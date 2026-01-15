from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from jsonschema import Draft202012Validator


class DecodingMode(str, Enum):
    UNCONSTRAINED = "UNCONSTRAINED"
    PROVIDER_STRUCTURED = "PROVIDER_STRUCTURED"
    PROVIDER_STRUCTURED_PLUS_VALIDATE = "PROVIDER_STRUCTURED_PLUS_VALIDATE"
    SPEC_DRIVEN = "SPEC_DRIVEN"
    SPEC_DRIVEN_PLUS_REPAIR = "SPEC_DRIVEN_PLUS_REPAIR"
    SPEC_DRIVEN_PLUS_SELFCONSISTENCY = "SPEC_DRIVEN_PLUS_SELFCONSISTENCY"

    @classmethod
    def from_str(cls, value: str) -> "DecodingMode":
        value = value.strip().upper()
        aliases = {
            "U": cls.UNCONSTRAINED,
            "P": cls.PROVIDER_STRUCTURED,
            "PV": cls.PROVIDER_STRUCTURED_PLUS_VALIDATE,
            "S": cls.SPEC_DRIVEN,
            "SR": cls.SPEC_DRIVEN_PLUS_REPAIR,
            "SSC": cls.SPEC_DRIVEN_PLUS_SELFCONSISTENCY,
        }
        if value in aliases:
            return aliases[value]
        return cls(value)


@dataclass
class Attempt:
    raw_text: str
    parsed_json: Optional[Dict[str, Any]]
    parse_error: Optional[str]
    schema_error: Optional[str]
    latency_ms: float
    ttft_ms: float
    tokens_in: int
    tokens_out: int
    timings: Dict[str, float]


@dataclass
class GenerationResult:
    final: Attempt
    attempts: List[Attempt]
    retry_count: int
    repair_count: int
    candidate_count: int
    total_latency_ms: float
    total_tokens_in: int
    total_tokens_out: int
    request_start: float
    request_end: float


def _parse_json(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, None
        return None, "parsed_json_not_object"
    except Exception as exc:
        return None, str(exc)


def _validate_schema(obj: Dict[str, Any], schema: Dict[str, Any]) -> Optional[str]:
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(obj), key=lambda e: e.path)
    if not errors:
        return None
    err = errors[0]
    path = ".".join([str(p) for p in err.path]) if err.path else "<root>"
    return f"{path}: {err.message}"


def _canonical_json(obj: Optional[Dict[str, Any]]) -> str:
    if not obj:
        return ""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _provider_generate_raw(
    prompt: str,
    schema: Dict[str, Any],
    mode: str,
    temperature: float,
    max_tokens: Optional[int],
) -> Tuple[str, Dict[str, Any], float, float, int, int]:
    backend = os.getenv("AOFW_PROVIDER", "lmstudio").lower()
    if backend == "lmstudio":
        from .providers.lmstudio_openai import generate_raw

        return generate_raw(prompt, schema, mode=mode, temperature=temperature, max_tokens=max_tokens)
    if backend == "ollama":
        from .providers.ollama_structured import generate_raw

        return generate_raw(prompt, schema, mode=mode, temperature=temperature, max_tokens=max_tokens)
    if backend == "vllm":
        from .providers.vllm_openai import generate_raw

        return generate_raw(prompt, schema, mode=mode, temperature=temperature, max_tokens=max_tokens)
    if backend == "hf_local":
        from .providers.hf_local import generate_raw

        return generate_raw(prompt, schema, mode=mode, temperature=temperature, max_tokens=max_tokens)
    # Fallback: empty output
    return "", {}, 5.0, 5.0, -1, -1


def provider_generate(
    prompt: str,
    schema: Dict[str, Any],
    mode: str | None = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> Tuple[Dict[str, Any], float, float, int]:
    decode_mode = mode or os.getenv("DECODE_MODE", "structured")
    _raw, parsed, lat_ms, ttft_ms, _tokens_in, tokens_out = _provider_generate_raw(
        prompt, schema, decode_mode, temperature, max_tokens
    )
    return parsed, lat_ms, ttft_ms, tokens_out


def _build_attempt(
    raw_text: str,
    parsed_json: Optional[Dict[str, Any]],
    parse_error: Optional[str],
    schema_error: Optional[str],
    latency_ms: float,
    ttft_ms: float,
    tokens_in: int,
    tokens_out: int,
    timings: Dict[str, float],
) -> Attempt:
    return Attempt(
        raw_text=raw_text,
        parsed_json=parsed_json,
        parse_error=parse_error,
        schema_error=schema_error,
        latency_ms=float(latency_ms),
        ttft_ms=float(ttft_ms),
        tokens_in=int(tokens_in),
        tokens_out=int(tokens_out),
        timings=timings,
    )


def generate_with_mode(
    prompt: str,
    schema: Dict[str, Any],
    mode: DecodingMode,
    temperature: float,
    max_tokens: Optional[int],
    max_retries: int,
    repair_max_attempts: int,
    self_consistency_samples: int,
    self_consistency_max_ms: int,
    self_consistency_selection: str = "majority_vote",
    include_error_in_repair_prompt: bool = True,
) -> GenerationResult:
    attempts: List[Attempt] = []
    retry_count = 0
    repair_count = 0
    candidate_count = 1
    total_tokens_in = 0
    total_tokens_out = 0
    total_latency_ms = 0.0

    def call_provider(cur_prompt: str, provider_mode: str) -> Attempt:
        call_start = time.perf_counter()
        raw, parsed, lat_ms, ttft_ms, tokens_in, tokens_out = _provider_generate_raw(
            cur_prompt, schema, provider_mode, temperature, max_tokens
        )
        call_end = time.perf_counter()
        val_start = time.perf_counter()
        obj, parse_err = _parse_json(raw) if provider_mode == "text" else (parsed, None if parsed else None)
        schema_err = None
        if obj is not None:
            schema_err = _validate_schema(obj, schema)
        val_end = time.perf_counter()
        timings = {
            "provider_call_start": call_start,
            "provider_call_end": call_end,
            "validation_start": val_start,
            "validation_end": val_end,
        }
        attempt = _build_attempt(raw, obj, parse_err, schema_err, lat_ms, ttft_ms, tokens_in, tokens_out, timings)
        return attempt

    def record_attempt(attempt: Attempt) -> None:
        nonlocal total_latency_ms, total_tokens_in, total_tokens_out
        attempts.append(attempt)
        total_latency_ms += attempt.latency_ms
        total_tokens_in += attempt.tokens_in
        total_tokens_out += attempt.tokens_out

    request_start = time.perf_counter()
    if mode == DecodingMode.UNCONSTRAINED:
        att = call_provider(prompt, "text")
        record_attempt(att)
        request_end = time.perf_counter()
        return GenerationResult(
            att,
            attempts,
            retry_count,
            repair_count,
            candidate_count,
            total_latency_ms,
            total_tokens_in,
            total_tokens_out,
            request_start,
            request_end,
        )

    if mode == DecodingMode.PROVIDER_STRUCTURED:
        att = call_provider(prompt, "structured")
        record_attempt(att)
        request_end = time.perf_counter()
        return GenerationResult(
            att,
            attempts,
            retry_count,
            repair_count,
            candidate_count,
            total_latency_ms,
            total_tokens_in,
            total_tokens_out,
            request_start,
            request_end,
        )

    def run_with_retries() -> Attempt:
        nonlocal retry_count
        cur_prompt = prompt
        for i in range(max_retries + 1):
            att = call_provider(cur_prompt, "structured")
            record_attempt(att)
            if att.parsed_json is not None and att.schema_error is None and att.parse_error is None:
                return att
            if i < max_retries:
                retry_count += 1
                err = att.parse_error or att.schema_error or "unknown_error"
                repair = "Your previous output was invalid. Return ONLY JSON that matches the schema."
                if include_error_in_repair_prompt:
                    repair += f" Error: {err}"
                cur_prompt = cur_prompt + "\n\n" + repair
        return attempts[-1]

    if mode == DecodingMode.PROVIDER_STRUCTURED_PLUS_VALIDATE:
        final = run_with_retries()
        request_end = time.perf_counter()
        return GenerationResult(
            final,
            attempts,
            retry_count,
            repair_count,
            candidate_count,
            total_latency_ms,
            total_tokens_in,
            total_tokens_out,
            request_start,
            request_end,
        )

    if mode == DecodingMode.SPEC_DRIVEN:
        final = run_with_retries()
        request_end = time.perf_counter()
        return GenerationResult(
            final,
            attempts,
            retry_count,
            repair_count,
            candidate_count,
            total_latency_ms,
            total_tokens_in,
            total_tokens_out,
            request_start,
            request_end,
        )

    if mode == DecodingMode.SPEC_DRIVEN_PLUS_REPAIR:
        final = run_with_retries()
        if final.parsed_json is None or final.schema_error is not None or final.parse_error is not None:
            err = final.parse_error or final.schema_error or "unknown_error"
            for _ in range(repair_max_attempts):
                repair_count += 1
                repair_prompt = (
                    "Fix the JSON to match the schema. Return ONLY valid JSON.\n"
                    f"Error: {err}\n"
                    f"Invalid JSON:\n{final.raw_text}"
                )
                repaired = call_provider(repair_prompt, "structured")
                record_attempt(repaired)
                if repaired.parsed_json is not None and repaired.schema_error is None and repaired.parse_error is None:
                    final = repaired
                    break
        request_end = time.perf_counter()
        return GenerationResult(
            final,
            attempts,
            retry_count,
            repair_count,
            candidate_count,
            total_latency_ms,
            total_tokens_in,
            total_tokens_out,
            request_start,
            request_end,
        )

    # SPEC_DRIVEN_PLUS_SELFCONSISTENCY: self-consistency over validated outputs
    t0 = time.perf_counter()
    candidates: List[Attempt] = []
    while len(candidates) < self_consistency_samples:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if elapsed_ms >= self_consistency_max_ms:
            break
        candidate = run_with_retries()
        candidates.append(candidate)
    candidate_count = len(candidates)

    valid = [c for c in candidates if c.parsed_json is not None and c.schema_error is None and c.parse_error is None]
    if not valid:
        final = candidates[-1] if candidates else run_with_retries()
        request_end = time.perf_counter()
        return GenerationResult(
            final,
            attempts,
            retry_count,
            repair_count,
            candidate_count,
            total_latency_ms,
            total_tokens_in,
            total_tokens_out,
            request_start,
            request_end,
        )

    if self_consistency_selection != "majority_vote":
        final = valid[0]
        request_end = time.perf_counter()
        return GenerationResult(
            final,
            attempts,
            retry_count,
            repair_count,
            candidate_count,
            total_latency_ms,
            total_tokens_in,
            total_tokens_out,
            request_start,
            request_end,
        )

    canon = [_canonical_json(v.parsed_json) for v in valid]
    best = max(set(canon), key=canon.count)
    for v, c in zip(valid, canon):
        if c == best:
            final = v
            request_end = time.perf_counter()
            return GenerationResult(
                final,
                attempts,
                retry_count,
                repair_count,
                candidate_count,
                total_latency_ms,
                total_tokens_in,
                total_tokens_out,
                request_start,
                request_end,
            )

    final = valid[0]
    request_end = time.perf_counter()
    return GenerationResult(
        final,
        attempts,
        retry_count,
        repair_count,
        candidate_count,
        total_latency_ms,
        total_tokens_in,
        total_tokens_out,
        request_start,
        request_end,
    )
