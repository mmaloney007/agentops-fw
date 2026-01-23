from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI


JUDGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "statements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "support_score_0_to_3": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 3,
                    },
                    "contradiction": {"type": "boolean"},
                },
                "required": ["statement", "support_score_0_to_3", "contradiction"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["statements"],
    "additionalProperties": False,
}


SYSTEM_PROMPT = (
    "You are a strict evaluator of factual support.\n"
    "Given a question, a context, and a candidate JSON output:\n"
    "1) Extract atomic factual statements from the candidate output (especially answer and reasoning_summary).\n"
    "2) Score each statement's support using an integer scale: 0 unsupported, 1 weak support, 2 supported, 3 strongly supported.\n"
    "3) contradiction=true if the statement contradicts the context.\n"
    "Return JSON that matches the provided schema only. Do not include any extra keys."
)


@dataclass
class FaithfulnessResult:
    faithfulness: float
    contradiction_rate: float
    mean_support: float
    judge_output: Optional[Dict[str, Any]]
    raw_output: str
    latency_ms: float
    parse_error: Optional[str]


def _build_messages(
    question: str, context: str, candidate_json: Dict[str, Any]
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\n"
                f"CANDIDATE_JSON:\n{json.dumps(candidate_json, ensure_ascii=True, sort_keys=True)}"
            ),
        },
    ]


def score_faithfulness(
    base_url: str,
    api_key: str,
    model: str,
    question: str,
    context: str,
    candidate_json: Dict[str, Any],
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens_out: int = 512,
) -> FaithfulnessResult:
    client = OpenAI(base_url=base_url, api_key=api_key)
    rf = {
        "type": "json_schema",
        "json_schema": {"name": "judge_output", "schema": JUDGE_SCHEMA, "strict": True},
    }

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=_build_messages(question, context, candidate_json),
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens_out,
        response_format=rf,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    raw = resp.choices[0].message.content or ""
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        return FaithfulnessResult(
            faithfulness=0.0,
            contradiction_rate=1.0,
            mean_support=0.0,
            judge_output=None,
            raw_output=raw,
            latency_ms=latency_ms,
            parse_error=str(exc),
        )

    statements = parsed.get("statements", [])
    if not statements:
        return FaithfulnessResult(
            faithfulness=1.0,
            contradiction_rate=0.0,
            mean_support=1.0,
            judge_output=parsed,
            raw_output=raw,
            latency_ms=latency_ms,
            parse_error=None,
        )

    scores: List[int] = []
    contras: List[int] = []
    for st in statements:
        s = int(st.get("support_score_0_to_3", 0))
        c = 1 if bool(st.get("contradiction", False)) else 0
        scores.append(max(0, min(3, s)))
        contras.append(c)

    mean_support = sum(scores) / (3.0 * len(scores))
    contradiction_rate = sum(contras) / float(len(contras))
    faithfulness = max(0.0, min(1.0, float(mean_support - contradiction_rate)))
    return FaithfulnessResult(
        faithfulness=faithfulness,
        contradiction_rate=contradiction_rate,
        mean_support=mean_support,
        judge_output=parsed,
        raw_output=raw,
        latency_ms=latency_ms,
        parse_error=None,
    )
