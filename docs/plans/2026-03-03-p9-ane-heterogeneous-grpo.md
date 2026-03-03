# P9: Heterogeneous ANE+MLX GRPO — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an ANE inference provider, evaluation harness, power monitor, and hybrid GRPO trainer that uses Apple's Neural Engine for rollout generation and MLX/GPU for gradient updates.

**Architecture:** Anemll wraps CoreML models for ANE inference. We build `ane_local.py` following the existing provider contract (`generate_raw` -> 6-tuple). The hybrid GRPO trainer generates rollouts on ANE, scores with existing reward functions, then computes gradients and updates LoRA weights on MLX. Weight sync between MLX and ANE is the key experimental bottleneck.

**Tech Stack:** Anemll (CoreML/ANE inference), coremltools >=9.0, mlx + mlx-lm (gradients), transformers (tokenizer), powermetrics (power measurement)

---

## Task 1: ANE Provider — Core Inference

**Files:**
- Create: `agent_stable_slo/rollout/providers/ane_local.py`
- Modify: `agent_stable_slo/rollout/engine.py:99-143`
- Test: `tests/test_ane_provider.py`

### Step 1: Write failing tests for ANE provider

```python
# tests/test_ane_provider.py
"""Tests for ANE local inference provider.

All Anemll/CoreML dependencies are mocked so tests run without
Apple Silicon or ANE models installed.
"""

import json
import os
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_schema():
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["intent", "confidence"],
    }


@pytest.fixture
def nested_schema():
    return {
        "type": "object",
        "properties": {
            "user": {"type": "object", "properties": {"id": {"type": "integer"}}},
            "tags": {"type": "array"},
        },
        "required": ["user", "tags"],
    }


# ---------------------------------------------------------------------------
# 1. Provider registration in engine dispatch
# ---------------------------------------------------------------------------

class TestProviderRegistration:
    """Verify engine._provider_generate_raw routes to ane_local correctly."""

    @patch.dict(os.environ, {"AOFW_PROVIDER": "ane_local"})
    @patch("agent_stable_slo.rollout.providers.ane_local.generate_raw")
    def test_routes_ane_local(self, mock_gen):
        mock_gen.return_value = ("raw", {"k": "v"}, 10.0, 10.0, 5, 8)
        from agent_stable_slo.rollout.engine import _provider_generate_raw

        result = _provider_generate_raw("prompt", {}, "structured", 0.0, None)
        mock_gen.assert_called_once()
        assert len(result) == 7
        assert result[-1] is None

    @patch.dict(os.environ, {"AOFW_PROVIDER": "ane"})
    @patch("agent_stable_slo.rollout.providers.ane_local.generate_raw")
    def test_routes_ane_alias(self, mock_gen):
        mock_gen.return_value = ("raw", {"k": "v"}, 10.0, 10.0, 5, 8)
        from agent_stable_slo.rollout.engine import _provider_generate_raw

        result = _provider_generate_raw("prompt", {}, "structured", 0.0, None)
        mock_gen.assert_called_once()
        assert result[0] == "raw"


# ---------------------------------------------------------------------------
# 2. generate_raw signature and return shape
# ---------------------------------------------------------------------------

class TestGenerateRawSignature:
    """Verify generate_raw returns the correct 6-tuple."""

    @patch("agent_stable_slo.rollout.providers.ane_local._ane_generate")
    @patch("agent_stable_slo.rollout.providers.ane_local._get_tokenizer")
    @patch("agent_stable_slo.rollout.providers.ane_local._get_ane_model")
    @patch.dict(os.environ, {"ANE_META_DIR": "/fake/model/dir"})
    def test_returns_6_tuple(self, mock_model, mock_tok, mock_gen, simple_schema):
        tok = MagicMock()
        tok.encode.return_value = [1, 2, 3, 4, 5]
        tok.decode.return_value = '{"intent": "greet", "confidence": 0.9}'
        tok.apply_chat_template.return_value = "formatted prompt"
        mock_tok.return_value = tok
        mock_model.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock(), {})
        mock_gen.return_value = ([101, 102, 103], 50.0, 15.0)

        from agent_stable_slo.rollout.providers.ane_local import generate_raw

        result = generate_raw("What is the intent?", simple_schema)
        assert len(result) == 6
        raw_text, parsed, lat_ms, ttft_ms, tokens_in, tokens_out = result
        assert isinstance(raw_text, str)
        assert isinstance(parsed, dict)
        assert isinstance(lat_ms, float)
        assert isinstance(ttft_ms, float)
        assert isinstance(tokens_in, int)
        assert isinstance(tokens_out, int)


# ---------------------------------------------------------------------------
# 3. JSON extraction (reuses same logic as MLX provider)
# ---------------------------------------------------------------------------

class TestJsonExtraction:
    """Verify JSON extraction from ANE model output."""

    def test_direct_json(self):
        from agent_stable_slo.rollout.providers.ane_local import _extract_json

        result = _extract_json('{"intent": "greet", "confidence": 0.9}')
        assert result == {"intent": "greet", "confidence": 0.9}

    def test_code_block(self):
        from agent_stable_slo.rollout.providers.ane_local import _extract_json

        text = 'Here is the JSON:\n```json\n{"intent": "bye"}\n```'
        assert _extract_json(text) == {"intent": "bye"}

    def test_embedded_json(self):
        from agent_stable_slo.rollout.providers.ane_local import _extract_json

        text = 'Some text {"intent": "greet"} more text'
        assert _extract_json(text) == {"intent": "greet"}

    def test_no_json(self):
        from agent_stable_slo.rollout.providers.ane_local import _extract_json

        assert _extract_json("no json here") == {}


# ---------------------------------------------------------------------------
# 4. Prompt building with schema
# ---------------------------------------------------------------------------

class TestPromptBuilding:
    """Verify prompt includes schema instruction."""

    def test_includes_schema_example(self, simple_schema):
        from agent_stable_slo.rollout.providers.ane_local import _build_prompt_with_schema

        result = _build_prompt_with_schema("Classify this", simple_schema)
        assert "JSON" in result
        assert "intent" in result
        assert "confidence" in result


# ---------------------------------------------------------------------------
# 5. Required-field backfill
# ---------------------------------------------------------------------------

class TestFieldBackfill:
    """Verify missing required fields are backfilled with defaults."""

    def test_backfill_missing_string(self, simple_schema):
        from agent_stable_slo.rollout.providers.ane_local import _backfill_required

        parsed = {"confidence": 0.5}
        result = _backfill_required(parsed, simple_schema)
        assert result["intent"] == ""
        assert result["confidence"] == 0.5

    def test_backfill_missing_array(self, nested_schema):
        from agent_stable_slo.rollout.providers.ane_local import _backfill_required

        parsed = {"user": {"id": 1}}
        result = _backfill_required(parsed, nested_schema)
        assert result["tags"] == []
```

### Step 2: Run tests to verify they fail

Run: `/Users/maloney/.local/share/mamba/bin/python -m pytest tests/test_ane_provider.py -v`
Expected: FAIL (module not found)

### Step 3: Create the ANE provider

```python
# agent_stable_slo/rollout/providers/ane_local.py
"""
ANE (Apple Neural Engine) local inference provider.
Uses Anemll to load CoreML models and run inference on the Neural Engine.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Cache loaded models to avoid reloading
_MODEL_CACHE: Dict[str, Any] = {}
_TOKENIZER_CACHE: Dict[str, Any] = {}


def _get_tokenizer(model_name: str):
    """Load or retrieve cached tokenizer from HuggingFace."""
    if model_name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_name]

    from transformers import AutoTokenizer

    print(f"[ane_local] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _TOKENIZER_CACHE[model_name] = tokenizer
    return tokenizer


def _get_ane_model(meta_dir: str):
    """Load or retrieve cached ANE model from Anemll-converted directory.

    Returns (embed_model, ffn_models, lmhead_model, state, metadata).
    """
    if meta_dir in _MODEL_CACHE:
        return _MODEL_CACHE[meta_dir]

    import yaml
    import numpy as np

    # Import Anemll's chat module functions
    # Anemll installs as a package but inference lives in tests/chat.py
    # We import coremltools directly and use the Anemll calling convention.
    import coremltools as ct

    print(f"[ane_local] Loading ANE model from: {meta_dir}")

    meta_path = Path(meta_dir) / "meta.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.yaml not found in {meta_dir}")

    with open(meta_path) as f:
        meta = yaml.safe_load(f)

    params = meta.get("model_info", {}).get("parameters", {})
    model_prefix = params.get("model_prefix", "model")
    context_length = int(params.get("context_length", 512))
    is_monolithic = meta.get("model_info", {}).get("model_type") == "monolithic"

    compute_unit = ct.ComputeUnit.CPU_AND_NE

    if is_monolithic:
        mono_name = params.get("monolithic_model", f"{model_prefix}_monolithic.mlmodelc")
        mono_path = str(Path(meta_dir) / mono_name)
        model = ct.models.CompiledMLModel(mono_path, compute_unit)
        # For monolithic models, functions are accessed via model[func_name]
        state = model.make_state()
        metadata = {
            "context_length": context_length,
            "is_monolithic": True,
            "vocab_size": int(params.get("vocab_size", 32000)),
            "argmax_in_model": params.get("argmax_in_model", False),
            "lm_head_chunk_sizes": params.get("lm_head_chunk_sizes"),
            "split_lm_head": int(params.get("split_lm_head", 1)),
            "batch_size": int(params.get("batch_size", 64)),
            "sliding_window": params.get("sliding_window"),
        }
        result = (model, None, None, state, metadata)
    else:
        # Chunked model: embed + ffn chunks + lmhead
        embed_name = params.get("embeddings", f"{model_prefix}_embeddings.mlmodelc")
        lmhead_name = params.get("lm_head", f"{model_prefix}_lmhead.mlmodelc")
        num_chunks = int(params.get("num_chunks", 1))

        embed_model = ct.models.CompiledMLModel(
            str(Path(meta_dir) / embed_name), compute_unit
        )
        lmhead_model = ct.models.CompiledMLModel(
            str(Path(meta_dir) / lmhead_name), compute_unit
        )

        ffn_models = []
        for i in range(num_chunks):
            ffn_template = params.get("ffn", f"{model_prefix}_ffn_chunk{{i}}of{num_chunks}.mlmodelc")
            ffn_name = ffn_template.replace("{i}", str(i))
            ffn_path = str(Path(meta_dir) / ffn_name)
            ffn = ct.models.CompiledMLModel(ffn_path, compute_unit)
            ffn_models.append(ffn)

        # Create state for KV cache
        states = []
        for ffn in ffn_models:
            states.append(ffn.make_state())

        metadata = {
            "context_length": context_length,
            "is_monolithic": False,
            "vocab_size": int(params.get("vocab_size", 32000)),
            "argmax_in_model": params.get("argmax_in_model", False),
            "lm_head_chunk_sizes": params.get("lm_head_chunk_sizes"),
            "split_lm_head": int(params.get("split_lm_head", 1)),
            "batch_size": int(params.get("batch_size", 64)),
            "num_chunks": num_chunks,
            "sliding_window": params.get("sliding_window"),
        }
        result = (embed_model, ffn_models, lmhead_model, states, metadata)

    _MODEL_CACHE[meta_dir] = result
    return result


def _build_prompt_with_schema(prompt: str, schema: dict) -> str:
    """Add JSON schema instruction to prompt."""
    example = {}
    for prop, spec in schema.get("properties", {}).items():
        typ = spec.get("type", "string")
        if typ == "string":
            example[prop] = "<value>"
        elif typ == "array":
            example[prop] = []
        elif typ == "object":
            example[prop] = {}
        elif typ in ("number", "integer"):
            example[prop] = 0

    return f"""{prompt}

Respond with ONLY a JSON object like this example:
{json.dumps(example)}

Your JSON response (no explanation, just the JSON):"""


def _extract_json(raw_text: str) -> dict:
    """Extract JSON from raw model output."""
    # Try direct parse
    try:
        obj = json.loads(raw_text.strip())
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Try code blocks
    if "```" in raw_text:
        parts = raw_text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            try:
                obj = json.loads(p)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue

    # Try to find JSON objects
    matches = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw_text)
    for m in matches:
        try:
            obj = json.loads(m)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    return {}


def _backfill_required(parsed: dict, schema: dict) -> dict:
    """Ensure required fields exist with type-appropriate defaults."""
    if not isinstance(parsed, dict):
        return parsed
    props = schema.get("properties", {})
    required = schema.get("required", [])
    for req in required:
        if req not in parsed:
            typ = props.get(req, {}).get("type")
            if typ == "string":
                parsed[req] = ""
            elif typ == "array":
                parsed[req] = []
            elif typ == "object":
                parsed[req] = {}
            elif typ == "boolean":
                parsed[req] = False
            elif typ in ("number", "integer"):
                parsed[req] = 0
    return parsed


def _ane_generate(
    model_components: tuple,
    tokenizer,
    input_ids: list,
    max_tokens: int,
    temperature: float,
    metadata: dict,
) -> Tuple[List[int], float, float]:
    """Run token-by-token generation on ANE.

    Returns (output_token_ids, total_latency_ms, ttft_ms).
    """
    import numpy as np

    context_length = metadata["context_length"]
    is_monolithic = metadata["is_monolithic"]

    if is_monolithic:
        model, _, _, state, _ = model_components
        return _generate_monolithic(
            model, state, input_ids, max_tokens, temperature, metadata
        )
    else:
        embed_model, ffn_models, lmhead_model, states, _ = model_components
        return _generate_chunked(
            embed_model, ffn_models, lmhead_model, states,
            input_ids, max_tokens, temperature, metadata
        )


def _make_causal_mask(length: int, start: int):
    """Create causal attention mask."""
    import numpy as np

    mask = np.full((1, 1, length, length), -np.inf, dtype=np.float16)
    for i in range(length):
        for j in range(start, start + i + 1):
            if j < length:
                mask[0, 0, i, j] = 0.0
    return mask


def _generate_chunked(
    embed_model, ffn_models, lmhead_model, states,
    input_ids, max_tokens, temperature, metadata,
):
    """Generate tokens using chunked ANE model."""
    import numpy as np

    context_length = metadata["context_length"]
    batch_size = metadata.get("batch_size", 64)
    vocab_size = metadata.get("vocab_size", 32000)
    argmax_mode = metadata.get("argmax_in_model", False)

    # Build stop token set
    stop_ids = {2}  # EOS
    if hasattr(embed_model, "eos_token_id"):
        stop_ids.add(embed_model.eos_token_id)

    # Prefill phase
    prefill_start = time.perf_counter()
    input_arr = np.array(input_ids, dtype=np.int32)
    context_pos = 0

    # Process input in batches
    for batch_start in range(0, len(input_ids), batch_size):
        batch_end = min(batch_start + batch_size, len(input_ids))
        batch_tokens = input_arr[batch_start:batch_end]
        batch_len = len(batch_tokens)

        # Embed
        embed_out = embed_model.predict(
            {"input_ids": batch_tokens.reshape(1, -1)}
        )
        hidden = embed_out["hidden_states"]

        # Position IDs
        pos_ids = np.arange(context_pos, context_pos + batch_len, dtype=np.int32)
        mask = _make_causal_mask(batch_len, 0)

        # Run through FFN chunks
        for chunk_idx, ffn in enumerate(ffn_models):
            inputs = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": pos_ids,
                "causal_mask": mask,
                "current_pos": np.array([context_pos], dtype=np.int32),
            }
            out = ffn.predict(inputs, states[chunk_idx])
            hidden = out["output_hidden_states"]

        context_pos += batch_len

    ttft_ms = (time.perf_counter() - prefill_start) * 1000.0

    # Decode phase
    decode_start = time.perf_counter()
    output_tokens = []
    current_token = input_ids[-1]

    for _ in range(max_tokens):
        token_arr = np.array([[current_token]], dtype=np.int32)
        embed_out = embed_model.predict({"input_ids": token_arr})
        hidden = embed_out["hidden_states"]

        pos_ids = np.array([context_pos], dtype=np.int32)
        mask = np.zeros((1, 1, 1, context_length), dtype=np.float16)
        mask[0, 0, 0, context_pos + 1:] = -np.inf

        for chunk_idx, ffn in enumerate(ffn_models):
            inputs = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": pos_ids,
                "causal_mask": mask,
                "current_pos": np.array([context_pos], dtype=np.int32),
            }
            out = ffn.predict(inputs, states[chunk_idx])
            hidden = out["output_hidden_states"]

        # LM head
        lm_out = lmhead_model.predict({"hidden_states": hidden.astype(np.float16)})

        if argmax_mode:
            next_token = int(lm_out["argmax_idx"])
        else:
            logits_key = next(
                (k for k in lm_out if "logits" in k.lower()), list(lm_out.keys())[0]
            )
            logits = lm_out[logits_key].flatten()
            if temperature > 0:
                probs = np.exp(logits / temperature)
                probs = probs / probs.sum()
                next_token = int(np.random.choice(len(probs), p=probs))
            else:
                next_token = int(np.argmax(logits))

        if next_token in stop_ids:
            break

        output_tokens.append(next_token)
        current_token = next_token
        context_pos += 1

    total_ms = ttft_ms + (time.perf_counter() - decode_start) * 1000.0
    return output_tokens, total_ms, ttft_ms


def _generate_monolithic(model, state, input_ids, max_tokens, temperature, metadata):
    """Generate tokens using monolithic ANE model."""
    import numpy as np

    context_length = metadata["context_length"]
    batch_size = metadata.get("batch_size", 64)
    argmax_mode = metadata.get("argmax_in_model", False)

    stop_ids = {2}

    # Prefill
    prefill_start = time.perf_counter()
    input_arr = np.array(input_ids, dtype=np.int32)
    context_pos = 0

    for batch_start in range(0, len(input_ids), batch_size):
        batch_end = min(batch_start + batch_size, len(input_ids))
        batch_tokens = input_arr[batch_start:batch_end]
        batch_len = len(batch_tokens)

        pos_ids = np.arange(context_pos, context_pos + batch_len, dtype=np.int32)
        mask = _make_causal_mask(batch_len, 0)

        inputs = {
            "input_ids": batch_tokens.reshape(1, -1).astype(np.int32),
            "position_ids": pos_ids,
            "causal_mask": mask,
            "current_pos": pos_ids,
        }
        model.predict(inputs, state)
        context_pos += batch_len

    ttft_ms = (time.perf_counter() - prefill_start) * 1000.0

    # Decode
    decode_start = time.perf_counter()
    output_tokens = []
    current_token = input_ids[-1]

    for _ in range(max_tokens):
        pos_ids = np.array([context_pos], dtype=np.int32)
        mask = np.zeros((1, 1, 1, context_length), dtype=np.float16)
        mask[0, 0, 0, context_pos + 1:] = -np.inf

        inputs = {
            "input_ids": np.array([[current_token]], dtype=np.int32),
            "position_ids": pos_ids,
            "causal_mask": mask,
            "current_pos": pos_ids,
        }
        out = model.predict(inputs, state)

        if argmax_mode:
            next_token = int(out["argmax_idx"])
        else:
            logits_key = next(
                (k for k in out if "logits" in k.lower()), list(out.keys())[0]
            )
            logits = out[logits_key].flatten()
            if temperature > 0:
                probs = np.exp(logits / temperature)
                probs = probs / probs.sum()
                next_token = int(np.random.choice(len(probs), p=probs))
            else:
                next_token = int(np.argmax(logits))

        if next_token in stop_ids:
            break

        output_tokens.append(next_token)
        current_token = next_token
        context_pos += 1

    total_ms = ttft_ms + (time.perf_counter() - decode_start) * 1000.0
    return output_tokens, total_ms, ttft_ms


def generate_raw(
    prompt: str,
    schema: dict,
    mode: str = "structured",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> Tuple[str, dict, float, float, int, int]:
    """Generate response using ANE model via Anemll/CoreML.

    Follows the same provider contract as mlx_local.py.
    """
    meta_dir = os.getenv("ANE_META_DIR")
    if not meta_dir:
        raise ValueError("ANE_META_DIR environment variable must be set")

    tokenizer_name = os.getenv("ANE_TOKENIZER", os.getenv("ANE_HF_MODEL", ""))
    if not tokenizer_name:
        raise ValueError("ANE_TOKENIZER or ANE_HF_MODEL must be set for tokenizer")

    max_new = max_tokens or int(os.getenv("ANE_MAX_TOKENS", "256"))

    tokenizer = _get_tokenizer(tokenizer_name)
    model_components = _get_ane_model(meta_dir)

    # Build prompt with schema
    full_prompt = _build_prompt_with_schema(prompt, schema)

    # Apply chat template
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": full_prompt}]
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
    else:
        formatted = full_prompt

    # Tokenize
    input_ids = tokenizer.encode(formatted)
    tokens_in = len(input_ids)

    # Generate on ANE
    _, _, _, _, metadata = model_components
    output_tokens, lat_ms, ttft_ms = _ane_generate(
        model_components, tokenizer, input_ids, max_new, temperature, metadata
    )

    # Decode
    raw_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
    tokens_out = len(output_tokens)

    # Parse JSON
    parsed = _extract_json(raw_text)
    parsed = _backfill_required(parsed, schema)

    return raw_text, parsed, lat_ms, ttft_ms, tokens_in, tokens_out
```

### Step 4: Register ANE in engine dispatch

Add to `engine.py` `_provider_generate_raw`, before the fallback return:

```python
    if backend in ("ane", "ane_local"):
        from .providers.ane_local import generate_raw

        result = generate_raw(
            prompt, schema, mode=mode, temperature=temperature, max_tokens=max_tokens
        )
        return (*result, None)
```

### Step 5: Run tests

Run: `/Users/maloney/.local/share/mamba/bin/python -m pytest tests/test_ane_provider.py -v`
Expected: All tests PASS

### Step 6: Commit

```bash
git add agent_stable_slo/rollout/providers/ane_local.py agent_stable_slo/rollout/engine.py tests/test_ane_provider.py
git commit -m "feat(p9): add ANE inference provider with engine dispatch"
```

---

## Task 2: Power Measurement Harness

**Files:**
- Create: `agent_stable_slo/bench/power_monitor.py`
- Test: `tests/test_power_monitor.py`

### Step 1: Write failing tests

```python
# tests/test_power_monitor.py
"""Tests for power monitoring harness.

Mocks subprocess calls to powermetrics.
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest


class TestPowerMonitor:
    """Verify power monitoring captures and correlates samples."""

    @patch("subprocess.Popen")
    def test_start_stop_captures_samples(self, mock_popen):
        """Monitor should capture power samples between start/stop."""
        from agent_stable_slo.bench.power_monitor import PowerMonitor

        # Mock powermetrics returning JSON samples
        mock_proc = MagicMock()
        sample = {
            "processor": {"clusters": [
                {"name": "E-Cluster", "hw_power": 1.5},
                {"name": "P-Cluster", "hw_power": 3.0},
            ]},
            "gpu": {"hw_power": 0.3},
        }
        mock_proc.stdout.readline.side_effect = [
            json.dumps(sample).encode() + b"\n",
            json.dumps(sample).encode() + b"\n",
            b"",  # EOF
        ]
        mock_proc.poll.side_effect = [None, None, 0]
        mock_popen.return_value = mock_proc

        mon = PowerMonitor(interval_ms=100, sudo_password=None)
        mon.start()
        time.sleep(0.05)
        result = mon.stop()

        assert "samples" in result
        assert "mean_cpu_w" in result
        assert "mean_gpu_w" in result

    def test_power_summary_fields(self):
        """Summary should include all expected power fields."""
        from agent_stable_slo.bench.power_monitor import PowerSummary

        summary = PowerSummary(
            samples=[],
            mean_cpu_w=2.5,
            mean_gpu_w=0.3,
            mean_ane_w=0.0,
            mean_total_w=2.8,
            duration_s=1.0,
            energy_j=2.8,
        )
        assert summary.mean_total_w == 2.8
        assert summary.energy_j == 2.8
```

### Step 2: Run tests to verify they fail

Run: `/Users/maloney/.local/share/mamba/bin/python -m pytest tests/test_power_monitor.py -v`
Expected: FAIL

### Step 3: Implement power monitor

```python
# agent_stable_slo/bench/power_monitor.py
"""Power measurement harness for Apple Silicon.

Wraps macOS `powermetrics` to capture CPU/GPU/ANE power draw
and correlate with inference or training windows.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PowerSample:
    timestamp: float
    cpu_w: float = 0.0
    gpu_w: float = 0.0
    ane_w: float = 0.0
    total_w: float = 0.0


@dataclass
class PowerSummary:
    samples: List[PowerSample]
    mean_cpu_w: float
    mean_gpu_w: float
    mean_ane_w: float
    mean_total_w: float
    duration_s: float
    energy_j: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_cpu_w": round(self.mean_cpu_w, 3),
            "mean_gpu_w": round(self.mean_gpu_w, 3),
            "mean_ane_w": round(self.mean_ane_w, 3),
            "mean_total_w": round(self.mean_total_w, 3),
            "duration_s": round(self.duration_s, 3),
            "energy_j": round(self.energy_j, 3),
            "num_samples": len(self.samples),
        }


class PowerMonitor:
    """Background power measurement using macOS powermetrics.

    Usage:
        mon = PowerMonitor(interval_ms=100)
        mon.start()
        # ... do inference or training ...
        summary = mon.stop()
        print(summary.mean_cpu_w, summary.mean_gpu_w)
    """

    def __init__(self, interval_ms: int = 100, sudo_password: Optional[str] = None):
        self.interval_ms = interval_ms
        self.sudo_password = sudo_password
        self._samples: List[PowerSample] = []
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._start_time: float = 0.0

    def start(self) -> None:
        """Start background power sampling."""
        self._samples = []
        self._stop_event.clear()
        self._start_time = time.time()

        cmd = [
            "sudo", "-n", "powermetrics",
            "--samplers", "cpu_power,gpu_power",
            "-i", str(self.interval_ms),
            "--format", "plist",
        ]

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()
        except (PermissionError, FileNotFoundError) as e:
            print(f"[power_monitor] WARNING: powermetrics not available: {e}")
            print("[power_monitor] Power data will be empty. Run with sudo for power measurement.")

    def _reader(self) -> None:
        """Read powermetrics output in background thread."""
        if not self._proc or not self._proc.stdout:
            return

        import plistlib
        buffer = b""

        while not self._stop_event.is_set():
            line = self._proc.stdout.readline()
            if not line:
                if self._proc.poll() is not None:
                    break
                continue

            buffer += line

            # plist output ends with </plist>
            if b"</plist>" in buffer:
                try:
                    data = plistlib.loads(buffer)
                    sample = self._parse_plist_sample(data)
                    self._samples.append(sample)
                except Exception:
                    pass
                buffer = b""

    def _parse_plist_sample(self, data: dict) -> PowerSample:
        """Extract power values from a powermetrics plist sample."""
        cpu_w = 0.0
        gpu_w = 0.0
        ane_w = 0.0

        # CPU power from processor clusters
        proc = data.get("processor", {})
        for cluster in proc.get("clusters", []):
            cpu_w += cluster.get("hw_power", 0.0)

        # GPU power
        gpu_data = data.get("gpu", {})
        gpu_w = gpu_data.get("hw_power", 0.0)

        # ANE power (if available in newer macOS)
        ane_data = data.get("ane", {})
        ane_w = ane_data.get("hw_power", 0.0)

        return PowerSample(
            timestamp=time.time(),
            cpu_w=cpu_w,
            gpu_w=gpu_w,
            ane_w=ane_w,
            total_w=cpu_w + gpu_w + ane_w,
        )

    def stop(self) -> PowerSummary:
        """Stop sampling and return summary."""
        self._stop_event.set()
        duration = time.time() - self._start_time

        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()

        if self._thread:
            self._thread.join(timeout=2)

        if not self._samples:
            return PowerSummary(
                samples=[], mean_cpu_w=0, mean_gpu_w=0, mean_ane_w=0,
                mean_total_w=0, duration_s=duration, energy_j=0,
            )

        n = len(self._samples)
        mean_cpu = sum(s.cpu_w for s in self._samples) / n
        mean_gpu = sum(s.gpu_w for s in self._samples) / n
        mean_ane = sum(s.ane_w for s in self._samples) / n
        mean_total = mean_cpu + mean_gpu + mean_ane
        energy = mean_total * duration

        return PowerSummary(
            samples=self._samples,
            mean_cpu_w=mean_cpu,
            mean_gpu_w=mean_gpu,
            mean_ane_w=mean_ane,
            mean_total_w=mean_total,
            duration_s=duration,
            energy_j=energy,
        )
```

### Step 4: Run tests

Run: `/Users/maloney/.local/share/mamba/bin/python -m pytest tests/test_power_monitor.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add agent_stable_slo/bench/power_monitor.py tests/test_power_monitor.py
git commit -m "feat(p9): add power measurement harness for ANE experiments"
```

---

## Task 3: Model Conversion Script

**Files:**
- Create: `scripts/convert_ane_models.py`

### Step 1: Write the conversion script

```python
#!/usr/bin/env python3
"""Convert HuggingFace models to Anemll/CoreML format for ANE inference.

Uses Anemll's converter pipeline to produce .mlmodelc files that run
on the Apple Neural Engine.

Usage:
    python scripts/convert_ane_models.py --model Qwen/Qwen3.5-0.8B --output models/ane/qwen3.5-0.8b
    python scripts/convert_ane_models.py --all --output models/ane
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Models to convert (HF ID -> short name)
MODEL_REGISTRY: Dict[str, str] = {
    "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen3.5-2b": "Qwen/Qwen3.5-2B",
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
}

# Default conversion parameters
DEFAULT_CONTEXT = 512
DEFAULT_BATCH = 64
DEFAULT_LUT1 = 6   # embeddings quantization bits
DEFAULT_LUT2 = 4   # FFN quantization bits
DEFAULT_LUT3 = 6   # LM head quantization bits


def find_anemll_converter() -> str:
    """Locate Anemll's convert_model.sh script."""
    # Check if anemll is installed as package
    try:
        import anemll
        pkg_dir = Path(anemll.__file__).parent
        script = pkg_dir / "utils" / "convert_model.sh"
        if script.exists():
            return str(script)
    except ImportError:
        pass

    # Check common clone locations
    for candidate in [
        Path.home() / "Anemll" / "anemll" / "utils" / "convert_model.sh",
        Path.home() / "Projects" / "Anemll" / "anemll" / "utils" / "convert_model.sh",
        Path("/opt/anemll/anemll/utils/convert_model.sh"),
    ]:
        if candidate.exists():
            return str(candidate)

    # Check ANEMLL_HOME env var
    home = os.getenv("ANEMLL_HOME")
    if home:
        script = Path(home) / "anemll" / "utils" / "convert_model.sh"
        if script.exists():
            return str(script)

    raise FileNotFoundError(
        "Cannot find Anemll's convert_model.sh. Set ANEMLL_HOME or install anemll."
    )


def check_preconverted(model_name: str, output_dir: Path) -> bool:
    """Check if a pre-converted model exists on HuggingFace anemll org."""
    # Pre-converted models at huggingface.co/anemll
    meta_path = output_dir / "meta.yaml"
    if meta_path.exists():
        print(f"  [skip] Already converted: {output_dir}")
        return True
    return False


def convert_model(
    hf_id: str,
    output_dir: Path,
    context: int = DEFAULT_CONTEXT,
    batch: int = DEFAULT_BATCH,
    lut1: int = DEFAULT_LUT1,
    lut2: int = DEFAULT_LUT2,
    lut3: int = DEFAULT_LUT3,
    monolithic: bool = True,
    argmax: bool = True,
) -> Dict[str, Any]:
    """Convert a single model to ANE format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if check_preconverted(hf_id, output_dir):
        return {"model": hf_id, "status": "skipped", "output": str(output_dir)}

    converter = find_anemll_converter()

    cmd = [
        "bash", converter,
        "--model", hf_id,
        "--output", str(output_dir),
        "--context", str(context),
        "--batch", str(batch),
        "--lut1", str(lut1),
        "--lut2", str(lut2),
        "--lut3", str(lut3),
    ]

    if monolithic:
        cmd.append("--monolithic")
    if argmax:
        cmd.append("--argmax")

    print(f"\n{'='*60}")
    print(f"  Converting: {hf_id}")
    print(f"  Output:     {output_dir}")
    print(f"  Context:    {context}")
    print(f"  Quant:      lut1={lut1} lut2={lut2} lut3={lut3}")
    print(f"{'='*60}\n")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    status = "ok" if result.returncode == 0 else "error"
    print(f"\n  --> {hf_id}: {status} ({elapsed:.1f}s)")

    # Save conversion metadata
    meta = {
        "hf_model": hf_id,
        "context_length": context,
        "batch_size": batch,
        "quantization": {"lut1": lut1, "lut2": lut2, "lut3": lut3},
        "monolithic": monolithic,
        "argmax": argmax,
        "conversion_time_s": round(elapsed, 1),
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "conversion_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {"model": hf_id, "status": status, "elapsed_s": elapsed, "output": str(output_dir)}


def main():
    ap = argparse.ArgumentParser(description="Convert models to ANE/CoreML format.")

    model_group = ap.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", help="HuggingFace model ID to convert.")
    model_group.add_argument("--all", action="store_true", help="Convert all registered models.")
    model_group.add_argument("--models", nargs="+", help="Short names from registry.")

    ap.add_argument("--output", default="models/ane", help="Output directory root.")
    ap.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--lut1", type=int, default=DEFAULT_LUT1)
    ap.add_argument("--lut2", type=int, default=DEFAULT_LUT2)
    ap.add_argument("--lut3", type=int, default=DEFAULT_LUT3)
    ap.add_argument("--no-monolithic", action="store_true")
    ap.add_argument("--no-argmax", action="store_true")

    args = ap.parse_args()
    output_root = Path(args.output)

    if args.all:
        models = list(MODEL_REGISTRY.items())
    elif args.models:
        models = [(name, MODEL_REGISTRY[name]) for name in args.models]
    else:
        slug = args.model.replace("/", "_").lower()
        models = [(slug, args.model)]

    results = []
    for short_name, hf_id in models:
        out_dir = output_root / short_name
        res = convert_model(
            hf_id, out_dir,
            context=args.context, batch=args.batch,
            lut1=args.lut1, lut2=args.lut2, lut3=args.lut3,
            monolithic=not args.no_monolithic,
            argmax=not args.no_argmax,
        )
        results.append(res)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Conversion Complete — {len(models)} models")
    for r in results:
        print(f"  {r['model']:<40} {r['status']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
```

### Step 2: Commit

```bash
git add scripts/convert_ane_models.py
git commit -m "feat(p9): add ANE model conversion script"
```

---

## Task 4: ANE Evaluation Harness

**Files:**
- Create: `scripts/eval_ane_suite.py`

### Step 1: Write the eval harness

```python
#!/usr/bin/env python3
"""
ANE evaluation wrapper -- runs eval_t_suite against Apple Neural Engine models.

Sets AOFW_PROVIDER=ane_local and runs each model sequentially.
Direct comparison to eval_mlx_suite.py results.

Usage:
    python scripts/eval_ane_suite.py --models qwen3.5-0.8b
    python scripts/eval_ane_suite.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Model registry: short-name -> (HF ID for tokenizer, ANE model dir)
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "qwen3.5-0.8b": {
        "hf_id": "Qwen/Qwen3.5-0.8B",
        "ane_dir": "models/ane/qwen3.5-0.8b",
    },
    "qwen3.5-2b": {
        "hf_id": "Qwen/Qwen3.5-2B",
        "ane_dir": "models/ane/qwen3.5-2b",
    },
    "qwen3.5-4b": {
        "hf_id": "Qwen/Qwen3.5-4B",
        "ane_dir": "models/ane/qwen3.5-4b",
    },
}

DEFAULT_TASKS = [
    "tasks/clinc_en.jsonl",
    "tasks/hotpot_dev.jsonl",
    "tasks/t3_tools.jsonl",
    "tasks/t4_bfcl.jsonl",
    "tasks/t5_swebench.jsonl",
    "tasks/public_gsm8k.jsonl",
]


# ---------------------------------------------------------------------------
# Hardware metadata
# ---------------------------------------------------------------------------

def _sysctl(key: str) -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", key], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return ""


def get_hardware_info() -> Dict[str, Any]:
    """Collect Apple Silicon + ANE hardware metadata."""
    info: Dict[str, Any] = {
        "chip": _sysctl("machdep.cpu.brand_string"),
        "core_count_total": _sysctl("hw.ncpu"),
        "core_count_perf": _sysctl("hw.perflevel0.logicalcpu_max"),
        "core_count_eff": _sysctl("hw.perflevel1.logicalcpu_max"),
        "memory_gb": "",
        "macos_version": platform.mac_ver()[0],
        "macos_build": platform.mac_ver()[2],
        "python_version": platform.python_version(),
        "coremltools_version": "",
        "anemll_version": "",
        "compute_target": "ANE (CPU_AND_NE)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        mem_bytes = int(_sysctl("hw.memsize"))
        info["memory_gb"] = round(mem_bytes / (1024**3), 1)
    except (ValueError, TypeError):
        pass

    try:
        import coremltools
        info["coremltools_version"] = coremltools.__version__
    except ImportError:
        info["coremltools_version"] = "not installed"

    try:
        import anemll
        info["anemll_version"] = getattr(anemll, "__version__", "unknown")
    except ImportError:
        info["anemll_version"] = "not installed"

    return info


# ---------------------------------------------------------------------------
# Resolve and slug
# ---------------------------------------------------------------------------

def resolve_model(name: str) -> Dict[str, str]:
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    raise ValueError(f"Unknown model '{name}'. Use: {list(MODEL_REGISTRY)}")


def slug_for(name: str) -> str:
    return name.replace("/", "_").replace(":", "-")


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def run_single_model(
    name: str,
    model_info: Dict[str, str],
    tasks: List[str],
    out_dir: Path,
    run_name: str,
    extra_args: List[str],
    measure_power: bool = False,
) -> Dict[str, Any]:
    """Evaluate one ANE model via subprocess."""
    slug = slug_for(name)
    hf_id = model_info["hf_id"]
    ane_dir = str(Path(model_info["ane_dir"]).resolve())

    # Verify ANE model exists
    if not Path(ane_dir).exists():
        print(f"  [SKIP] ANE model not found: {ane_dir}")
        print(f"  Run: python scripts/convert_ane_models.py --models {name}")
        return {"model": name, "status": "not_converted", "elapsed_s": 0}

    model_spec = f"ane_local:{name}"

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "eval_t_suite.py"),
        "--models", model_spec,
        "--tasks", *tasks,
        "--out-dir", str(out_dir),
        "--run-name", run_name,
        *extra_args,
    ]

    env = os.environ.copy()
    env["AOFW_PROVIDER"] = "ane_local"
    env["ANE_META_DIR"] = ane_dir
    env["ANE_HF_MODEL"] = hf_id
    project_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

    print(f"\n{'='*72}")
    print(f"  Model   : {name} ({hf_id})")
    print(f"  ANE Dir : {ane_dir}")
    print(f"  Tasks   : {len(tasks)} files")
    print(f"  Power   : {'enabled' if measure_power else 'disabled'}")
    print(f"{'='*72}\n")

    t0 = time.time()

    # Optional power measurement
    power_summary = None
    if measure_power:
        try:
            from agent_stable_slo.bench.power_monitor import PowerMonitor
            power_mon = PowerMonitor(interval_ms=100)
            power_mon.start()
        except Exception as e:
            print(f"  [WARN] Power monitoring unavailable: {e}")
            measure_power = False

    result = subprocess.run(cmd, env=env, capture_output=False)

    if measure_power:
        power_summary = power_mon.stop()
        power_path = out_dir / run_name / slug / "power_summary.json"
        power_path.parent.mkdir(parents=True, exist_ok=True)
        with open(power_path, "w") as f:
            json.dump(power_summary.to_dict(), f, indent=2)

    elapsed = time.time() - t0
    status = "ok" if result.returncode == 0 else "error"
    print(f"\n  --> {slug}: {status} ({elapsed:.1f}s)")

    entry: Dict[str, Any] = {
        "model": name,
        "hf_id": hf_id,
        "slug": slug,
        "status": status,
        "returncode": result.returncode,
        "elapsed_s": round(elapsed, 1),
    }
    if power_summary:
        entry["power"] = power_summary.to_dict()

    return entry


def main():
    ap = argparse.ArgumentParser(description="Run ANE eval suite on Apple Silicon models.")

    model_group = ap.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--models", nargs="+", help="Model short-names.")
    model_group.add_argument("--all", action="store_true", help="Run all registered models.")

    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--out-dir", default="results/ane_eval")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--measure-power", action="store_true", help="Capture power via powermetrics.")
    ap.add_argument("--slo-budget-ms", type=float, default=2000.0)
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--stability-runs", type=int, default=1)

    args = ap.parse_args()

    if args.all:
        model_list = list(MODEL_REGISTRY.items())
    else:
        model_list = [(m, resolve_model(m)) for m in args.models]

    for t in args.tasks:
        if not Path(t).exists():
            raise SystemExit(f"Task file not found: {t}")

    run_name = args.run_name or f"ane_eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hw_info = get_hardware_info()
    hw_path = out_dir / run_name
    hw_path.mkdir(parents=True, exist_ok=True)
    with open(hw_path / "hardware_info.json", "w") as f:
        json.dump(hw_info, f, indent=2)

    print(f"  Chip           : {hw_info['chip']}")
    print(f"  Memory         : {hw_info['memory_gb']} GB")
    print(f"  coremltools    : {hw_info['coremltools_version']}")
    print(f"  Compute Target : {hw_info['compute_target']}")
    print(f"  Models         : {len(model_list)}")

    extra_args: List[str] = []
    if args.stability_runs > 1:
        extra_args.extend(["--stability-runs", str(args.stability_runs)])
    if args.slo_budget_ms != 2000.0:
        extra_args.extend(["--slo-budget-ms", str(args.slo_budget_ms)])
    if args.max_records > 0:
        extra_args.extend(["--max-records", str(args.max_records)])

    run_log: List[Dict[str, Any]] = []
    overall_t0 = time.time()

    for i, (name, info) in enumerate(model_list, 1):
        print(f"\n[{i}/{len(model_list)}] Starting {name}...")
        summary = run_single_model(
            name=name, model_info=info, tasks=args.tasks,
            out_dir=out_dir, run_name=run_name,
            extra_args=extra_args, measure_power=args.measure_power,
        )
        run_log.append(summary)

    overall_elapsed = time.time() - overall_t0

    manifest = {
        "run_name": run_name,
        "models": [name for name, _ in model_list],
        "tasks": args.tasks,
        "hardware": hw_info,
        "results": run_log,
        "total_elapsed_s": round(overall_elapsed, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(hw_path / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*72}")
    print(f"  ANE Eval Complete -- {len(model_list)} models in {overall_elapsed:.0f}s")
    print(f"{'='*72}")
    for entry in run_log:
        pwr = f" | {entry['power']['mean_total_w']:.1f}W" if "power" in entry else ""
        print(f"  {entry['model']:<30} {entry['status']:<8} {entry['elapsed_s']:>7.1f}s{pwr}")

    failures = [e for e in run_log if e["status"] not in ("ok", "not_converted")]
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### Step 2: Commit

```bash
git add scripts/eval_ane_suite.py
git commit -m "feat(p9): add ANE evaluation harness with power measurement"
```

---

## Task 5: Hybrid ANE+MLX GRPO Trainer

**Files:**
- Create: `agent_stable_slo/train/ane_grpo_adapter.py`
- Test: `tests/test_ane_grpo.py`

### Step 1: Write failing tests

```python
# tests/test_ane_grpo.py
"""Tests for hybrid ANE+MLX GRPO trainer.

Mocks ANE inference and MLX gradient computation so tests run
without hardware dependencies.
"""

import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


class TestHybridGRPOConfig:
    """Verify config validation for hybrid trainer."""

    def test_requires_ane_meta_dir(self):
        from agent_stable_slo.train.ane_grpo_adapter import HybridGRPOConfig

        with pytest.raises((ValueError, TypeError)):
            HybridGRPOConfig(
                base_model="Qwen/Qwen3.5-0.8B",
                ane_meta_dir="",  # empty = invalid
                tasks=["tasks/clinc_en.jsonl"],
            )

    def test_valid_config(self):
        from agent_stable_slo.train.ane_grpo_adapter import HybridGRPOConfig

        cfg = HybridGRPOConfig(
            base_model="Qwen/Qwen3.5-0.8B",
            ane_meta_dir="/fake/ane/model",
            tasks=["tasks/clinc_en.jsonl"],
        )
        assert cfg.base_model == "Qwen/Qwen3.5-0.8B"
        assert cfg.batch_update_interval == 1


class TestWeightSync:
    """Verify weight sync logic between MLX and ANE."""

    def test_export_lora_weights_returns_dict(self):
        from agent_stable_slo.train.ane_grpo_adapter import _export_lora_weights

        mock_model = MagicMock()
        mock_model.trainable_parameters.return_value = [
            ("layers.0.lora_a", MagicMock(shape=(8, 768))),
            ("layers.0.lora_b", MagicMock(shape=(768, 8))),
        ]
        weights = _export_lora_weights(mock_model)
        assert isinstance(weights, dict)
        assert len(weights) == 2


class TestRolloutPhase:
    """Verify ANE rollout generation interface."""

    @patch("agent_stable_slo.rollout.providers.ane_local.generate_raw")
    def test_ane_rollout_returns_expected_shape(self, mock_gen):
        mock_gen.return_value = (
            '{"intent": "greet"}', {"intent": "greet"},
            50.0, 15.0, 10, 5,
        )
        from agent_stable_slo.train.ane_grpo_adapter import _ane_rollout

        result = _ane_rollout("Classify this", {"type": "object"})
        assert "text" in result
        assert "latency_ms" in result
        assert "parsed" in result
```

### Step 2: Run tests to verify failure

Run: `/Users/maloney/.local/share/mamba/bin/python -m pytest tests/test_ane_grpo.py -v`
Expected: FAIL

### Step 3: Implement hybrid GRPO trainer

```python
# agent_stable_slo/train/ane_grpo_adapter.py
"""Hybrid ANE+MLX GRPO trainer.

Uses Apple Neural Engine for rollout generation (inference) and
MLX/Metal GPU for gradient computation and LoRA weight updates.

The key experimental variable is the weight sync bottleneck:
after MLX updates LoRA weights, they must be re-converted to
CoreML format and reloaded on ANE.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent_stable_slo.rewards.composite import composite_reward
from agent_stable_slo.rewards.schema_reward import schema_valid


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class HybridGRPOConfig:
    """Configuration for hybrid ANE+MLX GRPO training."""

    # Model
    base_model: str                    # HuggingFace ID (for MLX + tokenizer)
    ane_meta_dir: str                  # Path to Anemll-converted model
    tasks: List[str]                   # Task JSONL files

    # GRPO
    num_steps: int = 200
    group_size: int = 4
    beta: float = 0.1                  # KL penalty

    # LoRA
    lora_rank: int = 8
    lora_layers: int = 16

    # Generation
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95

    # Reward weights
    lam_latency: float = 0.1
    mu_cost: float = 0.01

    # Training
    learning_rate: float = 1e-4
    seed: int = 42
    checkpoint_every: int = 50

    # Hybrid-specific
    batch_update_interval: int = 1     # Update ANE weights every N steps
    measure_power: bool = False

    # Paths
    adapter_path: str = ""
    log_path: str = ""

    def __post_init__(self):
        if not self.ane_meta_dir:
            raise ValueError("ane_meta_dir is required")
        if not self.tasks:
            raise ValueError("At least one task file is required")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _export_lora_weights(model) -> Dict[str, Any]:
    """Export LoRA adapter weights from MLX model as numpy arrays."""
    weights = {}
    for name, param in model.trainable_parameters():
        weights[name] = param
    return weights


def _ane_rollout(prompt: str, schema: dict, **kwargs) -> Dict[str, Any]:
    """Generate a single rollout using ANE inference.

    Delegates to the ane_local provider.
    """
    from agent_stable_slo.rollout.providers.ane_local import generate_raw

    raw_text, parsed, lat_ms, ttft_ms, tokens_in, tokens_out = generate_raw(
        prompt, schema, **kwargs
    )
    return {
        "text": raw_text,
        "parsed": parsed,
        "latency_ms": lat_ms,
        "ttft_ms": ttft_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }


def _parse_json(text: str, schema: dict) -> Dict[str, Any]:
    """Extract JSON from model output."""
    try:
        return json.loads(text)
    except Exception:
        pass

    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p_strip = p.strip()
            if p_strip.startswith("json"):
                p_strip = p_strip[4:].strip()
            try:
                return json.loads(p_strip)
            except Exception:
                continue

    matches = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)
    for m in matches:
        try:
            parsed = json.loads(m)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    if "answer" in schema.get("properties", {}):
        return {"answer": text.strip()}
    return {}


def _load_dataset(task_paths: List[str]) -> List[Dict[str, Any]]:
    """Load task records from JSONL files."""
    rows = []
    for path in task_paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                with open(rec["schema_path"]) as sf:
                    rec["schema"] = json.load(sf)
                rec["_source_task"] = path
                rows.append(rec)
    if not rows:
        raise ValueError(f"No records in {task_paths}")
    return rows


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class HybridGRPOTrainer:
    """GRPO trainer with ANE inference + MLX gradient updates.

    Training loop:
    1. Generate group rollouts on ANE (inference)
    2. Score with composite reward
    3. Compute GRPO loss and gradients on MLX (GPU)
    4. Update LoRA weights
    5. Periodically sync weights back to ANE (re-convert)

    The weight sync step is the key bottleneck being measured.
    """

    def __init__(self, cfg: HybridGRPOConfig):
        self.cfg = cfg
        self._setup_paths()
        self._load_data()

    def _setup_paths(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        if not self.cfg.adapter_path:
            self.adapter_dir = Path(f"out/ane_grpo_{ts}") / "adapter"
        else:
            self.adapter_dir = Path(self.cfg.adapter_path)
        self.adapter_dir.mkdir(parents=True, exist_ok=True)

        if not self.cfg.log_path:
            self.log_path = self.adapter_dir.parent / "train_log.jsonl"
        else:
            self.log_path = Path(self.cfg.log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_data(self):
        self.dataset = _load_dataset(self.cfg.tasks)
        print(f"[data] Loaded {len(self.dataset)} records from {len(self.cfg.tasks)} files")

    def _load_mlx_model(self):
        """Load model in MLX for gradient computation."""
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx_lm import load as mlx_load
        from mlx_lm.tuner.utils import linear_to_lora_layers

        print(f"[mlx] Loading {self.cfg.base_model} for gradient computation...")
        self.mlx_model, self.tokenizer = mlx_load(self.cfg.base_model)

        linear_to_lora_layers(
            model=self.mlx_model,
            num_layers=self.cfg.lora_layers,
            config={"rank": self.cfg.lora_rank, "scale": 20.0, "dropout": 0.0},
        )

        # Freeze MoE router gates
        for name, mod in self.mlx_model.named_modules():
            if name.endswith(".mlp.gate") and hasattr(mod, "lora_a"):
                mod.freeze()

        self.mlx_model.train()

        n_params = sum(p.size for _, p in self.mlx_model.trainable_parameters())
        n_total = sum(p.size for _, p in self.mlx_model.parameters())
        print(f"[mlx] Trainable: {n_params:,} / {n_total:,}")

        self.optimizer = optim.Adam(learning_rate=self.cfg.learning_rate)

    def _setup_ane(self):
        """Configure ANE environment for rollout generation."""
        os.environ["ANE_META_DIR"] = str(Path(self.cfg.ane_meta_dir).resolve())
        os.environ["ANE_HF_MODEL"] = self.cfg.base_model
        os.environ["ANE_MAX_TOKENS"] = str(self.cfg.max_tokens)
        print(f"[ane] Configured ANE meta dir: {self.cfg.ane_meta_dir}")

    def _generate_group_ane(self, prompt: str, schema: dict) -> List[Dict[str, Any]]:
        """Generate group rollouts using ANE inference."""
        outputs = []
        for _ in range(self.cfg.group_size):
            rollout = _ane_rollout(
                prompt, schema,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
            )
            outputs.append(rollout)
        return outputs

    def _compute_mlx_loss(self, prompt: str, group: List[Dict], advantages: List[float]):
        """Compute GRPO policy gradient loss on MLX."""
        import mlx.core as mx

        prompt_tokens = mx.array(
            self.tokenizer.encode(prompt, add_special_tokens=False)
        )

        total_loss = mx.array(0.0)
        n = len(group)

        for out, adv in zip(group, advantages):
            gen_tokens = mx.array(
                self.tokenizer.encode(out["text"], add_special_tokens=False) or [0]
            )

            full_tokens = mx.concatenate([prompt_tokens, gen_tokens], axis=-1)
            logits = self.mlx_model(full_tokens[None, :])[0]

            prompt_len = prompt_tokens.shape[0]
            gen_logits = logits[prompt_len - 1:-1, :]
            log_probs = gen_logits - mx.logsumexp(gen_logits, axis=-1, keepdims=True)

            gen_len = gen_tokens.shape[0]
            token_log_probs = log_probs[mx.arange(gen_len), gen_tokens]
            log_prob_sum = token_log_probs.sum()

            pg_loss = -adv * log_prob_sum
            kl_term = self.cfg.beta * log_prob_sum
            total_loss = total_loss + pg_loss + kl_term

        return total_loss / max(1, n)

    def _sync_weights_to_ane(self, step: int) -> float:
        """Export MLX LoRA weights and reload on ANE.

        Returns sync latency in milliseconds.
        This is the key bottleneck being measured.
        """
        t0 = time.time()

        # Export LoRA weights
        lora_weights = _export_lora_weights(self.mlx_model)

        # TODO: Merge LoRA into base weights and re-convert to CoreML
        # For now, measure just the export time.
        # Full pipeline: MLX LoRA -> merge -> coremltools convert -> ANE reload
        # This will be implemented iteratively as we understand the conversion API.

        sync_ms = (time.time() - t0) * 1000.0
        print(f"  [sync] Weight export: {sync_ms:.1f}ms (full conversion TBD)")
        return sync_ms

    def run(self) -> Path:
        """Execute the hybrid GRPO training loop."""
        random.seed(self.cfg.seed)

        self._load_mlx_model()
        self._setup_ane()

        import mlx.core as mx
        import mlx.nn as nn

        loss_and_grad = nn.value_and_grad(self.mlx_model, self._step_loss_fn)

        print(f"[train] Starting {self.cfg.num_steps} steps, group_size={self.cfg.group_size}")
        print(f"[train] Rollouts: ANE | Gradients: MLX | Sync every {self.cfg.batch_update_interval} steps")
        train_start = time.time()

        step_rewards = []
        power_mon = None

        if self.cfg.measure_power:
            try:
                from agent_stable_slo.bench.power_monitor import PowerMonitor
                power_mon = PowerMonitor(interval_ms=100)
                power_mon.start()
            except Exception as e:
                print(f"[warn] Power monitoring unavailable: {e}")

        with open(self.log_path, "w") as log_file:
            for step in range(self.cfg.num_steps):
                step_t0 = time.time()

                # Sample task
                row = random.choice(self.dataset)
                prompt = row["prompt"]
                schema = row["schema"]

                # Phase 1: ANE rollouts
                ane_t0 = time.time()
                group = self._generate_group_ane(prompt, schema)
                ane_ms = (time.time() - ane_t0) * 1000.0

                # Phase 2: Score
                rewards = []
                for out in group:
                    parsed = out.get("parsed") or _parse_json(out["text"], schema)
                    jv = schema_valid(parsed, schema)
                    r = composite_reward(
                        parsed, schema, ok_success=jv,
                        latency_ms=out["latency_ms"],
                        tokens=out["tokens_out"],
                        lam_latency=self.cfg.lam_latency,
                        mu_cost=self.cfg.mu_cost,
                    )
                    rewards.append(float(r))
                    out["reward"] = float(r)
                    out["json_valid"] = int(jv)

                mean_reward = sum(rewards) / max(1, len(rewards))
                advantages = [r - mean_reward for r in rewards]

                # Phase 3: MLX gradient update
                self._current_prompt = prompt
                self._current_group = group
                self._current_advantages = advantages

                mlx_t0 = time.time()
                loss_val, grads = loss_and_grad(self.mlx_model)
                self.optimizer.update(self.mlx_model, grads)
                mx.eval(loss_val)
                mx.eval(self.mlx_model.parameters())
                mlx_ms = (time.time() - mlx_t0) * 1000.0

                # Phase 4: Weight sync (periodic)
                sync_ms = 0.0
                if (step + 1) % self.cfg.batch_update_interval == 0:
                    sync_ms = self._sync_weights_to_ane(step)

                step_ms = (time.time() - step_t0) * 1000.0

                # Log
                best_idx = rewards.index(max(rewards))
                log_rec = {
                    "step": step,
                    "reward": rewards[best_idx],
                    "mean_reward": round(mean_reward, 4),
                    "json_valid": group[best_idx].get("json_valid", 0),
                    "loss": round(float(loss_val), 6),
                    "step_ms": round(step_ms, 1),
                    "ane_rollout_ms": round(ane_ms, 1),
                    "mlx_gradient_ms": round(mlx_ms, 1),
                    "weight_sync_ms": round(sync_ms, 1),
                    "group_rewards": [round(r, 4) for r in rewards],
                }
                log_file.write(json.dumps(log_rec) + "\n")
                log_file.flush()

                step_rewards.append(mean_reward)

                if (step + 1) % 10 == 0:
                    recent = step_rewards[-10:]
                    avg_r = sum(recent) / len(recent)
                    print(
                        f"[step {step + 1}/{self.cfg.num_steps}] "
                        f"reward={avg_r:.3f} loss={float(loss_val):.4f} "
                        f"ane={ane_ms:.0f}ms mlx={mlx_ms:.0f}ms sync={sync_ms:.0f}ms"
                    )

                # Checkpoint
                if self.cfg.checkpoint_every > 0 and (step + 1) % self.cfg.checkpoint_every == 0:
                    self._save_checkpoint(step)

                # Memory cleanup
                if (step + 1) % 50 == 0:
                    _clear = getattr(mx, "clear_cache", None) or getattr(mx.metal, "clear_cache", None)
                    if _clear:
                        _clear()

        # Final checkpoint
        self._save_checkpoint(self.cfg.num_steps - 1, final=True)

        # Stop power monitor
        power_summary = None
        if power_mon:
            power_summary = power_mon.stop()
            with open(self.adapter_dir.parent / "power_summary.json", "w") as f:
                json.dump(power_summary.to_dict(), f, indent=2)

        elapsed = time.time() - train_start
        print(
            f"[done] {self.cfg.num_steps} steps in {elapsed/60:.1f}min. "
            f"Adapter: {self.adapter_dir}"
        )
        return self.adapter_dir

    def _step_loss_fn(self, model):
        """Loss function closure for nn.value_and_grad."""
        import mlx.core as mx

        prompt_tokens = mx.array(
            self.tokenizer.encode(self._current_prompt, add_special_tokens=False)
        )

        total_loss = mx.array(0.0)
        n = len(self._current_group)

        for out, adv in zip(self._current_group, self._current_advantages):
            toks = self.tokenizer.encode(out["text"], add_special_tokens=False)
            if not toks:
                toks = [self.tokenizer.eos_token_id or 0]
            gen_tokens = mx.array(toks)

            full_tokens = mx.concatenate([prompt_tokens, gen_tokens], axis=-1)
            logits = model(full_tokens[None, :])[0]

            prompt_len = prompt_tokens.shape[0]
            gen_logits = logits[prompt_len - 1:-1, :]
            log_probs = gen_logits - mx.logsumexp(gen_logits, axis=-1, keepdims=True)

            gen_len = gen_tokens.shape[0]
            token_log_probs = log_probs[mx.arange(gen_len), gen_tokens]
            log_prob_sum = token_log_probs.sum()

            pg_loss = -adv * log_prob_sum
            kl_term = self.cfg.beta * log_prob_sum
            total_loss = total_loss + pg_loss + kl_term

        return total_loss / max(1, n)

    def _save_checkpoint(self, step: int, final: bool = False):
        """Save LoRA adapter weights."""
        import mlx.core as mx

        tag = "final" if final else f"step_{step + 1}"
        ckpt_dir = self.adapter_dir if final else self.adapter_dir / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        weights = {}
        for name, param in self.mlx_model.trainable_parameters():
            weights[name] = param
        mx.savez(str(ckpt_dir / "adapters.npz"), **weights)

        meta = {
            "step": step,
            "base_model": self.cfg.base_model,
            "ane_meta_dir": self.cfg.ane_meta_dir,
            "lora_rank": self.cfg.lora_rank,
            "lora_layers": self.cfg.lora_layers,
            "trainer": "hybrid_ane_mlx",
        }
        with open(ckpt_dir / "adapter_config.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[checkpoint] Saved {tag} to {ckpt_dir}")
```

### Step 4: Run tests

Run: `/Users/maloney/.local/share/mamba/bin/python -m pytest tests/test_ane_grpo.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add agent_stable_slo/train/ane_grpo_adapter.py tests/test_ane_grpo.py
git commit -m "feat(p9): add hybrid ANE+MLX GRPO trainer"
```

---

## Task 6: Paper Scaffold

**Files:**
- Create: `papers/P9_ane_heterogeneous/arxiv/main.tex`
- Create: `papers/P9_ane_heterogeneous/arxiv/refs.bib`

### Step 1: Create paper directory and scaffold

Create `papers/P9_ane_heterogeneous/arxiv/` directory.

Write `main.tex` with:
- Title: "Neural Engine Training for SLO-Aware Agents: Heterogeneous Compute on Apple Silicon"
- Abstract placeholder
- RQ1-RQ4 section structure
- Result table placeholders (ANE vs MLX comparison, hybrid GRPO metrics)
- Self-citations to P1-P8

Write `refs.bib` with self-citations to the 8 prior papers.

### Step 2: Commit

```bash
git add papers/P9_ane_heterogeneous/
git commit -m "feat(p9): scaffold paper directory with RQ structure"
```

---

## Task 7: Run Existing Tests (Regression Check)

### Step 1: Run full test suite

Run: `/Users/maloney/.local/share/mamba/bin/python -m pytest tests/ -v --tb=short`
Expected: All 102+ existing tests PASS, plus new ANE tests PASS

### Step 2: Fix any regressions

If engine.py changes break existing provider routing tests, fix the dispatch ordering.

---

## Task 8: Smoke Test (Integration)

### Step 1: Install Anemll dependencies

```bash
/Users/maloney/.local/share/mamba/bin/pip install anemll coremltools>=9.0 pyyaml
```

### Step 2: Convert Qwen3.5-0.8B to ANE format

```bash
/Users/maloney/.local/share/mamba/bin/python scripts/convert_ane_models.py --models qwen3.5-0.8b
```

### Step 3: Smoke test ANE provider

```bash
ANE_META_DIR=models/ane/qwen3.5-0.8b ANE_HF_MODEL=Qwen/Qwen3.5-0.8B AOFW_PROVIDER=ane_local \
  /Users/maloney/.local/share/mamba/bin/python -c "
from agent_stable_slo.rollout.engine import _provider_generate_raw
result = _provider_generate_raw('What is 2+2?', {'type': 'object', 'properties': {'answer': {'type': 'string'}}, 'required': ['answer']}, 'structured', 0.0, None)
print('Result:', result[:2])
print('Latency:', result[2], 'ms')
"
```

### Step 4: Commit final state

```bash
git add -A
git commit -m "feat(p9): complete P9 ANE infrastructure scaffold"
```
