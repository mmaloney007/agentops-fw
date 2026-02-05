"""Weave op wrappers for atomic-level instrumentation.

Provides @weave.op() wrapped versions of core functions for full traceability:
- Inference: llm_generate, validate_json, compute_reward
- Training: train_step, reward_breakdown

Usage:
    # Enable weave tracing for inference
    from agent_stable_slo.logging.weave_ops import enable_weave_tracing
    enable_weave_tracing("my-project")

    # Now all provider_generate calls are traced
    from agent_stable_slo.rollout.engine import provider_generate
    result = provider_generate(prompt, schema)  # Automatically traced!

The module uses monkey-patching to wrap core functions when enabled,
so existing code doesn't need modification.
"""
from __future__ import annotations

import json
import os
import time
from functools import wraps
from typing import Any, Dict, Optional, Tuple

# Lazy weave import
_weave = None
_weave_enabled = False


def _get_weave():
    """Lazy import weave."""
    global _weave
    if _weave is None:
        try:
            import weave
            _weave = weave
        except ImportError:
            _weave = False
    return _weave if _weave else None


def _check_wandb_auth() -> bool:
    """Check if wandb is authenticated (env var or .netrc)."""
    if os.getenv("WANDB_API_KEY"):
        return True
    # Check if wandb can authenticate via .netrc or other methods
    try:
        import wandb
        # wandb.login(anonymous="must") returns True if logged in
        return wandb.login(anonymous="must")
    except Exception:
        return False


def enable_weave_tracing(project: str) -> bool:
    """Enable weave tracing for all instrumented functions.

    Args:
        project: Weave project name (e.g., "agentslo-inference")

    Returns:
        True if weave was initialized successfully, False otherwise.
    """
    global _weave_enabled

    weave = _get_weave()
    if not weave:
        print("[weave] weave not installed, tracing disabled")
        return False

    if not _check_wandb_auth():
        print("[weave] Not authenticated to W&B (set WANDB_API_KEY or run wandb login)")
        return False

    try:
        weave.init(project)
        _weave_enabled = True
        _patch_inference_functions()
        print(f"[weave] Tracing enabled for project: {project}")
        return True
    except Exception as e:
        print(f"[weave] Failed to initialize: {e}")
        return False


def is_weave_enabled() -> bool:
    """Check if weave tracing is currently enabled."""
    return _weave_enabled


# ---------------------------------------------------------------------------
# Inference Ops
# ---------------------------------------------------------------------------

def _create_llm_generate_op():
    """Create a weave.op wrapped LLM generate function."""
    weave = _get_weave()
    if not weave:
        return None

    @weave.op()
    def llm_generate(
        prompt: str,
        schema: Dict[str, Any],
        mode: str = "structured",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        provider: str = "unknown",
        model: str = "unknown",
    ) -> Dict[str, Any]:
        """Traced LLM generation call.

        Returns a dict with all generation metadata for Weave logging.
        """
        from agent_stable_slo.rollout.engine import _provider_generate_raw

        t0 = time.time()
        raw_text, parsed, lat_ms, ttft_ms, tokens_in, tokens_out, logprobs = (
            _provider_generate_raw(prompt, schema, mode, temperature, max_tokens, False)
        )
        wall_time = (time.time() - t0) * 1000

        # Validate JSON
        from jsonschema import validate, ValidationError
        json_valid = True
        schema_error = None
        try:
            if parsed:
                validate(parsed, schema)
        except ValidationError as e:
            json_valid = False
            schema_error = str(e.message)[:200]
        except Exception:
            json_valid = False

        return {
            "raw_text": raw_text[:500] if raw_text else "",  # Truncate for logging
            "parsed_json": parsed,
            "json_valid": json_valid,
            "schema_error": schema_error,
            "latency_ms": lat_ms,
            "ttft_ms": ttft_ms,
            "wall_time_ms": wall_time,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "provider": provider,
            "model": model,
            "mode": mode,
            "temperature": temperature,
        }

    return llm_generate


def _create_compute_reward_op():
    """Create a weave.op wrapped reward computation."""
    weave = _get_weave()
    if not weave:
        return None

    @weave.op()
    def compute_reward(
        output_json: Dict[str, Any],
        schema: Dict[str, Any],
        ok_success: int,
        latency_ms: float,
        tokens: int,
        lam_latency: float = 0.1,
        mu_cost: float = 0.05,
        disagreement_rate: float = 0.0,
        gamma_stability: float = 0.1,
        faithfulness: float = 1.0,
        kappa_faithfulness: float = 0.0,
    ) -> Dict[str, Any]:
        """Traced reward computation with component breakdown."""
        from agent_stable_slo.rewards.composite import composite_reward
        from agent_stable_slo.rewards.schema_reward import schema_valid
        from agent_stable_slo.rewards.slo_reward import latency_penalty, cost_penalty
        from agent_stable_slo.rewards.stability_reward import stability_penalty

        # Compute individual components
        r_schema = float(schema_valid(output_json, schema))
        r_success = float(ok_success)
        r_latency = latency_penalty(latency_ms, lam_latency)
        r_cost = cost_penalty(tokens, mu_cost)
        r_stability = stability_penalty(disagreement_rate, gamma_stability)
        r_faithfulness = kappa_faithfulness * (faithfulness - 0.5)

        total = composite_reward(
            output_json, schema, ok_success, latency_ms, tokens,
            lam_latency, mu_cost, disagreement_rate, gamma_stability,
            faithfulness, kappa_faithfulness
        )

        return {
            "total_reward": total,
            "r_schema": r_schema,
            "r_success": r_success,
            "r_latency": r_latency,
            "r_cost": r_cost,
            "r_stability": r_stability,
            "r_faithfulness": r_faithfulness,
            "latency_ms": latency_ms,
            "tokens": tokens,
            "json_valid": bool(r_schema),
        }

    return compute_reward


# ---------------------------------------------------------------------------
# Training Ops
# ---------------------------------------------------------------------------

def _create_train_step_op():
    """Create a weave.op for training step logging."""
    weave = _get_weave()
    if not weave:
        return None

    @weave.op()
    def train_step(
        step: int,
        prompt: str,
        output_text: str,
        output_json: Optional[Dict[str, Any]],
        reward: float,
        advantage: float,
        loss: float,
        latency_ms: float,
        json_valid: bool,
        tokens_out: int,
        faithfulness: float = 1.0,
        disagreement_rate: float = 0.0,
        model_name: str = "unknown",
        task_type: str = "unknown",
    ) -> Dict[str, Any]:
        """Log a single GRPO training step with full context."""
        return {
            "step": step,
            "prompt": prompt[:200],  # Truncate for logging
            "output_text": output_text[:200],
            "output_json": output_json,
            "reward": reward,
            "advantage": advantage,
            "loss": loss,
            "latency_ms": latency_ms,
            "json_valid": json_valid,
            "tokens_out": tokens_out,
            "faithfulness": faithfulness,
            "disagreement_rate": disagreement_rate,
            "model_name": model_name,
            "task_type": task_type,
        }

    return train_step


# ---------------------------------------------------------------------------
# Monkey-patching for transparent instrumentation
# ---------------------------------------------------------------------------

_original_provider_generate = None
_original_provider_generate_raw = None


def _patch_inference_functions():
    """Patch inference functions to use weave-traced versions."""
    global _original_provider_generate, _original_provider_generate_raw

    weave = _get_weave()
    if not weave or not _weave_enabled:
        return

    from agent_stable_slo.rollout import engine

    # Save originals
    _original_provider_generate = engine.provider_generate
    _original_provider_generate_raw = engine.provider_generate_raw

    # Create traced wrapper
    llm_generate_op = _create_llm_generate_op()

    if llm_generate_op:
        @wraps(engine.provider_generate)
        def traced_provider_generate(
            prompt: str,
            schema: Dict[str, Any],
            mode: str | None = None,
            temperature: float = 0.0,
            max_tokens: Optional[int] = None,
            request_logprobs: bool = False,
        ) -> Tuple[Dict[str, Any], float, float, int, Optional[Dict[str, Any]]]:
            """Weave-traced provider_generate."""
            provider = os.getenv("AOFW_PROVIDER", "lmstudio")
            model = os.getenv("LMSTUDIO_MODEL", os.getenv("VLLM_MODEL", "unknown"))
            decode_mode = mode or os.getenv("DECODE_MODE", "structured")

            # Call the traced op
            result = llm_generate_op(
                prompt=prompt,
                schema=schema,
                mode=decode_mode,
                temperature=temperature,
                max_tokens=max_tokens,
                provider=provider,
                model=model,
            )

            # Return in original format
            return (
                result["parsed_json"],
                result["latency_ms"],
                result["ttft_ms"],
                result["tokens_out"],
                None,  # logprobs not traced yet
            )

        engine.provider_generate = traced_provider_generate
        print("[weave] Patched provider_generate for tracing")


def _unpatch_inference_functions():
    """Restore original inference functions."""
    global _original_provider_generate, _original_provider_generate_raw

    from agent_stable_slo.rollout import engine

    if _original_provider_generate:
        engine.provider_generate = _original_provider_generate
        _original_provider_generate = None
    if _original_provider_generate_raw:
        engine.provider_generate_raw = _original_provider_generate_raw
        _original_provider_generate_raw = None


# ---------------------------------------------------------------------------
# Training integration helper
# ---------------------------------------------------------------------------

def log_train_step(
    step: int,
    prompt: str,
    output_text: str,
    output_json: Optional[Dict[str, Any]],
    reward: float,
    advantage: float,
    loss: float,
    latency_ms: float,
    json_valid: bool,
    tokens_out: int,
    faithfulness: float = 1.0,
    disagreement_rate: float = 0.0,
    model_name: str = "unknown",
    task_type: str = "unknown",
) -> Optional[Dict[str, Any]]:
    """Log a training step to weave if enabled.

    Call this from the training loop to get atomic step-level tracing.
    Returns the logged data or None if weave is not enabled.
    """
    if not _weave_enabled:
        return None

    train_step_op = _create_train_step_op()
    if not train_step_op:
        return None

    return train_step_op(
        step=step,
        prompt=prompt,
        output_text=output_text,
        output_json=output_json,
        reward=reward,
        advantage=advantage,
        loss=loss,
        latency_ms=latency_ms,
        json_valid=json_valid,
        tokens_out=tokens_out,
        faithfulness=faithfulness,
        disagreement_rate=disagreement_rate,
        model_name=model_name,
        task_type=task_type,
    )


def log_reward_breakdown(
    output_json: Dict[str, Any],
    schema: Dict[str, Any],
    ok_success: int,
    latency_ms: float,
    tokens: int,
    lam_latency: float = 0.1,
    mu_cost: float = 0.05,
    disagreement_rate: float = 0.0,
    gamma_stability: float = 0.1,
    faithfulness: float = 1.0,
    kappa_faithfulness: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """Log reward computation with component breakdown to weave.

    Call this to get detailed reward decomposition in the Weave UI.
    Returns the breakdown dict or None if weave is not enabled.
    """
    if not _weave_enabled:
        return None

    compute_reward_op = _create_compute_reward_op()
    if not compute_reward_op:
        return None

    return compute_reward_op(
        output_json=output_json,
        schema=schema,
        ok_success=ok_success,
        latency_ms=latency_ms,
        tokens=tokens,
        lam_latency=lam_latency,
        mu_cost=mu_cost,
        disagreement_rate=disagreement_rate,
        gamma_stability=gamma_stability,
        faithfulness=faithfulness,
        kappa_faithfulness=kappa_faithfulness,
    )
