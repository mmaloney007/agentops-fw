"""
MoE (Mixture-of-Experts) profiler for MLX models on Apple Silicon.

Detects dense vs. MoE architectures, extracts parameter counts, and measures
runtime memory / throughput via Metal GPU.

Importable without mlx installed -- all MLX imports are guarded.
"""

from __future__ import annotations

import platform
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MoEProfile:
    """Profile of a single model's architecture and runtime characteristics."""

    model_name: str
    total_params: int = 0
    active_params: int = 0
    num_experts: int = 0
    num_active: int = 0
    architecture_type: str = "dense"  # "dense" or "moe"
    peak_memory_gb: float = 0.0
    tokens_per_second: float = 0.0
    memory_stats: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Architecture detection helpers
# ---------------------------------------------------------------------------


def is_moe(config: dict) -> bool:
    """Return True if a model config indicates MoE architecture.

    Checks for common MoE config keys across Qwen, Mixtral, DeepSeek, etc.
    """
    moe_keys = [
        "num_experts",
        "num_local_experts",
        "num_experts_per_tok",
        "num_experts_per_token",
        "moe_num_experts",
        "n_routed_experts",
    ]
    for key in moe_keys:
        val = config.get(key)
        if val is not None and int(val) > 1:
            return True

    # Some configs nest MoE info under model_type or architectures
    model_type = config.get("model_type", "").lower()
    if "moe" in model_type or "mixture" in model_type:
        return True

    return False


def _extract_expert_counts(config: dict) -> tuple[int, int]:
    """Extract (total_experts, active_experts) from config.

    Returns (0, 0) for dense models.
    """
    total = 0
    active = 0

    # Total experts
    for key in ("num_experts", "num_local_experts", "moe_num_experts", "n_routed_experts"):
        val = config.get(key)
        if val is not None:
            total = int(val)
            break

    # Active experts per token
    for key in (
        "num_experts_per_tok",
        "num_experts_per_token",
        "moe_top_k",
        "num_selected_experts",
        "top_k",
    ):
        val = config.get(key)
        if val is not None:
            active = int(val)
            break

    # Fallback: if we found total but not active, assume top-2
    if total > 0 and active == 0:
        active = min(2, total)

    return total, active


def _count_params_from_config(config: dict) -> tuple[int, int]:
    """Estimate total and active parameter counts from config fields.

    Returns (total_params, active_params). Active == total for dense models.
    """
    # Some configs expose this directly
    if "num_parameters" in config:
        total = int(config["num_parameters"])
    elif "n_params" in config:
        total = int(config["n_params"])
    else:
        total = 0

    active = total  # default for dense
    num_experts, num_active = _extract_expert_counts(config)
    if num_experts > 1 and total > 0:
        # Rough heuristic: MoE params ~ shared + (num_active/num_experts) * expert_block
        # Without full layer breakdown, approximate active = total * (active/experts)
        # This is a simplification; the shared params (attention, embeddings) are always active.
        ratio = num_active / num_experts
        # Assume ~50% of total params are in expert FFN blocks
        expert_fraction = 0.5
        active = int(total * (1 - expert_fraction + expert_fraction * ratio))

    return total, active


# ---------------------------------------------------------------------------
# Hardware info
# ---------------------------------------------------------------------------


def _sysctl(key: str) -> str:
    """Read a sysctl value; return empty string on failure."""
    try:
        return (
            subprocess.check_output(["sysctl", "-n", key], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return ""


def get_hardware_info() -> Dict[str, Any]:
    """Return chip, cores, memory, macOS version, and MLX version."""
    info: Dict[str, Any] = {
        "chip": _sysctl("machdep.cpu.brand_string"),
        "core_count_total": _sysctl("hw.ncpu"),
        "core_count_perf": _sysctl("hw.perflevel0.logicalcpu_max"),
        "core_count_eff": _sysctl("hw.perflevel1.logicalcpu_max"),
        "gpu_core_count": _sysctl("machdep.cpu.core_count")
        or _sysctl("hw.perflevel0.physicalcpu_max"),
        "memory_gb": "",
        "macos_version": platform.mac_ver()[0],
        "macos_build": platform.mac_ver()[2],
    }

    try:
        mem_bytes = int(_sysctl("hw.memsize"))
        info["memory_gb"] = round(mem_bytes / (1024**3), 1)
    except (ValueError, TypeError):
        pass

    try:
        import mlx

        info["mlx_version"] = getattr(mlx, "__version__", "unknown")
    except ImportError:
        info["mlx_version"] = "not installed"

    try:
        import mlx_lm

        info["mlx_lm_version"] = getattr(mlx_lm, "__version__", "unknown")
    except ImportError:
        info["mlx_lm_version"] = "not installed"

    return info


# ---------------------------------------------------------------------------
# Metal memory helpers
# ---------------------------------------------------------------------------


def _get_memory_stats() -> Dict[str, float]:
    """Snapshot Metal GPU memory (Apple Silicon only)."""
    try:
        import mlx.core as mx

        # Use non-deprecated API if available (mlx >= 0.28)
        _get_active = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
        _get_peak = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
        _get_cache = getattr(mx, "get_cache_memory", None) or mx.metal.get_cache_memory
        return {
            "active_memory_gb": _get_active() / (1024**3),
            "peak_memory_gb": _get_peak() / (1024**3),
            "cache_memory_gb": _get_cache() / (1024**3),
        }
    except Exception:
        return {}


def _reset_peak_memory() -> None:
    """Reset the peak-memory watermark so the next profile starts clean."""
    try:
        import mlx.core as mx

        _reset = getattr(mx, "reset_peak_memory", None) or mx.metal.reset_peak_memory
        _reset()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Profile entry point
# ---------------------------------------------------------------------------


def profile_model(
    model_path: str,
    prompt: str = "Hello, how are you?",
    max_tokens: int = 64,
    warmup_tokens: int = 8,
) -> MoEProfile:
    """Load an MLX model, detect its architecture, and measure performance.

    Parameters
    ----------
    model_path : str
        HuggingFace repo ID or local path (e.g. ``mlx-community/Qwen3.5-35B-A3B-4bit``).
    prompt : str
        Prompt used for the throughput benchmark.
    max_tokens : int
        Tokens to generate during the benchmark pass.
    warmup_tokens : int
        Tokens to generate in a throw-away warmup pass (stabilises Metal caches).

    Returns
    -------
    MoEProfile
        Populated profile dataclass.
    """
    from mlx_lm import load, generate as mlx_generate

    profile = MoEProfile(model_name=model_path)

    # -- Load model + tokenizer ------------------------------------------------
    model, tokenizer = load(model_path)

    # -- Read config -----------------------------------------------------------
    # mlx-lm models expose config either on model.config or via the loaded JSON.
    config: dict = {}
    if hasattr(model, "config"):
        cfg_obj = model.config
        if isinstance(cfg_obj, dict):
            config = cfg_obj
        elif hasattr(cfg_obj, "to_dict"):
            config = cfg_obj.to_dict()
        elif hasattr(cfg_obj, "__dict__"):
            config = dict(vars(cfg_obj))
    # Fallback: try loading config.json from the HF cache
    if not config:
        try:
            from huggingface_hub import hf_hub_download
            import json

            cfg_path = hf_hub_download(model_path, "config.json")
            with open(cfg_path, "r") as f:
                config = json.load(f)
        except Exception:
            pass

    # -- Architecture detection ------------------------------------------------
    moe_detected = is_moe(config)
    profile.architecture_type = "moe" if moe_detected else "dense"

    num_experts, num_active = _extract_expert_counts(config)
    profile.num_experts = num_experts
    profile.num_active = num_active

    total_params, active_params = _count_params_from_config(config)
    profile.total_params = total_params
    profile.active_params = active_params

    # If we can count parameters directly from the model weights, prefer that
    try:
        import mlx.core as mx

        param_count = sum(
            p.size for p in _iter_params(model)
        )
        if param_count > 0:
            profile.total_params = param_count
            if moe_detected and num_experts > 0:
                ratio = num_active / num_experts
                expert_fraction = 0.5
                profile.active_params = int(
                    param_count * (1 - expert_fraction + expert_fraction * ratio)
                )
            else:
                profile.active_params = param_count
    except Exception:
        pass

    # -- Warmup pass -----------------------------------------------------------
    _reset_peak_memory()

    # Format with chat template
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted = prompt

    # Warmup (don't measure)
    _ = mlx_generate(model, tokenizer, prompt=formatted, max_tokens=warmup_tokens, verbose=False)

    # -- Benchmark pass --------------------------------------------------------
    _reset_peak_memory()

    t0 = time.perf_counter()
    output = mlx_generate(model, tokenizer, prompt=formatted, max_tokens=max_tokens, verbose=False)
    elapsed = time.perf_counter() - t0

    # Count generated tokens
    output_ids = tokenizer.encode(output)
    gen_tokens = len(output_ids)
    profile.tokens_per_second = gen_tokens / elapsed if elapsed > 0 else 0.0

    # -- Memory stats ----------------------------------------------------------
    mem = _get_memory_stats()
    profile.memory_stats = mem
    profile.peak_memory_gb = mem.get("peak_memory_gb", 0.0)

    return profile


def _iter_params(model) -> list:
    """Recursively collect all MLX array leaves from a model tree."""
    import mlx.core as mx

    params = []

    def _walk(obj):
        if isinstance(obj, mx.array):
            params.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _walk(v)
        elif hasattr(obj, "parameters"):
            _walk(obj.parameters())

    try:
        _walk(model.parameters())
    except Exception:
        pass

    return params


# ---------------------------------------------------------------------------
# CLI entry point (for quick standalone profiling)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json as _json

    ap = argparse.ArgumentParser(description="Profile an MLX model.")
    ap.add_argument("model", help="HuggingFace model ID or local path.")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--prompt", default="Explain quantum computing in two sentences.")
    ap.add_argument("--json", action="store_true", help="Print profile as JSON.")
    args = ap.parse_args()

    print(f"Profiling {args.model} ...")
    prof = profile_model(args.model, prompt=args.prompt, max_tokens=args.max_tokens)

    if args.json:
        from dataclasses import asdict

        print(_json.dumps(asdict(prof), indent=2))
    else:
        print(f"  Model            : {prof.model_name}")
        print(f"  Architecture     : {prof.architecture_type}")
        print(f"  Total params     : {prof.total_params:,}")
        print(f"  Active params    : {prof.active_params:,}")
        print(f"  Experts          : {prof.num_experts} total, {prof.num_active} active")
        print(f"  Peak memory      : {prof.peak_memory_gb:.2f} GB")
        print(f"  Tokens/sec       : {prof.tokens_per_second:.1f}")
        print(f"  Memory stats     : {prof.memory_stats}")

    hw = get_hardware_info()
    print(f"\n  Hardware: {hw['chip']} / {hw['memory_gb']} GB / macOS {hw['macos_version']}")
    print(f"  MLX {hw.get('mlx_version', '?')} / mlx-lm {hw.get('mlx_lm_version', '?')}")
