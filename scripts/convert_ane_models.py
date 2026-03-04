#!/usr/bin/env python3
"""
ANE model conversion wrapper -- converts HuggingFace models to Anemll/CoreML
format for Apple Neural Engine inference.

Locates Anemll's convert_model.sh, then runs it for each requested model.
Converted models land in models/ane/<short-name>/ and are consumed by the
ane_local.py provider via the ANE_META_DIR environment variable.

Usage:
  python scripts/convert_ane_models.py --all
  python scripts/convert_ane_models.py --models qwen3.5-0.8b qwen3.5-2b
  python scripts/convert_ane_models.py --model Qwen/Qwen3.5-4B
  python scripts/convert_ane_models.py --models qwen3.5-4b --context 1024
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
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Model registry: short-name -> HuggingFace model ID
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, str] = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "gemma-3-1b": "google/gemma-3-1b-it",
}

# ---------------------------------------------------------------------------
# Default conversion parameters
# ---------------------------------------------------------------------------

DEFAULT_CONTEXT = 512
DEFAULT_BATCH = 64
DEFAULT_LUT1 = 6   # embeddings
DEFAULT_LUT2 = 4   # FFN
DEFAULT_LUT3 = 6   # LM head

# ---------------------------------------------------------------------------
# Locate Anemll converter
# ---------------------------------------------------------------------------


def find_anemll_converter() -> Path:
    """Locate Anemll's convert_model.sh script.

    Search order:
    1. ANEMLL_HOME environment variable
    2. Installed anemll package location
    3. Common clone paths: ~/, ~/Projects/

    Raises FileNotFoundError with a helpful message if not found.
    """
    script_name = "convert_model.sh"

    # 1. ANEMLL_HOME env var
    anemll_home = os.environ.get("ANEMLL_HOME", "")
    if anemll_home:
        candidate = Path(anemll_home) / script_name
        if candidate.is_file():
            return candidate
        # Also check a scripts/ subdirectory
        candidate = Path(anemll_home) / "scripts" / script_name
        if candidate.is_file():
            return candidate

    # 2. Installed anemll package
    try:
        import importlib.util
        spec = importlib.util.find_spec("anemll")
        if spec is not None and spec.origin:
            pkg_dir = Path(spec.origin).parent
            for rel in [script_name, f"scripts/{script_name}"]:
                candidate = pkg_dir / rel
                if candidate.is_file():
                    return candidate
            # Walk up one level (package root)
            pkg_root = pkg_dir.parent
            for rel in [script_name, f"scripts/{script_name}"]:
                candidate = pkg_root / rel
                if candidate.is_file():
                    return candidate
    except Exception:
        pass

    # 3. Common clone paths
    home = Path.home()
    search_roots = [
        home / "anemll",
        home / "Anemll",
        home / "Projects" / "anemll",
        home / "Projects" / "Anemll",
        home / "projects" / "anemll",
        home / "projects" / "Anemll",
    ]
    for root in search_roots:
        for rel in [script_name, f"scripts/{script_name}"]:
            candidate = root / rel
            if candidate.is_file():
                return candidate

    raise FileNotFoundError(
        f"Could not find Anemll's '{script_name}'.\n"
        "Please do one of the following:\n"
        "  1. Set the ANEMLL_HOME environment variable to your Anemll clone root:\n"
        "       export ANEMLL_HOME=~/Projects/anemll\n"
        "  2. Clone Anemll into ~/anemll or ~/Projects/anemll:\n"
        "       git clone https://github.com/Anemll/Anemll.git ~/Projects/anemll\n"
        "  3. Install the anemll Python package:\n"
        "       pip install anemll"
    )


# ---------------------------------------------------------------------------
# Pre-conversion check
# ---------------------------------------------------------------------------


def check_preconverted(model_name: str, output_dir: Path) -> bool:
    """Return True if the model has already been converted (meta.yaml exists).

    Args:
        model_name: Short name used as the subdirectory name under output_dir.
        output_dir: Root directory for converted models.
    """
    meta_path = output_dir / model_name / "meta.yaml"
    return meta_path.exists()


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------


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
    """Convert a HuggingFace model to CoreML/Anemll format.

    Runs Anemll's convert_model.sh in a subprocess.  On success, writes a
    conversion_meta.json file next to the converted artifacts.

    Args:
        hf_id:      HuggingFace model ID (e.g. ``Qwen/Qwen3.5-0.8B``).
        output_dir: Directory where the converted model will be stored.
        context:    Token context window length.
        batch:      Batch size for prefill.
        lut1:       LUT bits for embedding weights.
        lut2:       LUT bits for FFN weights.
        lut3:       LUT bits for LM-head weights.
        monolithic: Produce a single .mlmodelc (True) vs chunked layout.
        argmax:     Embed argmax into the model graph (True) vs raw logits.

    Returns:
        Dict with keys: ``hf_id``, ``output_dir``, ``status``, ``returncode``,
        ``elapsed_s``, ``converter``.
    """
    converter = find_anemll_converter()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        "bash",
        str(converter),
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

    env = os.environ.copy()
    # Ensure project root on PYTHONPATH so subprocesses can find the package
    project_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

    print(f"\n{'='*72}")
    print(f"  Model     : {hf_id}")
    print(f"  Output    : {output_dir}")
    print(f"  Context   : {context}  Batch: {batch}")
    print(f"  LUT       : embed={lut1}  ffn={lut2}  lmhead={lut3}")
    print(f"  Monolithic: {monolithic}  Argmax: {argmax}")
    print(f"  Converter : {converter}")
    print(f"{'='*72}\n")

    t0 = time.time()
    result = subprocess.run(cmd, env=env, capture_output=False)
    elapsed = time.time() - t0

    status = "ok" if result.returncode == 0 else "error"
    print(f"\n  --> {hf_id}: {status} ({elapsed:.1f}s)")

    # Save conversion metadata alongside the artifacts
    meta = {
        "hf_id": hf_id,
        "output_dir": str(output_dir),
        "converter": str(converter),
        "params": {
            "context": context,
            "batch": batch,
            "lut1": lut1,
            "lut2": lut2,
            "lut3": lut3,
            "monolithic": monolithic,
            "argmax": argmax,
        },
        "status": status,
        "returncode": result.returncode,
        "elapsed_s": round(elapsed, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.mac_ver()[0] or platform.platform(),
        "python_version": platform.python_version(),
    }

    try:
        meta_path = output_dir / "conversion_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except OSError:
        pass  # non-fatal; we already printed the result

    return meta


# ---------------------------------------------------------------------------
# Name resolution helpers
# ---------------------------------------------------------------------------


def short_name_for(hf_id: str) -> str:
    """Return registry short name for hf_id, or a slug derived from hf_id."""
    for short, hfid in MODEL_REGISTRY.items():
        if hfid == hf_id:
            return short
    # Derive a filesystem-safe slug: take the repo part after '/'
    return hf_id.split("/")[-1].lower().replace(".", "-")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert HuggingFace models to Anemll/CoreML format for ANE inference."
    )

    # Model selection (mutually exclusive)
    model_group = ap.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        metavar="HF_ID",
        help="Single model by full HuggingFace ID (e.g. Qwen/Qwen3.5-0.8B).",
    )
    model_group.add_argument(
        "--models",
        nargs="+",
        metavar="NAME",
        help="One or more registry short names (e.g. qwen3.5-0.8b qwen3.5-2b).",
    )
    model_group.add_argument(
        "--all",
        action="store_true",
        help="Convert all models in the registry.",
    )

    ap.add_argument(
        "--output",
        default="models/ane",
        metavar="DIR",
        help="Root output directory (default: models/ane).",
    )

    # Conversion parameters
    ap.add_argument(
        "--context",
        type=int,
        default=DEFAULT_CONTEXT,
        help=f"Token context window length (default: {DEFAULT_CONTEXT}).",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help=f"Prefill batch size (default: {DEFAULT_BATCH}).",
    )
    ap.add_argument(
        "--lut1",
        type=int,
        default=DEFAULT_LUT1,
        help=f"LUT bits for embedding weights (default: {DEFAULT_LUT1}).",
    )
    ap.add_argument(
        "--lut2",
        type=int,
        default=DEFAULT_LUT2,
        help=f"LUT bits for FFN weights (default: {DEFAULT_LUT2}).",
    )
    ap.add_argument(
        "--lut3",
        type=int,
        default=DEFAULT_LUT3,
        help=f"LUT bits for LM-head weights (default: {DEFAULT_LUT3}).",
    )
    ap.add_argument(
        "--no-monolithic",
        dest="monolithic",
        action="store_false",
        default=True,
        help="Produce chunked layout instead of a single .mlmodelc.",
    )
    ap.add_argument(
        "--no-argmax",
        dest="argmax",
        action="store_false",
        default=True,
        help="Emit raw logits instead of embedding argmax in the model graph.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-convert even if meta.yaml already exists (skip the pre-check).",
    )

    args = ap.parse_args()

    # Build the (short_name, hf_id) work list
    work: List[tuple[str, str]] = []
    if args.all:
        work = list(MODEL_REGISTRY.items())
    elif args.model:
        hf_id = args.model
        short = short_name_for(hf_id)
        work = [(short, hf_id)]
    else:
        for name in args.models:
            if name in MODEL_REGISTRY:
                work.append((name, MODEL_REGISTRY[name]))
            elif "/" in name:
                # Accept raw HF IDs
                work.append((short_name_for(name), name))
            else:
                ap.error(
                    f"Unknown model '{name}'. "
                    f"Use a registry key {list(MODEL_REGISTRY)} "
                    f"or a full HuggingFace ID (org/repo)."
                )

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    # Verify converter exists before starting any work
    try:
        converter = find_anemll_converter()
        print(f"Anemll converter : {converter}")
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Output root      : {output_root.resolve()}")
    print(f"Models to convert: {len(work)}")
    print(f"Parameters       : context={args.context} batch={args.batch} "
          f"lut1={args.lut1} lut2={args.lut2} lut3={args.lut3}")

    # Sequential conversion loop
    run_log: List[Dict[str, Any]] = []
    overall_t0 = time.time()

    for i, (short_name, hf_id) in enumerate(work, 1):
        model_out = output_root / short_name
        print(f"\n[{i}/{len(work)}] {hf_id}  -->  {model_out}")

        # Skip if already converted (unless --force)
        if not args.force and check_preconverted(short_name, output_root):
            print(f"  Skipping: meta.yaml already exists at {model_out / 'meta.yaml'}")
            run_log.append({
                "short_name": short_name,
                "hf_id": hf_id,
                "output_dir": str(model_out),
                "status": "skipped",
                "returncode": 0,
                "elapsed_s": 0.0,
            })
            continue

        result = convert_model(
            hf_id=hf_id,
            output_dir=model_out,
            context=args.context,
            batch=args.batch,
            lut1=args.lut1,
            lut2=args.lut2,
            lut3=args.lut3,
            monolithic=args.monolithic,
            argmax=args.argmax,
        )
        run_log.append({
            "short_name": short_name,
            **result,
        })

    overall_elapsed = time.time() - overall_t0

    # Save run manifest
    manifest = {
        "models": [entry["hf_id"] for entry in run_log if "hf_id" in entry],
        "params": {
            "context": args.context,
            "batch": args.batch,
            "lut1": args.lut1,
            "lut2": args.lut2,
            "lut3": args.lut3,
            "monolithic": args.monolithic,
            "argmax": args.argmax,
        },
        "results": run_log,
        "total_elapsed_s": round(overall_elapsed, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = output_root / "conversion_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Summary table
    print(f"\n{'='*72}")
    print(f"  ANE Conversion Complete -- {len(work)} model(s) in {overall_elapsed:.0f}s")
    print(f"{'='*72}")
    print(f"  {'Short Name':<20} {'HF Model ID':<35} {'Status':<9} {'Time':>8}")
    print(f"  {'-'*20} {'-'*35} {'-'*9} {'-'*8}")
    for entry in run_log:
        short = entry.get("short_name", "?")
        hfid  = entry.get("hf_id", "?")
        stat  = entry.get("status", "?")
        secs  = entry.get("elapsed_s", 0.0)
        time_str = f"{secs:>7.1f}s" if stat != "skipped" else "  skipped"
        print(f"  {short:<20} {hfid:<35} {stat:<9} {time_str}")

    print(f"\n  Manifest: {manifest_path}")
    print(f"\n  To use with ane_local provider, set:")
    for entry in run_log:
        if entry.get("status") in ("ok", "skipped"):
            short = entry.get("short_name", "?")
            print(f"    export ANE_META_DIR={output_root.resolve() / short}")
    print()

    # Exit with error if any model failed
    failures = [e for e in run_log if e.get("status") not in ("ok", "skipped")]
    if failures:
        print(f"  WARNING: {len(failures)} model(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
