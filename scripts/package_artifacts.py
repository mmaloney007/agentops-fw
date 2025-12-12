#!/usr/bin/env python3
"""
Package a run directory into a checksum manifest (and optional tar.gz archive).
Optionally logs to W&B if WANDB_PROJECT is set.
"""
import argparse, hashlib, os, tarfile
from pathlib import Path
from typing import Dict, Any, List

from agent_stable_slo.logging import wandb_utils as WL
from agent_stable_slo.utils.repro import atomic_write_json


def _sha256(path: Path, chunk: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_manifest(run_dir: Path, exclude_ext: List[str]) -> Dict[str, Any]:
    files = []
    for p in run_dir.rglob("*"):
        if p.is_dir():
            continue
        if any(str(p).endswith(ext) for ext in exclude_ext):
            continue
        files.append(
            {
                "path": str(p.relative_to(run_dir)),
                "size": p.stat().st_size,
                "sha256": _sha256(p),
            }
        )
    return {"run_dir": str(run_dir), "files": files}


def maybe_archive(run_dir: Path, archive_path: Path, exclude_ext: List[str]) -> None:
    with tarfile.open(archive_path, "w:gz") as tar:
        for p in run_dir.rglob("*"):
            if p.is_dir():
                continue
            if any(str(p).endswith(ext) for ext in exclude_ext):
                continue
            tar.add(p, arcname=p.relative_to(run_dir))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None, help="Manifest path (default: <run>/artifacts_manifest.json)")
    ap.add_argument("--archive", action="store_true", help="Create a tar.gz archive alongside manifest.")
    ap.add_argument("--exclude-ext", nargs="*", default=[".bin"], help="File extensions to skip (default: .bin)")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")

    manifest = build_manifest(run_dir, args.exclude_ext)
    manifest_path = Path(args.out) if args.out else run_dir / "artifacts_manifest.json"
    atomic_write_json(str(manifest_path), manifest)
    print(f"[done] wrote manifest to {manifest_path}")

    archive_path = None
    if args.archive:
        archive_path = run_dir / "artifacts.tar.gz"
        maybe_archive(run_dir, archive_path, args.exclude_ext)
        print(f"[done] wrote archive to {archive_path}")

    if WL._active():
        cfg = {"run_dir": str(run_dir), "archive": str(archive_path) if archive_path else None}
        with WL.maybe_run(name=f"artifacts-{run_dir.name}", config=cfg) as run:
            WL.log_artifact(run, str(manifest_path), name=f"{run_dir.name}-manifest", type_="artifact-manifest")
            if archive_path:
                WL.log_artifact(run, str(archive_path), name=f"{run_dir.name}-archive", type_="artifact-archive")


if __name__ == "__main__":
    main()
