#!/usr/bin/env python3
"""
Build arXiv-ready source archives for each paper under papers/*/arxiv containing a main.tex.

Examples:
  python tools/build_arxiv_sources.py               # package all detected arxiv dirs
  python tools/build_arxiv_sources.py --papers P1_stable_slo P2_reward_stability
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def find_arxiv_dirs(root: Path) -> list[Path]:
    """Return arxiv directories that contain a main.tex."""
    out = []
    for paper_dir in root.glob("*"):
        arxiv_dir = paper_dir / "arxiv"
        if (arxiv_dir / "main.tex").exists():
            out.append(arxiv_dir)
    return out


def build_archive(arxiv_dir: Path, out_dir: Path) -> Path:
    """Create a zip archive of the arxiv directory contents."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = out_dir / f"{arxiv_dir.parent.name}_arxiv"
    archive_path = shutil.make_archive(str(base_name), "zip", root_dir=arxiv_dir)
    return Path(archive_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--papers",
        nargs="*",
        help="Optional list of paper directory names under papers/ (e.g., P1_stable_slo). Default: auto-discover.",
    )
    ap.add_argument(
        "--out-dir",
        default="out/arxiv",
        help="Destination directory for zip archives (default: out/arxiv).",
    )
    args = ap.parse_args()

    root = Path("papers")
    arxiv_dirs = find_arxiv_dirs(root)
    if args.papers:
        wanted = set(args.papers)
        arxiv_dirs = [p for p in arxiv_dirs if p.parent.name in wanted]

    if not arxiv_dirs:
        raise SystemExit("No arxiv dirs with main.tex found.")

    out_dir = Path(args.out_dir)
    built = []
    for arxiv_dir in arxiv_dirs:
        archive = build_archive(arxiv_dir, out_dir)
        built.append(archive)
        print(f"[done] packaged {arxiv_dir} -> {archive}")


if __name__ == "__main__":
    main()
