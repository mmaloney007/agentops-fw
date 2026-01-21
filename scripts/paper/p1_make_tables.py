#!/usr/bin/env python3
"""
Generate Paper 1 tables from W&B episodes artifacts or local episodes.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _load_episodes(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _macro_f1(y_true: List[str], y_pred: List[str]) -> Optional[float]:
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return None
    f1s: List[float] = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        denom = precision + recall
        f1 = (2 * precision * recall / denom) if denom else 0.0
        f1s.append(f1)
    if not f1s:
        return None
    return float(sum(f1s) / len(f1s))


def _download_wandb_artifact(artifact: str, target_dir: Path) -> Path:
    import wandb

    api = wandb.Api()
    art = api.artifact(artifact)
    target_dir.mkdir(parents=True, exist_ok=True)
    art.download(root=str(target_dir))
    return target_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--episodes", default="", help="Path to local episodes.jsonl (optional)."
    )
    ap.add_argument(
        "--episodes-dir",
        default="",
        help="Directory containing one or more episodes.jsonl files.",
    )
    ap.add_argument(
        "--artifact", default="", help="W&B artifact name (entity/project/name:alias)."
    )
    ap.add_argument("--out-dir", default="papers/p1/tables")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes: List[Dict[str, Any]] = []
    if args.episodes_dir:
        for path in Path(args.episodes_dir).rglob("episodes.jsonl"):
            episodes.extend(_load_episodes(path))
    elif args.episodes:
        episodes = _load_episodes(Path(args.episodes))
    elif args.artifact:
        download_dir = _download_wandb_artifact(
            args.artifact, Path("out/wandb_artifacts")
        )
        candidates = list(download_dir.rglob("episodes.jsonl"))
        if not candidates:
            raise SystemExit("episodes.jsonl not found in artifact")
        episodes = _load_episodes(candidates[0])
    else:
        raise SystemExit("Provide --episodes, --episodes-dir, or --artifact.")
    if not episodes:
        raise SystemExit("No episodes loaded.")

    rows = []
    for ep in episodes:
        row = {
            "model": ep.get("model"),
            "decode_mode": ep.get("decode_mode"),
            "task_id": ep.get("task_id"),
            "latency_ms": ep.get("latency_ms"),
        }
        for k, v in (ep.get("metrics") or {}).items():
            row[k] = v
        gold = ep.get("gold", {})
        if ep.get("task_id") == "t1_clinc":
            row["gold_intent"] = str(gold.get("intent")) if gold else None
            if ep.get("parsed_output"):
                row["pred_intent"] = str(ep["parsed_output"].get("intent"))
        rows.append(row)

    df = pd.DataFrame(rows)
    grouped = df.groupby(["model", "decode_mode"], dropna=False)

    # Table 1: Structure
    table1 = grouped[["json_valid", "schema_valid"]].mean().reset_index()
    table1.to_csv(out_dir / "table1_structure.csv", index=False)

    # Table 2: Accuracy + Faithfulness
    table2_cols = [
        "clinc_intent_accuracy",
        "hotpot_answer_exact_match",
        "hotpot_answer_f1",
        "hotpot_faithfulness",
        "hotpot_contradiction_rate",
        "tool_success_rate",
    ]
    table2 = grouped[table2_cols].mean().reset_index()

    # Add CLINC macro F1 if gold/pred present
    macro_rows = []
    for (model, mode), gdf in grouped:
        gdf = gdf[gdf["task_id"] == "t1_clinc"]
        if "gold_intent" in gdf and "pred_intent" in gdf:
            macro = _macro_f1(
                [str(x) for x in gdf["gold_intent"].dropna().tolist()],
                [str(x) for x in gdf["pred_intent"].dropna().tolist()],
            )
        else:
            macro = None
        macro_rows.append(
            {"model": model, "decode_mode": mode, "clinc_intent_macro_f1": macro}
        )
    macro_df = pd.DataFrame(macro_rows)
    table2 = table2.merge(macro_df, on=["model", "decode_mode"], how="left")
    table2.to_csv(out_dir / "table2_accuracy_faithfulness.csv", index=False)

    # Table 3: Stability (compute directly from episodes list)
    stability_rows = []
    by_key: Dict[tuple, List[Dict[str, Any]]] = {}
    for ep in episodes:
        key = (ep.get("model"), ep.get("decode_mode"))
        by_key.setdefault(key, []).append(ep)

    for (model, mode), eps in by_key.items():
        by_prompt: Dict[str, List[str]] = {}
        for ep in eps:
            pid = ep.get("task_instance_id")
            canon = (ep.get("stability_features") or {}).get("canonical_json", "")
            by_prompt.setdefault(pid, []).append(canon)
        disagreements = []
        agreements = []
        for _pid, ck in by_prompt.items():
            if not ck:
                continue
            mode_key = max(set(ck), key=ck.count)
            disagreements.append(1.0 - ck.count(mode_key) / len(ck))
            agreements.append(ck.count(mode_key) / len(ck))
        stability_rows.append(
            {
                "model": model,
                "decode_mode": mode,
                "disagreement_at_k": sum(disagreements) / len(disagreements)
                if disagreements
                else None,
                "total_agreement_rate_at_k": sum(agreements) / len(agreements)
                if agreements
                else None,
            }
        )
    pd.DataFrame(stability_rows).to_csv(out_dir / "table3_stability.csv", index=False)

    print(f"[P1] wrote tables to {out_dir}")


if __name__ == "__main__":
    main()
