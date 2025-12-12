#!/usr/bin/env python3
"""Evaluate CLINC predictions vs gold JSONL; compute macro F1 over intent and accuracy for is_oos."""
import argparse, json
from pathlib import Path
from collections import defaultdict


def macro_f1(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    f1s = []
    for l in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == l and yp == l)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != l and yp == l)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == l and yp != l)
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def load_jsonl(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predictions JSONL with output_json per id")
    ap.add_argument("--gold", required=True, help="gold JSONL (tasks/clinc_en.jsonl)")
    args = ap.parse_args()

    pred_rows = {r["id"]: r for r in load_jsonl(args.pred)}
    gold_rows = load_jsonl(args.gold)
    y_true_intent, y_pred_intent = [], []
    y_true_oos, y_pred_oos = [], []
    missing = 0
    for g in gold_rows:
        gid = g["id"]
        gold = g["gold"] if "gold" in g else g
        if gid not in pred_rows:
            missing += 1
            continue
        pred = pred_rows[gid].get("output_json", {}) or pred_rows[gid].get("gold", {})
        y_true_intent.append(gold.get("intent", ""))
        y_pred_intent.append(pred.get("intent", ""))
        y_true_oos.append(bool(gold.get("is_oos", False)))
        y_pred_oos.append(bool(pred.get("is_oos", False)))
    f1_intent = macro_f1(y_true_intent, y_pred_intent)
    acc_oos = sum(1 for t, p in zip(y_true_oos, y_pred_oos) if t == p) / float(len(y_true_oos) or 1)
    print(f"macro_f1_intent: {f1_intent:.4f} over {len(y_true_intent)} examples")
    print(f"acc_is_oos: {acc_oos:.4f} over {len(y_true_oos)} examples")
    if missing:
        print(f"missing predictions for {missing} examples")


if __name__ == "__main__":
    main()
