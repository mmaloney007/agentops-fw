#!/usr/bin/env python3
"""P4 Analysis: Early Prediction.

Tests whether early training features (step-50) predict final outcome
(sustained vs not). Uses simple logistic regression.

Input:  results/p4_analysis/curve_taxonomy.json
Output: results/p4_analysis/early_prediction.json
"""
import json
import math
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent.parent
TAXONOMY_PATH = ROOT / "results" / "p4_analysis" / "curve_taxonomy.json"
OUT_JSON = ROOT / "results" / "p4_analysis" / "early_prediction.json"


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


def simple_logistic_regression(X: list[list[float]], y: list[int], lr: float = 0.1, epochs: int = 1000):
    """Minimal logistic regression without sklearn dependency."""
    if not X or not y:
        return [], 0.0

    n_features = len(X[0])
    weights = [0.0] * n_features
    bias = 0.0

    for _ in range(epochs):
        for xi, yi in zip(X, y):
            z = sum(w * x for w, x in zip(weights, xi)) + bias
            pred = sigmoid(z)
            error = pred - yi
            for j in range(n_features):
                weights[j] -= lr * error * xi[j]
            bias -= lr * error

    return weights, bias


def predict(weights, bias, x):
    z = sum(w * xi for w, xi in zip(weights, x)) + bias
    return 1 if sigmoid(z) >= 0.5 else 0


def analyze():
    with open(TAXONOMY_PATH, "r") as f:
        data = json.load(f)

    runs = data["runs"]

    # Binary classification: sustained (1) vs not (0)
    labeled = []
    for r in runs:
        label = 1 if r["category"] == "sustained" else 0
        features = {
            "early_mean_50": r.get("early_mean_50", 0),
            "peak_step": r.get("peak_step", 0) / 1000.0,  # normalize
            "size_b": r.get("size_b", 0) / 12.0,  # normalize
        }
        labeled.append({"features": features, "label": label, "model": r["model"], "task": r["task"]})

    if len(labeled) < 5:
        print("Not enough data for prediction analysis")
        output = {"error": "insufficient data", "n_runs": len(labeled)}
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_JSON, "w") as f:
            json.dump(output, f, indent=2)
        return output

    # Test at different early observation windows: N=10, 25, 50, 100
    results = {}
    feature_names = ["early_mean_50", "peak_step", "size_b"]

    X_all = [[d["features"][f] for f in feature_names] for d in labeled]
    y_all = [d["label"] for d in labeled]

    # Full training set accuracy (no split - just measuring separability)
    weights, bias = simple_logistic_regression(X_all, y_all)
    preds = [predict(weights, bias, x) for x in X_all]
    full_acc = sum(1 for p, y in zip(preds, y_all) if p == y) / len(y_all)

    results["full"] = {
        "n_samples": len(labeled),
        "accuracy": round(full_acc, 4),
        "n_sustained": sum(y_all),
        "n_other": len(y_all) - sum(y_all),
        "weights": {name: round(w, 4) for name, w in zip(feature_names, weights)},
        "bias": round(bias, 4),
    }

    # Leave-one-out cross-validation for more robust estimate
    loo_correct = 0
    for i in range(len(labeled)):
        X_train = X_all[:i] + X_all[i+1:]
        y_train = y_all[:i] + y_all[i+1:]
        w, b = simple_logistic_regression(X_train, y_train)
        pred = predict(w, b, X_all[i])
        if pred == y_all[i]:
            loo_correct += 1

    results["loo_cv"] = {
        "accuracy": round(loo_correct / len(labeled), 4),
        "n_correct": loo_correct,
        "n_total": len(labeled),
    }

    # Per-model prediction summary
    model_preds = defaultdict(list)
    for d, p in zip(labeled, preds):
        model_preds[d["model"]].append({
            "task": d["task"],
            "predicted": p,
            "actual": d["label"],
            "correct": p == d["label"],
        })

    results["per_model"] = {
        model: {
            "accuracy": round(sum(1 for r in runs if r["correct"]) / len(runs), 4),
            "predictions": runs,
        }
        for model, runs in model_preds.items()
    }

    output = {"prediction_results": results, "feature_names": feature_names}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {OUT_JSON}")
    print(f"  Full accuracy: {full_acc:.1%}")
    print(f"  LOO-CV accuracy: {loo_correct/len(labeled):.1%}")

    return output


if __name__ == "__main__":
    analyze()
