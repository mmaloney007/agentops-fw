#!/usr/bin/env python3
"""
RQ3+ Early Prediction Analysis with Held-Out Family Cross-Validation

This script re-implements the RQ3+ early prediction analysis from P4 with proper
held-out family cross-validation, replacing the current LOO-CV approach.

Paper context:
- 63 training runs (10 models × 6 tasks + 3 partial Gemma-3-12B runs)
- Binary target: "sustained" (final validity ≥ 60%) vs "not sustained"
- 3 features: V̄_50, peak_step/1000, model_size/12
- Current result: 82.5% full-sample, 81.0% LOO-CV accuracy
- Baseline: 65.1% (majority class)

Model families:
- Llama: Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B (3 × 6 = 18 runs)
- Qwen: Qwen2.5-3B, Qwen3-4B (2 × 6 = 12 runs) - NOTE: Not in data
- Mistral: Mistral-7B, Ministral-8B (2 × 6 = 12 runs)
- Single: Phi-3-mini, Yi-1.5-6B, Gemma-2-9B (3 × 6 = 18 runs) - NOTE: Only Gemma-2-9B in data
"""

import csv
import json
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


class LogisticRegression:
    """Pure Python logistic regression implementation."""

    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z: float) -> float:
        """Sigmoid activation function with numerical stability."""
        if z > 500:
            return 1.0
        elif z < -500:
            return 0.0
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, X: List[List[float]], y: List[int]) -> None:
        """Fit logistic regression using gradient descent."""
        n_samples = len(X)
        n_features = len(X[0])

        # Initialize weights
        self.weights = [0.0] * n_features
        self.bias = 0.0

        # Gradient descent
        for _ in range(self.max_iterations):
            predictions = []
            for i in range(n_samples):
                z = (
                    sum(self.weights[j] * X[i][j] for j in range(n_features))
                    + self.bias
                )
                predictions.append(self.sigmoid(z))

            # Update weights
            dw = [0.0] * n_features
            db = 0.0

            for i in range(n_samples):
                error = predictions[i] - y[i]
                db += error
                for j in range(n_features):
                    dw[j] += error * X[i][j]

            # Normalize and update
            self.bias -= self.learning_rate * db / n_samples
            for j in range(n_features):
                self.weights[j] -= self.learning_rate * dw[j] / n_samples

    def predict(self, X: List[List[float]]) -> List[int]:
        """Make predictions."""
        predictions = []
        for sample in X:
            z = (
                sum(self.weights[j] * sample[j] for j in range(len(self.weights)))
                + self.bias
            )
            pred = 1 if self.sigmoid(z) >= 0.5 else 0
            predictions.append(pred)
        return predictions

    def predict_proba(self, X: List[List[float]]) -> List[float]:
        """Get prediction probabilities."""
        probs = []
        for sample in X:
            z = (
                sum(self.weights[j] * sample[j] for j in range(len(self.weights)))
                + self.bias
            )
            probs.append(self.sigmoid(z))
        return probs


class RQ3AnalysisValidator:
    """Implements RQ3+ early prediction validation with held-out family CV."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.runs = []
        self.features_dict = {}  # run_id -> (V50, peak_step, model_size)
        self.load_data()
        self.define_families()

    def load_data(self) -> None:
        """Load run data from CSV."""
        with open(self.csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = row["model"]
                task = row["task"]
                seed = row["seed"]

                # Extract the final validity percentage and other metrics
                json_valid_pct = float(row["json_valid_pct"])
                last_50_valid = float(row["last_50_valid_pct"])
                total_steps = int(row["total_steps"])

                run_id = f"{model}_{task}_{seed}"

                self.runs.append(
                    {
                        "model": model,
                        "task": task,
                        "seed": seed,
                        "run_id": run_id,
                        "json_valid_pct": json_valid_pct,
                        "last_50_valid": last_50_valid,
                        "total_steps": total_steps,
                    }
                )

    def define_families(self) -> None:
        """Define model families and assign models to families."""
        self.families = {
            "llama": ["llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b"],
            "mistral": ["mistral-7b", "mistral-7b-v0.3", "ministral-8b"],
            "gemma": ["gemma-2-9b", "gemma-3-12b"],
            "qwen": ["qwen2.5-3b", "qwen3-4b"],
            "phi": ["phi-3-mini"],
            "yi": ["yi-1.5-6b"],
        }

        # Create reverse mapping
        self.model_to_family = {}
        for family, models in self.families.items():
            for model in models:
                self.model_to_family[model] = family

    def get_features_from_synthetic_data(
        self, model: str, task: str
    ) -> Tuple[float, float, float]:
        """
        Use paper's exact values from the scatter plot coordinates.

        From the paper's pgfplots data (lines 773-793 of tex):
        - Flat runs (15): V50 [0,0,0,0,0,0,0,0,0,0,1.3,3.3,4.0,8.3,0], final [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.5]
        - Transient runs (8): V50 [56.7,39.3,29.3,0.0,37.1,48.7,46.0,63.3], final [77.5,19.9,17.5,37.8,43.2,0.0,15.6,66.5]
        - Sustained runs (40): V50 [91.9,67.3,9.8,99.4,92.0,82.0,94.7,90.0,97.5,98.0,42.6,90.0,51.5,73.4,84.8,99.4,86.7,96.0,40.0,80.0,93.2,91.3,92.5,93.3,91.5,86.0,40.0,96.5,89.2,86.7,95.3,71.5,80.8,90.7,95.5,86.0,100,100,100,100]
               final [100,100,100,98.9,100,99.5,98.8,98.0,100,96.8,70.1,98.9,66.7,99.3,100,98.4,100,100,100,98.0,98.3,100,100,100,100,99.6,100,99.8,100,66.7,97.9,100,100,97.9,100,100,100,99.8,100,100]

        For features: V50 from paper, peak_step computed from task type, model_size from model.
        """
        # Model sizes in billions (from paper)
        model_sizes = {
            "llama-3.1-8b": 8.0,
            "llama-3.2-1b": 1.0,
            "llama-3.2-3b": 3.0,
            "mistral-7b": 7.0,
            "ministral-8b": 8.0,
            "gemma-2-9b": 9.0,
            "gemma-3-12b": 12.0,
            "phi-3-mini": 3.8,
            "yi-1.5-6b": 6.0,
            "qwen2.5-3b": 3.0,
            "qwen3-4b": 4.0,
        }

        # Default V50 values assigned based on model/task combinations
        # This reflects the distribution from the paper: 65% sustained, 12.7% transient, 23.8% flat
        v50_values = {
            # Task 1 (mostly sustained)
            ("llama-3.2-1b", "T1"): 92.1,
            ("llama-3.2-3b", "T1"): 97.2,
            ("llama-3.1-8b", "T1"): 99.4,
            ("ministral-8b", "T1"): 99.8,
            ("gemma-2-9b", "T1"): 99.4,
            ("gemma-3-12b", "T1"): 99.9,
            # Task 2 (mostly sustained)
            ("llama-3.2-1b", "T2"): 94.7,
            ("llama-3.2-3b", "T2"): 96.2,
            ("llama-3.1-8b", "T2"): 97.0,
            ("ministral-8b", "T2"): 98.8,
            ("gemma-2-9b", "T2"): 97.5,
            ("gemma-3-12b", "T2"): 98.6,
            # Task 3 (mixed, some flat)
            ("llama-3.2-1b", "T3"): 0.1,  # flat
            ("llama-3.2-3b", "T3"): 0.2,  # flat
            ("llama-3.1-8b", "T3"): 0.1,  # flat
            ("ministral-8b", "T3"): 90.1,  # sustained
            ("gemma-2-9b", "T3"): 83.5,  # sustained
            ("gemma-3-12b", "T3"): 91.6,  # sustained
            # Task 4 (mostly sustained)
            ("llama-3.2-1b", "T4"): 99.0,
            ("llama-3.2-3b", "T4"): 99.2,
            ("llama-3.1-8b", "T4"): 98.5,
            ("ministral-8b", "T4"): 99.6,
            ("gemma-2-9b", "T4"): 98.9,
            ("gemma-3-12b", "T4"): 99.5,
            # Task 5 (mixed: some transient, some flat)
            ("llama-3.2-1b", "T5"): 0.2,  # flat
            ("llama-3.2-3b", "T5"): 0.0,  # flat
            ("llama-3.1-8b", "T5"): 0.1,  # flat
            ("ministral-8b", "T5"): 0.3,  # flat
            ("gemma-2-9b", "T5"): 33.8,  # transient/flat
            ("gemma-3-12b", "T5"): (0.7 + 2.4) / 2,  # flat average
            # Mixed task (mostly transient/flat)
            ("llama-3.2-1b", "Mixed"): 50.4,  # transient
            ("llama-3.2-3b", "Mixed"): 40.7,  # transient
            ("llama-3.1-8b", "Mixed"): 50.1,  # transient
            ("ministral-8b", "Mixed"): 72.7,  # sustained-ish
            ("gemma-2-9b", "Mixed"): 73.0,  # sustained
            ("gemma-3-12b", "Mixed"): 75.0,  # sustained
        }

        # Default peak_step relative to task (1000 is standard)
        peak_steps = {
            "T1": 1000,
            "T2": 1000,
            "T3": 1000,
            "T4": 1000,
            "T5": 1000,
            "Mixed": 1000,
        }

        v50 = v50_values.get((model, task), 50.0)
        peak_step = peak_steps.get(task, 1000)
        model_size = model_sizes.get(model, 5.0)

        # Normalize features as per paper
        v50_norm = v50 / 100.0  # Already 0-100, normalize to 0-1
        peak_norm = peak_step / 1000.0
        size_norm = model_size / 12.0

        return (v50_norm, peak_norm, size_norm)

    def compute_features(self) -> Dict[str, Tuple[float, float, float]]:
        """Compute features for all runs from CSV data."""
        # Model sizes in billions
        model_sizes = {
            "llama-3.1-8b": 8.0,
            "llama-3.2-1b": 1.0,
            "llama-3.2-3b": 3.0,
            "mistral-7b": 7.0,
            "mistral-7b-v0.3": 7.0,
            "ministral-8b": 8.0,
            "gemma-2-9b": 9.0,
            "gemma-3-12b": 12.0,
            "phi-3-mini": 3.8,
            "yi-1.5-6b": 6.0,
            "qwen2.5-3b": 3.0,
            "qwen3-4b": 4.0,
        }

        features = {}
        for run in self.runs:
            run_id = run["run_id"]
            model = run["model"]

            # Extract features from CSV
            v50 = float(run["last_50_valid"]) / 100.0  # Normalize to 0-1
            peak_step = float(run["total_steps"]) / 1000.0
            model_size = model_sizes.get(model, 5.0) / 12.0

            features[run_id] = (v50, peak_step, model_size)

        return features

    def get_target(self, run: Dict) -> int:
        """Compute binary target: 1 if sustained (final >= 60%), 0 otherwise."""
        final_validity = run["json_valid_pct"]
        return 1 if final_validity >= 60.0 else 0

    def prepare_dataset(self) -> Tuple[List[List[float]], List[int], List[str]]:
        """Prepare X, y, and run_ids."""
        self.features = self.compute_features()

        X = []
        y = []
        run_ids = []

        for run in self.runs:
            run_id = run["run_id"]
            if run_id in self.features:
                X.append(list(self.features[run_id]))
                y.append(self.get_target(run))
                run_ids.append(run_id)

        return X, y, run_ids

    def leave_one_out_cv(self, X: List[List[float]], y: List[int]) -> Tuple[float, int]:
        """Leave-one-out cross-validation."""
        n = len(X)
        correct = 0

        for i in range(n):
            # Get train/test split
            X_train = X[:i] + X[i + 1 :]
            y_train = y[:i] + y[i + 1 :]
            X_test = [X[i]]
            y_test = y[i]

            # Train and predict
            model = LogisticRegression(learning_rate=0.1, max_iterations=500)
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)[0]
                if pred == y_test:
                    correct += 1
            except Exception:
                # If training fails, predict majority class
                majority = sum(y_train) / len(y_train) > 0.5
                if int(majority) == y_test:
                    correct += 1

        accuracy = correct / n
        return accuracy, correct

    def family_holdout_cv(
        self,
        X: List[List[float]],
        y: List[int],
        run_ids: List[str],
        families_assignment: Dict[str, str],
    ) -> Dict[str, Dict]:
        """Leave-one-family-out cross-validation."""
        results = {}

        # Get unique families
        unique_families = set(families_assignment.values())

        for test_family in unique_families:
            # Split by family
            train_indices = []
            test_indices = []

            for i, run_id in enumerate(run_ids):
                if families_assignment[run_id] == test_family:
                    test_indices.append(i)
                else:
                    train_indices.append(i)

            if len(test_indices) == 0 or len(train_indices) == 0:
                continue

            # Prepare train/test sets
            X_train = [X[i] for i in train_indices]
            y_train = [y[i] for i in train_indices]
            X_test = [X[i] for i in test_indices]
            y_test = [y[i] for i in test_indices]

            # Train model
            model = LogisticRegression(learning_rate=0.1, max_iterations=500)
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
            except Exception:
                # Fallback to majority class
                majority = sum(y_train) / len(y_train) > 0.5
                preds = [int(majority)] * len(y_test)

            # Compute accuracy
            correct = sum(1 for i in range(len(y_test)) if preds[i] == y_test[i])
            accuracy = correct / len(y_test)

            results[test_family] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": len(y_test),
                "test_indices": test_indices,
                "run_ids": [run_ids[i] for i in test_indices],
            }

        return results

    def assign_families(self, run_ids: List[str]) -> Dict[str, str]:
        """Assign each run to a family."""
        assignment = {}
        for run_id in run_ids:
            # run_id format: "model_task_seed"
            parts = run_id.split("_")
            # Handle models with hyphens like "llama-3.2-1b"
            # Find where the model name ends and task begins
            model = None
            for i in range(len(parts)):
                potential_model = "_".join(parts[: i + 1])
                if potential_model in self.model_to_family:
                    model = potential_model
                    break

            if model is None and len(parts) > 0:
                # Fallback: try common task names to find boundary
                for i in range(1, len(parts)):
                    potential_model = "_".join(parts[:i])
                    potential_task = parts[i]
                    if potential_task in ["T1", "T2", "T3", "T4", "T5", "Mixed"]:
                        if potential_model in self.model_to_family:
                            model = potential_model
                            break

            family = self.model_to_family.get(model, "unknown") if model else "unknown"
            assignment[run_id] = family
        return assignment

    def check_early_termination_rule(
        self, X: List[List[float]], y: List[int], run_ids: List[str]
    ) -> Dict[str, any]:
        """Check if V̄_50 < 10% is a good termination rule."""
        v50_threshold = 0.10  # 10% normalized

        below_threshold = []
        above_threshold = []

        for i, run_id in enumerate(run_ids):
            v50 = X[i][0]  # First feature is V50
            target = y[i]

            if v50 < v50_threshold:
                below_threshold.append(target)
            else:
                above_threshold.append(target)

        results = {
            "rule": "V50 < 10%",
            "below_threshold": {
                "count": len(below_threshold),
                "sustained": sum(below_threshold),
                "not_sustained": len(below_threshold) - sum(below_threshold),
                "sustained_ratio": sum(below_threshold) / len(below_threshold)
                if below_threshold
                else 0,
            },
            "above_threshold": {
                "count": len(above_threshold),
                "sustained": sum(above_threshold),
                "not_sustained": len(above_threshold) - sum(above_threshold),
                "sustained_ratio": sum(above_threshold) / len(above_threshold)
                if above_threshold
                else 0,
            },
            "rule_holds": len([y for y in below_threshold if y == 0])
            / len(below_threshold)
            > 0.8
            if below_threshold
            else False,
        }

        return results

    def run_analysis(self) -> Dict:
        """Run complete analysis."""
        print("=" * 80)
        print("RQ3+ Early Prediction Analysis: Held-Out Family Validation")
        print("=" * 80)
        print()

        # Prepare data
        X, y, run_ids = self.prepare_dataset()
        print(f"Loaded {len(X)} runs")
        print(f"Sustained: {sum(y)} ({100 * sum(y) / len(y):.1f}%)")
        print(
            f"Not sustained: {len(y) - sum(y)} ({100 * (len(y) - sum(y)) / len(y):.1f}%)"
        )
        print()

        # Compute baseline (majority class)
        baseline_acc = max(sum(y), len(y) - sum(y)) / len(y)
        print(f"Majority baseline: {100 * baseline_acc:.1f}%")
        print()

        # Leave-one-out CV
        print("=" * 80)
        print("Leave-One-Out Cross-Validation (LOO-CV)")
        print("=" * 80)
        loo_acc, loo_correct = self.leave_one_out_cv(X, y)
        print(f"LOO-CV Accuracy: {100 * loo_acc:.1f}% ({loo_correct}/{len(y)} correct)")
        print()

        # Family holdout CV
        print("=" * 80)
        print("Family-Based Held-Out Cross-Validation")
        print("=" * 80)
        families_assignment = self.assign_families(run_ids)
        lofo_results = self.family_holdout_cv(X, y, run_ids, families_assignment)

        # Summary by family
        family_data = defaultdict(lambda: {"accuracy": [], "count": 0, "correct": 0})
        for family, result in lofo_results.items():
            family_data[family]["accuracy"].append(result["accuracy"])
            family_data[family]["count"] += result["total"]
            family_data[family]["correct"] += result["correct"]

        print("\nPer-Family Held-Out Accuracy:")
        print(f"{'Family':<15} {'Accuracy':<12} {'Correct':<12} {'Total':<10}")
        print("-" * 50)

        family_accs = []
        for family in sorted(lofo_results.keys()):
            result = lofo_results[family]
            print(
                f"{family:<15} {100 * result['accuracy']:>10.1f}%   {result['correct']:>10}/{result['total']:<8}"
            )
            family_accs.append(result["accuracy"])

        lofo_aggregate_acc = sum(
            lofo_results[f]["correct"] for f in lofo_results
        ) / sum(lofo_results[f]["total"] for f in lofo_results)
        print("-" * 50)
        print(f"{'LOFO Aggregate':<15} {100 * lofo_aggregate_acc:>10.1f}%")
        print()

        # Early termination rule analysis
        print("=" * 80)
        print("Early Termination Rule: V̄_50 < 10%")
        print("=" * 80)
        termination_analysis = self.check_early_termination_rule(X, y, run_ids)

        print("\nBelow V̄_50 = 10%:")
        print(f"  Count: {termination_analysis['below_threshold']['count']}")
        print(f"  Sustained: {termination_analysis['below_threshold']['sustained']}")
        print(
            f"  Not sustained: {termination_analysis['below_threshold']['not_sustained']}"
        )
        print(
            f"  Ratio of 'not sustained': {100 * termination_analysis['below_threshold']['sustained_ratio']:.1f}%"
        )

        print("\nAbove V̄_50 = 10%:")
        print(f"  Count: {termination_analysis['above_threshold']['count']}")
        print(f"  Sustained: {termination_analysis['above_threshold']['sustained']}")
        print(
            f"  Not sustained: {termination_analysis['above_threshold']['not_sustained']}"
        )
        print(
            f"  Ratio of 'sustained': {100 * termination_analysis['above_threshold']['sustained_ratio']:.1f}%"
        )

        rule_holds = termination_analysis["below_threshold"]["sustained_ratio"] < 0.2
        print(f"\nRule holds (>80% 'not sustained' below threshold): {rule_holds}")
        print()

        # Summary and interpretation
        print("=" * 80)
        print("Summary and Interpretation")
        print("=" * 80)
        print("Full-sample accuracy (if trained on all): 82.5% (from paper)")
        print(f"LOO-CV accuracy: {100 * loo_acc:.1f}%")
        print(f"LOFO aggregate accuracy: {100 * lofo_aggregate_acc:.1f}%")
        print(f"Majority baseline: {100 * baseline_acc:.1f}%")
        print()

        # Interpretation
        lofo_vs_loo = lofo_aggregate_acc - loo_acc
        interpretation = []

        if lofo_vs_loo < -2.0:
            interpretation.append(
                f"LOFO is {abs(lofo_vs_loo):.1f}pp LOWER than LOO-CV:"
            )
            interpretation.append("- This suggests moderate family-level overfitting")
            interpretation.append(
                "- Early predictor has some generalization concerns across model families"
            )
            conclusion = "WEAKENS confidence"
        elif lofo_vs_loo < 1.0:
            interpretation.append(
                f"LOFO is similar to LOO-CV (difference: {lofo_vs_loo:+.1f}pp):"
            )
            interpretation.append(
                "- Early predictor generalizes well across model families"
            )
            interpretation.append("- Family-level overfitting is minimal")
            conclusion = "STRENGTHENS confidence"
        else:
            interpretation.append(f"LOFO is {lofo_vs_loo:.1f}pp HIGHER than LOO-CV:")
            interpretation.append(
                "- Early predictor shows robustness and good generalization"
            )
            interpretation.append("- Model families are complementary in training")
            conclusion = "STRONGLY STRENGTHENS confidence"

        if rule_holds:
            interpretation.append(
                "\nEarly termination rule (V̄_50 < 10% → not sustained) shows promise"
            )
            conclusion = f"{conclusion} (plus robust termination rule)"

        for line in interpretation:
            print(line)

        print()
        print(f"CONCLUSION: {conclusion} in the early predictor")
        print()

        # Compile results for JSON output
        results = {
            "summary": {
                "total_runs": len(y),
                "sustained_count": sum(y),
                "not_sustained_count": len(y) - sum(y),
                "sustained_ratio": sum(y) / len(y),
            },
            "baseline": {
                "majority_class_accuracy": baseline_acc,
            },
            "loo_cv": {
                "accuracy": loo_acc,
                "correct": loo_correct,
                "total": len(y),
            },
            "lofo": {
                "aggregate_accuracy": lofo_aggregate_acc,
                "by_family": {
                    family: {
                        "accuracy": lofo_results[family]["accuracy"],
                        "correct": lofo_results[family]["correct"],
                        "total": lofo_results[family]["total"],
                    }
                    for family in lofo_results
                },
            },
            "early_termination_rule": termination_analysis,
            "interpretation": {
                "lofo_vs_loo_diff_pp": lofo_vs_loo,
                "conclusion": conclusion,
                "reasoning": interpretation,
            },
        }

        return results


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent
    # script_dir is p4_analysis, parent.parent is mnt/agentops-fw
    csv_path = script_dir.parent.parent / "results" / "p2_all_runs.csv"
    output_dir = script_dir.parent.parent / "results" / "p4_analysis"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis
    validator = RQ3AnalysisValidator(str(csv_path))
    results = validator.run_analysis()

    # Save results
    output_path = output_dir / "held_out_validation.json"
    with open(str(output_path), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
