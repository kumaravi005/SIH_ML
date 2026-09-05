"""
AYUSH ML Model Optimization, Cross-Validation & Robustness Validation Engine (Section 13)

Performs:
- Stratified 5-Fold Cross-Validation (K=5)
- Hyperparameter Grid Search & Tuning
- Out-of-Distribution Noise Sensitivity & Robustness Testing (5%, 10%, 15%, 20% noise)
- Statistical Stability Metrics (Mean Accuracy, Mean Macro-F1, Standard Deviation sigma)

Strictly enforces 10 active parameters. Excludes Agni and Koshtha. Zero external mandatory dependencies.
"""

import json
import math
import time
import random
from pathlib import Path
from leakage_free_ml_engine import (
    AYUSHLeakageFreeMLEngine,
    ACTIVE_PARAMETERS,
    EXCLUDED_PARAMETERS
)
from ml_baseline_model import PurePythonKNNClassifier
from advanced_ml_engine import (
    PurePythonGaussianNB,
    PurePythonDecisionTree,
    PurePythonVotingEnsemble
)


class AYUSHMLRobustnessEngine:
    def __init__(self, dataset_path=None):
        base_dir = Path(__file__).resolve().parent.parent
        self.dataset_path = dataset_path or base_dir / "data" / "ayush_synthetic_dataset.json"
        self.output_report_file = base_dir / "output" / "ml_optimization_robustness_report.json"
        self.leak_free_engine = AYUSHLeakageFreeMLEngine(dataset_path=self.dataset_path)

    def _load_dataset(self):
        if Path(self.dataset_path).exists():
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _get_stratified_folds(self, k_folds=5):
        dataset = self._load_dataset()
        if not dataset:
            return []

        # Group by class
        by_class = {}
        for idx, case in enumerate(dataset):
            feats = self.leak_free_engine.extract_raw_features(case)
            dashavidha = case.get("ayushAssessment", {}).get("dashavidhaPariksha", {})
            p_val = dashavidha.get("prakriti", {}).get("value", "")
            target = str(p_val).lower().split("-")[0] if p_val else "unknown"

            if target not in by_class:
                by_class[target] = []
            by_class[target].append((feats, target))

        random.seed(42)
        folds = [[] for _ in range(k_folds)]

        for c, samples in by_class.items():
            random.shuffle(samples)
            for i, sample in enumerate(samples):
                folds[i % k_folds].append(sample)

        return folds

    def run_stratified_kfold_cv(self, k_folds=5, k_neighbors=5):
        """
        Execute Stratified K-Fold Cross Validation.
        """
        folds = self._get_stratified_folds(k_folds=k_folds)
        if not folds:
            return {"error": "Folds creation failed"}

        model_names = [
            "Model A (KNN)",
            "Model B (Gaussian Naive Bayes)",
            "Model C (Random Forest / DT)",
            "Model D (Soft-Voting Ensemble)"
        ]

        cv_results = {name: {"fold_accuracies": [], "fold_macro_f1s": []} for name in model_names}

        for i in range(k_folds):
            test_samples = folds[i]
            train_samples = []
            for j in range(k_folds):
                if j != i:
                    train_samples.extend(folds[j])

            X_train, y_train = [s[0] for s in train_samples], [s[1] for s in train_samples]
            X_test, y_test = [s[0] for s in test_samples], [s[1] for s in test_samples]

            knn = PurePythonKNNClassifier(k=k_neighbors)
            nb = PurePythonGaussianNB()
            dt = PurePythonDecisionTree(depth=5)
            ensemble = PurePythonVotingEnsemble(knn, nb, dt)

            candidates = [
                ("Model A (KNN)", knn),
                ("Model B (Gaussian Naive Bayes)", nb),
                ("Model C (Random Forest / DT)", dt),
                ("Model D (Soft-Voting Ensemble)", ensemble)
            ]

            for name, model in candidates:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                correct = sum(1 for p, t in zip(y_pred, y_test) if p == t)
                acc = round(correct / float(len(y_test)), 4) if y_test else 0.0

                classes = list(set(y_test))
                per_class_f1 = []
                for c in classes:
                    tp = sum(1 for p, t in zip(y_pred, y_test) if p == c and t == c)
                    fp = sum(1 for p, t in zip(y_pred, y_test) if p == c and t != c)
                    fn = sum(1 for p, t in zip(y_pred, y_test) if p != c and t == c)

                    prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
                    rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                    per_class_f1.append(f1)

                macro_f1 = round(sum(per_class_f1) / float(len(classes)), 4) if classes else 0.0

                cv_results[name]["fold_accuracies"].append(acc)
                cv_results[name]["fold_macro_f1s"].append(macro_f1)

        # Compute summary statistics (mean & std dev sigma)
        summary = {}
        for name, metrics in cv_results.items():
            accs = metrics["fold_accuracies"]
            f1s = metrics["fold_macro_f1s"]

            mean_acc = sum(accs) / float(len(accs))
            var_acc = sum((a - mean_acc) ** 2 for a in accs) / float(len(accs))
            std_acc = math.sqrt(var_acc)

            mean_f1 = sum(f1s) / float(len(f1s))
            var_f1 = sum((f - mean_f1) ** 2 for f in f1s) / float(len(f1s))
            std_f1 = math.sqrt(var_f1)

            summary[name] = {
                "mean_accuracy": round(mean_acc, 4),
                "std_accuracy_sigma": round(std_acc, 4),
                "mean_macro_f1": round(mean_f1, 4),
                "std_macro_f1_sigma": round(std_f1, 4),
                "fold_accuracies": accs,
                "fold_macro_f1s": f1s
            }

        return summary

    def grid_search_hyperparameters(self):
        """
        Hyperparameter grid search for KNN (k in [1, 3, 5, 7, 9]).
        """
        k_values = [1, 3, 5, 7, 9]
        k_tuning_results = {}

        for k in k_values:
            cv_summary = self.run_stratified_kfold_cv(k_folds=5, k_neighbors=k)
            knn_res = cv_summary.get("Model A (KNN)", {})
            k_tuning_results[f"k={k}"] = {
                "mean_accuracy": knn_res.get("mean_accuracy"),
                "std_accuracy_sigma": knn_res.get("std_accuracy_sigma"),
                "mean_macro_f1": knn_res.get("mean_macro_f1")
            }

        best_k_entry = max(k_tuning_results.items(), key=lambda item: item[1]["mean_macro_f1"])

        return {
            "knn_k_grid_search": k_tuning_results,
            "optimal_knn_k": best_k_entry[0],
            "optimal_cv_macro_f1": best_k_entry[1]["mean_macro_f1"]
        }

    def evaluate_noise_robustness(self, noise_levels=[0.05, 0.10, 0.15, 0.20]):
        """
        Out-of-Distribution & Noise Sensitivity Testing.
        Evaluates model degradation when feature vectors are perturbed with Gaussian noise.
        """
        splits = self.leak_free_engine.load_leak_free_splits(train_ratio=0.8)
        X_train, y_train = splits["X_train"], splits["y_train"]
        X_test, y_test = splits["X_test"], splits["y_test"]

        # Train champion model (KNN k=5)
        model = PurePythonKNNClassifier(k=5)
        model.fit(X_train, y_train)

        clean_pred = model.predict(X_test)
        clean_correct = sum(1 for p, t in zip(clean_pred, y_test) if p == t)
        baseline_acc = round(clean_correct / float(len(y_test)), 4) if y_test else 0.0

        random.seed(42)
        robustness_matrix = {}

        for eta in noise_levels:
            perturbed_X = []
            for sample in X_test:
                perturbed = [
                    min(1.0, max(0.0, val + random.gauss(0, eta)))
                    for val in sample
                ]
                perturbed_X.append(perturbed)

            p_pred = model.predict(perturbed_X)
            p_correct = sum(1 for p, t in zip(p_pred, y_test) if p == t)
            p_acc = round(p_correct / float(len(y_test)), 4) if y_test else 0.0

            retention_ratio = round(p_acc / baseline_acc, 4) if baseline_acc > 0 else 0.0

            robustness_matrix[f"noise_{int(eta*100)}pct"] = {
                "noise_level_sigma": eta,
                "perturbed_accuracy": p_acc,
                "baseline_clean_accuracy": baseline_acc,
                "robustness_retention_ratio": retention_ratio
            }

        return robustness_matrix

    def run_full_optimization_and_robustness_suite(self):
        """
        Execute Section 13 Model Optimization, Cross-Validation & Robustness Suite.
        """
        cv_summary = self.run_stratified_kfold_cv(k_folds=5, k_neighbors=5)
        tuning_summary = self.grid_search_hyperparameters()
        noise_summary = self.evaluate_noise_robustness()

        champion_cv = cv_summary.get("Model D (Soft-Voting Ensemble)", cv_summary.get("Model A (KNN)", {}))
        mean_cv_acc = champion_cv.get("mean_accuracy", 0.0)

        report = {
            "section": "Section 13 — ML Model Optimization, Cross-Validation & Robustness Validation",
            "cross_validation_strategy": "Stratified 5-Fold Cross-Validation (K=5)",
            "stratified_kfold_cv_results": cv_summary,
            "hyperparameter_optimization": tuning_summary,
            "noise_sensitivity_robustness": noise_summary,
            "active_parameters_evaluated": len(ACTIVE_PARAMETERS),
            "excluded_parameters": EXCLUDED_PARAMETERS,
            "synthetic_data_disclaimer": "Metrics evaluated on synthetic dataset (N=500). Clinical trial validation required before clinical deployment.",
            "overall_status": "PASSED" if mean_cv_acc >= 0.70 else "NEEDS_TUNING"
        }

        self.output_report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report


if __name__ == "__main__":
    engine = AYUSHMLRobustnessEngine()
    print("Running Section 13 ML Model Optimization, Cross-Validation & Robustness Validation...")
    report = engine.run_full_optimization_and_robustness_suite()
    print("Validation Suite Complete!")
    print(json.dumps(report, indent=2))
