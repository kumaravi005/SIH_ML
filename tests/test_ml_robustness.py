"""
Unit Tests for AYUSHMLRobustnessEngine (Section 13)
"""

import sys
import unittest
from pathlib import Path

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_robustness_engine import AYUSHMLRobustnessEngine, ACTIVE_PARAMETERS, EXCLUDED_PARAMETERS


class TestAYUSHMLRobustnessEngine(unittest.TestCase):
    def setUp(self):
        self.engine = AYUSHMLRobustnessEngine()

    def test_run_stratified_kfold_cv(self):
        cv_res = self.engine.run_stratified_kfold_cv(k_folds=5)
        self.assertIn("Model A (KNN)", cv_res)
        self.assertIn("Model D (Soft-Voting Ensemble)", cv_res)
        knn_res = cv_res["Model A (KNN)"]
        self.assertTrue(knn_res["mean_accuracy"] >= 0.85)
        self.assertTrue(knn_res["std_accuracy_sigma"] < 0.05)
        self.assertEqual(len(knn_res["fold_accuracies"]), 5)

    def test_grid_search_hyperparameters(self):
        grid_res = self.engine.grid_search_hyperparameters()
        self.assertIn("knn_k_grid_search", grid_res)
        self.assertIn("optimal_knn_k", grid_res)
        self.assertTrue(grid_res["optimal_cv_macro_f1"] >= 0.85)

    def test_evaluate_noise_robustness(self):
        noise_res = self.engine.evaluate_noise_robustness(noise_levels=[0.05, 0.10, 0.15, 0.20])
        self.assertIn("noise_10pct", noise_res)
        self.assertTrue(noise_res["noise_10pct"]["robustness_retention_ratio"] >= 0.70)

    def test_run_full_optimization_and_robustness_suite(self):
        report = self.engine.run_full_optimization_and_robustness_suite()
        self.assertIn("stratified_kfold_cv_results", report)
        self.assertIn("hyperparameter_optimization", report)
        self.assertIn("noise_sensitivity_robustness", report)
        self.assertEqual(report["overall_status"], "PASSED")
        self.assertEqual(report["excluded_parameters"], EXCLUDED_PARAMETERS)
        self.assertIn("synthetic_data_disclaimer", report)


if __name__ == "__main__":
    unittest.main()
