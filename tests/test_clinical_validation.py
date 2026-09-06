"""
Unit tests for Section 16: Independent Real-World / Expert Validation & Clinical Evaluation Framework
"""

import unittest
import json
from pathlib import Path
import sys

# Ensure src directory is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clinical_validation_engine import AYUSHClinicalValidationEngine, compute_cohens_kappa, build_confusion_matrix
from leakage_free_ml_engine import ACTIVE_PARAMETERS, EXCLUDED_PARAMETERS


class TestClinicalValidationEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = AYUSHClinicalValidationEngine()

    def test_cohens_kappa_calculation(self):
        """
        Test mathematical correctness of Cohen's Kappa calculation.
        """
        # Perfect agreement
        y_true = ["vata", "pitta", "kapha", "vata"]
        y_pred = ["vata", "pitta", "kapha", "vata"]
        labels = ["vata", "pitta", "kapha"]
        kappa_perfect = compute_cohens_kappa(y_true, y_pred, labels)
        self.assertEqual(kappa_perfect, 1.0)

        # Zero agreement / random
        y_true_rand = ["vata", "pitta", "kapha", "vata"]
        y_pred_rand = ["pitta", "kapha", "vata", "pitta"]
        kappa_rand = compute_cohens_kappa(y_true_rand, y_pred_rand, labels)
        self.assertLessEqual(kappa_rand, 0.0)

    def test_confusion_matrix_builder(self):
        """
        Test 3x3 confusion matrix construction.
        """
        y_true = ["vata", "pitta", "kapha", "vata"]
        y_pred = ["vata", "pitta", "kapha", "pitta"]
        labels = ["vata", "pitta", "kapha"]

        matrix = build_confusion_matrix(y_true, y_pred, labels)
        self.assertEqual(matrix["vata"]["vata"], 1)
        self.assertEqual(matrix["vata"]["pitta"], 1)
        self.assertEqual(matrix["pitta"]["pitta"], 1)
        self.assertEqual(matrix["kapha"]["kapha"], 1)

    def test_full_clinical_validation_run(self):
        """
        Test execution of full clinical validation protocol against expert benchmark cases.
        """
        report = self.engine.run_full_clinical_validation()

        self.assertIn("total_expert_benchmark_cases", report)
        self.assertGreaterEqual(report["total_expert_benchmark_cases"], 60)

        self.assertIn("blind_evaluation_protocol", report)
        self.assertIn("ACTIVE", report["blind_evaluation_protocol"])

        self.assertIn("overall_agreement_metrics", report)
        metrics = report["overall_agreement_metrics"]
        self.assertIn("accuracy", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.80)
        self.assertIn("cohens_kappa_agreement", metrics)

        self.assertIn("confidence_stratification_analysis", report)
        conf_analysis = report["confidence_stratification_analysis"]
        self.assertIn("low_confidence_flagged", conf_analysis)
        self.assertIn("action_required", conf_analysis["low_confidence_flagged"])

        self.assertIn("synthetic_vs_real_transfer_comparison", report)
        transfer = report["synthetic_vs_real_transfer_comparison"]
        self.assertIn("synthetic_transfer_degradation_gap", transfer)

        self.assertIn("clinical_safety_boundary", report)
        safety = report["clinical_safety_boundary"]
        self.assertFalse(safety["autonomous_diagnosis_permitted"])
        self.assertEqual(safety["excluded_parameters"], EXCLUDED_PARAMETERS)

    def test_blind_evaluation_isolation(self):
        """
        Verify that expert ground-truth labels are withheld from the model during prediction.
        """
        expert_cases = self.engine.load_expert_benchmark_cases(min_samples=10)
        sample_case = expert_cases[0]

        # Strip expert_validation
        blind_case = json.loads(json.dumps(sample_case))
        if "expert_validation" in blind_case:
            del blind_case["expert_validation"]

        # Prediction should work seamlessly on blind case
        res = self.engine.serving_engine.predict_with_explanation(blind_case)
        self.assertIn("predicted_prakriti", res)
        self.assertIn(res["predicted_prakriti"], ["vata", "pitta", "kapha"])


if __name__ == "__main__":
    unittest.main()
