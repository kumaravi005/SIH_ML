"""
Unit Tests for AYUSHAdvancedMLEngine (Section 9)
"""

import sys
import unittest
from pathlib import Path

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from advanced_ml_engine import AYUSHAdvancedMLEngine, ACTIVE_PARAMETERS, EXCLUDED_PARAMETERS


class TestAYUSHAdvancedMLEngine(unittest.TestCase):
    def setUp(self):
        self.engine = AYUSHAdvancedMLEngine()

    def test_train_and_compare_all_models(self):
        report = self.engine.train_and_compare_all_models()
        self.assertEqual(report["training_samples"], 400)
        self.assertEqual(report["testing_samples"], 100)
        self.assertIn("comparative_metrics_matrix", report)

        matrix = report["comparative_metrics_matrix"]
        matrix_keys_str = " ".join(matrix.keys())
        self.assertIn("Model A (KNN)", matrix)
        self.assertIn("Gaussian Naive Bayes", matrix_keys_str)
        self.assertIn("Random Forest", matrix_keys_str)
        self.assertIn("Soft-Voting Ensemble", matrix_keys_str)

        self.assertTrue(report["champion_accuracy"] > 0.33)
        self.assertEqual(report["excluded_parameters"], EXCLUDED_PARAMETERS)

    def test_predict_with_champion(self):
        sample_case = {
            "patient": {"age": 45},
            "ayushAssessment": {
                "dashavidhaPariksha": {
                    "prakriti": {"value": "pitta-kapha", "confidence": 0.9}
                }
            }
        }
        res = self.engine.predict_with_champion(sample_case)
        self.assertIn("predicted_prakriti", res)
        self.assertIn("champion_model", res)
        self.assertEqual(res["excluded_parameters"], EXCLUDED_PARAMETERS)


if __name__ == "__main__":
    unittest.main()
