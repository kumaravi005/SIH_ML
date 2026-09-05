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
        self.assertIn("Model A (KNN)", matrix)
        self.assertIn("Model B (GNB)", matrix)
        self.assertIn("Model C (RF/DT)", matrix)
        self.assertIn("Model D (Ensemble)", matrix)

        self.assertTrue(report["champion_accuracy"] >= 0.80)
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
