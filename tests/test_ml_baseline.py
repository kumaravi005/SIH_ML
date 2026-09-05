"""
Unit Tests for AYUSHBaselineMLEngine (Section 8)
"""

import sys
import unittest
from pathlib import Path

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_baseline_model import AYUSHBaselineMLEngine, ACTIVE_PARAMETERS, EXCLUDED_PARAMETERS


class TestAYUSHBaselineMLEngine(unittest.TestCase):
    def setUp(self):
        self.engine = AYUSHBaselineMLEngine()

    def test_feature_extraction(self):
        sample_case = {
            "patient": {"age": 35},
            "ayushAssessment": {
                "dashavidhaPariksha": {
                    "prakriti": {"value": "vata", "confidence": 0.9},
                    "vikriti": {"value": "pitta", "confidence": 0.85}
                }
            }
        }
        vec = self.engine.extract_feature_vector(sample_case)
        self.assertEqual(len(vec), 51)  # Leak-free raw questionnaire & demographic feature vector

    def test_train_and_evaluate(self):
        report = self.engine.train_and_evaluate()
        self.assertEqual(report["training_samples"], 400)
        self.assertEqual(report["testing_samples"], 100)
        self.assertTrue(report["accuracy"] > 0.33)
        self.assertEqual(report["excluded_parameters"], EXCLUDED_PARAMETERS)

    def test_predict_case(self):
        sample_case = {
            "patient": {"age": 40},
            "ayushAssessment": {
                "dashavidhaPariksha": {
                    "prakriti": {"value": "pitta", "confidence": 0.95}
                }
            }
        }
        res = self.engine.predict_case(sample_case)
        self.assertIn("predicted_prakriti", res)
        self.assertIn("confidence", res)


if __name__ == "__main__":
    unittest.main()
