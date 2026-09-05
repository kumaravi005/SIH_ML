"""
Unit Tests for AYUSHLeakageFreeMLEngine (Section 10)
"""

import sys
import unittest
from pathlib import Path

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from leakage_free_ml_engine import AYUSHLeakageFreeMLEngine, ACTIVE_PARAMETERS, EXCLUDED_PARAMETERS


class TestAYUSHLeakageFreeMLEngine(unittest.TestCase):
    def setUp(self):
        self.engine = AYUSHLeakageFreeMLEngine()

    def test_extract_raw_features_and_leak_free_verification(self):
        sample_case = {
            "patient": {"age": 42, "gender": "Male"},
            "questionnaire_answers": {
                "prakriti": [
                    {"questionId": "prakriti_q01", "answer": "Thin/light build"},
                    {"questionId": "prakriti_q02", "answer": "Usually dry"}
                ]
            },
            "unstructured_inputs": [
                {"text": "Patient has dry skin and mild headache", "source": "text"}
            ]
        }
        features = self.engine.extract_raw_features(sample_case)
        self.assertTrue(len(features) >= 40)
        # Verify zero post-assessment dashavidha values present
        self.engine.verify_zero_leakage(features, sample_case)

    def test_train_and_evaluate_all_leakage_free(self):
        report = self.engine.train_and_evaluate_all()
        self.assertTrue(report["leakage_free_verified"])
        self.assertEqual(report["training_samples"], 400)
        self.assertEqual(report["testing_samples"], 100)
        self.assertTrue(report["champion_accuracy"] > 0.33)
        self.assertEqual(report["excluded_parameters"], EXCLUDED_PARAMETERS)


if __name__ == "__main__":
    unittest.main()
