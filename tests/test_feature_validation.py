"""
Unit Tests for AYUSHFeatureValidationEngine (Section 12)
"""

import sys
import unittest
from pathlib import Path

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_validation_engine import AYUSHFeatureValidationEngine, ACTIVE_PARAMETERS, EXCLUDED_PARAMETERS


class TestAYUSHFeatureValidationEngine(unittest.TestCase):
    def setUp(self):
        self.engine = AYUSHFeatureValidationEngine()

    def test_question_label_agreement(self):
        report = self.engine.compute_question_label_agreement()
        self.assertIn("overall_questionnaire_label_agreement_pct", report)
        self.assertTrue(report["overall_questionnaire_label_agreement_pct"] >= 70.0)

    def test_feature_information_gain(self):
        report = self.engine.compute_feature_information_gain()
        self.assertIn("top_predictive_features", report)
        self.assertIn("average_information_gain", report)
        self.assertTrue(report["average_information_gain"] > 0.01)

    def test_run_full_feature_validation_audit(self):
        audit = self.engine.run_full_feature_validation_audit()
        self.assertIn("questionnaire_to_label_agreement", audit)
        self.assertIn("feature_information_gain_audit", audit)
        self.assertIn("retrained_leakage_free_evaluation", audit)
        self.assertEqual(audit["overall_status"], "PASSED")
        eval_dict = audit["retrained_leakage_free_evaluation"]
        self.assertTrue(eval_dict["champion_accuracy"] >= 0.70)
        self.assertEqual(eval_dict["excluded_parameters"], EXCLUDED_PARAMETERS)


if __name__ == "__main__":
    unittest.main()
