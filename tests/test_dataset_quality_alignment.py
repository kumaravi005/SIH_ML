"""
Unit Tests for AYUSHDatasetQualityAlignmentEngine (Section 11)
"""

import sys
import unittest
from pathlib import Path

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset_quality_alignment_engine import AYUSHDatasetQualityAlignmentEngine, ACTIVE_PARAMETERS, EXCLUDED_PARAMETERS


class TestAYUSHDatasetQualityAlignmentEngine(unittest.TestCase):
    def setUp(self):
        self.engine = AYUSHDatasetQualityAlignmentEngine()

    def test_analyze_feature_target_alignment(self):
        metrics = self.engine.analyze_feature_target_alignment()
        self.assertIn("class_distribution", metrics)
        self.assertIn("entropy_class_balance_score", metrics)
        self.assertTrue(metrics["entropy_class_balance_score"] >= 0.90)
        self.assertTrue(metrics["signal_to_noise_ratio"] > 1.0)
        self.assertFalse(metrics["target_leakage_present"])
        self.assertEqual(metrics["active_parameters_evaluated"], len(ACTIVE_PARAMETERS))
        self.assertEqual(metrics["excluded_parameters"], EXCLUDED_PARAMETERS)

    def test_run_full_alignment_audit(self):
        report = self.engine.run_full_alignment_audit()
        self.assertIn("dataset_alignment_metrics", report)
        self.assertIn("retrained_leakage_free_evaluation", report)
        self.assertIn("overall_status", report)
        eval_dict = report["retrained_leakage_free_evaluation"]
        self.assertTrue(eval_dict["leakage_free_verified"])
        self.assertEqual(eval_dict["excluded_parameters"], EXCLUDED_PARAMETERS)


if __name__ == "__main__":
    unittest.main()
