import unittest
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(BASE_DIR))

from eval_framework import MLEvaluationFramework, run_full_benchmark_suite


class TestMLEvaluationFramework(unittest.TestCase):

    def setUp(self):
        self.framework = MLEvaluationFramework()

    def test_extraction_precision_recall_f1(self):
        extracted = [{"entity": "Headache"}, {"entity": "Fatigue"}, {"entity": "Fever"}]
        ground_truth = [{"entity": "Headache"}, {"entity": "Fatigue"}]

        metrics = self.framework.evaluate_extraction_performance(extracted, ground_truth)
        self.assertEqual(metrics["true_positives"], 2)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["precision"], 0.6667)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertGreater(metrics["f1_score"], 0.7)

    def test_ayush_assessment_accuracy(self):
        assessed = {
            "prakriti": "pitta",
            "vikriti": "dryness",
            "sara": "madhyama",
            "samhanana": "madhyama",
            "pramana": "madhyama",
            "satmya": "madhyama",
            "sattva": "pravara",
            "aharaShakti": "madhyama",
            "vyayamaShakti": "madhyama",
            "vaya": "adulthood"
        }
        ground_truth = dict(assessed)

        metrics = self.framework.evaluate_ayush_assessment_accuracy(assessed, ground_truth)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["correct_parameters_count"], 10)
        self.assertEqual(metrics["active_parameters_count"], 10)
        self.assertEqual(metrics["excluded_parameters"], ["agni", "koshtha"])

    def test_mrr_calculation(self):
        retrieved = [{"patient_id": "HIST-001"}, {"patient_id": "TARGET-001"}]
        mrr = self.framework.evaluate_retrieval_mrr(retrieved, "TARGET-001")
        self.assertEqual(mrr, 0.5)

    def test_full_benchmark_suite(self):
        report = run_full_benchmark_suite()
        self.assertEqual(report["benchmark_status"], "PASSED")
        self.assertIn("clinical_extraction", report["evaluated_components"])
        self.assertIn("ayush_10_parameter_assessment", report["evaluated_components"])
        self.assertEqual(report["evaluated_components"]["ayush_10_parameter_assessment"]["active_parameters_evaluated"], 10)


if __name__ == "__main__":
    unittest.main()
