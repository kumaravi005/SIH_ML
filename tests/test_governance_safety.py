"""
Unit tests for Section 17: Clinical Validation Evidence, Model Governance & Production Safety Engine
"""

import unittest
import json
from pathlib import Path
import sys

# Ensure src directory is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from governance_safety_engine import (
    AYUSHGovernanceSafetyEngine,
    compute_input_sha256,
    compute_expected_calibration_error
)
from leakage_free_ml_engine import ACTIVE_PARAMETERS, EXCLUDED_PARAMETERS


class TestGovernanceSafetyEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = AYUSHGovernanceSafetyEngine()

    def test_input_sha256_computation(self):
        """
        Test cryptographic SHA-256 hash generation for patient case audit traceability.
        """
        patient_case = {
            "patient": {"patientId": "P_001", "age": 30, "gender": "female"},
            "questionnaire_answers": {"prakriti": [{"questionId": "q1", "answer": "Warm skin"}]}
        }
        hash1 = compute_input_sha256(patient_case)
        hash2 = compute_input_sha256(patient_case)
        
        self.assertEqual(len(hash1), 64)  # 64-char hex string
        self.assertEqual(hash1, hash2)   # Deterministic

    def test_expected_calibration_error(self):
        """
        Test Expected Calibration Error (ECE) calculation logic.
        """
        # Perfectly calibrated: confidence 0.9 with 90% accuracy
        confidences = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        y_true = ["vata"] * 9 + ["pitta"]
        y_pred = ["vata"] * 10
        
        ece, bins = compute_expected_calibration_error(confidences, y_true, y_pred, num_bins=5)
        self.assertLessEqual(ece, 0.05)
        self.assertEqual(len(bins), 5)

    def test_model_card_generation(self):
        """
        Test formal Model Card generation and serialization to disk.
        """
        card = self.engine.generate_model_card()

        self.assertIn("model_details", card)
        self.assertEqual(card["model_details"]["name"], "AYUSH Prakriti KNN Champion Classifier")
        self.assertIn("intended_use", card)
        self.assertIn("factors_and_features", card)
        self.assertEqual(card["factors_and_features"]["excluded_parameters"], EXCLUDED_PARAMETERS)
        
        # Verify file written to disk
        self.assertTrue(self.engine.model_card_file.exists())

    def test_prediction_audit_logging(self):
        """
        Test structured audit log stream writing to output/governance_audit_trail.log.
        """
        sample_case = {
            "patient": {"patientId": "AUDIT_TEST_PATIENT", "age": 45, "gender": "male"},
            "questionnaire_answers": {}
        }
        pred_res = {
            "predicted_prakriti": "vata",
            "confidence": 0.89,
            "overall_drift_status": "NO_DRIFT"
        }

        entry = self.engine.log_prediction_audit(sample_case, pred_res)

        self.assertIn("audit_id", entry)
        self.assertIn("input_sha256_hash", entry)
        self.assertEqual(entry["human_review_status"], "APPROVED_CDS")
        
        # Verify file contains log lines
        self.assertTrue(self.engine.audit_log_file.exists())

    def test_full_governance_safety_audit(self):
        """
        Test full Section 17 Governance, Audit & Production Safety Audit pipeline.
        """
        report = self.engine.run_full_governance_safety_audit()

        self.assertIn("dataset_provenance", report)
        self.assertIn("inter_practitioner_kappa", report["dataset_provenance"])

        self.assertIn("calibration_reliability_analysis", report)
        calib = report["calibration_reliability_analysis"]
        self.assertIn("expected_calibration_error_ece", calib)

        self.assertIn("error_analysis_summary", report)
        self.assertIn("model_card_serialization", report)

        self.assertIn("production_safety_boundaries", report)
        safety = report["production_safety_boundaries"]
        self.assertFalse(safety["autonomous_diagnosis_permitted"])
        self.assertTrue(safety["clinical_decision_support_only"])
        self.assertEqual(safety["excluded_parameters"], EXCLUDED_PARAMETERS)


if __name__ == "__main__":
    unittest.main()
