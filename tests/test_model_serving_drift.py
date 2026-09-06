"""
Unit tests for Section 15: Production Model Serving, Local Feature Explainability & Data Drift Engine
"""

import unittest
import json
from pathlib import Path
import sys

# Ensure src directory is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model_serving_drift_engine import AYUSHModelServingDriftEngine, FEATURE_NAMES
from dataset_generator import AYUSHDatasetGeneratorEngine
from leakage_free_ml_engine import ACTIVE_PARAMETERS, EXCLUDED_PARAMETERS


class TestModelServingAndDriftEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = AYUSHModelServingDriftEngine()
        cls.generator = AYUSHDatasetGeneratorEngine()

    def test_predict_with_explanation(self):
        """
        Test prediction with instance-level local feature explanations.
        """
        # Generate sample patient
        generator = AYUSHDatasetGeneratorEngine(seed=42)
        patient_case = generator.generate_patient_case(1)

        res = self.engine.predict_with_explanation(patient_case)
        
        self.assertIn("predicted_prakriti", res)
        self.assertIn(res["predicted_prakriti"], ["vata", "pitta", "kapha"])
        self.assertIn("confidence", res)
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)
        
        self.assertIn("class_probabilities", res)
        self.assertEqual(len(res["class_probabilities"]), 3)

        self.assertIn("local_explanation", res)
        explanation = res["local_explanation"]
        self.assertIn("top_supporting_features", explanation)
        self.assertIn("top_contrasting_features", explanation)
        self.assertIn("explanation_disclaimer", explanation)
        
        # Verify 10 active parameters only, Agni and Koshtha strictly excluded
        self.assertEqual(res["excluded_parameters"], EXCLUDED_PARAMETERS)
        self.assertNotIn("agni", res)
        self.assertNotIn("koshtha", res)

    def test_in_distribution_drift_monitoring(self):
        """
        Test data drift monitoring on in-distribution dataset (N=50).
        Should yield NO_DRIFT status or low severity.
        """
        generator = AYUSHDatasetGeneratorEngine(seed=999)
        cases = [generator.generate_patient_case(i) for i in range(1, 51)]

        report = self.engine.detect_data_drift(incoming_cases=cases)

        self.assertEqual(report["feature_dimension"], 51)
        self.assertIn("drift_summary", report)
        summary = report["drift_summary"]
        self.assertIn(summary["overall_drift_status"], ["NO_DRIFT", "SLIGHT_DRIFT"])
        self.assertFalse(summary["automatic_retraining_triggered"])
        
        self.assertIn("disclaimers", report)
        self.assertIn("synthetic_baseline_limitation", report["disclaimers"])

    def test_artificially_shifted_drift_detection(self):
        """
        Test drift detector on artificially shifted/perturbed patient inputs.
        Should detect distribution shift and report explicit KS & Wasserstein statistics without retraining.
        """
        generator = AYUSHDatasetGeneratorEngine(seed=888)
        cases = [generator.generate_patient_case(i) for i in range(1, 51)]

        # Artificially shift age, gender, and questionnaire answers in incoming cases
        shifted_cases = []
        for case in cases:
            case_copy = json.loads(json.dumps(case))
            # Shift age by +50 years
            case_copy["patient"]["age"] = 95
            case_copy["patient"]["gender"] = "other"
            # Shift questionnaire answers to foreign strings
            answers = case_copy.get("questionnaire_answers", {})
            for p in answers:
                for u_item in answers[p]:
                    if isinstance(u_item, dict):
                        u_item["answer"] = "SHIFTED_CUSTOM_RESPONSE_" + str(u_item.get("questionId", ""))
            shifted_cases.append(case_copy)

        report = self.engine.detect_data_drift(incoming_cases=shifted_cases)

        summary = report["drift_summary"]
        self.assertGreater(summary["drifted_features_count"], 5)
        self.assertIn(summary["overall_drift_status"], ["SLIGHT_DRIFT", "SIGNIFICANT_DRIFT"])
        self.assertFalse(summary["automatic_retraining_triggered"])

        # Check per-feature details
        details = report["per_feature_drift_details"]
        self.assertEqual(len(details), 51)
        # Age feature (index 0) should be flagged as drifted
        age_detail = details[0]
        self.assertEqual(age_detail["feature_name"], "age")
        self.assertTrue(age_detail["is_drifted"])
        self.assertGreater(age_detail["wasserstein_distance"], 0.15)

    def test_zero_target_leakage_in_drift_pipeline(self):
        """
        Verify that drift detection uses ONLY raw input features and does not inspect target labels.
        """
        generator = AYUSHDatasetGeneratorEngine(seed=777)
        cases = [generator.generate_patient_case(i) for i in range(1, 11)]

        # Strip all target labels from patient cases
        stripped_cases = []
        for case in cases:
            c = json.loads(json.dumps(case))
            if "dashavidhaPariksha" in c.get("ayushAssessment", {}):
                del c["ayushAssessment"]["dashavidhaPariksha"]
            stripped_cases.append(c)

        # Drift detection should execute perfectly without target labels
        report = self.engine.detect_data_drift(incoming_cases=stripped_cases)
        self.assertEqual(report["incoming_evaluation_samples"], 10)
        self.assertEqual(report["feature_dimension"], 51)


if __name__ == "__main__":
    unittest.main()
