import unittest
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(BASE_DIR))

from clinical_prediction_engine import ClinicalPredictionEngine, predict_clinical_outcome


class TestClinicalPredictionEngine(unittest.TestCase):

    def setUp(self):
        self.engine = ClinicalPredictionEngine()
        self.sample_case = {
            "patient_id": "P-1001",
            "patient": {"age": 35, "gender": "Male"},
            "ayush_assessment": {
                "prakriti": {"primary_dosha": "pitta", "prakriti_type": "Dwandvaja (Pitta-Vata)"},
                "vikriti": {"current_changes": ["increased dryness"]},
                "sara": {"quality": "madhyama"},
                "samhanana": {"compactness": "madhyama"},
                "pramana": {"proportions": "madhyama"},
                "satmya": {"adaptability": "madhyama"},
                "sattva": {"mental_strength": "pravara"},
                "aharaShakti": {"digestive_capacity": "madhyama"},
                "vyayamaShakti": {"physical_work_capacity": "madhyama"},
                "vaya": {"age": 35, "age_group": "Madhyama Vaya"}
            }
        }

    def test_stabilization_probability(self):
        prob = self.engine.predict_dosha_stabilization_probability(
            self.sample_case["ayush_assessment"], adherence_level="high"
        )
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 100.0)

    def test_timeframe_estimation(self):
        tf = self.engine.estimate_recovery_timeframe_weeks(
            resilience_score=70.0, imbalance_score=20.0, adherence_level="high"
        )
        self.assertEqual(tf["recovery_category"], "Rapid Recovery")
        self.assertIn("weeks", tf["timeframe_display"])

    def test_prognostic_factors(self):
        report = predict_clinical_outcome(self.sample_case, adherence_level="moderate")
        factors = report["prognostic_analysis"]
        self.assertIn("positive_prognostic_indicators", factors)
        self.assertIn("risk_and_delay_factors", factors)

    def test_agni_koshtha_exclusion(self):
        report = predict_clinical_outcome(self.sample_case)
        self.assertEqual(report["assessment_summary"]["active_parameters_evaluated"], 10)
        self.assertEqual(report["assessment_summary"]["excluded_parameters"], ["agni", "koshtha"])


if __name__ == "__main__":
    unittest.main()
