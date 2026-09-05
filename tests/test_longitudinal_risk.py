import unittest
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(BASE_DIR))

from longitudinal_risk_engine import LongitudinalRiskEngine, generate_longitudinal_risk_report


class TestLongitudinalRiskEngine(unittest.TestCase):

    def setUp(self):
        self.engine = LongitudinalRiskEngine()
        self.sample_case = {
            "patient_id": "P-1001",
            "visit_date": "2026-09-01",
            "ayush_assessment": {
                "prakriti": {"primary_dosha": "pitta", "prakriti_type": "Dwandvaja (Pitta-Vata)"},
                "vikriti": {"current_changes": ["increased dryness", "irritability"]},
                "sara": {"dominant_dhatu": "Rasa", "quality": "madhyama"},
                "samhanana": {"compactness": "madhyama"},
                "pramana": {"proportions": "madhyama"},
                "satmya": {"adaptability": "madhyama"},
                "sattva": {"mental_strength": "pravara"},
                "aharaShakti": {"digestive_capacity": "madhyama"},
                "vyayamaShakti": {"physical_work_capacity": "madhyama"},
                "vaya": {"age_group": "Madhyama Vaya"}
            },
            "clinical_evidence": {
                "extracted_symptoms": [
                    {"entity": "knee joint stiffness", "provenance": {"source": "unstructured_text", "confidence": 0.9}}
                ]
            }
        }

    def test_resilience_and_imbalance_scoring(self):
        res = self.engine.compute_health_resilience_score(
            self.sample_case["ayush_assessment"], self.sample_case["clinical_evidence"]
        )
        self.assertIn("resilience_score", res)
        self.assertGreaterEqual(res["resilience_score"], 0.0)
        self.assertLessEqual(res["resilience_score"], 100.0)

        imb = self.engine.compute_imbalance_severity_score(
            self.sample_case["ayush_assessment"], self.sample_case["clinical_evidence"]
        )
        self.assertIn("imbalance_severity_score", imb)
        self.assertGreater(imb["imbalance_severity_score"], 0.0)

    def test_risk_tier_determination(self):
        tier = self.engine.determine_risk_tier(resilience_score=75.0, imbalance_score=42.0)
        self.assertEqual(tier["risk_tier"], "High")
        self.assertIn("prioritized", tier["clinical_action_recommendation"].lower())

    def test_longitudinal_progress_tracking(self):
        baseline_case = {
            "patient_id": "P-1001",
            "visit_date": "2026-08-01",
            "ayush_assessment": {
                "prakriti": {"primary_dosha": "pitta"},
                "vikriti": {"current_changes": ["severe dryness", "anxiety", "insomnia", "acidity"]},
                "sara": {"quality": "madhyama"},
                "samhanana": {"compactness": "madhyama"},
                "sattva": {"mental_strength": "madhyama"},
                "vyayamaShakti": {"physical_work_capacity": "avara"}
            },
            "clinical_evidence": {
                "extracted_symptoms": [
                    {"entity": "joint stiffness", "provenance": {"confidence": 0.9}},
                    {"entity": "burning sensation", "provenance": {"confidence": 0.95}},
                    {"entity": "sleep disturbance", "provenance": {"confidence": 0.85}}
                ]
            }
        }

        report = generate_longitudinal_risk_report(self.sample_case, historical_cases=[baseline_case])
        self.assertEqual(report["patient_id"], "P-1001")
        self.assertEqual(report["longitudinal_progress"]["trajectory"], "improving")
        self.assertEqual(report["longitudinal_progress"]["visit_count"], 2)
        self.assertGreater(report["longitudinal_progress"]["vikriti_resolution_rate_percent"], 0.0)

    def test_agni_koshtha_exclusion(self):
        report = generate_longitudinal_risk_report(self.sample_case)
        self.assertEqual(report["assessment_summary"]["active_parameters_assessed"], 10)
        self.assertEqual(report["assessment_summary"]["excluded_parameters"], ["agni", "koshtha"])
        self.assertNotIn("agni", report["longitudinal_progress"].get("parameter_evolution_matrix", {}))
        self.assertNotIn("koshtha", report["longitudinal_progress"].get("parameter_evolution_matrix", {}))


if __name__ == "__main__":
    unittest.main()
