import unittest
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(BASE_DIR))

from patient_similarity_engine import PatientSimilarityEngine, find_similar_patient_cases


class TestPatientSimilarityEngine(unittest.TestCase):

    def setUp(self):
        self.engine = PatientSimilarityEngine()
        self.target_case = {
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
                "vaya": {"age_group": "Madhyama Vaya"}
            }
        }

        self.historical_cases = [
            {
                "patient_id": "HIST-001",
                "patient": {"age": 36, "gender": "Male"},
                "ayush_assessment": {
                    "prakriti": {"primary_dosha": "pitta", "prakriti_type": "Dwandvaja (Pitta-Vata)"},
                    "vikriti": {"current_changes": ["increased dryness"]},
                    "sara": {"quality": "madhyama"},
                    "samhanana": {"compactness": "madhyama"},
                    "sattva": {"mental_strength": "pravara"}
                }
            },
            {
                "patient_id": "HIST-002",
                "patient": {"age": 60, "gender": "Female"},
                "ayush_assessment": {
                    "prakriti": {"primary_dosha": "kapha", "prakriti_type": "Kapha"},
                    "vikriti": {"current_changes": []},
                    "sara": {"quality": "avara"},
                    "samhanana": {"compactness": "avara"},
                    "sattva": {"mental_strength": "avara"}
                }
            }
        ]

    def test_vectorization_length_and_values(self):
        vec = self.engine.vectorize_patient_case(self.target_case)
        self.assertEqual(len(vec), 12)

    def test_cosine_similarity(self):
        vec1 = [1.0, 0.0, 0.5, 0.5]
        vec2 = [1.0, 0.0, 0.5, 0.5]
        sim = self.engine.calculate_cosine_similarity(vec1, vec2)
        self.assertEqual(sim, 1.0)

    def test_find_similar_cases_ranking(self):
        report = find_similar_patient_cases(self.target_case, self.historical_cases, top_k=2)
        matches = report["retrieved_similar_cases"]
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0]["patient_id"], "HIST-001")
        self.assertGreater(matches[0]["similarity_score"], matches[1]["similarity_score"])

    def test_agni_koshtha_exclusion(self):
        report = find_similar_patient_cases(self.target_case, self.historical_cases)
        self.assertEqual(report["assessment_summary"]["active_parameters_evaluated"], 10)
        self.assertEqual(report["assessment_summary"]["excluded_parameters"], ["agni", "koshtha"])


if __name__ == "__main__":
    unittest.main()
