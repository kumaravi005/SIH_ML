import unittest
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(BASE_DIR))

from data_quality_engine import DataQualityEngine, generate_data_quality_report


class TestDataQualityEngine(unittest.TestCase):

    def setUp(self):
        self.engine = DataQualityEngine()
        self.valid_case = {
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

    def test_complete_case_integrity_score(self):
        report = generate_data_quality_report(self.valid_case)
        self.assertEqual(report["data_integrity"]["data_integrity_score"], 100.0)
        self.assertEqual(report["data_integrity"]["quality_grade"], "Excellent")
        self.assertTrue(report["is_valid_for_clinical_use"])

    def test_inter_parameter_consistency_warning(self):
        conflicting_case = {
            "patient_id": "P-1002",
            "ayush_assessment": {
                "sara": {"quality": "avara"},  # weak tissue excellence
                "vyayamaShakti": {"physical_work_capacity": "pravara"}  # high work capacity -> conflict!
            }
        }
        report = generate_data_quality_report(conflicting_case)
        warnings = report["consistency_warnings"]
        self.assertTrue(len(warnings) > 0)
        self.assertEqual(warnings[0]["rule_id"], "CONSISTENCY_01")
        self.assertLess(report["data_integrity"]["data_integrity_score"], 100.0)

    def test_incomplete_case_integrity(self):
        incomplete_case = {
            "patient_id": "P-1003",
            "ayush_assessment": {
                "prakriti": {"primary_dosha": "pitta"}
            }
        }
        report = generate_data_quality_report(incomplete_case)
        self.assertEqual(report["data_integrity"]["assessed_parameters_count"], 1)
        self.assertLess(report["data_integrity"]["data_integrity_score"], 70.0)
        self.assertFalse(report["is_valid_for_clinical_use"])

    def test_agni_koshtha_exclusion(self):
        report = generate_data_quality_report(self.valid_case)
        self.assertEqual(report["assessment_summary"]["active_parameters_evaluated"], 10)
        self.assertEqual(report["assessment_summary"]["excluded_parameters"], ["agni", "koshtha"])


if __name__ == "__main__":
    unittest.main()
