import unittest
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(BASE_DIR))

from regimen_optimizer_engine import RegimenOptimizerEngine, generate_personalized_regimen_report


class TestRegimenOptimizerEngine(unittest.TestCase):

    def setUp(self):
        self.engine = RegimenOptimizerEngine()
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
                "vyayamaShakti": {"physical_work_capacity": "pravara"},
                "vaya": {"age": 35, "age_group": "Madhyama Vaya"}
            }
        }

    def test_dinacharya_routine_generation(self):
        routine = self.engine.optimize_daily_dinacharya_schedule(self.sample_case["ayush_assessment"])
        self.assertEqual(len(routine), 4)
        self.assertIn("Morning", routine[0]["time_block"])
        self.assertIn("Night", routine[3]["time_block"])

    def test_ritucharya_seasonal_customization(self):
        plan = self.engine.optimize_seasonal_ritucharya(self.sample_case["ayush_assessment"], season="Sharad")
        self.assertIn("Sharad", plan["season_name"])
        self.assertIn("dietary_guidelines", plan)

    def test_full_report_generation(self):
        report = generate_personalized_regimen_report(self.sample_case, season="Vasanta")
        self.assertIn("daily_dinacharya_routine", report)
        self.assertIn("seasonal_ritucharya_plan", report)

    def test_agni_koshtha_exclusion(self):
        report = generate_personalized_regimen_report(self.sample_case)
        self.assertEqual(report["assessment_summary"]["active_parameters_evaluated"], 10)
        self.assertEqual(report["assessment_summary"]["excluded_parameters"], ["agni", "koshtha"])


if __name__ == "__main__":
    unittest.main()
