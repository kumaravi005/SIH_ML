"""
Unit Tests for AYUSHClinicalExplainerEngine (Section 6)
"""

import sys
import unittest
from pathlib import Path

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clinical_explainer_engine import AYUSHClinicalExplainerEngine, ACTIVE_PARAMETERS, EXCLUDED_PARAMETERS


class TestAYUSHClinicalExplainerEngine(unittest.TestCase):
    def setUp(self):
        self.engine = AYUSHClinicalExplainerEngine()
        self.sample_case = {
            "patient": {"patientId": "P100", "name": "Rahul Sharma"},
            "ayushAssessment": {
                "dashavidhaPariksha": {
                    "prakriti": {
                        "value": "pitta-kapha",
                        "confidence": 0.9,
                        "evidence": [{"questionId": "Q1", "answer": "Warm skin", "source": "questionnaire", "confidence": 0.9}]
                    },
                    "vikriti": {
                        "value": "vata-pitta",
                        "confidence": 0.85,
                        "evidence": [{"questionId": "Q2", "answer": "Dry skin & anxiety", "source": "transcript", "confidence": 0.85}]
                    },
                    "sara": {"value": "Rakta (Blood)", "confidence": 0.8},
                    "samhanana": {"value": "Madhyama (Medium)", "confidence": 0.85},
                    "pramana": {"value": "Madhyama (Medium)", "confidence": 0.9},
                    "satmya": {"value": "Madhyama (Medium)", "confidence": 0.8},
                    "sattva": {"value": "Pravara (High)", "confidence": 0.95},
                    "aharaShakti": {"value": "Madhyama (Medium)", "confidence": 0.85},
                    "vyayamaShakti": {"value": "Madhyama (Medium)", "confidence": 0.8},
                    "vaya": {"value": "Madhyama (Adult)", "confidence": 1.0}
                }
            }
        }

    def test_explain_parameter(self):
        assessment = self.sample_case["ayushAssessment"]["dashavidhaPariksha"]["prakriti"]
        exp = self.engine.explain_parameter("prakriti", assessment)
        self.assertEqual(exp["parameter"], "prakriti")
        self.assertEqual(exp["value"], "pitta-kapha")
        self.assertEqual(exp["evidenceCount"], 1)
        self.assertIn("Warm skin", exp["summary"])

    def test_excluded_parameters(self):
        for param in EXCLUDED_PARAMETERS:
            with self.assertRaises(ValueError):
                self.engine.explain_parameter(param, {})

    def test_explain_case(self):
        report = self.engine.explain_case(self.sample_case, practitioner_query="Why is Vata elevated in Vikriti?")
        self.assertEqual(report["patientId"], "P100")
        self.assertEqual(report["totalActiveParametersEvaluated"], 10)
        self.assertEqual(report["assessedParametersCount"], 10)
        self.assertIn("prakriti", report["parameterExplanations"])
        self.assertIn("vata", report["querySynthesis"].lower())
        self.assertIn("Clinical Decision Support Only", report["disclaimer"])

    def test_empty_case_fallback(self):
        empty_case = {"patient": {"patientId": "P999"}, "ayushAssessment": {"dashavidhaPariksha": {}}}
        report = self.engine.explain_case(empty_case)
        self.assertEqual(report["patientId"], "P999")
        self.assertEqual(report["assessedParametersCount"], 0)


if __name__ == "__main__":
    unittest.main()
