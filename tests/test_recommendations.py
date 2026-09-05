import json
import unittest
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(BASE_DIR))

from patient_case_builder import build_complete_patient_case
from recommendation_engine import (
    generate_clinical_recommendation_report,
    generate_clinical_interpretation,
    generate_evidence_backed_recommendations,
    generate_safety_cautions
)


class TestClinicalRecommendationEngine(unittest.TestCase):

    def setUp(self):
        self.sample_file = BASE_DIR / "tests" / "sample_patient_cases.json"
        with open(self.sample_file, "r", encoding="utf-8") as f:
            self.sample_cases = json.load(f)

    def test_complete_patient_case_recommendations(self):
        """
        Verify recommendation engine processes a complete multi-source patient case correctly.
        """
        case = self.sample_cases[0]
        patient_case = build_complete_patient_case(
            patient_answers=case["questionnaire_answers"],
            unstructured_inputs=case["unstructured_inputs"]
        )

        report = generate_clinical_recommendation_report(patient_case)

        # Verify top-level report keys
        self.assertIn("clinical_interpretation", report)
        self.assertIn("structured_recommendations", report)
        self.assertIn("safety_and_cautions", report)
        self.assertEqual(report["active_parameters_evaluated"], 10)
        self.assertTrue(report["agni_koshtha_excluded"])

        # Verify 10-parameter interpretations
        interpretations = report["clinical_interpretation"]
        expected_10 = [
            "prakriti", "vikriti", "sara", "samhanana", "pramana",
            "satmya", "sattva", "aharaShakti", "vyayamaShakti", "vaya"
        ]
        for param in expected_10:
            self.assertIn(param, interpretations)
            self.assertEqual(interpretations[param]["status"], "assessed")

        # Verify Agni and Koshtha are absent
        self.assertNotIn("agni", interpretations)
        self.assertNotIn("koshtha", interpretations)

        # Verify structured recommendations
        recs = report["structured_recommendations"]
        self.assertTrue(len(recs) > 0)
        for r in recs:
            self.assertIn("parameter", r)
            self.assertIn("category", r)
            self.assertIn("recommendation", r)
            self.assertIn("rationale", r)
            self.assertIn("evidence_references", r)
            self.assertTrue(len(r["evidence_references"]) > 0)
            for ev in r["evidence_references"]:
                self.assertIn("questionId", ev)
                self.assertIn("source", ev)
                self.assertIn("confidence", ev)

    def test_missing_information_insufficient_evidence(self):
        """
        Verify missing information is handled cleanly as insufficient_evidence without hallucinating data.
        """
        # Create minimal patient case with no questionnaire answers
        patient_case = build_complete_patient_case(patient_answers={}, unstructured_inputs=None)

        report = generate_clinical_recommendation_report(patient_case)
        interpretations = report["clinical_interpretation"]

        # Missing parameters should return insufficient_evidence status
        for param, data in interpretations.items():
            self.assertEqual(data["status"], "insufficient_evidence")
            self.assertIsNone(data["value"])
            self.assertIsNone(data["confidence"])

        # No fake recommendations should be generated
        recs = report["structured_recommendations"]
        self.assertEqual(len(recs), 0)

    def test_evidence_provenance_and_confidence_in_recommendations(self):
        """
        Verify recommendation evidence references retain original source (e.g. 'audio_transcript') and confidence.
        """
        raw_text = "Patient Rohan Sharma is 35 years old male. Severe headache and fatigue for 3 days. Takes Paracetamol 500 mg."
        patient_case = build_complete_patient_case(
            patient_answers=None,
            unstructured_inputs=[{"text": raw_text, "source": "audio_transcript", "confidence": 0.93}]
        )

        recs = generate_evidence_backed_recommendations(patient_case)
        self.assertTrue(len(recs) > 0)
        for r in recs:
            for ev in r["evidence_references"]:
                self.assertIn(ev["source"], ["audio_transcript", "questionnaire", "text"])
                self.assertTrue(0.0 <= ev["confidence"] <= 1.0)

    def test_safety_caution_and_non_diagnostic_disclaimer(self):
        """
        Verify non-diagnostic legal disclaimers and safety alerts are generated.
        """
        raw_text = "Patient reports headache getting worse. Allergic to Penicillin."
        patient_case = build_complete_patient_case(
            patient_answers=None,
            unstructured_inputs=[{"text": raw_text, "source": "text", "confidence": 0.90}]
        )

        cautions = generate_safety_cautions(patient_case)
        self.assertIn("DISCLAIMER", cautions["disclaimer"])
        self.assertTrue(len(cautions["alerts"]) > 0)
        self.assertTrue(any("worsening" in alert for alert in cautions["alerts"]))
        self.assertTrue(any("Penicillin" in alert for alert in cautions["alerts"]))


if __name__ == "__main__":
    unittest.main()
