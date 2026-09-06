"""
Unit tests for Section 19: End-to-End Physician Summary, Prescription Attachment & Review Workflow
"""

import unittest
import json
import os
import sys
from pathlib import Path

# Ensure src directory is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conversational_intake_engine import (
    AYUSHConversationalIntakeEngine,
    MockSpeechProvider,
    IntakeState,
    SUPPORTED_LANGUAGES
)
from physician_review_engine import AYUSHPhysicianReviewEngine
from leakage_free_ml_engine import ACTIVE_PARAMETERS, EXCLUDED_PARAMETERS


class TestPhysicianReviewWorkflow(unittest.TestCase):
    def setUp(self):
        self.intake_engine = AYUSHConversationalIntakeEngine()
        self.review_engine = AYUSHPhysicianReviewEngine(intake_engine=self.intake_engine)

    def test_multilingual_intake_preservation(self):
        for lang in ["en-US", "hi-IN", "gu-IN", "mr-IN", "bn-IN"]:
            sess = self.intake_engine.create_session(language_code=lang)
            self.assertEqual(sess["language"]["code"], lang)

    def test_text_voice_tap_multi_modal_inputs(self):
        sess_id = self.intake_engine.create_session()["session_id"]
        res_text = self.intake_engine.process_input(sess_id, text_content="Severe headache", input_mode="text")
        self.assertEqual(res_text["current_step"], "CHIEF_COMPLAINT")

        sess_id2 = self.intake_engine.create_session()["session_id"]
        res_voice = self.intake_engine.process_input(sess_id2, voice_audio=b"audio", input_mode="voice")
        self.assertEqual(res_voice["current_step"], "CHIEF_COMPLAINT")

        sess_id3 = self.intake_engine.create_session()["session_id"]
        res_tap = self.intake_engine.process_input(sess_id3, tap_options=["Headache"], input_mode="tap")
        self.assertEqual(res_tap["current_step"], "CHIEF_COMPLAINT")

    def test_prescription_question_and_no_prescription_path(self):
        sess = self.intake_engine.create_session()
        s_id = sess["session_id"]
        self.intake_engine.process_input(s_id, text_content="Digestion issues")
        summary = self.review_engine.generate_summary(s_id)
        self.assertEqual(summary["status"], "AI_GENERATED_DRAFT")
        self.assertEqual(len(summary["structured_content"]["original_medical_documents"]), 0)

    def test_yes_prescription_path_and_document_extensions(self):
        for filename in ["doc.pdf", "img.jpg", "pic.jpeg", "scan.png"]:
            sess = self.intake_engine.create_session()
            s_id = sess["session_id"]
            doc_entry = self.intake_engine.upload_document(s_id, filename=filename, file_size_bytes=1024)
            self.assertEqual(doc_entry["status"], "success")

            summary = self.review_engine.generate_summary(s_id)
            docs = summary["structured_content"]["original_medical_documents"]
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0]["file_name"], filename)

    def test_executable_and_invalid_files_rejection(self):
        sess = self.intake_engine.create_session()
        s_id = sess["session_id"]
        for invalid_file in ["script.exe", "notes.txt", "payload.sh"]:
            with self.assertRaises(ValueError):
                self.intake_engine.upload_document(s_id, filename=invalid_file)

    def test_summary_generation_and_non_hallucination(self):
        sess = self.intake_engine.create_session()
        s_id = sess["session_id"]
        summary = self.review_engine.generate_summary(s_id)

        struct = summary["structured_content"]
        self.assertEqual(struct["allergy_information"], "Not provided")
        self.assertEqual(struct["family_social_history"], "Not provided")
        self.assertEqual(struct["history_of_present_illness"]["socrates_findings"]["site"], "Not provided")

    def test_physician_retrieval_edit_approval_workflow(self):
        sess = self.intake_engine.create_session()
        s_id = sess["session_id"]
        self.intake_engine.upload_document(s_id, filename="previous_rx.pdf")

        summary = self.review_engine.generate_summary(s_id)
        summ_id = summary["summary_id"]
        self.assertEqual(summary["status"], "AI_GENERATED_DRAFT")

        # Physician Edit
        updated = self.review_engine.update_summary(
            summ_id,
            physician_edits={"chief_complaint": "Amended Chief Complaint"},
            physician_notes="Clinical notes added"
        )
        self.assertEqual(updated["status"], "PHYSICIAN_EDITED")
        self.assertEqual(updated["structured_content"]["chief_complaint"], "Amended Chief Complaint")

        # Physician Approval
        approved = self.review_engine.approve_summary(summ_id, physician_id="DR_TEST")
        self.assertEqual(approved["status"], "FINALIZED_APPROVED")
        self.assertEqual(approved["physician_review"]["physician_id"], "DR_TEST")
        self.assertIn("input_sha256_hash", approved["governance_audit"])

        # Retains original attachment
        self.assertEqual(len(approved["structured_content"]["original_medical_documents"]), 1)
        self.assertEqual(approved["structured_content"]["original_medical_documents"][0]["file_name"], "previous_rx.pdf")

    def test_physician_rejection_workflow(self):
        sess = self.intake_engine.create_session()
        s_id = sess["session_id"]
        summary = self.review_engine.generate_summary(s_id)
        summ_id = summary["summary_id"]

        rejected = self.review_engine.reject_summary(summ_id, physician_id="DR_TEST", rejection_reason="Incomplete clinical context")
        self.assertEqual(rejected["status"], "REJECTED_RETURNED")
        self.assertEqual(rejected["physician_review"]["rejection_reason"], "Incomplete clinical context")

    def test_ml_prediction_integration_and_zero_leakage(self):
        sess = self.intake_engine.create_session()
        summary = self.review_engine.generate_summary(sess["session_id"])

        ml_out = summary["structured_content"]["ml_prakriti_decision_support"]
        self.assertIn(ml_out["predicted_prakriti"], ["vata", "pitta", "kapha"])
        self.assertGreaterEqual(ml_out["confidence"], 0.0)

        # Zero leakage: no ground truth labels in input vector
        q_ans = summary["structured_content"]["ayush_dashavidha_pariksha"]["parameters"]
        self.assertNotIn("primary_prakriti", q_ans)

    def test_agni_and_koshtha_exclusion(self):
        sess = self.intake_engine.create_session()
        summary = self.review_engine.generate_summary(sess["session_id"])
        ayush_section = summary["structured_content"]["ayush_dashavidha_pariksha"]

        self.assertEqual(ayush_section["active_parameters_evaluated"], 10)
        self.assertIn("agni", ayush_section["excluded_parameters"])
        self.assertIn("koshtha", ayush_section["excluded_parameters"])
        self.assertNotIn("Agni", ayush_section["parameters"])
        self.assertNotIn("Koshtha", ayush_section["parameters"])

    def test_red_flag_interruption_preservation(self):
        sess = self.intake_engine.create_session(language_code="en-US")
        res = self.intake_engine.process_input(sess["session_id"], text_content="Severe chest pain and unconsciousness")
        self.assertEqual(res["status"], IntakeState.RED_FLAG_TRIAGE.value)

        summary = self.review_engine.generate_summary(sess["session_id"])
        self.assertTrue(summary["structured_content"]["red_flag_emergency_status"]["detected"])


if __name__ == "__main__":
    unittest.main()
