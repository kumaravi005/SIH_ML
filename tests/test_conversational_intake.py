"""
Tests for AYUSH Patient Multilingual Conversational Intake Engine (Section 18)
"""

import unittest
import os
import sys
from pathlib import Path

# Ensure src directory is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conversational_intake_engine import (
    AYUSHConversationalIntakeEngine,
    MockSpeechProvider,
    LanguageLocalizationProvider,
    IntakeState,
    SUPPORTED_LANGUAGES
)
from leakage_free_ml_engine import ACTIVE_PARAMETERS, EXCLUDED_PARAMETERS


class TestConversationalIntakeEngine(unittest.TestCase):
    def setUp(self):
        self.engine = AYUSHConversationalIntakeEngine()

    def test_supported_languages_and_localization(self):
        self.assertIn("en-US", SUPPORTED_LANGUAGES)
        self.assertIn("hi-IN", SUPPORTED_LANGUAGES)
        self.assertIn("gu-IN", SUPPORTED_LANGUAGES)
        self.assertIn("mr-IN", SUPPORTED_LANGUAGES)
        self.assertIn("bn-IN", SUPPORTED_LANGUAGES)

        provider = LanguageLocalizationProvider()
        for lang in SUPPORTED_LANGUAGES:
            prompt = provider.get_prompt("welcome", lang)
            self.assertTrue(len(prompt) > 0)
            self.assertIn("welcome", provider.prompts[lang])

    def test_mock_speech_provider(self):
        speech_provider = MockSpeechProvider()
        stt_result = speech_provider.transcribe_audio(b"audio_dummy_bytes", language_code="hi-IN")
        self.assertEqual(stt_result["language_code"], "hi-IN")
        self.assertTrue(len(stt_result["transcript"]) > 0)

        tts_result = speech_provider.synthesize_speech("नमस्ते", language_code="hi-IN")
        self.assertEqual(tts_result["language_code"], "hi-IN")
        self.assertTrue(len(tts_result["audio_url"]) > 0)

    def test_session_lifecycle(self):
        session = self.engine.create_session(language_code="hi-IN")
        self.assertIsNotNone(session["session_id"])
        self.assertEqual(session["language"]["code"], "hi-IN")
        self.assertEqual(session["stage"], IntakeState.WELCOME.value)

        state = self.engine.get_session_state(session["session_id"])
        self.assertEqual(state["session_id"], session["session_id"])
        self.assertEqual(state["current_step"], "LANG_SELECT")

    def test_language_switching(self):
        session = self.engine.create_session(language_code="en-US")
        self.assertEqual(session["language"]["code"], "en-US")

        updated = self.engine.set_session_language(session["session_id"], "gu-IN")
        self.assertEqual(updated["language_code"], "gu-IN")

    def test_multi_modal_equivalence(self):
        # Test text, voice, tap produce uniform representation
        res_text = self.engine.process_input(
            session_id=self.engine.create_session()["session_id"],
            text_content="Headache for 2 days",
            input_mode="text"
        )
        self.assertEqual(res_text["current_step"], IntakeState.CHIEF_COMPLAINT.value)

        res_voice = self.engine.process_input(
            session_id=self.engine.create_session()["session_id"],
            voice_audio=b"audio_bytes",
            input_mode="voice"
        )
        self.assertEqual(res_voice["current_step"], IntakeState.CHIEF_COMPLAINT.value)

        res_tap = self.engine.process_input(
            session_id=self.engine.create_session()["session_id"],
            tap_options=["Headache"],
            input_mode="tap"
        )
        self.assertEqual(res_tap["current_step"], IntakeState.CHIEF_COMPLAINT.value)

    def test_red_flag_emergency_triage_interruption(self):
        session = self.engine.create_session(language_code="en-US")
        res = self.engine.process_input(
            session_id=session["session_id"],
            text_content="I am having severe chest pain and breathing difficulty",
            input_mode="text"
        )
        self.assertEqual(res["status"], IntakeState.RED_FLAG_TRIAGE.value)
        self.assertTrue(res["red_flag_triggered"])
        self.assertIn("urgent medical attention", res["next_prompt"])

    def test_red_flag_multilingual(self):
        # Test Hindi red flag
        session_hi = self.engine.create_session(language_code="hi-IN")
        res_hi = self.engine.process_input(
            session_id=session_hi["session_id"],
            text_content="मुझे सीने में दर्द हो रहा है",
            input_mode="text"
        )
        self.assertEqual(res_hi["status"], IntakeState.RED_FLAG_TRIAGE.value)

    def test_socrates_flow_and_completion(self):
        session = self.engine.create_session(language_code="en-US")
        s_id = session["session_id"]

        # 1. Chief Complaint input
        r1 = self.engine.process_input(s_id, text_content="Abdominal pain and bloating")
        self.assertEqual(r1["current_step"], "CHIEF_COMPLAINT")

        # 2. SOCRATES site input
        r2 = self.engine.process_input(s_id, text_content="Abdomen upper right")
        self.assertEqual(r2["current_step"], "SOCRATES_PROBING")

        # Complete intake
        summary = self.engine.complete_session(s_id)
        self.assertEqual(summary["status"], IntakeState.COMPLETED.value)
        self.assertIn("ml_prediction", summary)
        self.assertIn("governance_audit", summary)

    def test_document_upload_validation(self):
        session = self.engine.create_session()
        s_id = session["session_id"]

        # Valid document
        val_doc = self.engine.upload_document(s_id, filename="report.pdf", content_type="application/pdf", file_size_bytes=1024)
        self.assertEqual(val_doc["status"], "success")

        # Invalid document extension
        with self.assertRaises(ValueError):
            self.engine.upload_document(s_id, filename="malware.exe", content_type="application/x-msdownload", file_size_bytes=2048)

    def test_zero_target_leakage_and_10_parameter_isolation(self):
        session = self.engine.create_session()
        summary = self.engine.complete_session(session["session_id"])
        patient_case = summary["patient_case"]

        # Ensure ground truth label is absent or excluded from inference
        q_ans = patient_case.get("questionnaire_answers", {})
        self.assertNotIn("primary_prakriti", q_ans)
        self.assertNotIn("prakriti", q_ans)

        # Verify active parameters = 10 and excluded parameters = agni, koshtha
        self.assertEqual(len(ACTIVE_PARAMETERS), 10)
        self.assertIn("agni", EXCLUDED_PARAMETERS)
        self.assertIn("koshtha", EXCLUDED_PARAMETERS)
        self.assertNotIn("Agni", ACTIVE_PARAMETERS)
        self.assertNotIn("Koshtha", ACTIVE_PARAMETERS)


if __name__ == "__main__":
    unittest.main()
