"""
AYUSH Patient Multilingual Conversational Intake Engine (Section 18)

Provides:
1. First-class multilingual intake in English (en-US), Hindi (hi-IN), Gujarati (gu-IN), Marathi (mr-IN), and Bengali (bn-IN).
2. Multi-modal input handling (Text, Voice, Tap) mapping to a unified CommonAnswerRepresentation.
3. Speech-to-Text and Text-to-Speech provider abstractions with deterministic MockSpeechProvider for testing.
4. Adaptive clinical questioning state machine with SOCRATES symptom investigation (Site, Onset, Character, Radiation, Associated, Timing, Exacerbating, Severity).
5. Safety-first Red-Flag emergency triage detection and immediate flow interruption.
6. Integration with existing 10 active AYUSH parameters (strictly excluding Agni and Koshtha).
7. Medical document upload handoff (PDF, JPG, JPEG, PNG validation & metadata preservation).
8. Handoff of structured intake to existing KNN k=9 ML prediction & governance audit engines without target leakage.
"""

import json
import math
import time
import os
import re
from enum import Enum
from pathlib import Path

from leakage_free_ml_engine import (
    ACTIVE_PARAMETERS,
    EXCLUDED_PARAMETERS
)
from model_serving_drift_engine import AYUSHModelServingDriftEngine
from governance_safety_engine import AYUSHGovernanceSafetyEngine
from patient_case_builder import build_complete_patient_case


class IntakeState(str, Enum):
    WELCOME = "LANG_SELECT"
    LANG_SELECT = "LANG_SELECT"
    CHIEF_COMPLAINT = "CHIEF_COMPLAINT"
    SOCRATES_SITE = "SOCRATES_PROBING"
    SOCRATES_PROBING = "SOCRATES_PROBING"
    MEDICAL_HISTORY = "MEDICAL_HISTORY"
    AYUSH_ASSESSMENT = "AYUSH_ASSESSMENT"
    DOCUMENT_UPLOAD = "DOCUMENT_UPLOAD"
    COMPLETED = "COMPLETED"
    RED_FLAG_TRIAGE = "RED_FLAG_TRIAGE"


SUPPORTED_LANGUAGES = {
    "en-US": {"name": "English", "native": "English"},
    "hi-IN": {"name": "Hindi", "native": "हिन्दी"},
    "gu-IN": {"name": "Gujarati", "native": "ગુજરાતી"},
    "mr-IN": {"name": "Marathi", "native": "मराठी"},
    "bn-IN": {"name": "Bengali", "native": "বাংলা"}
}

ALLOWED_DOCUMENT_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}

# Red-flag urgent keywords across supported languages
RED_FLAG_KEYWORDS = {
    "en-US": ["chest pain", "breathing difficulty", "shortness of breath", "unconscious", "uncontrolled bleeding", "severe allergic reaction", "sudden paralysis", "stroke"],
    "hi-IN": ["सीने में दर्द", "छाती में दर्द", "सांस लेने में तकलीफ", "बेहोश", "गंभीर खून बहना", "लकवा"],
    "gu-IN": ["છાતીમાં દુખાવો", "શ્વાસ લેવામાં તકલીફ", "બેભાન", "અચાનક લકવો"],
    "mr-IN": ["छातीत दुखणे", "श्वास घेण्यास त्रास", "बेशुद्ध", "पक्षाघात"],
    "bn-IN": ["বুকে ব্যথা", "শ্বাসকষ্ট", "অজ্ঞান", "পক্ষাঘাত"]
}


class LanguageLocalizationProvider:
    """
    Handles clinical UI string localization across supported languages
    while preserving medical terminology, dosages, and numbers unaltered.
    """
    def __init__(self):
        self.prompts = {
            "en-US": {
                "welcome": "Welcome to AYUSH Health Assessment. Please select your preferred language.",
                "chief_complaint": "What is your primary health concern or symptom today?",
                "socrates_site": "Where exactly is the symptom or pain located?",
                "socrates_onset": "When did this symptom start?",
                "socrates_character": "How would you describe the sensation (e.g., sharp, dull, burning, aching)?",
                "socrates_radiation": "Does the discomfort spread or radiate to any other body part?",
                "socrates_associated": "Are there any other associated symptoms (e.g., nausea, dizziness)?",
                "socrates_timing": "Is the symptom constant or does it come and go?",
                "socrates_exacerbating": "What makes the symptom better or worse?",
                "socrates_severity": "On a scale of 0 to 10, how severe is your discomfort?",
                "medical_history": "Do you have any ongoing medical conditions or current medications?",
                "ayush_intro": "Now we will assess your constitution based on 10 AYUSH parameters.",
                "document_prompt": "Do you have any previous prescription, lab report, or discharge summary?",
                "red_flag_alert": "This information may indicate a situation requiring urgent medical attention. Please seek immediate medical care / follow your healthcare facility's emergency process.",
                "completion": "Intake completed successfully. Generating clinical decision support summary."
            },
            "hi-IN": {
                "welcome": "आयुष स्वास्थ्य मूल्यांकन में आपका स्वागत है। कृपया अपनी पसंदीदा भाषा चुनें।",
                "chief_complaint": "आज आपकी मुख्य स्वास्थ्य समस्या या लक्षण क्या है?",
                "socrates_site": "दर्द या लक्षण शरीर के किस हिस्से में है?",
                "socrates_onset": "यह लक्षण कब शुरू हुआ था?",
                "socrates_character": "यह दर्द या अनुभव कैसा महसूस होता है (जैसे तेज, हल्का, जलन, चुभन)?",
                "socrates_radiation": "क्या यह दर्द शरीर के किसी अन्य हिस्से में फैलता है?",
                "socrates_associated": "क्या इसके साथ कोई अन्य लक्षण भी हैं (जैसे जी मिचलाना, चक्कर)?",
                "socrates_timing": "क्या यह लक्षण लगातार बना रहता है या आता-जाता है?",
                "socrates_exacerbating": "किस चीज़ से यह लक्षण कम या ज्यादा होता है?",
                "socrates_severity": "0 से 10 के पैमाने पर, आपकी तकलीफ कितनी गंभीर है?",
                "medical_history": "क्या आपको कोई मौजूदा बीमारी है या आप कोई दवा ले रहे हैं?",
                "ayush_intro": "अब हम 10 आयुष मापदंडों के आधार पर आपकी प्रकृति का मूल्यांकन करेंगे।",
                "document_prompt": "क्या आपके पास कोई पिछली prescription, lab report या discharge summary है?",
                "red_flag_alert": "यह जानकारी किसी गंभीर आपातकालीन स्थिति का संकेत हो सकती है। कृपया तुरंत निकटतम आपातकालीन चिकित्सा सेवा से संपर्क करें।",
                "completion": "मूल्यांकन सफलतापूर्वक पूरा हुआ। नैदानिक सहायता रिपोर्ट तैयार की जा रही है।"
            },
            "gu-IN": {
                "welcome": "આયુષ સ્વાસ્થ્ય મૂલ્યાંકનમાં આપનું સ્વાગત છે. કૃપા કરીને તમારી ભાષા પસંદ કરો.",
                "chief_complaint": "આજે તમારી મુખ્ય સ્વાસ્થ્ય સમસ્યા શું છે?",
                "socrates_site": "દુખાવો અથવા લક્ષણ શરીરના કયા ભાગમાં છે?",
                "socrates_onset": "આ લક્ષણ ક્યારે શરૂ થયું?",
                "socrates_character": "આ દુખાવો કેવો અનુભવાય છે?",
                "socrates_radiation": "શું આ દુખાવો શરીરના અન્ય ભાગમાં ફેલાય છે?",
                "socrates_associated": "શું સાથે અન્ય કોઈ લક્ષણો છે?",
                "socrates_timing": "શું આ લક્ષણ સતત રહે છે કે આવે-જાય છે?",
                "socrates_exacerbating": "શાનાથી દુખાવો વધે કે ઘટે છે?",
                "socrates_severity": "0 થી 10 ના માપદંડ પર તમારી તકલીફ કેટલી ગંભીર છે?",
                "medical_history": "શું તમને કોઈ જૂની બીમારી છે અથવા દવાઓ ચાલે છે?",
                "ayush_intro": "હવે આપણે 10 આયુષ પરિમાણોના આધારે તમારું મૂલ્યાંકન કરીશું.",
                "document_prompt": "શું તમારી પાસે કોઈ જૂનું પ્રિસ્ક્રિપ્શન અથવા લેબ રિપોર્ટ છે?",
                "red_flag_alert": "આ માહિતી ગંભીર તબીબી પરિસ્થિતિ સૂચવી શકે છે. કૃપા કરીને તાત્કાલિક કટોકટી તબીબી સંભાળ મેળવો.",
                "completion": "મૂલ્યાંકન સફળતાપૂર્વક પૂર્ણ થયું."
            },
            "mr-IN": {
                "welcome": "आयुष आरोग्य मूल्यांकनात आपले स्वागत आहे. कृपया तुमची भाषा निवडा.",
                "chief_complaint": "आज तुमची मुख्य आरोग्य समस्या काय आहे?",
                "socrates_site": "त्रास किंवा वेदना शरीराच्या कोणत्या भागात आहे?",
                "socrates_onset": "हे लक्षण कधी सुरू झाले?",
                "socrates_character": "ही वेदना कशी वाटते?",
                "socrates_radiation": "ही वेदना शरीराच्या इतर भागात पसरते का?",
                "socrates_associated": "सोबत इतर काही लक्षणे आहेत का?",
                "socrates_timing": "हे लक्षण सतत असते की येते-जाते?",
                "socrates_exacerbating": "कशाने त्रास वाढतो किंवा कमी होतो?",
                "socrates_severity": "0 ते 10 च्या स्केलवर त्रास किती तीव्र आहे?",
                "medical_history": "तुम्हाला काही आजार किंवा औषधे सुरू आहेत का?",
                "ayush_intro": "आता आपण 10 आयुष निकषांवर आधारित मूल्यांकन करूया.",
                "document_prompt": "तुमच्याकडे पूर्वीचे प्रिस्क्रिप्शन किंवा लॅब रिपोर्ट आहे का?",
                "red_flag_alert": "ही माहिती तातडीच्या वैद्यकीय मदतीची गरज दर्शवू शकते. कृपया त्वरित जवळच्या रुग्णालयाशी संपर्क साधा.",
                "completion": "मूल्यांकन यशस्वीपणे पूर्ण झाले."
            },
            "bn-IN": {
                "welcome": "আয়ুষ স্বাস্থ্য মূল্যায়নে স্বাগতম। অনুগ্রহ করে আপনার ভাষা নির্বাচন করুন।",
                "chief_complaint": "আজ আপনার প্রধান স্বাস্থ্য সমস্যা কী?",
                "socrates_site": "ব্যথা বা সমস্যাটি শরীরের কোথায় অবস্থিত?",
                "socrates_onset": "এই উপসর্গটি কখন শুরু হয়েছিল?",
                "socrates_character": "এই অনুভূতিটি কেমন?",
                "socrates_radiation": "ব্যথা কি অন্য কোথাও ছড়িয়ে পড়ে?",
                "socrates_associated": "অন্য কোনো উপসর্গ আছে কি?",
                "socrates_timing": "উপসর্গটি কি ক্রমাগত থাকে নাকি আসে-যায়?",
                "socrates_exacerbating": "কিসে সমস্যা বাড়ে বা কমে?",
                "socrates_severity": "০ থেকে ১০ স্কেলে সমস্যাটি কতটা তীব্র?",
                "medical_history": "আপনার কি কোনো শারীরিক সমস্যা বা ওষুধ চলছে?",
                "ayush_intro": "এখন আমরা ১০টি আয়ুষ পরামিতির ভিত্তিতে আপনার প্রকৃতি মূল্যায়ন করব।",
                "document_prompt": "আপনার কি কোনো আগের প্রেসক্রিপশন বা ল্যাব রিপোর্ট আছে?",
                "red_flag_alert": "এই তথ্যটি জরুরি চিকিৎসার প্রয়োজনীয়তা নির্দেশ করতে পারে। অনুগ্রহ করে অবিলম্বে জরুরি চিকিৎসা সহায়তা নিন।",
                "completion": "মূল্যায়ন সফলভাবে সম্পন্ন হয়েছে।"
            }
        }

    def get_prompt(self, key, lang_code="en-US"):
        lang_dict = self.prompts.get(lang_code, self.prompts["en-US"])
        return lang_dict.get(key, self.prompts["en-US"].get(key, ""))


class SpeechToTextProvider:
    """Interface for Speech-to-Text provider."""
    def transcribe_audio(self, audio_reference, language_code="en-US"):
        raise NotImplementedError


class TextToSpeechProvider:
    """Interface for Text-to-Speech provider."""
    def synthesize_speech(self, text, language_code="en-US"):
        raise NotImplementedError


class MockSpeechProvider(SpeechToTextProvider, TextToSpeechProvider):
    """
    Deterministic local mock speech provider for testing and offline CLI operation.
    """
    def transcribe_audio(self, audio_reference, language_code="en-US"):
        # If audio_reference is already text string (e.g. from CLI or test)
        if isinstance(audio_reference, str):
            transcript = audio_reference
        else:
            transcript = "Sample spoken clinical complaint"

        return {
            "transcript": transcript,
            "language_code": language_code,
            "confidence": 0.95,
            "provider": "MockSpeechProvider",
            "audio_reference": str(audio_reference)
        }

    def synthesize_speech(self, text, language_code="en-US"):
        return {
            "audio_url": f"mock_audio://tts/{hash(text)}.mp3",
            "text": text,
            "language_code": language_code,
            "provider": "MockSpeechProvider"
        }


class CommonAnswerRepresentation:
    """
    Standardized internal answer representation regardless of input mode (Text, Voice, Tap).
    """
    def __init__(self, question_id, input_mode, language_code, original_response, normalized_value, confidence=1.0):
        self.question_id = question_id
        self.input_mode = input_mode  # "text", "voice", "tap"
        self.language_code = language_code
        self.original_response = original_response
        self.normalized_value = normalized_value
        self.confidence = confidence

    def to_dict(self):
        return {
            "question_id": self.question_id,
            "input_mode": self.input_mode,
            "language_code": self.language_code,
            "original_response": self.original_response,
            "normalized_value": self.normalized_value,
            "confidence": self.confidence
        }


class AYUSHConversationalIntakeEngine:
    def __init__(self, speech_provider=None, model_path=None):
        base_dir = Path(__file__).resolve().parent.parent
        self.model_path = model_path or base_dir / "models" / "ayush_prakriti_knn_model.json"
        
        self.localization = LanguageLocalizationProvider()
        self.speech_provider = speech_provider or MockSpeechProvider()
        self.serving_engine = AYUSHModelServingDriftEngine(model_path=self.model_path)
        self.governance_engine = AYUSHGovernanceSafetyEngine(model_path=self.model_path)

        self.sessions = {}

    def create_session(self, patient_id=None, language_code="en-US", input_mode="text"):
        """
        Creates a new structured patient intake session.
        """
        if language_code not in SUPPORTED_LANGUAGES:
            language_code = "en-US"

        session_id = f"SESS_{int(time.time() * 1000)}_{len(self.sessions) + 1}"
        session_state = {
            "session_id": session_id,
            "patient_id": patient_id or f"PATIENT_{int(time.time())}",
            "created_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "language": {
                "code": language_code,
                "name": SUPPORTED_LANGUAGES[language_code]["name"]
            },
            "input_mode": input_mode,  # "text", "voice", "tap"
            "stage": "LANG_SELECT",
            "socrates_index": 0,
            "conversation_history": [],
            "common_answers": {},
            "clinical_history": {
                "chief_complaint": None,
                "socrates": {},
                "medical_history": None
            },
            "red_flags": {
                "detected": False,
                "items": []
            },
            "ayush_answers": {
                "prakriti": [],
                "vikriti": [],
                "sara": [],
                "samhanana": [],
                "pramana": [],
                "satmya": [],
                "sattva": [],
                "aharaShakti": [],
                "vyayamaShakti": [],
                "vaya": []
            },
            "documents": [],
            "status": "IN_PROGRESS"
        }

        self.sessions[session_id] = session_state
        return session_state

    def set_language(self, session_id, language_code):
        """
        Updates session language preference.
        """
        if session_id not in self.sessions:
            raise KeyError(f"Session {session_id} not found.")

        if language_code in SUPPORTED_LANGUAGES:
            self.sessions[session_id]["language"] = {
                "code": language_code,
                "name": SUPPORTED_LANGUAGES[language_code]["name"]
            }
        return self.sessions[session_id]

    def check_red_flags(self, text, lang_code="en-US"):
        """
        Scans input text for emergency red-flag signals.
        """
        if not text:
            return False, []

        keywords = RED_FLAG_KEYWORDS.get(lang_code, RED_FLAG_KEYWORDS["en-US"])
        text_lower = text.lower()

        detected = []
        for kw in keywords:
            if kw.lower() in text_lower:
                detected.append(kw)

        return (len(detected) > 0), detected

    def normalize_input_to_common_answer(self, question_id, raw_response, input_mode, lang_code):
        """
        Maps Text, Voice transcript, or Tap choice to a unified CommonAnswerRepresentation.
        """
        norm_val = raw_response

        # Parse numeric severity if scale question
        if "severity" in question_id:
            numbers = re.findall(r'\d+', str(raw_response))
            if numbers:
                val = int(numbers[0])
                norm_val = min(10, max(0, val))
            else:
                norm_val = 5

        # Tap option index or string
        elif input_mode == "tap":
            norm_val = str(raw_response).strip()

        return CommonAnswerRepresentation(
            question_id=question_id,
            input_mode=input_mode,
            language_code=lang_code,
            original_response=str(raw_response),
            normalized_value=norm_val,
            confidence=0.98 if input_mode == "tap" else 0.90
        )

    def process_message(self, session_id, raw_response, input_mode=None, question_id=None):
        """
        Advances the intake state machine with text/tap/voice input.
        """
        if session_id not in self.sessions:
            raise KeyError(f"Session {session_id} not found.")

        session = self.sessions[session_id]
        if session["status"] == "RED_FLAG_TRIAGE":
            lang_code = session["language"]["code"]
            alert_msg = self.localization.get_prompt("red_flag_alert", lang_code)
            return {
                "session_id": session_id,
                "status": "RED_FLAG_TRIAGE",
                "system_message": alert_msg,
                "red_flags": session["red_flags"]
            }

        lang_code = session["language"]["code"]
        mode = input_mode or session["input_mode"]

        # 1. Voice transcription if voice mode
        if mode == "voice":
            trans_res = self.speech_provider.transcribe_audio(raw_response, language_code=lang_code)
            patient_text = trans_res["transcript"]
        else:
            patient_text = str(raw_response)

        # Record conversation line
        session["conversation_history"].append({
            "speaker": "patient",
            "input_mode": mode,
            "text": patient_text,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        })

        # 2. Safety Red-Flag Detection Check
        has_red_flag, red_items = self.check_red_flags(patient_text, lang_code=lang_code)
        if has_red_flag:
            session["red_flags"]["detected"] = True
            session["red_flags"]["items"].extend(red_items)
            session["status"] = "RED_FLAG_TRIAGE"

            alert_msg = self.localization.get_prompt("red_flag_alert", lang_code)
            session["conversation_history"].append({
                "speaker": "system",
                "text": alert_msg,
                "type": "red_flag_emergency_alert",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            })

            return {
                "session_id": session_id,
                "status": "RED_FLAG_TRIAGE",
                "system_message": alert_msg,
                "red_flags": session["red_flags"]
            }

        # 3. State Machine Question Progression
        stage = session["stage"]
        socrates_steps = [
            "socrates_site", "socrates_onset", "socrates_character",
            "socrates_radiation", "socrates_associated", "socrates_timing",
            "socrates_exacerbating", "socrates_severity"
        ]

        if stage == "LANG_SELECT":
            session["stage"] = "CHIEF_COMPLAINT"
            next_prompt = self.localization.get_prompt("chief_complaint", lang_code)
            options = ["Headache", "Fever", "Digestive Issue", "Joint Pain", "Skin Concern", "Other"]

        elif stage == "CHIEF_COMPLAINT":
            q_id = "chief_complaint"
            common_ans = self.normalize_input_to_common_answer(q_id, patient_text, mode, lang_code)
            session["common_answers"][q_id] = common_ans.to_dict()
            session["clinical_history"]["chief_complaint"] = common_ans.normalized_value

            session["stage"] = "SOCRATES_PROBING"
            session["socrates_index"] = 0
            curr_soc_key = socrates_steps[0]
            next_prompt = self.localization.get_prompt(curr_soc_key, lang_code)
            options = ["Head / Forehead", "Abdomen", "Chest", "Back", "Limbs / Joints", "Skin"]

        elif stage == "SOCRATES_PROBING":
            idx = session["socrates_index"]
            curr_soc_key = socrates_steps[idx]
            common_ans = self.normalize_input_to_common_answer(curr_soc_key, patient_text, mode, lang_code)
            session["common_answers"][curr_soc_key] = common_ans.to_dict()
            session["clinical_history"]["socrates"][curr_soc_key] = common_ans.normalized_value

            idx += 1
            session["socrates_index"] = idx

            if idx < len(socrates_steps):
                next_key = socrates_steps[idx]
                next_prompt = self.localization.get_prompt(next_key, lang_code)
                options = ["Mild (1-3)", "Moderate (4-6)", "Severe (7-10)"] if next_key == "socrates_severity" else ["Constant", "Intermittent", "Occasional"]
            else:
                session["stage"] = "MEDICAL_HISTORY"
                next_prompt = self.localization.get_prompt("medical_history", lang_code)
                options = ["No major illness", "Hypertension", "Diabetes", "Asthma", "Taking regular medication"]

        elif stage == "MEDICAL_HISTORY":
            q_id = "medical_history"
            common_ans = self.normalize_input_to_common_answer(q_id, patient_text, mode, lang_code)
            session["common_answers"][q_id] = common_ans.to_dict()
            session["clinical_history"]["medical_history"] = common_ans.normalized_value

            session["stage"] = "AYUSH_ASSESSMENT"
            next_prompt = self.localization.get_prompt("ayush_intro", lang_code)
            options = ["Dry / Rough skin", "Warm / Sensitive skin", "Smooth / Oily skin"]

        elif stage == "AYUSH_ASSESSMENT":
            # Record sample AYUSH answers for the 10 active parameters
            q_id = "prakriti_q1"
            common_ans = self.normalize_input_to_common_answer(q_id, patient_text, mode, lang_code)
            session["common_answers"][q_id] = common_ans.to_dict()
            
            # Map into active parameter questionnaire answers
            for p in ACTIVE_PARAMETERS:
                session["ayush_answers"][p] = [
                    {
                        "questionId": f"{p}_q1",
                        "answer": patient_text if p == "prakriti" else "Moderate / Normal",
                        "source": mode,
                        "confidence": common_ans.confidence
                    }
                ]

            session["stage"] = "DOCUMENT_UPLOAD"
            next_prompt = self.localization.get_prompt("document_prompt", lang_code)
            options = ["Yes (Upload prescription / report)", "No (Skip)"]

        elif stage == "DOCUMENT_UPLOAD":
            q_id = "document_decision"
            common_ans = self.normalize_input_to_common_answer(q_id, patient_text, mode, lang_code)
            session["common_answers"][q_id] = common_ans.to_dict()

            session["stage"] = "COMPLETED"
            session["status"] = "COMPLETED"
            next_prompt = self.localization.get_prompt("completion", lang_code)
            options = []

        else:
            next_prompt = self.localization.get_prompt("completion", lang_code)
            options = []

        session["conversation_history"].append({
            "speaker": "system",
            "text": next_prompt,
            "options": options,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        })

        return {
            "session_id": session_id,
            "stage": session["stage"],
            "status": session["status"],
            "system_message": next_prompt,
            "tap_options": options
        }

    def attach_document(self, session_id, file_path, document_type="prescription"):
        """
        Validates file type (PDF, JPG, JPEG, PNG) and associates uploaded prescription/report with session.
        """
        if session_id not in self.sessions:
            raise KeyError(f"Session {session_id} not found.")

        path = Path(file_path)
        ext = path.suffix.lower()

        if ext not in ALLOWED_DOCUMENT_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{ext}'. Allowed formats: PDF, JPG, JPEG, PNG.")

        doc_entry = {
            "document_id": f"DOC_{int(time.time() * 1000)}",
            "file_name": path.name,
            "file_path": str(path),
            "document_type": document_type,
            "extension": ext,
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "upload_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

        self.sessions[session_id]["documents"].append(doc_entry)
        return doc_entry

    def complete_intake(self, session_id):
        """
        Finalizes structured intake and hands off raw clinical information to existing ML & governance pipeline.
        """
        if session_id not in self.sessions:
            raise KeyError(f"Session {session_id} not found.")

        session = self.sessions[session_id]
        
        # Build patient case using existing pipeline builder
        complaint_text = str(session["clinical_history"].get("chief_complaint") or "General health assessment")
        unstructured = [
            {
                "text": f"Chief complaint: {complaint_text}. Medical history: {session['clinical_history'].get('medical_history')}",
                "source": "conversational_intake",
                "confidence": 0.95
            }
        ]

        patient_case = build_complete_patient_case(
            patient_answers=session["ayush_answers"],
            unstructured_inputs=unstructured
        )
        patient_case["patient"]["patientId"] = session["patient_id"]
        patient_case["questionnaire_answers"] = session["ayush_answers"]
        patient_case["unstructured_inputs"] = unstructured

        # Run KNN k=9 ML Prediction via existing serving engine
        pred_result = self.serving_engine.predict_with_explanation(patient_case)

        # Log prediction to immutable governance audit trail
        audit_entry = self.governance_engine.log_prediction_audit(patient_case, pred_result)

        session["status"] = "COMPLETED"
        session["stage"] = "COMPLETED"

        return {
            "session_id": session_id,
            "status": "COMPLETED",
            "patient_id": session["patient_id"],
            "language": session["language"],
            "clinical_summary": {
                "chief_complaint": session["clinical_history"]["chief_complaint"],
                "socrates_findings": session["clinical_history"]["socrates"],
                "medical_history": session["clinical_history"]["medical_history"],
                "uploaded_documents_count": len(session["documents"])
            },
            "ayush_ml_prediction": pred_result,
            "governance_audit": {
                "audit_id": audit_entry["audit_id"],
                "input_sha256_hash": audit_entry["input_sha256_hash"],
                "human_review_status": audit_entry["human_review_status"]
            },
            "cds_disclaimer": "Clinical Decision Support (CDS) System. Non-diagnostic. Must be validated clinically."
        }

    def get_session_state(self, session_id):
        if session_id not in self.sessions:
            raise KeyError(f"Session {session_id} not found.")
        sess = self.sessions[session_id]
        return {
            "session_id": sess["session_id"],
            "patient_id": sess["patient_id"],
            "created_timestamp": sess["created_timestamp"],
            "language_code": sess["language"]["code"],
            "current_step": sess["stage"],
            "status": sess["status"],
            "red_flags": sess["red_flags"],
            "clinical_history": sess["clinical_history"],
            "documents_count": len(sess["documents"])
        }

    def set_session_language(self, session_id, language_code):
        sess = self.set_language(session_id, language_code)
        return {
            "session_id": session_id,
            "language_code": sess["language"]["code"],
            "language_name": sess["language"]["name"]
        }

    def process_input(self, session_id, text_content=None, voice_audio=None, tap_options=None, input_mode=None):
        if input_mode == "voice" or (voice_audio is not None and not text_content and not tap_options):
            mode = "voice"
            raw = voice_audio or text_content or ""
        elif input_mode == "tap" or (tap_options is not None and not text_content):
            mode = "tap"
            raw = tap_options[0] if isinstance(tap_options, list) and tap_options else (tap_options or "")
        else:
            mode = "text"
            raw = text_content or ""

        res = self.process_message(session_id=session_id, raw_response=raw, input_mode=mode)
        red_flag_triggered = (res.get("status") == "RED_FLAG_TRIAGE")
        return {
            "session_id": session_id,
            "status": res["status"],
            "current_step": res.get("stage", res.get("status")),
            "next_prompt": res["system_message"],
            "tap_options": res.get("tap_options", []),
            "red_flag_triggered": red_flag_triggered,
            "red_flags": res.get("red_flags", {})
        }

    def upload_document(self, session_id, filename, content_type=None, file_size_bytes=0, file_bytes=None):
        if session_id not in self.sessions:
            raise KeyError(f"Session {session_id} not found.")
        path = Path(filename)
        ext = path.suffix.lower()
        if ext not in ALLOWED_DOCUMENT_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{ext}'. Allowed formats: PDF, JPG, JPEG, PNG.")

        doc_entry = {
            "document_id": f"DOC_{int(time.time() * 1000)}",
            "file_name": path.name,
            "document_type": "prescription" if "presc" in filename.lower() else "lab_report",
            "extension": ext,
            "size_bytes": file_size_bytes,
            "content_type": content_type or f"image/{ext[1:]}",
            "upload_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        self.sessions[session_id]["documents"].append(doc_entry)
        return {
            "status": "success",
            "document": doc_entry
        }

    def complete_session(self, session_id):
        res = self.complete_intake(session_id)
        sess = self.sessions[session_id]
        return {
            "session_id": session_id,
            "status": res["status"],
            "language_code": res["language"]["code"],
            "clinical_summary": res["clinical_summary"],
            "ml_prediction": res["ayush_ml_prediction"],
            "governance_audit": res["governance_audit"],
            "patient_case": build_complete_patient_case(patient_answers=sess["ayush_answers"])
        }
