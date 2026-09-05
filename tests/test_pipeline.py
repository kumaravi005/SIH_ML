import json
import unittest
import urllib.request
from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(BASE_DIR))

from ayush_processor import process_ayush_answers
from ayush_assessment import assess_all
from clinical_extractor import process_unstructured_input
from patient_case_builder import build_complete_patient_case
from ayush_explainer import generate_ayush_explainability_report
from llm_extractor import LLMClinicalExtractor


class TestAYUSHMLPipeline(unittest.TestCase):

    def setUp(self):
        self.sample_cases_file = Path(__file__).resolve().parent / "sample_patient_cases.json"
        with open(self.sample_cases_file, "r", encoding="utf-8") as f:
            self.sample_cases = json.load(f)

    def test_ayush_10_parameter_assessment(self):
        """
        Verify all 10 AYUSH parameters are correctly assessed and Agni/Koshtha are excluded.
        """
        case = self.sample_cases[0]
        patient_answers = case["questionnaire_answers"]

        validated = process_ayush_answers(patient_answers)
        evidence_map = {param: res["evidence"] for param, res in validated.items()}
        results = assess_all(evidence_map)

        expected_10 = [
            "prakriti", "vikriti", "sara", "samhanana", "pramana",
            "satmya", "sattva", "aharaShakti", "vyayamaShakti", "vaya"
        ]

        # Check all 10 present
        for param in expected_10:
            self.assertIn(param, results, f"Missing parameter {param} in assessment results")
            self.assertIn("value", results[param])
            self.assertIn("confidence", results[param])
            self.assertIn("evidence", results[param])

        # Verify Agni and Koshtha are NOT in Dashavidha assessment output
        self.assertNotIn("agni", results)
        self.assertNotIn("koshtha", results)

    def test_multi_source_patient_case_building(self):
        """
        Verify end-to-end multi-source patient case construction and clinical entity extraction.
        """
        case = self.sample_cases[0]
        patient_answers = case["questionnaire_answers"]
        unstructured_inputs = case["unstructured_inputs"]

        built_case = build_complete_patient_case(
            patient_answers=patient_answers,
            unstructured_inputs=unstructured_inputs
        )

        # 1. Verify general clinical sections populated
        hpi = built_case["historyOfPresentIllness"]
        self.assertTrue(len(hpi) > 0)
        self.assertIn("Headache", hpi[0]["complaint"])

        meds = built_case["medications"]
        self.assertTrue(len(meds) > 0)
        self.assertEqual(meds[0]["name"], "Paracetamol")

        allergies = built_case["allergies"]
        self.assertTrue(len(allergies) > 0)
        self.assertEqual(allergies[0]["substance"], "Penicillin")

        # 2. Verify Dashavidha Pariksha section contains exactly 10 active parameters
        dashavidha = built_case["ayushAssessment"]["dashavidhaPariksha"]
        expected_10 = [
            "prakriti", "vikriti", "sara", "samhanana", "pramana",
            "satmya", "sattva", "aharaShakti", "vyayamaShakti", "vaya"
        ]
        for param in expected_10:
            self.assertIn(param, dashavidha)
            self.assertIsNotNone(dashavidha[param]["value"])

        # 3. Verify Agni and Koshtha are kept null outside Dashavidha
        self.assertIn("agni", built_case["ayushAssessment"])
        self.assertIn("koshtha", built_case["ayushAssessment"])
        self.assertIsNone(built_case["ayushAssessment"]["agni"]["value"])
        self.assertIsNone(built_case["ayushAssessment"]["koshtha"]["value"])

        # 4. Verify additional metadata preservation
        self.assertIn("measurements", dashavidha["pramana"])
        self.assertEqual(dashavidha["pramana"]["measurements"], "Height: 170 cm, Weight: 65 kg")
        self.assertIn("dietary_evidence", dashavidha["satmya"])
        self.assertIn("free_text_evidence", dashavidha["sattva"])
        self.assertIn("age", dashavidha["vaya"])

    def test_source_provenance_preservation(self):
        """
        Verify that evidence retains original source ('audio_transcript', 'text', 'questionnaire').
        """
        text_input = "Patient reports headache and fatigue for 3 days."
        extracted = process_unstructured_input(text_input, source="audio_transcript", base_confidence=0.91)
        ayush_ev = extracted["ayush_evidence"]

        self.assertIn("vikriti", ayush_ev)
        for item in ayush_ev["vikriti"]:
            self.assertEqual(item["source"], "audio_transcript")
            self.assertEqual(item["confidence"], 0.91)

    def test_ayush_explainability_report(self):
        """
        Verify generation of human-readable explainability reports for doctors.
        """
        case = self.sample_cases[0]
        built_case = build_complete_patient_case(
            patient_answers=case["questionnaire_answers"],
            unstructured_inputs=case["unstructured_inputs"]
        )
        report = generate_ayush_explainability_report(built_case)

        self.assertIn("explanations", report)
        self.assertEqual(len(report["explanations"]), 10)
        self.assertIn("prakriti", report["explanations"])
        self.assertIn("summary", report["explanations"]["prakriti"])
        self.assertIn("vata", report["explanations"]["prakriti"]["summary"])

    def test_llm_extractor_fallback(self):
        """
        Verify LLM extractor gracefully falls back to rule-based engine offline.
        """
        extractor = LLMClinicalExtractor(provider="rule_based")
        res = extractor.extract("Severe headache and fatigue for 3 days", source="text")
        self.assertIn("clinical", res)
        self.assertIn("ayush_evidence", res)

    def test_python_sdk_exports(self):
        """
        Verify SDK exports from src package.
        """
        import src
        self.assertTrue(callable(src.build_complete_patient_case))
        self.assertTrue(callable(src.assess_all))
        self.assertTrue(callable(src.generate_ayush_explainability_report))

    def test_rest_api_server_endpoints(self):
        """
        Verify REST API server endpoints (GET /health, POST /process-case, POST /extract).
        """
        import threading
        import time
        import urllib.request
        from api import HealthcareMLRequestHandler
        from http.server import HTTPServer

        server = HTTPServer(("127.0.0.1", 8089), HealthcareMLRequestHandler)
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()
        time.sleep(0.1)

        try:
            # 1. GET /health
            req = urllib.request.Request("http://127.0.0.1:8089/health")
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(data["status"], "healthy")
                self.assertEqual(data["ayush_parameters"], 10)

            # 2. POST /extract
            payload = json.dumps({"text": "Patient has severe headache and fatigue for 3 days", "source": "audio_transcript"}).encode("utf-8")
            req = urllib.request.Request("http://127.0.0.1:8089/extract", data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertIn("clinical", data)
                self.assertIn("ayush_evidence", data)
        finally:
            server.shutdown()
            server.server_close()

    def test_unstructured_clinical_text_and_audio_transcript_flow(self):
        """
        Test complete end-to-end pipeline flow driven purely by raw unstructured clinical text & audio transcripts:
        raw transcript -> clinical extraction -> AYUSH evidence -> 10-parameter assessment -> patient case -> explainability.
        """
        raw_transcript = (
            "Patient Rohan Sharma is a 35 years old male. "
            "He reports severe headache and fatigue for 3 days which is getting worse. "
            "Patient has a family history of hypertension in family. "
            "Height: 170 cm, Weight: 65 kg, well-proportioned build with good appetite. "
            "Takes Paracetamol 500 mg for headache. Allergic to Penicillin. "
            "Eats home-cooked food, walks 30 minutes daily and stays calm under stress."
        )

        # 1. Build complete patient case from audio transcript
        patient_case = build_complete_patient_case(
            patient_answers=None,
            unstructured_inputs=[
                {"text": raw_transcript, "source": "audio_transcript", "confidence": 0.94}
            ]
        )

        # 2. Verify clinical sections extracted into patient case schema
        self.assertEqual(patient_case["patient"]["name"], "Rohan Sharma")
        self.assertEqual(patient_case["patient"]["age"], 35)
        self.assertEqual(patient_case["patient"]["gender"], "Male")

        complaints = patient_case["chiefComplaint"]["complaints"]
        self.assertTrue(len(complaints) > 0)
        self.assertIn("Headache", complaints[0]["description"])

        meds = patient_case["medications"]
        self.assertTrue(len(meds) > 0)
        self.assertEqual(meds[0]["name"], "Paracetamol")
        self.assertEqual(meds[0]["source"], "audio_transcript")

        allergies = patient_case["allergies"]
        self.assertTrue(len(allergies) > 0)
        self.assertEqual(allergies[0]["substance"], "Penicillin")

        # 3. Verify Dashavidha Pariksha contains all 10 active parameters
        dashavidha = patient_case["ayushAssessment"]["dashavidhaPariksha"]
        self.assertEqual(len(dashavidha), 10)
        self.assertNotIn("agni", dashavidha)
        self.assertNotIn("koshtha", dashavidha)

        # 4. Verify evidence provenance retained as 'audio_transcript'
        for param_key, param_val in dashavidha.items():
            if param_val["evidence"]:
                for ev in param_val["evidence"]:
                    self.assertIn(ev["source"], ["audio_transcript", "questionnaire", "text"])

        # 5. Verify explainability report generated cleanly
        report = generate_ayush_explainability_report(patient_case)
        self.assertIn("explanations", report)
        self.assertEqual(len(report["explanations"]), 10)

    def test_longitudinal_risk_and_api_endpoints(self):
        import threading
        import time
        from api import HealthcareMLRequestHandler
        from http.server import HTTPServer

        patient_case = build_complete_patient_case(
            patient_answers=self.sample_cases[0]["questionnaire_answers"]
        )

        from longitudinal_risk_engine import generate_longitudinal_risk_report
        risk_report = generate_longitudinal_risk_report(patient_case)

        self.assertIn("health_resilience", risk_report)
        self.assertIn("imbalance_severity", risk_report)
        self.assertIn("risk_stratification", risk_report)
        self.assertEqual(risk_report["assessment_summary"]["active_parameters_assessed"], 10)
        self.assertEqual(risk_report["assessment_summary"]["excluded_parameters"], ["agni", "koshtha"])

        # Test REST API /risk-stratification endpoint
        server = HTTPServer(("127.0.0.1", 8090), HealthcareMLRequestHandler)
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()
        time.sleep(0.1)

        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8090/risk-stratification",
                data=json.dumps({"patient_case": patient_case}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertIn("risk_stratification", data)
        finally:
            server.shutdown()
            server.server_close()

    def test_patient_similarity_api_endpoint(self):
        import threading
        import time
        from api import HealthcareMLRequestHandler
        from http.server import HTTPServer

        patient_case = build_complete_patient_case(
            patient_answers=self.sample_cases[0]["questionnaire_answers"]
        )

        from patient_similarity_engine import find_similar_patient_cases
        sim_report = find_similar_patient_cases(patient_case, self.sample_cases, top_k=2)
        self.assertIn("retrieved_similar_cases", sim_report)

        server = HTTPServer(("127.0.0.1", 8091), HealthcareMLRequestHandler)
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()
        time.sleep(0.1)

        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8091/similar-cases",
                data=json.dumps({"target_case": patient_case, "historical_cases": self.sample_cases}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertIn("retrieved_similar_cases", data)
        finally:
            server.shutdown()
            server.server_close()

    def test_data_quality_api_endpoint(self):
        import threading
        import time
        from api import HealthcareMLRequestHandler
        from http.server import HTTPServer

        patient_case = build_complete_patient_case(
            patient_answers=self.sample_cases[0]["questionnaire_answers"]
        )

        from data_quality_engine import generate_data_quality_report
        quality_report = generate_data_quality_report(patient_case)
        self.assertIn("data_integrity", quality_report)

        server = HTTPServer(("127.0.0.1", 8092), HealthcareMLRequestHandler)
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()
        time.sleep(0.1)

        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8092/validate-quality",
                data=json.dumps({"patient_case": patient_case}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertIn("data_integrity", data)
        finally:
            server.shutdown()
            server.server_close()

    def test_predict_outcome_api_endpoint(self):
        import threading
        import time
        from api import HealthcareMLRequestHandler
        from http.server import HTTPServer

        patient_case = build_complete_patient_case(
            patient_answers=self.sample_cases[0]["questionnaire_answers"]
        )

        from clinical_prediction_engine import predict_clinical_outcome
        pred_report = predict_clinical_outcome(patient_case, adherence_level="high")
        self.assertIn("predicted_dosha_stabilization_probability", pred_report)

        server = HTTPServer(("127.0.0.1", 8093), HealthcareMLRequestHandler)
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()
        time.sleep(0.1)

        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8093/predict-outcome",
                data=json.dumps({"patient_case": patient_case, "adherence_level": "high"}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertIn("predicted_dosha_stabilization_probability", data)
        finally:
            server.shutdown()
            server.server_close()

    def test_evaluate_pipeline_api_endpoint(self):
        import threading
        import time
        from api import HealthcareMLRequestHandler
        from http.server import HTTPServer

        from eval_framework import run_full_benchmark_suite
        eval_report = run_full_benchmark_suite()
        self.assertEqual(eval_report["benchmark_status"], "PASSED")

        server = HTTPServer(("127.0.0.1", 8094), HealthcareMLRequestHandler)
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()
        time.sleep(0.1)

        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8094/evaluate-pipeline",
                data=json.dumps({}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(data["benchmark_status"], "PASSED")
        finally:
            server.shutdown()
            server.server_close()

    def test_optimize_regimen_api_endpoint(self):
        import threading
        import time
        from api import HealthcareMLRequestHandler
        from http.server import HTTPServer

        patient_case = build_complete_patient_case(
            patient_answers=self.sample_cases[0]["questionnaire_answers"]
        )

        from regimen_optimizer_engine import generate_personalized_regimen_report
        reg_report = generate_personalized_regimen_report(patient_case, season="Vasanta")
        self.assertIn("daily_dinacharya_routine", reg_report)

        server = HTTPServer(("127.0.0.1", 8095), HealthcareMLRequestHandler)
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()
        time.sleep(0.1)

        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8095/optimize-regimen",
                data=json.dumps({"patient_case": patient_case, "season": "Vasanta"}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertIn("daily_dinacharya_routine", data)
        finally:
            server.shutdown()
            server.server_close()

    def test_explain_case_api_endpoint(self):
        import threading
        import time
        from api import HealthcareMLRequestHandler
        from http.server import HTTPServer

        patient_case = build_complete_patient_case(
            patient_answers=self.sample_cases[0]["questionnaire_answers"]
        )

        server = HTTPServer(("127.0.0.1", 8096), HealthcareMLRequestHandler)
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()
        time.sleep(0.1)

        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8096/explain-case",
                data=json.dumps({"patient_case": patient_case, "query": "Explain Vata imbalance"}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertIn("parameterExplanations", data)
                self.assertIn("querySynthesis", data)
        finally:
            server.shutdown()
            server.server_close()

    def test_generate_dataset_api_endpoint(self):
        import threading
        import time
        from api import HealthcareMLRequestHandler
        from http.server import HTTPServer

        server = HTTPServer(("127.0.0.1", 8097), HealthcareMLRequestHandler)
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()
        time.sleep(0.1)

        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8097/generate-dataset",
                data=json.dumps({"num_samples": 5}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(data["total_generated"], 5)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()







