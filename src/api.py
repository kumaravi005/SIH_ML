import json
import sys
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

# Add src/ directory to path if executed standalone
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from patient_case_builder import build_complete_patient_case
from clinical_extractor import process_unstructured_input
from ayush_explainer import generate_ayush_explainability_report
from recommendation_engine import generate_clinical_recommendation_report
from longitudinal_risk_engine import generate_longitudinal_risk_report, LongitudinalRiskEngine
from patient_similarity_engine import find_similar_patient_cases, PatientSimilarityEngine
from data_quality_engine import generate_data_quality_report, DataQualityEngine
from clinical_prediction_engine import predict_clinical_outcome, ClinicalPredictionEngine
from eval_framework import run_full_benchmark_suite, MLEvaluationFramework
from regimen_optimizer_engine import generate_personalized_regimen_report, RegimenOptimizerEngine
from clinical_explainer_engine import AYUSHClinicalExplainerEngine
from dataset_generator import AYUSHDatasetGeneratorEngine
from ml_baseline_model import AYUSHBaselineMLEngine


class HealthcareMLRequestHandler(BaseHTTPRequestHandler):
    """
    Zero-dependency HTTP REST API handler for SIH_ML.
    Integrates clinical extraction, 10-parameter AYUSH assessment, recommendation engine,
    longitudinal risk tracking, patient similarity matching, data quality validation, outcome prediction, model evaluation framework, and regimen optimization.
    """

    def _set_headers(self, status_code=200, content_type="application/json"):
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        if self.path == "/health":
            self._set_headers(200)
            self.wfile.write(json.dumps({
                "status": "healthy",
                "module": "SIH_ML",
                "ayush_parameters": 10,
                "agni_koshtha_excluded": True
            }).encode("utf-8"))
        elif self.path == "/":
            self._set_headers(200)
            self.wfile.write(json.dumps({
                "name": "SIH_ML REST API Server",
                "endpoints": [
                    "GET /health",
                    "POST /process-case",
                    "POST /extract",
                    "POST /recommendations",
                    "POST /risk-stratification",
                    "POST /longitudinal-track",
                    "POST /similar-cases",
                    "POST /validate-quality",
                    "POST /predict-outcome",
                    "POST /evaluate-pipeline",
                    "POST /optimize-regimen",
                    "POST /explain-case",
                    "POST /generate-dataset",
                    "POST /train-baseline"
                ]
            }).encode("utf-8"))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body_str = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"

        try:
            payload = json.loads(body_str) if body_str else {}
        except json.JSONDecodeError:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": "Invalid JSON body"}).encode("utf-8"))
            return

        if self.path == "/process-case":
            patient_answers = payload.get("questionnaire_answers")
            unstructured_inputs = payload.get("unstructured_inputs")
            generate_explain = payload.get("generate_explainability", True)

            patient_case = build_complete_patient_case(
                patient_answers=patient_answers,
                unstructured_inputs=unstructured_inputs
            )

            report = None
            if generate_explain:
                report = generate_ayush_explainability_report(patient_case)

            rec_report = generate_clinical_recommendation_report(patient_case)

            self._set_headers(200)
            self.wfile.write(json.dumps({
                "patient_case": patient_case,
                "explainability_report": report,
                "recommendation_report": rec_report
            }, ensure_ascii=False).encode("utf-8"))

        elif self.path == "/extract":
            text = payload.get("text", "")
            source = payload.get("source", "text")
            confidence = payload.get("confidence", 0.90)

            extracted = process_unstructured_input(
                raw_input=text,
                source=source,
                base_confidence=confidence
            )

            self._set_headers(200)
            self.wfile.write(json.dumps(extracted, ensure_ascii=False).encode("utf-8"))

        elif self.path == "/recommendations":
            patient_answers = payload.get("questionnaire_answers")
            unstructured_inputs = payload.get("unstructured_inputs")

            patient_case = payload.get("patient_case")
            if not patient_case:
                patient_case = build_complete_patient_case(
                    patient_answers=patient_answers,
                    unstructured_inputs=unstructured_inputs
                )

            rec_report = generate_clinical_recommendation_report(patient_case)

            self._set_headers(200)
            self.wfile.write(json.dumps(rec_report, ensure_ascii=False).encode("utf-8"))

        elif self.path == "/risk-stratification":
            patient_case = payload.get("patient_case")
            if not patient_case:
                patient_answers = payload.get("questionnaire_answers")
                unstructured_inputs = payload.get("unstructured_inputs")
                patient_case = build_complete_patient_case(
                    patient_answers=patient_answers,
                    unstructured_inputs=unstructured_inputs
                )

            risk_report = generate_longitudinal_risk_report(patient_case)

            self._set_headers(200)
            self.wfile.write(json.dumps(risk_report, ensure_ascii=False).encode("utf-8"))

        elif self.path == "/longitudinal-track":
            current_case = payload.get("current_case")
            historical_cases = payload.get("historical_cases", [])

            if not current_case:
                patient_answers = payload.get("questionnaire_answers")
                unstructured_inputs = payload.get("unstructured_inputs")
                current_case = build_complete_patient_case(
                    patient_answers=patient_answers,
                    unstructured_inputs=unstructured_inputs
                )

            track_report = generate_longitudinal_risk_report(current_case, historical_cases)

            self._set_headers(200)
            self.wfile.write(json.dumps(track_report, ensure_ascii=False).encode("utf-8"))

        elif self.path == "/similar-cases":
            target_case = payload.get("target_case")
            historical_cases = payload.get("historical_cases", [])
            top_k = payload.get("top_k", 3)

            if not target_case:
                patient_answers = payload.get("questionnaire_answers")
                unstructured_inputs = payload.get("unstructured_inputs")
                target_case = build_complete_patient_case(
                    patient_answers=patient_answers,
                    unstructured_inputs=unstructured_inputs
                )

            sim_report = find_similar_patient_cases(target_case, historical_cases, top_k=top_k)

            self._set_headers(200)
            self.wfile.write(json.dumps(sim_report, ensure_ascii=False).encode("utf-8"))

        elif self.path == "/validate-quality":
            patient_case = payload.get("patient_case")
            if not patient_case:
                patient_answers = payload.get("questionnaire_answers")
                unstructured_inputs = payload.get("unstructured_inputs")
                patient_case = build_complete_patient_case(
                    patient_answers=patient_answers,
                    unstructured_inputs=unstructured_inputs
                )

            quality_report = generate_data_quality_report(patient_case)

            self._set_headers(200)
            self.wfile.write(json.dumps(quality_report, ensure_ascii=False).encode("utf-8"))

        elif self.path == "/predict-outcome":
            patient_case = payload.get("patient_case")
            adherence_level = payload.get("adherence_level", "moderate")
            if not patient_case:
                patient_answers = payload.get("questionnaire_answers")
                unstructured_inputs = payload.get("unstructured_inputs")
                patient_case = build_complete_patient_case(
                    patient_answers=patient_answers,
                    unstructured_inputs=unstructured_inputs
                )

            pred_report = predict_clinical_outcome(patient_case, adherence_level=adherence_level)

            self._set_headers(200)
            self.wfile.write(json.dumps(pred_report, ensure_ascii=False).encode("utf-8"))

        elif self.path == "/evaluate-pipeline":
            benchmark_samples = payload.get("benchmark_samples")
            eval_report = run_full_benchmark_suite(benchmark_samples=benchmark_samples)

            self._set_headers(200)
            self.wfile.write(json.dumps(eval_report, ensure_ascii=False).encode("utf-8"))

        elif self.path == "/optimize-regimen":
            patient_case = payload.get("patient_case")
            season = payload.get("season", "Vasanta")
            if not patient_case:
                patient_answers = payload.get("questionnaire_answers")
                unstructured_inputs = payload.get("unstructured_inputs")
                patient_case = build_complete_patient_case(
                    patient_answers=patient_answers,
                    unstructured_inputs=unstructured_inputs
                )

            reg_report = generate_personalized_regimen_report(patient_case, season=season)

            self._set_headers(200)
            self.wfile.write(json.dumps(reg_report, ensure_ascii=False).encode("utf-8"))

        elif self.path == "/explain-case":
            patient_case = payload.get("patient_case")
            query = payload.get("query")
            if not patient_case:
                patient_answers = payload.get("questionnaire_answers")
                unstructured_inputs = payload.get("unstructured_inputs")
                patient_case = build_complete_patient_case(
                    patient_answers=patient_answers,
                    unstructured_inputs=unstructured_inputs
                )

            explainer = AYUSHClinicalExplainerEngine()
            explain_report = explainer.explain_case(patient_case, practitioner_query=query)

            self._set_headers(200)
            self.wfile.write(json.dumps(explain_report, ensure_ascii=False).encode("utf-8"))

        elif self.path == "/generate-dataset":
            num_samples = payload.get("num_samples", 50)
            generator = AYUSHDatasetGeneratorEngine()
            stats = generator.generate_dataset(num_samples=num_samples)

            self._set_headers(200)
            self.wfile.write(json.dumps(stats, ensure_ascii=False).encode("utf-8"))

        elif self.path == "/train-baseline":
            engine = AYUSHBaselineMLEngine()
            eval_report = engine.train_and_evaluate()

            self._set_headers(200)
            self.wfile.write(json.dumps(eval_report, ensure_ascii=False).encode("utf-8"))

        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))








def run_server(port=8000, host="0.0.0.0"):
    server_address = (host, port)
    httpd = HTTPServer(server_address, HealthcareMLRequestHandler)
    print(f"SIH_ML REST API Server running on http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping SIH_ML REST API Server...")
        httpd.server_close()


if __name__ == "__main__":
    run_server(port=8000)
