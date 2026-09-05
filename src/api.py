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


class HealthcareMLRequestHandler(BaseHTTPRequestHandler):
    """
    Zero-dependency HTTP REST API handler for SIH_ML.
    Integrates clinical extraction, 10-parameter AYUSH assessment, recommendation engine,
    longitudinal risk tracking, and patient similarity matching.
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
                    "POST /similar-cases"
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
