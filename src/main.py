import argparse
import json
from pathlib import Path

from patient_case_builder import build_complete_patient_case, save_patient_case, OUTPUT_FILE
from ayush_explainer import generate_ayush_explainability_report
from recommendation_engine import generate_clinical_recommendation_report
from longitudinal_risk_engine import generate_longitudinal_risk_report


EXPLAINABILITY_OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent
    / "output"
    / "ayush_explainability_report.json"
)

RECOMMENDATION_OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent
    / "output"
    / "clinical_recommendation_output.json"
)

LONGITUDINAL_RISK_OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent
    / "output"
    / "longitudinal_risk_report.json"
)



def run_pipeline(input_case_path=None, text_input=None, source="audio_transcript", generate_explanation=True):
    """
    Execute end-to-end ML pipeline.
    """
    patient_answers = None
    unstructured_inputs = []

    if input_case_path and Path(input_case_path).exists():
        with open(input_case_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list) and len(data) > 0:
            case = data[0]
            patient_answers = case.get("questionnaire_answers")
            unstructured_inputs = case.get("unstructured_inputs", [])
        elif isinstance(data, dict):
            patient_answers = data.get("questionnaire_answers")
            unstructured_inputs = data.get("unstructured_inputs", [])

    if text_input:
        unstructured_inputs.append({
            "text": text_input,
            "source": source,
            "confidence": 0.92
        })

    # Build patient case
    patient_case = build_complete_patient_case(
        patient_answers=patient_answers,
        unstructured_inputs=unstructured_inputs if unstructured_inputs else None
    )

    # Save output
    save_patient_case(patient_case)

    # Generate explainability report
    report = None
    if generate_explanation:
        report = generate_ayush_explainability_report(patient_case)
        with open(EXPLAINABILITY_OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    # Generate clinical recommendations report
    rec_report = generate_clinical_recommendation_report(patient_case)
    with open(RECOMMENDATION_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(rec_report, f, indent=2, ensure_ascii=False)

    # Generate longitudinal progress & risk report
    risk_report = generate_longitudinal_risk_report(patient_case)
    with open(LONGITUDINAL_RISK_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(risk_report, f, indent=2, ensure_ascii=False)

    return patient_case, report, rec_report, risk_report


def main():
    parser = argparse.ArgumentParser(description="SIH_ML Healthcare Pipeline & AYUSH Assessment CLI")
    parser.add_argument("--input", type=str, help="Path to input patient case JSON")
    parser.add_argument("--text", type=str, help="Raw clinical text or transcript input")
    parser.add_argument("--source", type=str, default="audio_transcript", choices=["text", "audio_transcript", "questionnaire", "prescription"])
    parser.add_argument("--no-explain", action="store_true", help="Disable generating explainability report")

    args = parser.parse_args()

    default_sample = Path(__file__).resolve().parent.parent / "tests" / "sample_patient_cases.json"
    input_file = args.input or (str(default_sample) if default_sample.exists() else None)

    patient_case, report, rec_report, risk_report = run_pipeline(
        input_case_path=input_file,
        text_input=args.text,
        source=args.source,
        generate_explanation=not args.no_explain
    )

    print("============================================================")
    print("AYUSH 10-PARAMETER ASSESSMENT & CLINICAL CASE GENERATED")
    print("============================================================")
    print(f"Patient Case Output Saved: {OUTPUT_FILE}")
    if report:
        print(f"Explainability Report Saved: {EXPLAINABILITY_OUTPUT_FILE}")
    print(f"Clinical Recommendation Output Saved: {RECOMMENDATION_OUTPUT_FILE}")
    print(f"Longitudinal Risk Report Saved: {LONGITUDINAL_RISK_OUTPUT_FILE}")



if __name__ == "__main__":
    main()
