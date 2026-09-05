import json
from pathlib import Path


def generate_parameter_explanation(parameter, result):
    """
    Generate structured, explainable clinical narrative for a single AYUSH parameter result.
    """
    val = result.get("value")
    conf = result.get("confidence")
    evidence = result.get("evidence", [])

    if val is None:
        return {
            "parameter": parameter,
            "summary": f"{parameter.capitalize()} could not be determined due to lack of specific evidence.",
            "value": None,
            "confidence": None,
            "evidence_count": 0,
            "traceability": []
        }

    traceability = []
    for item in evidence:
        traceability.append({
            "questionId": item.get("questionId"),
            "answer": item.get("answer"),
            "source": item.get("source", "unknown"),
            "confidence": item.get("confidence", 1.0)
        })

    # Human-readable rationale text
    evidence_desc = ", ".join([f"'{t['answer']}' (via {t['source']}, confidence {t['confidence']})" for t in traceability])
    summary = (
        f"{parameter.capitalize()} assessed as '{val}' with overall confidence {conf}. "
        f"Derived from {len(traceability)} evidence item(s): {evidence_desc}."
    )

    return {
        "parameter": parameter,
        "summary": summary,
        "value": val,
        "confidence": conf,
        "evidence_count": len(traceability),
        "traceability": traceability
    }


def generate_ayush_explainability_report(patient_case):
    """
    Generate a full explainability report for all 10 AYUSH Dashavidha parameters in a patient case.
    """
    dashavidha = (
        patient_case.get("ayushAssessment", {})
        .get("dashavidhaPariksha", {})
    )

    report = {
        "patientId": patient_case.get("patient", {}).get("patientId"),
        "patientName": patient_case.get("patient", {}).get("name"),
        "total_parameters_assessed": len(dashavidha),
        "explanations": {}
    }

    for param, res in dashavidha.items():
        report["explanations"][param] = generate_parameter_explanation(param, res)

    return report


if __name__ == "__main__":
    output_file = Path(__file__).resolve().parent.parent / "output" / "patient_case_output.json"
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            case_data = json.load(f)

        report = generate_ayush_explainability_report(case_data)
        print("AYUSH EXPLAINABILITY REPORT:")
        print(json.dumps(report, indent=2))
