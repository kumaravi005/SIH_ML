import json
from pathlib import Path

RULES_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "ayush_recommendation_rules.json"
)


def load_recommendation_rules():
    """
    Load recommendation rules from data/ayush_recommendation_rules.json.
    """
    if RULES_FILE.exists():
        with open(RULES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("rules", {})
    return {}


def generate_clinical_interpretation(patient_case):
    """
    Generate structured clinical interpretation narratives based on the 10 AYUSH parameters
    and clinical patient case data.
    """
    dashavidha = (
        patient_case.get("ayushAssessment", {})
        .get("dashavidhaPariksha", {})
    )

    interpretations = {}

    active_10 = [
        "prakriti", "vikriti", "sara", "samhanana", "pramana",
        "satmya", "sattva", "aharaShakti", "vyayamaShakti", "vaya"
    ]

    for param in active_10:
        param_data = dashavidha.get(param, {})
        val = param_data.get("value")
        conf = param_data.get("confidence")
        evidence = param_data.get("evidence", [])

        if val is None or conf is None:
            interpretations[param] = {
                "status": "insufficient_evidence",
                "summary": f"Insufficient data available to formulate clinical interpretation for {param}.",
                "value": None,
                "confidence": None
            }
            continue

        if param == "prakriti":
            summary = f"Primary constitutional trait evaluated as '{val.capitalize()}' based on constitutional evidence."
        elif param == "vikriti":
            curr = val.get("current_changes", []) if isinstance(val, dict) else []
            prog = val.get("progression") if isinstance(val, dict) else None
            summary = f"Active symptom alterations detected: {', '.join(curr) if curr else 'Mild discomfort'}. Progression noted as '{prog or 'stable'}'."
        elif param == "sara":
            summary = f"Overall tissue essence and constitutional strength (Dhatu Sara) evaluated as '{val}'."
        elif param == "samhanana":
            summary = f"Musculoskeletal build and structural compactness (Samhanana) evaluated as '{val}'."
        elif param == "pramana":
            meas = param_data.get("measurements")
            summary = f"Body proportions and anthropometric balance (Pramana) evaluated as '{val}'." + (f" Measurements: {meas}" if meas else "")
        elif param == "satmya":
            summary = f"Dietary habituation and food tolerance (Satmya) evaluated as '{val}'."
        elif param == "sattva":
            summary = f"Mental strength, emotional resilience, and coping capacity (Sattva) evaluated as '{val}'."
        elif param == "aharaShakti":
            summary = f"Appetite and digestive ingestion capacity (Ahara Shakti) evaluated as '{val}'."
        elif param == "vyayamaShakti":
            summary = f"Physical exertion capacity and workout tolerance (Vyayama Shakti) evaluated as '{val}'."
        elif param == "vaya":
            age = param_data.get("age")
            summary = f"Biological age stage evaluated as '{val}'" + (f" (Age {int(age)})" if age else "") + "."
        else:
            summary = f"{param.capitalize()} evaluated as '{val}'."

        interpretations[param] = {
            "status": "assessed",
            "summary": summary,
            "value": val,
            "confidence": conf
        }

    return interpretations


def generate_evidence_backed_recommendations(patient_case):
    """
    Generate structured, evidence-backed lifestyle, dietary, and physical activity recommendations
    based on the 10 active AYUSH parameters.
    """
    rules = load_recommendation_rules()
    dashavidha = (
        patient_case.get("ayushAssessment", {})
        .get("dashavidhaPariksha", {})
    )

    recommendations = []
    active_10 = [
        "prakriti", "vikriti", "sara", "samhanana", "pramana",
        "satmya", "sattva", "aharaShakti", "vyayamaShakti", "vaya"
    ]

    for param in active_10:
        param_data = dashavidha.get(param, {})
        val = param_data.get("value")
        evidence = param_data.get("evidence", [])

        if not val or not evidence or val == "insufficient" or param not in rules:
            continue

        param_rules = rules[param]
        if isinstance(val, str):
            lookup_key = val
        elif isinstance(val, dict):
            has_content = any(v for v in val.values() if v)
            lookup_key = "present" if has_content else None
        else:
            lookup_key = None

        if lookup_key and lookup_key in param_rules:
            matched_rules = param_rules[lookup_key]
            for r in matched_rules:
                # Format evidence references
                evidence_refs = [
                    {
                        "questionId": e.get("questionId"),
                        "answer": e.get("answer"),
                        "source": e.get("source", "unknown"),
                        "confidence": e.get("confidence", 1.0)
                    }
                    for e in evidence
                ]

                recommendations.append({
                    "parameter": param,
                    "category": r.get("category", "general_care"),
                    "recommendation": r.get("recommendation"),
                    "rationale": r.get("rationale"),
                    "evidence_references": evidence_refs
                })

    return recommendations


def generate_safety_cautions(patient_case):
    """
    Generate clinical safety cautions and non-diagnostic legal disclaimers.
    """
    hpi = patient_case.get("historyOfPresentIllness", [])
    allergies = patient_case.get("allergies", [])
    dashavidha = (
        patient_case.get("ayushAssessment", {})
        .get("dashavidhaPariksha", {})
    )

    alerts = []

    # Check for worsening progression
    vikriti_data = dashavidha.get("vikriti", {}).get("value", {})
    if isinstance(vikriti_data, dict):
        prog = vikriti_data.get("progression", "")
        if prog and "worse" in str(prog).lower():
            alerts.append("Patient symptoms are reported as worsening; recommend clinical review by a registered healthcare professional.")

    # Check for known allergies
    if allergies and any(a.get("substance") for a in allergies):
        known_allergies = [a["substance"] for a in allergies if a.get("substance")]
        alerts.append(f"Patient has reported allergies: {', '.join(known_allergies)}. Avoid allergen exposure in dietary or lifestyle interventions.")

    disclaimer = (
        "DISCLAIMER: This clinical interpretation and recommendation output provides supportive AYUSH lifestyle, "
        "dietary, and habituation guidance based on patient-reported parameters. It does NOT constitute a formal medical "
        "diagnosis, clinical prescription, or emergency medical advice. Consult a qualified medical practitioner for acute medical concerns."
    )

    return {
        "disclaimer": disclaimer,
        "alerts": alerts
    }


def generate_clinical_recommendation_report(patient_case):
    """
    Main entry point for generating the complete Clinical Interpretation & Recommendation Report.
    Consumes patient_case containing 10-parameter AYUSH assessment.
    """
    interpretations = generate_clinical_interpretation(patient_case)
    recommendations = generate_evidence_backed_recommendations(patient_case)
    safety = generate_safety_cautions(patient_case)

    patient_info = patient_case.get("patient", {})

    return {
        "patientId": patient_info.get("patientId"),
        "patientName": patient_info.get("name"),
        "active_parameters_evaluated": 10,
        "agni_koshtha_excluded": True,
        "clinical_interpretation": interpretations,
        "structured_recommendations": recommendations,
        "safety_and_cautions": safety
    }


if __name__ == "__main__":
    output_file = Path(__file__).resolve().parent.parent / "output" / "patient_case_output.json"
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            case_data = json.load(f)

        report = generate_clinical_recommendation_report(case_data)
        print("CLINICAL INTERPRETATION & RECOMMENDATION REPORT:")
        print(json.dumps(report, indent=2))
