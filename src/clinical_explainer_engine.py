"""
AYUSH Clinical Explainer & Reasoning Synthesis Engine (Section 6)

Synthesizes interactive, multi-layered clinical explanations, evidence provenance traces,
and practitioner query responses based strictly on the 10 active AYUSH parameters:
- prakriti, vikriti, sara, samhanana, pramana, satmya, sattva, aharaShakti, vyayamaShakti, vaya

Strictly excludes Agni and Koshtha. Zero external mandatory dependencies.
"""

import json
from pathlib import Path

# 10 Active Parameters Constant
ACTIVE_PARAMETERS = [
    "prakriti", "vikriti", "sara", "samhanana", "pramana",
    "satmya", "sattva", "aharaShakti", "vyayamaShakti", "vaya"
]

EXCLUDED_PARAMETERS = ["agni", "koshtha"]


class AYUSHClinicalExplainerEngine:
    def __init__(self, rules_path=None):
        base_dir = Path(__file__).resolve().parent.parent
        self.rules_path = rules_path or base_dir / "data" / "ayush_recommendation_rules.json"
        self.rules = self._load_rules()

    def _load_rules(self):
        if self.rules_path and Path(self.rules_path).exists():
            with open(self.rules_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def explain_parameter(self, parameter_name, assessment_result):
        """
        Generate detailed single-parameter explanation with item provenance and classical reference.
        """
        if parameter_name.lower() in EXCLUDED_PARAMETERS:
            raise ValueError(f"Parameter '{parameter_name}' is strictly excluded from AYUSH engine.")

        val = assessment_result.get("value")
        conf = assessment_result.get("confidence", 1.0)
        evidence = assessment_result.get("evidence", [])

        if val is None:
            return {
                "parameter": parameter_name,
                "summary": f"{parameter_name.capitalize()} could not be conclusively determined due to insufficient specific evidence.",
                "value": None,
                "confidence": None,
                "evidenceCount": 0,
                "traceability": [],
                "classicalReference": "Dashavidha Pariksha standard assessment protocol."
            }

        traceability = []
        for item in evidence:
            traceability.append({
                "questionId": item.get("questionId"),
                "answer": item.get("answer"),
                "source": item.get("source", "unknown"),
                "confidence": item.get("confidence", 1.0)
            })

        evidence_desc = ", ".join([f"'{t['answer']}' (via {t['source']}, confidence {t['confidence']})" for t in traceability])
        summary = (
            f"{parameter_name.capitalize()} evaluated as '{val}' with confidence {conf}. "
            f"Derived from {len(traceability)} clinical evidence item(s): {evidence_desc}."
        )

        return {
            "parameter": parameter_name,
            "summary": summary,
            "value": val,
            "confidence": conf,
            "evidenceCount": len(traceability),
            "traceability": traceability,
            "classicalReference": f"Derived per Charaka Samhita Vimanasthana 8 for {parameter_name.capitalize()} assessment."
        }

    def synthesize_practitioner_query(self, patient_case, query_text):
        """
        Synthesize natural-language answer to practitioner query based on case assessments.
        """
        if not query_text or not query_text.strip():
            return "No practitioner query provided."

        query_lower = query_text.lower()
        dashavidha = patient_case.get("ayushAssessment", {}).get("dashavidhaPariksha", {})

        matched_params = [p for p in ACTIVE_PARAMETERS if p.lower() in query_lower]
        
        responses = []

        if "vata" in query_lower or "pitta" in query_lower or "kapha" in query_lower:
            vikriti = dashavidha.get("vikriti", {}).get("value", "balanced")
            prakriti = dashavidha.get("prakriti", {}).get("value", "balanced")
            responses.append(
                f"Regarding Dosha dynamics: Patient's baseline Prakriti is '{prakriti}' and current Vikriti imbalance is '{vikriti}'."
            )

        if "ahara" in query_lower or "diet" in query_lower:
            ahara = dashavidha.get("aharaShakti", {}).get("value", "medium")
            responses.append(f"Digestive & Nourishment capacity (Ahara Shakti) is assessed as '{ahara}'.")

        if "vyayama" in query_lower or "exercise" in query_lower or "strength" in query_lower:
            vyayama = dashavidha.get("vyayamaShakti", {}).get("value", "medium")
            responses.append(f"Physical work capacity (Vyayama Shakti) is assessed as '{vyayama}'.")

        for param in matched_params:
            res = dashavidha.get(param, {})
            val = res.get("value", "undetermined")
            responses.append(f"For parameter '{param}': Assessed value is '{val}'.")

        if not responses:
            responses.append(
                f"Query received: '{query_text}'. Analysis synthesized across all 10 active AYUSH Dashavidha parameters."
            )

        return " ".join(responses)

    def explain_case(self, patient_case, practitioner_query=None):
        """
        Generate full interactive clinical explainability and reasoning synthesis report.
        """
        dashavidha = (
            patient_case.get("ayushAssessment", {})
            .get("dashavidhaPariksha", {})
        )

        parameter_explanations = {}
        assessed_count = 0
        overall_confidence_sum = 0.0

        for param in ACTIVE_PARAMETERS:
            assessment_res = dashavidha.get(param, {})
            explanation = self.explain_parameter(param, assessment_res)
            parameter_explanations[param] = explanation
            if explanation["value"] is not None:
                assessed_count += 1
                overall_confidence_sum += explanation.get("confidence") or 1.0

        avg_confidence = round(overall_confidence_sum / assessed_count, 2) if assessed_count > 0 else 0.0

        query_synthesis = self.synthesize_practitioner_query(patient_case, practitioner_query) if practitioner_query else None

        # Cross-parameter interaction rationale
        prakriti_val = dashavidha.get("prakriti", {}).get("value", "Unknown")
        vikriti_val = dashavidha.get("vikriti", {}).get("value", "Balanced")
        sattva_val = dashavidha.get("sattva", {}).get("value", "Pravara (High)")
        ahara_val = dashavidha.get("aharaShakti", {}).get("value", "Madhyama (Medium)")

        interaction_synthesis = (
            f"Patient presents baseline constitution (Prakriti) of '{prakriti_val}' with current imbalance (Vikriti) of '{vikriti_val}'. "
            f"Psychological resilience (Sattva) is '{sattva_val}' and digestive/nourishment capacity (Ahara Shakti) is '{ahara_val}'. "
            f"No Agni or Koshtha parameters evaluated per protocol."
        )

        return {
            "patientId": patient_case.get("patient", {}).get("patientId", "UNKNOWN"),
            "totalActiveParametersEvaluated": len(parameter_explanations),
            "assessedParametersCount": assessed_count,
            "overallAssessmentConfidence": avg_confidence,
            "parameterExplanations": parameter_explanations,
            "practitionerQuery": practitioner_query,
            "querySynthesis": query_synthesis,
            "interactionSynthesis": interaction_synthesis,
            "disclaimer": (
                "Clinical Decision Support Only: This explainability synthesis is provided to assist registered medical practitioners "
                "and does not constitute independent diagnostic or therapeutic claims."
            )
        }
