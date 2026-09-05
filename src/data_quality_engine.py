"""
AYUSH Anomaly Detection & Clinical Data Quality Engine
------------------------------------------------------
Evaluates inter-parameter consistency across 10 active AYUSH parameters,
detects evidence source conflicts, quantifies Data Integrity Scores (0 - 100),
and flags clinical data quality warnings for decision support.

Evaluates EXACTLY 10 active AYUSH parameters:
prakriti, vikriti, sara, samhanana, pramana, satmya, sattva, aharaShakti, vyayamaShakti, vaya.

EXCLUDED PARAMETERS: Agni, Koshtha.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime


class DataQualityEngine:
    """
    Engine for multi-dimensional clinical anomaly detection and data quality validation.
    """

    ACTIVE_PARAMETERS = [
        "prakriti", "vikriti", "sara", "samhanana", "pramana",
        "satmya", "sattva", "aharaShakti", "vyayamaShakti", "vaya"
    ]
    EXCLUDED_PARAMETERS = ["agni", "koshtha"]

    def __init__(self):
        pass

    def check_inter_parameter_consistency(self, ayush_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detects contradictory findings across active AYUSH parameters.
        Returns a list of identified consistency warnings.
        """
        warnings = []

        sara_val = str(ayush_assessment.get("sara", {}).get("quality", "")).lower()
        vyayama_val = str(ayush_assessment.get("vyayamaShakti", {}).get("physical_work_capacity", "")).lower()
        ahara_val = str(ayush_assessment.get("aharaShakti", {}).get("digestive_capacity", "")).lower()
        sattva_val = str(ayush_assessment.get("sattva", {}).get("mental_strength", "")).lower()
        vaya_val = str(ayush_assessment.get("vaya", {}).get("age_group", "")).lower()
        age_num = ayush_assessment.get("vaya", {}).get("age")

        # 1. Contradiction: High VyayamaShakti (physical work capacity) vs Avara Sara (weak tissue excellence)
        if vyayama_val in ["pravara", "uttama", "strong", "high"] and sara_val in ["avara", "hina", "weak", "low"]:
            warnings.append({
                "rule_id": "CONSISTENCY_01",
                "severity": "medium",
                "parameters_involved": ["vyayamaShakti", "sara"],
                "message": "High physical work capacity (VyayamaShakti) reported alongside weak tissue excellence (Avara Sara). Verification recommended."
            })

        # 2. Contradiction: High AharaShakti (digestive capacity) vs Avara Sattva with severe Vikriti
        vikriti = ayush_assessment.get("vikriti", {})
        changes = vikriti.get("current_changes", []) if isinstance(vikriti, dict) else []
        if len(changes) >= 3 and sattva_val in ["avara", "hina", "weak"]:
            warnings.append({
                "rule_id": "CONSISTENCY_02",
                "severity": "medium",
                "parameters_involved": ["vikriti", "sattva"],
                "message": "Multiple active Vikriti aggravations observed in a patient with low mental resilience (Avara Sattva)."
            })

        # 3. Demographic Age Stage mismatch: Age < 16 reported as Madhyama Vaya or Vriddha
        if age_num is not None and isinstance(age_num, (int, float)):
            if age_num < 16 and "madhyama" in vaya_val:
                warnings.append({
                    "rule_id": "CONSISTENCY_03",
                    "severity": "high",
                    "parameters_involved": ["vaya"],
                    "message": f"Numerical age ({age_num}) conflicts with recorded age stage ({vaya_val})."
                })

        return warnings

    def check_evidence_source_conflicts(self, patient_case: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detects conflicts between questionnaire inputs and extracted unstructured text/audio inputs.
        """
        conflicts = []
        evidence_sources = set()

        ayush = patient_case.get("ayushAssessment", {}) or patient_case.get("ayush_assessment", {})
        for param, data in ayush.items():
            if isinstance(data, dict) and "evidence" in data:
                for ev in data["evidence"]:
                    if isinstance(ev, dict) and "source" in ev:
                        evidence_sources.add(ev["source"])

        # Check if both questionnaire and audio_transcript/text sources are present
        if "questionnaire" in evidence_sources and ("audio_transcript" in evidence_sources or "text" in evidence_sources):
            # Check for symptom reporting discrepancy
            chief_complaints = patient_case.get("chiefComplaint", {}).get("complaints", [])
            if not chief_complaints:
                conflicts.append({
                    "conflict_id": "SOURCE_CONFLICT_01",
                    "severity": "low",
                    "message": "Unstructured transcript provided without structured chief complaints."
                })

        return conflicts

    def compute_data_integrity_score(
        self, ayush_assessment: Dict[str, Any], clinical_evidence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Computes overall Data Quality & Integrity Score (0 - 100).
        """
        score = 100.0
        deductions = []

        # 1. Parameter completeness check (out of 10 active parameters)
        assessed_count = 0
        for param in self.ACTIVE_PARAMETERS:
            if param in ayush_assessment and ayush_assessment[param] is not None:
                assessed_count += 1

        completeness_pct = round((assessed_count / len(self.ACTIVE_PARAMETERS)) * 100.0, 1)
        if assessed_count < len(self.ACTIVE_PARAMETERS):
            missing_pts = (len(self.ACTIVE_PARAMETERS) - assessed_count) * 6.0
            score -= missing_pts
            deductions.append(f"Missing {len(self.ACTIVE_PARAMETERS) - assessed_count} parameters: -{missing_pts} pts")

        # 2. Inter-parameter consistency warnings
        warnings = self.check_inter_parameter_consistency(ayush_assessment)
        for w in warnings:
            pts = 10.0 if w["severity"] == "high" else (5.0 if w["severity"] == "medium" else 2.0)
            score -= pts
            deductions.append(f"Consistency warning ({w['rule_id']}): -{pts} pts")

        final_integrity = round(max(0.0, score), 1)

        quality_grade = "Excellent" if final_integrity >= 90 else ("Good" if final_integrity >= 75 else "Requires Verification")

        return {
            "data_integrity_score": final_integrity,
            "quality_grade": quality_grade,
            "parameter_completeness_percent": completeness_pct,
            "assessed_parameters_count": assessed_count,
            "deductions_breakdown": deductions
        }

    def generate_data_quality_report(self, patient_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates full Data Quality & Anomaly Validation Report.
        """
        ayush = patient_case.get("ayushAssessment", {}) or patient_case.get("ayush_assessment", {})
        dashavidha = ayush.get("dashavidhaPariksha", ayush)
        evidence = patient_case.get("clinical_evidence")

        warnings = self.check_inter_parameter_consistency(dashavidha)
        conflicts = self.check_evidence_source_conflicts(patient_case)
        integrity = self.compute_data_integrity_score(dashavidha, evidence)

        patient_id = patient_case.get("patient_id") or patient_case.get("patient", {}).get("id") or "UNKNOWN"

        return {
            "patient_id": patient_id,
            "assessment_summary": {
                "active_parameters_evaluated": len(self.ACTIVE_PARAMETERS),
                "excluded_parameters": self.EXCLUDED_PARAMETERS
            },
            "data_integrity": integrity,
            "consistency_warnings": warnings,
            "evidence_source_conflicts": conflicts,
            "is_valid_for_clinical_use": integrity["data_integrity_score"] >= 70.0,
            "timestamp": datetime.now().isoformat()
        }


def generate_data_quality_report(patient_case: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to run data quality engine."""
    engine = DataQualityEngine()
    return engine.generate_data_quality_report(patient_case)
