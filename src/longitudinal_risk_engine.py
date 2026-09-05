"""
Longitudinal Patient Progress Tracking & Health Risk Stratification Engine
-------------------------------------------------------------------------
Computes quantitative AYUSH health resilience scores, Vikriti imbalance severity scores,
assigns clinical risk tiers, and tracks longitudinal progress across patient visits.

Evaluates EXACTLY 10 active AYUSH parameters:
prakriti, vikriti, sara, samhanana, pramana, satmya, sattva, aharaShakti, vyayamaShakti, vaya.

EXCLUDED PARAMETERS: Agni, Koshtha.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime


class LongitudinalRiskEngine:
    """
    Engine for evaluating patient health resilience, calculating imbalance severity,
    determining clinical risk tiers, and tracking progress across longitudinal visits.
    """

    ACTIVE_PARAMETERS = [
        "prakriti", "vikriti", "sara", "samhanana", "pramana",
        "satmya", "sattva", "aharaShakti", "vyayamaShakti", "vaya"
    ]
    EXCLUDED_PARAMETERS = ["agni", "koshtha"]

    # Parameter score weightings for Resilience calculation
    QUALITATIVE_SCORES = {
        "pravara": 100.0,
        "uttama": 100.0,
        "madhyama": 65.0,
        "avara": 30.0,
        "hina": 30.0,
        "dura": 50.0
    }

    def __init__(self):
        pass

    def compute_health_resilience_score(
        self, ayush_assessment: Dict[str, Any], clinical_evidence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Computes AYUSH Health Resilience Score (0 - 100).
        High score indicates strong Ojas, Sattva, Sara, and physical endurance.
        """
        sub_scores = {}
        valid_weights = 0.0
        weighted_sum = 0.0

        # 1. Tissue Excellence (Sara) - weight 0.25
        sara = ayush_assessment.get("sara", {})
        sara_quality = str(sara.get("quality", "")).lower()
        if sara_quality in self.QUALITATIVE_SCORES:
            s_score = self.QUALITATIVE_SCORES[sara_quality]
            sub_scores["sara"] = s_score
            weighted_sum += s_score * 0.25
            valid_weights += 0.25

        # 2. Compactness (Samhanana) - weight 0.20
        samhanana = ayush_assessment.get("samhanana", {})
        sam_quality = str(samhanana.get("compactness", "")).lower()
        if sam_quality in self.QUALITATIVE_SCORES:
            sam_score = self.QUALITATIVE_SCORES[sam_quality]
            sub_scores["samhanana"] = sam_score
            weighted_sum += sam_score * 0.20
            valid_weights += 0.20

        # 3. Mental Strength (Sattva) - weight 0.25
        sattva = ayush_assessment.get("sattva", {})
        sat_quality = str(sattva.get("mental_strength", "")).lower()
        if sat_quality in self.QUALITATIVE_SCORES:
            sat_score = self.QUALITATIVE_SCORES[sat_quality]
            sub_scores["sattva"] = sat_score
            weighted_sum += sat_score * 0.25
            valid_weights += 0.25

        # 4. Work Capacity (VyayamaShakti) - weight 0.20
        vyayama = ayush_assessment.get("vyayamaShakti", {})
        vy_quality = str(vyayama.get("physical_work_capacity", "")).lower()
        if vy_quality in self.QUALITATIVE_SCORES:
            vy_score = self.QUALITATIVE_SCORES[vy_quality]
            sub_scores["vyayamaShakti"] = vy_score
            weighted_sum += vy_score * 0.20
            valid_weights += 0.20

        # 5. Adaptability (Satmya) - weight 0.10
        satmya = ayush_assessment.get("satmya", {})
        satmya_quality = str(satmya.get("adaptability", "")).lower()
        if satmya_quality in self.QUALITATIVE_SCORES:
            satmya_score = self.QUALITATIVE_SCORES[satmya_quality]
            sub_scores["satmya"] = satmya_score
            weighted_sum += satmya_score * 0.10
            valid_weights += 0.10

        if valid_weights == 0.0:
            final_resilience = 50.0  # Neutral fallback if no qualitative data exists
        else:
            final_resilience = round(weighted_sum / valid_weights, 1)

        return {
            "resilience_score": final_resilience,
            "sub_scores": sub_scores,
            "evaluation_confidence": round(valid_weights, 2)
        }

    def compute_imbalance_severity_score(
        self, ayush_assessment: Dict[str, Any], clinical_evidence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Computes Vikriti Imbalance Severity Score (0 - 100).
        High score indicates severe dosha aggravation or heavy clinical symptom burden.
        """
        severity_score = 0.0
        factors = []

        # 1. Vikriti changes count
        vikriti = ayush_assessment.get("vikriti", {})
        changes = vikriti.get("current_changes", []) if isinstance(vikriti, dict) else []
        change_count = len(changes)
        if change_count > 0:
            vikriti_points = min(40.0, change_count * 15.0)
            severity_score += vikriti_points
            factors.append(f"Vikriti aggravations ({change_count}): +{vikriti_points} pts")

        # 2. Extracted clinical symptoms
        symptoms = []
        if clinical_evidence and isinstance(clinical_evidence, dict):
            symptoms = clinical_evidence.get("extracted_symptoms", []) or []

        symptom_count = len(symptoms)
        if symptom_count > 0:
            symptom_points = min(45.0, symptom_count * 12.0)
            severity_score += symptom_points
            factors.append(f"Clinical symptoms extracted ({symptom_count}): +{symptom_points} pts")

        # 3. Avara Sattva or Avara Vyayama multiplier
        sattva_val = str(ayush_assessment.get("sattva", {}).get("mental_strength", "")).lower()
        if sattva_val in ["avara", "hina"]:
            severity_score += 15.0
            factors.append("Low Mental Strength (Avara Sattva) vulnerability: +15 pts")

        final_imbalance = round(min(100.0, severity_score), 1)

        return {
            "imbalance_severity_score": final_imbalance,
            "contributing_factors": factors,
            "aggravated_symptom_count": symptom_count
        }

    def determine_risk_tier(self, resilience_score: float, imbalance_score: float) -> Dict[str, Any]:
        """
        Categorizes patient into risk tiers based on resilience vs imbalance score.
        """
        if imbalance_score >= 65.0 or (imbalance_score >= 45.0 and resilience_score < 45.0):
            tier = "Critical"
            action = "Urgent consultation with senior Ayurvedic physician required."
        elif imbalance_score >= 40.0 or (imbalance_score >= 30.0 and resilience_score < 60.0):
            tier = "High"
            action = "Prioritized clinical follow-up and active dosha pacification advised."
        elif imbalance_score >= 20.0 or resilience_score < 50.0:
            tier = "Moderate"
            action = "Routine lifestyle correction and preventive Ayurvedic care recommended."
        else:
            tier = "Low"
            action = "Maintain healthy routine (Swasthavritta) and seasonal regimen (Ritucharya)."

        return {
            "risk_tier": tier,
            "clinical_action_recommendation": action,
            "resilience_score": resilience_score,
            "imbalance_severity_score": imbalance_score
        }

    def track_longitudinal_progress(
        self, current_case: Dict[str, Any], historical_cases: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Tracks longitudinal progress by comparing baseline/historical visits with current visit.
        """
        if not historical_cases:
            return {
                "trajectory": "insufficient_history",
                "visit_count": 1,
                "vikriti_resolution_rate_percent": None,
                "parameter_evolution_matrix": {},
                "summary": "Baseline visit registered. Longitudinal progress tracking will activate on subsequent visits."
            }

        # Sort all cases by date or visit_id
        all_visits = historical_cases + [current_case]
        
        def get_visit_key(c):
            val = c.get("visit_date") or c.get("timestamp") or c.get("created_at") or c.get("patient_id") or ""
            return str(val)

        sorted_visits = sorted(all_visits, key=get_visit_key)
        baseline_case = sorted_visits[0]
        latest_case = sorted_visits[-1]

        # Calculate scores for baseline vs current
        base_assess = baseline_case.get("ayush_assessment", {})
        curr_assess = latest_case.get("ayush_assessment", {})

        base_evidence = baseline_case.get("clinical_evidence", {})
        curr_evidence = latest_case.get("clinical_evidence", {})

        base_imb = self.compute_imbalance_severity_score(base_assess, base_evidence)["imbalance_severity_score"]
        curr_imb = self.compute_imbalance_severity_score(curr_assess, curr_evidence)["imbalance_severity_score"]

        # Resolution rate calculation
        if base_imb > 0:
            resolution_rate = round(((base_imb - curr_imb) / base_imb) * 100.0, 1)
        else:
            resolution_rate = 0.0 if curr_imb == 0 else -100.0

        # Determine overall trajectory
        if resolution_rate >= 15.0:
            trajectory = "improving"
            summary = f"Patient shows positive recovery trajectory with {resolution_rate}% imbalance resolution."
        elif resolution_rate <= -15.0:
            trajectory = "aggravated"
            summary = f"Patient shows aggravation trajectory with imbalance increasing by {abs(resolution_rate)}%."
        else:
            trajectory = "stable"
            summary = "Patient condition remains stable with minimal score fluctuation."

        # Parameter Evolution Matrix for 10 Active Parameters
        evolution_matrix = {}
        for param in self.ACTIVE_PARAMETERS:
            base_val = base_assess.get(param)
            curr_val = curr_assess.get(param)

            if base_val == curr_val:
                status = "unchanged"
            elif base_val is None:
                status = "newly_assessed"
            elif curr_val is None:
                status = "unassessed"
            else:
                status = "modified"

            evolution_matrix[param] = {
                "baseline": base_val,
                "current": curr_val,
                "status": status
            }

        return {
            "trajectory": trajectory,
            "visit_count": len(sorted_visits),
            "baseline_imbalance_score": base_imb,
            "current_imbalance_score": curr_imb,
            "vikriti_resolution_rate_percent": resolution_rate,
            "parameter_evolution_matrix": evolution_matrix,
            "summary": summary
        }

    def generate_longitudinal_risk_report(
        self, current_case: Dict[str, Any], historical_cases: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Assembles complete Longitudinal Progress & Risk Report.
        """
        patient_id = current_case.get("patient_id", "UNKNOWN")
        ayush_assessment = current_case.get("ayush_assessment", {})
        clinical_evidence = current_case.get("clinical_evidence", {})

        # 1. Compute Resilience & Imbalance
        resilience_data = self.compute_health_resilience_score(ayush_assessment, clinical_evidence)
        imbalance_data = self.compute_imbalance_severity_score(ayush_assessment, clinical_evidence)

        # 2. Risk Tier
        risk_data = self.determine_risk_tier(
            resilience_data["resilience_score"], imbalance_data["imbalance_severity_score"]
        )

        # 3. Longitudinal Progress
        progress_data = self.track_longitudinal_progress(current_case, historical_cases)

        return {
            "patient_id": patient_id,
            "assessment_summary": {
                "active_parameters_assessed": len(self.ACTIVE_PARAMETERS),
                "excluded_parameters": self.EXCLUDED_PARAMETERS,
                "prakriti": ayush_assessment.get("prakriti", {}).get("prakriti_type"),
                "vikriti_status": "Aggravated" if imbalance_data["imbalance_severity_score"] > 20 else "Normal"
            },
            "health_resilience": resilience_data,
            "imbalance_severity": imbalance_data,
            "risk_stratification": risk_data,
            "longitudinal_progress": progress_data,
            "timestamp": datetime.now().isoformat()
        }


def generate_longitudinal_risk_report(
    current_case: Dict[str, Any], historical_cases: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Helper function to run longitudinal risk engine."""
    engine = LongitudinalRiskEngine()
    return engine.generate_longitudinal_risk_report(current_case, historical_cases)
