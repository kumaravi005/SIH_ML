"""
AYUSH Clinical Outcome & Recovery Prediction Engine
---------------------------------------------------
Projects expected recovery timelines, computes dosha stabilization probabilities,
and identifies prognostic factors for clinical decision support.

Evaluates EXACTLY 10 active AYUSH parameters:
prakriti, vikriti, sara, samhanana, pramana, satmya, sattva, aharaShakti, vyayamaShakti, vaya.

EXCLUDED PARAMETERS: Agni, Koshtha.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from longitudinal_risk_engine import LongitudinalRiskEngine


class ClinicalPredictionEngine:
    """
    Engine for multi-variable clinical outcome prognosis and recovery timeline prediction.
    """

    ACTIVE_PARAMETERS = [
        "prakriti", "vikriti", "sara", "samhanana", "pramana",
        "satmya", "sattva", "aharaShakti", "vyayamaShakti", "vaya"
    ]
    EXCLUDED_PARAMETERS = ["agni", "koshtha"]

    ADHERENCE_MULTIPLIERS = {
        "high": 1.25,
        "moderate": 1.0,
        "low": 0.65
    }

    def __init__(self):
        self.risk_engine = LongitudinalRiskEngine()

    def predict_dosha_stabilization_probability(
        self, ayush_assessment: Dict[str, Any], clinical_evidence: Optional[Dict[str, Any]] = None, adherence_level: str = "moderate"
    ) -> float:
        """
        Calculates projected probability (0 - 100%) of achieving dosha balance within standard treatment duration.
        High resilience + high adherence -> higher probability.
        High imbalance severity -> lower probability.
        """
        res_data = self.risk_engine.compute_health_resilience_score(ayush_assessment, clinical_evidence)
        imb_data = self.risk_engine.compute_imbalance_severity_score(ayush_assessment, clinical_evidence)

        res_score = res_data["resilience_score"]
        imb_score = imb_data["imbalance_severity_score"]

        # Base probability calculation
        base_prob = 50.0 + (res_score * 0.4) - (imb_score * 0.35)

        # Adherence adjustment
        adh_key = str(adherence_level).lower()
        mult = self.ADHERENCE_MULTIPLIERS.get(adh_key, 1.0)
        adj_prob = base_prob * mult

        return round(min(98.0, max(15.0, adj_prob)), 1)

    def estimate_recovery_timeframe_weeks(
        self, resilience_score: float, imbalance_score: float, adherence_level: str = "moderate"
    ) -> Dict[str, Any]:
        """
        Estimates expected recovery timeframe range (in weeks).
        """
        adh_key = str(adherence_level).lower()

        if imbalance_score < 25.0 and resilience_score >= 65.0:
            min_w, max_w = 2, 4
            category = "Rapid Recovery"
        elif imbalance_score < 45.0 or resilience_score >= 50.0:
            min_w, max_w = 4, 8
            category = "Standard Recovery"
        elif imbalance_score < 65.0:
            min_w, max_w = 8, 12
            category = "Extended Recovery"
        else:
            min_w, max_w = 12, 16
            category = "Prolonged Management"

        if adh_key == "low":
            min_w = int(min_w * 1.5)
            max_w = int(max_w * 1.5)
        elif adh_key == "high":
            min_w = max(2, int(min_w * 0.8))
            max_w = max(3, int(max_w * 0.8))

        return {
            "estimated_min_weeks": min_w,
            "estimated_max_weeks": max_w,
            "timeframe_display": f"{min_w}–{max_w} weeks",
            "recovery_category": category
        }

    def identify_prognostic_factors(
        self, ayush_assessment: Dict[str, Any], clinical_evidence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[str]]:
        """
        Identifies positive prognostic indicators vs risk factors based on 10 active parameters.
        """
        positive_factors = []
        risk_factors = []

        # 1. Sattva (mental strength)
        sattva = str(ayush_assessment.get("sattva", {}).get("mental_strength", "")).lower()
        if sattva in ["pravara", "uttama", "strong"]:
            positive_factors.append("Strong mental resilience (Pravara Sattva) accelerates healing response.")
        elif sattva in ["avara", "hina", "weak"]:
            risk_factors.append("Low mental strength (Avara Sattva) may require enhanced psychological support.")

        # 2. Sara (tissue excellence)
        sara = str(ayush_assessment.get("sara", {}).get("quality", "")).lower()
        if sara in ["pravara", "uttama", "strong"]:
            positive_factors.append("Excellent tissue quality (Pravara Sara) supports rapid recovery.")
        elif sara in ["avara", "hina", "weak"]:
            risk_factors.append("Weak tissue excellence (Avara Sara) increases recovery timeframe.")

        # 3. VyayamaShakti (work capacity)
        vyayama = str(ayush_assessment.get("vyayamaShakti", {}).get("physical_work_capacity", "")).lower()
        if vyayama in ["pravara", "uttama", "strong"]:
            positive_factors.append("High physical work capacity (Pravara VyayamaShakti) enhances vitality.")

        # 4. Vikriti aggravations
        vikriti = ayush_assessment.get("vikriti", {})
        changes = vikriti.get("current_changes", []) if isinstance(vikriti, dict) else []
        if len(changes) >= 3:
            risk_factors.append(f"Multiple active Vikriti aggravations ({len(changes)}) require targeted multi-dosha pacification.")

        return {
            "positive_prognostic_indicators": positive_factors,
            "risk_and_delay_factors": risk_factors
        }

    def generate_clinical_prediction_report(
        self, patient_case: Dict[str, Any], adherence_level: str = "moderate"
    ) -> Dict[str, Any]:
        """
        Assembles complete Clinical Outcome & Recovery Prediction Report.
        """
        ayush = patient_case.get("ayushAssessment", {}) or patient_case.get("ayush_assessment", {})
        evidence = patient_case.get("clinical_evidence")

        res_data = self.risk_engine.compute_health_resilience_score(ayush, evidence)
        imb_data = self.risk_engine.compute_imbalance_severity_score(ayush, evidence)

        prob = self.predict_dosha_stabilization_probability(ayush, evidence, adherence_level=adherence_level)
        timeframe = self.estimate_recovery_timeframe_weeks(
            res_data["resilience_score"], imb_data["imbalance_severity_score"], adherence_level=adherence_level
        )
        factors = self.identify_prognostic_factors(ayush, evidence)

        patient_id = patient_case.get("patient_id") or patient_case.get("patient", {}).get("id") or "UNKNOWN"

        return {
            "patient_id": patient_id,
            "assessment_summary": {
                "active_parameters_evaluated": len(self.ACTIVE_PARAMETERS),
                "excluded_parameters": self.EXCLUDED_PARAMETERS,
                "assumed_adherence_level": adherence_level
            },
            "predicted_dosha_stabilization_probability": prob,
            "predicted_dosha_stabilization_percentage": f"{prob}%",
            "recovery_timeframe": timeframe,
            "prognostic_analysis": factors,
            "timestamp": datetime.now().isoformat(),
            "disclaimer": "Predicted recovery outcomes are decision-support estimates based on AYUSH parameters and compliance. They do not constitute a medical prognosis guarantee."
        }


def predict_clinical_outcome(
    patient_case: Dict[str, Any], adherence_level: str = "moderate"
) -> Dict[str, Any]:
    """Helper function to run clinical outcome prediction engine."""
    engine = ClinicalPredictionEngine()
    return engine.generate_clinical_prediction_report(patient_case, adherence_level=adherence_level)
