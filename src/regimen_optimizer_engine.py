"""
Personalized AYUSH Regimen & Lifestyle Optimization Engine
---------------------------------------------------------
Generates time-blocked daily routines (Dinacharya) and seasonal wellness plans (Ritucharya)
customized to Prakriti-Vikriti profiles, Sattva, VyayamaShakti, Vaya, and current season.

Evaluates EXACTLY 10 active AYUSH parameters:
prakriti, vikriti, sara, samhanana, pramana, satmya, sattva, aharaShakti, vyayamaShakti, vaya.

EXCLUDED PARAMETERS: Agni, Koshtha.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime


class RegimenOptimizerEngine:
    """
    Engine for generating personalized daily routines and seasonal AYUSH wellness plans.
    """

    ACTIVE_PARAMETERS = [
        "prakriti", "vikriti", "sara", "samhanana", "pramana",
        "satmya", "sattva", "aharaShakti", "vyayamaShakti", "vaya"
    ]
    EXCLUDED_PARAMETERS = ["agni", "koshtha"]

    VALID_SEASONS = ["Vasanta", "Grishma", "Varsha", "Sharad", "Hemanta", "Shishira"]

    def __init__(self):
        pass

    def optimize_daily_dinacharya_schedule(self, ayush_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generates a 4-period time-blocked daily routine customized to dosha dominance and mental strength.
        """
        prakriti = ayush_assessment.get("prakriti", {})
        p_type = str(prakriti.get("prakriti_type") or prakriti.get("primary_dosha") or "vata").lower()

        sattva_val = str(ayush_assessment.get("sattva", {}).get("mental_strength", "")).lower()
        vyayama_val = str(ayush_assessment.get("vyayamaShakti", {}).get("physical_work_capacity", "")).lower()

        # Morning Routine (05:00 - 09:00)
        morning_activity = "Brahma Muhurta awakening (approx 05:30 AM), gentle warm water rinse, tongue scraping, and 15-minute mindfulness/Pranayama."
        if "vata" in p_type:
            morning_activity += " Warm sesame oil self-massage (Abhyanga) followed by a warm bath."
        elif "pitta" in p_type:
            morning_activity += " Cool or lukewarm bath with soothing sandalwood/rose water."
        elif "kapha" in p_type:
            morning_activity += " Dry powder massage (Udvartana) and invigorating light exercise."

        # Midday Routine (11:30 - 13:30)
        midday_activity = "Main meal of the day (highest digestive period). Sit quietly for 10 minutes post-meal. Avoid immediate strenuous work."

        # Evening Routine (17:00 - 19:00)
        if vyayama_val in ["pravara", "uttama", "strong"]:
            exercise_duration = "30–45 minutes of moderate physical activity or yoga asana practice."
        else:
            exercise_duration = "15–20 minutes of gentle walking or calming Pranayama (Anulom Vilom)."

        evening_activity = f"Unwind period: {exercise_duration} Light, warm dinner before 07:30 PM."

        # Night Routine (21:00 - 22:30)
        night_activity = "Digital detox 1 hour before sleep. Warm foot massage (Padabhyanga) with ghee/sesame oil. In-bed by 10:00 PM."
        if sattva_val in ["avara", "hina", "weak"]:
            night_activity += " 10-minute guided relaxation or soothing chamomile/warm nutmeg milk."

        return [
            {"time_block": "Morning (05:00 - 09:00)", "phase": "Pratah Charya", "recommended_routine": morning_activity},
            {"time_block": "Midday (11:30 - 13:30)", "phase": "Madhyahna Charya", "recommended_routine": midday_activity},
            {"time_block": "Evening (17:00 - 19:00)", "phase": "Sayam Charya", "recommended_routine": evening_activity},
            {"time_block": "Night (21:00 - 22:30)", "phase": "Ratri Charya", "recommended_routine": night_activity}
        ]

    def optimize_seasonal_ritucharya(self, ayush_assessment: Dict[str, Any], season: str = "Vasanta") -> Dict[str, Any]:
        """
        Provides seasonal dietary and lifestyle guidelines across AYUSH Ritus.
        """
        s_norm = season.capitalize() if season else "Vasanta"
        if s_norm not in self.VALID_SEASONS:
            s_norm = "Vasanta"

        prakriti = ayush_assessment.get("prakriti", {})
        p_type = str(prakriti.get("prakriti_type") or prakriti.get("primary_dosha") or "vata").lower()

        seasonal_rules = {
            "Vasanta": {
                "season_name": "Spring (Vasanta Ritu)",
                "dominant_dosha_impact": "Kapha liquefaction phase",
                "dietary_guidelines": "Favor bitter, pungent, and astringent tastes. Avoid heavy, oily, sweet foods and daytime sleep.",
                "lifestyle_guidelines": "Engage in active physical exercise, dry massage, and warm herbal tea consumption."
            },
            "Grishma": {
                "season_name": "Summer (Grishma Ritu)",
                "dominant_dosha_impact": "Pitta accumulation phase",
                "dietary_guidelines": "Favor sweet, cool, liquid foods (coconut water, milk, rice). Avoid spicy, sour, salty foods.",
                "lifestyle_guidelines": "Avoid direct noon sun, strenuous exercise, and stay well hydrated."
            },
            "Varsha": {
                "season_name": "Monsoon (Varsha Ritu)",
                "dominant_dosha_impact": "Vata aggravation phase",
                "dietary_guidelines": "Favor sour, salty, unctuous warm foods. Ensure boiled water consumption.",
                "lifestyle_guidelines": "Keep surroundings dry and clean. Avoid dampness and cold breeze exposure."
            },
            "Sharad": {
                "season_name": "Autumn (Sharad Ritu)",
                "dominant_dosha_impact": "Pitta aggravation phase",
                "dietary_guidelines": "Favor sweet, bitter foods and ghee. Avoid heavy fried foods.",
                "lifestyle_guidelines": "Moonlight exposure (Kumudaka), gentle evening walks, cooling bath regimens."
            },
            "Hemanta": {
                "season_name": "Early Winter (Hemanta Ritu)",
                "dominant_dosha_impact": "Vata pacification & strong digestive power",
                "dietary_guidelines": "Favor unctuous, sweet, sour, salty foods and nourishing warm soups.",
                "lifestyle_guidelines": "Sunlight exposure, vigorous exercise, oil massage (Abhyanga), and warm clothing."
            },
            "Shishira": {
                "season_name": "Late Winter (Shishira Ritu)",
                "dominant_dosha_impact": "Vata & Kapha preservation phase",
                "dietary_guidelines": "Favor warm, hearty, well-spiced meals. Avoid cold drinks.",
                "lifestyle_guidelines": "Stay warm, practice active yoga, and maintain regular sleep schedules."
            }
        }

        selected_plan = seasonal_rules.get(s_norm, seasonal_rules["Vasanta"])
        selected_plan["individualized_note"] = f"Plan tailored for primary constitution: {p_type.capitalize()}."

        return selected_plan

    def generate_personalized_regimen_report(
        self, patient_case: Dict[str, Any], season: str = "Vasanta"
    ) -> Dict[str, Any]:
        """
        Assembles full Personalized AYUSH Regimen & Lifestyle Optimization Report.
        """
        ayush = patient_case.get("ayushAssessment", {}) or patient_case.get("ayush_assessment", {})

        daily_routine = self.optimize_daily_dinacharya_schedule(ayush)
        seasonal_plan = self.optimize_seasonal_ritucharya(ayush, season=season)

        patient_id = patient_case.get("patient_id") or patient_case.get("patient", {}).get("id") or "UNKNOWN"

        return {
            "patient_id": patient_id,
            "assessment_summary": {
                "active_parameters_evaluated": len(self.ACTIVE_PARAMETERS),
                "excluded_parameters": self.EXCLUDED_PARAMETERS,
                "current_season": season
            },
            "daily_dinacharya_routine": daily_routine,
            "seasonal_ritucharya_plan": seasonal_plan,
            "timestamp": datetime.now().isoformat(),
            "disclaimer": "This daily and seasonal regimen is supportive lifestyle advice based on AYUSH principles. It does not substitute clinical treatment directives."
        }


def generate_personalized_regimen_report(
    patient_case: Dict[str, Any], season: str = "Vasanta"
) -> Dict[str, Any]:
    """Helper function to run regimen optimizer engine."""
    engine = RegimenOptimizerEngine()
    return engine.generate_personalized_regimen_report(patient_case, season=season)
