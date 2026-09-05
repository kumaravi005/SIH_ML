"""
Patient Similarity & Clinical Case Retrieval Engine
---------------------------------------------------
Vectorizes multi-dimensional AYUSH patient cases (10 active parameters + clinical symptoms)
and computes Cosine Similarity scores for retrieval-augmented clinical decision support.

Evaluates EXACTLY 10 active AYUSH parameters:
prakriti, vikriti, sara, samhanana, pramana, satmya, sattva, aharaShakti, vyayamaShakti, vaya.

EXCLUDED PARAMETERS: Agni, Koshtha.
"""

import math
from typing import Dict, Any, List, Optional, Tuple


class PatientSimilarityEngine:
    """
    Engine for feature vectorization and cosine similarity / k-NN retrieval of patient cases.
    """

    ACTIVE_PARAMETERS = [
        "prakriti", "vikriti", "sara", "samhanana", "pramana",
        "satmya", "sattva", "aharaShakti", "vyayamaShakti", "vaya"
    ]
    EXCLUDED_PARAMETERS = ["agni", "koshtha"]

    DOSHA_ENCODINGS = {
        "vata": [1.0, 0.0, 0.0],
        "pitta": [0.0, 1.0, 0.0],
        "kapha": [0.0, 0.0, 1.0],
        "vata-pitta": [0.7, 0.7, 0.0],
        "pitta-vata": [0.7, 0.7, 0.0],
        "pitta-kapha": [0.0, 0.7, 0.7],
        "kapha-pitta": [0.0, 0.7, 0.7],
        "vata-kapha": [0.7, 0.0, 0.7],
        "kapha-vata": [0.7, 0.0, 0.7],
        "tridoshaja": [0.6, 0.6, 0.6]
    }

    QUALITATIVE_SCALARS = {
        "pravara": 1.0, "uttama": 1.0, "strong": 1.0, "high": 1.0,
        "madhyama": 0.65, "balanced": 0.65, "moderate": 0.65,
        "avara": 0.3, "hina": 0.3, "weak": 0.3, "low": 0.3
    }

    def __init__(self):
        pass

    def vectorize_patient_case(self, patient_case: Dict[str, Any]) -> List[float]:
        """
        Converts patient case (10 AYUSH parameters) into a standardized numerical feature vector.
        Vector dimensions (12-dim):
        0-2: Prakriti (Vata, Pitta, Kapha components)
        3: Vikriti severity (0.0 to 1.0)
        4: Sara quality scalar
        5: Samhanana compactness scalar
        6: Pramana proportion scalar
        7: Satmya adaptability scalar
        8: Sattva mental strength scalar
        9: AharaShakti digestive capacity scalar
        10: VyayamaShakti work capacity scalar
        11: Vaya age group scalar
        """
        vector = []

        ayush = patient_case.get("ayushAssessment", {}) or patient_case.get("ayush_assessment", {})

        # 1. Prakriti (3-dim)
        prakriti = ayush.get("prakriti", {})
        p_type = str(prakriti.get("prakriti_type") or prakriti.get("primary_dosha") or "madhyama").lower()
        matched_encoded = [0.33, 0.33, 0.33]
        for k, enc in self.DOSHA_ENCODINGS.items():
            if k in p_type:
                matched_encoded = enc
                break
        vector.extend(matched_encoded)

        # 2. Vikriti severity (1-dim)
        vikriti = ayush.get("vikriti", {})
        changes = vikriti.get("current_changes", []) if isinstance(vikriti, dict) else []
        v_score = min(1.0, len(changes) * 0.25)
        vector.append(v_score)

        # 3. Qualitative parameters (8-dim)
        for param in ["sara", "samhanana", "pramana", "satmya", "sattva", "aharaShakti", "vyayamaShakti", "vaya"]:
            data = ayush.get(param, {})
            val_str = ""
            if isinstance(data, dict):
                val_str = str(data.get("quality") or data.get("compactness") or data.get("proportions") or
                              data.get("adaptability") or data.get("mental_strength") or
                              data.get("digestive_capacity") or data.get("physical_work_capacity") or
                              data.get("age_group") or "").lower()
            elif isinstance(data, str):
                val_str = data.lower()

            scalar = 0.5  # default neutral
            for q_key, q_val in self.QUALITATIVE_SCALARS.items():
                if q_key in val_str:
                    scalar = q_val
                    break
            vector.append(scalar)

        return vector

    def calculate_cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """
        Computes Cosine Similarity score between two feature vectors.
        Returns value between 0.0 and 1.0.
        """
        if len(vec_a) != len(vec_b) or not vec_a:
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        sim = dot_product / (norm_a * norm_b)
        return round(min(1.0, max(0.0, sim)), 4)

    def find_similar_cases(
        self, target_case: Dict[str, Any], historical_cases: List[Dict[str, Any]], top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Finds top-k most similar patient cases from historical database.
        """
        if not historical_cases:
            return []

        target_vec = self.vectorize_patient_case(target_case)
        matches = []

        for idx, h_case in enumerate(historical_cases):
            h_vec = self.vectorize_patient_case(h_case)
            sim_score = self.calculate_cosine_similarity(target_vec, h_vec)

            pid = h_case.get("patient_id") or h_case.get("patient", {}).get("id") or f"CASE-{idx+1}"

            matches.append({
                "patient_id": pid,
                "similarity_score": sim_score,
                "similarity_percentage": f"{round(sim_score * 100, 1)}%",
                "prakriti": h_case.get("ayushAssessment", {}).get("prakriti") or h_case.get("ayush_assessment", {}).get("prakriti"),
                "matched_case_summary": {
                    "age": h_case.get("patient", {}).get("age"),
                    "gender": h_case.get("patient", {}).get("gender"),
                    "chief_complaint": h_case.get("chiefComplaint", {}).get("complaints", [{}])[0].get("description") if h_case.get("chiefComplaint") else None
                }
            })

        matches.sort(key=lambda x: x["similarity_score"], reverse=True)
        return matches[:top_k]

    def generate_case_retrieval_report(
        self, target_case: Dict[str, Any], historical_cases: List[Dict[str, Any]], top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Generates structured case retrieval report.
        """
        similar_matches = self.find_similar_cases(target_case, historical_cases, top_k=top_k)
        target_pid = target_case.get("patient_id") or target_case.get("patient", {}).get("id") or "TARGET-001"

        return {
            "target_patient_id": target_pid,
            "assessment_summary": {
                "active_parameters_evaluated": len(self.ACTIVE_PARAMETERS),
                "excluded_parameters": self.EXCLUDED_PARAMETERS,
                "vector_dimensions": 12
            },
            "retrieved_similar_cases": similar_matches,
            "top_match_similarity": similar_matches[0]["similarity_percentage"] if similar_matches else "0%",
            "clinical_decision_support_note": "Case retrieval matches similar historical profiles for physician reference. Does not replace individualized clinical judgement."
        }


def find_similar_patient_cases(
    target_case: Dict[str, Any], historical_cases: List[Dict[str, Any]], top_k: int = 3
) -> Dict[str, Any]:
    """Helper function to run patient similarity engine."""
    engine = PatientSimilarityEngine()
    return engine.generate_case_retrieval_report(target_case, historical_cases, top_k=top_k)
