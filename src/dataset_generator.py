"""
Synthetic Clinical Patient Dataset Generation & Dataset Validation Engine (Section 7 & 11)

Synthesizes N=500 realistic, diverse patient cases with multi-source evidence inputs
(questionnaire answers + clinical text / audio transcripts) and ground-truth target labels
based strictly on the 10 active AYUSH parameters:
- prakriti, vikriti, sara, samhanana, pramana, satmya, sattva, aharaShakti, vyayamaShakti, vaya

Section 11 Enhancements:
- Latent Clinical Archetype Engine for probabilistic, noise-controlled feature-target alignment.
- Zero feature leakage: latent archetype is used ONLY during synthetic generation and is NEVER passed into X or stored in patient cases.
- Strictly excludes Agni and Koshtha. Zero external mandatory dependencies.
"""

import json
import random
from pathlib import Path

# 10 Active Parameters Constant
ACTIVE_PARAMETERS = [
    "prakriti", "vikriti", "sara", "samhanana", "pramana",
    "satmya", "sattva", "aharaShakti", "vyayamaShakti", "vaya"
]

EXCLUDED_PARAMETERS = ["agni", "koshtha"]

ARCHETYPES = {
    "vata_dominant": {"vata": 0.85, "pitta": 0.08, "kapha": 0.07},
    "pitta_dominant": {"vata": 0.07, "pitta": 0.85, "kapha": 0.08},
    "kapha_dominant": {"vata": 0.08, "pitta": 0.07, "kapha": 0.85},
    "vata_pitta": {"vata": 0.50, "pitta": 0.45, "kapha": 0.05},
    "pitta_kapha": {"vata": 0.05, "pitta": 0.50, "kapha": 0.45},
    "vata_kapha": {"vata": 0.45, "pitta": 0.05, "kapha": 0.50}
}


class AYUSHDatasetGeneratorEngine:
    def __init__(self, seed=42):
        self.seed = seed
        random.seed(seed)
        base_dir = Path(__file__).resolve().parent.parent
        self.questions_file = base_dir / "data" / "ayush_questions.json"
        self.prakriti_map_file = base_dir / "data" / "prakriti_mapping.json"
        self.output_dataset_file = base_dir / "data" / "ayush_synthetic_dataset.json"
        self.question_tree = self._load_questions()
        self.prakriti_rules = self._load_prakriti_rules()

    def _load_questions(self):
        if self.questions_file.exists():
            with open(self.questions_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("ayushQuestionTree", {})
        return {}

    def _load_prakriti_rules(self):
        if self.prakriti_map_file.exists():
            with open(self.prakriti_map_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("prakriti", {}).get("questions", {})
        return {}

    def _generate_demographics(self, idx):
        first_names = ["Rohan", "Ananya", "Vikram", "Priya", "Amit", "Sneha", "Karan", "Pooja", "Rahul", "Meera"]
        last_names = ["Sharma", "Verma", "Patel", "Gupta", "Iyer", "Nair", "Reddy", "Singh", "Joshi", "Chopra"]

        gender = "Male" if idx % 2 == 0 else "Female"
        age = random.randint(18, 80)
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        patient_id = f"P{1000 + idx}"

        return {
            "patientId": patient_id,
            "name": name,
            "age": age,
            "gender": gender,
            "preferredLanguage": "English"
        }

    def _generate_questionnaire_answers(self, age, archetype_weights=None):
        answers = {}
        if archetype_weights is None:
            archetype_weights = {"vata": 0.33, "pitta": 0.33, "kapha": 0.33}

        # 1. Prakriti Questions (probabilistically aligned with archetype)
        prakriti_q = self.question_tree.get("prakriti", {}).get("questions", [])
        answers["prakriti"] = []
        for q in prakriti_q:
            opts = q.get("options", [])
            q_id = q["id"]
            if opts:
                q_rules = self.prakriti_rules.get(q_id, {})
                weights = []
                for o in opts:
                    if o == "Not sure":
                        w = 0.01
                    else:
                        s = q_rules.get(o, {})
                        w = (
                            0.02
                            + archetype_weights["vata"] * (s.get("vata", 0.0) ** 2)
                            + archetype_weights["pitta"] * (s.get("pitta", 0.0) ** 2)
                            + archetype_weights["kapha"] * (s.get("kapha", 0.0) ** 2)
                        )
                    weights.append(w)

                tot = sum(weights) or 1.0
                norm_weights = [w / tot for w in weights]
                chosen = random.choices(opts, weights=norm_weights, k=1)[0]

                answers["prakriti"].append({
                    "questionId": q_id,
                    "answer": chosen,
                    "source": "questionnaire",
                    "confidence": round(random.uniform(0.85, 1.0), 2)
                })

        # 2. Sara
        sara_q = self.question_tree.get("sara", {}).get("questions", [])
        answers["sara"] = []
        for q in sara_q:
            if q["answerType"] == "single_choice":
                chosen = random.choice(q.get("options", ["Mostly"]))
            else:
                chosen = random.choice(["Strong tissue quality", "Moderate strength", "Good physical appearance"])
            answers["sara"].append({
                "questionId": q["id"],
                "answer": chosen,
                "source": "questionnaire",
                "confidence": round(random.uniform(0.80, 0.95), 2)
            })

        # 3. Samhanana
        sam_q = self.question_tree.get("samhanana", {}).get("questions", [])
        answers["samhanana"] = []
        for q in sam_q:
            opts = [o for o in q.get("options", ["Well developed"]) if o != "Not sure"]
            chosen = random.choice(opts if opts else ["Well developed"])
            answers["samhanana"].append({
                "questionId": q["id"],
                "answer": chosen,
                "source": "questionnaire",
                "confidence": round(random.uniform(0.80, 0.95), 2)
            })

        # 4. Pramana
        pramana_q = self.question_tree.get("pramana", {}).get("questions", [])
        answers["pramana"] = []
        height = random.randint(150, 185)
        weight = random.randint(50, 90)
        for q in pramana_q:
            if q["answerType"] == "free_text":
                chosen = f"Height: {height} cm, Weight: {weight} kg"
            else:
                opts = [o for o in q.get("options", ["Generally proportionate"]) if o != "Not sure"]
                chosen = random.choice(opts if opts else ["Generally proportionate"])
            answers["pramana"].append({
                "questionId": q["id"],
                "answer": chosen,
                "source": "questionnaire",
                "confidence": round(random.uniform(0.85, 0.98), 2)
            })

        # 5. Satmya
        satmya_q = self.question_tree.get("satmya", {}).get("questions", [])
        answers["satmya"] = []
        for q in satmya_q:
            if q["answerType"] == "single_choice":
                chosen = random.choice(["Very well", "Mostly well", "Sometimes causes discomfort"])
            else:
                chosen = random.choice(["Vegetarian diet", "Balanced home-cooked meals", "Low spicy food"])
            answers["satmya"].append({
                "questionId": q["id"],
                "answer": chosen,
                "source": "questionnaire",
                "confidence": round(random.uniform(0.80, 0.95), 2)
            })

        # 6. Sattva
        sattva_q = self.question_tree.get("sattva", {}).get("questions", [])
        answers["sattva"] = []
        for q in sattva_q:
            if q["answerType"] == "single_choice":
                chosen = random.choice(["Usually calm and focused", "Generally confident", "Rarely"])
            else:
                chosen = random.choice(["I practice meditation and stay calm", "Focus on problem solving"])
            answers["sattva"].append({
                "questionId": q["id"],
                "answer": chosen,
                "source": "questionnaire",
                "confidence": round(random.uniform(0.80, 0.95), 2)
            })

        # 7. Ahara Shakti
        ahara_q = self.question_tree.get("aharaShakti", {}).get("questions", [])
        answers["aharaShakti"] = []
        for q in ahara_q:
            if q["answerType"] == "single_choice":
                chosen = random.choice(["Good", "Moderate", "Comfortable and satisfied", "Mostly regularly"])
            else:
                chosen = random.choice(["Normal meal size", "Two full meals daily"])
            answers["aharaShakti"].append({
                "questionId": q["id"],
                "answer": chosen,
                "source": "questionnaire",
                "confidence": round(random.uniform(0.80, 0.95), 2)
            })

        # 8. Vyayama Shakti
        vyayama_q = self.question_tree.get("vyayamaShakti", {}).get("questions", [])
        answers["vyayamaShakti"] = []
        for q in vyayama_q:
            if q["answerType"] == "single_choice":
                chosen = random.choice(["Moderately active", "Mildly tired but recover quickly", "Quickly"])
            else:
                chosen = random.choice(["30 minutes daily walk", "45 minutes light exercise"])
            answers["vyayamaShakti"].append({
                "questionId": q["id"],
                "answer": chosen,
                "source": "questionnaire",
                "confidence": round(random.uniform(0.80, 0.95), 2)
            })

        # 9. Vaya
        vaya_q = self.question_tree.get("vaya", {}).get("questions", [])
        answers["vaya"] = []
        stage = "Adulthood" if age < 50 else ("Middle age" if age < 65 else "Older adulthood")
        for q in vaya_q:
            if q["answerType"] == "number":
                chosen = age
            elif q["answerType"] == "single_choice":
                chosen = stage
            else:
                chosen = "No major age-related changes"
            answers["vaya"].append({
                "questionId": q["id"],
                "answer": chosen,
                "source": "questionnaire",
                "confidence": 1.0
            })

        return answers

    def _generate_unstructured_inputs(self, archetype_name):
        symptoms_map = {
            "vata_dominant": "dry skin, constipation, joint pain, and anxiety",
            "pitta_dominant": "hyperacidity, skin rashes, burning sensation, and irritability",
            "kapha_dominant": "lethargy, chest heaviness, weight gain, and excessive sleepiness",
            "vata_pitta": "dry skin, hyperacidity, and mild joint pain",
            "pitta_kapha": "acidity, lethargy, and skin sensitivity",
            "vata_kapha": "dry skin, lethargy, and joint stiffness"
        }
        chosen_symptom = symptoms_map.get(archetype_name, "mild fatigue and body ache")

        return [
            {
                "text": f"Patient reports {chosen_symptom}. Taking light diet and resting.",
                "source": random.choice(["audio_transcript", "text"]),
                "confidence": round(random.uniform(0.88, 0.96), 2)
            }
        ]

    def generate_patient_case(self, idx):
        """
        Generate a single synthetic patient case using latent archetype probabilistic sampling.
        The archetype is used solely for response generation and is NOT saved in patient_case.
        """
        archetypes_keys = list(ARCHETYPES.keys())
        archetype_name = archetypes_keys[idx % len(archetypes_keys)]
        archetype_weights = ARCHETYPES[archetype_name]

        demographics = self._generate_demographics(idx)
        questionnaire_answers = self._generate_questionnaire_answers(demographics["age"], archetype_weights)
        unstructured_inputs = self._generate_unstructured_inputs(archetype_name)

        # Build full case using existing pipeline case builder
        from patient_case_builder import build_complete_patient_case
        patient_case = build_complete_patient_case(
            patient_answers=questionnaire_answers,
            unstructured_inputs=unstructured_inputs
        )
        patient_case["patient"]["patientId"] = demographics["patientId"]
        patient_case["patient"]["name"] = demographics["name"]
        patient_case["patient"]["age"] = demographics["age"]
        patient_case["patient"]["gender"] = demographics["gender"]

        return patient_case

    def generate_dataset(self, num_samples=500, save_path=None):
        """
        Generate N synthetic patient cases, validate quality, and save to disk.
        """
        path = Path(save_path) if save_path else self.output_dataset_file
        path.parent.mkdir(parents=True, exist_ok=True)

        dataset = []
        for idx in range(1, num_samples + 1):
            case = self.generate_patient_case(idx)
            dataset.append(case)

        # Validate dataset quality via DataQualityEngine
        from data_quality_engine import DataQualityEngine
        quality_engine = DataQualityEngine()

        valid_count = 0
        for case in dataset:
            report = quality_engine.generate_data_quality_report(case)
            score = report.get("data_integrity", {}).get("data_integrity_score", 0)
            if score >= 70.0:
                valid_count += 1

        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            saved_to_str = str(path)
        elif num_samples == 500:
            path = self.output_dataset_file
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            saved_to_str = str(path)
        else:
            saved_to_str = "in-memory (not saved to main dataset)"

        stats = self.get_dataset_statistics(dataset)
        stats["total_generated"] = len(dataset)
        stats["quality_validated_count"] = valid_count
        stats["saved_to"] = saved_to_str

        return stats

    def get_dataset_statistics(self, dataset):
        """
        Compute dataset distribution metrics across 10 active parameters.
        """
        total = len(dataset)
        prakriti_dist = {}
        vikriti_dist = {}

        for case in dataset:
            dashavidha = case.get("ayushAssessment", {}).get("dashavidhaPariksha", {})
            p_res = dashavidha.get("prakriti", {})
            v_res = dashavidha.get("vikriti", {})

            p_val = p_res.get("value") if isinstance(p_res, dict) else str(p_res)
            v_val = v_res.get("value") if isinstance(v_res, dict) else str(v_res)

            p_str = str(p_val) if p_val is not None else "Unknown"
            v_str = str(v_val) if v_val is not None else "Unknown"

            prakriti_dist[p_str] = prakriti_dist.get(p_str, 0) + 1
            vikriti_dist[v_str] = vikriti_dist.get(v_str, 0) + 1

        return {
            "total_cases": total,
            "active_parameters_count": len(ACTIVE_PARAMETERS),
            "excluded_parameters": EXCLUDED_PARAMETERS,
            "prakriti_distribution": prakriti_dist,
            "vikriti_distribution": vikriti_dist
        }


if __name__ == "__main__":
    generator = AYUSHDatasetGeneratorEngine()
    print("Generating synthetic AYUSH patient dataset with latent archetype alignment (N=500)...")
    stats = generator.generate_dataset(num_samples=500)
    print("Dataset generation complete!")
    print(json.dumps(stats, indent=2))
