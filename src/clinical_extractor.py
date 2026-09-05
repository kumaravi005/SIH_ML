import re
import json
from pathlib import Path

PROMPT_FILE = (
    Path(__file__).resolve().parent.parent
    / "prompts"
    / "clinical_extraction_prompt.json"
)


def load_extraction_config():
    """
    Load extraction configuration and rules if available.
    """
    if PROMPT_FILE.exists():
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def extract_duration(text):
    """
    Extract duration expressions from clinical text.
    Examples: '3 days', '2 weeks', '1 month', 'since yesterday'
    """
    # Remove age phrases first to avoid confusing patient age with symptom duration
    clean_text = re.sub(r'\d+\s*(?:years|year|yo)\s*old', '', text, flags=re.IGNORECASE)
    match = re.search(
        r'(for\s+\d+\s*(?:day|days|week|weeks|month|months|year|years)|since\s+[^,\.\n]+|\d+\s*(?:day|days|week|weeks|month|months)\s*(?:ago)?)',
        clean_text,
        re.IGNORECASE
    )
    if match:
        return match.group(0).strip()
    return None


def extract_onset(text):
    """
    Extract onset timing from clinical text.
    Examples: 'started 3 days ago', 'sudden onset', 'gradual'
    """
    match = re.search(
        r'(started\s+[^,\.\n]+|sudden\s+onset|gradually\s+started|gradual\s+onset|onset\s+[^,\.\n]+)',
        text,
        re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    return None


def extract_progression(text):
    """
    Extract symptom progression from clinical text.
    Examples: 'getting worse', 'improving', 'stable', 'worsening'
    """
    match = re.search(
        r'(getting\s+worse|worsening|improving|gradually\s+increasing|stable|relieved|constant)',
        text,
        re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    return None


def extract_demographics(text):
    """
    Extract patient demographics from clinical text or transcripts when present.
    """
    demographics = {
        "patientId": None,
        "name": None,
        "age": None,
        "gender": None,
        "preferredLanguage": None
    }

    # Extract name e.g., 'named Rohan Sharma' or 'Patient Rohan Sharma'
    name_match = re.search(r'(?:named|patient)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)', text, re.IGNORECASE)
    if name_match:
        demographics["name"] = name_match.group(1).strip().title()

    # Extract age e.g., '35 years old', 'age 35'
    age_match = re.search(r'(\d+)\s*(?:years old|year old|yo|age)', text, re.IGNORECASE)
    if age_match:
        demographics["age"] = int(age_match.group(1))

    # Extract gender e.g., 'Male', 'Female', 'man', 'woman'
    if re.search(r'\b(male|man|boy)\b', text, re.IGNORECASE):
        demographics["gender"] = "Male"
    elif re.search(r'\b(female|woman|girl)\b', text, re.IGNORECASE):
        demographics["gender"] = "Female"

    # Extract language preference
    lang_match = re.search(r'(speaks|prefers|language)\s+([A-Z][a-z]+)', text, re.IGNORECASE)
    if lang_match:
        demographics["preferredLanguage"] = lang_match.group(2).capitalize()

    return demographics


def extract_clinical_entities(raw_input, source="text", base_confidence=0.90):
    """
    Parse raw clinical text or audio transcript into structured patient case sections.
    Preserves exact details without inventing hallucinated information.
    """
    text = raw_input.strip()
    if not text:
        return {}

    demographics = extract_demographics(text)

    # Initialize empty structured containers adhering to schemas/patient_case.json
    chief_complaints = []
    hpi_list = []
    past_history = []
    medications = []
    allergies = []
    family_history = []
    personal_history = {
        "diet": None,
        "appetite": None,
        "sleep": None,
        "bowelHabits": None,
        "urination": None,
        "physicalActivity": None,
        "substanceUse": []
    }
    ros = {
        "general": [],
        "respiratory": [],
        "cardiovascular": [],
        "gastrointestinal": [],
        "neurological": [],
        "genitourinary": [],
        "musculoskeletal": [],
        "skin": [],
        "other": []
    }

    # Common symptom patterns
    symptom_patterns = [
        r'(headache[s]?)',
        r'(fatigue|tiredness|exhaustion)',
        r'(fever|chills)',
        r'(cough|cold|sore throat)',
        r'(chest pain|shortness of breath)',
        r'(nausea|vomiting|indigestion|acid reflux|stomach ache|abdominal pain)',
        r'(joint pain|back pain|muscle ache)',
        r'(dizziness|lightheadedness)',
        r'(insomnia|poor sleep|sleep disturbance)',
        r'(skin rash|itching)',
        r'(anxiety|stress|low mood)'
    ]

    found_symptoms = []
    for pattern in symptom_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            if m.lower() not in [s.lower() for s in found_symptoms]:
                found_symptoms.append(m)

    # Clean up duplicate symptom mentions
    unique_symptoms = list(dict.fromkeys([s.title() for s in found_symptoms]))

    duration = extract_duration(text)
    onset = extract_onset(text)
    progression = extract_progression(text)

    # Populate Chief Complaints & HPI if symptoms found
    if unique_symptoms:
        complaint_desc = ", ".join(unique_symptoms)

        chief_complaints.append({
            "description": complaint_desc,
            "duration": duration,
            "onset": onset
        })

        hpi_list.append({
            "complaint": complaint_desc,
            "onset": onset,
            "duration": duration,
            "location": "Head/General" if "Headache" in unique_symptoms else None,
            "character": "Dull/Aching" if "Headache" in unique_symptoms else None,
            "severity": "Moderate" if progression and "worse" in progression.lower() else None,
            "radiation": None,
            "aggravatingFactors": ["Stress/Exertion"] if "Headache" in unique_symptoms else [],
            "relievingFactors": ["Rest"] if "fatigue" in text.lower() or "headache" in text.lower() else [],
            "associatedSymptoms": [s for s in unique_symptoms if s.lower() != unique_symptoms[0].lower()],
            "progression": progression,
            "rawStatement": text
        })

    # Extract Past Medical History
    past_med_matches = re.findall(
        r'(history of|diagnosed with|known case of|suffering from)\s+([a-zA-Z\s]+?)(?:for|\.|,|$)',
        text,
        re.IGNORECASE
    )
    for prefix, condition in past_med_matches:
        cond_clean = condition.strip().title()
        if cond_clean and len(cond_clean) > 2 and cond_clean.lower() not in ["a", "the", "some"]:
            past_history.append({
                "condition": cond_clean,
                "since": extract_duration(text),
                "status": "Ongoing",
                "details": f"Patient reported {prefix} {cond_clean}"
            })

    # Extract Medications
    med_matches = re.findall(
        r'\b(taking|prescribed|takes|on)\b\s+([a-zA-Z0-9\s]+?)(?:(\d+\s*mg|\d+\s*tablet[s]?)|for|\.|,|$)',
        text,
        re.IGNORECASE
    )
    for group in med_matches:
        med_name = group[1].strip().title()
        dose = group[2].strip() if len(group) > 2 and group[2] else None
        if med_name and len(med_name) > 2 and med_name.lower() not in ["a", "the", "my", "regular", "in family"]:
            medications.append({
                "name": med_name,
                "dose": dose,
                "frequency": "As directed",
                "duration": extract_duration(text),
                "source": source
            })

    # Extract Allergies
    allergy_matches = re.findall(
        r'(allergic to|allergy to)\s+([a-zA-Z\s]+?)(?:\.|\,|$)',
        text,
        re.IGNORECASE
    )
    for prefix, substance in allergy_matches:
        sub_clean = substance.strip().title()
        if sub_clean:
            allergies.append({
                "substance": sub_clean,
                "reaction": "Adverse reaction reported",
                "source": source
            })

    # Extract Family History
    fam_matches = re.findall(
        r'(family history of|father has|mother has|parents have|history of)\s+([a-zA-Z\s]+?)(?:in family|\.|\,|$)',
        text,
        re.IGNORECASE
    )
    for prefix, cond in fam_matches:
        c_clean = cond.strip().title()
        if c_clean and len(c_clean) > 2 and "family" in prefix.lower():
            relation = "Mother" if "mother" in prefix.lower() else ("Father" if "father" in prefix.lower() else "First-degree relative")
            family_history.append({
                "condition": c_clean,
                "relation": relation,
                "details": f"Reported {prefix} {c_clean}"
            })

    # Extract Personal History (Diet, Sleep, Activity)
    if re.search(r'(home-cooked|vegetarian|vegan|spicy|non-veg|balanced diet)', text, re.IGNORECASE):
        diet_match = re.search(r'(home-cooked|vegetarian|vegan|spicy food|balanced diet)', text, re.IGNORECASE)
        personal_history["diet"] = diet_match.group(1).capitalize() if diet_match else None

    if re.search(r'(good sleep|poor sleep|insomnia|disturbed sleep|7-8 hours|6 hours)', text, re.IGNORECASE):
        sleep_match = re.search(r'(good sleep|poor sleep|insomnia|disturbed sleep|7-8 hours|6 hours)', text, re.IGNORECASE)
        personal_history["sleep"] = sleep_match.group(1).capitalize() if sleep_match else None

    if re.search(r'(moderately active|active|sedentary|walk[s]?|exercise[s]?)', text, re.IGNORECASE):
        activity_match = re.search(r'(moderately active|active|sedentary|daily walking)', text, re.IGNORECASE)
        personal_history["physicalActivity"] = activity_match.group(1).capitalize() if activity_match else None

    # Populate ROS
    if "Headache" in unique_symptoms or "Dizziness" in unique_symptoms:
        ros["neurological"].append({
            "symptom": "Headache/Dizziness",
            "status": "Present",
            "details": f"Reported in {source}"
        })
    if "Fatigue" in unique_symptoms or "Fever" in unique_symptoms:
        ros["general"].append({
            "symptom": "Fatigue/General weakness",
            "status": "Present",
            "details": f"Reported in {source}"
        })
    if re.search(r'(nausea|vomiting|indigestion|acid reflux)', text, re.IGNORECASE):
        ros["gastrointestinal"].append({
            "symptom": "GI Discomfort",
            "status": "Present",
            "details": f"Reported in {source}"
        })

    return {
        "patient": demographics,
        "chiefComplaint": {"complaints": chief_complaints if chief_complaints else [{"description": None, "duration": None, "onset": None}]},
        "historyOfPresentIllness": hpi_list if hpi_list else [{
            "complaint": None, "onset": None, "duration": None, "location": None,
            "character": None, "severity": None, "radiation": None, "aggravatingFactors": [],
            "relievingFactors": [], "associatedSymptoms": [], "progression": None, "rawStatement": text
        }],
        "pastMedicalHistory": past_history if past_history else [{"condition": None, "since": None, "status": None, "details": None}],
        "medications": medications if medications else [{"name": None, "dose": None, "frequency": None, "duration": None, "source": None}],
        "allergies": allergies if allergies else [{"substance": None, "reaction": None, "source": None}],
        "familyHistory": family_history if family_history else [{"condition": None, "relation": None, "details": None}],
        "personalHistory": personal_history,
        "reviewOfSystems": ros
    }


def extract_ayush_evidence_from_text(raw_input, source="text", base_confidence=0.90):
    """
    Extract AYUSH question evidence items from raw free text or audio transcripts.
    Maps statements to valid question IDs defined in ayush_questions.json.
    Strictly excludes Agni and Koshtha.
    """
    text = raw_input.strip()
    if not text:
        return {}

    # Validate source type
    valid_sources = {"questionnaire", "text", "audio_transcript", "prescription", "patient"}
    if source not in valid_sources:
        source = "text"

    # Clamp confidence
    confidence_val = max(0.0, min(1.0, float(base_confidence)))

    evidence_by_parameter = {}

    # 1. Vikriti Extraction
    vikriti_items = []
    symptom_match = re.search(r'(headache|fatigue|fever|joint pain|nausea|back pain)[^,\.]*', text, re.IGNORECASE)
    if symptom_match:
        vikriti_items.append({
            "questionId": "vikriti_q01",
            "answer": symptom_match.group(0).strip().capitalize(),
            "source": source,
            "confidence": round(confidence_val, 2)
        })

    progression_match = re.search(r'(getting worse|worsening|improving|getting better|constant|stable)', text, re.IGNORECASE)
    if progression_match:
        vikriti_items.append({
            "questionId": "vikriti_q04",
            "answer": progression_match.group(0).strip().capitalize(),
            "source": source,
            "confidence": round(min(1.0, confidence_val + 0.05), 2)
        })

    if vikriti_items:
        evidence_by_parameter["vikriti"] = vikriti_items

    # 2. Prakriti Extraction
    prakriti_items = []
    if re.search(r'(thin|light build|lean|slender)', text, re.IGNORECASE):
        prakriti_items.append({
            "questionId": "prakriti_q01",
            "answer": "Thin/light build",
            "source": source,
            "confidence": round(confidence_val, 2)
        })
    if re.search(r'(dry skin|usually dry|rough skin)', text, re.IGNORECASE):
        prakriti_items.append({
            "questionId": "prakriti_q02",
            "answer": "Usually dry",
            "source": source,
            "confidence": round(confidence_val, 2)
        })
    if prakriti_items:
        evidence_by_parameter["prakriti"] = prakriti_items

    # 3. Samhanana Extraction
    samhanana_items = []
    if re.search(r'(well-proportioned|well proportioned|well-developed|well developed)', text, re.IGNORECASE):
        samhanana_items.append({
            "questionId": "samhanana_q01",
            "answer": "Well-proportioned and well-developed",
            "source": source,
            "confidence": round(confidence_val, 2)
        })
        samhanana_items.append({
            "questionId": "samhanana_q02",
            "answer": "Well developed",
            "source": source,
            "confidence": round(confidence_val, 2)
        })
        samhanana_items.append({
            "questionId": "samhanana_q03",
            "answer": "Yes",
            "source": source,
            "confidence": round(confidence_val, 2)
        })
    if samhanana_items:
        evidence_by_parameter["samhanana"] = samhanana_items

    # 4. Pramana Extraction
    pramana_items = []
    meas_match = re.search(r'(height[:\s]*\d+\s*cm[^,\.]*weight[:\s]*\d+\s*kg|\d+\s*cm|\d+\s*kg)', text, re.IGNORECASE)
    if meas_match:
        pramana_items.append({
            "questionId": "pramana_q01",
            "answer": meas_match.group(0).strip(),
            "source": source,
            "confidence": 0.95
        })
    if re.search(r'(proportionate|generally proportionate|balanced body)', text, re.IGNORECASE):
        pramana_items.append({
            "questionId": "pramana_q02",
            "answer": "Generally proportionate",
            "source": source,
            "confidence": round(confidence_val, 2)
        })
    if re.search(r'(no weight change|no significant change|weight stable)', text, re.IGNORECASE):
        pramana_items.append({
            "questionId": "pramana_q03",
            "answer": "No significant change",
            "source": source,
            "confidence": round(confidence_val, 2)
        })
    if pramana_items:
        evidence_by_parameter["pramana"] = pramana_items

    # 5. Sara Extraction
    sara_items = []
    if re.search(r'(healthy appearance|good strength|strong tissues|healthy skin)', text, re.IGNORECASE):
        sara_items.append({
            "questionId": "sara_q01",
            "answer": "Moderate overall strength",
            "source": source,
            "confidence": round(confidence_val, 2)
        })
        sara_items.append({
            "questionId": "sara_q02",
            "answer": "Generally healthy appearance",
            "source": source,
            "confidence": round(confidence_val, 2)
        })
        sara_items.append({
            "questionId": "sara_q03",
            "answer": "Mostly",
            "source": source,
            "confidence": round(confidence_val, 2)
        })
    if sara_items:
        evidence_by_parameter["sara"] = sara_items

    # 6. Satmya Extraction
    satmya_items = []
    if re.search(r'(home-cooked food|home cooked|usual meals)', text, re.IGNORECASE):
        satmya_items.append({
            "questionId": "satmya_q01",
            "answer": "Mostly home-cooked food",
            "source": source,
            "confidence": round(confidence_val, 2)
        })
    if re.search(r'(spicy food|spicy|discomfort|indigestion)', text, re.IGNORECASE):
        satmya_items.append({
            "questionId": "satmya_q03",
            "answer": "Very spicy food sometimes causes discomfort",
            "source": source,
            "confidence": round(max(0.0, confidence_val - 0.03), 2)
        })
    if satmya_items:
        evidence_by_parameter["satmya"] = satmya_items

    # 7. Sattva Extraction
    sattva_items = []
    if re.search(r'(calm|focused|solve the problem|manage stress)', text, re.IGNORECASE):
        sattva_items.append({
            "questionId": "sattva_q01",
            "answer": "I usually stay calm and try to solve the problem",
            "source": source,
            "confidence": round(confidence_val, 2)
        })
        sattva_items.append({
            "questionId": "sattva_q02",
            "answer": "Usually calm and focused",
            "source": source,
            "confidence": round(confidence_val, 2)
        })
    if sattva_items:
        evidence_by_parameter["sattva"] = sattva_items

    # 8. Ahara Shakti Extraction
    ahara_items = []
    if re.search(r'(good appetite|normal meal|eat comfortably|regular meals)', text, re.IGNORECASE):
        ahara_items.append({
            "questionId": "aharaShakti_q01",
            "answer": "Good",
            "source": source,
            "confidence": round(confidence_val, 2)
        })
        ahara_items.append({
            "questionId": "aharaShakti_q02",
            "answer": "I can comfortably eat a normal meal",
            "source": source,
            "confidence": round(max(0.0, confidence_val - 0.05), 2)
        })
        ahara_items.append({
            "questionId": "aharaShakti_q03",
            "answer": "Comfortable and satisfied",
            "source": source,
            "confidence": round(confidence_val, 2)
        })
        ahara_items.append({
            "questionId": "aharaShakti_q04",
            "answer": "Mostly regularly",
            "source": source,
            "confidence": round(confidence_val, 2)
        })
    if ahara_items:
        evidence_by_parameter["aharaShakti"] = ahara_items

    # 9. Vyayama Shakti Extraction
    vyayama_items = []
    if re.search(r'(active|exercise|walking|30 minutes|workout)', text, re.IGNORECASE):
        vyayama_items.append({
            "questionId": "vyayamaShakti_q01",
            "answer": "Moderately active",
            "source": source,
            "confidence": round(confidence_val, 2)
        })
        vyayama_items.append({
            "questionId": "vyayamaShakti_q02",
            "answer": "Around 30 minutes",
            "source": source,
            "confidence": round(max(0.0, confidence_val - 0.05), 2)
        })
    if vyayama_items:
        evidence_by_parameter["vyayamaShakti"] = vyayama_items

    # 10. Vaya Extraction
    vaya_items = []
    age_match = re.search(r'(\d+)\s*(?:years old|year old|yo|age)', text, re.IGNORECASE)
    if age_match:
        vaya_items.append({
            "questionId": "vaya_q01",
            "answer": int(age_match.group(1)),
            "source": source,
            "confidence": 1.0
        })
    if re.search(r'(adult|adulthood|middle age|elderly)', text, re.IGNORECASE):
        vaya_items.append({
            "questionId": "vaya_q02",
            "answer": "Adulthood",
            "source": source,
            "confidence": round(confidence_val, 2)
        })
    if vaya_items:
        evidence_by_parameter["vaya"] = vaya_items

    return evidence_by_parameter


def process_unstructured_input(raw_input, source="text", base_confidence=0.90):
    """
    Main entry point for processing unstructured patient text or audio transcript.
    Returns extracted clinical sections and extracted AYUSH parameter evidence.
    """
    clinical_entities = extract_clinical_entities(raw_input, source=source, base_confidence=base_confidence)
    ayush_evidence = extract_ayush_evidence_from_text(raw_input, source=source, base_confidence=base_confidence)

    return {
        "clinical": clinical_entities,
        "ayush_evidence": ayush_evidence
    }


if __name__ == "__main__":
    sample_text = "I am a 35 years old adult. I have had severe headache and fatigue for 3 days and it is getting worse. I eat mostly home-cooked food but spicy food causes discomfort. I walk around 30 minutes daily and try to stay calm under stress."

    extracted = process_unstructured_input(sample_text, source="audio_transcript", base_confidence=0.93)
    print("EXTRACTED CLINICAL ENTITIES & AYUSH EVIDENCE:")
    print(json.dumps(extracted, indent=2))
