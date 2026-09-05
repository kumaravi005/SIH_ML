import json
from pathlib import Path

from ayush_processor import process_ayush_answers
from ayush_assessment import assess_all
from clinical_extractor import process_unstructured_input


# ==========================================================
# FILE PATHS
# ==========================================================

PATIENT_CASE_FILE = (
    Path(__file__).resolve().parent.parent
    / "schemas"
    / "patient_case.json"
)

OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent
    / "output"
    / "patient_case_output.json"
)


# ==========================================================
# LOAD BASE PATIENT CASE
# ==========================================================

def load_patient_case():
    """
    Load the base patient case JSON.
    """

    with open(
        PATIENT_CASE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ==========================================================
# ADD AYUSH RESULTS
# ==========================================================

def add_ayush_results(
    patient_case,
    ayush_results
):
    """
    Add calculated AYUSH assessment results
    to the patient case.

    All fields returned by ayush_assessment.py
    are preserved automatically.

    This means fields such as:

        value
        evidence
        confidence
        measurements
        dietary_evidence
        free_text_evidence
        age
        age_group
        stage
        age_evidence
        stage_evidence

    are retained whenever they are present.
    """

    dashavidha = patient_case[
        "ayushAssessment"
    ][
        "dashavidhaPariksha"
    ]

    for parameter, result in ayush_results.items():

        # --------------------------------------------------
        # Validate parameter
        # --------------------------------------------------

        if parameter not in dashavidha:

            raise ValueError(
                f"Unknown Dashavidha parameter: {parameter}"
            )

        # --------------------------------------------------
        # Copy every field returned by the
        # AYUSH assessment engine.
        #
        # This prevents additional fields from
        # being lost.
        # --------------------------------------------------

        for field, value in result.items():

            # "parameter" is metadata used by the
            # assessment engine and is not required
            # inside the patient-case parameter object.
            if field == "parameter":
                continue

            dashavidha[parameter][field] = value

    return patient_case


# ==========================================================
# MERGE CLINICAL SECTIONS & AYUSH EVIDENCE
# ==========================================================

def merge_clinical_entities(patient_case, extracted_clinical):
    """
    Merge extracted clinical entities into patient_case JSON structure.
    Only overwrites empty/null fields to preserve existing clinical data.
    """
    if not extracted_clinical:
        return patient_case

    for key, value in extracted_clinical.items():
        if key in patient_case:
            if isinstance(value, dict) and isinstance(patient_case[key], dict):
                for sub_k, sub_v in value.items():
                    target = patient_case[key].get(sub_k)
                    is_template = (
                        isinstance(target, list) and len(target) == 1 and
                        isinstance(target[0], dict) and
                        all(v is None for v in target[0].values())
                    )
                    if target in (None, [], {}) or is_template:
                        patient_case[key][sub_k] = sub_v
            elif isinstance(value, list) and isinstance(patient_case[key], list):
                is_template = (
                    len(patient_case[key]) == 1 and
                    isinstance(patient_case[key][0], dict) and
                    all(v is None for v in patient_case[key][0].values() if not isinstance(v, list))
                )
                if is_template:
                    patient_case[key] = value
                else:
                    patient_case[key].extend(value)
            elif patient_case.get(key) is None:
                patient_case[key] = value

    return patient_case


def build_complete_patient_case(patient_answers=None, unstructured_inputs=None):
    """
    End-to-end multi-source patient case builder.
    Combines structured questionnaire answers with unstructured text & audio transcripts.
    """
    if patient_answers is None:
        patient_answers = {}
    else:
        # Create a deep copy of parameter answers
        patient_answers = {k: list(v) for k, v in patient_answers.items()}

    patient_case = load_patient_case()

    # Process unstructured inputs if provided
    if unstructured_inputs:
        for item in unstructured_inputs:
            text = item.get("text", "")
            source = item.get("source", "text")
            confidence = item.get("confidence", 0.90)

            extracted = process_unstructured_input(
                raw_input=text,
                source=source,
                base_confidence=confidence
            )

            # 1. Update clinical sections
            patient_case = merge_clinical_entities(patient_case, extracted.get("clinical", {}))

            # 2. Merge extracted AYUSH evidence items into questionnaire answers
            extracted_ayush = extracted.get("ayush_evidence", {})
            for param, items in extracted_ayush.items():
                if param not in patient_answers:
                    patient_answers[param] = []
                existing_qids = {e["questionId"] for e in patient_answers[param]}
                for new_item in items:
                    if new_item["questionId"] not in existing_qids:
                        patient_answers[param].append(new_item)

    # Validate and structure answers
    validated_answers = process_ayush_answers(patient_answers)

    # Extract evidence lists
    evidence_by_parameter = {
        param: res["evidence"] for param, res in validated_answers.items()
    }

    # Calculate AYUSH assessments
    ayush_assessments = assess_all(evidence_by_parameter)

    # Add AYUSH results to patient case
    patient_case = add_ayush_results(patient_case, ayush_assessments)

    return patient_case


# ==========================================================
# SAVE PATIENT CASE
# ==========================================================

def save_patient_case(
    patient_case
):
    """
    Save the final patient case to
    the output JSON file.
    """

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            patient_case,
            file,
            indent=2,
            ensure_ascii=False
        )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    # ======================================================
    # RAW PATIENT ANSWERS
    # ======================================================

    patient_answers = {

        # --------------------------------------------------
        # 1. PRAKRITI
        # --------------------------------------------------

        "prakriti": [

            {
                "questionId": "prakriti_q01",
                "answer": "Thin/light build",
                "source": "questionnaire",
                "confidence": 1.0
            },

            {
                "questionId": "prakriti_q02",
                "answer": "Usually dry",
                "source": "questionnaire",
                "confidence": 1.0
            }
        ],


        # --------------------------------------------------
        # 2. VIKRITI
        # --------------------------------------------------

        "vikriti": [

            {
                "questionId": "vikriti_q01",
                "answer": "Headache and fatigue",
                "source": "audio_transcript",
                "confidence": 0.91
            },

            {
                "questionId": "vikriti_q04",
                "answer": "Getting worse",
                "source": "text",
                "confidence": 0.96
            }
        ],


        # --------------------------------------------------
        # 3. SARA
        # --------------------------------------------------

        "sara": [

            {
                "questionId": "sara_q01",
                "answer": "Moderate overall strength",
                "source": "questionnaire",
                "confidence": 0.84
            },

            {
                "questionId": "sara_q02",
                "answer": "Generally healthy appearance",
                "source": "questionnaire",
                "confidence": 0.85
            },

            {
                "questionId": "sara_q03",
                "answer": "Mostly",
                "source": "questionnaire",
                "confidence": 0.90
            }
        ],


        # --------------------------------------------------
        # 4. SAMHANANA
        # --------------------------------------------------

        "samhanana": [

            {
                "questionId": "samhanana_q01",
                "answer": "Well-proportioned and well-developed",
                "source": "questionnaire",
                "confidence": 0.90
            },

            {
                "questionId": "samhanana_q02",
                "answer": "Well developed",
                "source": "questionnaire",
                "confidence": 0.88
            },

            {
                "questionId": "samhanana_q03",
                "answer": "Yes",
                "source": "questionnaire",
                "confidence": 0.92
            }
        ],


        # --------------------------------------------------
        # 5. PRAMANA
        # --------------------------------------------------

        "pramana": [

            {
                "questionId": "pramana_q01",
                "answer": "Height: 170 cm, Weight: 65 kg",
                "source": "questionnaire",
                "confidence": 0.95
            },

            {
                "questionId": "pramana_q02",
                "answer": "Generally proportionate",
                "source": "questionnaire",
                "confidence": 0.90
            },

            {
                "questionId": "pramana_q03",
                "answer": "No significant change",
                "source": "questionnaire",
                "confidence": 0.88
            }
        ],


        # --------------------------------------------------
        # 6. SATMYA
        # --------------------------------------------------

        "satmya": [

            {
                "questionId": "satmya_q01",
                "answer": "Mostly home-cooked food",
                "source": "questionnaire",
                "confidence": 0.90
            },

            {
                "questionId": "satmya_q02",
                "answer": "I tolerate my usual meals well",
                "source": "questionnaire",
                "confidence": 0.88
            },

            {
                "questionId": "satmya_q03",
                "answer": "Very spicy food sometimes causes discomfort",
                "source": "questionnaire",
                "confidence": 0.87
            },

            {
                "questionId": "satmya_q04",
                "answer": "Very well",
                "source": "questionnaire",
                "confidence": 0.90
            }
        ],


        # --------------------------------------------------
        # 7. SATTVA
        # --------------------------------------------------

        "sattva": [

            {
                "questionId": "sattva_q01",
                "answer": "I usually stay calm and try to solve the problem",
                "source": "questionnaire",
                "confidence": 0.90
            },

            {
                "questionId": "sattva_q02",
                "answer": "Usually calm and focused",
                "source": "questionnaire",
                "confidence": 0.90
            },

            {
                "questionId": "sattva_q03",
                "answer": "Generally confident",
                "source": "questionnaire",
                "confidence": 0.88
            },

            {
                "questionId": "sattva_q04",
                "answer": "Rarely",
                "source": "questionnaire",
                "confidence": 0.90
            }
        ],


        # --------------------------------------------------
        # 8. AHARA SHAKTI
        # --------------------------------------------------

        "aharaShakti": [

            {
                "questionId": "aharaShakti_q01",
                "answer": "Good",
                "source": "questionnaire",
                "confidence": 0.90
            },

            {
                "questionId": "aharaShakti_q02",
                "answer": "I can comfortably eat a normal meal",
                "source": "questionnaire",
                "confidence": 0.85
            },

            {
                "questionId": "aharaShakti_q03",
                "answer": "Comfortable and satisfied",
                "source": "questionnaire",
                "confidence": 0.90
            },

            {
                "questionId": "aharaShakti_q04",
                "answer": "Mostly regularly",
                "source": "questionnaire",
                "confidence": 0.90
            }
        ],


        # --------------------------------------------------
        # 9. VYAYAMA SHAKTI
        # --------------------------------------------------

        "vyayamaShakti": [

            {
                "questionId": "vyayamaShakti_q01",
                "answer": "Moderately active",
                "source": "questionnaire",
                "confidence": 0.90
            },

            {
                "questionId": "vyayamaShakti_q02",
                "answer": "Around 30 minutes",
                "source": "questionnaire",
                "confidence": 0.85
            },

            {
                "questionId": "vyayamaShakti_q03",
                "answer": "Mildly tired but recover quickly",
                "source": "questionnaire",
                "confidence": 0.90
            },

            {
                "questionId": "vyayamaShakti_q04",
                "answer": "Quickly",
                "source": "questionnaire",
                "confidence": 0.90
            }
        ],


        # --------------------------------------------------
        # 10. VAYA
        # --------------------------------------------------

        "vaya": [

            {
                "questionId": "vaya_q01",
                "answer": 35,
                "source": "questionnaire",
                "confidence": 1.0
            },

            {
                "questionId": "vaya_q02",
                "answer": "Adulthood",
                "source": "questionnaire",
                "confidence": 0.90
            },

            {
                "questionId": "vaya_q03",
                "answer": "No major age-related changes",
                "source": "questionnaire",
                "confidence": 0.85
            }
        ]
    }


    # ======================================================
    # UNSTRUCTURED AUDIO & TEXT INPUTS
    # ======================================================

    unstructured_inputs = [
        {
            "text": "I have had headache and fatigue for 3 days and it is getting worse. Patient is taking Paracetamol 500 mg. Allergic to Penicillin.",
            "source": "audio_transcript",
            "confidence": 0.91
        }
    ]

    # ======================================================
    # BUILD COMPLETE MULTI-SOURCE PATIENT CASE
    # ======================================================

    updated_case = build_complete_patient_case(
        patient_answers=patient_answers,
        unstructured_inputs=unstructured_inputs
    )

    # Save final patient case
    save_patient_case(updated_case)

    # Display final patient case
    print("FINAL PATIENT CASE:")
    print(json.dumps(updated_case, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {OUTPUT_FILE}")