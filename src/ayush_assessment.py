import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


PRAKRITI_MAPPING_FILE = DATA_DIR / "prakriti_mapping.json"
SARA_MAPPING_FILE = DATA_DIR / "sara_mapping.json"
SAMHANANA_MAPPING_FILE = DATA_DIR / "samhanana_mapping.json"
PRAMANA_MAPPING_FILE = DATA_DIR / "pramana_mapping.json"
SATMYA_MAPPING_FILE = DATA_DIR / "satmya_mapping.json"
SATTVA_MAPPING_FILE = DATA_DIR / "sattva_mapping.json"
AHARA_SHAKTI_MAPPING_FILE = DATA_DIR / "ahara_shakti_mapping.json"
VYAYAMA_SHAKTI_MAPPING_FILE = DATA_DIR / "vyayama_shakti_mapping.json"
VAYA_MAPPING_FILE = DATA_DIR / "vaya_mapping.json"


# ============================================================
# Generic JSON loader
# ============================================================

def load_mapping(file_path, parameter):
    """
    Load a parameter mapping JSON file.
    """

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    return data[parameter]["questions"]


# ============================================================
# Generic confidence helper
# ============================================================

def average_confidence(confidences):
    """
    Calculate average confidence.
    """

    if not confidences:
        return None

    return round(
        sum(confidences) / len(confidences),
        3
    )


# ============================================================
# PRAKRITI
# ============================================================

def assess_prakriti(evidence):
    """
    Calculate preliminary Prakriti assessment.
    """

    mapping = load_mapping(
        PRAKRITI_MAPPING_FILE,
        "prakriti"
    )

    scores = {
        "vata": 0.0,
        "pitta": 0.0,
        "kapha": 0.0
    }

    valid_answers = 0

    for item in evidence:

        question_id = item["questionId"]
        answer = item["answer"]

        if question_id not in mapping:
            continue

        question_mapping = mapping[question_id]

        if answer not in question_mapping:
            continue

        answer_scores = question_mapping[answer]

        confidence = item.get(
            "confidence",
            1.0
        )

        scores["vata"] += (
            answer_scores.get("vata", 0.0)
            * confidence
        )

        scores["pitta"] += (
            answer_scores.get("pitta", 0.0)
            * confidence
        )

        scores["kapha"] += (
            answer_scores.get("kapha", 0.0)
            * confidence
        )

        if sum(answer_scores.values()) > 0:
            valid_answers += 1

    if valid_answers == 0:

        return {
            "parameter": "prakriti",
            "value": None,
            "evidence": evidence,
            "confidence": None
        }

    dominant_dosha = max(
        scores,
        key=scores.get
    )

    total_score = sum(
        scores.values()
    )

    confidence = (
        scores[dominant_dosha] / total_score
        if total_score > 0
        else None
    )

    return {
        "parameter": "prakriti",
        "value": dominant_dosha,
        "evidence": evidence,
        "confidence": round(
            confidence,
            3
        ) if confidence is not None else None
    }


# ============================================================
# VIKRITI
# ============================================================

def assess_vikriti(evidence):
    """
    Structure current Vikriti-related evidence.

    Free-text answers are retained as evidence.
    """

    assessment = {
        "current_changes": [],
        "primary_concern": None,
        "onset": None,
        "progression": None
    }

    confidences = []

    for item in evidence:

        question_id = item["questionId"]
        answer = item["answer"]

        confidence = item.get(
            "confidence",
            1.0
        )

        confidences.append(
            confidence
        )

        if question_id == "vikriti_q01":

            assessment[
                "current_changes"
            ].append(answer)

        elif question_id == "vikriti_q02":

            assessment[
                "primary_concern"
            ] = answer

        elif question_id == "vikriti_q03":

            assessment[
                "onset"
            ] = answer

        elif question_id == "vikriti_q04":

            assessment[
                "progression"
            ] = answer

    return {
        "parameter": "vikriti",
        "value": assessment,
        "evidence": evidence,
        "confidence": average_confidence(
            confidences
        )
    }


# ============================================================
# GENERIC SCALED PARAMETER ASSESSMENT
# ============================================================

def assess_scaled_parameter(
    parameter,
    evidence,
    mapping_file,
    value_key,
    thresholds,
    mapping_parameter=None
):
    """
    Generic assessment for parameters whose mapping
    produces a numerical score.

    The mapping must contain:

    {
        "questionId": {
            "answer": {
                "score_name": number
            }
        }
    }

    Free-text answers remain evidence and are not
    assigned arbitrary scores.
    """

    if mapping_parameter is None:
        mapping_parameter = parameter

    mapping = load_mapping(
        mapping_file,
        mapping_parameter
    )

    weighted_scores = []
    confidences = []

    for item in evidence:

        question_id = item["questionId"]
        answer = item["answer"]

        confidence = item.get(
            "confidence",
            1.0
        )

        if question_id not in mapping:
            continue

        question_mapping = mapping[
            question_id
        ]

        if answer not in question_mapping:
            continue

        answer_mapping = question_mapping[
            answer
        ]

        score = answer_mapping.get(
            value_key
        )

        if score is None:
            continue

        weighted_scores.append(
            score * confidence
        )

        confidences.append(
            confidence
        )

    if not weighted_scores:

        return {
            "parameter": parameter,
            "value": None,
            "evidence": evidence,
            "confidence": None
        }

    total_confidence = sum(
        confidences
    )

    weighted_average = (
        sum(weighted_scores)
        / total_confidence
        if total_confidence > 0
        else None
    )

    value = None

    if weighted_average is not None:

        for minimum, maximum, label in thresholds:

            if (
                weighted_average >= minimum
                and weighted_average < maximum
            ):
                value = label
                break

    return {
        "parameter": parameter,
        "value": value,
        "evidence": evidence,
        "confidence": average_confidence(
            confidences
        )
    }


# ============================================================
# SARA
# ============================================================

def assess_sara(evidence):

    return assess_scaled_parameter(
        parameter="sara",
        evidence=evidence,
        mapping_file=SARA_MAPPING_FILE,
        value_key="sara",
        thresholds=[
            (0.75, 1.01, "strong"),
            (0.50, 0.75, "moderate"),
            (0.25, 0.50, "low"),
            (0.00, 0.25, "insufficient")
        ]
    )


# ============================================================
# SAMHANANA
# ============================================================

def assess_samhanana(evidence):

    mapping = load_mapping(
        SAMHANANA_MAPPING_FILE,
        "samhanana"
    )

    scores = {
        "strong": 0.0,
        "moderate": 0.0,
        "weak": 0.0
    }

    confidences = []

    for item in evidence:

        question_id = item["questionId"]
        answer = item["answer"]

        confidence = item.get(
            "confidence",
            1.0
        )

        if question_id not in mapping:
            continue

        question_mapping = mapping[
            question_id
        ]

        if answer not in question_mapping:
            continue

        answer_scores = question_mapping[
            answer
        ]

        confidences.append(
            confidence
        )

        for category in scores:

            scores[category] += (
                answer_scores.get(
                    category,
                    0.0
                )
                * confidence
            )

    if not confidences:

        return {
            "parameter": "samhanana",
            "value": None,
            "evidence": evidence,
            "confidence": None
        }

    value = max(
        scores,
        key=scores.get
    )

    return {
        "parameter": "samhanana",
        "value": value,
        "evidence": evidence,
        "confidence": average_confidence(
            confidences
        )
    }


# ============================================================
# PRAMANA
# ============================================================

def assess_pramana(evidence):

    mapping = load_mapping(
        PRAMANA_MAPPING_FILE,
        "pramana"
    )

    proportion_scores = []
    stability_scores = []

    confidences = []

    measurements = None

    for item in evidence:

        question_id = item["questionId"]
        answer = item["answer"]

        confidence = item.get(
            "confidence",
            1.0
        )

        confidences.append(
            confidence
        )

        if question_id == "pramana_q01":

            measurements = answer
            continue

        if question_id not in mapping:
            continue

        question_mapping = mapping[
            question_id
        ]

        if answer not in question_mapping:
            continue

        answer_mapping = question_mapping[
            answer
        ]

        if "proportion" in answer_mapping:

            proportion_scores.append(
                answer_mapping["proportion"]
                * confidence
            )

        if "stability" in answer_mapping:

            stability_scores.append(
                answer_mapping["stability"]
                * confidence
            )

    if not proportion_scores:

        value = None
        overall_confidence = None

    else:

        proportion_average = (
            sum(proportion_scores)
            / sum(
                item.get(
                    "confidence",
                    1.0
                )
                for item in evidence
                if item["questionId"]
                == "pramana_q02"
            )
        )

        stability_average = (
            sum(stability_scores)
            / sum(
                item.get(
                    "confidence",
                    1.0
                )
                for item in evidence
                if item["questionId"]
                == "pramana_q03"
            )
            if stability_scores
            else 0.0
        )

        combined_score = (
            proportion_average * 0.7
            + stability_average * 0.3
        )

        if combined_score >= 0.75:
            value = "balanced"

        elif combined_score >= 0.50:
            value = "mostly_balanced"

        elif combined_score >= 0.25:
            value = "somewhat_unbalanced"

        else:
            value = "insufficient"

        relevant_confidences = [
            item.get("confidence", 1.0)
            for item in evidence
            if item["questionId"]
            in (
                "pramana_q02",
                "pramana_q03"
            )
        ]

        overall_confidence = average_confidence(
            relevant_confidences
        )

    return {
        "parameter": "pramana",
        "value": value,
        "evidence": evidence,
        "confidence": overall_confidence,
        "measurements": measurements
    }


# ============================================================
# SATMYA
# ============================================================

def assess_satmya(evidence):

    mapping = load_mapping(
        SATMYA_MAPPING_FILE,
        "satmya"
    )

    tolerance_scores = []
    confidences = []

    free_text = []

    for item in evidence:

        question_id = item["questionId"]
        answer = item["answer"]

        confidence = item.get(
            "confidence",
            1.0
        )

        if question_id in (
            "satmya_q01",
            "satmya_q02",
            "satmya_q03"
        ):

            free_text.append({
                "questionId": question_id,
                "answer": answer,
                "confidence": confidence
            })

        if question_id not in mapping:
            continue

        question_mapping = mapping[
            question_id
        ]

        if answer not in question_mapping:
            continue

        answer_mapping = question_mapping[
            answer
        ]

        score = answer_mapping.get(
            "tolerance"
        )

        if score is not None:

            tolerance_scores.append(
                score * confidence
            )

            confidences.append(
                confidence
            )

    if not tolerance_scores:

        value = "insufficient"

        overall_confidence = None

    else:

        score = (
            sum(tolerance_scores)
            / sum(confidences)
        )

        if score >= 0.75:
            value = "well_tolerated"

        elif score >= 0.50:
            value = "mostly_tolerated"

        elif score >= 0.25:
            value = "sometimes_intolerant"

        else:
            value = "poorly_tolerated"

        overall_confidence = average_confidence(
            confidences
        )

    return {
        "parameter": "satmya",
        "value": value,
        "evidence": evidence,
        "confidence": overall_confidence,
        "dietary_evidence": free_text
    }


# ============================================================
# SATTVA
# ============================================================

def assess_sattva(evidence):

    mapping = load_mapping(
        SATTVA_MAPPING_FILE,
        "sattva"
    )

    scores = []
    confidences = []

    free_text = []

    for item in evidence:

        question_id = item["questionId"]
        answer = item["answer"]

        confidence = item.get(
            "confidence",
            1.0
        )

        if question_id == "sattva_q01":

            free_text.append({
                "questionId": question_id,
                "answer": answer,
                "confidence": confidence
            })

        if question_id not in mapping:
            continue

        question_mapping = mapping[
            question_id
        ]

        if answer not in question_mapping:
            continue

        score = question_mapping[
            answer
        ].get("sattva")

        if score is None:
            continue

        scores.append(
            score * confidence
        )

        confidences.append(
            confidence
        )

    if not scores:

        return {
            "parameter": "sattva",
            "value": None,
            "evidence": evidence,
            "confidence": None,
            "free_text_evidence": free_text
        }

    average_score = (
        sum(scores)
        / sum(confidences)
    )

    if average_score >= 0.75:
        value = "strong"

    elif average_score >= 0.50:
        value = "moderate"

    elif average_score >= 0.25:
        value = "low"

    else:
        value = "weak"

    return {
        "parameter": "sattva",
        "value": value,
        "evidence": evidence,
        "confidence": average_confidence(
            confidences
        ),
        "free_text_evidence": free_text
    }


# ============================================================
# AHARA SHAKTI
# ============================================================

def assess_ahara_shakti(evidence):

    mapping = load_mapping(
        AHARA_SHAKTI_MAPPING_FILE,
        "aharaShakti"
    )

    scores = []
    confidences = []

    free_text = []

    for item in evidence:

        question_id = item["questionId"]
        answer = item["answer"]

        confidence = item.get(
            "confidence",
            1.0
        )

        if question_id == "aharaShakti_q02":

            free_text.append({
                "questionId": question_id,
                "answer": answer,
                "confidence": confidence
            })

        if question_id not in mapping:
            continue

        question_mapping = mapping[
            question_id
        ]

        if answer not in question_mapping:
            continue

        score = question_mapping[
            answer
        ].get("strength")

        if score is None:
            continue

        scores.append(
            score * confidence
        )

        confidences.append(
            confidence
        )

    if not scores:

        return {
            "parameter": "aharaShakti",
            "value": None,
            "evidence": evidence,
            "confidence": None,
            "free_text_evidence": free_text
        }

    average_score = (
        sum(scores)
        / sum(confidences)
    )

    if average_score >= 0.75:
        value = "strong"

    elif average_score >= 0.50:
        value = "moderate"

    elif average_score >= 0.25:
        value = "low"

    else:
        value = "weak"

    return {
        "parameter": "aharaShakti",
        "value": value,
        "evidence": evidence,
        "confidence": average_confidence(
            confidences
        ),
        "free_text_evidence": free_text
    }


# ============================================================
# VYAYAMA SHAKTI
# ============================================================

def assess_vyayama_shakti(evidence):

    mapping = load_mapping(
        VYAYAMA_SHAKTI_MAPPING_FILE,
        "vyayamaShakti"
    )

    scores = []
    confidences = []

    free_text = []

    for item in evidence:

        question_id = item["questionId"]
        answer = item["answer"]

        confidence = item.get(
            "confidence",
            1.0
        )

        if question_id == "vyayamaShakti_q02":

            free_text.append({
                "questionId": question_id,
                "answer": answer,
                "confidence": confidence
            })

        if question_id not in mapping:
            continue

        question_mapping = mapping[
            question_id
        ]

        if answer not in question_mapping:
            continue

        score = question_mapping[
            answer
        ].get("capacity")

        if score is None:
            continue

        scores.append(
            score * confidence
        )

        confidences.append(
            confidence
        )

    if not scores:

        return {
            "parameter": "vyayamaShakti",
            "value": None,
            "evidence": evidence,
            "confidence": None,
            "free_text_evidence": free_text
        }

    average_score = (
        sum(scores)
        / sum(confidences)
    )

    if average_score >= 0.75:
        value = "strong"

    elif average_score >= 0.50:
        value = "moderate"

    elif average_score >= 0.25:
        value = "low"

    else:
        value = "weak"

    return {
        "parameter": "vyayamaShakti",
        "value": value,
        "evidence": evidence,
        "confidence": average_confidence(
            confidences
        ),
        "free_text_evidence": free_text
    }


# ============================================================
# VAYA
# ============================================================

def assess_vaya(evidence):

    mapping = load_mapping(
        VAYA_MAPPING_FILE,
        "vaya"
    )

    age = None
    stage = None
    age_confidence = None
    stage_confidence = None

    age_evidence = None
    stage_evidence = None

    age_groups = []

    for item in evidence:

        question_id = item["questionId"]
        answer = item["answer"]

        confidence = item.get(
            "confidence",
            1.0
        )

        if question_id == "vaya_q01":

            try:
                age = float(answer)
                age_confidence = confidence
                age_evidence = answer
            except (
                ValueError,
                TypeError
            ):
                pass

        elif question_id == "vaya_q02":

            if answer in mapping.get(
                "vaya_q02",
                {}
            ):

                stage = mapping[
                    "vaya_q02"
                ][answer].get(
                    "stage"
                )

                stage_confidence = confidence
                stage_evidence = answer

    if age is not None:

        if age < 13:
            age_group = "childhood"

        elif age < 20:
            age_group = "adolescence"

        elif age < 40:
            age_group = "adulthood"

        elif age < 60:
            age_group = "middle_age"

        else:
            age_group = "older_adulthood"

    else:

        age_group = None

    if stage is not None:

        value = stage

    else:

        value = age_group

    relevant_confidences = []

    if age_confidence is not None:
        relevant_confidences.append(
            age_confidence
        )

    if stage_confidence is not None:
        relevant_confidences.append(
            stage_confidence
        )

    return {
        "parameter": "vaya",
        "value": value,
        "evidence": evidence,
        "confidence": average_confidence(
            relevant_confidences
        ),
        "age": age,
        "age_group": age_group,
        "stage": stage,
        "age_evidence": age_evidence,
        "stage_evidence": stage_evidence
    }


# ============================================================
# MAIN ROUTER
# ============================================================

def assess_ayush_parameter(
    parameter,
    evidence
):

    if parameter == "prakriti":

        return assess_prakriti(
            evidence
        )

    elif parameter == "vikriti":

        return assess_vikriti(
            evidence
        )

    elif parameter == "sara":

        return assess_sara(
            evidence
        )

    elif parameter == "samhanana":

        return assess_samhanana(
            evidence
        )

    elif parameter == "pramana":

        return assess_pramana(
            evidence
        )

    elif parameter == "satmya":

        return assess_satmya(
            evidence
        )

    elif parameter == "sattva":

        return assess_sattva(
            evidence
        )

    elif parameter == "aharaShakti":

        return assess_ahara_shakti(
            evidence
        )

    elif parameter == "vyayamaShakti":

        return assess_vyayama_shakti(
            evidence
        )

    elif parameter == "vaya":

        return assess_vaya(
            evidence
        )

    else:

        raise ValueError(
            f"Unsupported AYUSH parameter: {parameter}"
        )


# ============================================================
# ASSESS ALL 10
# ============================================================

def assess_all(
    answers_by_parameter
):
    """
    Assess every supplied AYUSH parameter.

    Supported parameters:

    1. prakriti
    2. vikriti
    3. sara
    4. samhanana
    5. pramana
    6. satmya
    7. sattva
    8. aharaShakti
    9. vyayamaShakti
    10. vaya

    Agni and Koshtha are intentionally excluded.
    """

    results = {}

    supported_parameters = [
        "prakriti",
        "vikriti",
        "sara",
        "samhanana",
        "pramana",
        "satmya",
        "sattva",
        "aharaShakti",
        "vyayamaShakti",
        "vaya"
    ]

    for parameter in supported_parameters:

        evidence = answers_by_parameter.get(
            parameter,
            []
        )

        results[parameter] = (
            assess_ayush_parameter(
                parameter=parameter,
                evidence=evidence
            )
        )

    return results


# ============================================================
# TEST DATA
# ============================================================

if __name__ == "__main__":

    patient_answers = {

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

    results = assess_all(
        patient_answers
    )

    print(
        "10-PARAMETER AYUSH ASSESSMENT:"
    )

    print(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False
        )
    )