import json
from pathlib import Path


QUESTIONS_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "ayush_questions.json"
)


def load_questions():
    """
    Load the AYUSH question tree from ayush_questions.json.
    """

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_question(parameter, question_id):
    """
    Check whether the question exists
    and belongs to the given AYUSH parameter.
    """

    data = load_questions()
    question_tree = data["ayushQuestionTree"]

    if parameter not in question_tree:
        return False

    questions = question_tree[parameter]["questions"]

    for question in questions:
        if question["id"] == question_id:
            return True

    return False


def create_ayush_evidence(parameter, answers):
    """
    Create structured evidence for one AYUSH parameter.
    """

    evidence = []

    for item in answers:

        question_id = item["questionId"]
        answer = item["answer"]
        source = item.get("source", "patient")
        confidence = item.get("confidence", 1.0)

        if not validate_question(parameter, question_id):
            raise ValueError(
                f"Invalid question '{question_id}' "
                f"for parameter '{parameter}'"
            )
        valid_sources = {
            "questionnaire",
            "text",
            "audio_transcript",
            "prescription",
            "patient"
        }

        if source not in valid_sources:
            raise ValueError(
                f"Invalid source '{source}' for '{question_id}'"
            )
        if not isinstance(confidence, (int, float)):
            raise ValueError(
                f"Confidence for '{question_id}' "
                f"must be a number between 0 and 1"
            )

        if not 0 <= confidence <= 1:
            raise ValueError(
                f"Confidence for '{question_id}' "
                f"must be between 0 and 1"
            )

        evidence.append(
            {
                "questionId": question_id,
                "answer": answer,
                "source": source,
                "confidence": confidence
            }
        )

    return {
        "parameter": parameter,
        "evidence": evidence
    }


def process_ayush_answers(answers_by_parameter):
    """
    Process answers belonging to multiple AYUSH parameters.
    """

    results = {}

    for parameter, answers in answers_by_parameter.items():

        results[parameter] = create_ayush_evidence(
            parameter=parameter,
            answers=answers
        )

    return results


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
                "source": "patient",
                "confidence": 0.84
            }
        ]
    }

    result = process_ayush_answers(patient_answers)

    print("EVIDENCE WITH SOURCE AND CONFIDENCE:")
    print(json.dumps(result, indent=2))