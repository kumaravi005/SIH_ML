import argparse
import json
from pathlib import Path

from patient_case_builder import build_complete_patient_case, save_patient_case, OUTPUT_FILE
from ayush_explainer import generate_ayush_explainability_report
from recommendation_engine import generate_clinical_recommendation_report
from longitudinal_risk_engine import generate_longitudinal_risk_report
from patient_similarity_engine import find_similar_patient_cases
from data_quality_engine import generate_data_quality_report
from clinical_prediction_engine import predict_clinical_outcome
from eval_framework import run_full_benchmark_suite
from regimen_optimizer_engine import generate_personalized_regimen_report
from clinical_explainer_engine import AYUSHClinicalExplainerEngine


EXPLAINABILITY_OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent
    / "output"
    / "ayush_explainability_report.json"
)

EXPLAINER_SYNTHESIS_OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent
    / "output"
    / "clinical_explainer_synthesis.json"
)

RECOMMENDATION_OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent
    / "output"
    / "clinical_recommendation_output.json"
)

LONGITUDINAL_RISK_OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent
    / "output"
    / "longitudinal_risk_report.json"
)

SIMILARITY_OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent
    / "output"
    / "patient_similarity_report.json"
)

QUALITY_OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent
    / "output"
    / "data_quality_report.json"
)

PREDICTION_OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent
    / "output"
    / "clinical_outcome_prediction.json"
)

EVALUATION_OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent
    / "output"
    / "ml_evaluation_report.json"
)

REGIMEN_OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent
    / "output"
    / "personalized_regimen_report.json"
)








def run_pipeline(input_case_path=None, text_input=None, source="audio_transcript", generate_explanation=True):
    """
    Execute end-to-end ML pipeline.
    """
    patient_answers = None
    unstructured_inputs = []

    if input_case_path and Path(input_case_path).exists():
        with open(input_case_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list) and len(data) > 0:
            case = data[0]
            patient_answers = case.get("questionnaire_answers")
            unstructured_inputs = case.get("unstructured_inputs", [])
        elif isinstance(data, dict):
            patient_answers = data.get("questionnaire_answers")
            unstructured_inputs = data.get("unstructured_inputs", [])

    if text_input:
        unstructured_inputs.append({
            "text": text_input,
            "source": source,
            "confidence": 0.92
        })

    # Build patient case
    patient_case = build_complete_patient_case(
        patient_answers=patient_answers,
        unstructured_inputs=unstructured_inputs if unstructured_inputs else None
    )

    # Save output
    save_patient_case(patient_case)

    # Generate explainability report
    report = None
    if generate_explanation:
        report = generate_ayush_explainability_report(patient_case)
        with open(EXPLAINABILITY_OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    # Generate clinical recommendations report
    rec_report = generate_clinical_recommendation_report(patient_case)
    with open(RECOMMENDATION_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(rec_report, f, indent=2, ensure_ascii=False)

    # Generate longitudinal progress & risk report
    risk_report = generate_longitudinal_risk_report(patient_case)
    with open(LONGITUDINAL_RISK_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(risk_report, f, indent=2, ensure_ascii=False)

    # Generate patient similarity & case retrieval report
    sample_file = Path(__file__).resolve().parent.parent / "tests" / "sample_patient_cases.json"
    historical_cases = []
    if sample_file.exists():
        with open(sample_file, "r", encoding="utf-8") as f:
            historical_cases = json.load(f)

    sim_report = find_similar_patient_cases(patient_case, historical_cases, top_k=3)
    with open(SIMILARITY_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(sim_report, f, indent=2, ensure_ascii=False)

    # Generate data quality & anomaly validation report
    quality_report = generate_data_quality_report(patient_case)
    with open(QUALITY_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2, ensure_ascii=False)

    # Generate clinical outcome & recovery prediction report
    pred_report = predict_clinical_outcome(patient_case, adherence_level="moderate")
    with open(PREDICTION_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(pred_report, f, indent=2, ensure_ascii=False)

    # Generate ML evaluation & performance metrics report
    eval_report = run_full_benchmark_suite()
    with open(EVALUATION_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2, ensure_ascii=False)

    # Generate personalized AYUSH regimen & lifestyle optimization report
    reg_report = generate_personalized_regimen_report(patient_case, season="Vasanta")
    with open(REGIMEN_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(reg_report, f, indent=2, ensure_ascii=False)

    # Generate interactive clinical explainer & reasoning synthesis report
    explainer = AYUSHClinicalExplainerEngine()
    synthesis_report = explainer.explain_case(patient_case, practitioner_query="Explain Dosha dynamics and Ahara Shakti status.")
    with open(EXPLAINER_SYNTHESIS_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(synthesis_report, f, indent=2, ensure_ascii=False)

    return patient_case, report, rec_report, risk_report, sim_report, quality_report, pred_report, eval_report, reg_report, synthesis_report


def main():
    parser = argparse.ArgumentParser(description="SIH_ML Healthcare Pipeline & AYUSH Assessment CLI")
    parser.add_argument("--input", type=str, help="Path to input patient case JSON")
    parser.add_argument("--text", type=str, help="Raw clinical text or transcript input")
    parser.add_argument("--source", type=str, default="audio_transcript", choices=["text", "audio_transcript", "questionnaire", "prescription"])
    parser.add_argument("--no-explain", action="store_true", help="Disable generating explainability report")
    parser.add_argument("--generate-dataset", type=int, nargs="?", const=500, help="Generate synthetic patient dataset (default N=500)")
    parser.add_argument("--train-baseline", action="store_true", help="Train and evaluate baseline ML models on synthetic dataset")
    parser.add_argument("--train-advanced-ml", action="store_true", help="Train, compare, and benchmark advanced ML model candidates")
    parser.add_argument("--validate-leakage-free-ml", action="store_true", help="Run 100% leak-free feature engineering and model validation")
    parser.add_argument("--validate-dataset-quality", action="store_true", help="Run pre-training feature/label alignment and dataset quality audit")
    parser.add_argument("--validate-features", action="store_true", help="Run feature engineering & ground-truth target label validation audit")
    parser.add_argument("--validate-robustness", action="store_true", help="Run K-Fold Cross-Validation, hyperparameter tuning & noise robustness audit")
    parser.add_argument("--predict-prakriti-ml", action="store_true", help="Run Section 14 final ML model inference on patient case")

    args = parser.parse_args()

    if args.generate_dataset:
        from dataset_generator import AYUSHDatasetGeneratorEngine
        print(f"Generating synthetic AYUSH dataset with N={args.generate_dataset} cases...")
        generator = AYUSHDatasetGeneratorEngine()
        stats = generator.generate_dataset(num_samples=args.generate_dataset)
        print("Dataset generation complete!")
        print(json.dumps(stats, indent=2))
        return

    if args.train_baseline:
        from ml_baseline_model import AYUSHBaselineMLEngine
        print("Training baseline ML models on dataset...")
        engine = AYUSHBaselineMLEngine()
        eval_report = engine.train_and_evaluate()
        print("Baseline ML Training complete!")
        print(json.dumps(eval_report, indent=2))
        return

    if args.train_advanced_ml:
        from advanced_ml_engine import AYUSHAdvancedMLEngine
        print("Training and benchmarking Advanced ML Model Candidates...")
        engine = AYUSHAdvancedMLEngine()
        comp_report = engine.train_and_compare_all_models()
        print("Advanced ML Comparative Evaluation complete!")
        print(json.dumps(comp_report, indent=2))
        return

    if args.validate_leakage_free_ml:
        from leakage_free_ml_engine import AYUSHLeakageFreeMLEngine
        print("Executing Leakage-Free Model Training & Validation...")
        engine = AYUSHLeakageFreeMLEngine()
        leak_report = engine.train_and_evaluate_all()
        print("Leakage-Free Model Validation complete!")
        print(json.dumps(leak_report, indent=2))
        return

    if args.validate_dataset_quality:
        from dataset_quality_alignment_engine import AYUSHDatasetQualityAlignmentEngine
        print("Executing Section 11 ML Dataset Quality & Feature/Label Alignment Audit...")
        engine = AYUSHDatasetQualityAlignmentEngine()
        audit_report = engine.run_full_alignment_audit()
        print("Dataset Quality & Alignment Audit complete!")
        print(json.dumps(audit_report, indent=2))
        return

    if args.validate_features:
        from feature_validation_engine import AYUSHFeatureValidationEngine
        print("Executing Section 12 ML Feature Engineering & Ground-Truth Label Validation Audit...")
        engine = AYUSHFeatureValidationEngine()
        feat_report = engine.run_full_feature_validation_audit()
        print("Feature Validation & Ground-Truth Audit complete!")
        print(json.dumps(feat_report, indent=2))
        return

    if args.validate_robustness:
        from ml_robustness_engine import AYUSHMLRobustnessEngine
        print("Executing Section 13 ML Model Optimization, Cross-Validation & Robustness Validation Audit...")
        engine = AYUSHMLRobustnessEngine()
        robustness_report = engine.run_full_optimization_and_robustness_suite()
        print("ML Optimization & Robustness Validation complete!")
        print(json.dumps(robustness_report, indent=2))
        return

    if args.predict_prakriti_ml:
        from final_model_trainer import AYUSHFinalModelTrainer
        print("Executing Section 14 Final Model Inference & Out-of-Sample Validation Pipeline...")
        trainer = AYUSHFinalModelTrainer()
        val_report = trainer.run_full_final_validation_pipeline()
        print("Final Model Inference & Validation complete!")
        print(json.dumps(val_report, indent=2))
        return

    default_sample = Path(__file__).resolve().parent.parent / "tests" / "sample_patient_cases.json"
    input_file = args.input or (str(default_sample) if default_sample.exists() else None)

    patient_case, report, rec_report, risk_report, sim_report, quality_report, pred_report, eval_report, reg_report, synthesis_report = run_pipeline(
        input_case_path=input_file,
        text_input=args.text,
        source=args.source,
        generate_explanation=not args.no_explain
    )

    print("============================================================")
    print("AYUSH 10-PARAMETER ASSESSMENT & CLINICAL CASE GENERATED")
    print("============================================================")
    print(f"Patient Case Output Saved: {OUTPUT_FILE}")
    if report:
        print(f"Explainability Report Saved: {EXPLAINABILITY_OUTPUT_FILE}")
    print(f"Clinical Recommendation Output Saved: {RECOMMENDATION_OUTPUT_FILE}")
    print(f"Longitudinal Risk Report Saved: {LONGITUDINAL_RISK_OUTPUT_FILE}")
    print(f"Patient Similarity Report Saved: {SIMILARITY_OUTPUT_FILE}")
    print(f"Data Quality Report Saved: {QUALITY_OUTPUT_FILE}")
    print(f"Clinical Outcome Prediction Saved: {PREDICTION_OUTPUT_FILE}")
    print(f"ML Evaluation Report Saved: {EVALUATION_OUTPUT_FILE}")
    print(f"Personalized Regimen Report Saved: {REGIMEN_OUTPUT_FILE}")
    print(f"Clinical Explainer Synthesis Saved: {EXPLAINER_SYNTHESIS_OUTPUT_FILE}")








if __name__ == "__main__":
    main()
