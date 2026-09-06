"""
AYUSH Production Model Serving, Local Feature Explainability & Data Drift Detection Engine (Section 15)

Provides:
1. Fast production model inference with instance-level local feature explanations using KNN k=9.
2. Data drift detection monitoring comparing incoming patient feature vectors against the N=500 synthetic baseline (seed=42).
3. Pure-python 2-sample Kolmogorov-Smirnov (KS) test and Wasserstein distance computation.
4. Explicit statistical thresholds and drift classification without automatic retraining.

Strictly enforces:
- Single 51-feature extraction pipeline (AYUSHLeakageFreeFeatureEngine).
- 10 active parameters only (Agni and Koshtha strictly EXCLUDED).
- Zero target leakage.
- Synthetic baseline disclaimers.
"""

import json
import math
import time
from pathlib import Path

from leakage_free_ml_engine import (
    AYUSHLeakageFreeMLEngine,
    ACTIVE_PARAMETERS,
    EXCLUDED_PARAMETERS
)
from final_model_trainer import AYUSHFinalModelTrainer


# Feature index names for explainability and drift reports (51 total)
FEATURE_NAMES = ["age", "gender"]
# 10 active parameters x 4 questions = 40 features
for param in ACTIVE_PARAMETERS:
    for q_idx in range(1, 5):
        FEATURE_NAMES.append(f"{param}_q{q_idx}")
# 9 clinical note terms
CLINICAL_TERMS = ["dry", "light", "cold", "hot", "sharp", "oily", "heavy", "stable", "soft"]
for term in CLINICAL_TERMS:
    FEATURE_NAMES.append(f"note_term_{term}")


def ks_two_sample_statistic(data1, data2):
    """
    Computes the 2-sample Kolmogorov-Smirnov statistic D and asymptotic p-value
    for two 1D empirical distributions data1 and data2.
    """
    n1 = len(data1)
    n2 = len(data2)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    s1 = sorted(data1)
    s2 = sorted(data2)

    # Combine unique values
    all_vals = sorted(list(set(s1 + s2)))
    
    max_d = 0.0
    i1 = 0
    i2 = 0

    for val in all_vals:
        while i1 < n1 and s1[i1] <= val:
            i1 += 1
        while i2 < n2 and s2[i2] <= val:
            i2 += 1
        
        cdf1 = i1 / float(n1)
        cdf2 = i2 / float(n2)
        diff = abs(cdf1 - cdf2)
        if diff > max_d:
            max_d = diff

    # Asymptotic p-value calculation (Kolmogorov distribution approximation)
    en = math.sqrt((n1 * n2) / float(n1 + n2))
    lambda_val = (en + 0.12 + 0.11 / en) * max_d

    # Sum asymptotic series
    p_val = 0.0
    for k in range(1, 101):
        term = 2.0 * ((-1) ** (k - 1)) * math.exp(-2.0 * (k * lambda_val) ** 2)
        p_val += term
        if abs(term) < 1e-10:
            break
    
    p_val = max(0.0, min(1.0, p_val))
    return round(max_d, 4), round(p_val, 4)


def wasserstein_distance_1d(data1, data2):
    """
    Computes 1D Wasserstein-1 Distance (Earth Mover's Distance)
    W_1 = integral |CDF1(x) - CDF2(x)| dx
    """
    n1 = len(data1)
    n2 = len(data2)
    if n1 == 0 or n2 == 0:
        return 0.0

    s1 = sorted(data1)
    s2 = sorted(data2)

    all_vals = sorted(list(set(s1 + s2)))
    if len(all_vals) <= 1:
        return 0.0

    total_w = 0.0
    i1 = 0
    i2 = 0

    for idx in range(len(all_vals) - 1):
        v_curr = all_vals[idx]
        v_next = all_vals[idx + 1]
        width = v_next - v_curr

        while i1 < n1 and s1[i1] <= v_curr:
            i1 += 1
        while i2 < n2 and s2[i2] <= v_curr:
            i2 += 1

        cdf1 = i1 / float(n1)
        cdf2 = i2 / float(n2)
        total_w += abs(cdf1 - cdf2) * width

    return round(total_w, 4)


class AYUSHModelServingDriftEngine:
    def __init__(self, model_path=None, dataset_path=None):
        base_dir = Path(__file__).resolve().parent.parent
        self.model_path = model_path or base_dir / "models" / "ayush_prakriti_knn_model.json"
        self.dataset_path = dataset_path or base_dir / "data" / "ayush_synthetic_dataset.json"
        self.output_drift_report_file = base_dir / "output" / "model_drift_monitoring_report.json"
        
        self.trainer = AYUSHFinalModelTrainer(model_path=self.model_path, dataset_path=self.dataset_path)
        self.leak_free_engine = self.trainer.leak_free_engine
        
        # Ensure model is loaded
        if self.model_path.exists():
            self.trainer.load_model_artifact()
        else:
            self.trainer.train_final_model(k=9)
            self.trainer.save_model_artifact()

    def predict_with_explanation(self, patient_case):
        """
        Runs leakage-free prediction using trained KNN (k=9) and provides instance-level local feature explanations.
        
        Uses EXACT SAME 51-feature extraction pipeline.
        """
        t0 = time.time()
        # 1. Extract 51-dimensional feature vector using the single unified pipeline
        features = self.leak_free_engine.extract_raw_features(patient_case)
        
        # 2. Perform KNN prediction
        model = self.trainer.model
        if not model or not hasattr(model, "X_train"):
            self.trainer.load_model_artifact()
            model = self.trainer.model

        pred_prakriti = model.predict_one(features)
        latency_ms = round((time.time() - t0) * 1000.0, 3)

        # 3. Compute Euclidean distances to all baseline training samples
        X_train = model.X_train
        y_train = model.y_train

        distances = []
        for i, train_vec in enumerate(X_train):
            d = model._euclidean_distance(features, train_vec)
            distances.append((d, y_train[i], train_vec))
        
        distances.sort(key=lambda x: x[0])
        top_k = distances[:model.k]

        vote_counts = {"vata": 0, "pitta": 0, "kapha": 0}
        for _, label, _ in top_k:
            vote_counts[label] = vote_counts.get(label, 0) + 1

        confidence = round(vote_counts[pred_prakriti] / float(model.k), 3)

        # Probabilities
        probabilities = {
            cls: round(vote_counts[cls] / float(model.k), 3)
            for cls in ["vata", "pitta", "kapha"]
        }

        # 4. Instance-level local feature explainability analysis
        # Analyze average feature differences between input vector and top-k neighbors
        feature_diffs = []
        for feat_idx in range(len(features)):
            feat_val = features[feat_idx]
            avg_neighbor_val = sum(neighbor_vec[feat_idx] for _, _, neighbor_vec in top_k) / float(len(top_k))
            diff = abs(feat_val - avg_neighbor_val)
            
            feat_name = FEATURE_NAMES[feat_idx] if feat_idx < len(FEATURE_NAMES) else f"feature_{feat_idx}"
            feature_diffs.append({
                "feature_index": feat_idx,
                "feature_name": feat_name,
                "input_value": round(feat_val, 4),
                "neighbors_avg_value": round(avg_neighbor_val, 4),
                "absolute_difference": round(diff, 4)
            })

        # Features with smallest difference to nearest neighbors strongly support the cluster
        feature_diffs.sort(key=lambda x: x["absolute_difference"])
        top_supporting_features = feature_diffs[:5]
        
        # Features with largest difference show local contrast/variance
        feature_diffs.sort(key=lambda x: x["absolute_difference"], reverse=True)
        top_contrasting_features = feature_diffs[:3]

        return {
            "predicted_prakriti": pred_prakriti,
            "confidence": confidence,
            "class_probabilities": probabilities,
            "inference_latency_ms": latency_ms,
            "model_version": f"KNN (k={model.k})",
            "feature_pipeline": "51-dimensional AYUSHLeakageFreeFeatureEngine",
            "local_explanation": {
                "top_supporting_features": top_supporting_features,
                "top_contrasting_features": top_contrasting_features,
                "explanation_disclaimer": "Local feature contribution analysis indicates statistical proximity in synthetic feature space, NOT clinical causation."
            },
            "excluded_parameters": EXCLUDED_PARAMETERS,
            "synthetic_disclaimer": "Synthetic ML prediction. Must be validated clinically before deployment."
        }

    def detect_data_drift(self, incoming_cases, alpha_overall=0.05, wasserstein_threshold=0.15):
        """
        Monitors incoming patient feature distributions against the serialized N=500 training baseline (seed=42).
        
        Uses EXACT SAME 51-feature extraction pipeline.
        
        Statistical Methods:
        - 2-Sample Kolmogorov-Smirnov (KS) Test per feature.
        - 1D Wasserstein-1 Distance (Earth Mover's Distance) per feature.
        - Bonferroni-corrected alpha threshold: alpha_adj = alpha_overall / 51.
        - Explicit feature drift classification & overall system verdict.
        
        Constraint: Does NOT trigger automatic retraining.
        """
        if not incoming_cases:
            raise ValueError("incoming_cases batch cannot be empty for drift analysis.")

        # Load baseline training features from model artifact
        model = self.trainer.model
        if not model or not hasattr(model, "X_train"):
            self.trainer.load_model_artifact()
            model = self.trainer.model

        X_baseline = model.X_train  # Shape: (500, 51)
        num_baseline = len(X_baseline)
        num_features = len(X_baseline[0])

        # Extract 51-dimensional features for incoming batch using EXACT SAME PIPELINE
        X_incoming = [self.leak_free_engine.extract_raw_features(case) for case in incoming_cases]
        num_incoming = len(X_incoming)

        # Adjusted alpha using Bonferroni correction for 51 simultaneous hypothesis tests
        alpha_adj = round(alpha_overall / float(num_features), 6)
        
        # KS critical value threshold for 2-sample test @ alpha=0.05
        # D_crit = 1.36 * sqrt((N1 + N2) / (N1 * N2))
        ks_crit_value = round(1.36 * math.sqrt((num_baseline + num_incoming) / float(num_baseline * num_incoming)), 4)

        per_feature_drift_results = []
        drifted_feature_count = 0

        for f_idx in range(num_features):
            feat_name = FEATURE_NAMES[f_idx] if f_idx < len(FEATURE_NAMES) else f"feature_{f_idx}"
            
            vals_baseline = [sample[f_idx] for sample in X_baseline]
            vals_incoming = [sample[f_idx] for sample in X_incoming]

            # Compute KS Statistic & p-value
            ks_stat, p_val = ks_two_sample_statistic(vals_baseline, vals_incoming)

            # Compute Wasserstein Distance
            w_dist = wasserstein_distance_1d(vals_baseline, vals_incoming)

            # Feature is classified as DRIFTED if statistically significant shift (p < alpha_adj)
            # AND Wasserstein distance exceeds physical range threshold
            is_drifted = (p_val < alpha_adj) and (w_dist > wasserstein_threshold)
            if is_drifted:
                drifted_feature_count += 1

            per_feature_drift_results.append({
                "feature_index": f_idx,
                "feature_name": feat_name,
                "ks_statistic": ks_stat,
                "ks_p_value": p_val,
                "wasserstein_distance": w_dist,
                "is_drifted": is_drifted,
                "baseline_mean": round(sum(vals_baseline) / float(num_baseline), 4),
                "incoming_mean": round(sum(vals_incoming) / float(num_incoming), 4)
            })

        drift_percentage = round((drifted_feature_count / float(num_features)) * 100.0, 2)

        # Overall Drift Status Categorization
        if drift_percentage < 10.0:
            overall_status = "NO_DRIFT"
            severity = "Low"
            recommendation = "In-distribution patient data. No operational action required."
        elif drift_percentage <= 25.0:
            overall_status = "SLIGHT_DRIFT"
            severity = "Moderate"
            recommendation = "Minor distribution shift detected. Log and monitor feature shifts. Manual review recommended before considering retraining."
        else:
            overall_status = "SIGNIFICANT_DRIFT"
            severity = "High"
            recommendation = "FLAGGED FOR CLINICAL & DATA ARCHITECTURE REVIEW. Automatic model retraining is DISABLED. Inspect feature distribution shifts before taking manual operational action."

        report = {
            "monitoring_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "baseline_training_samples": num_baseline,
            "incoming_evaluation_samples": num_incoming,
            "feature_dimension": num_features,
            "statistical_thresholds": {
                "overall_alpha": alpha_overall,
                "bonferroni_adjusted_alpha": alpha_adj,
                "ks_critical_value_threshold": ks_crit_value,
                "wasserstein_distance_threshold": wasserstein_threshold
            },
            "drift_summary": {
                "total_features": num_features,
                "drifted_features_count": drifted_feature_count,
                "drift_percentage": drift_percentage,
                "overall_drift_status": overall_status,
                "severity_level": severity,
                "action_recommendation": recommendation,
                "automatic_retraining_triggered": False
            },
            "per_feature_drift_details": per_feature_drift_results,
            "disclaimers": {
                "synthetic_baseline_limitation": "Drift is evaluated relative to the N=500 synthetic baseline training dataset (seed=42). Drift detection does NOT automatically imply clinical population shift or model invalidity.",
                "retraining_policy": "Automatic retraining is strictly disabled. Retraining must be a deliberate, validated offline process.",
                "excluded_parameters": EXCLUDED_PARAMETERS
            }
        }

        # Save drift monitoring report to disk
        self.output_drift_report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_drift_report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report
