"""
Unit Tests for AYUSHFinalModelTrainer (Section 14)
"""

import sys
import unittest
from pathlib import Path

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from final_model_trainer import AYUSHFinalModelTrainer, ACTIVE_PARAMETERS, EXCLUDED_PARAMETERS


class TestAYUSHFinalModelTrainer(unittest.TestCase):
    def setUp(self):
        self.trainer = AYUSHFinalModelTrainer()

    def test_train_and_serialize_model(self):
        train_info = self.trainer.train_final_model(k=9)
        self.assertEqual(train_info["training_samples"], 500)
        self.assertEqual(train_info["feature_dimension"], 51)

        saved_path = self.trainer.save_model_artifact()
        self.assertTrue(Path(saved_path).exists())

    def test_load_and_predict_case(self):
        self.trainer.train_final_model(k=9)
        self.trainer.save_model_artifact()

        new_trainer = AYUSHFinalModelTrainer()
        loaded = new_trainer.load_model_artifact()
        self.assertTrue(loaded)

        sample_case = {
            "patient": {"age": 45, "gender": "Female"},
            "questionnaire_answers": {
                "prakriti": [
                    {"questionId": "prakriti_q01", "answer": "Thin/light build"},
                    {"questionId": "prakriti_q02", "answer": "Usually dry"}
                ]
            }
        }
        pred = new_trainer.predict_case(sample_case)
        self.assertIn("predicted_prakriti", pred)
        self.assertIn("confidence", pred)
        self.assertEqual(pred["excluded_parameters"], EXCLUDED_PARAMETERS)

    def test_evaluate_independent_out_of_sample(self):
        eval_report = self.trainer.evaluate_independent_out_of_sample(num_samples=100, seed=999)
        self.assertEqual(eval_report["eval_dataset_seed"], 999)
        self.assertEqual(eval_report["independent_samples_eval"], 100)
        self.assertTrue(eval_report["independent_accuracy"] >= 0.85)
        self.assertIn("per_class_metrics", eval_report)

    def test_run_full_final_validation_pipeline(self):
        report = self.trainer.run_full_final_validation_pipeline()
        self.assertEqual(report["overall_status"], "PASSED")
        self.assertIn("serialized_artifact_path", report)
        self.assertIn("cross_validation_5fold_benchmark", report)
        self.assertIn("independent_out_of_sample_validation_100", report)
        self.assertEqual(report["excluded_parameters"], EXCLUDED_PARAMETERS)


if __name__ == "__main__":
    unittest.main()
