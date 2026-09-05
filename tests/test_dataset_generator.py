"""
Unit Tests for AYUSHDatasetGeneratorEngine (Section 7)
"""

import sys
import unittest
from pathlib import Path

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset_generator import AYUSHDatasetGeneratorEngine, ACTIVE_PARAMETERS, EXCLUDED_PARAMETERS


class TestAYUSHDatasetGeneratorEngine(unittest.TestCase):
    def setUp(self):
        self.generator = AYUSHDatasetGeneratorEngine(seed=42)

    def test_generate_single_patient_case(self):
        case = self.generator.generate_patient_case(1)
        self.assertIn("patient", case)
        self.assertEqual(case["patient"]["patientId"], "P1001")
        self.assertTrue(18 <= case["patient"]["age"] <= 80)

        dashavidha = case["ayushAssessment"]["dashavidhaPariksha"]
        self.assertEqual(len(dashavidha), 10)
        for param in ACTIVE_PARAMETERS:
            self.assertIn(param, dashavidha)

        for param in EXCLUDED_PARAMETERS:
            self.assertNotIn(param, dashavidha)

    def test_generate_dataset_batch(self):
        test_path = BASE_DIR / "data" / "test_dataset_temp.json"
        try:
            stats = self.generator.generate_dataset(num_samples=10, save_path=test_path)
            self.assertEqual(stats["total_generated"], 10)
            self.assertEqual(stats["quality_validated_count"], 10)
            self.assertTrue(test_path.exists())
        finally:
            if test_path.exists():
                test_path.unlink()

    def test_agni_koshtha_exclusion(self):
        case = self.generator.generate_patient_case(2)
        dashavidha = case["ayushAssessment"]["dashavidhaPariksha"]
        self.assertNotIn("agni", dashavidha)
        self.assertNotIn("koshtha", dashavidha)


if __name__ == "__main__":
    unittest.main()
