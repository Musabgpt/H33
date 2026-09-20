import unittest

from training.evaluate_partial import (
    case_passes,
    compute_gate,
    normalize_answer,
)


class EvaluatePartialTest(unittest.TestCase):
    def test_normalizes_arabic_digits_and_whitespace(self):
        self.assertEqual("42", normalize_answer("  ٤٢.  "))
        self.assertEqual("باريس", normalize_answer("باريس!"))

    def test_case_checks_exact_or_contains_without_brittle_formatting(self):
        exact_case = {
            "check": {"type": "exact_any", "values": ["42", "٤٢"]}
        }
        self.assertTrue(case_passes("٤٢", exact_case))
        self.assertFalse(case_passes("الجواب هو 42", exact_case))

        contains_case = {
            "check": {"type": "contains_any", "values": ["باريس"]}
        }
        self.assertTrue(case_passes("الإجابة هي باريس.", contains_case))

    def test_gate_blocks_more_than_five_point_general_regression(self):
        gate = compute_gate(
            baseline_general_pass_rate=0.90,
            trained_general_pass_rate=0.84,
            baseline_preference_margin=0.10,
            trained_preference_margin=0.20,
            preference_pairs=10,
        )
        self.assertFalse(gate["general_ok"])
        self.assertFalse(gate["passes_gate"])

    def test_gate_allows_exactly_five_point_general_drop(self):
        gate = compute_gate(
            baseline_general_pass_rate=0.90,
            trained_general_pass_rate=0.85,
            baseline_preference_margin=0.10,
            trained_preference_margin=0.10,
            preference_pairs=10,
        )
        self.assertTrue(gate["general_ok"])
        self.assertTrue(gate["preference_ok"])
        self.assertTrue(gate["passes_gate"])

    def test_gate_blocks_preference_margin_regression_when_pairs_exist(self):
        gate = compute_gate(
            baseline_general_pass_rate=0.80,
            trained_general_pass_rate=0.82,
            baseline_preference_margin=0.12,
            trained_preference_margin=0.11,
            preference_pairs=4,
        )
        self.assertTrue(gate["general_ok"])
        self.assertFalse(gate["preference_ok"])
        self.assertFalse(gate["passes_gate"])

    def test_gate_skips_preference_requirement_when_no_pairs_exist(self):
        gate = compute_gate(
            baseline_general_pass_rate=0.80,
            trained_general_pass_rate=0.80,
            baseline_preference_margin=0.0,
            trained_preference_margin=0.0,
            preference_pairs=0,
        )
        self.assertTrue(gate["preference_ok"])
        self.assertTrue(gate["passes_gate"])


if __name__ == "__main__":
    unittest.main()
