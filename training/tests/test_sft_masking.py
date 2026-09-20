import unittest

from training.train_partial_sft import build_masked_sequence


class SftMaskingTest(unittest.TestCase):
    def test_masks_prompt_and_keeps_assistant_tokens(self):
        input_ids, labels = build_masked_sequence(
            prompt_ids=[1, 2, 3],
            full_ids=[1, 2, 3, 4, 5],
            max_length=8,
        )

        self.assertEqual([1, 2, 3, 4, 5], input_ids)
        self.assertEqual([-100, -100, -100, 4, 5], labels)

    def test_left_truncation_masks_only_surviving_prompt_tokens(self):
        input_ids, labels = build_masked_sequence(
            prompt_ids=[1, 2, 3, 4],
            full_ids=[1, 2, 3, 4, 5, 6],
            max_length=4,
        )

        self.assertEqual([3, 4, 5, 6], input_ids)
        self.assertEqual([-100, -100, 5, 6], labels)

    def test_rejects_when_truncation_removes_entire_assistant_answer(self):
        with self.assertRaisesRegex(ValueError, "assistant response"):
            build_masked_sequence(
                prompt_ids=[1, 2, 3, 4],
                full_ids=[1, 2, 3, 4],
                max_length=2,
            )

    def test_requires_positive_max_length(self):
        with self.assertRaises(ValueError):
            build_masked_sequence([1], [1, 2], 0)


if __name__ == "__main__":
    unittest.main()
