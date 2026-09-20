import unittest

from training.train_partial_sft import model_dtype_kwargs


class TransformersCompatTest(unittest.TestCase):
    def test_transformers_v5_uses_dtype(self):
        marker = object()
        self.assertEqual(
            {"dtype": marker},
            model_dtype_kwargs("5.9.0", marker),
        )

    def test_transformers_v4_uses_legacy_torch_dtype(self):
        marker = object()
        self.assertEqual(
            {"torch_dtype": marker},
            model_dtype_kwargs("4.57.3", marker),
        )

    def test_rejects_unparseable_version(self):
        with self.assertRaises(ValueError):
            model_dtype_kwargs("unknown", object())


if __name__ == "__main__":
    unittest.main()
