import json
import unittest
from pathlib import Path


NOTEBOOK = Path("training/H33_PARTIAL_QWEN_KAGGLE.ipynb")


class KaggleNotebookContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.source = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.data.get("cells", [])
        )

    def test_notebook_is_valid_and_small(self):
        self.assertEqual(4, self.data["nbformat"])
        self.assertLess(NOTEBOOK.stat().st_size, 1_000_000)

    def test_does_not_replace_kaggle_torch_or_transformers(self):
        lowered = self.source.lower()
        self.assertNotIn("pip install torch", lowered)
        self.assertNotIn("pip install transformers", lowered)
        self.assertIn("training/check_environment.py", self.source)
        self.assertIn("pip','freeze", self.source)

    def test_real_weight_smoke_resume_and_gate_are_present(self):
        self.assertIn("training/train_partial_sft.py", self.source)
        self.assertIn("--max-optimizer-steps','2'", self.source)
        self.assertIn("--resume-from-checkpoint", self.source)
        self.assertIn("--max-optimizer-steps','3'", self.source)
        self.assertIn("training/evaluate_partial.py", self.source)
        self.assertIn("original_weights_changed", self.source)

    def test_int4_export_is_opt_in(self):
        self.assertIn("EXPORT_INT4 = False", self.source)
        self.assertIn("training/export_mobile_int4.py", self.source)


if __name__ == "__main__":
    unittest.main()
