import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "model_smoke", Path(__file__).resolve().parents[1] / "scripts/model_smoke.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class ModelSmokeTest(unittest.TestCase):
    def test_prompt_echo_is_not_a_generated_answer(self):
        with self.assertRaises(ValueError):
            smoke.generated_answer(smoke.PROMPT + "\n", 0)

    def test_runtime_log_is_not_a_generated_answer(self):
        with self.assertRaises(ValueError):
            smoke.generated_answer("llama_model_load: loaded 24 tensors\n", 0)

    def test_nonzero_exit_rejects_even_visible_output(self):
        with self.assertRaises(ValueError):
            smoke.generated_answer(smoke.PROMPT + "Hello", 2)

    def test_accepts_text_after_echoed_prompt(self):
        self.assertEqual(smoke.generated_answer(
            "<BOS>" + smoke.PROMPT + "Hello!\n", 0), "Hello!")


if __name__ == "__main__":
    unittest.main()
