import tempfile
import unittest
from pathlib import Path

from training.export_mobile_int4 import (
    build_builder_command,
    require_passing_evaluation,
    require_original_weight_update,
)


class ExportMobileInt4Test(unittest.TestCase):
    def test_refuses_failed_evaluation_gate(self):
        with self.assertRaisesRegex(ValueError, "evaluation gate failed"):
            require_passing_evaluation(
                {
                    "schema_version": 1,
                    "general_ok": True,
                    "preference_ok": False,
                    "passes_gate": False,
                }
            )

    def test_accepts_only_consistent_passing_report(self):
        report = {
            "schema_version": 1,
            "general_ok": True,
            "preference_ok": True,
            "passes_gate": True,
        }
        self.assertEqual(report, require_passing_evaluation(report))

        inconsistent = dict(report)
        inconsistent["general_ok"] = False
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            require_passing_evaluation(inconsistent)

    def test_builder_command_uses_trained_local_model(self):
        command = build_builder_command(
            python_executable="/usr/bin/python",
            trained_model=Path("/tmp/h33-trained"),
            output_dir=Path("/tmp/h33-int4"),
        )
        joined = " ".join(command)
        self.assertIn("/tmp/h33-trained", joined)
        self.assertIn("/tmp/h33-int4", joined)
        self.assertIn("int4", joined)
        self.assertIn("hf_remote=false", joined)

    def test_mobile_export_requires_proof_original_weights_changed(self):
        self.assertTrue(
            require_original_weight_update(
                {
                    "optimizer_steps": 4,
                    "original_weights_changed": True,
                }
            )
        )

        with self.assertRaisesRegex(ValueError, "original Qwen weights"):
            require_original_weight_update(
                {
                    "optimizer_steps": 4,
                    "original_weights_changed": False,
                }
            )



if __name__ == "__main__":
    unittest.main()
