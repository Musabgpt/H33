import tempfile
import unittest
from pathlib import Path

from training.train_partial_sft import (
    ResumeCursor,
    read_checkpoint_metadata,
    save_optimizer_checkpoint,
    validate_resume_metadata,
)


class FakeSaveable:
    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "marker").write_text("ok", encoding="utf-8")


class FakeStateful:
    def state_dict(self):
        return {"value": 1}


class FakeCuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def get_rng_state_all():
        return []


class FakeTorch:
    cuda = FakeCuda()

    @staticmethod
    def get_rng_state():
        return "cpu-rng"

    @staticmethod
    def save(value, path):
        Path(path).write_text(repr(value), encoding="utf-8")


class CheckpointMetadataTest(unittest.TestCase):
    def test_persists_and_validates_training_fingerprint(self):
        metadata = {
            "dataset_sha256": "abc123",
            "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "seed": 3407,
            "last_n_blocks": 2,
            "gradient_accumulation_steps": 16,
        }

        with tempfile.TemporaryDirectory() as td:
            checkpoint = save_optimizer_checkpoint(
                output_dir=Path(td),
                cursor=ResumeCursor(
                    global_step=2,
                    epoch=0,
                    next_order_offset=4,
                ),
                model=FakeSaveable(),
                tokenizer=FakeSaveable(),
                optimizer=FakeStateful(),
                scheduler=FakeStateful(),
                torch=FakeTorch,
                metadata=metadata,
            )

            self.assertEqual(metadata, read_checkpoint_metadata(checkpoint))
            validate_resume_metadata(checkpoint, metadata)

            changed = dict(metadata)
            changed["gradient_accumulation_steps"] = 8
            with self.assertRaisesRegex(ValueError, "gradient_accumulation_steps"):
                validate_resume_metadata(checkpoint, changed)

    def test_resume_rejects_checkpoint_without_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            checkpoint = save_optimizer_checkpoint(
                output_dir=Path(td),
                cursor=ResumeCursor(
                    global_step=1,
                    epoch=0,
                    next_order_offset=1,
                ),
                model=FakeSaveable(),
                tokenizer=FakeSaveable(),
                optimizer=FakeStateful(),
                scheduler=FakeStateful(),
                torch=FakeTorch,
            )

            with self.assertRaisesRegex(ValueError, "metadata"):
                validate_resume_metadata(
                    checkpoint,
                    {"dataset_sha256": "abc"},
                )


if __name__ == "__main__":
    unittest.main()
