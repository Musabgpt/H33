import json
import tempfile
import unittest
from pathlib import Path

from training.train_partial_sft import (
    ResumeCursor,
    read_checkpoint_cursor,
    save_optimizer_checkpoint,
)


class FakeSaveable:
    def __init__(self, marker):
        self.marker = marker

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / self.marker).write_text("ok", encoding="utf-8")


class FakeStateful:
    def __init__(self, value):
        self.value = value

    def state_dict(self):
        return {"value": self.value}


class FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def get_rng_state_all():
        return ["cuda-rng"]


class FakeTorch:
    cuda = FakeCuda()

    @staticmethod
    def get_rng_state():
        return "cpu-rng"

    @staticmethod
    def save(value, path):
        Path(path).write_text(repr(value), encoding="utf-8")


class CheckpointLayoutTest(unittest.TestCase):
    def test_checkpoint_is_published_atomically_after_optimizer_boundary(self):
        cursor = ResumeCursor(global_step=3, epoch=1, next_order_offset=4)

        with tempfile.TemporaryDirectory() as td:
            output = Path(td)
            checkpoint = save_optimizer_checkpoint(
                output_dir=output,
                cursor=cursor,
                model=FakeSaveable("model.marker"),
                tokenizer=FakeSaveable("tokenizer.marker"),
                optimizer=FakeStateful("optimizer"),
                scheduler=FakeStateful("scheduler"),
                torch=FakeTorch,
            )

            self.assertEqual("checkpoint-00000003", checkpoint.name)
            self.assertTrue((checkpoint / "model" / "model.marker").exists())
            self.assertTrue((checkpoint / "model" / "tokenizer.marker").exists())
            self.assertTrue((checkpoint / "optimizer.pt").exists())
            self.assertTrue((checkpoint / "scheduler.pt").exists())
            self.assertTrue((checkpoint / "rng_state.pt").exists())
            self.assertTrue((checkpoint / "trainer_state.json").exists())
            self.assertFalse((output / ".checkpoint-00000003.tmp").exists())
            self.assertEqual(cursor, read_checkpoint_cursor(checkpoint))

            manifest = json.loads(
                (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(1, manifest["schema_version"])
            self.assertEqual(cursor.to_dict(), manifest["cursor"])

    def test_refuses_to_overwrite_existing_checkpoint(self):
        cursor = ResumeCursor(global_step=1, epoch=0, next_order_offset=1)

        with tempfile.TemporaryDirectory() as td:
            output = Path(td)
            (output / "checkpoint-00000001").mkdir()

            with self.assertRaises(FileExistsError):
                save_optimizer_checkpoint(
                    output_dir=output,
                    cursor=cursor,
                    model=FakeSaveable("model.marker"),
                    tokenizer=FakeSaveable("tokenizer.marker"),
                    optimizer=FakeStateful("optimizer"),
                    scheduler=FakeStateful("scheduler"),
                    torch=FakeTorch,
                )


if __name__ == "__main__":
    unittest.main()
