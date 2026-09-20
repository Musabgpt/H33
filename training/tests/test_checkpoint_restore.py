import pickle
import random
import tempfile
import unittest
from pathlib import Path

from training.train_partial_sft import (
    ResumeCursor,
    restore_optimizer_checkpoint,
    save_optimizer_checkpoint,
)


class FakeSaveable:
    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "marker").write_text("ok", encoding="utf-8")


class FakeStateful:
    def __init__(self, value):
        self.value = value

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = state["value"]


class FakeCuda:
    current = ["initial-cuda"]

    @classmethod
    def is_available(cls):
        return True

    @classmethod
    def get_rng_state_all(cls):
        return list(cls.current)

    @classmethod
    def set_rng_state_all(cls, value):
        cls.current = list(value)


class FakeTorch:
    cuda = FakeCuda
    current_cpu_rng = "initial-cpu"

    @classmethod
    def get_rng_state(cls):
        return cls.current_cpu_rng

    @classmethod
    def set_rng_state(cls, value):
        cls.current_cpu_rng = value

    @staticmethod
    def save(value, path):
        with Path(path).open("wb") as handle:
            pickle.dump(value, handle)

    @staticmethod
    def load(path, **kwargs):
        with Path(path).open("rb") as handle:
            return pickle.load(handle)


class CheckpointRestoreTest(unittest.TestCase):
    def test_restores_optimizer_scheduler_and_rng_at_saved_boundary(self):
        cursor = ResumeCursor(global_step=7, epoch=2, next_order_offset=3)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            optimizer = FakeStateful("optimizer-before")
            scheduler = FakeStateful("scheduler-before")

            random.seed(3407)
            expected_python_state = random.getstate()
            FakeTorch.current_cpu_rng = "cpu-saved"
            FakeCuda.current = ["cuda-saved-0", "cuda-saved-1"]

            checkpoint = save_optimizer_checkpoint(
                output_dir=root,
                cursor=cursor,
                model=FakeSaveable(),
                tokenizer=FakeSaveable(),
                optimizer=optimizer,
                scheduler=scheduler,
                torch=FakeTorch,
            )

            optimizer.value = "optimizer-corrupted"
            scheduler.value = "scheduler-corrupted"
            random.seed(999)
            FakeTorch.current_cpu_rng = "cpu-corrupted"
            FakeCuda.current = ["cuda-corrupted"]

            restored = restore_optimizer_checkpoint(
                checkpoint_dir=checkpoint,
                optimizer=optimizer,
                scheduler=scheduler,
                torch=FakeTorch,
                map_location="cpu",
            )

            self.assertEqual(cursor, restored)
            self.assertEqual("optimizer-before", optimizer.value)
            self.assertEqual("scheduler-before", scheduler.value)
            self.assertEqual(expected_python_state, random.getstate())
            self.assertEqual("cpu-saved", FakeTorch.current_cpu_rng)
            self.assertEqual(
                ["cuda-saved-0", "cuda-saved-1"],
                FakeCuda.current,
            )


if __name__ == "__main__":
    unittest.main()
