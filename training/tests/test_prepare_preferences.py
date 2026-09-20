import json
import tempfile
import unittest
from pathlib import Path

from training.prepare_preferences import (
    InputStats,
    prepare_from_paths,
    split_effective_records,
)


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if isinstance(row, str):
                handle.write(row + "\n")
            else:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class PreparePreferencesTest(unittest.TestCase):
    def test_latest_sft_response_supersedes_older_machine_choice(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sft_old = root / "old_sft.jsonl"
            sft_new = root / "new_sft.jsonl"
            dpo = root / "dpo.jsonl"

            write_jsonl(sft_old, [
                {"id": "t1", "prompt": "سؤال", "response": "جواب البحث"},
            ])
            write_jsonl(sft_new, [
                {"id": "t1", "prompt": "سؤال", "response": "تصحيحي النهائي"},
                {"id": "t2", "prompt": "سؤال 2", "response": "جواب 2"},
            ])
            write_jsonl(dpo, [
                {
                    "id": "t1",
                    "prompt": "سؤال",
                    "chosen": "جواب البحث",
                    "rejected": "جواب محلي",
                },
                {
                    "id": "t1",
                    "prompt": "سؤال",
                    "chosen": "جواب البحث",
                    "rejected": "جواب Google",
                },
                {
                    "id": "t2",
                    "prompt": "سؤال 2",
                    "chosen": "جواب 2",
                    "rejected": "جواب آخر",
                },
            ])

            prepared = prepare_from_paths(
                sft_paths=[sft_old, sft_new],
                preference_paths=[dpo],
            )

            self.assertEqual("تصحيحي النهائي", prepared.sft["t1"]["response"])
            pairs = prepared.preferences["t1"]
            self.assertTrue(all(p["chosen"] == "تصحيحي النهائي" for p in pairs))
            rejected = {p["rejected"] for p in pairs}
            self.assertIn("جواب البحث", rejected)
            self.assertIn("جواب محلي", rejected)
            self.assertIn("جواب Google", rejected)

    def test_malformed_and_empty_rows_are_counted_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sft = root / "sft.jsonl"
            dpo = root / "dpo.jsonl"

            write_jsonl(sft, [
                '{"id":',
                {"id": "bad-empty", "prompt": "س", "response": "   "},
                {"id": "ok1", "prompt": "س1", "response": "ج1"},
                {"id": "ok2", "prompt": "س2", "response": "ج2"},
            ])
            write_jsonl(dpo, [
                "not-json",
                {"id": "ok1", "prompt": "س1", "chosen": "", "rejected": "x"},
                {
                    "id": "ok1",
                    "prompt": "س1",
                    "chosen": "ج1",
                    "rejected": "خطأ",
                },
            ])

            prepared = prepare_from_paths([sft], [dpo])

            self.assertEqual(2, len(prepared.sft))
            self.assertGreaterEqual(prepared.stats.malformed_lines, 2)
            self.assertGreaterEqual(prepared.stats.empty_or_invalid_rows, 2)

    def test_split_is_deterministic_and_keeps_turns_together(self):
        sft = {
            f"id-{i:02d}": {
                "id": f"id-{i:02d}",
                "prompt": f"p{i}",
                "response": f"r{i}",
            }
            for i in range(20)
        }
        prefs = {
            key: [
                {
                    "id": key,
                    "prompt": row["prompt"],
                    "chosen": row["response"],
                    "rejected": "other-" + key,
                }
            ]
            for key, row in sft.items()
        }

        first = split_effective_records(sft, prefs, seed=3407)
        second = split_effective_records(sft, prefs, seed=3407)

        self.assertEqual(first.eval_ids, second.eval_ids)
        self.assertEqual(2, len(first.eval_ids))
        self.assertTrue(set(first.train_ids).isdisjoint(first.eval_ids))

    def test_single_turn_refuses_train_eval_split(self):
        sft = {"t1": {"id": "t1", "prompt": "p", "response": "r"}}
        with self.assertRaisesRegex(ValueError, "not enough data"):
            split_effective_records(sft, {}, seed=3407)


if __name__ == "__main__":
    unittest.main()
