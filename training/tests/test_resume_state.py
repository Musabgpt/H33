import unittest

from training.train_partial_sft import (
    ResumeCursor,
    build_epoch_order,
    cursor_after_optimizer_step,
    optimizer_step_groups,
    optimizer_limit_reached,
)


def simulate_optimizer_steps(weight, cursor, *, example_count, seed, steps):
    seen = []
    done = 0
    current = cursor

    while done < steps:
        order = build_epoch_order(example_count, seed, current.epoch)
        if current.next_order_offset >= len(order):
            current = ResumeCursor(
                global_step=current.global_step,
                epoch=current.epoch + 1,
                next_order_offset=0,
            )
            continue

        position = current.next_order_offset
        example_index = order[position]

        # Tiny deterministic stand-in for an optimizer update.
        weight = weight * 0.97 + (example_index + 1) * 0.001
        seen.append((current.epoch, position, example_index))

        current = cursor_after_optimizer_step(
            global_step=current.global_step + 1,
            epoch=current.epoch,
            completed_order_offset=position,
            order_length=len(order),
        )
        done += 1

    return weight, current, seen


class ResumeStateTest(unittest.TestCase):
    def test_interrupted_sequence_matches_continuous_optimizer_steps(self):
        start = ResumeCursor(global_step=0, epoch=0, next_order_offset=0)

        continuous_weight, continuous_cursor, continuous_seen = (
            simulate_optimizer_steps(
                1.0, start, example_count=5, seed=3407, steps=6
            )
        )

        first_weight, saved_cursor, first_seen = simulate_optimizer_steps(
            1.0, start, example_count=5, seed=3407, steps=3
        )
        restored_cursor = ResumeCursor.from_dict(saved_cursor.to_dict())
        resumed_weight, resumed_cursor, second_seen = simulate_optimizer_steps(
            first_weight,
            restored_cursor,
            example_count=5,
            seed=3407,
            steps=3,
        )

        self.assertAlmostEqual(continuous_weight, resumed_weight, places=14)
        self.assertEqual(continuous_cursor, resumed_cursor)
        self.assertEqual(continuous_seen, first_seen + second_seen)

    def test_cursor_advances_epoch_only_after_last_optimizer_boundary(self):
        cursor = cursor_after_optimizer_step(
            global_step=4,
            epoch=2,
            completed_order_offset=3,
            order_length=4,
        )
        self.assertEqual(4, cursor.global_step)
        self.assertEqual(3, cursor.epoch)
        self.assertEqual(0, cursor.next_order_offset)

    def test_epoch_order_is_reproducible_but_changes_across_epochs(self):
        first = build_epoch_order(10, seed=3407, epoch=0)
        again = build_epoch_order(10, seed=3407, epoch=0)
        next_epoch = build_epoch_order(10, seed=3407, epoch=1)

        self.assertEqual(first, again)
        self.assertNotEqual(first, next_epoch)
        self.assertEqual(list(range(10)), sorted(first))

    def test_optimizer_groups_reset_accumulation_after_partial_epoch_flush(self):
        self.assertEqual(
            [(0, 5)],
            optimizer_step_groups(
                order_length=5,
                start_offset=0,
                gradient_accumulation_steps=16,
            ),
        )

        self.assertEqual(
            [(4, 8), (8, 10)],
            optimizer_step_groups(
                order_length=10,
                start_offset=4,
                gradient_accumulation_steps=4,
            ),
        )

    def test_optimizer_groups_reject_invalid_resume_offset(self):
        with self.assertRaises(ValueError):
            optimizer_step_groups(
                order_length=5,
                start_offset=6,
                gradient_accumulation_steps=2,
            )


    def test_step_limit_is_checked_before_resuming_more_work(self):
        self.assertTrue(optimizer_limit_reached(10, 10))
        self.assertTrue(optimizer_limit_reached(11, 10))
        self.assertFalse(optimizer_limit_reached(9, 10))
        self.assertFalse(optimizer_limit_reached(999, 0))



if __name__ == "__main__":
    unittest.main()
