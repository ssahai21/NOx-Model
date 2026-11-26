import unittest

from src.refit_peak_selection import learning_rates


class RefitScheduleTests(unittest.TestCase):
    def test_schedule_is_fixed_from_inner_validation_and_truncated_to_chosen_epoch(self):
        history = [{'selection_score': 1.}] * 13
        self.assertEqual(learning_rates(history, 4), [.001] * 4)
        self.assertEqual(learning_rates(history, 13), [.001] * 12 + [.0005])
        improved = history.copy()
        improved[9] = {'selection_score': .5}
        self.assertEqual(learning_rates(improved, 13), [.001] * 13)
        with self.assertRaises(ValueError):
            learning_rates(history, 14)


if __name__ == '__main__':
    unittest.main()
