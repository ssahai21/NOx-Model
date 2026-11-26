import unittest

import numpy as np
import pandas as pd

from src.peak_validation_study import PeakScorer, prepare_inputs, sample_weights, select_arm, split_fires, training_statistics


class PeakValidationTests(unittest.TestCase):
    def data(self):
        return pd.DataFrame([
            {'fire_name': fire, 'plot_id': 'plot', 'date': pd.Timestamp('2020-01-01') + pd.Timedelta(days=day),
             'days_post_fire': day, 'soil_nox_tau5hr': float(day + (100 if day == 15 else 0)),
             'x': float(day), 'pct_a': .5}
            for fire in ['A', 'B', 'C', 'D', 'E', 'F', 'G'] for day in range(40)
        ])

    def test_outer_and_validation_labels_do_not_change_split_or_training_fits(self):
        data = self.data()
        train, val, test = split_fires(data, 'A')
        self.assertFalse(set(data.iloc[train].fire_name) & set(data.iloc[val].fire_name))
        self.assertNotIn('A', set(data.iloc[train].fire_name) | set(data.iloc[val].fire_name))
        changed = data.copy()
        changed.loc[np.concatenate([val, test]), 'soil_nox_tau5hr'] *= 100000
        changed.loc[test, 'x'] = 1e12
        other = split_fires(changed, 'A')
        for before, after in zip((train, val, test), other):
            np.testing.assert_array_equal(before, after)
        self.assertEqual(training_statistics(data.iloc[train]), training_statistics(changed.iloc[train]))
        a, state_a = prepare_inputs(data, ['x', 'pct_a'], train, val, test)
        b, state_b = prepare_inputs(changed, ['x', 'pct_a'], train, val, test)
        self.assertEqual(state_a, state_b)
        for part in ['train', 'validation']:
            np.testing.assert_array_equal(a[part]['standardized'], b[part]['standardized'])

    def test_score_is_zero_for_perfect_predictions_and_balances_fire_sizes(self):
        data = self.data()
        stats = training_statistics(data)
        scorer = PeakScorer(data, stats['high_threshold'], stats['peak_threshold'], stats['scales'])
        self.assertEqual(scorer.score(data.soil_nox_tau5hr.to_numpy()), 0)
        prediction = data.soil_nox_tau5hr.to_numpy().copy()
        prediction[data.fire_name == 'A'] += 7
        self.assertAlmostEqual(scorer.components(prediction)['full'], 1)
        duplicated = pd.concat([data, data[data.fire_name == 'A']], ignore_index=True)
        extra_prediction = np.concatenate([prediction, prediction[data.fire_name == 'A']])
        extra = PeakScorer(duplicated, stats['high_threshold'], stats['peak_threshold'], stats['scales'])
        self.assertAlmostEqual(extra.components(extra_prediction)['full'], 1)
        self.assertGreater(scorer.score(prediction), 0)

    def test_training_weights_are_bounded_and_positive_tail_only(self):
        data = pd.DataFrame({'days_post_fire': [0, 0, 100, 100], 'soil_nox_tau5hr': [-100, 100, 100, 0]})
        np.testing.assert_array_equal(sample_weights(data, 50, 5), [3, 15, 5, 1])
        np.testing.assert_array_equal(sample_weights(data, 50, 1), [3, 3, 1, 1])

    def test_validation_guardrail_excludes_large_bulk_error_regression(self):
        values = {
            'inner_arcsinh_capped': {'full': 10., 'peak_score': 1.},
            'inner_peak_capped': {'full': 10.4, 'peak_score': .8},
            'inner_peak_uncapped': {'full': 11., 'peak_score': .1},
            'inner_peak_weighted': {'full': 10.3, 'peak_score': .9},
        }
        self.assertEqual(select_arm(values), 'inner_peak_capped')
        values['inner_peak_capped']['peak_score'] = 1.1
        values['inner_peak_weighted']['peak_score'] = 1.2
        self.assertEqual(select_arm(values), 'inner_arcsinh_capped')


if __name__ == '__main__':
    unittest.main()
