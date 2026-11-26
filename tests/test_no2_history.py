import unittest

import pandas as pd

from src.build_no2_history import HISTORY_FEATURES, history_features


class HistoryCausalityTests(unittest.TestCase):
    def observations(self):
        return pd.DataFrame({
            'fire_name': ['A'] * 4, 'plot_id': ['plot'] * 4,
            'date': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-04', '2020-01-12']),
            'no2': [10., 20., 40., 120.], 'ref_no2': [1., 2., 4., 12.],
        })

    def test_current_and_future_observations_cannot_change_history(self):
        original = self.observations()
        changed = original.copy()
        changed.loc[changed.date >= '2020-01-04', ['no2', 'ref_no2']] *= 100
        before, after = history_features(original), history_features(changed)
        pd.testing.assert_frame_equal(
            before.loc[before.date <= '2020-01-04', HISTORY_FEATURES],
            after.loc[after.date <= '2020-01-04', HISTORY_FEATURES],
        )
        day4 = before.loc[before.date == '2020-01-04'].iloc[0]
        self.assertEqual(day4.no2_previous_observed, 20)
        self.assertEqual(day4.delta_no2_previous_observed, 18)
        self.assertEqual(day4.no2_past_week_mean, 15)
        self.assertEqual(day4.no2_history_age_days, 2)

    def test_stale_observations_are_missing_and_groups_are_isolated(self):
        a = self.observations()
        b = a.copy()
        b['fire_name'] = 'B'
        b['no2'] *= 1000
        c = a.copy()
        c['plot_id'] = 'other'
        c['no2'] *= 100
        result = history_features(pd.concat([a, b, c], ignore_index=True))
        actual = result[(result.fire_name == 'A') & (result.plot_id == 'plot')].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, history_features(a).reset_index(drop=True))
        stale = actual.iloc[-1]
        self.assertEqual(stale.no2_history_available, 0)
        self.assertTrue(pd.isna(stale.no2_previous_observed))
        self.assertTrue(pd.isna(stale.no2_past_week_mean))
        self.assertEqual(stale.no2_history_age_days, 8)


if __name__ == '__main__':
    unittest.main()
