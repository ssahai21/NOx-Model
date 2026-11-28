import unittest

import numpy as np
import pandas as pd

from src.history_scaling_study import SIGNALS, choose_scaling, prepare_asinh_inputs, transform_raw
from src.peak_validation_study import prepare_inputs


class HistoryScalingTests(unittest.TestCase):
    def data(self):
        return pd.DataFrame({**{f: [1e-5, 2e-5, -3e-5, 5e-5, 1e-2, -1e-2] for f in SIGNALS},
                             'soil': [1., 2., 3., 4., 100., -100.], 'pct_a': [.1, .2, .3, .4, .5, .6]})

    def test_validation_and_test_values_cannot_affect_fitted_transform(self):
        data = self.data();features = data.columns.tolist()
        train, val, test = np.array([0, 1, 2, 3]), np.array([4]), np.array([5])
        arrays, state = prepare_asinh_inputs(data, features, train, val, test)
        changed = data.copy();changed.loc[[4, 5], SIGNALS] *= 1e6
        altered, other = prepare_asinh_inputs(changed, features, train, val, test)
        self.assertEqual(state, other)
        np.testing.assert_array_equal(arrays['train']['standardized'], altered['train']['standardized'])
        raw_arrays, raw_state = prepare_inputs(data, features, train, val, test)
        for partition in ['train', 'validation', 'test']:
            for mode in ['standardized', 'native']:
                np.testing.assert_array_equal(arrays[partition][mode][:, -2:], raw_arrays[partition][mode][:, -2:])
        for f in SIGNALS:
            self.assertAlmostEqual(state['signal_scales'][f], 2.5e-5)

    def test_transform_preserves_sign_and_order_including_large_values(self):
        frame = pd.DataFrame({SIGNALS[0]: [-1e6, -1., 0., 1., 1e6]})
        result = transform_raw(frame, frame.columns.tolist(), {SIGNALS[0]: 1.})[:, 0]
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue((np.diff(result) > 0).all())
        self.assertEqual(result[2], 0)
        self.assertAlmostEqual(result[0], -result[-1])
        self.assertLess(result[-1], 20)

    def test_zero_and_missing_training_signals_have_finite_fallback(self):
        data = pd.DataFrame({f: [0., np.nan, 0., 1e-5] for f in SIGNALS})
        arrays, state = prepare_asinh_inputs(data, SIGNALS, [0, 1], [2], [3])
        self.assertEqual(set(state['signal_scales'].values()), {1.})
        for partition in arrays.values():
            self.assertTrue(np.isfinite(partition['standardized']).all())

    def test_scaling_choice_keeps_original_bulk_guardrail(self):
        raw = {'peak_score': .8, 'full': 10.4}
        anchor = {'full': 10.}
        self.assertEqual(choose_scaling(raw, {'peak_score': .7, 'full': 10.6}, anchor), 'raw')
        self.assertEqual(choose_scaling(raw, {'peak_score': .7, 'full': 10.3}, anchor), 'asinh')
        self.assertEqual(choose_scaling(raw, {'peak_score': .9, 'full': 9.}, anchor), 'raw')


if __name__ == '__main__':
    unittest.main()
