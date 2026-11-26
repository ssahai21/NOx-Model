"""Describe history-input saturation under frozen refit preprocessing."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def audit(root):
    manifest = json.loads((root / 'study_manifest.json').read_text())
    data = pd.read_csv(manifest['data'])
    data = data[data.days_post_fire >= 0].copy()
    thresholds = json.loads(Path(manifest['shared_thresholds']).read_text())['thresholds']
    features = ['no2_previous_observed', 'delta_no2_previous_observed', 'no2_past_week_mean']
    rows = []
    frames = []
    for fire, group in data.groupby('fire_name'):
        report = json.loads((root / 'refit_selected' / f'{fire}_report.json').read_text())
        state = report['preprocessing']
        positions = [state['features'].index(f) for f in features]
        median = np.asarray(state['medians'])[positions]
        mean = np.asarray(state['mean'])[positions]
        scale = np.asarray(state['scale'])[positions]
        filled = group[features].replace([np.inf, -np.inf], np.nan).fillna(dict(zip(features, median))).to_numpy()
        z = (filled - mean) / scale
        limit = state['input_z_limit']
        group = group[['fire_name', 'soil_nox_tau5hr', 'days_post_fire']].copy()
        group['history_signal_clipped'] = (np.abs(z) > limit).any(axis=1)
        group['history_signal_upper_clipped'] = (z > limit).any(axis=1)
        group['all_history_signals_upper_clipped'] = (z > limit).all(axis=1)
        group['extreme'] = group.soil_nox_tau5hr > thresholds[fire]['extreme_flux_row']
        frames.append(group)
        rows.append({'fire': fire, 'rows': len(group), 'clipped_rows': int(group.history_signal_clipped.sum()),
                     'extreme_rows': int(group.extreme.sum()),
                     'clipped_extreme_rows': int((group.history_signal_clipped & group.extreme).sum()),
                     **{f'{f}_clipped_rows': int((np.abs(z[:, i]) > limit).sum()) for i, f in enumerate(features)}})
    combined = pd.concat(frames, ignore_index=True)
    result = {'features': features, 'interpretation': 'Post-evaluation diagnostic of saturation. No new model was selected or fitted from this audit.', 'cohorts': {}}
    for name, mask in [('all', np.ones(len(combined), dtype=bool)),
                       ('first_month', combined.days_post_fire.between(0, 30)), ('extreme', combined.extreme)]:
        group = combined.loc[mask]
        result['cohorts'][name] = {'rows': len(group),
            'history_signal_clipped_rows': int(group.history_signal_clipped.sum()),
            'history_signal_clipped_fraction': float(group.history_signal_clipped.mean()),
            'history_signal_upper_clipped_rows': int(group.history_signal_upper_clipped.sum()),
            'all_history_signals_upper_clipped_rows': int(group.all_history_signals_upper_clipped.sum())}
    pd.DataFrame(rows).to_csv(root / 'history_clipping_per_fire.csv', index=False)
    (root / 'history_clipping_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    audit(Path('results/phase2/peak_validation_study'))
