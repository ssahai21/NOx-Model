"""Build strictly past satellite features without modifying existing model rows."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ['fire_name', 'plot_id', 'date']
HISTORY_FEATURES = [
    'no2_previous_observed', 'delta_no2_previous_observed', 'no2_past_week_mean',
    'no2_history_age_days', 'no2_history_available',
]


def history_features(observations):
    if observations.duplicated(KEYS).any():
        raise ValueError('Satellite history must have unique daily plot keys.')
    frames = []
    for _, rows in observations.groupby(['fire_name', 'plot_id']):
        rows = rows.sort_values('date').copy()
        previous_date = rows.date.shift(1)
        age = (rows.date - previous_date).dt.days
        available = age.between(1, 7)
        rows['no2_previous_observed'] = rows.no2.shift(1).where(available)
        rows['delta_no2_previous_observed'] = (rows.no2 - rows.ref_no2).shift(1).where(available)
        # The upper endpoint is excluded, so today's satellite value never enters the mean.
        rows['no2_past_week_mean'] = (
            rows.set_index('date').no2.rolling('7D', closed='left').mean().to_numpy()
        )
        rows['no2_history_age_days'] = age.clip(upper=8).fillna(8)
        rows['no2_history_available'] = available.astype(int)
        frames.append(rows[KEYS + HISTORY_FEATURES])
    return pd.concat(frames, ignore_index=True)


def main():
    model_path = Path('data/phase_2_all_fires_model.csv')
    observation_path = Path('data/phase_2_daily_clean.csv')
    model = pd.read_csv(model_path, parse_dates=['date'])
    observations = pd.read_csv(observation_path, parse_dates=['date'])
    features = history_features(observations)
    merged = model.merge(features, on=KEYS, how='left', validate='one_to_one', indicator=True)
    if len(merged) != len(model) or not merged['_merge'].eq('both').all():
        raise ValueError('History feature coverage changed the model rows.')
    merged.drop(columns='_merge', inplace=True)
    merged.to_csv('data/phase_2_no2_history_model.csv', index=False)
    output = Path('results/phase2/information_physics_study')
    output.mkdir(parents=True, exist_ok=True)
    audit = {
        'source_sha256': {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (model_path, observation_path)
        },
        'features': HISTORY_FEATURES, 'lookback_days': 7,
        'row_count': len(merged),
        'history_available_fraction': float(merged.no2_history_available.mean()),
        'history_available_per_fire': merged.groupby('fire_name').no2_history_available.mean().to_dict(),
        'interpretation': 'Past satellite assisted estimation. Prior observations from the held-out fire are available as inputs, while its target labels stay outside training and validation.',
    }
    (output / 'history_feature_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
