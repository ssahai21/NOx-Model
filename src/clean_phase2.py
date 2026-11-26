"""Aggregate repeated daily satellite observations and join existing model inputs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.data_loader import get_features

def main():
    source = Path('data/phase_2_nox_df.csv')
    raw = pd.read_csv(source)
    raw['fire_name'] = raw.fire_name.replace({'Camp': 'Camp Fire', 'Dixie': 'Dixie Fire'})
    raw['date'] = pd.to_datetime(raw.date, errors='raise')
    keys = ['fire_name', 'plot_id', 'date']
    if raw[keys].isna().any().any():
        raise ValueError('Missing record identifiers.')
    numeric = raw.select_dtypes(include=np.number).columns.tolist()
    if not np.isfinite(raw[numeric]).all().all():
        raise ValueError('Nonfinite numeric observations require review.')
    # Multiple satellite observations on a plot/date become one daily mean.
    daily = raw.groupby(keys, as_index=False)[numeric].mean()
    daily['month'] = daily.date.dt.month
    daily['day_of_year'] = daily.date.dt.dayofyear
    daily.to_csv('data/phase_2_daily_clean.csv', index=False)

    old = pd.read_csv('data/nox_clean.csv', parse_dates=['date'])
    features = get_features(old)
    # Join only exact plot/date matches. Never fill new-fire inputs from other fires.
    old_inputs = old[keys + features + ['days_post_fire']].loc[:, lambda x: ~x.columns.duplicated()]
    if old_inputs.duplicated(keys).any():
        raise ValueError('Existing feature keys are not unique.')
    model_data = daily.merge(old_inputs, on=keys, how='inner', validate='one_to_one',
                             suffixes=('', '_saved'))
    for name in ['centroid_lat', 'centroid_lon']:
        if not np.allclose(model_data[name], model_data[name + '_saved'], atol=1e-8, rtol=0):
            raise ValueError('Plot coordinates changed.')
        model_data.drop(columns=name + '_saved', inplace=True)
    if set(get_features(model_data)) != set(features):
        raise ValueError('Joined model feature schema changed.')
    model_data.to_csv('data/phase_2_model_clean.csv', index=False)
    audit = {
        'source': str(source), 'raw_rows': len(raw), 'daily_rows': len(daily),
        'aggregated_extra_observations': len(raw) - len(daily),
        'aggregation': 'Arithmetic mean of numeric observations within fire/plot/date.',
        'matched_feature_rows': len(model_data),
        'unmatched_daily_rows': len(daily) - len(model_data),
        'daily_rows_per_fire': daily.groupby('fire_name').size().to_dict(),
        'matched_rows_per_fire': model_data.groupby('fire_name').size().to_dict(),
    }
    out = Path('results/phase2')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'cleaning_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))

if __name__ == '__main__':
    main()
