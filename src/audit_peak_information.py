"""Diagnose label components and missing satellite information without fitting an oracle."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    root = Path('results/phase2/information_physics_study')
    data = pd.read_csv('data/phase_2_no2_history_model.csv', parse_dates=['date'])
    data = data[data.days_post_fire >= 0].copy()
    predictions = pd.concat([
        pd.read_csv(file, parse_dates=['date'])[['fire_name', 'plot_id', 'date', 'baseline_ensemble']]
        for file in Path('results/phase2/target_cap_study/flux_capped').glob('*_predictions.csv')
    ], ignore_index=True)
    data = data.merge(predictions, on=['fire_name', 'plot_id', 'date'], validate='one_to_one')
    factor = 14.0067e9 / 6.022e23
    tau = 3600 * (10 + 5 * np.cos(2 * np.pi * (data.day_of_year - 355) / 365.25))
    data['advection_flux'] = data.advection_term * factor
    data['deposition_flux'] = data.deposition_term * factor
    data['chemical_flux_tau5'] = data.no2 * 14.0067e9 / tau
    data['current_delta_no2'] = data.no2 - data.ref_no2
    data['prediction_error'] = data.soil_nox_tau5hr - data.baseline_ensemble
    reconstruction = data.advection_flux + data.deposition_flux + data.chemical_flux_tau5
    offset = reconstruction - data.soil_nox_tau5hr
    correlations = []
    peak_days = []
    for fire, rows in data.groupby('fire_name'):
        record = {
            'fire': fire, 'rows': len(rows),
            'negative_label_fraction': float((rows.soil_nox_tau5hr < 0).mean()),
        }
        for feature in ['current_delta_no2', 'advection_flux', 'no2_previous_observed',
                        'delta_no2_previous_observed', 'no2_past_week_mean']:
            record[f'{feature}_correlation'] = rows[feature].corr(rows.soil_nox_tau5hr)
            record[f'{feature}_observations'] = int(rows[feature].notna().sum())
        correlations.append(record)
        daily = rows.groupby('date')[[
            'soil_nox_tau5hr', 'baseline_ensemble', 'advection_flux',
            'deposition_flux', 'chemical_flux_tau5', 'prediction_error',
        ]].mean()
        for date in daily.prediction_error.nlargest(3).index:
            row = daily.loc[date]
            observed = rows[rows.date == date]
            peak_days.append({
                'fire': fire, 'date': str(date.date()), 'plots': len(observed),
                **row.to_dict(),
                'label_median': float(observed.soil_nox_tau5hr.median()),
                'fraction_underpredicted': float((observed.soil_nox_tau5hr > observed.baseline_ensemble).mean()),
            })
    pd.DataFrame(correlations).to_csv(root / 'information_correlations.csv', index=False)
    peaks = pd.DataFrame(peak_days)
    peaks.to_csv(root / 'largest_missed_days.csv', index=False)
    summary = {
        'rows': len(data), 'negative_label_fraction': float((data.soil_nox_tau5hr < 0).mean()),
        'label_advection_correlation': data.soil_nox_tau5hr.corr(data.advection_flux),
        'current_delta_no2_correlation': data.soil_nox_tau5hr.corr(data.current_delta_no2),
        'previous_delta_no2_correlation': data.soil_nox_tau5hr.corr(data.delta_no2_previous_observed),
        'previous_no2_correlation': data.soil_nox_tau5hr.corr(data.no2_previous_observed),
        'reconstruction_offset_mean': float(offset.mean()),
        'reconstruction_offset_std': float(offset.std()),
        'largest_missed_days_advection_share_of_label_sum': float(peaks.advection_flux.sum()/peaks.soil_nox_tau5hr.sum()),
        'interpretation': 'Contemporaneous NO2 and box components are label-construction diagnostics, not independent predictive features or field validation.',
    }
    (root / 'information_audit.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
