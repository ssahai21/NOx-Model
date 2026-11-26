"""Verify the four label-cap/loss experiments and compare shared error subsets."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.final_pinn import metrics


KEYS = ['fire_name', 'plot_id', 'date']


def subset_scores(frame, column, mask):
    if mask.sum() < 2:
        return None
    return metrics(frame.loc[mask, 'actual'].to_numpy(), frame.loc[mask, column].to_numpy())


def compare(study_path):
    manifest = json.loads((study_path / 'study_manifest.json').read_text())
    data_path = Path(manifest['data'])
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != manifest['data_sha256']:
        raise ValueError('The study dataset changed.')
    data = pd.read_csv(data_path, parse_dates=['date'])
    data = data[data.days_post_fire >= 0].sort_values(KEYS).reset_index(drop=True)
    all_fires = set(data.fire_name)
    frames = {}
    reports = {}
    for arm, config in manifest['arms'].items():
        directory = Path(config['directory'])
        report = json.loads((directory / 'metrics.json').read_text())
        if {fold['fire'] for fold in report['folds']} != all_fires:
            raise ValueError(f'{arm}: evaluation is incomplete.')
        if report['seeds'] != manifest['seeds']:
            raise ValueError(f'{arm}: seeds changed.')
        model_config = report['model_configs']['baseline']
        if model_config.get('flux_loss_beta') != config['flux_loss_beta']:
            raise ValueError(f'{arm}: loss configuration changed.')
        if report.get('clip_training_target', True) != config['clip_training_target']:
            raise ValueError(f'{arm}: target cap configuration changed.')
        if model_config['acute_weight'] != manifest['acute_weight']:
            raise ValueError(f'{arm}: acute weights changed.')
        arm_frames = []
        for fold in report['folds']:
            fire = fold['fire']
            frame = pd.read_csv(directory / f'{fire}_predictions.csv', parse_dates=['date'])
            columns = [f'baseline_{seed}' for seed in manifest['seeds']]
            if not np.isfinite(frame[columns].to_numpy()).all():
                raise ValueError(f'{arm}/{fire}: predictions are nonfinite.')
            np.testing.assert_allclose(frame.baseline_ensemble, frame[columns].mean(axis=1))
            for seed in manifest['seeds']:
                if not (directory / f'{fire}_baseline_{seed}.pt').is_file():
                    raise ValueError(f'{arm}/{fire}: seed {seed} checkpoint missing.')
            arm_frames.append(frame[KEYS + ['actual', 'baseline_ensemble'] + columns])
        combined = pd.concat(arm_frames, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
        pd.testing.assert_frame_equal(combined[KEYS], data[KEYS], check_dtype=False)
        np.testing.assert_allclose(combined.actual, data.soil_nox_tau5hr, rtol=1e-12, atol=1e-12)
        combined['days_post_fire'] = data.days_post_fire
        frames[arm] = combined
        reports[arm] = report

    control = reports['arcsinh_capped']
    for arm, report in reports.items():
        for setting in ('max_epochs', 'device', 'input_z_limit', 'select_land_cover', 'post_fire_only'):
            if report[setting] != control[setting]:
                raise ValueError(f'{arm}: shared setting changed: {setting}')
        control_sizes = {fold['fire']: fold['rows'] for fold in control['folds']}
        sizes = {fold['fire']: fold['rows'] for fold in report['folds']}
        if sizes != control_sizes:
            raise ValueError(f'{arm}: train/validation/test sizes changed.')

    thresholds = {}
    for fire in sorted(all_fires):
        outer = data[data.fire_name != fire]
        train = outer[outer.groupby('fire_name').date.rank(method='dense', pct=True) <= .8]
        daily = train.groupby(['fire_name', 'date']).soil_nox_tau5hr.mean()
        thresholds[fire] = {
            'high_flux_row': float(train.soil_nox_tau5hr.quantile(.95)),
            'extreme_flux_row': float(train.soil_nox_tau5hr.quantile(.99)),
            'peak_day_mean': float(daily.quantile(.95)),
        }

    for arm, report in reports.items():
        for fold in report['folds']:
            if 'training_target' in fold:
                np.testing.assert_allclose(
                    fold['training_target']['p99'],
                    thresholds[fold['fire']]['extreme_flux_row'],
                )

    per_fire = []
    per_seed = []
    summary = {'rows': len(data), 'fires': len(all_fires), 'thresholds': thresholds, 'arms': {}}
    for arm, frame in frames.items():
        frame['high_flux'] = [
            value > thresholds[fire]['high_flux_row']
            for fire, value in zip(frame.fire_name, frame.actual)
        ]
        frame['extreme_flux'] = [
            value > thresholds[fire]['extreme_flux_row']
            for fire, value in zip(frame.fire_name, frame.actual)
        ]
        score_columns = ['actual', 'baseline_ensemble'] + [f'baseline_{s}' for s in manifest['seeds']]
        daily = frame.groupby(['fire_name', 'date'], as_index=False)[score_columns].mean()
        daily['peak_day'] = [
            value > thresholds[fire]['peak_day_mean']
            for fire, value in zip(daily.fire_name, daily.actual)
        ]
        for fire, group in frame.groupby('fire_name'):
            acute = group.days_post_fire.between(0, 30)
            day_group = daily[daily.fire_name == fire]
            scores = metrics(group.actual.to_numpy(), group.baseline_ensemble.to_numpy())
            early = subset_scores(group, 'baseline_ensemble', acute)
            high = subset_scores(group, 'baseline_ensemble', group.high_flux)
            extreme = subset_scores(group, 'baseline_ensemble', group.extreme_flux)
            peak = subset_scores(day_group, 'baseline_ensemble', day_group.peak_day)
            per_fire.append({
                'arm': arm, 'fire': fire, 'rows': len(group), **scores,
                'first_month_rows': int(acute.sum()),
                'first_month_mae': early['mae'] if early else None,
                'first_month_bias': early['percent_bias'] if early else None,
                'high_flux_rows': int(group.high_flux.sum()),
                'high_flux_mae': high['mae'] if high else None,
                'extreme_flux_rows': int(group.extreme_flux.sum()),
                'extreme_flux_mae': extreme['mae'] if extreme else None,
                'peak_days': int(day_group.peak_day.sum()),
                'peak_day_mae': peak['mae'] if peak else None,
            })
            for seed in manifest['seeds']:
                column = f'baseline_{seed}'
                per_seed.append({
                    'arm': arm, 'fire': fire, 'seed': seed,
                    **metrics(group.actual.to_numpy(), group[column].to_numpy()),
                })
        rows = pd.DataFrame([row for row in per_fire if row['arm'] == arm])
        early_mask = frame.days_post_fire.between(0, 30)
        result = {
            'equal_fire_mean_mae': float(rows.mae.mean()),
            'equal_fire_mean_r2_physical': float(rows.r2_physical.mean()),
            'pooled': metrics(frame.actual.to_numpy(), frame.baseline_ensemble.to_numpy()),
            'first_month_rows': int(early_mask.sum()),
            'first_month_fires': int(frame.loc[early_mask, 'fire_name'].nunique()),
            'first_month': subset_scores(frame, 'baseline_ensemble', early_mask),
            'high_flux_rows': int(frame.high_flux.sum()),
            'high_flux': subset_scores(frame, 'baseline_ensemble', frame.high_flux),
            'extreme_flux_rows': int(frame.extreme_flux.sum()),
            'extreme_flux': subset_scores(frame, 'baseline_ensemble', frame.extreme_flux),
            'peak_days': int(daily.peak_day.sum()),
            'peak_days_fires': int(daily.loc[daily.peak_day, 'fire_name'].nunique()),
            'peak_days_scores': subset_scores(daily, 'baseline_ensemble', daily.peak_day),
        }
        seed_table = pd.DataFrame([row for row in per_seed if row['arm'] == arm])
        seed_means = seed_table.groupby('seed').mae.mean()
        result['seed_macro_mae_mean'] = float(seed_means.mean())
        result['seed_macro_mae_std'] = float(seed_means.std(ddof=0))
        summary['arms'][arm] = result

    table = pd.DataFrame(per_fire)
    table.to_csv(study_path / 'per_fire_comparison.csv', index=False)
    pd.DataFrame(per_seed).to_csv(study_path / 'per_seed_comparison.csv', index=False)
    (study_path / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = ['Transformed\ncapped', 'Transformed\nuncapped', 'Flux loss\ncapped', 'Flux loss\nuncapped']
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout='constrained')
    colors = ['#0072B2', '#56B4E9', '#D55E00', '#E69F00']
    arms = list(summary['arms'].values())
    values = [
        [arm['equal_fire_mean_mae'] for arm in arms],
        [arm['extreme_flux']['mae'] for arm in arms],
        [arm['peak_days_scores']['mae'] for arm in arms],
    ]
    titles = ['Full period · equal-fire MAE', 'Above training cap · MAE', 'Peak daily means · MAE']
    for axis, scores, title in zip(axes, values, titles):
        axis.bar(np.arange(4), scores, color=colors)
        axis.set_xticks(np.arange(4), labels, fontsize=8)
        axis.set_title(title, fontsize=10)
        axis.set_ylabel('ng N m⁻² s⁻¹')
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
        for index, value in enumerate(scores):
            axis.text(index, value, f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        axis.set_ylim(0, max(scores) * 1.15)
    figure.suptitle('Training-label cap experiment · 16 held-out fires · three-seed ensembles')
    figure.savefig(study_path / 'comparison.png', dpi=180)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study_path', type=Path)
    args = parser.parse_args()
    compare(args.study_path)
