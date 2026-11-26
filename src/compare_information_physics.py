"""Verify and compare the past-satellite and physics-repair experiments."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.final_pinn import metrics
from src.compare_target_caps import subset_scores

KEYS = ['fire_name', 'plot_id', 'date']


def compare(path):
    manifest = json.loads((path / 'study_manifest.json').read_text())
    for name, expected in {**manifest['sources'], **manifest['code_sha256']}.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Study source changed: {name}')
    data = pd.read_csv('data/phase_2_all_fires_model.csv', parse_dates=['date'])
    data = data[data.days_post_fire >= 0].sort_values(KEYS).reset_index(drop=True)
    history_data = pd.read_csv('data/phase_2_no2_history_model.csv', parse_dates=['date'])
    history_data = history_data[history_data.days_post_fire >= 0].sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(data, history_data[data.columns], check_exact=False, rtol=1e-12, atol=1e-12)
    thresholds = json.loads(Path('results/phase2/target_cap_study/summary.json').read_text())['thresholds']
    frames, reports = {}, {}
    for arm, config in manifest['arms'].items():
        directory, model = Path(config['directory']), config['model']
        report = json.loads((directory / 'metrics.json').read_text())
        if {f['fire'] for f in report['folds']} != set(data.fire_name):
            raise ValueError(f'{arm}: incomplete folds')
        assert report['seeds'] == manifest['seeds']
        assert report['clip_training_target']
        assert report['model_configs'][model]['flux_loss_beta'] == 10
        assert report['model_configs'][model]['acute_weight'] == manifest['acute_weight']
        if arm.endswith('pinn'):
            configuration = report['model_configs'][model]
            assert configuration['lambda_physics'] == manifest['physics_weight']
            assert configuration['lambda_box'] == configuration['lambda_deposition'] == 0
            assert configuration.get('same_day_pairs', False) == (arm != 'legacy_pinn')
            assert configuration.get('stable_regularization', False) == (arm != 'legacy_pinn')
            if arm == 'pair_only_pinn':
                assert configuration['lambda_nonneg'] == configuration['lambda_decay'] == 0
        parts = []
        for fold in report['folds']:
            fire = fold['fire']
            frame = pd.read_csv(directory / f'{fire}_predictions.csv', parse_dates=['date'])
            columns = [f'{model}_{seed}' for seed in manifest['seeds']]
            assert np.isfinite(frame[columns].to_numpy()).all()
            np.testing.assert_allclose(frame[f'{model}_ensemble'], frame[columns].mean(axis=1))
            score = metrics(frame.actual.to_numpy(), frame[f'{model}_ensemble'].to_numpy())
            for key, value in score.items():
                np.testing.assert_allclose(value, fold['models'][model]['ensemble'][key])
            for seed in manifest['seeds']:
                checkpoint = torch.load(directory / f'{fire}_{model}_{seed}.pt', map_location='cpu', weights_only=True)
                weight = next(v for v in checkpoint.values() if v.ndim == 2)
                assert weight.shape[1] == config['features'], (arm, fire, weight.shape)
            np.testing.assert_allclose(fold['training_target']['p99'], thresholds[fire]['extreme_flux_row'])
            frame = frame[KEYS + ['actual', f'{model}_ensemble'] + columns].rename(columns={
                f'{model}_ensemble': 'prediction', **{f'{model}_{s}': f'seed_{s}' for s in manifest['seeds']}})
            parts.append(frame)
        frame = pd.concat(parts, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
        pd.testing.assert_frame_equal(frame[KEYS], data[KEYS], check_dtype=False)
        np.testing.assert_allclose(frame.actual, data.soil_nox_tau5hr, rtol=1e-12, atol=1e-12)
        frame['days_post_fire'] = data.days_post_fire
        frame['high'] = [v > thresholds[f]['high_flux_row'] for f, v in zip(frame.fire_name, frame.actual)]
        frame['extreme'] = [v > thresholds[f]['extreme_flux_row'] for f, v in zip(frame.fire_name, frame.actual)]
        frames[arm], reports[arm] = frame, report
    control = reports['environment_nn']
    for arm, report in reports.items():
        for setting in ['seeds', 'max_epochs', 'device', 'input_z_limit', 'select_land_cover', 'post_fire_only', 'excluded_rows']:
            assert report[setting] == control[setting], (arm, setting)
        assert {f['fire']: f['rows'] for f in report['folds']} == {f['fire']: f['rows'] for f in control['folds']}
    per_fire, per_seed, telemetry, daily_predictions = [], [], [], []
    summary = {'rows': len(data), 'fires': data.fire_name.nunique(), 'thresholds': thresholds, 'arms': {},
               'caveat': manifest['caveat']}
    for arm, frame in frames.items():
        columns = ['actual', 'prediction'] + [f'seed_{s}' for s in manifest['seeds']]
        daily = frame.groupby(['fire_name', 'date'], as_index=False)[columns].mean()
        daily['peak'] = [v > thresholds[f]['peak_day_mean'] for f, v in zip(daily.fire_name, daily.actual)]
        daily_predictions.append(daily.assign(arm=arm))
        for fire, group in frame.groupby('fire_name'):
            day_group = daily[daily.fire_name == fire]
            row = {'arm': arm, 'fire': fire, **metrics(group.actual.to_numpy(), group.prediction.to_numpy())}
            for name, source, mask in [('first_month', group, group.days_post_fire.between(0, 30)),
                                       ('high_flux', group, group.high), ('extreme_flux', group, group.extreme),
                                       ('negative_label', group, group.actual < 0),
                                       ('nonnegative_label', group, group.actual >= 0),
                                       ('peak_days', day_group, day_group.peak)]:
                score = subset_scores(source, 'prediction', mask)
                row[f'{name}_rows'] = int(mask.sum())
                row[f'{name}_mae'] = score['mae'] if score else None
                row[f'{name}_bias'] = score['percent_bias'] if score else None
            per_fire.append(row)
            for seed in manifest['seeds']:
                per_seed.append({'arm': arm, 'fire': fire, 'seed': seed,
                                 **metrics(group.actual.to_numpy(), group[f'seed_{seed}'].to_numpy())})
        table = pd.DataFrame([r for r in per_fire if r['arm'] == arm])
        result = {'equal_fire_mean_mae': float(table.mae.mean()),
                  'pooled': metrics(frame.actual.to_numpy(), frame.prediction.to_numpy())}
        model_name = manifest['arms'][arm]['model']
        selected_epochs = [s['epochs_run'] for fold in reports[arm]['folds']
                           for s in fold['models'][model_name]['seeds'].values()]
        result['max_selected_epochs_run'] = max(selected_epochs)
        result['selected_models_at_epoch_ceiling'] = sum(e == reports[arm]['max_epochs'] for e in selected_epochs)
        for name, source, mask in [('first_month', frame, frame.days_post_fire.between(0, 30)),
                                   ('high_flux', frame, frame.high), ('extreme_flux', frame, frame.extreme),
                                   ('negative_label', frame, frame.actual < 0),
                                   ('nonnegative_label', frame, frame.actual >= 0),
                                   ('peak_days', daily, daily.peak)]:
            result[name] = subset_scores(source, 'prediction', mask)
            result[f'{name}_rows'] = int(mask.sum())
        seed_table = pd.DataFrame([r for r in per_seed if r['arm'] == arm])
        result['per_seed_macro_mae'] = {str(k): float(v) for k, v in seed_table.groupby('seed').mae.mean().items()}
        summary['arms'][arm] = result
        config = manifest['arms'][arm]
        if arm.endswith('pinn'):
            for fold in reports[arm]['folds']:
                train_rows = fold['rows'][0]
                batches = train_rows // 512 + int(train_rows % 512 > 1)
                for seed in manifest['seeds']:
                    history = json.loads((Path(config['directory']) / f"{fold['fire']}_{config['model']}_{seed}_history.json").read_text())
                    for mode, candidate in history['candidates'].items():
                        components = pd.DataFrame(candidate['loss_components'])
                        telemetry.append({'arm': arm, 'fire': fold['fire'], 'seed': seed, 'mode': mode,
                                          'selected': mode == history['selected_mode'], 'epochs': len(components),
                                          'batches': batches * len(components),
                                          **{column: float(components[column].mean()) for column in components}})
    table = pd.DataFrame(per_fire)
    control_mae = table[table.arm == 'environment_nn'].set_index('fire').mae
    for arm in frames:
        scores = table[table.arm == arm].set_index('fire').mae
        summary['arms'][arm]['fires_better_than_control'] = int((scores < control_mae).sum())
        summary['arms'][arm]['macro_mae_change_percent'] = float(100 * (scores.mean() / control_mae.mean() - 1))
    telemetry_table = pd.DataFrame(telemetry)
    summary['physics_telemetry'] = {}
    for arm, group in telemetry_table.groupby('arm'):
        summary['physics_telemetry'][arm] = {}
        for scope, subset in [('all_candidates', group), ('selected_candidates', group[group.selected])]:
            fields = ['hard_clamp_active', 'raw_above_cap', 'physics_total', 'physics_effective', 'non_neg', 'decay', 'spatial', 'wind']
            if arm != 'legacy_pinn':
                fields.append('same_day_pair_count')
            summary['physics_telemetry'][arm][scope] = {
                'training_batches': int(subset.batches.sum()),
                **{f: float(np.average(subset[f], weights=subset.batches)) for f in fields}}
    table.to_csv(path / 'per_fire_comparison.csv', index=False)
    pd.DataFrame(per_seed).to_csv(path / 'per_seed_comparison.csv', index=False)
    telemetry_table.to_csv(path / 'physics_training_telemetry.csv', index=False)
    pd.concat(daily_predictions, ignore_index=True).to_csv(path / 'daily_predictions.csv', index=False)
    (path / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    plot(path, summary)
    print(json.dumps({arm: result for arm, result in summary['arms'].items()}, indent=2))


def plot(path, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['Environment\nNN', 'Past NO₂\nNN', 'Legacy\nPINN', 'Repaired\nPINN', 'Pair-only\nPINN']
    colors = ['#0072B2', '#009E73', '#999999', '#D55E00', '#CC79A7']
    arms = list(summary['arms'].values())
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout='constrained')
    panels = [('Full period · equal-fire MAE', [r['equal_fire_mean_mae'] for r in arms]),
              ('Extreme flux rows · MAE', [r['extreme_flux']['mae'] for r in arms]),
              ('Peak daily means · MAE', [r['peak_days']['mae'] for r in arms])]
    for axis, (title, values) in zip(axes, panels):
        axis.bar(range(5), values, color=colors)
        axis.set_xticks(range(5), labels, fontsize=8)
        axis.set_title(title, fontsize=10)
        axis.set_ylabel('ng N m⁻² s⁻¹')
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
        axis.set_ylim(0, max(values) * 1.16)
        for i, value in enumerate(values):
            axis.text(i, value, f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    figure.suptitle('Information and physics audit · 16 held-out fires · three-seed ensembles')
    figure.savefig(path / 'comparison.png', dpi=180)
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study_path', type=Path)
    compare(parser.parse_args().study_path)
