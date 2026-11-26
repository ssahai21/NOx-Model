"""Verify complete LOGO predictions and summarize ensembles and seed variation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.final_pinn import metrics


def summarize(directory, data_path):
    report = json.loads((directory / 'metrics.json').read_text())
    manifest_path = directory / 'run_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, expected_hash in manifest['sha256'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f'Run source changed during evaluation: {name}')
    source = pd.read_csv(data_path)
    if report['post_fire_only']:
        source = source.loc[source.days_post_fire >= 0]
    fires = set(source.fire_name)
    completed = {fold['fire'] for fold in report['folds']}
    if completed != fires:
        raise ValueError(f'Incomplete evaluation. Missing fires: {sorted(fires - completed)}')
    models = list(report['model_configs'])
    rows = []
    prediction_frames = []
    keys = ['fire_name', 'plot_id', 'date']
    for fold in report['folds']:
        fire = fold['fire']
        predictions = pd.read_csv(directory / f'{fire}_predictions.csv')
        expected = source.loc[source.fire_name == fire, keys + ['soil_nox_tau5hr', 'days_post_fire']]
        joined = predictions.merge(expected, on=keys, validate='one_to_one')
        if len(joined) != len(predictions) or len(joined) != len(expected):
            raise ValueError(f'{fire}: incomplete prediction coverage.')
        np.testing.assert_allclose(joined.actual, joined.soil_nox_tau5hr, rtol=1e-12, atol=1e-12)
        prediction_frames.append(joined)
        for model in models:
            seed_columns = [f'{model}_{seed}' for seed in report['seeds']]
            values = predictions[seed_columns].mean(axis=1)
            np.testing.assert_allclose(predictions[f'{model}_ensemble'], values, rtol=1e-12, atol=1e-12)
            scores = metrics(predictions.actual.to_numpy(), values.to_numpy())
            for key, value in scores.items():
                saved = fold['models'][model]['ensemble'][key]
                if value is None:
                    assert saved is None
                else:
                    np.testing.assert_allclose(value, saved)
            for seed in report['seeds']:
                if not (directory / f'{fire}_{model}_{seed}.pt').is_file():
                    raise ValueError(f'{fire}: missing {model} seed {seed} checkpoint.')
                seed_scores = metrics(
                    predictions.actual.to_numpy(), predictions[f'{model}_{seed}'].to_numpy())
                saved_seed = fold['models'][model]['seeds'][str(seed)]
                for key, value in seed_scores.items():
                    if value is None:
                        assert saved_seed[key] is None
                    else:
                        np.testing.assert_allclose(value, saved_seed[key])
            rows.append({
                'fire': fire, 'model': model, 'rows': len(predictions),
                **scores,
                'seed_mae_mean': fold['models'][model]['summary']['mae']['mean'],
                'seed_mae_std': fold['models'][model]['summary']['mae']['std'],
            })
        for name, scores in fold.get('benchmarks', {}).items():
            rows.append({'fire': fire, 'model': name, 'rows': len(predictions), **scores})
    table = pd.DataFrame(rows)
    table.to_csv(directory / 'per_fire_summary.csv', index=False)
    pooled = pd.concat(prediction_frames, ignore_index=True)
    summary = {
        'fires': len(fires), 'rows': len(pooled), 'seeds': report['seeds'],
        'coverage': {
            fire: {
                'min_day': int(rows.days_post_fire.min()),
                'max_day': int(rows.days_post_fire.max()),
                'first_month_rows': int(rows.days_post_fire.between(0, 30).sum()),
            }
            for fire, rows in pooled.groupby('fire_name')
        },
        'models': {},
    }
    for model in table.model.unique():
        values = table.loc[table.model == model]
        column = f'{model}_ensemble' if model in models else model
        overall = metrics(pooled.actual.to_numpy(), pooled[column].to_numpy())
        acute = pooled.days_post_fire.between(0, 30)
        result = {
            'equal_fire_mean_mae': values.mae.mean(),
            'equal_fire_mean_r2_physical': values.r2_physical.mean(),
            'pooled': overall,
            'days_0_to_30_rows': int(acute.sum()),
            'days_0_to_30_fires': int(pooled.loc[acute, 'fire_name'].nunique()),
            'days_0_to_30_pooled': metrics(
                pooled.loc[acute, 'actual'].to_numpy(), pooled.loc[acute, column].to_numpy()),
        }
        if model in models:
            seed_macro = [
                float(np.mean([fold['models'][model]['seeds'][str(seed)]['mae']
                               for fold in report['folds']]))
                for seed in report['seeds']
            ]
            result['seed_equal_fire_mean_mae'] = seed_macro
            result['seed_equal_fire_mae_mean'] = float(np.mean(seed_macro))
            result['seed_equal_fire_mae_std'] = float(np.std(seed_macro))
        summary['models'][model] = result
    nn = table.loc[table.model == models[0]].set_index('fire')
    pinn = table.loc[table.model == models[1]].set_index('fire')
    summary['pinn_mae_wins'] = int((pinn.mae < nn.mae).sum())
    summary['pinn_mae_ties'] = int((pinn.mae == nn.mae).sum())
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fire_order = [fold['fire'] for fold in report['folds']]
    positions = np.arange(len(fire_order))
    figure, axis = plt.subplots(figsize=(9, 7), layout='constrained')
    for name, label, offset, color, marker in [
        (models[0], 'Plain NN', -.14, '#0072B2', 'o'),
        (models[1], 'Simpler PINN', .14, '#D55E00', 'o'),
        ('wind_season_ridge', 'Wind + season ridge', 0, '#666666', 'x'),
    ]:
        values = table.loc[table.model == name].set_index('fire').reindex(fire_order)
        axis.scatter(values.mae, positions + offset, label=label, color=color,
                     marker=marker, s=35)
    axis.set_yticks(positions, fire_order)
    axis.invert_yaxis()
    axis.set_xlabel('Mean absolute error (ng N m⁻² s⁻¹), lower is better')
    axis.set_title('Held-out fire evaluation · 16 fires · three-seed ensembles')
    axis.grid(axis='x', alpha=.2)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc='lower right')
    figure.savefig(directory / 'mae_by_fire.png', dpi=180)
    plt.close(figure)
    print(table.pivot(index='fire', columns='model', values='mae').round(3).to_string())
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--data', type=Path, default=Path('data/phase_2_all_fires_model.csv'))
    args = parser.parse_args()
    summarize(args.directory, args.data)
