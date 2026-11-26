"""Verify matched XGBoost/NN outputs and summarize the comparison."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from src.final_pinn import metrics


def compare(root):
    per_fire = []
    summary_rows = []
    keys = ['fire_name', 'plot_id', 'date']
    source = pd.read_csv('data/phase_2_all_fires_model.csv', parse_dates=['date'])
    source = source[source.days_post_fire >= 0]
    for loss in ['arcsinh', 'flux']:
        directory = root / loss
        report = json.loads((directory / 'metrics.json').read_text())
        summary = json.loads((directory / 'summary.json').read_text())
        if len(report['folds']) != 16 or summary['rows'] != len(source):
            raise ValueError(f'{loss}: evaluation is incomplete.')
        expected_fires = set(source.fire_name)
        if {fold['fire'] for fold in report['folds']} != expected_fires:
            raise ValueError('Test fires differ.')
        for fold in report['folds']:
            fire = fold['fire']
            frame = pd.read_csv(directory / f'{fire}_predictions.csv', parse_dates=['date'])
            expected = source.loc[source.fire_name == fire, keys + ['soil_nox_tau5hr']]
            joined = frame.merge(expected, on=keys, validate='one_to_one')
            if len(joined) != len(frame) or len(frame) != len(expected):
                raise ValueError(f'{loss}/{fire}: incomplete rows.')
            np.testing.assert_allclose(joined.actual, joined.soil_nox_tau5hr, rtol=1e-12, atol=1e-12)
            columns = [f'xgb_{seed}' for seed in report['seeds']]
            np.testing.assert_allclose(frame.xgb_ensemble, frame[columns].mean(axis=1))
            for seed in report['seeds']:
                if not (directory / f'{fire}_{seed}.ubj').is_file():
                    raise ValueError('Missing seed model.')
                actual_scores = metrics(frame.actual.to_numpy(), frame[f'xgb_{seed}'].to_numpy())
                for name, value in actual_scores.items():
                    np.testing.assert_allclose(value, fold['seeds'][str(seed)][name])
            for model in ['nn', 'xgb']:
                scores = metrics(frame.actual.to_numpy(), frame[f'{model}_ensemble'].to_numpy())
                for name, value in scores.items():
                    np.testing.assert_allclose(value, fold['models'][model]['ensemble'][name])
                record = {'loss': loss, 'model': model, 'fire': fire, 'rows': len(frame), **scores}
                for segment in ['first_month', 'extreme_flux', 'peak_days_scores']:
                    subset = fold['models'][model][segment]
                    record[f'{segment}_mae'] = subset['mae'] if subset else None
                    record[f'{segment}_bias'] = subset['percent_bias'] if subset else None
                per_fire.append(record)
        for model in ['nn', 'xgb']:
            scores = summary['models'][model]
            summary_rows.append({
                'loss': loss, 'model': model,
                'equal_fire_mean_mae': scores['equal_fire_mean_mae'],
                'pooled_mae': scores['pooled']['mae'],
                'r2_physical': scores['pooled']['r2_physical'],
                'r2_arcsinh': scores['pooled']['r2_arcsinh'],
                'first_month_mae': scores['first_month']['mae'],
                'first_month_bias': scores['first_month']['percent_bias'],
                'extreme_flux_mae': scores['extreme_flux']['mae'],
                'extreme_flux_bias': scores['extreme_flux']['percent_bias'],
                'peak_day_mae': scores['peak_days_scores']['mae'],
                'peak_day_bias': scores['peak_days_scores']['percent_bias'],
            })
    full = pd.DataFrame(per_fire)
    full.to_csv(root / 'per_fire_comparison.csv', index=False)
    table = pd.DataFrame(summary_rows)
    table.to_csv(root / 'comparison.csv', index=False)
    final = {
        'fires': 16, 'rows': len(source),
        'results': table.to_dict('records'),
        'xgb_wins_by_loss': {
            loss: json.loads((root / loss / 'summary.json').read_text())['xgb_mae_wins']
            for loss in ['arcsinh', 'flux']
        },
        'budget_hits_by_loss': {
            loss: json.loads((root / loss / 'summary.json').read_text())['seeds_at_tree_budget']
            for loss in ['arcsinh', 'flux']
        },
    }
    (root / 'summary.json').write_text(json.dumps(final, indent=2, allow_nan=False))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['NN\ntransformed', 'XGBoost\ntransformed', 'NN\nflux loss', 'XGBoost\nflux loss']
    colors = ['#0072B2', '#56B4E9', '#D55E00', '#E69F00']
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout='constrained')
    for axis, column, title in zip(
        axes, ['equal_fire_mean_mae', 'extreme_flux_mae', 'peak_day_mae'],
        ['Full period · equal-fire MAE', 'Above training cap · MAE', 'Peak daily means · MAE'],
    ):
        values = table[column].to_numpy()
        axis.bar(np.arange(4), values, color=colors)
        axis.set_xticks(np.arange(4), labels, fontsize=8)
        axis.set_title(title, fontsize=10)
        axis.set_ylabel('ng N m⁻² s⁻¹')
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
        axis.set_ylim(0, values.max() * 1.15)
        for index, value in enumerate(values):
            axis.text(index, value, f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    figure.suptitle('Matched inputs · 16 held-out fires · three-seed ensembles')
    figure.savefig(root / 'comparison.png', dpi=180)
    plt.close(figure)
    print(table.round(4).to_string(index=False))
    print(json.dumps(final, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, default=Path('results/phase2/xgboost_evaluation'), nargs='?')
    args = parser.parse_args()
    compare(args.directory)
