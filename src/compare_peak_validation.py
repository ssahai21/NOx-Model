"""Freeze validation-only arm choices, then evaluate untouched outer-fire predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.final_pinn import metrics
from src.baseline_nn import BaselineNN
from src.compare_target_caps import subset_scores
from src.peak_validation_study import ARMS, KEYS, PeakScorer, predict, prepare_inputs, select_arm, split_fires, training_statistics
from src.refit_peak_selection import learning_rates


def load_predictions(path, expected, seeds, prefix='seed_', ensemble='prediction'):
    frame = pd.read_csv(path, parse_dates=['date'])
    pd.testing.assert_frame_equal(frame[KEYS].reset_index(drop=True), expected[KEYS].reset_index(drop=True), check_dtype=False)
    np.testing.assert_allclose(frame.actual, expected.soil_nox_tau5hr, rtol=1e-12, atol=1e-12)
    columns = [f'{prefix}{seed}' for seed in seeds]
    if not np.isfinite(frame[columns + [ensemble]].to_numpy()).all():
        raise ValueError(f'Nonfinite prediction: {path}')
    np.testing.assert_allclose(frame[ensemble], frame[columns].mean(axis=1))
    return frame[KEYS + ['actual', ensemble] + columns].rename(columns={ensemble: 'prediction',
                            **{f'{prefix}{s}': f'seed_{s}' for s in seeds}})


def compare(root, select_only=False):
    manifest = json.loads((root / 'study_manifest.json').read_text())
    for name, expected in {**manifest['source_sha256'], **manifest['training_code_sha256']}.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    data = pd.read_csv(manifest['data'], parse_dates=['date'])
    data = data[data.days_post_fire >= 0].sort_values(KEYS).reset_index(drop=True)
    seeds = manifest['seeds']
    plans = {f['fire']: f for f in json.loads((root / 'split_plan.json').read_text())}
    assert set(plans) == set(data.fire_name)
    reports, folds = {}, {}
    for arm, config in ARMS.items():
        report = json.loads((root / arm / 'report.json').read_text())
        assert report['config'] == config and report['seeds'] == seeds
        assert report['data_sha256'] == manifest['source_sha256'][manifest['data']]
        assert report['max_epochs'] == manifest['max_epochs']
        assert len(report['features']) == 60
        assert not set(report['features']) & {'no2', 'ref_no2', 'soil_nox_tau5hr', 'fire_name', 'date'}
        assert len(report['folds']) == len(plans)
        folds[arm] = {f['fire']: f for f in report['folds']}
        assert set(folds[arm]) == set(plans)
        reports[arm] = report
    decisions = {}
    for fire in sorted(plans):
        train, val, test = split_fires(data, fire)
        plan = plans[fire]
        stats = training_statistics(data.iloc[train])
        assert stats == plan['training_statistics']
        scorer = PeakScorer(data.iloc[val], stats['high_threshold'], stats['peak_threshold'], stats['scales'])
        scores = {}
        reference = folds['inner_arcsinh_capped'][fire]
        for arm in ARMS:
            fold = folds[arm][fire]
            assert fold['rows'] == [len(train), len(val), len(test)] == plan['rows']
            assert fold['training_statistics'] == stats
            assert fold['validation_fires'] == sorted(data.iloc[val].fire_name.unique()) == plan['validation_fires']
            assert fold['train_fires'] == sorted(data.iloc[train].fire_name.unique()) == plan['train_fires']
            assert not set(fold['train_fires']) & set(fold['validation_fires'])
            assert fire not in fold['train_fires'] + fold['validation_fires']
            assert fold['preprocessing'] == reference['preprocessing']
            assert set(fold['seeds']) == {str(s) for s in seeds}
            frame = load_predictions(root / arm / f'{fire}_validation.csv', data.iloc[val], seeds)
            score = {'peak_score': scorer.score(frame.prediction.to_numpy()), **scorer.components(frame.prediction.to_numpy())}
            for key, value in score.items():
                np.testing.assert_allclose(value, fold['validation_ensemble'][key])
            scores[arm] = score
            for seed in seeds:
                history = json.loads((root / arm / f'{fire}_{seed}_history.json').read_text())
                selected_mode = fold['seeds'][str(seed)]['selected_mode']
                minima = {mode: min(h['selection_score'] for h in rows) for mode, rows in history.items()}
                assert minima[selected_mode] == min(minima.values())
                np.testing.assert_allclose(minima[selected_mode], fold['seeds'][str(seed)]['selection_score'])
                seed_prediction = frame[f'seed_{seed}'].to_numpy()
                seed_score = scorer.score(seed_prediction) if ARMS[arm]['selection'] == 'peak_score' else float(
                    np.mean((np.arcsinh(seed_prediction) - scorer.asinh) ** 2))
                np.testing.assert_allclose(seed_score, minima[selected_mode])
                weights = torch.load(root / arm / f'{fire}_{seed}.pt', map_location='cpu', weights_only=True)
                assert next(w for w in weights.values() if w.ndim == 2).shape[1] == 60
        selected = select_arm(scores)
        decisions[fire] = {'selected_arm': selected, 'validation_scores': scores,
                           'eligible': {arm: s['full'] <= scores['inner_arcsinh_capped']['full'] * 1.05 for arm, s in scores.items()}}
    # Persist all choices before opening any outer test prediction files.
    decision_path = root / 'validation_selection.json'
    if decision_path.exists():
        previous_decisions = json.loads(decision_path.read_text())
        assert {f: d['selected_arm'] for f, d in decisions.items()} == {
            f: d['selected_arm'] for f, d in previous_decisions.items()}
        for fire, decision in decisions.items():
            for arm, scores in decision['validation_scores'].items():
                for key, value in scores.items():
                    np.testing.assert_allclose(value, previous_decisions[fire]['validation_scores'][arm][key])
    else:
        decision_path.write_text(json.dumps(decisions, indent=2, allow_nan=False))
    if select_only:
        print('All validation-only choices frozen. No test prediction files were read.')
        return
    if (root / 'refit_extension.json').exists():
        if not (root / 'refit_plan.json').exists():
            raise ValueError('Freeze the refit plan and complete refits before outer evaluation.')
        for strategy in ['refit_control', 'refit_selected']:
            if any(not (root / strategy / f'{fire}_report.json').exists() for fire in plans):
                raise ValueError('Complete every refit before outer evaluation.')
    frames = {arm: [] for arm in ARMS}
    selected_frames, previous_frames = [], []
    previous_dir = Path(manifest['previous_history_nn'])
    previous_report = json.loads((previous_dir / 'metrics.json').read_text())
    assert previous_report['seeds'] == seeds
    for fire in sorted(plans):
        expected = data[data.fire_name == fire]
        replay = fire == sorted(plans)[0]
        if replay:
            train, val, test = split_fires(data, fire)
            arrays, preprocessing = prepare_inputs(data, reports['inner_arcsinh_capped']['features'], train, val, test)
        for arm in ARMS:
            frame = load_predictions(root / arm / f'{fire}_predictions.csv', expected, seeds)
            if replay:
                assert preprocessing == folds[arm][fire]['preprocessing']
                validation_frame = load_predictions(root / arm / f'{fire}_validation.csv', data.iloc[val], seeds)
                for seed in seeds:
                    mode = folds[arm][fire]['seeds'][str(seed)]['selected_mode']
                    model = BaselineNN(60)
                    model.load_state_dict(torch.load(root / arm / f'{fire}_{seed}.pt', map_location='cpu', weights_only=True))
                    for partition, saved in [('test', frame), ('validation', validation_frame)]:
                        _, values = predict(model, arrays[partition][mode])
                        np.testing.assert_allclose(values, saved[f'seed_{seed}'], rtol=1e-6, atol=1e-6)
            frames[arm].append(frame)
            if arm == decisions[fire]['selected_arm']:
                selected_frames.append(frame.copy())
        previous_frames.append(load_predictions(previous_dir / f'{fire}_predictions.csv', expected, seeds,
                                                 prefix='baseline_', ensemble='baseline_ensemble'))
    combined = {'previous_history_nn': pd.concat(previous_frames, ignore_index=True),
                **{arm: pd.concat(parts, ignore_index=True) for arm, parts in frames.items()},
                'validation_selected': pd.concat(selected_frames, ignore_index=True)}
    if (root / 'refit_plan.json').exists():
        refit_plan = json.loads((root / 'refit_plan.json').read_text())
        for name, expected_hash in refit_plan['code_sha256'].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected_hash
        assert hashlib.sha256(decision_path.read_bytes()).hexdigest() == refit_plan['decision_sha256']
        for strategy in ['refit_control', 'refit_selected']:
            refit_frames = []
            for fire in sorted(plans):
                report = json.loads((root / strategy / f'{fire}_report.json').read_text())
                source_arm = 'inner_arcsinh_capped' if strategy == 'refit_control' else decisions[fire]['selected_arm']
                assert report['source_arm'] == source_arm
                outer_train = np.flatnonzero(data.fire_name != fire)
                outer_test = np.flatnonzero(data.fire_name == fire)
                assert report['train_fires'] == sorted(data.iloc[outer_train].fire_name.unique())
                assert report['rows'] == [len(outer_train), len(outer_test)]
                assert fire not in report['train_fires'] and len(report['train_fires']) == 15
                assert report['training_statistics'] == training_statistics(data.iloc[outer_train])
                frame = load_predictions(root / strategy / f'{fire}_predictions.csv', data.iloc[outer_test], seeds)
                for seed in seeds:
                    settings = report['seeds'][str(seed)]
                    inner = folds[source_arm][fire]['seeds'][str(seed)]
                    assert settings['selected_mode'] == inner['selected_mode']
                    assert settings['epochs'] == inner['best_epoch']
                    history = json.loads((root / source_arm / f'{fire}_{seed}_history.json').read_text())[inner['selected_mode']]
                    assert settings['learning_rates'] == learning_rates(history, inner['best_epoch'])
                    checkpoint = torch.load(root / strategy / f'{fire}_{seed}.pt', map_location='cpu', weights_only=True)
                    assert next(w for w in checkpoint.values() if w.ndim == 2).shape[1] == 60
                    if fire == sorted(plans)[0]:
                        refit_arrays, refit_preprocessing = prepare_inputs(data, reports[source_arm]['features'], outer_train, outer_train, outer_test)
                        assert refit_preprocessing == report['preprocessing']
                        model = BaselineNN(60)
                        model.load_state_dict(checkpoint)
                        _, values = predict(model, refit_arrays['test'][settings['selected_mode']])
                        np.testing.assert_allclose(values, frame[f'seed_{seed}'], rtol=1e-6, atol=1e-6)
                refit_frames.append(frame)
            combined[strategy] = pd.concat(refit_frames, ignore_index=True)
    thresholds = json.loads(Path(manifest['shared_thresholds']).read_text())['thresholds']
    summary = {'rows': len(data), 'fires': len(plans), 'seeds': seeds, 'arms': {},
               'checkpoint_replay': {'fire': sorted(plans)[0], 'arms': 4, 'seeds_per_arm': len(seeds), 'partitions': ['validation', 'test']},
               'selection_counts': pd.Series([d['selected_arm'] for d in decisions.values()]).value_counts().to_dict(),
               'comparison_note': 'Four candidate arms share whole-fire inner validation and training rows. The previous history NN used late-date validation and different training rows. Refit pipelines train on all 15 outer-training fires after validation freezes settings. Outer evaluation rows and reporting thresholds are identical.'}
    per_fire, per_seed, daily_frames = [], [], []
    for arm, frame in combined.items():
        frame = frame.sort_values(KEYS).reset_index(drop=True)
        pd.testing.assert_frame_equal(frame[KEYS], data[KEYS], check_dtype=False)
        frame['days_post_fire'] = data.days_post_fire
        frame['high'] = [v > thresholds[f]['high_flux_row'] for f, v in zip(frame.fire_name, frame.actual)]
        frame['extreme'] = [v > thresholds[f]['extreme_flux_row'] for f, v in zip(frame.fire_name, frame.actual)]
        if arm in ['validation_selected', 'refit_selected']:
            frame['selected_arm'] = frame.fire_name.map({f: d['selected_arm'] for f, d in decisions.items()})
            name = 'selected_predictions.csv' if arm == 'validation_selected' else 'refit_selected_predictions.csv'
            frame.to_csv(root / name, index=False)
        daily = frame.groupby(['fire_name', 'date'], as_index=False)[['actual', 'prediction']].mean()
        daily['peak'] = [v > thresholds[f]['peak_day_mean'] for f, v in zip(daily.fire_name, daily.actual)]
        daily_frames.append(daily.assign(arm=arm))
        for fire, group in frame.groupby('fire_name'):
            days = daily[daily.fire_name == fire]
            row = {'arm': arm, 'fire': fire, **metrics(group.actual.to_numpy(), group.prediction.to_numpy())}
            for name, source, mask in [('first_month', group, group.days_post_fire.between(0, 30)),
                                       ('high_flux', group, group.high), ('extreme_flux', group, group.extreme),
                                       ('peak_days', days, days.peak)]:
                scores = subset_scores(source, 'prediction', mask)
                row[f'{name}_rows'] = int(mask.sum())
                row[f'{name}_mae'] = scores['mae'] if scores else None
                row[f'{name}_bias'] = scores['percent_bias'] if scores else None
            per_fire.append(row)
            for seed in seeds:
                per_seed.append({'arm': arm, 'fire': fire, 'seed': seed,
                                 **metrics(group.actual.to_numpy(), group[f'seed_{seed}'].to_numpy())})
        table = pd.DataFrame([r for r in per_fire if r['arm'] == arm])
        scores = {'equal_fire_mean_mae': float(table.mae.mean()),
                  'pooled': metrics(frame.actual.to_numpy(), frame.prediction.to_numpy())}
        for name, source, mask in [('first_month', frame, frame.days_post_fire.between(0, 30)),
                                   ('high_flux', frame, frame.high), ('extreme_flux', frame, frame.extreme),
                                   ('negative_label', frame, frame.actual < 0), ('peak_days', daily, daily.peak)]:
            scores[name] = subset_scores(source, 'prediction', mask)
            scores[f'{name}_rows'] = int(mask.sum())
        summary['arms'][arm] = scores
    per_fire = pd.DataFrame(per_fire)
    previous = per_fire[per_fire.arm == 'previous_history_nn'].set_index('fire').mae
    for arm, scores in summary['arms'].items():
        group = per_fire[per_fire.arm == arm].set_index('fire')
        scores['fires_better_than_previous'] = int((group.mae < previous).sum())
        scores['macro_mae_change_percent'] = 100 * (scores['equal_fire_mean_mae'] / previous.mean() - 1)
    summary['epoch_ceiling'] = {arm: {
        'candidates': sum(n == reports[arm]['max_epochs'] for f in reports[arm]['folds']
                          for seed in f['seeds'].values() for n in seed['candidate_epochs'].values()),
        'total_candidates': len(plans) * len(seeds) * 2}
        for arm in ARMS}
    per_fire.to_csv(root / 'per_fire_comparison.csv', index=False)
    pd.DataFrame(per_seed).to_csv(root / 'per_seed_comparison.csv', index=False)
    pd.concat(daily_frames, ignore_index=True).to_csv(root / 'daily_predictions.csv', index=False)
    (root / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    plot(root, summary)
    print(json.dumps(summary, indent=2))


def plot(root, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['Previous\nhistory NN', 'Inner-fire\nMSE control', 'Peak score\ncapped', 'Peak score\nuncapped', 'Peak score\nweighted', 'Validation\nselected', 'Refit\ncontrol', 'Refit\nselected'][:len(summary['arms'])]
    colors = ['#0072B2', '#999999', '#56B4E9', '#E69F00', '#CC79A7', '#009E73', '#666666', '#007A5E'][:len(labels)]
    arms = list(summary['arms'].values())
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout='constrained')
    panels = [('Full period · equal-fire MAE', [r['equal_fire_mean_mae'] for r in arms]),
              ('First month · pooled MAE', [r['first_month']['mae'] for r in arms]),
              ('Extreme rows · pooled MAE', [r['extreme_flux']['mae'] for r in arms]),
              ('Peak daily means · MAE', [r['peak_days']['mae'] for r in arms])]
    for ax, (title, values) in zip(axes.flat, panels):
        ax.bar(range(len(labels)), values, color=colors)
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel('ng N m⁻² s⁻¹')
        ax.set_ylim(0, max(values) * 1.15)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        for i, value in enumerate(values):
            ax.text(i, value, f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    fig.suptitle('Peak-focused validation study · 16 outer fires · three-seed ensembles')
    fig.savefig(root / 'comparison.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--select-only', action='store_true')
    torch.set_num_threads(2)
    args = parser.parse_args()
    compare(args.root, args.select_only)
