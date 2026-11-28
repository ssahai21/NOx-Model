"""Verify a paired history-transform experiment before inspecting test performance."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.baseline_nn import BaselineNN
from src.compare_peak_validation import load_predictions
from src.compare_target_caps import subset_scores
from src.final_pinn import metrics
from src.history_scaling_study import SIGNALS, apply_preprocessing, choose_scaling, prepare_asinh_inputs
from src.peak_validation_study import ARMS, KEYS, PeakScorer, predict, prepare_inputs, split_fires, training_statistics
from src.refit_peak_selection import learning_rates


def assert_scores(actual, saved):
    for key, value in actual.items():
        np.testing.assert_allclose(value, saved[key])


def compare(root):
    manifest = json.loads((root / 'study_manifest.json').read_text())
    for name, expected in {**manifest['source_sha256'], **manifest['training_code_sha256']}.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    source = Path(manifest['control_study'])
    old_choices = json.loads((source / 'validation_selection.json').read_text())
    old_reports = {a: json.loads((source / a / 'report.json').read_text()) for a in ARMS}
    old_folds = {a: {f['fire']: f for f in r['folds']} for a, r in old_reports.items()}
    data = pd.read_csv(manifest['data'], parse_dates=['date'])
    data = data[data.days_post_fire >= 0].sort_values(KEYS).reset_index(drop=True)
    assert set(old_choices) == set(data.fire_name)
    seeds = manifest['seeds']
    features = old_reports['inner_arcsinh_capped']['features']
    decisions, inner, refits = {}, {}, {}
    for fire in sorted(old_choices):
        train, val, test = split_fires(data, fire)
        full_train = np.flatnonzero(data.fire_name != fire)
        arm = old_choices[fire]['selected_arm']
        fold = json.loads((root / 'inner_asinh' / f'{fire}_report.json').read_text())
        refit = json.loads((root / 'refit_asinh' / f'{fire}_report.json').read_text())
        raw = old_folds[arm][fire]
        stats = training_statistics(data.iloc[train])
        assert fold['source_arm'] == refit['source_arm'] == arm
        assert fold['config'] == ARMS[arm]
        assert fold['rows'] == raw['rows'] == [len(train), len(val), len(test)]
        assert fold['train_fires'] == raw['train_fires']
        assert fold['validation_fires'] == raw['validation_fires']
        assert fire not in fold['train_fires'] + fold['validation_fires']
        assert not set(fold['train_fires']) & set(fold['validation_fires'])
        assert fold['training_statistics'] == raw['training_statistics'] == stats
        arrays, state = prepare_asinh_inputs(data, features, train, val, test)
        assert state == fold['preprocessing']
        scorer = PeakScorer(data.iloc[val], stats['high_threshold'], stats['peak_threshold'], stats['scales'])
        frame = load_predictions(root / 'inner_asinh' / f'{fire}_validation.csv', data.iloc[val], seeds)
        scores = {'peak_score': scorer.score(frame.prediction.to_numpy()), **scorer.components(frame.prediction.to_numpy())}
        assert_scores(scores, fold['validation_ensemble'])
        decision_path = root / 'decisions' / f'{fire}.json'
        decision = json.loads(decision_path.read_text())
        anchor = old_folds['inner_arcsinh_capped'][fire]['validation_ensemble']
        assert decision['source_arm'] == arm
        assert_scores(anchor, decision['bulk_anchor'])
        assert_scores(raw['validation_ensemble'], decision['raw_validation'])
        assert_scores(scores, decision['asinh_validation'])
        assert decision['selected_scaling'] == choose_scaling(raw['validation_ensemble'], scores, anchor)
        assert refit['decision_sha256'] == hashlib.sha256(decision_path.read_bytes()).hexdigest()
        assert refit['rows'] == [len(full_train), len(test)]
        assert refit['train_fires'] == sorted(data.iloc[full_train].fire_name.unique())
        assert len(refit['train_fires']) == 15 and fire not in refit['train_fires']
        assert refit['training_statistics'] == training_statistics(data.iloc[full_train])
        full_arrays, full_state = prepare_asinh_inputs(data, features, full_train, full_train, test)
        assert full_state == refit['preprocessing']
        assert set(fold['seeds']) == set(refit['seeds']) == {str(s) for s in seeds}
        for seed in seeds:
            settings = fold['seeds'][str(seed)]
            mode = settings['selected_mode']
            history = json.loads((root / 'inner_asinh' / f'{fire}_{seed}_history.json').read_text())
            minima = {m: min(h['selection_score'] for h in rows) for m, rows in history.items()}
            assert minima[mode] == min(minima.values())
            np.testing.assert_allclose(minima[mode], settings['selection_score'])
            assert settings['best_epoch'] == min(history[mode], key=lambda h: h['selection_score'])['epoch']
            seed_values = frame[f'seed_{seed}'].to_numpy()
            seed_score = scorer.score(seed_values) if ARMS[arm]['selection'] == 'peak_score' else float(np.mean((np.arcsinh(seed_values) - scorer.asinh) ** 2))
            np.testing.assert_allclose(seed_score, minima[mode])
            assert refit['seeds'][str(seed)]['selected_mode'] == mode
            assert refit['seeds'][str(seed)]['epochs'] == settings['best_epoch']
            assert refit['seeds'][str(seed)]['learning_rates'] == learning_rates(history[mode], settings['best_epoch'])
            for directory in ['inner_asinh', 'refit_asinh']:
                checkpoint = torch.load(root / directory / f'{fire}_{seed}.pt', map_location='cpu', weights_only=True)
                assert next(w for w in checkpoint.values() if w.ndim == 2).shape[1] == 60
            if fire == sorted(old_choices)[0]:
                model = BaselineNN(60)
                model.load_state_dict(torch.load(root / 'inner_asinh' / f'{fire}_{seed}.pt', map_location='cpu', weights_only=True))
                np.testing.assert_allclose(predict(model, arrays['validation'][mode])[1], seed_values, rtol=1e-6, atol=1e-6)
        decisions[fire], inner[fire], refits[fire] = decision, fold, refit
    # This aggregate is frozen before any outer prediction CSV is opened here.
    (root / 'scaling_selection.json').write_text(json.dumps(decisions, indent=2, allow_nan=False))
    parts = {arm: [] for arm in ['raw_inner', 'asinh_inner', 'raw_refit', 'asinh_refit', 'validation_selected_scaling']}
    clipping = []
    for fire in sorted(old_choices):
        train, val, test = split_fires(data, fire)
        full_train = np.flatnonzero(data.fire_name != fire)
        expected = data.iloc[test]
        arm = old_choices[fire]['selected_arm']
        locations = {'raw_inner': source / arm, 'asinh_inner': root / 'inner_asinh',
                     'raw_refit': source / 'refit_selected', 'asinh_refit': root / 'refit_asinh'}
        loaded = {name: load_predictions(directory / f'{fire}_predictions.csv', expected, seeds)
                  for name, directory in locations.items()}
        for name, frame in loaded.items():
            parts[name].append(frame)
        selected = 'asinh_refit' if decisions[fire]['selected_scaling'] == 'asinh' else 'raw_refit'
        parts['validation_selected_scaling'].append(loaded[selected].copy())
        raw_refit = json.loads((source / 'refit_selected' / f'{fire}_report.json').read_text())
        assert raw_refit['rows'] == refits[fire]['rows']
        assert raw_refit['train_fires'] == refits[fire]['train_fires']
        assert raw_refit['training_statistics'] == refits[fire]['training_statistics']
        for name, state in [('raw', raw_refit['preprocessing']), ('asinh', refits[fire]['preprocessing'])]:
            if name == 'asinh':
                z = apply_preprocessing(expected, state)[1]
            else:
                filled = expected[features].replace([np.inf, -np.inf], np.nan).fillna(dict(zip(features, state['medians'])))
                z = (filled.to_numpy() - np.asarray(state['mean'])) / np.asarray(state['scale'])
            positions = [features.index(f) for f in SIGNALS]
            signal_z = z[:, positions]
            clip = expected[KEYS + ['soil_nox_tau5hr', 'days_post_fire']].copy()
            clip['scaling'] = name
            clip['any_signal_clipped'] = (np.abs(signal_z) > 5).any(axis=1)
            clip['all_signals_upper_clipped'] = (signal_z > 5).all(axis=1)
            clipping.append(clip)
        if fire == sorted(old_choices)[0]:
            inner_arrays = prepare_asinh_inputs(data, features, train, val, test)[0]
            refit_arrays = prepare_asinh_inputs(data, features, full_train, full_train, test)[0]
            raw_arrays, raw_state = prepare_inputs(data, features, full_train, full_train, test)
            assert raw_state == raw_refit['preprocessing']
            for seed in seeds:
                for name, directory, arrays, settings in [
                    ('asinh_inner', root / 'inner_asinh', inner_arrays, inner[fire]['seeds'][str(seed)]),
                    ('asinh_refit', root / 'refit_asinh', refit_arrays, refits[fire]['seeds'][str(seed)]),
                    ('raw_refit', source / 'refit_selected', raw_arrays, raw_refit['seeds'][str(seed)])]:
                    model = BaselineNN(60)
                    model.load_state_dict(torch.load(directory / f'{fire}_{seed}.pt', map_location='cpu', weights_only=True))
                    prediction = predict(model, arrays['test'][settings['selected_mode']])[1]
                    np.testing.assert_allclose(prediction, loaded[name][f'seed_{seed}'], rtol=1e-6, atol=1e-6)
    thresholds = json.loads(Path(manifest['shared_thresholds']).read_text())['thresholds']
    summary = {'rows': len(data), 'fires': len(old_choices), 'seeds': seeds, 'arms': {},
               'selection_counts': pd.Series([d['selected_scaling'] for d in decisions.values()]).value_counts().to_dict(),
               'clipping': {}, 'caveat': manifest['caveat']}
    per_fire, per_seed, daily_frames = [], [], []
    for name, frames in parts.items():
        frame = pd.concat(frames, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
        pd.testing.assert_frame_equal(frame[KEYS], data[KEYS], check_dtype=False)
        frame['days_post_fire'] = data.days_post_fire
        frame['high'] = [v > thresholds[f]['high_flux_row'] for f, v in zip(frame.fire_name, frame.actual)]
        frame['extreme'] = [v > thresholds[f]['extreme_flux_row'] for f, v in zip(frame.fire_name, frame.actual)]
        daily = frame.groupby(['fire_name', 'date'], as_index=False)[['actual', 'prediction']].mean()
        daily['peak'] = [v > thresholds[f]['peak_day_mean'] for f, v in zip(daily.fire_name, daily.actual)]
        daily_frames.append(daily.assign(arm=name))
        for fire, group in frame.groupby('fire_name'):
            days = daily[daily.fire_name == fire]
            row = {'arm': name, 'fire': fire, **metrics(group.actual.to_numpy(), group.prediction.to_numpy())}
            for cohort, source_frame, mask in [('first_month', group, group.days_post_fire.between(0, 30)),
                                               ('extreme_flux', group, group.extreme), ('peak_days', days, days.peak)]:
                score = subset_scores(source_frame, 'prediction', mask)
                row[f'{cohort}_mae'] = score['mae'] if score else None
                row[f'{cohort}_bias'] = score['percent_bias'] if score else None
            per_fire.append(row)
            for seed in seeds:
                per_seed.append({'arm': name, 'fire': fire, 'seed': seed,
                                 **metrics(group.actual.to_numpy(), group[f'seed_{seed}'].to_numpy())})
        table = pd.DataFrame([r for r in per_fire if r['arm'] == name])
        result = {'equal_fire_mean_mae': float(table.mae.mean()), 'pooled': metrics(frame.actual.to_numpy(), frame.prediction.to_numpy())}
        for cohort, source_frame, mask in [('first_month', frame, frame.days_post_fire.between(0, 30)),
                                           ('high_flux', frame, frame.high), ('extreme_flux', frame, frame.extreme),
                                           ('negative_label', frame, frame.actual < 0), ('peak_days', daily, daily.peak)]:
            result[cohort] = subset_scores(source_frame, 'prediction', mask)
            result[f'{cohort}_rows'] = int(mask.sum())
        summary['arms'][name] = result
        if name == 'validation_selected_scaling':
            frame['selected_scaling'] = frame.fire_name.map({f: d['selected_scaling'] for f, d in decisions.items()})
            frame.to_csv(root / 'selected_predictions.csv', index=False)
    per_fire = pd.DataFrame(per_fire)
    raw = per_fire[per_fire.arm == 'raw_refit'].set_index('fire').mae
    for name, result in summary['arms'].items():
        group = per_fire[per_fire.arm == name].set_index('fire')
        result['fires_better_than_raw_refit'] = int((group.mae < raw).sum())
        result['macro_mae_change_percent'] = 100 * (group.mae.mean() / raw.mean() - 1)
    clip_data = pd.concat(clipping, ignore_index=True)
    clip_data['extreme'] = [v > thresholds[f]['extreme_flux_row'] for f, v in zip(clip_data.fire_name, clip_data.soil_nox_tau5hr)]
    for scaling, group in clip_data.groupby('scaling'):
        summary['clipping'][scaling] = {}
        for cohort, mask in [('all', np.ones(len(group), dtype=bool)), ('first_month', group.days_post_fire.between(0, 30)), ('extreme', group.extreme)]:
            selected = group.loc[mask]
            summary['clipping'][scaling][cohort] = {'rows': len(selected), 'any_signal_clipped_rows': int(selected.any_signal_clipped.sum()),
                                                  'any_signal_clipped_fraction': float(selected.any_signal_clipped.mean()),
                                                  'all_signals_upper_clipped_rows': int(selected.all_signals_upper_clipped.sum())}
    summary['epoch_ceiling_candidates'] = sum(n == manifest['max_epochs'] for fold in inner.values()
                                             for seed in fold['seeds'].values() for n in seed['candidate_epochs'].values())
    per_fire.to_csv(root / 'per_fire_comparison.csv', index=False)
    pd.DataFrame(per_seed).to_csv(root / 'per_seed_comparison.csv', index=False)
    pd.concat(daily_frames, ignore_index=True).to_csv(root / 'daily_predictions.csv', index=False)
    clip_data.to_csv(root / 'clipping_rows.csv', index=False)
    (root / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    plot(root, summary)
    print(json.dumps(summary, indent=2))


def plot(root, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['Raw\ninner', 'Signed-log\ninner', 'Raw\nrefit', 'Signed-log\nrefit', 'Validation\nselected']
    colors = ['#999999', '#56B4E9', '#0072B2', '#E69F00', '#009E73']
    arms = list(summary['arms'].values())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout='constrained')
    panels = [('Full period · equal-fire MAE', [r['equal_fire_mean_mae'] for r in arms]),
              ('Extreme rows · MAE', [r['extreme_flux']['mae'] for r in arms]),
              ('Peak daily means · MAE', [r['peak_days']['mae'] for r in arms])]
    for ax, (title, values) in zip(axes, panels):
        ax.bar(range(5), values, color=colors)
        ax.set_xticks(range(5), labels, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('ng N m⁻² s⁻¹')
        ax.set_ylim(0, max(values) * 1.15)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        for i, value in enumerate(values):
            ax.text(i, value, f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    fig.suptitle('Past-NO₂ signed-log scaling · 16 held-out fires · three-seed ensembles')
    fig.savefig(root / 'comparison.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    torch.set_num_threads(2)
    compare(parser.parse_args().root)
