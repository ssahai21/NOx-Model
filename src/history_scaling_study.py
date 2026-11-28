"""Train-only signed-log history scaling with matched, frozen loss choices."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.peak_validation_study import ARMS, KEYS, PeakScorer, fit_candidate, predict, sample_weights, split_fires, training_statistics
from src.refit_peak_selection import fit_fixed, learning_rates

SIGNALS = ['no2_previous_observed', 'delta_no2_previous_observed', 'no2_past_week_mean']


def transform_raw(filled, features, signal_scales):
    values = filled.to_numpy(dtype=float).copy()
    for feature, scale in signal_scales.items():
        position = features.index(feature)
        values[:, position] = np.arcsinh(values[:, position] / scale)
    return values


def apply_preprocessing(frame, state):
    features = state['features']
    filled = frame[features].replace([np.inf, -np.inf], np.nan).fillna(dict(zip(features, state['medians'])))
    transformed = transform_raw(filled, features, state['signal_scales'])
    z = (transformed - np.asarray(state['mean'])) / np.asarray(state['scale'])
    standard = np.clip(z, -state['input_z_limit'], state['input_z_limit']).astype(np.float32)
    native = standard.copy()
    land = [i for i, name in enumerate(features) if name.startswith('pct_')]
    native[:, land] = filled.iloc[:, land].to_numpy(dtype=np.float32)
    return {'standardized': standard, 'native': native}, z


def prepare_asinh_inputs(data, features, train, validation, test):
    inputs = data[features].replace([np.inf, -np.inf], np.nan)
    medians = inputs.iloc[train].median().fillna(0)
    filled = inputs.iloc[train].fillna(medians)
    signal_scales = {}
    for feature in SIGNALS:
        magnitudes = np.abs(filled[feature].to_numpy(dtype=float))
        positive = magnitudes[magnitudes > 0]
        signal_scales[feature] = float(np.median(positive)) if len(positive) else 1.
    transformed = transform_raw(filled, features, signal_scales)
    scaler = StandardScaler().fit(transformed)
    state = {'features': features, 'medians': medians.to_list(), 'signal_scales': signal_scales,
             'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(), 'input_z_limit': 5,
             'transform': 'asinh(raw / median_nonzero_absolute_training_value) on three history signals'}
    arrays = {name: apply_preprocessing(data.iloc[indices], state)[0]
              for name, indices in [('train', train), ('validation', validation), ('test', test)]}
    return arrays, state


def choose_scaling(raw, transformed, bulk_anchor):
    return 'asinh' if transformed['full'] <= bulk_anchor['full'] * 1.05 and transformed['peak_score'] < raw['peak_score'] else 'raw'


def run(root, shard, shards, only_fires=None, epochs=None, seeds=None):
    manifest = json.loads((root / 'study_manifest.json').read_text())
    for name, expected in {**manifest['source_sha256'], **manifest['training_code_sha256']}.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    source = Path(manifest['control_study'])
    decisions = json.loads((source / 'validation_selection.json').read_text())
    data = pd.read_csv(manifest['data'], parse_dates=['date'])
    data = data[data.days_post_fire >= 0].sort_values(KEYS).reset_index(drop=True)
    seeds = manifest['seeds'] if seeds is None else seeds
    epochs = manifest['max_epochs'] if epochs is None else epochs
    reports = {arm: json.loads((source / arm / 'report.json').read_text()) for arm in ARMS}
    folds = {arm: {f['fire']: f for f in report['folds']} for arm, report in reports.items()}
    features = reports['inner_arcsinh_capped']['features']
    assert len(features) == 60
    for index, fire in enumerate(sorted(decisions)):
        if index % shards != shard or (only_fires is not None and fire not in only_fires):
            continue
        arm = decisions[fire]['selected_arm']
        config = ARMS[arm]
        train, validation, test = split_fires(data, fire)
        raw_fold = folds[arm][fire]
        stats = training_statistics(data.iloc[train])
        assert raw_fold['training_statistics'] == stats
        scorer = PeakScorer(data.iloc[validation], stats['high_threshold'], stats['peak_threshold'], stats['scales'])
        arrays, state = prepare_asinh_inputs(data, features, train, validation, test)
        target = data.iloc[train].soil_nox_tau5hr.to_numpy()
        if config['capped']:
            target = np.clip(target, stats['p1'], stats['p99'])
        weights = sample_weights(data.iloc[train], stats['high_threshold'], config['tail_weight'])
        output = root / 'inner_asinh'
        output.mkdir(parents=True, exist_ok=True)
        if (output / f'{fire}_report.json').exists():
            raise ValueError(f'Completed scaling fold already exists: {fire}')
        validation_predictions = data.iloc[validation][KEYS + ['soil_nox_tau5hr']].rename(columns={'soil_nox_tau5hr': 'actual'}).copy()
        predictions = data.iloc[test][KEYS + ['soil_nox_tau5hr']].rename(columns={'soil_nox_tau5hr': 'actual'}).copy()
        fold = {'fire': fire, 'source_arm': arm, 'config': config, 'rows': [len(train), len(validation), len(test)],
                'train_fires': sorted(data.iloc[train].fire_name.unique()),
                'validation_fires': sorted(data.iloc[validation].fire_name.unique()),
                'training_statistics': stats, 'preprocessing': state, 'seeds': {}}
        for seed in seeds:
            best, histories = float('inf'), {}
            for mode in ['standardized', 'native']:
                print(f'{fire} | asinh inner | {arm} | seed={seed} | {mode}', flush=True)
                model, history, val_prediction = fit_candidate(arrays['train'][mode], target, weights,
                    arrays['validation'][mode], scorer, config['selection'], seed, epochs)
                histories[mode] = history
                score = min(h['selection_score'] for h in history)
                if score < best:
                    best, selected, best_model, best_val = score, mode, model, val_prediction
            validation_predictions[f'seed_{seed}'] = best_val
            predictions[f'seed_{seed}'] = predict(best_model, arrays['test'][selected])[1]
            fold['seeds'][str(seed)] = {'selected_mode': selected, 'selection_score': best,
                'best_epoch': min(histories[selected], key=lambda h: h['selection_score'])['epoch'],
                'candidate_epochs': {mode: len(h) for mode, h in histories.items()}}
            torch.save(best_model.state_dict(), output / f'{fire}_{seed}.pt')
            (output / f'{fire}_{seed}_history.json').write_text(json.dumps(histories, indent=2, allow_nan=False))
        columns = [f'seed_{seed}' for seed in seeds]
        validation_predictions['prediction'] = validation_predictions[columns].mean(axis=1)
        predictions['prediction'] = predictions[columns].mean(axis=1)
        val_scores = {'peak_score': scorer.score(validation_predictions.prediction.to_numpy()),
                      **scorer.components(validation_predictions.prediction.to_numpy())}
        fold['validation_ensemble'] = val_scores
        validation_predictions.to_csv(output / f'{fire}_validation.csv', index=False)
        predictions.to_csv(output / f'{fire}_predictions.csv', index=False)
        (output / f'{fire}_report.json').write_text(json.dumps(fold, indent=2, allow_nan=False))
        anchor = folds['inner_arcsinh_capped'][fire]['validation_ensemble']
        decision = {'fire': fire, 'source_arm': arm,
                    'selected_scaling': choose_scaling(raw_fold['validation_ensemble'], val_scores, anchor),
                    'bulk_anchor': anchor,
                    'raw_validation': raw_fold['validation_ensemble'], 'asinh_validation': val_scores}
        decision_path = root / 'decisions' / f'{fire}.json'
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        decision_path.write_text(json.dumps(decision, indent=2, allow_nan=False))
        # The choice above is frozen before refitting and before any test metrics.
        full_train = np.flatnonzero(data.fire_name != fire)
        full_arrays, full_state = prepare_asinh_inputs(data, features, full_train, full_train, test)
        full_stats = training_statistics(data.iloc[full_train])
        full_target = data.iloc[full_train].soil_nox_tau5hr.to_numpy()
        if config['capped']:
            full_target = np.clip(full_target, full_stats['p1'], full_stats['p99'])
        full_weights = sample_weights(data.iloc[full_train], full_stats['high_threshold'], config['tail_weight'])
        refit = root / 'refit_asinh'
        refit.mkdir(parents=True, exist_ok=True)
        refit_predictions = data.iloc[test][KEYS + ['soil_nox_tau5hr']].rename(columns={'soil_nox_tau5hr': 'actual'}).copy()
        refit_report = {'fire': fire, 'source_arm': arm, 'rows': [len(full_train), len(test)],
                        'train_fires': sorted(data.iloc[full_train].fire_name.unique()),
                        'training_statistics': full_stats, 'preprocessing': full_state,
                        'decision_sha256': hashlib.sha256(decision_path.read_bytes()).hexdigest(), 'seeds': {}}
        for seed in seeds:
            settings = fold['seeds'][str(seed)]
            mode, budget = settings['selected_mode'], settings['best_epoch']
            history = json.loads((output / f'{fire}_{seed}_history.json').read_text())[mode]
            rates = learning_rates(history, budget)
            print(f'{fire} | asinh refit | seed={seed} | epochs={budget}', flush=True)
            model = fit_fixed(full_arrays['train'][mode], full_target, full_weights, rates, seed)
            refit_predictions[f'seed_{seed}'] = predict(model, full_arrays['test'][mode])[1]
            torch.save(model.state_dict(), refit / f'{fire}_{seed}.pt')
            refit_report['seeds'][str(seed)] = {'selected_mode': mode, 'epochs': budget, 'learning_rates': rates}
        refit_predictions['prediction'] = refit_predictions[columns].mean(axis=1)
        refit_predictions.to_csv(refit / f'{fire}_predictions.csv', index=False)
        (refit / f'{fire}_report.json').write_text(json.dumps(refit_report, indent=2, allow_nan=False))
        print(f'{fire} scaling fold completed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--fires', nargs='+')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--seeds', nargs='+', type=int)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error('Invalid shard')
    torch.set_num_threads(2)
    run(args.root, args.shard, args.shards, args.fires, args.epochs, args.seeds)
