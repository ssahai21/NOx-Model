"""Nested fire validation and predeclared tail-loss comparisons for the history NN."""
import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.baseline_nn import BaselineNN
from src.data_loader import get_features

ARMS = {
    'inner_arcsinh_capped': {'selection': 'arcsinh_mse', 'capped': True, 'tail_weight': 1.},
    'inner_peak_capped': {'selection': 'peak_score', 'capped': True, 'tail_weight': 1.},
    'inner_peak_uncapped': {'selection': 'peak_score', 'capped': False, 'tail_weight': 1.},
    'inner_peak_weighted': {'selection': 'peak_score', 'capped': False, 'tail_weight': 5.},
}
KEYS = ['fire_name', 'plot_id', 'date']
COMPONENT_WEIGHTS = {'full': .4, 'first_month': .2, 'high_rows': .2, 'peak_days': .2}


def select_arm(validation_scores):
    control = validation_scores['inner_arcsinh_capped']
    eligible = [arm for arm in ARMS if validation_scores[arm]['full'] <= control['full'] * 1.05]
    return min(eligible, key=lambda arm: validation_scores[arm]['peak_score'])


def split_fires(data, outer_fire):
    """Choose inner fires using only identifiers and early-date availability, never flux."""
    if outer_fire not in set(data.fire_name):
        raise ValueError('Unknown outer fire')
    outer = data[data.fire_name != outer_fire]
    eligible = outer[outer.days_post_fire.between(0, 30)].groupby('fire_name').date.nunique()
    eligible = eligible[eligible >= 3].index.tolist()
    if len(eligible) < 4:
        raise ValueError('At least four inner fires with early observations are required')
    # Avoid tiny or dominant validation groups using row counts only. A broad
    # budget permits large fires to appear in validation rather than always train.
    counts = outer.groupby('fire_name').size()
    candidates = list(combinations(sorted(eligible), 3))
    errors = {combo: abs(sum(counts[fire] for fire in combo) - .2 * len(outer)) for combo in candidates}
    balanced = [combo for combo in candidates
                if .15 <= sum(counts[fire] for fire in combo) / len(outer) <= .35]
    if not balanced:
        balanced = [combo for combo in candidates if errors[combo] == min(errors.values())]
    validation_fires = list(min(balanced, key=lambda combo: hashlib.sha256(
        f'peak-study-v1|{outer_fire}|{"|".join(combo)}'.encode()).hexdigest()))
    train = np.flatnonzero((data.fire_name != outer_fire) & ~data.fire_name.isin(validation_fires))
    validation = np.flatnonzero(data.fire_name.isin(validation_fires))
    test = np.flatnonzero(data.fire_name == outer_fire)
    return train, validation, test


class PeakScorer:
    """Equal-fire errors with fixed training-only scales and tail thresholds."""
    def __init__(self, frame, high_threshold, peak_threshold, scales=None):
        self.actual = frame.soil_nox_tau5hr.to_numpy(dtype=float)
        self.asinh = np.arcsinh(self.actual)
        fires = frame.fire_name.to_numpy()
        self.row_weights = {}
        masks = {'full': np.ones(len(frame), dtype=bool),
                 'first_month': frame.days_post_fire.between(0, 30).to_numpy(),
                 'high_rows': self.actual > high_threshold}
        for name, mask in masks.items():
            weights = np.zeros(len(frame), dtype=float)
            included = np.unique(fires[mask])
            if not len(included):
                raise ValueError(f'Validation/training cohort is empty: {name}')
            for fire in included:
                selected = mask & (fires == fire)
                weights[selected] = 1 / (len(included) * selected.sum())
            self.row_weights[name] = weights
        daily_keys = pd.MultiIndex.from_frame(frame[['fire_name', 'date']])
        self.daily_codes, unique = pd.factorize(daily_keys, sort=True)
        self.daily_counts = np.bincount(self.daily_codes)
        self.daily_actual = np.bincount(self.daily_codes, weights=self.actual) / self.daily_counts
        daily_fires = unique.get_level_values(0).to_numpy()
        selected = self.daily_actual > peak_threshold
        included = np.unique(daily_fires[selected])
        if not len(included):
            raise ValueError('Validation/training cohort is empty: peak_days')
        self.daily_weights = np.zeros(len(unique), dtype=float)
        for fire in included:
            mask = selected & (daily_fires == fire)
            self.daily_weights[mask] = 1 / (len(included) * mask.sum())
        self.scales = scales
        self.coverage = {name: {'rows': int((weights > 0).sum()),
                                'fires': int(len(np.unique(fires[weights > 0])))}
                         for name, weights in self.row_weights.items()}
        self.coverage['peak_days'] = {'days': int(selected.sum()), 'fires': int(len(included))}

    def components(self, prediction):
        prediction = np.asarray(prediction, dtype=float)
        if prediction.shape != self.actual.shape or not np.isfinite(prediction).all():
            raise ValueError('Invalid predictions')
        error = np.abs(prediction - self.actual)
        scores = {name: float(weights @ error) for name, weights in self.row_weights.items()}
        daily_prediction = np.bincount(self.daily_codes, weights=prediction) / self.daily_counts
        scores['peak_days'] = float(self.daily_weights @ np.abs(daily_prediction - self.daily_actual))
        return scores

    def score(self, prediction):
        if self.scales is None:
            raise ValueError('Training-derived normalizers are required')
        parts = self.components(prediction)
        return sum(COMPONENT_WEIGHTS[name] * value / self.scales[name] for name, value in parts.items())


def training_statistics(frame):
    target = frame.soil_nox_tau5hr.to_numpy(dtype=float)
    daily = frame.groupby(['fire_name', 'date']).soil_nox_tau5hr.mean()
    lower, upper = np.quantile(target, [.01, .99])
    high, peak = float(np.quantile(target, .95)), float(daily.quantile(.95))
    scorer = PeakScorer(frame, high, peak)
    scales = {key: max(value, 1.) for key, value in scorer.components(np.full(len(frame), np.median(target))).items()}
    return {'p1': float(lower), 'p99': float(upper), 'high_threshold': high,
            'peak_threshold': peak, 'scales': scales, 'training_coverage': scorer.coverage}


def sample_weights(frame, high_threshold, tail_weight):
    acute = frame.days_post_fire.between(0, 30).to_numpy()
    high = frame.soil_nox_tau5hr.to_numpy() > high_threshold
    return (np.where(acute, 3., 1.) * np.where(high, tail_weight, 1.)).astype(np.float32)


def prepare_inputs(data, features, train, validation, test):
    inputs = data[features].replace([np.inf, -np.inf], np.nan)
    medians = inputs.iloc[train].median().fillna(0)
    scaler = StandardScaler().fit(inputs.iloc[train].fillna(medians))
    arrays = {}
    land = [i for i, feature in enumerate(features) if feature.startswith('pct_')]
    for name, indices in [('train', train), ('validation', validation), ('test', test)]:
        filled = inputs.iloc[indices].fillna(medians)
        standard = np.clip(scaler.transform(filled), -5, 5).astype(np.float32)
        native = standard.copy()
        native[:, land] = filled.iloc[:, land].to_numpy(dtype=np.float32)
        arrays[name] = {'standardized': standard, 'native': native}
    state = {'features': features, 'medians': medians.to_list(), 'mean': scaler.mean_.tolist(),
             'scale': scaler.scale_.tolist(), 'input_z_limit': 5}
    return arrays, state


def predict(model, values):
    model.eval()
    with torch.no_grad():
        transformed = np.concatenate([model(torch.from_numpy(values[i:i+512])).numpy().flatten()
                                      for i in range(0, len(values), 512)]).astype(np.float64)
    return transformed, np.sinh(transformed)


def fit_candidate(x, y, weights, val_x, scorer, selection, seed, epochs):
    torch.manual_seed(seed)
    model = BaselineNN(x.shape[1])
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(np.arcsinh(y).astype(np.float32)),
                            torch.from_numpy(weights))
    loader = DataLoader(dataset, batch_size=512, shuffle=True, drop_last=len(dataset) % 512 == 1,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=.5)
    best, best_state, best_prediction, patience = float('inf'), None, None, 0
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch_x, batch_y, batch_w in loader:
            optimizer.zero_grad()
            transformed = model(batch_x).flatten()
            errors = torch.nn.functional.smooth_l1_loss(torch.sinh(transformed), torch.sinh(batch_y), beta=10., reduction='none')
            loss = (errors * batch_w).sum() / batch_w.sum()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(loss.item())
        transformed, prediction = predict(model, val_x)
        parts = scorer.components(prediction)
        peak_score = scorer.score(prediction)
        arcsinh_mse = float(np.mean((transformed - scorer.asinh) ** 2))
        value = peak_score if selection == 'peak_score' else arcsinh_mse
        history.append({'epoch': epoch + 1, 'train_loss': float(np.mean(losses)),
                        'selection_score': value, 'peak_score': peak_score,
                        'arcsinh_mse': arcsinh_mse, **parts})
        scheduler.step(value)
        if value < best:
            best, patience = value, 0
            best_state = {key: tensor.clone() for key, tensor in model.state_dict().items()}
            best_prediction = prediction.copy()
        else:
            patience += 1
        if (epoch + 1) % 20 == 0:
            print(f'  epoch {epoch + 1} validation={value:.6f} patience={patience}/20', flush=True)
        if patience >= 20:
            break
    model.load_state_dict(best_state)
    return model, history, best_prediction


def run(data_path, output, arm, epochs=200, seeds=(42, 123, 7), only_fires=None):
    data = pd.read_csv(data_path, parse_dates=['date'])
    data = data[data.days_post_fire >= 0].sort_values(KEYS).reset_index(drop=True)
    features = get_features(data)
    if len(features) != 60:
        raise ValueError('Expected 60 history-model inputs')
    output.mkdir(parents=True, exist_ok=False)
    config = ARMS[arm]
    report = {'arm': arm, 'config': config, 'data_sha256': hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
              'seeds': list(seeds), 'max_epochs': epochs, 'features': features, 'folds': []}
    for outer_fire in sorted(data.fire_name.unique()):
        if only_fires is not None and outer_fire not in only_fires:
            continue
        train, validation, test = split_fires(data, outer_fire)
        train_data, validation_data = data.iloc[train], data.iloc[validation]
        stats = training_statistics(train_data)
        scorer = PeakScorer(validation_data, stats['high_threshold'], stats['peak_threshold'], stats['scales'])
        arrays, preprocessing = prepare_inputs(data, features, train, validation, test)
        y = train_data.soil_nox_tau5hr.to_numpy()
        if config['capped']:
            y = np.clip(y, stats['p1'], stats['p99'])
        weights = sample_weights(train_data, stats['high_threshold'], config['tail_weight'])
        print(f'{outer_fire} | {arm} | train={len(train)} validation={len(validation)} test={len(test)}', flush=True)
        predictions = data.iloc[test][KEYS + ['soil_nox_tau5hr']].rename(columns={'soil_nox_tau5hr': 'actual'}).copy()
        val_predictions = validation_data[KEYS + ['soil_nox_tau5hr']].rename(columns={'soil_nox_tau5hr': 'actual'}).copy()
        fold = {'fire': outer_fire, 'rows': [len(train), len(validation), len(test)],
                'train_fires': sorted(train_data.fire_name.unique()),
                'validation_fires': sorted(validation_data.fire_name.unique()), 'training_statistics': stats,
                'validation_coverage': scorer.coverage, 'preprocessing': preprocessing,
                'training_weight_mean': float(weights.mean()), 'training_weight_max': float(weights.max()), 'seeds': {}}
        for seed in seeds:
            best, histories, selected = float('inf'), {}, None
            for mode in ['standardized', 'native']:
                print(f'{outer_fire} | {arm} | seed={seed} | {mode}', flush=True)
                model, history, val_prediction = fit_candidate(
                    arrays['train'][mode], y, weights, arrays['validation'][mode], scorer,
                    config['selection'], seed, epochs)
                histories[mode] = history
                score = min(row['selection_score'] for row in history)
                if score < best:
                    best, selected, best_model, best_val = score, mode, model, val_prediction
            _, prediction = predict(best_model, arrays['test'][selected])
            predictions[f'seed_{seed}'] = prediction
            val_predictions[f'seed_{seed}'] = best_val
            fold['seeds'][str(seed)] = {'selected_mode': selected, 'selection_score': best,
                                      'best_epoch': min(histories[selected], key=lambda row: row['selection_score'])['epoch'],
                                      'candidate_epochs': {mode: len(h) for mode, h in histories.items()}}
            torch.save(best_model.state_dict(), output / f'{outer_fire}_{seed}.pt')
            (output / f'{outer_fire}_{seed}_history.json').write_text(json.dumps(histories, indent=2, allow_nan=False))
        columns = [f'seed_{seed}' for seed in seeds]
        predictions['prediction'] = predictions[columns].mean(axis=1)
        val_predictions['prediction'] = val_predictions[columns].mean(axis=1)
        fold['validation_ensemble'] = {'peak_score': scorer.score(val_predictions.prediction.to_numpy()),
                                       **scorer.components(val_predictions.prediction.to_numpy())}
        predictions.to_csv(output / f'{outer_fire}_predictions.csv', index=False)
        val_predictions.to_csv(output / f'{outer_fire}_validation.csv', index=False)
        report['folds'].append(fold)
        (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        print(f'{outer_fire} completed ({len(report["folds"])} folds)', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/phase_2_no2_history_model.csv'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 7])
    parser.add_argument('--fires', nargs='+')
    args = parser.parse_args()
    if args.epochs < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Positive epochs and distinct seeds are required')
    torch.set_num_threads(2)
    run(args.data, args.output, args.arm, args.epochs, args.seeds, args.fires)
