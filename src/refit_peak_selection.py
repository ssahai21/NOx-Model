"""Refit validation-chosen settings on all outer-training fires, with fixed budgets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.baseline_nn import BaselineNN
from src.peak_validation_study import ARMS, KEYS, predict, prepare_inputs, sample_weights, training_statistics


def learning_rates(history, epochs):
    if not 1 <= epochs <= len(history):
        raise ValueError('The refit budget must be an observed inner-training epoch')
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([parameter], lr=.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=.5)
    rates = []
    for row in history[:epochs]:
        rates.append(optimizer.param_groups[0]['lr'])
        scheduler.step(row['selection_score'])
    return rates


def fit_fixed(x, y, weights, rates, seed):
    torch.manual_seed(seed)
    model = BaselineNN(x.shape[1])
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(np.arcsinh(y).astype(np.float32)),
                            torch.from_numpy(weights))
    loader = DataLoader(dataset, batch_size=512, shuffle=True, drop_last=len(dataset) % 512 == 1,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-4)
    for epoch, rate in enumerate(rates):
        for group in optimizer.param_groups:
            group['lr'] = rate
        model.train()
        for batch_x, batch_y, batch_w in loader:
            optimizer.zero_grad()
            transformed = model(batch_x).flatten()
            errors = torch.nn.functional.smooth_l1_loss(torch.sinh(transformed), torch.sinh(batch_y), beta=10, reduction='none')
            loss = (batch_w * errors).sum() / batch_w.sum()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite refit loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f'  refit epoch {epoch + 1}/{len(rates)}', flush=True)
    return model


def run(root, shard, shards, only_fires=None):
    manifest = json.loads((root / 'study_manifest.json').read_text())
    decisions = json.loads((root / 'validation_selection.json').read_text())
    plan = json.loads((root / 'refit_plan.json').read_text())
    assert hashlib.sha256((root / 'validation_selection.json').read_bytes()).hexdigest() == plan['decision_sha256']
    for name, expected in plan['code_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
    assert hashlib.sha256(Path(manifest['data']).read_bytes()).hexdigest() == manifest['source_sha256'][manifest['data']]
    data = pd.read_csv(manifest['data'], parse_dates=['date'])
    data = data[data.days_post_fire >= 0].sort_values(KEYS).reset_index(drop=True)
    reports = {arm: json.loads((root / arm / 'report.json').read_text()) for arm in ARMS}
    folds = {arm: {f['fire']: f for f in report['folds']} for arm, report in reports.items()}
    features = reports['inner_arcsinh_capped']['features']
    for index, fire in enumerate(sorted(decisions)):
        if index % shards != shard or (only_fires is not None and fire not in only_fires):
            continue
        train = np.flatnonzero(data.fire_name != fire)
        test = np.flatnonzero(data.fire_name == fire)
        assert fire not in set(data.iloc[train].fire_name)
        # The unused validation slot is training data; no held-out target is scored.
        arrays, preprocessing = prepare_inputs(data, features, train, train, test)
        stats = training_statistics(data.iloc[train])
        control_predictions = None
        for strategy, arm in [('refit_control', 'inner_arcsinh_capped'),
                              ('refit_selected', decisions[fire]['selected_arm'])]:
            output = root / strategy
            output.mkdir(parents=True, exist_ok=True)
            report_path = output / f'{fire}_report.json'
            if report_path.exists():
                raise ValueError(f'Refit already exists: {report_path}')
            config = ARMS[arm]
            target = data.iloc[train].soil_nox_tau5hr.to_numpy()
            if config['capped']:
                target = np.clip(target, stats['p1'], stats['p99'])
            weights = sample_weights(data.iloc[train], stats['high_threshold'], config['tail_weight'])
            predictions = data.iloc[test][KEYS + ['soil_nox_tau5hr']].rename(columns={'soil_nox_tau5hr': 'actual'}).copy()
            report = {'fire': fire, 'strategy': strategy, 'source_arm': arm,
                      'train_fires': sorted(data.iloc[train].fire_name.unique()),
                      'rows': [len(train), len(test)], 'training_statistics': stats,
                      'preprocessing': preprocessing, 'seeds': {}}
            for seed in manifest['seeds']:
                inner = folds[arm][fire]['seeds'][str(seed)]
                mode, epochs = inner['selected_mode'], inner['best_epoch']
                history = json.loads((root / arm / f'{fire}_{seed}_history.json').read_text())[mode]
                rates = learning_rates(history, epochs)
                reused = strategy == 'refit_selected' and arm == 'inner_arcsinh_capped'
                print(f'{fire} | {strategy} | {arm} | seed={seed} | epochs={epochs} reuse={reused}', flush=True)
                if reused:
                    predictions[f'seed_{seed}'] = control_predictions[f'seed_{seed}'].to_numpy()
                    shutil.copy2(root / 'refit_control' / f'{fire}_{seed}.pt', output / f'{fire}_{seed}.pt')
                else:
                    model = fit_fixed(arrays['train'][mode], target, weights, rates, seed)
                    _, prediction = predict(model, arrays['test'][mode])
                    predictions[f'seed_{seed}'] = prediction
                    torch.save(model.state_dict(), output / f'{fire}_{seed}.pt')
                report['seeds'][str(seed)] = {'selected_mode': mode, 'epochs': epochs, 'learning_rates': rates,
                                               'reused_control': reused}
            predictions['prediction'] = predictions[[f'seed_{s}' for s in manifest['seeds']]].mean(axis=1)
            predictions.to_csv(output / f'{fire}_predictions.csv', index=False)
            report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
            if strategy == 'refit_control':
                control_predictions = predictions.copy()
        print(f'{fire} refits completed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--fires', nargs='+')
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error('Shard must be between zero and shards minus one')
    torch.set_num_threads(2)
    run(args.root, args.shard, args.shards, args.fires)
