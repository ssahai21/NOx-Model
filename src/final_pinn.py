"""Paired baseline and PINN evaluation with an untouched outer test fire."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baseline_nn import BaselineNN
from src.data_loader import get_features, load_data
from src.physics_loss import PhysicsLoss
from src.pinn import PINNDataset, evaluate_pinn, train_pinn


PHYSICS_FEATURES = [
    'no2', 'ref_no2', 'wind_speed', 'wind_u', 'wind_v', 'days_post_fire',
    'centroid_lat', 'centroid_lon', 'day_of_year_sin_1', 'day_of_year_cos_1',
]


def metrics(actual, predicted):
    return {
        'r2_arcsinh': r2_score(np.arcsinh(actual), np.arcsinh(predicted)),
        'r2_physical': r2_score(actual, predicted),
        'mae': mean_absolute_error(actual, predicted),
        'percent_bias': 100 * float((predicted - actual).sum()) / float(actual.sum())
        if actual.sum() else None,
    }


def run_comparison(df, output, seeds, epochs, device, input_z_limit=None,
                   select_land_cover=False, post_fire_only=False, physics_ablation=False,
                   acute_weight=1.0, flux_loss_beta=None,
                   clip_training_target=True, baseline_only=False,
                   physics_repairs=False, model_names=None, save_histories=False):
    excluded_rows = 0
    if post_fire_only:
        days = df['days_post_fire']
        if not np.isfinite(days).all():
            raise ValueError('The post-fire filter requires finite day offsets.')
        excluded_rows = int((days < 0).sum())
        df = df.loc[days >= 0].reset_index(drop=True)
        print(f'Excluded {excluded_rows} pre-ignition rows.')
    features = get_features(df)
    physics_features = [name for name in PHYSICS_FEATURES if name in df]
    inputs = df[features].replace([np.inf, -np.inf], np.nan)
    model_configs = {'baseline': {'lambda_physics': 0.0}, 'pinn': {'lambda_physics': 0.01}}
    if physics_ablation:
        physics_features += ['day_of_year', 'month']
        model_configs.update({
            'pinn_molar': {'lambda_physics': 0.01, 'molar_columns': True},
            'pinn_without_box_deposition': {
                'lambda_physics': 0.01, 'molar_columns': True,
                'lambda_box': 0.0, 'lambda_deposition': 0.0,
            },
        })
    if acute_weight != 1.0:
        model_configs = {
            name: {**model_configs[name], 'acute_weight': acute_weight}
            for name in ('baseline', 'pinn_without_box_deposition')
        }
    if physics_repairs:
        for name in ['day_of_year', 'month']:
            if name not in physics_features:
                physics_features.append(name)
        common = {
            'lambda_physics': .01, 'lambda_box': 0.0, 'lambda_deposition': 0.0,
            'molar_columns': True, 'acute_weight': acute_weight,
        }
        repaired = {**common, 'same_day_pairs': True, 'stable_regularization': True}
        model_configs = {
            'baseline': {'lambda_physics': 0.0, 'acute_weight': acute_weight},
            'legacy_pinn': common,
            'repaired_pinn': repaired,
            'pair_only_pinn': {**repaired, 'lambda_nonneg': 0.0, 'lambda_decay': 0.0},
        }
    if flux_loss_beta is not None:
        if physics_repairs:
            model_configs = {name: {**config, 'flux_loss_beta': flux_loss_beta}
                             for name, config in model_configs.items()}
        else:
            model_configs = {'baseline': {**model_configs['baseline'], 'flux_loss_beta': flux_loss_beta}}
    if baseline_only:
        model_configs = {'baseline': model_configs['baseline']}
    if model_names is not None:
        missing_models = set(model_names) - set(model_configs)
        if missing_models:
            raise ValueError(f'Unknown model configurations: {sorted(missing_models)}')
        model_configs = {name: model_configs[name] for name in model_names}
    physics = df[physics_features].replace([np.inf, -np.inf], np.nan)
    if physics_repairs:
        # Absolute dates and fire IDs are physical context only, never NN inputs.
        physics['_physics_date'] = pd.to_datetime(df.date).to_numpy(dtype='datetime64[D]').astype(np.int64)
        physics['_physics_fire'] = pd.factorize(df.fire_name, sort=True)[0]
        physics_features = physics.columns.tolist()
    target = df['soil_nox_tau5hr'].to_numpy()
    if not np.isfinite(target).all():
        raise ValueError('The tau5hr target must be finite.')

    output.mkdir(parents=True, exist_ok=False)
    report = {
        'seeds': seeds, 'max_epochs': epochs, 'device': device,
        'input_z_limit': input_z_limit, 'select_land_cover': select_land_cover,
        'post_fire_only': post_fire_only, 'excluded_rows': excluded_rows,
        'model_configs': model_configs, 'clip_training_target': clip_training_target,
        'folds': [],
    }
    for outer_train, test in LeaveOneGroupOut().split(inputs, groups=df['fire_name']):
        fire = df.iloc[test[0]]['fire_name']
        outer_rows = df.iloc[outer_train]
        # Keep whole dates together. Reserve each training fire's last 20% of dates.
        date_rank = outer_rows.groupby('fire_name')['date'].rank(method='dense', pct=True)
        validation_mask = date_rank.to_numpy() > 0.8
        train, val = outer_train[~validation_mask], outer_train[validation_mask]
        if not len(train) or not len(val):
            raise ValueError('The training fires need enough dates for a validation block.')
        print(f'{fire} | train={len(train)} validation={len(val)} test={len(test)}')
        median = inputs.iloc[train].median().fillna(0)
        physics_median = physics.iloc[train].median().fillna(0)
        scaler = StandardScaler().fit(inputs.iloc[train].fillna(median))
        lower, upper = np.quantile(target[train], [0.01, 0.99])

        land_cover = [i for i, feature in enumerate(features) if feature.startswith('pct_')]
        modes = ['standardized', 'native'] if select_land_cover else ['standardized']
        datasets = {mode: [] for mode in modes}
        clipped_rows = []
        for indices in (train, val, test):
            raw_target = target[indices]
            if indices is train and clip_training_target:
                raw_target = np.clip(raw_target, lower, upper)
            filled = inputs.iloc[indices].fillna(median)
            scaled = scaler.transform(filled)
            if input_z_limit is not None:
                clipped_rows.append(int((np.abs(scaled) > input_z_limit).any(axis=1).sum()))
                scaled = np.clip(scaled, -input_z_limit, input_z_limit)
            raw_physics = physics.iloc[indices].fillna(physics_median).to_numpy()
            for mode in modes:
                model_inputs = scaled
                if mode == 'native':
                    model_inputs = scaled.copy()
                    model_inputs[:, land_cover] = filled.iloc[:, land_cover].to_numpy()
                datasets[mode].append(PINNDataset(
                    model_inputs, np.arcsinh(raw_target), raw_physics, physics_features,
                ))
        predictions = df.iloc[test][['fire_name', 'plot_id', 'date']].copy()
        predictions['actual'] = target[test]
        acute = df.iloc[test]['days_post_fire'].between(0, 30).to_numpy()
        tail_rows = int(((target[train] < lower) | (target[train] > upper)).sum())
        fold = {
            'fire': fire, 'rows': [len(train), len(val), len(test)], 'models': {},
            'training_target': {
                'p1': float(lower), 'p99': float(upper),
                'clipped_rows': tail_rows if clip_training_target else 0,
            },
        }
        if clipped_rows:
            fold['input_clipped_rows'] = dict(zip(('train', 'validation', 'test'), clipped_rows))
        if select_land_cover:
            benchmark_features = ['wind_speed'] + [
                f'day_of_year_{term}_{i}' for term in ('sin', 'cos') for i in (1, 2, 3)
            ]
            benchmark = inputs[benchmark_features].fillna(median[benchmark_features])
            benchmark_scaler = StandardScaler().fit(benchmark.iloc[train])
            ridge = Ridge(alpha=1).fit(
                benchmark_scaler.transform(benchmark.iloc[train]),
                np.arcsinh(np.clip(target[train], lower, upper)),
            )
            predictions['wind_season_ridge'] = np.sinh(
                ridge.predict(benchmark_scaler.transform(benchmark.iloc[test]))
            )
            predictions['train_median'] = np.median(target[train])
            fold['benchmarks'] = {
                name: metrics(target[test], predictions[name].to_numpy())
                for name in ('wind_season_ridge', 'train_median')
            }
        for name, config in model_configs.items():
            by_seed = {}
            columns = []
            for seed in seeds:
                print(f'{fire} | {name} | seed={seed}')
                criterion = PhysicsLoss(**config)
                validation_mse = {}
                best_model = None
                candidate_histories = {}
                for mode, (training, validation, _) in datasets.items():
                    torch.manual_seed(seed)
                    model = BaselineNN(len(features)).to(device)
                    # BatchNorm requires at least two samples in each training batch.
                    train_loader = DataLoader(
                        training, batch_size=512, shuffle=True,
                        drop_last=len(train) % 512 == 1,
                        generator=torch.Generator().manual_seed(seed),
                    )
                    model, history = train_pinn(
                        model, train_loader, DataLoader(validation, batch_size=512),
                        criterion, physics_features, n_epochs=epochs, device=device,
                        data_only_validation=True,
                    )
                    candidate_histories[mode] = history
                    score = min(history['val_loss'])
                    validation_mse[mode] = score
                    if best_model is None or score < validation_mse[selected_mode]:
                        best_model, selected_mode, best_history = model, mode, history
                print(f'  Selected {selected_mode} | validation MSE {validation_mse}', flush=True)
                _, pred, _ = evaluate_pinn(
                    best_model, DataLoader(datasets[selected_mode][2], batch_size=512),
                    criterion, physics_features, device, data_only=True,
                )
                column = f'{name}_{seed}'
                predictions[column] = np.sinh(pred.astype(np.float64))
                columns.append(column)
                by_seed[seed] = {
                    **metrics(target[test], predictions[column].to_numpy()),
                    'best_epoch': int(np.argmin(best_history['val_loss'])) + 1,
                    'epochs_run': len(best_history['val_loss']),
                    'land_cover_mode': selected_mode,
                    'validation_mse': validation_mse,
                }
                torch.save(best_model.state_dict(), output / f'{fire}_{column}.pt')
                if save_histories:
                    (output / f'{fire}_{column}_history.json').write_text(json.dumps({
                        'selected_mode': selected_mode, 'candidates': candidate_histories,
                    }, indent=2, allow_nan=False))

            ensemble = predictions[columns].mean(axis=1)
            predictions[f'{name}_ensemble'] = ensemble
            summary = {}
            for key in ('r2_arcsinh', 'r2_physical', 'mae', 'percent_bias'):
                values = [result[key] for result in by_seed.values()]
                if values[0] is not None:
                    summary[key] = {'mean': np.mean(values), 'std': np.std(values)}
            fold['models'][name] = {
                'lambda_physics': config['lambda_physics'], 'seeds': by_seed, 'summary': summary,
                'ensemble': metrics(target[test], ensemble.to_numpy()),
                'days_0_to_30': metrics(target[test][acute], ensemble.to_numpy()[acute])
                if acute.sum() >= 2 else None,
            }
        predictions.to_csv(output / f'{fire}_predictions.csv', index=False)
        report['folds'].append(fold)
        (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/nox_clean.csv')
    parser.add_argument('--output', type=Path, default=Path('results/honest_logo'))
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 7])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--input-z-limit', type=float, help='Cap standardized model inputs symmetrically.')
    parser.add_argument('--select-land-cover', action='store_true',
                        help='Select standardized or native land-cover inputs using validation MSE.')
    parser.add_argument('--post-fire-only', action='store_true',
                        help='Include ignition day and later rows in every fold.')
    parser.add_argument('--physics-ablation', action='store_true',
                        help='Compare molar physics with and without box and deposition terms.')
    parser.add_argument('--acute-weight', type=float, default=1.0,
                        help='Multiply data loss weights on days 0–30 in the two-model experiment.')
    parser.add_argument('--flux-loss-beta', type=float,
                        help='Train only the plain NN with smooth L1 loss in flux units.')
    parser.add_argument('--uncapped-target', action='store_true',
                        help='Retain all training label values, including the extreme tails.')
    parser.add_argument('--baseline-only', action='store_true',
                        help='Evaluate only the plain neural network.')
    parser.add_argument('--physics-repairs', action='store_true',
                        help='Expose legacy, corrected, and pair-only physics configurations.')
    parser.add_argument('--models', nargs='+', help='Restrict the model configurations to evaluate.')
    parser.add_argument('--save-histories', action='store_true',
                        help='Save all candidate histories, including physics clamp telemetry.')
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    args = parser.parse_args()
    if args.epochs < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Use positive epochs and distinct seeds.')
    if args.input_z_limit is not None and (
        not np.isfinite(args.input_z_limit) or args.input_z_limit <= 0
    ):
        parser.error('The input z limit must be positive and finite.')
    if args.select_land_cover and args.input_z_limit is None:
        parser.error('Land-cover selection requires an input z limit.')
    if not np.isfinite(args.acute_weight) or args.acute_weight < 1:
        parser.error('Acute weight must be finite and at least 1.')
    if args.acute_weight != 1 and not (args.physics_ablation and args.post_fire_only):
        parser.error('Acute weighting requires --physics-ablation and --post-fire-only.')
    if args.flux_loss_beta is not None and (
        not np.isfinite(args.flux_loss_beta) or args.flux_loss_beta <= 0
    ):
        parser.error('Flux loss beta must be finite and positive.')
    if args.baseline_only and args.models is not None:
        parser.error('Choose either --baseline-only or --models.')
    if args.device == 'cpu':
        torch.set_num_threads(2)
    run_comparison(
        load_data(args.data), args.output, args.seeds, args.epochs, args.device,
        input_z_limit=args.input_z_limit, select_land_cover=args.select_land_cover,
        post_fire_only=args.post_fire_only, physics_ablation=args.physics_ablation, acute_weight=args.acute_weight,
        flux_loss_beta=args.flux_loss_beta,
        clip_training_target=not args.uncapped_target, baseline_only=args.baseline_only,
        physics_repairs=args.physics_repairs, model_names=args.models,
        save_histories=args.save_histories,
    )
