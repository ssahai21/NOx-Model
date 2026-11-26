"""Matched-input XGBoost evaluation with held-out fires excluded from validation."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from src.compare_target_caps import subset_scores
from src.data_loader import get_features, load_data
from src.final_pinn import metrics


KEYS = ['fire_name', 'plot_id', 'date']


def arcsinh_mse(actual, predicted):
    return float(np.mean((actual.astype(float) - predicted.astype(float)) ** 2))


def flux_arcsinh_mse(actual, predicted):
    return arcsinh_mse(np.arcsinh(actual), np.arcsinh(predicted))


def run(args):
    study = json.loads(Path('results/phase2/target_cap_study/study_manifest.json').read_text())
    data_path = Path(study['data'])
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != study['data_sha256']:
        raise ValueError('Dataset differs from the NN comparison.')
    reference_path = Path(study['arms'][f'{args.loss}_capped']['directory'])
    reference = json.loads((reference_path / 'metrics.json').read_text())
    reference_folds = {fold['fire']: fold for fold in reference['folds']}
    thresholds = json.loads(Path('results/phase2/target_cap_study/summary.json').read_text())['thresholds']
    data = load_data(data_path)
    data = data.loc[data.days_post_fire >= 0].reset_index(drop=True)
    features = get_features(data)
    inputs = data[features].replace([np.inf, -np.inf], np.nan)
    target = data.soil_nox_tau5hr.to_numpy()
    if not np.isfinite(target).all():
        raise ValueError('Target must be finite.')
    land_cover = [i for i, name in enumerate(features) if name.startswith('pct_')]
    parameters = {
        'n_estimators': args.trees, 'learning_rate': .01, 'max_depth': 5,
        'subsample': .7, 'colsample_bytree': .7, 'min_child_weight': 20,
        'reg_lambda': 1, 'tree_method': 'hist', 'n_jobs': 2,
        'early_stopping_rounds': 50, 'disable_default_eval_metric': True,
    }
    if args.loss == 'flux':
        parameters.update(objective='reg:pseudohubererror', huber_slope=10)
        validation_metric = flux_arcsinh_mse
    else:
        parameters.update(objective='reg:squarederror')
        validation_metric = arcsinh_mse
    reuse = None
    if args.reuse_run is not None:
        reuse = json.loads((args.reuse_run / 'metrics.json').read_text())
        if reuse['loss'] != args.loss or reuse['data_sha256'] != study['data_sha256']:
            raise ValueError('Reuse run uses a different loss or dataset.')
        if reuse['seeds'] != args.seeds:
            raise ValueError('Reuse run uses different seeds.')
        if reuse['features'] != features or reuse['nn_reference'] != str(reference_path):
            raise ValueError('Reuse run uses different features or NN preprocessing.')
        for key, value in parameters.items():
            if key != 'n_estimators' and reuse['parameters'][key] != value:
                raise ValueError(f'Reuse setting differs: {key}')
        if args.trees <= reuse['parameters']['n_estimators']:
            raise ValueError('The extended budget must exceed the original budget.')
    reuse_folds = {fold['fire']: fold for fold in reuse['folds']} if reuse else {}
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        'reused_run': str(args.reuse_run) if args.reuse_run else None,
        'algorithm': 'XGBoost', 'version': xgb.__version__, 'loss': args.loss,
        'parameters': parameters, 'seeds': args.seeds, 'data_sha256': study['data_sha256'],
        'features': features, 'nn_reference': str(reference_path),
        'preprocessing': 'Same train imputation/scaling/input cap and NN-selected land-cover mode per seed.',
        'validation': study['validation'], 'training_target_cap': [1, 99],
        'acute_weight': 3, 'folds': [],
    }
    pooled = []
    for train_outer, test in LeaveOneGroupOut().split(inputs, groups=data.fire_name):
        fire = data.iloc[test[0]].fire_name
        outer = data.iloc[train_outer]
        validation_mask = outer.groupby('fire_name').date.rank(method='dense', pct=True).to_numpy() > .8
        train, validation = train_outer[~validation_mask], train_outer[validation_mask]
        old_fold = reference_folds[fire]
        if [len(train), len(validation), len(test)] != old_fold['rows']:
            raise ValueError(f'{fire}: train/validation/test sizes changed.')
        assert set(data.iloc[train].fire_name).isdisjoint({fire})
        assert set(data.iloc[validation].fire_name).isdisjoint({fire})
        assert not set(train) & set(validation)
        median = inputs.iloc[train].median().fillna(0)
        scaler = StandardScaler().fit(inputs.iloc[train].fillna(median))
        filled = [inputs.iloc[index].fillna(median) for index in (train, validation, test)]
        scaled = [np.clip(scaler.transform(frame), -5, 5) for frame in filled]
        native = []
        for frame, values in zip(filled, scaled):
            raw_land = values.copy()
            raw_land[:, land_cover] = frame.iloc[:, land_cover].to_numpy()
            native.append(raw_land)
        modes = {'standardized': scaled, 'native': native}
        lower, upper = np.quantile(target[train], [.01, .99])
        y_train = np.clip(target[train], lower, upper)
        y_validation = target[validation]
        if args.loss == 'arcsinh':
            y_train, y_validation = np.arcsinh(y_train), np.arcsinh(y_validation)
        weights = np.where(data.iloc[train].days_post_fire.between(0, 30), 3., 1.)
        predictions = data.iloc[test][KEYS + ['days_post_fire']].copy()
        predictions['actual'] = target[test]
        nn_predictions = pd.read_csv(reference_path / f'{fire}_predictions.csv', parse_dates=['date'])
        pd.testing.assert_frame_equal(
            predictions[KEYS].reset_index(drop=True), nn_predictions[KEYS], check_dtype=False)
        np.testing.assert_allclose(predictions.actual, nn_predictions.actual, rtol=1e-12, atol=1e-12)
        predictions['nn_ensemble'] = nn_predictions.baseline_ensemble.to_numpy()
        seed_results = {}
        for seed in args.seeds:
            start = time.monotonic()
            mode = old_fold['models']['baseline']['seeds'][str(seed)]['land_cover_mode']
            x_train, x_validation, x_test = [values.astype(np.float32) for values in modes[mode]]
            model = xgb.XGBRegressor(**parameters, random_state=seed, eval_metric=validation_metric)
            old_seed = reuse_folds.get(fire, {}).get('seeds', {}).get(str(seed))
            reused = old_seed is not None and old_seed['trees_fitted'] < reuse['parameters']['n_estimators']
            if reused:
                # Early stopping was reached before the old limit, so a higher limit gives the same fit.
                if old_seed['land_cover_mode'] != mode:
                    raise ValueError('Reused preprocessing differs.')
                model.load_model(args.reuse_run / f'{fire}_{seed}.ubj')
            else:
                model.fit(x_train, y_train, sample_weight=weights,
                          eval_set=[(x_validation, y_validation)], verbose=False)
            values = model.predict(x_test, iteration_range=(0, model.best_iteration + 1)).astype(float)
            if args.loss == 'arcsinh':
                values = np.sinh(values)
            if not np.isfinite(values).all():
                raise ValueError(f'{fire}/{seed}: nonfinite predictions.')
            predictions[f'xgb_{seed}'] = values
            model.save_model(args.output / f'{fire}_{seed}.ubj')
            seed_results[seed] = {
                **metrics(target[test], values), 'best_iteration': model.best_iteration,
                'trees_fitted': model.get_booster().num_boosted_rounds(),
                'reused_early_stopped_model': reused,
                'land_cover_mode': mode,
                'best_validation_arcsinh_mse': float(model.best_score),
            }
            print(f'{fire} seed={seed} trees={model.best_iteration+1} '
                  f'MAE={seed_results[seed]["mae"]:.3f} elapsed={time.monotonic()-start:.1f}s', flush=True)
        predictions['xgb_ensemble'] = predictions[[f'xgb_{s}' for s in args.seeds]].mean(axis=1)
        predictions['high_flux'] = predictions.actual > thresholds[fire]['high_flux_row']
        predictions['extreme_flux'] = predictions.actual > thresholds[fire]['extreme_flux_row']
        daily = predictions.groupby('date', as_index=False)[['actual', 'xgb_ensemble', 'nn_ensemble']].mean()
        daily['peak_day'] = daily.actual > thresholds[fire]['peak_day_mean']
        fold = {'fire': fire, 'rows': old_fold['rows'], 'seeds': seed_results, 'models': {}}
        for name in ('xgb', 'nn'):
            column = f'{name}_ensemble'
            fold['models'][name] = {
                'ensemble': metrics(target[test], predictions[column].to_numpy()),
                'first_month': subset_scores(predictions, column, predictions.days_post_fire.between(0, 30)),
                'high_flux': subset_scores(predictions, column, predictions.high_flux),
                'extreme_flux': subset_scores(predictions, column, predictions.extreme_flux),
                'peak_days_scores': subset_scores(daily, column, daily.peak_day),
            }
        predictions.to_csv(args.output / f'{fire}_predictions.csv', index=False)
        pooled.append(predictions)
        report['folds'].append(fold)
        (args.output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    frame = pd.concat(pooled, ignore_index=True)
    daily = frame.groupby(['fire_name', 'date'], as_index=False)[['actual', 'xgb_ensemble', 'nn_ensemble']].mean()
    daily['peak_day'] = [value > thresholds[fire]['peak_day_mean']
                         for fire, value in zip(daily.fire_name, daily.actual)]
    summary = {'rows': len(frame), 'fires': len(report['folds']), 'loss': args.loss, 'models': {}}
    for name in ('xgb', 'nn'):
        column = f'{name}_ensemble'
        summary['models'][name] = {
            'equal_fire_mean_mae': float(np.mean([f['models'][name]['ensemble']['mae'] for f in report['folds']])),
            'equal_fire_mean_r2_physical': float(np.mean([f['models'][name]['ensemble']['r2_physical'] for f in report['folds']])),
            'pooled': metrics(frame.actual.to_numpy(), frame[column].to_numpy()),
            'first_month': subset_scores(frame, column, frame.days_post_fire.between(0, 30)),
            'extreme_flux': subset_scores(frame, column, frame.extreme_flux),
            'peak_days_scores': subset_scores(daily, column, daily.peak_day),
        }
    summary['xgb_mae_wins'] = sum(
        fold['models']['xgb']['ensemble']['mae'] < fold['models']['nn']['ensemble']['mae']
        for fold in report['folds'])
    summary['seeds_at_tree_budget'] = sum(
        seed['trees_fitted'] == args.trees for fold in report['folds'] for seed in fold['seeds'].values())
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loss', choices=['arcsinh', 'flux'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 7])
    parser.add_argument('--trees', type=int, default=1000)
    parser.add_argument('--reuse-run', type=Path,
                        help='Reuse converged models and refit only models that hit a smaller tree budget.')
    args = parser.parse_args()
    if args.trees < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Use positive tree count and distinct seeds.')
    run(args)
