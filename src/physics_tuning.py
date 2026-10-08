import torch
import numpy as np
import json
import os
import sys
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_data, get_features, prepare_target
from src.physics_loss import PhysicsLoss
from src.hyperparameter_tuning import (FlexNN, make_pinn_dataloaders,
                                        run_final_pinn_with_config)
from src.pinn import train_pinn, evaluate_pinn

BEST_CONFIG = {
    'hidden_dims':  [256, 128, 64],
    'dropout':      0.3,
    'lr':           0.001,
    'batch_size':   512,
    'weight_decay': 0.0001
}
LAMBDA_PHYSICS = 0.01

# =====================================================================
# PHYSICS TERM WEIGHT SEARCH
# =====================================================================

# From the improvement analysis we know:
# - Box model and non-negativity help most
# - Decay term was causing instability
# - Spatial and wind are firing correctly now
# Try varying the per-term weights

PHYSICS_SEARCH = [
    # name,          box,  nonneg, decay, depos, spatial, wind
    ('baseline',     1.0,  2.0,   0.1,   0.5,   0.3,    0.5),
    ('strong_box',   3.0,  2.0,   0.1,   0.5,   0.3,    0.5),
    ('weak_decay',   1.0,  2.0,   0.01,  0.5,   0.3,    0.5),
    ('strong_nonneg',1.0,  4.0,   0.1,   0.5,   0.3,    0.5),
    ('no_spatial',   1.0,  2.0,   0.1,   0.5,   0.0,    0.5),
    ('strong_wind',  1.0,  2.0,   0.1,   0.5,   0.3,    1.5),
    ('no_decay',     1.0,  2.0,   0.0,   0.5,   0.3,    0.5),
    ('box_nonneg',   2.0,  3.0,   0.0,   1.0,   0.0,    0.0),
    ('all_strong',   2.0,  3.0,   0.1,   1.0,   0.5,    1.0),
    ('minimal',      1.0,  1.0,   0.0,   0.0,   0.0,    0.0),
]


def evaluate_physics_config(df, features, physics_config,
                              val_fires, device):
    physics_feature_names = [
        f for f in ['no2', 'ref_no2', 'wind_speed', 'wind_u', 'wind_v',
                    'days_post_fire', 'centroid_lat', 'centroid_lon',
                    'day_of_year_sin_1', 'day_of_year_cos_1']
        if f in df.columns
    ]

    logo   = LeaveOneGroupOut()
    groups = df['fire_name']
    X      = df[features].replace([np.inf, -np.inf], np.nan)
    X_raw  = df[physics_feature_names].replace([np.inf, -np.inf], np.nan)
    y      = df['nox_transformed'].values

    r2s, maes = [], []

    for train_idx, test_idx in logo.split(X, y, groups=groups):
        held_out = groups.iloc[test_idx].iloc[0]
        if held_out not in val_fires:
            continue

        X_tr_raw = X.iloc[train_idx].fillna(X.iloc[train_idx].median())
        X_te_raw = X.iloc[test_idx].fillna(X.iloc[train_idx].median())
        y_tr     = y[train_idx]
        y_te     = np.clip(y[test_idx], y[train_idx].min(), y[train_idx].max())

        scaler      = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_raw.values)
        X_te_scaled = scaler.transform(X_te_raw.values)

        X_raw_tr = X_raw.iloc[train_idx].fillna(
            X_raw.iloc[train_idx].median()).values
        X_raw_te = X_raw.iloc[test_idx].fillna(
            X_raw.iloc[train_idx].median()).values

        train_loader, test_loader = make_pinn_dataloaders(
            X_tr_scaled, y_tr, X_raw_tr,
            X_te_scaled,  y_te, X_raw_te,
            batch_size=BEST_CONFIG['batch_size']
        )

        name, box, nonneg, decay, depos, spatial, wind = physics_config
        criterion = PhysicsLoss(
            lambda_physics    = LAMBDA_PHYSICS,
            lambda_box        = box,
            lambda_nonneg     = nonneg,
            lambda_decay      = decay,
            lambda_deposition = depos,
            lambda_spatial    = spatial,
            lambda_wind       = wind
        )

        model = FlexNN(
            X_tr_scaled.shape[1],
            BEST_CONFIG['hidden_dims'],
            BEST_CONFIG['dropout']
        ).to(device)

        model, _ = train_pinn(
            model, train_loader, test_loader, criterion,
            physics_feature_names,
            n_epochs=150, lr=BEST_CONFIG['lr'],
            patience=15, device=device
        )

        _, preds, labels = evaluate_pinn(
            model, test_loader, criterion,
            physics_feature_names, device
        )

        r2s.append(r2_score(labels, preds))
        maes.append(mean_absolute_error(
            np.sinh(np.clip(preds, labels.min(), labels.max())),
            np.sinh(labels)
        ))

    return np.mean(r2s) if r2s else None, np.mean(maes) if maes else None


if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    df = load_data('data/nox_clean.csv')
    features = get_features(df)
    df['nox_transformed'] = prepare_target(df)

    val_fires = ['Dixie Fire', 'North Complex']

    print(f"\nTesting {len(PHYSICS_SEARCH)} physics weight configs")
    print(f"Validation fires: {val_fires}\n")

    results = []

    for config in PHYSICS_SEARCH:
        name = config[0]
        r2, mae = evaluate_physics_config(
            df, features, config, val_fires, device
        )
        results.append({'name': name, 'r2': r2, 'mae': mae,
                        'config': config})
        print(f"  {name:<20} R²={r2:.4f}  MAE={mae:.4f}")

    results.sort(key=lambda x: x['r2'], reverse=True)

    print(f"\n{'='*50}")
    print("PHYSICS WEIGHT RESULTS (sorted by R²)")
    print(f"{'='*50}")
    for r in results:
        print(f"  {r['name']:<20} R²={r['r2']:.4f}  MAE={r['mae']:.4f}")

    best = results[0]
    print(f"\nBest physics config: {best['name']}")
    print(f"Config: {best['config']}")

    with open('results/physics_tuning_results.json', 'w') as f:
        serializable = [
            {'name': r['name'], 'r2': float(r['r2']),
             'mae': float(r['mae'])}
            for r in results
        ]
        json.dump(serializable, f, indent=2)
    print("Saved: results/physics_tuning_results.json")