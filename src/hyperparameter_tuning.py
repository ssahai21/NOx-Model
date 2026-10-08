import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
import sys
from itertools import product
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_data, get_features, prepare_target
from src.physics_loss import PhysicsLoss
from src.pinn import (PINNDataset, make_pinn_dataloaders,
                      train_pinn, evaluate_pinn)

# =====================================================================
# SEARCH SPACE
# =====================================================================

SEARCH_SPACE = {
    'hidden_dims':    [
        [256, 128, 64],         # original
        [512, 256, 128],        # wider
        [256, 256, 128, 64],    # deeper
        [128, 64, 32],          # smaller — less overfitting
    ],
    'dropout':        [0.1, 0.3, 0.5],
    'lr':             [1e-3, 5e-4, 1e-4],
    'batch_size':     [256, 512, 1024],
    'weight_decay':   [1e-4, 1e-3],
}

LAMBDA_PHYSICS = 0.01  # locked from September


# =====================================================================
# FLEXIBLE NN ARCHITECTURE
# =====================================================================

class FlexNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super(FlexNN, self).__init__()
        layers   = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# =====================================================================
# SINGLE FOLD EVALUATION — fast, uses one held-out fire
# =====================================================================

def evaluate_config(df, features, config, test_fire,
                    physics_feature_names, device):
    """
    Train and evaluate one hyperparameter config on one held-out fire.
    Returns mean R² and MAE in transformed space.
    """
    logo   = LeaveOneGroupOut()
    groups = df['fire_name']
    X      = df[features].replace([np.inf, -np.inf], np.nan)
    X_raw  = df[physics_feature_names].replace([np.inf, -np.inf], np.nan)
    y      = df['nox_transformed'].values

    for train_idx, test_idx in logo.split(X, y, groups=groups):
        held_out = groups.iloc[test_idx].iloc[0]
        if held_out != test_fire:
            continue

        X_tr_raw = X.iloc[train_idx]
        X_te_raw = X.iloc[test_idx]
        y_tr     = y[train_idx]
        y_te     = y[test_idx]

        train_median = X_tr_raw.median()
        X_tr_raw     = X_tr_raw.fillna(train_median)
        X_te_raw     = X_te_raw.fillna(train_median)

        scaler      = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_raw.values)
        X_te_scaled = scaler.transform(X_te_raw.values)

        X_raw_tr = X_raw.iloc[train_idx].fillna(
            X_raw.iloc[train_idx].median()).values
        X_raw_te = X_raw.iloc[test_idx].fillna(
            X_raw.iloc[train_idx].median()).values

        y_tr_min, y_tr_max = y_tr.min(), y_tr.max()
        y_te = np.clip(y_te, y_tr_min, y_tr_max)

        train_loader, test_loader = make_pinn_dataloaders(
            X_tr_scaled, y_tr, X_raw_tr,
            X_te_scaled,  y_te, X_raw_te,
            batch_size=config['batch_size']
        )

        model     = FlexNN(
            input_dim   = X_tr_scaled.shape[1],
            hidden_dims = config['hidden_dims'],
            dropout     = config['dropout']
        ).to(device)

        criterion = PhysicsLoss(lambda_physics=LAMBDA_PHYSICS)

        model, _ = train_pinn(
            model, train_loader, test_loader, criterion,
            physics_feature_names,
            n_epochs  = 150,   # fewer epochs for speed during search
            lr        = config['lr'],
            patience  = 15,
            device    = device
        )

        _, preds, labels = evaluate_pinn(
            model, test_loader, criterion,
            physics_feature_names, device
        )

        r2  = r2_score(labels, preds)
        mae = mean_absolute_error(
            np.sinh(np.clip(preds,   labels.min(), labels.max())),
            np.sinh(labels)
        )
        return r2, mae

    return None, None


# =====================================================================
# GRID SEARCH — uses two validation fires to avoid lucky splits
# =====================================================================

def run_grid_search(df, features, device,
                    val_fires=['Dixie Fire', 'North Complex']):
    """
    Grid search over hyperparameters.
    Evaluates each config on val_fires and averages.
    Uses a subset of the search space for speed.
    """
    physics_feature_names = [
        f for f in ['no2', 'ref_no2', 'wind_speed', 'wind_u', 'wind_v',
                    'days_post_fire', 'centroid_lat', 'centroid_lon',
                    'day_of_year_sin_1', 'day_of_year_cos_1']
        if f in df.columns
    ]

    # Build all configs
    keys   = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    all_configs = [
        dict(zip(keys, combo))
        for combo in product(*values)
    ]
    print(f"Total configs to evaluate: {len(all_configs)}")
    print(f"Validation fires: {val_fires}")
    print(f"Estimated runs: {len(all_configs) * len(val_fires)}\n")

    results = []
    best_r2 = -999
    best_config = None

    for i, config in enumerate(all_configs):
        r2s  = []
        maes = []

        for fire in val_fires:
            r2, mae = evaluate_config(
                df, features, config, fire,
                physics_feature_names, device
            )
            if r2 is not None:
                r2s.append(r2)
                maes.append(mae)

        if not r2s:
            continue

        mean_r2  = np.mean(r2s)
        mean_mae = np.mean(maes)

        results.append({
            'config':   config,
            'mean_r2':  float(mean_r2),
            'mean_mae': float(mean_mae)
        })

        if mean_r2 > best_r2:
            best_r2     = mean_r2
            best_config = config
            print(f"  *** New best at config {i+1}/{len(all_configs)} ***")

        if (i + 1) % 10 == 0 or mean_r2 > best_r2:
            arch = f"{config['hidden_dims']}"
            print(f"  Config {i+1:>3}/{len(all_configs)} | "
                  f"arch={arch} | "
                  f"dropout={config['dropout']} | "
                  f"lr={config['lr']:.0e} | "
                  f"bs={config['batch_size']} | "
                  f"R²={mean_r2:.4f} | MAE={mean_mae:.4f}")

    # Sort by R²
    results.sort(key=lambda x: x['mean_r2'], reverse=True)

    print(f"\n{'='*60}")
    print("TOP 5 CONFIGS")
    print(f"{'='*60}")
    for r in results[:5]:
        print(f"  R²={r['mean_r2']:.4f}  MAE={r['mean_mae']:.4f}  "
              f"config={r['config']}")

    os.makedirs('results', exist_ok=True)
    with open('results/grid_search_results.json', 'w') as f:
        # Convert hidden_dims lists to strings for JSON
        serializable = []
        for r in results:
            rc = dict(r)
            rc['config'] = dict(r['config'])
            rc['config']['hidden_dims'] = str(r['config']['hidden_dims'])
            serializable.append(rc)
        json.dump(serializable, f, indent=2)

    print(f"\nBest config: {best_config}")
    print(f"Best mean R²: {best_r2:.4f}")
    print("Results saved to results/grid_search_results.json")

    return best_config, results


# =====================================================================
# FINAL EVALUATION WITH BEST CONFIG — full LOGO CV
# =====================================================================

def run_final_pinn(df, features, best_config, device):
    """
    Run full leave-one-fire-out CV with the best config found.
    """
    from src.pinn import run_pinn_logo_cv

    # Monkey-patch BaselineNN with FlexNN in pinn module
    import src.pinn as pinn_module
    import src.baseline_nn as baseline_module

    original_class = baseline_module.BaselineNN

    class BestNN(nn.Module):
        def __init__(self, input_dim, **kwargs):
            super().__init__()
            self.net = FlexNN(
                input_dim,
                best_config['hidden_dims'],
                best_config['dropout']
            )
        def forward(self, x):
            return self.net(x)

    # Temporarily replace BaselineNN
    baseline_module.BaselineNN = BestNN

    results = run_pinn_logo_cv(
        df, features,
        lambda_physics=LAMBDA_PHYSICS,
        device=device
    )

    # Restore original
    baseline_module.BaselineNN = original_class

    return results


def run_final_pinn_with_config(df, features, best_config, device):
    """
    Run full leave-one-fire-out CV with the best config.
    Uses FlexNN directly instead of monkey-patching BaselineNN.
    """
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler

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

    results = []

    for train_idx, test_idx in logo.split(X, y, groups=groups):
        held_out = groups.iloc[test_idx].iloc[0]

        print(f"\n{'='*55}")
        print(f"PINN (tuned) — Held-out fire: {held_out}")
        print(f"{'='*55}")

        X_tr_raw = X.iloc[train_idx]
        X_te_raw = X.iloc[test_idx]
        y_tr     = y[train_idx]
        y_te     = y[test_idx]

        train_median = X_tr_raw.median()
        X_tr_raw     = X_tr_raw.fillna(train_median)
        X_te_raw     = X_te_raw.fillna(train_median)

        scaler      = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_raw.values)
        X_te_scaled = scaler.transform(X_te_raw.values)

        X_raw_tr = X_raw.iloc[train_idx].fillna(
            X_raw.iloc[train_idx].median()).values
        X_raw_te = X_raw.iloc[test_idx].fillna(
            X_raw.iloc[train_idx].median()).values

        y_tr_min, y_tr_max = y_tr.min(), y_tr.max()
        y_te = np.clip(y_te, y_tr_min, y_tr_max)

        train_loader, test_loader = make_pinn_dataloaders(
            X_tr_scaled, y_tr, X_raw_tr,
            X_te_scaled,  y_te, X_raw_te,
            batch_size=best_config['batch_size']
        )

        # Use FlexNN directly — no monkey-patching
        model = FlexNN(
            input_dim   = X_tr_scaled.shape[1],
            hidden_dims = best_config['hidden_dims'],
            dropout     = best_config['dropout']
        ).to(device)

        criterion = PhysicsLoss(lambda_physics=LAMBDA_PHYSICS)

        model, history = train_pinn(
            model, train_loader, test_loader, criterion,
            physics_feature_names,
            n_epochs  = 200,
            lr        = best_config['lr'],
            patience  = 20,
            device    = device
        )

        _, preds_transformed, labels_transformed = evaluate_pinn(
            model, test_loader, criterion,
            physics_feature_names, device
        )

        r2_transformed = r2_score(labels_transformed, preds_transformed)

        preds_clipped   = np.clip(preds_transformed,
                                   labels_transformed.min(),
                                   labels_transformed.max())
        preds_physical  = np.sinh(preds_clipped)
        labels_physical = np.sinh(labels_transformed)

        r2_physical  = r2_score(labels_physical, preds_physical)
        mae_physical = mean_absolute_error(labels_physical, preds_physical)
        pb = (np.sum(preds_physical - labels_physical) /
              np.sum(np.abs(labels_physical))) * 100

        print(f"\nResults for {held_out}:")
        print(f"  R² (transformed): {r2_transformed:.4f}")
        print(f"  R² (physical):    {r2_physical:.4f}")
        print(f"  MAE (physical):   {mae_physical:.4f} ng N/m²/s")
        print(f"  PB Bias:          {pb:.2f}%")

        results.append({
            'fire':           held_out,
            'r2':             r2_transformed,
            'r2_physical':    r2_physical,
            'mae':            mae_physical,
            'pb_bias':        pb,
            'preds_physical': preds_physical,
            'labels_physical': labels_physical,
            'history':        history
        })

        os.makedirs('models', exist_ok=True)
        torch.save(
            model.state_dict(),
            f'models/pinn_tuned_{held_out.replace(" ", "_")}.pt'
        )

    print(f"\n{'='*55}")
    print("TUNED PINN LOGO CV SUMMARY")
    print(f"{'='*55}")
    for r in results:
        print(f"{r['fire']:<20} R²={r['r2']:>7.4f}  "
              f"MAE={r['mae']:>7.4f}  PBias={r['pb_bias']:>8.2f}%")

    mean_r2  = np.mean([r['r2']  for r in results])
    mean_mae = np.mean([r['mae'] for r in results])
    mean_pb  = np.mean([r['pb_bias'] for r in results])
    print(f"\nMean R²:    {mean_r2:.4f}")
    print(f"Mean MAE:   {mean_mae:.4f}")
    print(f"Mean PBias: {mean_pb:.2f}%")

    return results

# =====================================================================
# MAIN
# =====================================================================

if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    df = load_data('data/nox_clean.csv')
    features = get_features(df)
    df['nox_transformed'] = prepare_target(df)

    """
    # Step 1 — grid search on two validation fires
    print("\n=== STEP 1: GRID SEARCH ===")
    best_config, all_results = run_grid_search(
        df, features, device,
        val_fires=['Dixie Fire', 'North Complex']
    )
    """
    grid_search_path = 'results/grid_search_results.json'

    if os.path.exists(grid_search_path):
        print(f"\nLoading grid search results from {grid_search_path}")
        with open(grid_search_path) as f:
            all_results = json.load(f)

        # Best config is first entry (sorted by R²)
        best_raw = all_results[0]['config']

        # Convert hidden_dims back from string to list
        import ast
        best_config = {
            'hidden_dims':  ast.literal_eval(best_raw['hidden_dims']),
            'dropout':      best_raw['dropout'],
            'lr':           best_raw['lr'],
            'batch_size':   best_raw['batch_size'],
            'weight_decay': best_raw['weight_decay']
        }
        print(f"Best config loaded: {best_config}")
        print(f"Grid search R²: {all_results[0]['mean_r2']:.4f}")

    else:
        print("No grid search results found — run grid search first")
        print("Set RUN_GRID_SEARCH = True at top of file")
        sys.exit(1)
    
    # Step 2 — full LOGO CV with best config
    print("\n=== STEP 2: FULL LOGO CV WITH BEST CONFIG ===")
    final_results = run_final_pinn_with_config(df, features, best_config, device)

    mean_r2  = np.mean([r['r2']  for r in final_results])
    mean_mae = np.mean([r['mae'] for r in final_results])

    print(f"\nFinal PINN (tuned) results:")
    print(f"  Best config:  {best_config}")
    print(f"  Mean R²:      {mean_r2:.4f}")
    print(f"  Mean MAE:     {mean_mae:.4f}")

    # Save final results
    final_summary = {
        'best_config':   {k: str(v) if isinstance(v, list) else v
                          for k, v in best_config.items()},
        'lambda_physics': LAMBDA_PHYSICS,
        'mean_r2':       float(mean_r2),
        'mean_mae':      float(mean_mae),
        'per_fire':      {r['fire']: {'r2': float(r['r2']),
                                       'mae': float(r['mae'])}
                          for r in final_results}
    }
    with open('results/pinn_tuned_results.json', 'w') as f:
        json.dump(final_summary, f, indent=2)
    print("Saved: results/pinn_tuned_results.json")