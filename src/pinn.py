import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import (
    load_data, get_features, prepare_target,
    get_logo_splits, make_dataloaders
)
from src.physics_loss import PhysicsLoss
from src.baseline_nn import BaselineNN, evaluate

# =====================================================================
# PINN — Physics Informed Neural Network
# Same architecture as baseline but trained with PhysicsLoss
# =====================================================================

class PINNDataset(torch.utils.data.Dataset):
    """
    Extended dataset that also returns raw features needed
    by the physics loss (no2, ref_no2, etc.) separately
    from the scaled features used by the network.
    """
    def __init__(self, X_scaled, y, X_raw, raw_feature_names):
        X_scaled = np.array(X_scaled)
        y        = np.array(y)
        X_raw    = np.array(X_raw)

        assert len(X_scaled) == len(y) == len(X_raw)

        self.X_scaled = torch.tensor(X_scaled, dtype=torch.float32)
        self.y        = torch.tensor(y,        dtype=torch.float32).unsqueeze(1)
        self.X_raw    = torch.tensor(X_raw,    dtype=torch.float32)
        self.raw_feature_names = raw_feature_names

    def __len__(self):
        return len(self.X_scaled)

    def __getitem__(self, idx):
        return self.X_scaled[idx], self.y[idx], self.X_raw[idx]


def make_pinn_dataloaders(X_scaled_tr, y_tr, X_raw_tr,
                           X_scaled_te, y_te, X_raw_te,
                           batch_size=256):
    train_dataset = PINNDataset(X_scaled_tr, y_tr, X_raw_tr,
                                 raw_feature_names=[])
    test_dataset  = PINNDataset(X_scaled_te, y_te, X_raw_te,
                                 raw_feature_names=[])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


# =====================================================================
# PINN TRAINING LOOP
# =====================================================================

def train_pinn_epoch(model, loader, optimizer, criterion,
                      feature_names, device):
    model.train()
    total_loss   = 0
    total_data   = 0
    total_physics = 0
    loss_comp_sum = {}

    for batch_X, batch_y, batch_raw in loader:
        batch_X   = batch_X.to(device)
        batch_y   = batch_y.to(device)
        batch_raw = batch_raw.to(device)

        optimizer.zero_grad()
        preds = model(batch_X)

        total, data, physics, components = criterion(
            preds, batch_y, batch_raw, feature_names
        )

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss    += total.item()
        total_data    += data.item()
        total_physics += physics.item()

        for k, v in components.items():
            loss_comp_sum[k] = loss_comp_sum.get(k, 0) + v

    n = len(loader)
    avg_components = {k: v / n for k, v in loss_comp_sum.items()}

    return total_loss / n, total_data / n, total_physics / n, avg_components


def evaluate_pinn(model, loader, criterion, feature_names, device, data_only=False):
    model.eval()
    total_loss = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y, batch_raw in loader:
            batch_X   = batch_X.to(device)
            batch_y   = batch_y.to(device)
            batch_raw = batch_raw.to(device)

            preds = model(batch_X)
            if data_only:
                total = torch.nn.functional.mse_loss(preds, batch_y, reduction="sum")
            else:
                total, _, _, _ = criterion(preds, batch_y, batch_raw, feature_names)

            total_loss += total.item()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())

    all_preds  = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    denominator = len(all_labels) if data_only else len(loader)
    return total_loss / denominator, all_preds, all_labels


def train_pinn(model, train_loader, test_loader, criterion,
               feature_names, n_epochs=200, lr=1e-3,
               patience=20, device='cpu', data_only_validation=False):

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    history = {
        'train_total': [], 'train_data': [], 'train_physics': [],
        'val_loss': [], 'loss_components': []
    }

    best_val_loss    = float('inf')
    best_weights     = None
    patience_counter = 0

    for epoch in range(n_epochs):
        train_total, train_data, train_physics, components = train_pinn_epoch(
            model, train_loader, optimizer, criterion, feature_names, device
        )
        val_loss, _, _ = evaluate_pinn(
            model, test_loader, criterion, feature_names, device,
            data_only=data_only_validation
        )

        history['train_total'].append(train_total)
        history['train_data'].append(train_data)
        history['train_physics'].append(train_physics)
        history['val_loss'].append(val_loss)
        history['loss_components'].append(components)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_weights     = {k: v.clone()
                                 for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:>3} | "
                  f"Total: {train_total:.4f} | "
                  f"Data: {train_data:.4f} | "
                  f"Physics: {train_physics:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_weights:
        model.load_state_dict(best_weights)

    return model, history


# =====================================================================
# LOGO CV WITH PINN
# =====================================================================

def run_pinn_logo_cv(df, features, lambda_physics=0.1, device='cpu'):
    """
    Leave-one-fire-out CV with physics-informed loss.
    Also returns raw features for physics loss computation.
    """
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler

    # Raw feature names needed by physics loss
    # These are passed unscaled to physics terms
    physics_feature_names = [
        'no2', 'ref_no2', 'wind_speed', 'wind_u', 'wind_v',
        'days_post_fire', 'centroid_lat', 'centroid_lon',
        'day_of_year_sin_1', 'day_of_year_cos_1'
    ]
    # Use all available physics features that exist in df
    physics_feature_names = [
        f for f in physics_feature_names if f in df.columns
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
        print(f"PINN — Held-out fire: {held_out}")
        print(f"{'='*55}")

        # Scaled features for NN input
        X_tr_raw = X.iloc[train_idx]
        X_te_raw = X.iloc[test_idx]
        y_tr     = y[train_idx]
        y_te     = y[test_idx]

        train_median = X_tr_raw.median()
        X_tr_raw     = X_tr_raw.fillna(train_median)
        X_te_raw     = X_te_raw.fillna(train_median)

        scaler       = StandardScaler()
        X_tr_scaled  = scaler.fit_transform(X_tr_raw.values)
        X_te_scaled  = scaler.transform(X_te_raw.values)

        # Raw features for physics loss (unscaled)
        X_raw_tr = X_raw.iloc[train_idx].fillna(
            X_raw.iloc[train_idx].median()
        ).values
        X_raw_te = X_raw.iloc[test_idx].fillna(
            X_raw.iloc[train_idx].median()
        ).values

        # Clip target to training range
        y_tr_min, y_tr_max = y_tr.min(), y_tr.max()
        y_te = np.clip(y_te, y_tr_min, y_tr_max)

        train_loader, test_loader = make_pinn_dataloaders(
            X_tr_scaled, y_tr, X_raw_tr,
            X_te_scaled,  y_te, X_raw_te,
            batch_size=512
        )

        model     = BaselineNN(input_dim=X_tr_scaled.shape[1]).to(device)
        criterion = PhysicsLoss(lambda_physics=lambda_physics)

        model, history = train_pinn(
            model, train_loader, test_loader, criterion,
            physics_feature_names,
            n_epochs=200, lr=1e-3, patience=20, device=device
        )

        val_loss, preds_transformed, labels_transformed = evaluate_pinn(
            model, test_loader, criterion, physics_feature_names, device
        )

        # Evaluate in transformed space
        r2_transformed = r2_score(labels_transformed, preds_transformed)

        # Evaluate in physical space with clipping
        preds_clipped  = np.clip(preds_transformed, y_te.min(), y_te.max())
        preds_physical = np.sinh(preds_clipped)
        labels_physical = np.sinh(labels_transformed)

        r2_physical  = r2_score(labels_physical, preds_physical)
        mae_physical = mean_absolute_error(labels_physical, preds_physical)
        pb = (np.sum(preds_physical - labels_physical) /
              np.sum(np.abs(labels_physical))) * 100

        print(f"\nPINN Results for {held_out}:")
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
            f'models/pinn_{held_out.replace(" ", "_")}.pt'
        )

    # Summary
    print(f"\n{'='*55}")
    print("PINN LOGO CV SUMMARY")
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
# PLOT PINN vs BASELINE COMPARISON
# =====================================================================

def plot_pinn_vs_baseline(pinn_results, baseline_results_path):
    with open(baseline_results_path) as f:
        baseline = json.load(f)

    fire_order = ['August Complex', 'Camp Fire', 'Dixie Fire', 'North Complex']
    fire_colors = {
        'August Complex': '#d62728',
        'Camp Fire':      '#2ca02c',
        'Dixie Fire':     '#ff7f0e',
        'North Complex':  '#1f77b4'
    }

    pinn_r2  = {r['fire']: r['r2']  for r in pinn_results}
    pinn_mae = {r['fire']: r['mae'] for r in pinn_results}
    base_r2  = {f: baseline['per_fire'][f]['r2_transformed']
                for f in fire_order}
    base_mae = {f: baseline['per_fire'][f]['mae_physical']
                for f in fire_order}

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x     = np.arange(len(fire_order))
    width = 0.35

    # R² comparison
    axes[0].bar(x - width/2,
                [base_r2[f]  for f in fire_order], width,
                label='Baseline NN', color='steelblue',
                alpha=0.8, edgecolor='black')
    axes[0].bar(x + width/2,
                [pinn_r2[f]  for f in fire_order], width,
                label='PINN', color='darkorange',
                alpha=0.8, edgecolor='black')
    axes[0].axhline(y=0, color='black', linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f.replace(' ', '\n') for f in fire_order])
    axes[0].set_ylabel('R² (transformed space)')
    axes[0].set_title('R² — Baseline NN vs PINN\nLeave-One-Fire-Out')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # MAE comparison
    axes[1].bar(x - width/2,
                [base_mae[f] for f in fire_order], width,
                label='Baseline NN', color='steelblue',
                alpha=0.8, edgecolor='black')
    axes[1].bar(x + width/2,
                [pinn_mae[f] for f in fire_order], width,
                label='PINN', color='darkorange',
                alpha=0.8, edgecolor='black')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f.replace(' ', '\n') for f in fire_order])
    axes[1].set_ylabel('MAE (ng N/m²/s)')
    axes[1].set_title('MAE — Baseline NN vs PINN\nLeave-One-Fire-Out')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Physics-Informed Neural Network vs Baseline\n'
                 'Leave-One-Fire-Out Cross Validation',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/pinn_vs_baseline.png', dpi=150, bbox_inches='tight')
    print("Saved: results/pinn_vs_baseline.png")
    plt.show()


# =====================================================================
# MAIN
# =====================================================================

if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    df = load_data('data/nox_clean.csv')
    features = get_features(df)
    df['nox_transformed'] = prepare_target(df)

    """
    stability_results = {}

    for lambda_val in [0.005, 0.05, 0.01]:
        r2s  = []
        maes = []

        for seed in [42, 123, 7]:
            torch.manual_seed(seed)
            np.random.seed(seed)

            results = run_pinn_logo_cv(df, features,
                                    lambda_physics=lambda_val,
                                    device=device)
            r2s.append(float(np.mean([r['r2']  for r in results])))
            maes.append(float(np.mean([r['mae'] for r in results])))
            print(f"lambda={lambda_val}, seed={seed} → "
                f"R²={r2s[-1]:.4f}, MAE={maes[-1]:.4f}")

        stability_results[str(lambda_val)] = {
            'mean_r2':  float(np.mean(r2s)),
            'std_r2':   float(np.std(r2s)),
            'mean_mae': float(np.mean(maes)),
            'std_mae':  float(np.std(maes))
        }

    print("\n=== STABILITY TEST ===")
    for lam, res in stability_results.items():
        print(f"lambda={lam}: "
            f"R²={res['mean_r2']:.4f}±{res['std_r2']:.4f}  "
            f"MAE={res['mean_mae']:.4f}±{res['std_mae']:.4f}")

    BEST_LAMBDA = 0.01

    with open('results/lambda_stability.json', 'w') as f:
        json.dump({
            'best_lambda': BEST_LAMBDA,
            'stability_results': stability_results,
            'selection_reason': (
                'lambda=0.01 selected for lowest std (0.007 vs 0.086-0.167), '
                'positive mean R², and lowest MAE among stable configs'
            )
        }, f, indent=2)

    print(f"Best lambda locked: {BEST_LAMBDA}")
    """

    # Run PINN LOGO CV

    pinn_results = run_pinn_logo_cv(df, features,
        lambda_physics=0.01,
        device=device)


    """

    sweep_results = {}

    for lambda_val in [0.001, 0.005, 0.01, 0.05, 0.1]:
        print(f"\n{'='*55}")
        print(f"Testing lambda_physics = {lambda_val}")
        results = run_pinn_logo_cv(df, features,
                                lambda_physics=lambda_val,
                                device=device)
        mean_r2  = float(np.mean([r['r2']  for r in results]))
        mean_mae = float(np.mean([r['mae'] for r in results]))
        mean_pb  = float(np.mean([r['pb_bias'] for r in results]))

        sweep_results[str(lambda_val)] = {
            'mean_r2':  mean_r2,
            'mean_mae': mean_mae,
            'mean_pb':  mean_pb
        }
        print(f"lambda={lambda_val} → R²={mean_r2:.4f}, MAE={mean_mae:.4f}")

    # Save and print summary
    with open('results/lambda_sweep.json', 'w') as f:
        json.dump(sweep_results, f, indent=2)

    print("\n=== LAMBDA SWEEP SUMMARY ===")
    print(f"{'Lambda':<10} {'Mean R²':>10} {'Mean MAE':>10}")
    print("-" * 32)
    for lam, res in sweep_results.items():
        print(f"{lam:<10} {res['mean_r2']:>10.4f} {res['mean_mae']:>10.4f}")
    # Compare against baseline

    """

    plot_pinn_vs_baseline(
        pinn_results,
        'results/baseline_nn_results.json'
    )

    # Save PINN results
    pinn_summary = {
        'lambda_physics': 0.01,
        'mean_r2_transformed': float(np.mean([r['r2']  for r in pinn_results])),
        'mean_mae_physical':   float(np.mean([r['mae'] for r in pinn_results])),
        'mean_pbias':          float(np.mean([r['pb_bias'] for r in pinn_results])),
        'per_fire': {
            r['fire']: {
                'r2_transformed': float(r['r2']),
                'r2_physical':    float(r['r2_physical']),
                'mae_physical':   float(r['mae']),
                'pbias':          float(r['pb_bias'])
            }
            for r in pinn_results
        }
    }
    with open(f'results/pinn_results.json', 'w') as f:
        json.dump(pinn_summary, f, indent=2)
    print(f"PINN results saved to results/pinn_results.json")