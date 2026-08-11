import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import (
    load_data, get_features, prepare_target, get_logo_splits, make_dataloaders
)

#Baseline Feedforward NN
class BaselineNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256,128,64], dropout=0.3):
        super(BaselineNN, self).__init__()

        layers = []
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

#Training Time - Loop
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        preds = model(batch_X)
        loss = criterion(preds, batch_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            total_loss += loss.item()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    return total_loss / len(loader), all_preds, all_labels

#Full training

def train_model(model, train_loader, test_loader, n_epochs=200, lr=1e-3, patience=20, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _, _ = evaluate(model, test_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter +=1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:>3} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_weights:
        model.load_state_dict(best_weights)

    return model, history

#LOGO CV
def run_logo_cv(df, features, device='cpu'):
    results = []

    for X_tr, X_te, y_tr, y_te, fire, scaler in get_logo_splits(df, features):
        print(f"\n{'='*55}")
        print(f"Held-out fire: {fire}")
        print(f"Train: {X_tr.shape} | Test: {X_te.shape}")
        print(f"{'='*55}")

        print(f"X_tr: {X_tr.shape}, y_tr: {y_tr.shape}")
        print(f"X_te: {X_te.shape}, y_te: {y_te.shape}")

        train_loader, test_loader = make_dataloaders(
            X_tr, y_tr, X_te, y_te, batch_size=512
        )

        model = BaselineNN(input_dim=X_tr.shape[1]).to(device)

        model, history = train_model(
            model, train_loader, test_loader, n_epochs=200, lr=1e-3, patience=20, device=device
        )

        criterion = nn.MSELoss()
        _, preds_transformed, labels_transformed = evaluate(
            model, test_loader, criterion, device
        )

        preds_physical = np.sinh(preds_transformed)
        labels_physical = np.sinh(labels_transformed)

        r2 = r2_score(labels_physical, preds_physical)
        mae = mean_absolute_error(labels_physical, preds_physical)
        pb = (np.sum(preds_physical-labels_physical) / np.sum(np.abs(labels_physical))) * 100

        print(f"\nResults for {fire}:")
        print(f"  R²:        {r2:.4f}")
        print(f"  MAE:       {mae:.4f} ng N/m²/s")
        print(f"  PB Bias:   {pb:.2f}%")

        results.append({
            'fire':             fire,
            'r2':               r2,
            'mae':              mae,
            'pb_bias':          pb,
            'preds_physical':   preds_physical,
            'labels_physical':  labels_physical,
            'history':          history
        })

        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), f'models/baseline_nn_{fire.replace(" ","_")}.pt')

    print(f"\n{'='*55}")
    print("LOGO CV SUMMARY")
    print(f"{'='*55}")
    r2s  = [r['r2']  for r in results]
    maes = [r['mae'] for r in results]
    pbs  = [r['pb_bias'] for r in results]

    for r in results:
        print(f"{r['fire']:<20} R²={r['r2']:>7.4f}  "
              f"MAE={r['mae']:>7.4f}  PBias={r['pb_bias']:>8.2f}%")

    print(f"\nMean R²:     {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
    print(f"Mean MAE:    {np.mean(maes):.4f} ± {np.std(maes):.4f}")
    print(f"Mean PBias:  {np.mean(pbs):.2f}% ± {np.std(pbs):.2f}%")

    return results

#Plot training history
def plot_training_history(results):
    fig, axes = plt.subplots(2,2, figsize=(14,10))
    axes = axes.flatten()

    for i, r in enumerate(results):
        ax = axes[i]
        ax.plot(r['history']['train_loss'], label='Train loss', color='steelblue')
        ax.plot(r['history']['val_loss'], label='Val loss', color='darkorange')
        ax.set_title(f"{r['fire']} — R²={r['r2']:.3f}")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Baseline NN Training History — Leave-One-Fire-Out',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/baseline_nn_training_history.png', dpi=150)
    print("\nSaved: results/baseline_nn_training_history.png")
    plt.show()

if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    df = load_data('data/nox_clean.csv')
    features = get_features(df)
    df['nox_transformed'] = prepare_target(df)

    results = run_logo_cv(df, features, device=device)

    plot_training_history(results)