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

if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    df = load_data('data/nox_clean.csv')
    features = get_features(df)
    df['nox_transformed'] = prepare_target(df)
    
    FINAL_CONFIG = {
        'hidden_dims':  [256, 128, 64],
        'dropout':      0.3,
        'lr':           0.001,
        'batch_size':   512,
        'weight_decay': 0.0001
    }

    # Run 3 seeds for stability
    import torch
    import numpy as np

    all_r2s  = []
    all_maes = []

    for seed in [42, 123, 7]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"\n=== Seed {seed} ===")

        results = run_final_pinn_with_config(
            df, features, FINAL_CONFIG, device
        )
        r2  = np.mean([r['r2']  for r in results])
        mae = np.mean([r['mae'] for r in results])
        all_r2s.append(r2)
        all_maes.append(mae)
        print(f"Seed {seed}: R²={r2:.4f}, MAE={mae:.4f}")

    print(f"\n{'='*50}")
    print("FINAL PINN RESULTS (3-seed average)")
    print(f"{'='*50}")
    print(f"Mean R²:  {np.mean(all_r2s):.4f} ± {np.std(all_r2s):.4f}")
    print(f"Mean MAE: {np.mean(all_maes):.4f} ± {np.std(all_maes):.4f}")

    final = {
        'mean_r2':  float(np.mean(all_r2s)),
        'std_r2':   float(np.std(all_r2s)),
        'mean_mae': float(np.mean(all_maes)),
        'std_mae':  float(np.std(all_maes)),
        'config':   FINAL_CONFIG,
        'lambda_physics': 0.01,
        'seeds': [42, 123, 7]
    }
    with open('results/final_pinn_results.json', 'w') as f:
        json.dump(final, f, indent=2)
    print("Saved: results/final_pinn_results.json")