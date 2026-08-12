import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
import torch
from torch.utils.data import Dataset, DataLoader

#Load data

def load_data(file):
    df = pd.read_csv(file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['fire_name','plot_id','date']).reset_index(drop=True)

    print(f"Loaded {len(df)} rows across {df['fire_name'].nunique()} fires")
    print(f"Fires: {df['fire_name'].unique().tolist()}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Columns: {df.columns.tolist()}")

    return df

#Define features

EXCLUDE_COLS = [ #not to be included as features to avoid leakage
    # targets
    'soil_nox_tau3hr', 'soil_nox_tau5hr', 'soil_nox_tau7hr',
    'nox_transformed', 'no2', 'ref_no2',
    # box model intermediates derived from no2
    'advection_term', 'deposition_term', 'chemical_loss_term',
    # identifiers
    'fire_name', 'date', 'plot_id', 'month_key',
    'system:index', '.geo',
    # redundant
    'day_of_year', 'month', 'year', 'southernly_wind', 'westernly_wind', 'urban_baseline',
    'u_component_of_wind_10m','v_component_of_wind_10m','wind_angle'
]

def get_features(df):
    features = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in EXCLUDE_COLS
    ]
    print(f"\nFeatures ({len(features)}):")
    print(features)
    return features

#Prepare target variable

def prepare_target(df, lifetime='tau5hr'):
    col = f'soil_nox_{lifetime}'
    raw = df[col].copy()

    p1 = raw.quantile(0.01)
    p99 = raw.quantile(0.99)
    raw = raw.clip(lower=p1, upper=p99)

    transformed = np.arcsinh(raw)

    print(f"\nTarget ({col}):")
    print(f"  Raw mean:         {df[col].mean():.4f}")
    print(f"  Raw std:          {df[col].std():.4f}")
    print(f"  Transformed mean: {transformed.mean():.4f}")
    print(f"  Transformed std:  {transformed.std():.4f}")
    print(f"  Skewness before:  {df[col].skew():.4f}")
    print(f"  Skewness after:   {transformed.skew():.4f}")

    return transformed

#Scale features

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

#PYTorch dataset

class NOxDataset(Dataset):

    def __init__(self, X, y):
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y

        assert len(X) == len(y), \
            f"X length {len(X)} != y length {len(y)}"
        
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def make_dataloaders(X_train, y_train, X_test, y_test, batch_size=256, shuffle_train=True):
    train_dataset = NOxDataset(X_train, y_train)
    test_dataset = NOxDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

#Leave One Fire Out split

def get_logo_splits(df, features, target_col='nox_transformed'):
    logo = LeaveOneGroupOut()
    groups=df["fire_name"]
    X = df[features].replace([np.inf, -np.inf], np.nan)
    y = df[target_col].values

    for train_idx, test_idx in logo.split(X,y,groups=groups):
        held_out = groups.iloc[test_idx].iloc[0]

        X_tr_raw = X.iloc[train_idx]
        X_te_raw = X.iloc[test_idx]
        y_tr = y[train_idx]
        y_te = y[test_idx]

        y_tr_min, y_tr_max = y_tr.min(), y_tr.max()
        y_te = np.clip(y_te, y_tr_min, y_tr_max)

        train_median = X_tr_raw.median()
        X_tr_raw = X_tr_raw.fillna(train_median)
        X_te_raw = X_te_raw.fillna(train_median)

        X_tr_scaled, X_te_scaled, scaler = scale_features(X_tr_raw.values, X_te_raw.values)

        assert len(X_tr_scaled) == len(y_tr), \
            f"Train size mismatch: {len(X_tr_scaled)} vs {len(y_tr)}"
        assert len(X_te_scaled) == len(y_te), \
            f"Test size mismatch: {len(X_te_scaled)} vs {len(y_te)}"
        
        yield X_tr_scaled, X_te_scaled, y_tr, y_te, held_out, scaler

#Test

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/data_loader.py <path_to_csv>")
        sys.exit(1)

    filepath = sys.argv[1]
    df = load_data(filepath)

    features = get_features(df)
    df['nox_transformed'] = prepare_target(df)

    print("\nTesting LOGO splits:")
    for X_tr, X_te, y_tr, y_te, fire, scaler in get_logo_splits(
        df, features
    ):
        print(f"  Held out: {fire:<20} "
              f"train={X_tr.shape} test={X_te.shape}")

    print("\nTesting DataLoader:")
    X_tr, X_te, y_tr, y_te, fire, scaler = next(
        get_logo_splits(df, features)
    )
    train_loader, test_loader = make_dataloaders(X_tr, y_tr, X_te, y_te)
    batch_X, batch_y = next(iter(train_loader))
    print(f"  Batch X shape: {batch_X.shape}")
    print(f"  Batch y shape: {batch_y.shape}")
    print("\nData pipeline ready.")