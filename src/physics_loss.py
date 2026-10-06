import torch
import torch.nn as nn
import numpy as np

# =====================================================================
# PHYSICS-INFORMED LOSS FUNCTION FOR NOx EMISSIONS
#
# Based on the Huber et al. (2020) box model:
#
# E_soil = (U * delta_NO2 / L) + (Vd * NO2 / Zpbl) + (NO2 / tau) - E_NEI
#
# The physics loss penalizes predictions that violate:
#   1. NOx mass balance (box model residual)
#   2. Wind advection consistency
#   3. Dry deposition constraints
#   4. Temporal decay after fire
#   5. Spatial diffusion smoothness
#   6. Non-negativity of emissions
#   7. Seasonal deposition velocity
# =====================================================================


# =====================================================================
# CONSTANTS
# =====================================================================

Z_PBL    = 1000.0          # boundary layer height (m)
L_FETCH  = 5000.0          # fetch distance (m)

# Seasonal deposition velocity (m/s) — from Yang et al. (2010)
# Keys are month ranges: (start_month, end_month) inclusive
VD_SEASONAL = {
    'winter': (0.0002, [12, 1,  2]),   # Dec-Feb
    'spring': (0.0016, [3,  4,  5]),   # Mar-May
    'summer': (0.0029, [6,  7,  8]),   # Jun-Aug
    'fall':   (0.0006, [9,  10, 11]),  # Sep-Nov
}

# NOx lifetime — sinusoidal, min on June 21 (day 172), max on Dec 21 (day 355)
# Three scenarios: 3hr, 5hr, 7hr minimum
TAU_MAX_HR  = 15.0
TAU_PEAK_DAY = 355


def get_vd(month_tensor):
    """
    Returns seasonal deposition velocity for each sample.
    month_tensor: (N,) int tensor of month values (1-12)
    Returns: (N,) float tensor of Vd values
    """
    vd = torch.full_like(month_tensor, 0.0002, dtype=torch.float32)
    vd = torch.where((month_tensor >= 3)  & (month_tensor <= 5),
                     torch.full_like(vd, 0.0016), vd)
    vd = torch.where((month_tensor >= 6)  & (month_tensor <= 8),
                     torch.full_like(vd, 0.0029), vd)
    vd = torch.where((month_tensor >= 9)  & (month_tensor <= 11),
                     torch.full_like(vd, 0.0006), vd)
    return vd


def get_tau(day_of_year_tensor, min_lifetime_hr=5.0):
    """
    Sinusoidal NOx lifetime in seconds.
    Minimum on June 21 (day 172), maximum on Dec 21 (day 355).
    """
    min_s  = min_lifetime_hr * 3600.0
    max_s  = TAU_MAX_HR      * 3600.0
    amp    = (max_s - min_s) / 2.0
    mean   = (max_s + min_s) / 2.0
    tau    = mean + amp * torch.cos(
        2 * np.pi * (day_of_year_tensor - TAU_PEAK_DAY) / 365.25
    )
    return tau


# =====================================================================
# PHYSICS LOSS TERMS
# =====================================================================

def box_model_residual(pred_nox, no2_vcd, no2_upwind, wind_speed,
                        day_of_year, month, min_lifetime_hr=5.0):
    """
    Compute box model residual entirely in arcsinh-transformed space.
    Instead of converting to physical units, we normalize both sides
    to have unit variance to avoid numerical explosion.
    """
    pred_physical = torch.sinh(pred_nox)

    AVOGADRO     = 6.022e23
    N_MOLAR_MASS = 14.0067

    delta_no2       = no2_vcd - no2_upwind
    vd              = get_vd(month)
    tau             = get_tau(day_of_year.float(), min_lifetime_hr)

    advection_term  = (wind_speed * delta_no2) / L_FETCH
    deposition_term = (vd * no2_vcd) / Z_PBL
    chemical_term   = no2_vcd / tau

    box_physical = (advection_term + deposition_term + chemical_term) \
                   * (N_MOLAR_MASS / AVOGADRO) * 1e9

    # Normalize both to zero mean unit variance before comparing
    pred_norm = (pred_physical - pred_physical.mean()) / \
                (pred_physical.std() + 1e-8)
    box_norm  = (box_physical  - box_physical.mean())  / \
                (box_physical.std()  + 1e-8)

    return ((pred_norm - box_norm) ** 2).mean()


def non_negativity_loss(pred_nox):
    """
    Term 2: Penalize negative NOx predictions.
    Soil emissions physically cannot be negative
    (or at most very slightly negative for uptake).
    """
    pred_physical = torch.sinh(pred_nox)
    violations    = torch.clamp(-pred_physical, min=0.0)
    return (violations ** 2).mean()


def temporal_decay_loss(pred_nox, days_post_fire):
    """
    Term 3: NOx should decay exponentially after fire.
    Within a batch, predictions for later days should generally
    be lower than predictions for earlier days.

    We penalize cases where pred(day_late) > pred(day_early)
    by more than a physically reasonable amount.

    Decay timescale ~30 days based on literature.
    """
    pred_physical  = torch.sinh(pred_nox).squeeze()
    days           = days_post_fire.squeeze().float()

    # Expected decay: E(t) = E(0) * exp(-t/30)
    # Normalized relative to day 0
    decay_rate     = 1.0 / 30.0
    expected_decay = torch.exp(-decay_rate * days)

    # Penalize predictions that are too HIGH relative to expected decay
    # i.e. model thinks NOx is still very high long after fire
    pred_normalized = pred_physical / (pred_physical.mean() + 1e-8)
    violation       = torch.clamp(pred_normalized - expected_decay * 3, min=0.0)

    return (violation ** 2).mean()


def deposition_consistency_loss(pred_nox, no2_vcd, month):
    """
    Normalized deposition consistency.
    Penalize predictions much larger than deposition flux,
    but in normalized space to avoid unit explosion.
    """
    pred_physical = torch.sinh(pred_nox)

    AVOGADRO     = 6.022e23
    N_MOLAR_MASS = 14.0067

    vd              = get_vd(month)
    deposition_flux = (vd * no2_vcd / Z_PBL) \
                      * (N_MOLAR_MASS / AVOGADRO) * 1e9

    # Normalize
    pred_norm = pred_physical / (pred_physical.abs().mean() + 1e-8)
    dep_norm  = deposition_flux / (deposition_flux.abs().mean() + 1e-8)

    # Penalize if pred is more than 5x the deposition flux (in normalized space)
    violation = torch.clamp(pred_norm - 5.0 * dep_norm, min=0.0)
    return (violation ** 2).mean()


def spatial_smoothness_loss(pred_nox, centroid_lat, centroid_lon):
    """
    Term 5: Spatial diffusion / smoothness.
    Adjacent plots should have similar NOx values —
    NOx diffuses spatially and sharp discontinuities
    between neighboring plots are physically unrealistic.

    Penalizes large differences between spatially close predictions.
    """
    pred_physical = torch.sinh(pred_nox).squeeze()

    # Compute pairwise distances between plots in the batch
    lat_scale = 111320.0  # meters per degree lat
    lon_scale = 111320.0  # approximate meters per degree lon

    dlat = (centroid_lat.unsqueeze(1) - centroid_lat.unsqueeze(0)) * lat_scale
    dlon = (centroid_lon.unsqueeze(1) - centroid_lon.unsqueeze(0)) * lon_scale
    dist = torch.sqrt(dlat ** 2 + dlon ** 2 + 1e-8)  # meters

    # Only consider nearby plots (within 20km)
    nearby_mask = (dist < 20000.0) & (dist > 0.0)

    if nearby_mask.sum() == 0:
        return torch.tensor(0.0)

    # Penalize large NOx differences between nearby plots
    pred_diff   = (pred_physical.unsqueeze(1) - pred_physical.unsqueeze(0)) ** 2
    spatial_pen = (pred_diff * nearby_mask.float()).sum() / (nearby_mask.sum() + 1e-8)

    return spatial_pen


def wind_advection_consistency_loss(pred_nox, wind_u, wind_v,
                                     centroid_lat, centroid_lon):
    """
    Term 6: Wind advection direction consistency.
    NOx is transported downwind — plots that are downwind
    should have higher NOx than upwind plots.

    Penalizes cases where a downwind plot has much lower
    NOx than an upwind plot.
    """
    pred_physical = torch.sinh(pred_nox).squeeze()
    wind_speed    = torch.sqrt(wind_u ** 2 + wind_v ** 2 + 1e-8)

    # Unit wind vector (direction of transport)
    unit_u = wind_u / (wind_speed + 1e-8)
    unit_v = wind_v / (wind_speed + 1e-8)

    # Vector from each plot to every other plot
    dlat = (centroid_lat.unsqueeze(1) - centroid_lat.unsqueeze(0)) * 111320.0
    dlon = (centroid_lon.unsqueeze(1) - centroid_lon.unsqueeze(0)) * 111320.0

    # Dot product: positive = downwind, negative = upwind
    downwind_score = unit_u.unsqueeze(1) * dlon + unit_v.unsqueeze(1) * dlat

    # Only consider nearby plots
    dist        = torch.sqrt(dlat ** 2 + dlon ** 2 + 1e-8)
    nearby_mask = (dist < 20000.0) & (dist > 0.0)

    # Downwind plots should have >= upwind plots (allowing some tolerance)
    pred_i      = pred_physical.unsqueeze(1).expand_as(downwind_score)
    pred_j      = pred_physical.unsqueeze(0).expand_as(downwind_score)

    # If j is downwind of i (score > 0), pred_j should not be much less than pred_i
    downwind_mask = (downwind_score > 0) & nearby_mask
    violation     = torch.clamp(pred_i - pred_j - 5.0, min=0.0)
    penalty       = (violation * downwind_mask.float()).sum() / \
                    (downwind_mask.float().sum() + 1e-8)

    return penalty


# =====================================================================
# COMBINED PHYSICS LOSS
# =====================================================================

class PhysicsLoss(nn.Module):
    """
    Combined physics-informed loss for NOx prediction.

    Total loss = data_loss + lambda_physics * physics_loss

    physics_loss is a weighted sum of individual physical constraints.
    """

    def __init__(self,
                 lambda_physics=0.1,
                 lambda_box=1.0,
                 lambda_nonneg=2.0,
                 lambda_decay=0.1,
                 lambda_deposition=0.5,
                 lambda_spatial=0.3,
                 lambda_wind=0.3,
                 min_lifetime_hr=5.0):

        super(PhysicsLoss, self).__init__()

        self.lambda_physics    = lambda_physics
        self.lambda_box        = lambda_box
        self.lambda_nonneg     = lambda_nonneg
        self.lambda_decay      = lambda_decay
        self.lambda_deposition = lambda_deposition
        self.lambda_spatial    = lambda_spatial
        self.lambda_wind       = lambda_wind
        self.min_lifetime_hr   = min_lifetime_hr

        self.data_criterion    = nn.MSELoss()

    def forward(self, pred, target, batch_features, feature_names):
        """
        Args:
            pred:          (N, 1) model predictions in arcsinh space
            target:        (N, 1) true values in arcsinh space
            batch_features:(N, F) raw feature tensor
            feature_names: list of feature column names

        Returns:
            total_loss, data_loss, physics_loss, loss_components dict
        """

        def get_feat(name):
            """Extract a feature column by name."""
            if name in feature_names:
                idx = feature_names.index(name)
                return batch_features[:, idx]
            return torch.zeros(pred.shape[0], device=pred.device)

        # Extract needed features
        no2_vcd    = get_feat('no2')     # keep in mol/m² scaled units
        no2_upwind = get_feat('ref_no2') # keep in mol/m² scaled units
        wind_speed    = get_feat('wind_speed')
        wind_u        = get_feat('wind_u')
        wind_v        = get_feat('wind_v')
        days_post     = get_feat('days_post_fire')
        centroid_lat  = get_feat('centroid_lat')
        centroid_lon  = get_feat('centroid_lon')

        # Month — need to reconstruct from Fourier features
        # day_of_year_sin_1 = sin(2π * doy / 365.25)
        # → doy ≈ arcsin(sin1) * 365.25 / (2π)
        sin1          = get_feat('day_of_year_sin_1')
        cos1          = get_feat('day_of_year_cos_1')
        day_of_year   = (torch.atan2(sin1, cos1) * 365.25 / (2 * np.pi)) % 365
        day_of_year   = day_of_year.long()
        month         = ((day_of_year // 30) + 1).clamp(1, 12)

        # Data loss
        data_loss = self.data_criterion(pred, target)

        # Physics terms
        loss_box        = box_model_residual(
            pred.squeeze(), no2_vcd, no2_upwind,
            wind_speed, day_of_year.float(), month, self.min_lifetime_hr
        )
        loss_nonneg     = non_negativity_loss(pred.squeeze())
        loss_decay      = temporal_decay_loss(pred.squeeze(), days_post)
        loss_deposition = deposition_consistency_loss(
            pred.squeeze(), no2_vcd, month
        )
        loss_spatial    = spatial_smoothness_loss(
            pred.squeeze(), centroid_lat, centroid_lon
        )
        loss_wind       = wind_advection_consistency_loss(
            pred.squeeze(), wind_u, wind_v, centroid_lat, centroid_lon
        )

        # Weighted sum of physics terms
        physics_loss = (
            self.lambda_box        * loss_box        +
            self.lambda_nonneg     * loss_nonneg     +
            self.lambda_decay      * loss_decay      +
            self.lambda_deposition * loss_deposition +
            self.lambda_spatial    * loss_spatial    +
            self.lambda_wind       * loss_wind
        )

        total_loss = data_loss + self.lambda_physics * physics_loss

        loss_components = {
            'data':        data_loss.item(),
            'box_model':   loss_box.item(),
            'non_neg':     loss_nonneg.item(),
            'decay':       loss_decay.item(),
            'deposition':  loss_deposition.item(),
            'spatial':     loss_spatial.item(),
            'wind':        loss_wind.item(),
            'physics_total': physics_loss.item(),
            'total':       total_loss.item()
        }

        return total_loss, data_loss, physics_loss, loss_components


# =====================================================================
# QUICK SANITY CHECK
# =====================================================================

if __name__ == '__main__':
    print("Testing PhysicsLoss...")

    N = 32  # batch size
    F = 55  # number of features

    # Fake batch
    pred           = torch.randn(N, 1)
    target         = torch.randn(N, 1)
    batch_features = torch.randn(N, F)

    # Fake feature names — must include the ones physics loss needs
    feature_names = [
        'no2', 'ref_no2', 'wind_speed', 'wind_u', 'wind_v',
        'days_post_fire', 'centroid_lat', 'centroid_lon',
        'day_of_year_sin_1', 'day_of_year_cos_1'
    ] + [f'feat_{i}' for i in range(F - 10)]

    criterion = PhysicsLoss(lambda_physics=0.1)
    total, data, physics, components = criterion(
        pred, target, batch_features, feature_names
    )

    print(f"Total loss:   {total.item():.4f}")
    print(f"Data loss:    {data.item():.4f}")
    print(f"Physics loss: {physics.item():.4f}")
    print("\nLoss components:")
    for k, v in components.items():
        print(f"  {k:<20}: {v:.4f}")
    print("\nPhysicsLoss sanity check passed.")