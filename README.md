# Predicting Soil NOₓ Emissions After Northern California Wildfires

A 2026 NASA SEES Air Quality Initiative project by **Shreya Sahai and Lilybet Downey**. We investigated whether satellite and environmental data could estimate post-fire soil nitrogen oxide (NOₓ) emissions, and whether a model trained on several fires could generalize to a new one.

**Status** · Exploratory research. In the archived leave-one-fire-out runs, the physics-informed network had slightly higher transformed-scale R² and slightly higher flux-scale MAE than the data-only network. The comparison does not establish a benefit from adding the physics loss.

## Background

The project began with Google Earth Engine and an XGBoost experiment, then continued in PyTorch with a feedforward network and a box-model-inspired loss. The original project report (Sections 3.2–3.3) records R² ≈ **0.82** with a random row split and ≈ **0.09** with an entire fire held out. Those figures describe the early XGBoost model’s **NO₂ column target**. The experiments in this repository predict a derived **soil NOₓ flux**.

## How we derived the target

The project used Sentinel-5P tropospheric NO₂ columns and adapted a box-model approach from [Huber et al. (2020)](https://doi.org/10.1029/2020GL089949). The project report describes the target calculation schematically as:

```text
E_soil ≈ U · ΔNO₂ / L + Vd · NO₂ / Z_pbl + NO₂ / τ − E_NEI
         advection     dry deposition      chemical loss  anthropogenic correction
```

Here, ΔNO₂ is the plot column minus the upwind reference (`ref_no2`), U is wind speed, L is fetch distance, Vd is deposition velocity, Z_pbl is boundary-layer height, τ is atmospheric NOₓ lifetime, and E_NEI is the anthropogenic emissions adjustment. The export contains three seasonal lifetime scenarios with minima of 3, 5, or 7 hours and a maximum of 15 hours. The training scripts use `soil_nox_tau5hr`. Flux is reported in **ng N m⁻² s⁻¹**.

The target is a satellite-derived estimate. The project did not use direct soil flux measurements for validation. Combustion NOₓ could contribute to the observed enhancement during or soon after burning. The project report also notes unusually negative estimates around rain events, including a Camp Fire estimate of approximately −194 ng N m⁻² s⁻¹. The available analysis does not establish what caused that particular estimate.

### Data sources

The original project used Google Earth Engine to access Sentinel-5P OFFL NO₂, Sentinel-2 surface reflectance for NBR and dNBR, SMAP L4 soil moisture and temperature, ERA5-family weather data, NASADEM elevation, USDA Cropland Data Layer, and MTBS fire perimeters. This repository starts from the team’s exported table. The Earth Engine scripts and input CSV are outside the repository.

## Modeling workflow

1. **Prepare the export.** Deduplicate by fire, plot, and date, then parse dates and sort. Use numeric predictors and exclude direct identifiers, soil NOₓ target variants, NO₂ columns, and NO₂-derived box-model components. The original feature set covered soil conditions, weather, burn severity, elevation, cropland, plot coordinates, days since fire, wind, and seasonal Fourier terms. The exact feature count depends on the export.
2. **Transform the target.** Clip flux at the 1st and 99th percentiles, then apply `arcsinh` to handle negative values and compress the tails. Reverse the transformation with `sinh` for flux-scale metrics. In each fold, use training-fire medians for imputation and fit feature scaling on the training fires. The archived target percentiles were computed before splitting.
3. **Train the baseline.** Fit a data-only neural network with hidden layers of 256, 128, and 64 units, batch normalization, ReLU, and 0.3 dropout. The baseline minimizes mean squared error on the transformed target.
4. **Add physics-informed penalties.** Minimize the data loss plus λ times a weighted sum of penalties for box-model consistency, non-negativity, post-fire decay, deposition, spatial smoothness, and wind consistency. The six default term weights in that order are **1.0, 2.0, 0.1, 0.5, 0.3, and 0.3**. The final run uses **λ = 0.01**. NO₂ and reference NO₂ enter through a separate, unscaled loss input. The box term standardizes the predictions and its reference within each batch, which emphasizes relative variation. Its implemented residual omits the anthropogenic subtraction in the label equation above and uses seasonal deposition velocities. The spatial and wind penalties operate on rows within a mini-batch.
5. **Tune and rerun.** The final run uses hidden layers of 256/128/64, dropout 0.3, learning rate 0.001, batch size 512, Adam weight decay 0.0001, and seeds **42, 123, and 7**. The architecture, physics-term weights, and global physics weight were explored in separate experiments.

## Evaluation

Random row splitting can place nearby observations from the same fire on both sides of the split. Leave-one-fire-out instead trains on three fires per fold and holds out the fourth. The archived folds cover **August Complex, Camp Fire, Dixie Fire, and North Complex**.

R² is reported on the transformed target. Mean absolute error (MAE) is reported on the back-transformed flux scale. Percent bias is the sum of prediction errors divided by the sum of absolute target values, multiplied by 100. Negative values indicate net underprediction.

## Results

| Archived experiment | Mean R² (transformed) | Mean MAE (ng N m⁻² s⁻¹) |
| --- | ---: | ---: |
| Data-only network, one archived run | −0.001 | 18.869 |
| Tuned PINN, three-seed mean ± standard deviation | 0.038 ± 0.007 | 19.267 ± 0.346 |

The PINN’s mean transformed R² was about 0.039 higher and its mean flux MAE about 0.40 higher. The PINN standard deviations describe variation across three seed averages. The baseline has one archived run, so this table cannot tell us how much of the difference is due to run-to-run variation. The baseline’s mean percent bias was **−29.5%**. The final three-seed PINN JSON does not record percent bias. Camp Fire had the lowest transformed R² in the archived baseline and tuned per-fire results. Seasonal mismatch is one hypothesis to test.

[Baseline results](results/baseline_nn_results.json) · [Final PINN results](results/final_pinn_results.json) · [Tuned per-fire results](results/pinn_tuned_results.json) · [Physics-weight stability](results/lambda_stability.json)

## Repository map

| File | Purpose |
| --- | --- |
| [`src/fix_dedup.py`](src/fix_dedup.py) | Remove duplicate fire, plot, and date rows |
| [`src/data_loader.py`](src/data_loader.py) | Load data, select features, transform targets, and build grouped folds |
| [`src/baseline_nn.py`](src/baseline_nn.py) | Train and evaluate the data-only network |
| [`src/physics_loss.py`](src/physics_loss.py) | Define the physics penalties |
| [`src/pinn.py`](src/pinn.py) | Train and evaluate the physics-informed network |
| [`src/hyperparameter_tuning.py`](src/hyperparameter_tuning.py) | Define the configurable network and run architecture and training-configuration searches |
| [`src/physics_tuning.py`](src/physics_tuning.py) | Search physics-term weights |
| [`src/final_pinn.py`](src/final_pinn.py) | Run the final configuration across three seeds |
| [`results/`](results/) | Archived metrics and figures |

## Reproduce the archived runs

Run from the repository root with a Python environment compatible with the pinned packages. The training entry points use Apple MPS when available and CPU otherwise.

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

Place the team’s prepared export at `data/nox_clean.csv`. If starting with `data/df_final.csv`, run the deduplication command first.

```bash
python -m src.fix_dedup
python -m src.baseline_nn
python -m src.final_pinn
```

The scripts write metrics to `results/`. The final PINN script saves checkpoints by fire under `models/`, so later seeds overwrite earlier checkpoints for the same fire. Code and archived outputs are available for inspection. Rerunning training requires the original export.

## Limitations

- **Target validity.** Performance measures agreement with the box-model estimate. Combustion emissions and other influences on the satellite column make it difficult to isolate the soil contribution.
- **Evaluation used the held-out fire during development.** Early stopping and learning-rate scheduling used the held-out fire, and tuning reused results from the grouped folds. Treat the archived metrics as exploratory.
- **Target and prediction clipping.** Target percentiles were computed across all fires before splitting. The pipeline also clips held-out targets to the training target range and clips predictions for flux-scale scoring using the held-out labels’ range.
- **Overlap between the target and physics loss.** The label and the box-model penalty draw on the same NO₂ observations. The original workflow used a constant deposition velocity according to the project documentation, while the loss uses seasonal values. The penalty may reinforce assumptions already present in the label.
- **Few independent fires.** Four fires and one archived baseline run limit conclusions about both cross-fire generalization and seed variability.

For a stricter rerun, fit target preprocessing within each training fold, use training fires for early stopping and tuning, and score the untouched held-out labels without using their range to clip predictions. Compare the data-only network and PINN with matched architectures, splits, and seeds.

## References and credit

Huber, D. E., Steiner, A. L., & Kort, E. A. (2020). Daily cropland soil NOₓ emissions identified by TROPOMI and SMAP. *Geophysical Research Letters*, 47(22), e2020GL089949. [Paper](https://doi.org/10.1029/2020GL089949)

Developed by Shreya Sahai and Lilybet Downey through the 2026 NASA SEES Air Quality Initiative and continued after the internship. The team presented its initial work at the SEES symposium.
