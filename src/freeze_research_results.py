"""Create or verify a presentation snapshot without retraining or changing studies."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import sys
import zipfile

DEFAULT_OUTPUT = Path('results/final/2026-11-27')
STUDIES = {
    'target': Path('results/phase2/target_cap_study'),
    'physics': Path('results/phase2/information_physics_study'),
    'peak': Path('results/phase2/peak_validation_study'),
    'scaling': Path('results/phase2/history_scaling_study'),
    'xgb': Path('results/phase2/xgboost_evaluation'),
}
# A canonical record for each unique completed arm. Cross-study duplicates are omitted.
SPECS = [
    ('env_arcsinh_capped', 'target', 'arcsinh_capped', 'Environment NN, transformed loss, capped', 55, False),
    ('env_arcsinh_uncapped', 'target', 'arcsinh_uncapped', 'Environment NN, transformed loss, uncapped', 55, False),
    ('environment_nn', 'physics', 'environment_nn', 'Environment NN, flux loss', 55, True),
    ('env_flux_uncapped', 'target', 'flux_uncapped', 'Environment NN, flux loss, uncapped', 55, False),
    ('history_nn', 'physics', 'history_nn', 'NN with past NO2', 60, True),
    ('legacy_pinn', 'physics', 'legacy_pinn', 'Legacy physics penalties', 55, False),
    ('repaired_pinn', 'physics', 'repaired_pinn', 'Repaired physics penalties', 55, True),
    ('pair_only_pinn', 'physics', 'pair_only_pinn', 'Repaired spatial/wind penalties only', 55, True),
    ('inner_mse_control', 'peak', 'inner_arcsinh_capped', 'Inner-fire MSE control', 60, False),
    ('inner_peak_capped', 'peak', 'inner_peak_capped', 'Peak validation, capped', 60, False),
    ('inner_peak_uncapped', 'peak', 'inner_peak_uncapped', 'Peak validation, uncapped', 60, False),
    ('inner_peak_weighted', 'peak', 'inner_peak_weighted', 'Peak validation, fivefold tail weighting', 60, False),
    ('inner_validation_selected', 'peak', 'validation_selected', 'Validation-selected loss before refit', 60, False),
    ('refit_mse_control', 'peak', 'refit_control', 'Refitted MSE control', 60, True),
    ('raw_peak_refit', 'peak', 'refit_selected', 'Peak-focused refit, raw inputs', 60, True),
    ('asinh_inner', 'scaling', 'asinh_inner', 'Signed-log inputs before refit', 60, False),
    ('asinh_refit', 'scaling', 'asinh_refit', 'Signed-log refit everywhere', 60, True),
    ('scaling_selected', 'scaling', 'validation_selected_scaling', 'Validation-selected input scaling', 60, True),
    ('xgb_arcsinh', 'xgb', 'arcsinh', 'XGBoost, transformed loss', 55, False),
    ('xgb_flux', 'xgb', 'flux', 'XGBoost, physical loss', 55, True),
]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_lock(root):
    records = {str(p.relative_to(root)): {'sha256': digest(p), 'bytes': p.stat().st_size}
               for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'snapshot_lock.json'}
    (root / 'snapshot_lock.json').write_text(json.dumps({'format_version': 1, 'files': records}, indent=2) + '\n')


def verify(root, sources=False):
    lock = json.loads((root / 'snapshot_lock.json').read_text())
    expected_paths = set(lock['files'])
    actual_paths = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name != 'snapshot_lock.json'}
    if expected_paths != actual_paths:
        raise ValueError(f'Snapshot file set changed. Missing={expected_paths-actual_paths}, extra={actual_paths-expected_paths}')
    for name, record in lock['files'].items():
        path = root / name
        if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise ValueError(f'Snapshot checksum mismatch: {name}')
    source_count = 0
    if sources:
        inventory = json.loads((root / 'source_inventory.json').read_text())
        for name, record in inventory['files'].items():
            if not Path(name).is_file() or digest(name) != record['sha256']:
                raise ValueError(f'Original source artifact missing or changed: {name}')
            source_count += 1
    return {'snapshot_files_verified': len(lock['files']), 'original_source_files_verified': source_count}


def validate_coverage(per_fire, per_seed, fires, seeds):
    identifiers = set(per_fire.experiment_id)
    if identifiers != set(per_seed.experiment_id):
        raise ValueError('Experiment sets differ between fire and seed tables')
    expected_seeds = {(fire, seed) for fire in fires for seed in seeds}
    for identifier in identifiers:
        a = per_fire[per_fire.experiment_id == identifier]
        b = per_seed[per_seed.experiment_id == identifier]
        if len(a) != len(fires) or set(a.fire) != set(fires) or a.fire.duplicated().any():
            raise ValueError(f'Incomplete or duplicated fire coverage: {identifier}')
        actual = set(zip(b.fire, b.seed))
        if len(b) != len(expected_seeds) or actual != expected_seeds or b[['fire', 'seed']].duplicated().any():
            raise ValueError(f'Incomplete or duplicated seed coverage: {identifier}')


def load_tables():
    import numpy as np
    import pandas as pd
    summaries = {name: json.loads((path / 'summary.json').read_text()) for name, path in STUDIES.items()}
    fire_tables = {name: pd.read_csv(path / 'per_fire_comparison.csv') for name, path in STUDIES.items()}
    seed_tables = {name: pd.read_csv(path / 'per_seed_comparison.csv') for name, path in STUDIES.items() if name != 'xgb'}
    models, per_fire, per_seed, registry = [], [], [], []
    for identifier, family, arm, label, features, headline in SPECS:
        path = STUDIES[family]
        if family == 'xgb':
            score = next(r for r in summaries[family]['results'] if r['loss'] == arm and r['model'] == 'xgb')
            record = {'equal_fire_mean_mae': score['equal_fire_mean_mae'], 'pooled_mae': score['pooled_mae'],
                      'r2_physical': score['r2_physical'], 'r2_arcsinh': score['r2_arcsinh'],
                      'first_month_mae': score['first_month_mae'], 'extreme_flux_mae': score['extreme_flux_mae'],
                      'peak_day_mae': score['peak_day_mae'], 'extreme_flux_bias': score['extreme_flux_bias'],
                      'peak_day_bias': score['peak_day_bias']}
            fire = fire_tables[family].query('loss == @arm and model == "xgb"').copy()
            report = json.loads((path / arm / 'metrics.json').read_text())
            seed = pd.DataFrame([{'fire': f['fire'], 'seed': int(s), **{k: v[k] for k in ['mae', 'r2_physical', 'r2_arcsinh', 'percent_bias']}}
                                 for f in report['folds'] for s, v in f['seeds'].items()])
        else:
            score = summaries[family]['arms'][arm]
            peak_key = 'peak_days_scores' if family == 'target' else 'peak_days'
            record = {'equal_fire_mean_mae': score['equal_fire_mean_mae'], 'pooled_mae': score['pooled']['mae'],
                      'r2_physical': score['pooled']['r2_physical'], 'r2_arcsinh': score['pooled']['r2_arcsinh'],
                      'first_month_mae': score['first_month']['mae'], 'extreme_flux_mae': score['extreme_flux']['mae'],
                      'peak_day_mae': score[peak_key]['mae'], 'extreme_flux_bias': score['extreme_flux']['percent_bias'],
                      'peak_day_bias': score[peak_key]['percent_bias']}
            fire = fire_tables[family][fire_tables[family].arm == arm].copy()
            seed = seed_tables[family][seed_tables[family].arm == arm].copy()
        peak_column = next(c for c in ['peak_days_mae', 'peak_day_mae', 'peak_days_scores_mae'] if c in fire)
        fire = fire.rename(columns={peak_column: 'peak_day_mae'})
        fire = fire[['fire', 'mae', 'r2_physical', 'r2_arcsinh', 'percent_bias', 'first_month_mae', 'extreme_flux_mae', 'peak_day_mae']]
        seed = seed[['fire', 'seed', 'mae', 'r2_physical', 'r2_arcsinh', 'percent_bias']]
        if not np.isfinite(fire[['mae', 'r2_physical', 'r2_arcsinh']].to_numpy()).all():
            raise ValueError(f'Nonfinite full-period fire metric: {identifier}')
        if not np.isfinite(seed[['mae', 'r2_physical', 'r2_arcsinh']].to_numpy()).all():
            raise ValueError(f'Nonfinite full-period seed metric: {identifier}')
        fire.insert(0, 'experiment_id', identifier)
        seed.insert(0, 'experiment_id', identifier)
        np.testing.assert_allclose(fire.mae.mean(), record['equal_fire_mean_mae'], rtol=1e-10)
        registry.append({'experiment_id': identifier, 'label': label, 'study': family, 'source_arm': arm,
                         'features': features, 'headline': headline, 'source_directory': str(path),
                         'validation': 'late-date blocks inside training fires' if family in ['target', 'physics', 'xgb'] else 'whole inner fires; refit where specified',
                         'satellite_history_available': features == 60})
        models.append({'experiment_id': identifier, 'label': label, 'features': features, 'headline': headline, **record})
        per_fire.append(fire)
        per_seed.append(seed)
    per_fire, per_seed = pd.concat(per_fire, ignore_index=True), pd.concat(per_seed, ignore_index=True)
    fires = sorted(per_fire.fire.unique())
    validate_coverage(per_fire, per_seed, fires, [42, 123, 7])
    assert len(fires) == 16
    return pd.DataFrame(models), per_fire, per_seed, registry


def collect_inventory():
    # Keep superseded/incomplete runs out of the final source inventory.
    directories = {
        'target': ['arcsinh_capped', 'arcsinh_uncapped', 'flux_capped', 'flux_uncapped'],
        'physics': ['lagged_nn', 'legacy_pinn', 'repaired_pinn', 'pair_only_pinn'],
        'peak': ['inner_arcsinh_capped', 'inner_peak_capped', 'inner_peak_uncapped', 'inner_peak_weighted', 'refit_control', 'refit_selected'],
        'scaling': ['inner_asinh', 'refit_asinh', 'decisions'],
        'xgb': ['arcsinh', 'flux'],
    }
    paths = set()
    for name, root in STUDIES.items():
        paths.update(p for p in root.iterdir() if p.is_file())
        for folder in directories[name]:
            paths.update(p for p in (root / folder).rglob('*') if p.is_file())
    paths.update(Path('src').rglob('*.py'))
    paths.update(Path('tests').rglob('*.py'))
    paths.update([Path('README.md'), Path('requirements.txt'), Path('requirements-research.txt'), Path('.gitignore'),
                  Path('data/phase_2_all_fires_model.csv'), Path('data/phase_2_no2_history_model.csv'), Path('data/phase_2_daily_clean.csv')])
    return {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(paths)}


def make_figure(output, table):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    raw = table[table.experiment_id == 'raw_peak_refit'].set_index('fire')
    scaled = table[table.experiment_id == 'scaling_selected'].set_index('fire')
    order = raw.mae.sort_values(ascending=False).index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.5), sharey=True, layout='constrained')
    for axis, metric, title in zip(axes, ['mae', 'peak_day_mae'], ['Full-period MAE by fire', 'Peak-day MAE by fire']):
        for i, fire in enumerate(order):
            a, b = raw.loc[fire, metric], scaled.loc[fire, metric]
            if np.isfinite(a) and np.isfinite(b):
                axis.plot([a, b], [i, i], color='#BBBBBB', linewidth=1, zorder=1)
                axis.scatter(a, i, color='#0072B2', marker='o', s=34, label='Raw-input peak refit' if i == 0 else None, zorder=2)
                axis.scatter(b, i, color='#009E73', marker='x', s=42, label='Validation-selected scaling' if i == 0 else None, zorder=3)
            else:
                axis.text(.02, i, 'No qualifying peak score', transform=axis.get_yaxis_transform(), fontsize=8, color='#666666')
        axis.set_title(title)
        axis.set_xlabel('MAE (ng N m⁻² s⁻¹); lower is better')
        axis.grid(axis='x', alpha=.2)
        axis.set_axisbelow(True)
        axis.set_xlim(left=0)
    axes[0].set_yticks(range(len(order)), order)
    axes[0].invert_yaxis()
    axes[0].legend(loc='lower right', fontsize=8)
    fig.suptitle('Held-out fire variability · three-seed ensembles · exploratory benchmark')
    fig.savefig(output / 'per_fire_variability.png', dpi=200)
    fig.savefig(output / 'per_fire_variability.svg')
    plt.close(fig)


def write_report(output, models, variability, paired):
    lines = ['# Frozen presentation results', '',
             'Snapshot date 2026-11-27, America/Los_Angeles. This freezes the completed experiment set for presentation and portfolio review. No training was run to create it. Scientific/domain review remains pending.', '',
             '## Evaluation status', '',
             'All canonical arms cover the same 16 outer fires, 74,581 post-ignition rows and three seeds (42, 123, 7). Model fitting, preprocessing and within-fold selection excluded the outer fire. The benchmark was repeatedly examined during project development. These are exploratory cross-validation results, not a fresh independent final test.', '',
             'The target is the satellite box-model-derived `soil_nox_tau5hr` estimate in ng N m⁻² s⁻¹. No independent soil-flux measurements were used. Agreement with this target does not establish accuracy against actual soil emissions.', '',
             '## Main comparisons', '',
             '| Model | Inputs | Equal-fire MAE | Pooled MAE | Physical pooled R² | First-month MAE | Extreme-row MAE | Peak-day MAE |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in models[models.headline].to_dict('records'):
        lines.append(f"| {row['label']} | {row['features']} | {row['equal_fire_mean_mae']:.3f} | {row['pooled_mae']:.3f} | {row['r2_physical']:.3f} | {row['first_month_mae']:.3f} | {row['extreme_flux_mae']:.3f} | {row['peak_day_mae']:.3f} |")
    lines += ['', 'Lower MAE is better. Equal-fire MAE averages the 16 fire-specific ensemble errors. Pooled MAE weights each observation equally. Physical R² is pooled across raw flux values; transformed R² is preserved in the CSV but is a different measure.', '',
              'The 55-input models use environmental predictors. The 60-input models additionally assume strictly earlier satellite NO2 observations from the outer fire are available at prediction time. This is estimation assisted by past satellite observations. Whole-fire inner validation and refits also change training rows relative to late-date validation. The table is not an architecture-only ranking.', '',
              'XGBoost is matched to the environment-only NN inputs. Its pseudo-Huber objective differs slightly from NN SmoothL1. Physics comparisons use matched environment inputs, physical data loss and seeds. Repaired physics fixes improve implementation correctness, but the tested penalties do not beat the plain environment NN.', '',
              '## Frozen interpretation', '',
              '- The raw-input peak-focused refit is retained as the equal-fire MAE reference profile. This does not assert it has the smallest test MAE among all exploratory candidates.',
              '- Validation-selected scaling is the extreme-flux candidate. It improves extreme-row error modestly while slightly increasing equal-fire MAE. Scaling is selected by inner validation, not outer test scores.',
              '- No single pipeline is best on every metric or every fire. Do not present the smallest observed test metric as a separately validated global model choice.',
              '- Large values remain substantially underestimated. Extreme-row aggregate bias is approximately −82.6% for the raw peak refit and −79.8% for validation-selected scaling.',
              '- The fixed fivefold tail-weight arm has stronger peak-day performance with worse average and negative-label errors. It remains in the complete 20-arm registry rather than being hidden.', '',
              'The common reporting subsets contain 5,771 first-month rows (days 0–30, 14 fires), 726 extreme rows (above each held-out fold’s frozen training p99), and 214 peak daily means (above the frozen training daily-mean p95). Thresholds are shared across models and stored in `protocols/reporting_thresholds.json`. Internal training/validation thresholds can differ because those fits use different training subsets.', '',
              'Percent bias is `100 * sum(prediction - actual) / sum(actual)` on the specified cohort. The denominator is signed, not the sum of absolute labels. Bias can be misleading when signed targets cancel; use MAE alongside it. Positive-tail cohort biases are easier to interpret.', '',
              '## Per-fire and seed variability', '',
              '| Model | Fire MAE median | Fire MAE IQR | Fire MAE min–max | Individual-seed macro MAE range |',
              '| --- | ---: | ---: | ---: | ---: |']
    for row in variability[variability.experiment_id.isin(models[models.headline].experiment_id)].to_dict('records'):
        label = models.set_index('experiment_id').loc[row['experiment_id'], 'label']
        lines.append(f"| {label} | {row['fire_mae_median']:.3f} | {row['fire_mae_q25']:.3f}–{row['fire_mae_q75']:.3f} | {row['fire_mae_min']:.3f}–{row['fire_mae_max']:.3f} | {row['seed_macro_mae_min']:.3f}–{row['seed_macro_mae_max']:.3f} |")
    wins = int((paired.mae_change < -1e-10).sum()); losses = int((paired.mae_change > 1e-10).sum())
    lines += ['', f'For scaling-selected versus raw peak-refit full-period MAE, {wins} fires improved, {losses} worsened, and {16-wins-losses} were unchanged. Missing peak/first-month scores mean the fire has too few qualifying observations; they are not zero error.', '',
              'Fire distributions describe heterogeneity across 16 evaluated fires. Seed ranges and population standard deviations describe three fixed optimization seeds. Neither is a confidence interval. Individual-seed macro MAE is computed before ensembling and differs from the macro MAE of the averaged predictions. Rows within a fire are correlated and do not create tens of thousands of independent tests.', '',
              '![Per-fire variability](per_fire_variability.png)', '',
              '## Snapshot contents and verification', '',
              '- `model_results.csv` contains all 20 unique completed canonical arms; `headline_results.csv` is the nine-row main table.',
              '- `per_fire_results.csv` and `per_seed_results.csv` contain all 320 fire rows and 960 fire/seed rows. `variability_summary.csv` and `paired_scaling_changes.csv` make the spread and tradeoffs explicit.',
              '- `experiment_registry.json` pins each canonical arm to its original study and feature/validation assumptions. Repeated aliases across study tables are deduplicated.',
              '- The final set covers the five study families in that registry. Earlier four-fire runs and initial all-fire physics/unit explorations remain outside this presentation snapshot and are not used for its headline claims.',
              '- `protocols/` copies original manifests, selection records and reporting thresholds. Superseded/incomplete physics outputs and the initial capped XGBoost ceiling pilot are excluded.',
              '- `code_snapshot.zip` captures current repository Python sources, tests, dependency files and README. Original study manifests retain historical code hashes, which can differ from the current source snapshot. The archive does not claim bitwise reconstruction of every earlier training execution.',
              '- `source_inventory.json` hashes original study outputs, predictions, checkpoints and input tables in their existing locations. They are not copied into this portable summary snapshot. Data and weights remain subject to their separate availability and sharing arrangements.',
              '- `snapshot_lock.json` hashes every packaged file. The builder refuses to replace an existing snapshot directory. A changed snapshot requires a new output/version.', '',
              'From the project root, run `python -m src.freeze_research_results --verify` for packaged-file checks, or add `--verify-sources` to check all original local inputs and artifacts. The latter requires the local data/checkpoints. Neither command trains a model.', '',
              '## Claims to keep out of the presentation', '',
              'Do not claim validated field-soil emissions, operational forecasting accuracy, a PINN advantage, a fresh untouched final test set, or statistical significance from three seeds. Do not use the original four-fire/random-row scores as the current benchmark. Archived historical descriptions are in `docs/archive/README_original_four_fire.md` at the repository root.', '',
              'Mentor review of box-model units, combustion contamination and the soil-specific interpretation is still pending. More independent fires or field observations would strengthen the scientific claims.']
    (output / 'README.md').write_text('\n'.join(lines) + '\n')


def build(output):
    import numpy as np
    import pandas as pd
    if output.exists():
        raise FileExistsError(f'Refusing to replace frozen snapshot: {output}')
    latest = json.loads((STUDIES['scaling'] / 'study_manifest.json').read_text())
    assert latest['completion']['status'] == 'complete'
    for group in ['source_sha256', 'training_code_sha256', 'analysis_code_sha256']:
        for name, expected in latest[group].items():
            if digest(name) != expected:
                raise ValueError(f'Latest completed study provenance changed: {name}')
    for name, expected in latest['decision_sha256'].items():
        if digest(STUDIES['scaling'] / 'decisions' / name) != expected:
            raise ValueError(f'Frozen scaling decision changed: {name}')
    models, fire, seed, registry = load_tables()
    data = pd.read_csv('data/phase_2_all_fires_model.csv')
    post = data[data.days_post_fire >= 0]
    assert len(post) == 74581 and post.fire_name.nunique() == 16
    assert set(post.fire_name) == set(fire.fire)
    variability = []
    for identifier, group in fire.groupby('experiment_id', sort=False):
        values = seed[seed.experiment_id == identifier].groupby('seed').mae.mean()
        variability.append({'experiment_id': identifier, 'fire_mae_median': group.mae.median(),
                            'fire_mae_q25': group.mae.quantile(.25), 'fire_mae_q75': group.mae.quantile(.75),
                            'fire_mae_min': group.mae.min(), 'fire_mae_max': group.mae.max(),
                            'fire_mae_sd_population': group.mae.std(ddof=0),
                            'seed_macro_mae_mean': values.mean(), 'seed_macro_mae_sd_population': values.std(ddof=0),
                            'seed_macro_mae_min': values.min(), 'seed_macro_mae_max': values.max()})
    variability = pd.DataFrame(variability)
    raw = fire[fire.experiment_id == 'raw_peak_refit'].set_index('fire')
    scaled = fire[fire.experiment_id == 'scaling_selected'].set_index('fire')
    paired = pd.DataFrame({'fire': raw.index})
    for column in ['mae', 'first_month_mae', 'extreme_flux_mae', 'peak_day_mae']:
        paired[f'{column}_raw'] = raw[column].to_numpy()
        paired[f'{column}_scaling'] = scaled.loc[raw.index, column].to_numpy()
        paired[f'{column}_change'] = paired[f'{column}_scaling'] - paired[f'{column}_raw']
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in [('model_results', models), ('headline_results', models[models.headline]),
                        ('per_fire_results', fire), ('per_seed_results', seed), ('variability_summary', variability),
                        ('paired_scaling_changes', paired)]:
        frame.to_csv(output / f'{name}.csv', index=False)
    (output / 'experiment_registry.json').write_text(json.dumps(registry, indent=2) + '\n')
    environment = {'snapshot_date_local': '2026-11-27', 'timezone': 'America/Los_Angeles',
                   'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
                   'packages': {p: version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'torch', 'matplotlib', 'xgboost']},
                   'note': 'Observed environment at snapshot time; historical environments are not independently reconstructed.',
                   'dataset': {'raw_model_rows': len(data), 'evaluated_rows': len(post), 'fires': 16,
                               'fire_plot_pairs': data.groupby(['fire_name', 'plot_id']).ngroups,
                               'negative_evaluated_labels': int((post.soil_nox_tau5hr < 0).sum())}}
    (output / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    protocols = output / 'protocols'; protocols.mkdir()
    for family, root in STUDIES.items():
        for path in root.glob('*.json'):
            if any(word in path.name for word in ['manifest', 'plan', 'selection', 'extension']):
                shutil.copyfile(path, protocols / f'{family}_{path.name}')
    threshold = json.loads((STUDIES['target'] / 'summary.json').read_text())['thresholds']
    (protocols / 'reporting_thresholds.json').write_text(json.dumps(threshold, indent=2) + '\n')
    for folder in ['decisions']:
        if (STUDIES['scaling'] / folder).exists():
            shutil.copytree(STUDIES['scaling'] / folder, protocols / 'scaling_decisions')
    make_figure(output, fire)
    write_report(output, models, variability, paired)
    with zipfile.ZipFile(output / 'code_snapshot.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        paths = sorted(set(Path('src').rglob('*.py')) | set(Path('tests').rglob('*.py')) |
                       {Path('README.md'), Path('requirements.txt'), Path('requirements-research.txt'), Path('.gitignore'), Path('docs/archive/README_original_four_fire.md')})
        for path in paths:
            archive.write(path, str(path))
    (output / 'source_inventory.json').write_text(json.dumps({'files': collect_inventory()}, indent=2) + '\n')
    write_lock(output)
    print(json.dumps(verify(output, sources=True), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--verify-sources', action='store_true')
    args = parser.parse_args()
    if args.verify or args.verify_sources:
        print(json.dumps(verify(args.output, sources=args.verify_sources), indent=2))
    else:
        build(args.output)
