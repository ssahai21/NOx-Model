"""Audit the first seed-42 training epoch's spatial pair topology in every outer fold."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def main():
    data = pd.read_csv('data/phase_2_all_fires_model.csv', parse_dates=['date'])
    data = data[data.days_post_fire >= 0].sort_values(['fire_name', 'plot_id', 'date']).reset_index(drop=True)
    records = []
    for held_out in sorted(data.fire_name.unique()):
        outer = data[data.fire_name != held_out]
        train = outer[outer.groupby('fire_name').date.rank(method='dense', pct=True) <= .8]
        coords = train[['centroid_lat', 'centroid_lon']].to_numpy(dtype=np.float32)
        dates = train.date.to_numpy(dtype='datetime64[D]').astype(np.int64)
        fires = train.fire_name.to_numpy()
        loader = DataLoader(
            torch.arange(len(train)), batch_size=512, shuffle=True,
            drop_last=len(train) % 512 == 1, generator=torch.Generator().manual_seed(42),
        )
        old_count = 0
        old_valid = 0
        new_counts = []
        for indices in loader:
            indices = indices.numpy()
            lat, lon = coords[indices].T
            north = (lat[:, None] - lat[None, :]) * 111320.
            east = (lon[:, None] - lon[None, :]) * 111320.
            squared = north ** 2 + east ** 2
            # The legacy epsilon makes self-pairs and repeated locations pass dist > 0.
            legacy_near = np.sqrt(squared + 1e-8) < 20000
            context = ((dates[indices, None] == dates[None, indices])
                       & (fires[indices, None] == fires[None, indices]))
            distinct = squared > 0
            old_count += int(legacy_near.sum())
            old_valid += int((legacy_near & context & distinct).sum())
            corrected = north ** 2 + (east * np.cos(np.deg2rad((lat[:, None] + lat[None, :]) / 2))) ** 2
            new_counts.append(int((context & distinct & (corrected < 20000 ** 2)).sum()))
        records.append({
            'held_out_fire': held_out, 'batches': len(loader),
            'legacy_pairs': old_count, 'legacy_same_date_fire_distinct_pairs': old_valid,
            'legacy_eligible_fraction': old_valid / old_count,
            'repaired_mean_pairs_per_batch': float(np.mean(new_counts)),
            'repaired_zero_pair_batch_fraction': float(np.mean(np.array(new_counts) == 0)),
        })
    output = Path('results/phase2/information_physics_study')
    pd.DataFrame(records).to_csv(output / 'pair_topology_per_fire.csv', index=False)
    total = sum(row['legacy_pairs'] for row in records)
    eligible = sum(row['legacy_same_date_fire_distinct_pairs'] for row in records)
    summary = {
        'audit_scope': 'First training epoch, seed 42, all 16 outer folds, same DataLoader shuffle.',
        'legacy_pairs': total, 'eligible_legacy_pairs': eligible,
        'eligible_fraction': eligible / total,
        'median_repaired_pairs_per_batch': float(np.median([r['repaired_mean_pairs_per_batch'] for r in records])),
        'max_zero_pair_batch_fraction': max(r['repaired_zero_pair_batch_fraction'] for r in records),
    }
    (output / 'pair_topology_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
