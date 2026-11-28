import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.freeze_research_results import validate_coverage, verify, write_lock


class FrozenSnapshotTests(unittest.TestCase):
    def test_checksum_detects_changed_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'result.csv').write_text('mae\n1.0\n')
            write_lock(root)
            self.assertEqual(verify(root)['snapshot_files_verified'], 1)
            (root / 'result.csv').write_text('mae\n9.0\n')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                verify(root)

    def test_lock_detects_missing_or_extra_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'result.json').write_text('{}')
            write_lock(root)
            (root / 'extra.txt').write_text('unrecorded')
            with self.assertRaisesRegex(ValueError, 'file set changed'):
                verify(root)
            (root / 'extra.txt').unlink()
            (root / 'result.json').unlink()
            with self.assertRaisesRegex(ValueError, 'file set changed'):
                verify(root)

    def test_incomplete_or_duplicate_coverage_is_rejected(self):
        fires, seeds = ['A', 'B'], [42, 123, 7]
        fire = pd.DataFrame({'experiment_id': ['model', 'model'], 'fire': fires})
        seed = pd.DataFrame([{'experiment_id': 'model', 'fire': f, 'seed': s} for f in fires for s in seeds])
        validate_coverage(fire, seed, fires, seeds)
        with self.assertRaisesRegex(ValueError, 'fire coverage'):
            validate_coverage(fire.iloc[:1], seed, fires, seeds)
        with self.assertRaisesRegex(ValueError, 'seed coverage'):
            validate_coverage(fire, seed.iloc[:-1], fires, seeds)
        duplicate = pd.concat([seed.iloc[:-1], seed.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'seed coverage'):
            validate_coverage(fire, duplicate, fires, seeds)


if __name__ == '__main__':
    unittest.main()
