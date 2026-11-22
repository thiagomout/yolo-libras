import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from auditar_dataset import audit


class AuditTests(unittest.TestCase):
    def test_reports_cross_split_similarities_and_respects_sample_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ('train', 'val', 'test'):
                folder = root / split / 'A'
                folder.mkdir(parents=True)
                for index in range(3):
                    image = np.full((64, 64, 3), 220, dtype=np.uint8)
                    cv2.rectangle(image, (10 + index * 4, 15), (30 + index * 4, 50), (60, 80, 100), -1)
                    cv2.imwrite(str(folder / f'{index}.png'), image)
            report = audit(root, samples_per_class=2)
            for split in ('val', 'test'):
                self.assertEqual(report['splits'][split]['sampled'], 2)
                self.assertTrue(all(row['distance_63_bits'] == 0 for row in report['splits'][split]['pairs']))
                self.assertTrue(all(Path(row['nearest_train']).parent.parent.name == 'train' for row in report['splits'][split]['pairs']))

    def test_missing_train_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'train'):
                audit(Path(tmp))
