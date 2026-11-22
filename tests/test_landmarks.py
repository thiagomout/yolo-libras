import argparse
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from coletar import save_recording
from landmarks import MotionSegmenter, hand_scale, longest_motion, normalize_hand, resample, sequence_features
from modelos_mao import HandSignModel, SequenceGRU, StaticMLP, augment_points, save_bundle, time_warp
from treinar_landmarks import MOTION_CONFIG, load_recordings, train_dynamic, train_static

RNG = np.random.default_rng(0)
HAND = (RNG.normal(0, 0.05, (21, 3)) + [0.5, 0.5, 0]).astype(np.float32)
HAND[0] = [0.5, 0.6, 0]  # pulso abaixo da palma


def moving_recording(direction, fps=30, still=0.5, move=0.5, jitter=0.0005):
    """Mão parada, depois se desloca 2 palmas na direção dada, depois parada de novo."""
    times, frames = [], []
    total = 2 * still + move
    step = np.array(direction + [0], dtype=np.float32) * 0.2
    for i in range(int(total * fps)):
        t = i / fps
        progress = min(max((t - still) / move, 0), 1)
        frames.append(HAND + step * progress + RNG.normal(0, jitter, (21, 3)).astype(np.float32))
        times.append(t)
    return np.stack(frames), np.array(times, dtype=np.float32)


class LandmarkTests(unittest.TestCase):
    def test_normalization_ignores_position_and_size(self):
        moved = HAND * 2.5 + np.array([0.3, -0.1, 0.05], dtype=np.float32)
        np.testing.assert_allclose(normalize_hand(HAND), normalize_hand(moved), atol=1e-5)

    def test_resample_keeps_endpoints(self):
        sequence = np.arange(10, dtype=np.float32).reshape(10, 1)
        result = resample(sequence, 32)
        self.assertEqual(result.shape, (32, 1))
        self.assertEqual(result[0, 0], 0)
        self.assertEqual(result[-1, 0], 9)

    def test_segmenter_finds_one_motion_and_ignores_jitter(self):
        points, times = moving_recording([1, 0])
        segmenter = MotionSegmenter(**MOTION_CONFIG)
        found = [s for p, t in zip(points, times) if (s := segmenter.update(p, float(t))) is not None]
        self.assertEqual(len(found), 1)
        duration = found[0][1][-1] - found[0][1][0]
        self.assertTrue(0.3 < duration < 1.0, duration)

        segmenter.reset()
        still = [segmenter.update(HAND + RNG.normal(0, 0.0005, (21, 3)).astype(np.float32), i / 30) for i in range(60)]
        self.assertTrue(all(s is None for s in still))
        self.assertFalse(segmenter.moving)

    def test_too_long_motion_is_discarded(self):
        segmenter = MotionSegmenter(start=1.0, max_duration=1.0)
        results = []
        for i in range(120):  # 4 s balançando a mão
            offset = np.float32(0.1 * np.sin(i / 3))
            results.append(segmenter.update(HAND + offset, i / 30))
        for i in range(120, 150):
            results.append(segmenter.update(HAND, i / 30))
        self.assertTrue(all(r is None for r in results))

    def test_longest_motion_and_features(self):
        points, times = moving_recording([0, 1])
        segment = longest_motion(points, times, MotionSegmenter(**MOTION_CONFIG))
        self.assertIsNotNone(segment)
        features = sequence_features(segment)
        self.assertEqual(features.shape[1:], (22, 3))
        expected = 0.2 / hand_scale(HAND)  # pulso desceu 0.2 em unidades da imagem
        self.assertAlmostEqual(float(features[-1, 21, 1]), expected, delta=0.1 * expected)

    def test_augmentation_and_time_warp_shapes(self):
        generator = torch.Generator().manual_seed(0)
        batch = torch.randn(8, 32, 22, 3)
        self.assertEqual(augment_points(batch, generator).shape, batch.shape)
        self.assertEqual(time_warp(np.random.rand(40, 66).astype(np.float32), np.random.default_rng(0)).shape, (32, 66))


class TrainingTests(unittest.TestCase):
    def test_bundle_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mao.pt"
            save_bundle(path, {"classes": ["A", "B"], "state": StaticMLP(2).state_dict()},
                        {"classes": ["J", "NADA"], "state": SequenceGRU(2).state_dict()}, {"motion": MOTION_CONFIG})
            model = HandSignModel.load(path)
            self.assertEqual(model.static_probabilities(HAND).shape, (2,))
            label, confidence = model.predict_dynamic(moving_recording([1, 0])[0][:20])
            self.assertIn(label, {"J", None})
            self.assertGreaterEqual(confidence, 0)

    def test_collect_and_train_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for person in ("ana", "bia"):
                for label, direction in (("J", [1, 0]), ("Z", [0, 1])):
                    for _ in range(6):
                        points, times = moving_recording(direction, jitter=0.002)
                        save_recording(root / "dinamico" / label, person, label, "dinamico", list(points),
                                       list(times), False)
                # Diferença que não some com o espelhamento do treino: B tem os dedos esticados.
                stretched = HAND.copy()
                stretched[1:] = HAND[0] + (HAND[1:] - HAND[0]) * [1, 2.5, 1]
                for label, hand in (("A", HAND), ("B", stretched)):
                    for _ in range(3):
                        frames = [hand + RNG.normal(0, 0.003, (21, 3)) for _ in range(10)]
                        save_recording(root / "estatico" / label, person, label, "estatico", frames,
                                       list(range(10)), False)
            dynamic = load_recordings(root / "dinamico")
            self.assertEqual(len(dynamic), 24)
            self.assertEqual({r["person"] for r in dynamic}, {"ana", "bia"})
            args = argparse.Namespace(static=root / "nao_existe.npz", test_person="bia", val_fraction=0.2, frame_stride=2,
                                      batch=32, epochs=10, seq_epochs=20, seq_batch=8, collected_share=0.3, seed=0)
            generator = torch.Generator().manual_seed(0)
            static, static_metrics = train_static(args, random.Random(0), generator, torch.device("cpu"),
                                     load_recordings(root / "estatico"))
            dyn, metrics = train_dynamic(args, random.Random(0), generator, torch.device("cpu"), dynamic)
            self.assertEqual(static["classes"], ["A", "B"])
            self.assertGreaterEqual(static_metrics["coleta_test"]["acuracia"], 0.9)
            self.assertEqual(dyn["classes"], ["J", "Z"])
            self.assertGreaterEqual(metrics["coleta_test"]["acuracia"], 0.9)


if __name__ == "__main__":
    unittest.main()
