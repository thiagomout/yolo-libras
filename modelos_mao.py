"""Classificadores sobre pontos da mão: MLP para letras estáticas e GRU para letras com movimento."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from landmarks import (DYNAMIC_DIM, NOTHING_LABEL, SEQUENCE_LENGTH, STATIC_DIM, MotionSegmenter,
                       resample, sequence_features, static_features)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_BUNDLE = PROJECT_DIR / "model" / "mao.pt"


class StaticMLP(nn.Module):
    def __init__(self, classes: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATIC_DIM, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 63)
        return self.net(x)


class SequenceGRU(nn.Module):
    def __init__(self, classes: int, hidden: int = 96, dropout: float = 0.3):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(DYNAMIC_DIM, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 4, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, 66)
        output, _ = self.gru(self.embed(x))
        return self.head(torch.cat([output.mean(dim=1), output.max(dim=1).values], dim=1))


def random_rotation(batch: int, generator: torch.Generator, roll: float, tilt: float) -> torch.Tensor:
    """Rotações 3D pequenas: `roll` no plano da câmera e `tilt` em profundidade (graus)."""
    def angles(limit):
        return (torch.rand(batch, generator=generator) * 2 - 1) * math.radians(limit)

    a, b, c = angles(roll), angles(tilt), angles(tilt)
    zeros, ones = torch.zeros(batch), torch.ones(batch)
    rz = torch.stack([torch.stack([a.cos(), -a.sin(), zeros], -1), torch.stack([a.sin(), a.cos(), zeros], -1),
                      torch.stack([zeros, zeros, ones], -1)], 1)
    rx = torch.stack([torch.stack([ones, zeros, zeros], -1), torch.stack([zeros, b.cos(), -b.sin()], -1),
                      torch.stack([zeros, b.sin(), b.cos()], -1)], 1)
    ry = torch.stack([torch.stack([c.cos(), zeros, c.sin()], -1), torch.stack([zeros, ones, zeros], -1),
                      torch.stack([-c.sin(), zeros, c.cos()], -1)], 1)
    return rz @ rx @ ry


def augment_points(points: torch.Tensor, generator: torch.Generator, mirror: bool = True,
                   roll: float = 15.0, tilt: float = 20.0, noise: float = 0.03) -> torch.Tensor:
    """Aumenta (B, ..., 3) com a MESMA transformação para todos os pontos de cada exemplo:
    espelho (mão esquerda/direita), rotação, escala anisotrópica (perspectiva) e ruído."""
    batch = points.shape[0]
    flat = points.reshape(batch, -1, 3)
    transform = random_rotation(batch, generator, roll, tilt)
    stretch = 1 + (torch.rand(batch, 3, generator=generator) * 2 - 1) * torch.tensor([0.12, 0.12, 0.3])
    transform = transform * stretch[:, None, :]
    if mirror:
        flip = torch.where(torch.rand(batch, generator=generator) < 0.5, -1.0, 1.0)
        transform[:, :, 0] = transform[:, :, 0] * flip[:, None]
    flat = flat @ transform.transpose(1, 2)
    flat = flat + torch.randn(flat.shape, generator=generator) * noise
    return flat.reshape(points.shape)


def time_warp(sequence: np.ndarray, rng: np.random.Generator, crop: float = 0.12) -> np.ndarray:
    """Corta um pouco das pontas e deforma o tempo antes de reamostrar (sinais mais rápidos/lentos)."""
    count = len(sequence)
    start = int(rng.uniform(0, crop) * count)
    end = count - int(rng.uniform(0, crop) * count)
    part = sequence[start:max(end, start + 2)]
    positions = np.linspace(0, 1, SEQUENCE_LENGTH) ** rng.uniform(0.8, 1.25)
    dense = resample(part, max(len(part) * 4, SEQUENCE_LENGTH * 2))
    index = np.clip((positions * (len(dense) - 1)).round().astype(int), 0, len(dense) - 1)
    return dense[index]


def save_bundle(path: Path, static: dict | None, dynamic: dict | None, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"static": static, "dynamic": dynamic, "config": config}, path)


class HandSignModel:
    """Carrega model/mao.pt e faz as previsões usadas pela webcam."""

    def __init__(self, bundle: dict, device: str = "cpu"):
        self.config = bundle["config"]
        self.device = torch.device(device)
        self.static_classes: list[str] = []
        self.dynamic_classes: list[str] = []
        self.static_model = self.dynamic_model = None
        if bundle.get("static"):
            self.static_classes = list(bundle["static"]["classes"])
            self.static_model = StaticMLP(len(self.static_classes))
            self.static_model.load_state_dict(bundle["static"]["state"])
            self.static_model.to(self.device).eval()
        if bundle.get("dynamic"):
            self.dynamic_classes = list(bundle["dynamic"]["classes"])
            self.dynamic_model = SequenceGRU(len(self.dynamic_classes))
            self.dynamic_model.load_state_dict(bundle["dynamic"]["state"])
            self.dynamic_model.to(self.device).eval()

    @classmethod
    def load(cls, path: Path = DEFAULT_BUNDLE, device: str = "cpu") -> "HandSignModel":
        return cls(torch.load(path, map_location="cpu", weights_only=True), device)

    def segmenter(self, **overrides) -> MotionSegmenter:
        options = dict(self.config.get("motion", {}))
        options.update({key: value for key, value in overrides.items() if value is not None})
        return MotionSegmenter(**options)

    @torch.no_grad()
    def static_probabilities(self, points: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(static_features(points))[None].to(self.device)
        return torch.softmax(self.static_model(x), dim=1)[0].cpu().numpy()

    @torch.no_grad()
    def predict_dynamic(self, points: np.ndarray) -> tuple[str | None, float]:
        """Recebe o segmento (T, 21, 3) já recortado pelo MotionSegmenter."""
        if self.dynamic_model is None or len(points) < 2:
            return None, 0.0
        features = resample(sequence_features(points)).reshape(SEQUENCE_LENGTH, -1)
        x = torch.from_numpy(features)[None].to(self.device)
        probabilities = torch.softmax(self.dynamic_model(x), dim=1)[0].cpu().numpy()
        index = int(probabilities.argmax())
        label = self.dynamic_classes[index]
        return (None if label == NOTHING_LABEL else label), float(probabilities[index])

