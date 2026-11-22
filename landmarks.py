"""Pontos da mão (MediaPipe) e as transformações usadas igualmente no treino e na webcam.

Os pontos são guardados "crus" (coordenadas da imagem, com x e z corrigidos pela proporção
largura/altura). A normalização acontece depois, para poder mudar sem reextrair o dataset.
"""
from __future__ import annotations

import os
import tempfile
import urllib.request
from collections import deque
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
HAND_MODEL = PROJECT_DIR / "model" / "hand_landmarker.task"
# Pasta das gravações. Defina LIBRAS_COLETA para usar uma pasta compartilhada (ex.: no Google Drive).
COLLECTION_DIR = Path(os.environ.get("LIBRAS_COLETA") or PROJECT_DIR / "coleta").expanduser()
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Letras do alfabeto manual de Libras que envolvem movimento.
DYNAMIC_LETTERS = ("H", "J", "K", "X", "Z", "Ç")
# Classe gravada com movimentos que NÃO são letras (transições, mão entrando no quadro...).
NOTHING_LABEL = "NADA"
SEQUENCE_LENGTH = 32
N_POINTS = 21
WRIST = 0
PALM_POINTS = (5, 9, 13, 17)
MOTION_POINTS = (0, 4, 8, 12, 16, 20)  # pulso e pontas dos dedos
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
    (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
STATIC_DIM = N_POINTS * 3
DYNAMIC_DIM = (N_POINTS + 1) * 3  # forma da mão + trajetória do pulso


def canonical_label(label: str) -> str:
    """Aceita CC/CEDILHA para Ç, útil em terminais sem acentos."""
    label = label.strip().upper()
    if label in {"CC", "C_CEDILHA", "CEDILHA", "C-CEDILHA"}:
        return "Ç"
    return label


def ensure_hand_model(path: Path = HAND_MODEL) -> Path:
    """Baixa o modelo do MediaPipe uma única vez (~8 MB)."""
    if path.is_file() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Baixando o modelo de pontos da mão para {path} ...")
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".part")
    os.close(descriptor)
    try:
        urllib.request.urlretrieve(HAND_MODEL_URL, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


class HandTracker:
    """Encapsula o HandLandmarker do MediaPipe. `video=True` usa rastreamento entre quadros."""

    def __init__(self, video: bool = False, model_path: Path | None = None, min_confidence: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self.video = video
        self._last_timestamp = -1
        mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_hand_model(model_path or HAND_MODEL))),
            running_mode=mode,
            num_hands=1,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, bgr: np.ndarray, timestamp_s: float | None = None) -> np.ndarray | None:
        """Retorna (21, 3) float32 ou None se não houver mão."""
        import cv2

        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if self.video:
            timestamp = int((timestamp_s or 0.0) * 1000)
            timestamp = max(timestamp, self._last_timestamp + 1)  # precisa ser estritamente crescente
            self._last_timestamp = timestamp
            result = self._detector.detect_for_video(image, timestamp)
        else:
            result = self._detector.detect(image)
        if not result.hand_landmarks:
            return None
        height, width = bgr.shape[:2]
        aspect = width / height
        points = np.array([(p.x * aspect, p.y, p.z * aspect) for p in result.hand_landmarks[0]], dtype=np.float32)
        return points

    def close(self) -> None:
        self._detector.close()


def draw_hand(frame: np.ndarray, points: np.ndarray, color=(80, 220, 80)) -> None:
    import cv2

    height = frame.shape[0]
    pixels = [(int(x * height), int(y * height)) for x, y, _ in points]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pixels[a], pixels[b], color, 2, cv2.LINE_AA)
    for pixel in pixels:
        cv2.circle(frame, pixel, 3, (255, 255, 255), -1, cv2.LINE_AA)


def hand_scale(points: np.ndarray) -> float:
    """Tamanho da palma: distância média do pulso às bases dos dedos (estável ao fechar a mão)."""
    distances = np.linalg.norm(points[list(PALM_POINTS)] - points[WRIST], axis=-1)
    return float(max(distances.mean(), 1e-6))


def normalize_hand(points: np.ndarray) -> np.ndarray:
    """Remove posição e tamanho: pulso na origem, palma com tamanho 1. Mantém a rotação,
    porque em Libras a orientação da mão diferencia letras."""
    centered = points - points[WRIST]
    return (centered / hand_scale(points)).astype(np.float32)


def static_features(points: np.ndarray) -> np.ndarray:
    return normalize_hand(points).reshape(-1)


def sequence_features(sequence: np.ndarray) -> np.ndarray:
    """(T, 21, 3) -> (T, 22, 3): forma normalizada por quadro + deslocamento do pulso
    desde o início, medido em tamanhos de palma."""
    scale = float(np.median([hand_scale(frame) for frame in sequence]))
    shapes = np.stack([normalize_hand(frame) for frame in sequence])
    trajectory = (sequence[:, WRIST] - sequence[0, WRIST]) / scale
    return np.concatenate([shapes, trajectory[:, None, :]], axis=1).astype(np.float32)


def resample(sequence: np.ndarray, length: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Reamostra no tempo por interpolação linear, para sinais rápidos e lentos terem o mesmo tamanho."""
    count = len(sequence)
    if count == 1:
        return np.repeat(sequence, length, axis=0)
    positions = np.linspace(0, count - 1, length)
    low = np.floor(positions).astype(int)
    high = np.minimum(low + 1, count - 1)
    weight = (positions - low).reshape(-1, *([1] * (sequence.ndim - 1)))
    return (sequence[low] * (1 - weight) + sequence[high] * weight).astype(np.float32)


class MotionSegmenter:
    """Detecta o início e o fim de um movimento da mão.

    A velocidade é medida em tamanhos de palma por segundo, comparando o quadro atual com o de
    ~`window` segundos antes (isso suaviza o tremor dos pontos). Usa histerese: começa acima de
    `start`, termina quando fica abaixo de `stop` por `still` segundos. O mesmo código recorta
    as gravações no treino, para treino e webcam verem segmentos equivalentes.
    """

    def __init__(self, start: float = 1.5, stop: float | None = None, still: float = 0.2,
                 window: float = 0.1, pre_roll: float = 0.15, min_duration: float = 0.25,
                 max_duration: float = 3.0):
        self.start = start
        self.stop = stop if stop is not None else start * 0.6
        self.still = still
        self.window = window
        self.pre_roll = pre_roll
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.history: deque[tuple[float, np.ndarray]] = deque()
        self.speed = 0.0
        self.moving = False
        self._segment: list[tuple[float, np.ndarray]] = []
        self._still_since: float | None = None
        self._too_long = False

    def reset(self) -> None:
        self.history.clear()
        self._segment = []
        self.moving = False
        self.speed = 0.0
        self._still_since = None
        self._too_long = False

    def _finish(self) -> tuple[np.ndarray, np.ndarray] | None:
        segment, self._segment = self._segment, []
        too_long, self._too_long = self._too_long, False
        self.moving = False
        self._still_since = None
        if not segment or too_long:
            return None
        duration = segment[-1][0] - segment[0][0]
        if not self.min_duration <= duration <= self.max_duration:
            return None
        times = np.array([t for t, _ in segment], dtype=np.float32)
        points = np.stack([p for _, p in segment]).astype(np.float32)
        return points, times

    def update(self, points: np.ndarray | None, now: float) -> tuple[np.ndarray, np.ndarray] | None:
        """Retorna (pontos (T,21,3), tempos (T,)) quando um movimento termina."""
        if points is None:
            self.history.clear()
            self.speed = 0.0
            return self._finish() if self.moving else None

        self.history.append((now, points))
        horizon = now - max(self.pre_roll, self.window) - 0.05
        while self.history and self.history[0][0] < horizon:
            self.history.popleft()
        reference = next(((t, p) for t, p in reversed(self.history) if now - t >= self.window), None)
        if reference is None:
            self.speed = 0.0
        else:
            elapsed = max(now - reference[0], 1e-3)
            idx = list(MOTION_POINTS)
            displacement = np.linalg.norm(points[idx, :2] - reference[1][idx, :2], axis=-1).mean()
            self.speed = float(displacement / hand_scale(points) / elapsed)

        if not self.moving:
            if self.speed >= self.start:
                self.moving = True
                self._still_since = None
                self._segment = [(t, p) for t, p in self.history if now - t <= self.pre_roll + self.window]
            return None

        self._segment.append((now, points))
        if self._segment[-1][0] - self._segment[0][0] > self.max_duration:
            # Movimento longo demais (ex.: mão balançando): descarta tudo e espera parar.
            self._too_long = True
            self._segment = self._segment[-1:]
        if self.speed < self.stop:
            if self._still_since is None:
                self._still_since = now
            if now - self._still_since >= self.still:
                return self._finish()
        else:
            self._still_since = None
        return None


def longest_motion(points: np.ndarray, times: np.ndarray, segmenter: MotionSegmenter) -> np.ndarray | None:
    """Passa uma gravação pelo segmentador e devolve o maior movimento encontrado."""
    segmenter.reset()
    best = None
    for frame, now in zip(points, times):
        found = segmenter.update(frame, float(now))
        if found is not None and (best is None or len(found[0]) > len(best)):
            best = found[0]
    if segmenter.moving:
        found = segmenter.update(None, float(times[-1]))
        if found is not None and (best is None or len(found[0]) > len(best)):
            best = found[0]
    segmenter.reset()
    return best
