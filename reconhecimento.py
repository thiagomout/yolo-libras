"""Confirma sinais por duração, independentemente da velocidade da câmera."""
from __future__ import annotations


class StableRecognition:
    def __init__(self, hold: float = 0.6, release: float = 0.5):
        self.hold = hold
        self.release = release
        self.candidate: str | None = None
        self.since = 0.0
        self.stable: str | None = None
        self.last_announced: str | None = None
        self.absent_since: float | None = None

    def update(self, label: str | None, now: float) -> tuple[str | None, bool]:
        if label is None:
            self.candidate = None
            self.stable = None
            if self.absent_since is None:
                self.absent_since = now
            if now - self.absent_since >= self.release:
                self.last_announced = None
            return None, False
        # A pausa pode terminar entre dois quadros; considere sua duração aqui também.
        if self.absent_since is not None and now - self.absent_since >= self.release:
            self.last_announced = None
        self.absent_since = None
        if label != self.candidate:
            self.candidate = label
            self.since = now
            self.stable = None
        if now - self.since < self.hold:
            return None, False
        self.stable = label
        announce = label != self.last_announced
        if announce:
            self.last_announced = label
        return label, announce
