"""Grava exemplos de sinais pela webcam como pontos da mão (não salva imagens).

Exemplos:
  python coletar.py --pessoa ana --letras H J K X Z CC          # letras com movimento
  python coletar.py --pessoa ana --letras NADA                  # movimentos que não são letras
  python coletar.py --pessoa ana --letras A B C --tipo estatico # mais variação para estáticas

Teclas: ESPAÇO grava/para | N/P próxima/anterior letra | D apaga a última gravação | Q/ESC sai.
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from landmarks import COLLECTION_DIR, DYNAMIC_LETTERS, NOTHING_LABEL, HandTracker, MotionSegmenter, canonical_label, draw_hand

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = COLLECTION_DIR
MAX_DYNAMIC_SECONDS = 4.0


def recording_type(label: str, forced: str) -> str:
    if forced != "auto":
        return forced
    return "dinamico" if label in DYNAMIC_LETTERS or label == NOTHING_LABEL else "estatico"


def save_recording(folder: Path, person: str, label: str, kind: str, points: list, times: list, mirrored: bool) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = folder / f"{person}_{stamp}.npz"
    np.savez_compressed(path, points=np.stack(points).astype(np.float32),
                        times=np.array(times, dtype=np.float32) - times[0],
                        label=label, person=person, kind=kind, mirrored=mirrored)
    return path


def put(frame, text, y, color=(255, 255, 255), scale=0.6):
    cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1 if scale < 0.7 else 2, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description="Grava sinais de Libras como pontos da mão.")
    parser.add_argument("--pessoa", required=True, help="Identifica quem grava; usado para separar treino e teste.")
    parser.add_argument("--letras", nargs="+", default=list(DYNAMIC_LETTERS),
                        help="Letras a gravar (use CC para Ç e NADA para movimentos que não são letras).")
    parser.add_argument("--tipo", choices=["auto", "estatico", "dinamico"], default="auto")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--duracao", type=float, default=2.0, help="Segundos gravados por exemplo estático.")
    parser.add_argument("--mirror-input", action="store_true", help="Use igual ao webcam.py.")
    parser.add_argument("--saida", default=str(DEFAULT_OUTPUT),
                        help="Pasta das gravações (padrão: variável LIBRAS_COLETA ou ./coleta).")
    args = parser.parse_args()

    person = "".join(ch for ch in args.pessoa.strip().lower() if ch.isalnum() or ch in "-_")
    if not person:
        print("--pessoa precisa ter letras ou números.")
        return 2
    labels = [canonical_label(label) for label in args.letras]
    output = Path(args.saida).expanduser().resolve()
    print(f"Gravando em: {output}")

    if "microsoft" in platform.release().lower() and not list(Path("/dev").glob("video*")):
        print("WSL sem webcam. No PowerShell do Windows execute:")
        print(f"  powershell -ExecutionPolicy Bypass -File .\\iniciar_windows.ps1 coletar.py --pessoa {person} --letras ...")
        return 2

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    camera = cv2.VideoCapture(args.camera, backend)
    if not camera.isOpened():
        camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print(f"Não consegui abrir a câmera {args.camera}.")
        return 2

    tracker = HandTracker(video=True)
    motion = MotionSegmenter()
    index = 0
    recording = False
    rec_points: list[np.ndarray] = []
    rec_times: list[float] = []
    rec_started = 0.0
    saved: list[Path] = []
    message, message_until = "", 0.0
    print(__doc__)

    def folder_for(label: str) -> Path:
        return output / recording_type(label, args.tipo) / label

    def count(label: str) -> int:
        folder = folder_for(label)
        return len(list(folder.glob(f"{person}_*.npz"))) if folder.is_dir() else 0

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Não consegui ler a câmera.")
                break
            if args.mirror_input:
                frame = cv2.flip(frame, 1)
            now = time.monotonic()
            points = tracker.detect(frame, now)
            motion.update(points, now)
            label = labels[index]
            kind = recording_type(label, args.tipo)

            if points is not None:
                draw_hand(frame, points, (0, 0, 255) if recording else (80, 220, 80))
                if recording:
                    rec_points.append(points)
                    rec_times.append(now)

            elapsed = now - rec_started
            limit = args.duracao if kind == "estatico" else MAX_DYNAMIC_SECONDS
            stop_now = recording and elapsed >= limit
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if recording:
                    stop_now = True
                else:
                    recording, rec_points, rec_times, rec_started = True, [], [], now
            if stop_now:
                recording = False
                if len(rec_points) < 5:
                    message = "Gravacao descartada: mao nao apareceu o suficiente."
                else:
                    path = save_recording(folder_for(label), person, label, kind, rec_points, rec_times, args.mirror_input)
                    saved.append(path)
                    message = f"Salvo {label} ({len(rec_points)} quadros)"
                message_until = now + 2
            elif key in (ord("n"), ord("N"), 9) and not recording:
                index = (index + 1) % len(labels)
            elif key in (ord("p"), ord("P")) and not recording:
                index = (index - 1) % len(labels)
            elif key in (ord("d"), ord("D")) and saved and not recording:
                removed = saved.pop()
                removed.unlink(missing_ok=True)
                message, message_until = f"Apagado: {removed.name}", now + 2
            elif key in (ord("q"), 27):
                break

            # Interface
            width = frame.shape[1]
            cv2.rectangle(frame, (0, 0), (width, 95), (25, 25, 25), -1)
            put(frame, f"Letra: {label}  ({kind})  gravadas por {person}: {count(label)}", 30, (0, 220, 255), 0.75)
            if kind == "dinamico":
                tip = "ESPACO, faca o sinal partindo da mao parada, pare e ESPACO" if not recording else "Gravando... ESPACO para parar"
            else:
                tip = f"ESPACO grava {args.duracao:.0f}s: mova/gire levemente a mao" if not recording else f"Gravando {elapsed:.1f}s..."
            put(frame, tip, 58)
            bar = int(min(motion.speed / (motion.start * 2), 1.0) * 200)
            cv2.rectangle(frame, (15, 70), (215, 85), (80, 80, 80), 1)
            cv2.rectangle(frame, (15, 70), (15 + bar, 85), (0, 0, 255) if motion.moving else (0, 200, 0), -1)
            start_x = 15 + int(200 / 2)
            cv2.line(frame, (start_x, 66), (start_x, 89), (255, 255, 255), 1)
            cv2.putText(frame, f"movimento {motion.speed:.1f} (limiar {motion.start:.1f})", (225, 83),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            if recording:
                cv2.circle(frame, (width - 30, 30), 12, (0, 0, 255), -1)
            if points is None:
                put(frame, "Mao nao detectada", frame.shape[0] - 45, (0, 0, 255))
            if now < message_until:
                put(frame, message, frame.shape[0] - 18, (0, 255, 0))
            else:
                put(frame, "N/P: letra | D: apaga ultima | Q: sair", frame.shape[0] - 18)
            cv2.imshow("Coleta de sinais - Libras", frame)
    finally:
        camera.release()
        cv2.destroyAllWindows()
        tracker.close()
    print(f"{len(saved)} gravações salvas nesta sessão em {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
