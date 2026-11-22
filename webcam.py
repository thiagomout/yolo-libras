from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO
from voz import SpeechWorker
from reconhecimento import StableRecognition

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "model" / "best.pt"
DEFAULT_METADATA = PROJECT_DIR / "model" / "metadata.json"
DEFAULT_HAND_MODEL = PROJECT_DIR / "model" / "mao.pt"



def load_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def get_prediction(result: Any, task: str, margin: float = 0.0) -> tuple[str | None, float]:
    """Retorna a classe mais confiante para classificação ou detecção."""
    if task == "classify":
        probs = getattr(result, "probs", None)
        if probs is None:
            return None, 0.0
        index = int(probs.top1)
        confidence = float(probs.top1conf)
        scores = sorted(probs.data.cpu().tolist(), reverse=True)
        if len(scores) > 1 and scores[0] - scores[1] < margin:
            return None, confidence
        return str(result.names[index]), confidence

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None, 0.0
    confidences = boxes.conf.cpu().tolist()
    best_index = max(range(len(confidences)), key=lambda index: confidences[index])
    class_id = int(boxes.cls[best_index].item())
    confidence = float(confidences[best_index])
    return str(result.names[class_id]), confidence


def draw_detection_boxes(frame, result, threshold: float) -> None:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return
    for box in boxes:
        confidence = float(box.conf.item())
        if confidence < threshold:
            continue
        class_id = int(box.cls.item())
        label = str(result.names[class_id])
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 60), 2)
        cv2.putText(
            frame,
            f"{label} {confidence:.0%}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (60, 220, 60),
            2,
            cv2.LINE_AA,
        )


def open_camera(index: int):
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    camera = cv2.VideoCapture(index, backend)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(index)
    return camera


def wsl_without_camera() -> bool:
    return "microsoft" in platform.release().lower() and not list(Path("/dev").glob("video*"))


class HandRecognizer:
    """Pontos da mão -> letra estática (mão parada) ou letra com movimento (ao fim do movimento)."""

    def __init__(self, model, conf: float, margin: float, dyn_conf: float, smooth: int, motion: float | None):
        from collections import deque
        from landmarks import NOTHING_LABEL

        self.model = model
        self.conf, self.margin, self.dyn_conf = conf, margin, dyn_conf
        self.segmenter = model.segmenter(start=motion)
        self.history = deque(maxlen=max(1, smooth))
        self.nothing = NOTHING_LABEL
        self.cooldown_until = 0.0

    def update(self, points, now: float) -> tuple[str | None, float, tuple[str, float] | None]:
        """Retorna (letra estática aceita, confiança, evento dinâmico (letra, confiança) ou None)."""
        event = None
        segment = self.segmenter.update(points, now)
        if segment is not None and self.model.dynamic_model is not None:
            label, confidence = self.model.predict_dynamic(segment[0])
            if label is not None and confidence >= self.dyn_conf:
                event = (label, confidence)
                # Evita anunciar a pose final do movimento como se fosse uma letra estática.
                self.cooldown_until = now + 0.8
        if points is None or self.segmenter.moving or self.model.static_model is None:
            self.history.clear()
            return None, 0.0, event
        self.history.append(self.model.static_probabilities(points))
        probabilities = sum(self.history) / len(self.history)
        order = probabilities.argsort()[::-1]
        confidence = float(probabilities[order[0]])
        label = self.model.static_classes[order[0]]
        if now < self.cooldown_until or label == self.nothing or confidence < self.conf:
            return None, confidence, event
        if len(order) > 1 and confidence - float(probabilities[order[1]]) < self.margin:
            return None, confidence, event
        return label, confidence, event


def run_hand_mode(args, model_path: Path) -> int:
    if not 0 <= args.conf <= 1 or not 0 <= args.margin <= 1 or args.hold <= 0 or args.release <= 0:
        print("conf/margem devem estar entre 0 e 1; hold/release devem ser positivos.")
        return 2
    if wsl_without_camera():
        print("WSL sem webcam disponível: nenhum /dev/video* encontrado.")
        print("Execute iniciar_windows.ps1 no PowerShell do Windows; o modelo treinado no WSL será reutilizado.")
        return 2
    from landmarks import HandTracker, draw_hand
    from modelos_mao import HandSignModel

    model = HandSignModel.load(model_path)
    recognizer = HandRecognizer(model, args.conf, args.margin, args.dyn_conf, args.smooth, args.motion)
    tracker = HandTracker(video=True)
    stability = StableRecognition(args.hold, args.release)
    speech = None if args.no_voice else SpeechWorker(args.voice_backend)
    camera = open_camera(args.camera)
    if not camera.isOpened():
        if speech:
            speech.close()
        tracker.close()
        print(f"Não consegui abrir a câmera {args.camera}. Tente `--camera 1` ou confira a permissão da webcam.")
        return 2

    print("Modo pontos da mão. Estáticas:", " ".join(model.static_classes) or "-")
    print("Com movimento:", " ".join(c for c in model.dynamic_classes if c != recognizer.nothing) or "- (grave com coletar.py)")
    print("Pressione Q ou ESC para sair; R repete a última letra.")
    last_spoken, event_text, event_until = None, "", 0.0
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Não consegui ler um quadro da câmera.")
                break
            if args.mirror_input:
                frame = cv2.flip(frame, 1)
            now = time.monotonic()
            points = tracker.detect(frame, now)
            label, confidence, event = recognizer.update(points, now)
            stable_label, announce = stability.update(label, now)
            if event:
                event_text, event_until = f"Letra: {event[0]} (movimento {event[1]:.0%})", now + 1.5
                last_spoken = event[0]
                if speech:
                    speech.say(event[0])
            elif announce:
                last_spoken = stable_label
                if speech:
                    speech.say(stable_label)

            if points is not None:
                draw_hand(frame, points, (0, 0, 255) if recognizer.segmenter.moving else (80, 220, 80))
            if now < event_until:
                text, color = event_text, (255, 160, 0)
            elif stable_label:
                text, color = f"Letra: {stable_label} ({confidence:.0%})", (50, 220, 50)
            elif points is None:
                text, color = "Mostre a mao para a camera", (0, 200, 255)
            elif recognizer.segmenter.moving:
                text, color = "Movimento...", (0, 160, 255)
            else:
                text, color = f"Aguardando sinal... ({confidence:.0%})", (0, 200, 255)
            cv2.rectangle(frame, (10, 10), (520, 58), (25, 25, 25), -1)
            cv2.putText(frame, text, (22, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
            if speech and speech.error:
                cv2.putText(frame, "Voz indisponivel: execute python voz.py", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, f"Q/ESC: sair | R: repetir | mov {recognizer.segmenter.speed:.1f}", (15, frame.shape[0] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Reconhecimento de Libras", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r") and last_spoken and speech:
                speech.say(last_spoken)
    finally:
        camera.release()
        cv2.destroyAllWindows()
        tracker.close()
        if speech:
            speech.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconhece letras de Libras pela webcam e fala a letra.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Caminho para o best.pt treinado.")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA), help="Caminho para metadata.json.")
    parser.add_argument("--camera", type=int, default=0, help="Índice da webcam (padrão: 0).")
    parser.add_argument("--conf", type=float, default=0.70, help="Confiança mínima para aceitar uma previsão (0-1).")
    parser.add_argument("--no-voice", action="store_true", help="Desliga a fala, mas mantém a previsão na tela.")
    parser.add_argument("--voice-backend", choices=["auto", "espeak", "pyttsx3", "windows"], default="auto")
    parser.add_argument("--hold", type=float, default=0.6, help="Segundos para confirmar a letra.")
    parser.add_argument("--release", type=float, default=0.5, help="Pausa para permitir repetir a mesma letra.")
    parser.add_argument("--margin", type=float, default=0.15, help="Diferença mínima entre as duas classes mais prováveis.")
    parser.add_argument("--roi", type=float, default=0.65, help="Lado do quadrado central / menor dimensão do quadro; 0 usa quadro inteiro.")
    parser.add_argument("--mirror-input", action="store_true", help="Espelha a entrada, caso seu dataset tenha essa orientação.")
    parser.add_argument("--modo", choices=["auto", "mao", "yolo"], default="auto",
                        help="mao: pontos da mão (model/mao.pt); yolo: classificador de imagem; auto: mao se existir.")
    parser.add_argument("--mao-model", default=str(DEFAULT_HAND_MODEL), help="Caminho para mao.pt.")
    parser.add_argument("--dyn-conf", type=float, default=0.6, help="Confiança mínima para letras com movimento.")
    parser.add_argument("--motion", type=float, default=None, help="Limiar de movimento (palmas/s); padrão do treino.")
    parser.add_argument("--smooth", type=int, default=5, help="Quadros para suavizar a previsão estática.")
    args = parser.parse_args()

    hand_model_path = Path(args.mao_model).expanduser().resolve()
    if args.modo == "mao" or (args.modo == "auto" and hand_model_path.is_file()):
        if not hand_model_path.is_file():
            print(f"Modelo de pontos não encontrado: {hand_model_path}. Rode treinar_landmarks.py.")
            return 2
        return run_hand_mode(args, hand_model_path)

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        print(f"Modelo não encontrado: {model_path}")
        print("Execute primeiro: python treinar.py --zip dataset.zip")
        return 2
    if not 0.0 <= args.conf <= 1.0:
        print("--conf precisa estar entre 0 e 1.")
        return 2

    if not 0 <= args.roi <= 1 or not 0 <= args.margin <= 1 or args.hold <= 0 or args.release <= 0:
        print("ROI/margem devem estar entre 0 e 1; hold/release devem ser positivos.")
        return 2

    if wsl_without_camera():
        print("WSL sem webcam disponível: nenhum /dev/video* encontrado.")
        print("Execute iniciar_windows.ps1 no PowerShell do Windows; o modelo treinado no WSL será reutilizado.")
        return 2

    metadata = load_metadata(Path(args.metadata).expanduser().resolve())
    try:
        model = YOLO(str(model_path))
    except Exception as exc:
        print(f"Não consegui carregar o modelo: {exc}")
        return 2

    task = getattr(model, "task", None) or metadata.get("task") or ("detect" if getattr(model, "task", "classify") == "detect" else "classify")
    img_size = int(metadata.get("imgsz", 640 if task == "detect" else 224))
    predict_options = {}
    if task == "classify" and metadata.get("preprocessing") == "resize_full_image":
        from classificacao import FullImagePredictor
        predict_options["predictor"] = FullImagePredictor
    stability = StableRecognition(args.hold, args.release)
    current_stable = None
    speech = None if args.no_voice else SpeechWorker(args.voice_backend)

    camera = open_camera(args.camera)
    if not camera.isOpened():
        if speech:
            speech.close()
        print(f"Não consegui abrir a câmera {args.camera}. Tente `--camera 1` ou confira a permissão da webcam.")
        return 2

    print("Webcam iniciada. Pressione Q ou ESC para sair; R repete a última letra estável.")
    print("Dica: coloque a mao inteira dentro do quadrado e ocupe boa parte dele. Use fundo neutro e boa luz.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Não consegui ler um quadro da câmera.")
                break
            if args.mirror_input:
                frame = cv2.flip(frame, 1)
            input_image = frame
            roi_box = None
            if task == "classify" and args.roi > 0:
                height, width = frame.shape[:2]
                side = max(1, int(min(height, width) * args.roi))
                x, y = (width - side) // 2, (height - side) // 2
                input_image = frame[y:y + side, x:x + side].copy()
                roi_box = (x, y, x + side, y + side)

            try:
                result = model.predict(input_image, imgsz=img_size, verbose=False, **predict_options)[0]
                if task == "detect":
                    draw_detection_boxes(frame, result, args.conf)
                label, confidence = get_prediction(result, task, args.margin)
            except Exception as exc:
                cv2.putText(frame, f"Erro de previsao: {str(exc)[:45]}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                label, confidence = None, 0.0

            accepted = label if label is not None and confidence >= args.conf else None
            stable_label, announce = stability.update(accepted, time.monotonic())
            current_stable = stable_label
            if announce and speech:
                speech.say(stable_label)
            if roi_box:
                cv2.rectangle(frame, roi_box[:2], roi_box[2:], (0, 200, 255), 2)
            if speech and speech.error:
                cv2.putText(frame, "Voz indisponivel: execute python voz.py", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if stable_label:
                display_label = f"Letra: {stable_label} ({confidence:.0%})"
                display_color = (50, 220, 50)
            else:
                display_label = f"Aguardando sinal... ({confidence:.0%})"
                display_color = (0, 200, 255)

            cv2.rectangle(frame, (10, 10), (470, 58), (25, 25, 25), -1)
            cv2.putText(frame, display_label, (22, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.85, display_color, 2, cv2.LINE_AA)
            cv2.putText(frame, "Q/ESC: sair | R: repetir letra", (15, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Reconhecimento de Libras - Parte 1", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r") and current_stable:
                if speech:
                    speech.say(current_stable)
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if speech:
            speech.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
