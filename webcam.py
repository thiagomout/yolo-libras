from __future__ import annotations

import argparse
import json
import queue
import threading
from collections import Counter, deque
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "model" / "best.pt"
DEFAULT_METADATA = PROJECT_DIR / "model" / "metadata.json"


class SpeechWorker:
    """Fala as letras em uma thread separada para não travar o vídeo."""

    def __init__(self) -> None:
        self.messages: queue.Queue[str | None] = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def say(self, text: str) -> None:
        self.messages.put(text)

    @staticmethod
    def _select_portuguese_voice(engine: Any) -> None:
        try:
            voices = engine.getProperty("voices")
            for voice in voices:
                details = f"{getattr(voice, 'name', '')} {getattr(voice, 'id', '')}".lower()
                languages = " ".join(
                    item.decode("utf-8", errors="ignore") if isinstance(item, bytes) else str(item)
                    for item in getattr(voice, "languages", [])
                ).lower()
                if any(word in details + " " + languages for word in ("portuguese", "português", "brazil", "pt-br", "pt_br")):
                    engine.setProperty("voice", voice.id)
                    break
            engine.setProperty("rate", 155)
        except Exception:
            # O sintetizador padrão continua sendo usado se a seleção de voz falhar.
            pass

    def _run(self) -> None:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            self._select_portuguese_voice(engine)
            while True:
                message = self.messages.get()
                if message is None:
                    break
                try:
                    engine.say(message)
                    engine.runAndWait()
                except Exception as exc:
                    print(f"Não foi possível falar '{message}': {exc}")
            try:
                engine.stop()
            except Exception:
                pass
        except Exception as exc:
            print("Não consegui iniciar a síntese de voz. Confira se há uma voz instalada em português.")
            print(f"Detalhe do sintetizador: {exc}")

    def close(self) -> None:
        self.messages.put(None)
        self.thread.join(timeout=2.0)


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def get_prediction(result: Any, task: str) -> tuple[str | None, float]:
    """Retorna a classe mais confiante para classificação ou detecção."""
    if task == "classify":
        probs = getattr(result, "probs", None)
        if probs is None:
            return None, 0.0
        index = int(probs.top1)
        confidence = float(probs.top1conf)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconhece letras de Libras pela webcam e fala a letra.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Caminho para o best.pt treinado.")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA), help="Caminho para metadata.json.")
    parser.add_argument("--camera", type=int, default=0, help="Índice da webcam (padrão: 0).")
    parser.add_argument("--conf", type=float, default=0.70, help="Confiança mínima para aceitar uma previsão (0-1).")
    parser.add_argument("--no-voice", action="store_true", help="Desliga a fala, mas mantém a previsão na tela.")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        print(f"Modelo não encontrado: {model_path}")
        print("Execute primeiro: python treinar.py --zip dataset.zip")
        return 2
    if not 0.0 <= args.conf <= 1.0:
        print("--conf precisa estar entre 0 e 1.")
        return 2

    metadata = load_metadata(Path(args.metadata).expanduser().resolve())
    try:
        model = YOLO(str(model_path))
    except Exception as exc:
        print(f"Não consegui carregar o modelo: {exc}")
        return 2

    task = metadata.get("task") or ("detect" if getattr(model, "task", "classify") == "detect" else "classify")
    img_size = int(metadata.get("imgsz", 640 if task == "detect" else 224))
    labels_history: deque[str | None] = deque(maxlen=7)
    last_spoken: str | None = None
    current_stable: str | None = None
    speech = None if args.no_voice else SpeechWorker()

    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        if speech:
            speech.close()
        print(f"Não consegui abrir a câmera {args.camera}. Tente `--camera 1` ou confira a permissão da webcam.")
        return 2

    print("Webcam iniciada. Pressione Q ou ESC para sair; R repete a última letra estável.")
    print("Dica: deixe uma mão visível, centralizada e bem iluminada.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Não consegui ler um quadro da câmera.")
                break
            frame = cv2.flip(frame, 1)

            try:
                result = model.predict(frame, imgsz=img_size, verbose=False)[0]
                if task == "detect":
                    draw_detection_boxes(frame, result, args.conf)
                label, confidence = get_prediction(result, task)
            except Exception as exc:
                cv2.putText(frame, f"Erro de previsao: {str(exc)[:45]}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                label, confidence = None, 0.0

            if label is not None and confidence >= args.conf:
                labels_history.append(label)
            else:
                labels_history.append(None)

            valid_labels = [item for item in labels_history if item is not None]
            stable_label = None
            if valid_labels:
                counts = Counter(valid_labels)
                candidate, count = counts.most_common(1)[0]
                if count >= 5:
                    stable_label = candidate

            if stable_label != current_stable:
                current_stable = stable_label
                if current_stable is not None and current_stable != last_spoken:
                    if speech:
                        speech.say(current_stable)
                    last_spoken = current_stable

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
                last_spoken = current_stable
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if speech:
            speech.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
