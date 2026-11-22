"""Converte o dataset de imagens (pastas por letra) em pontos da mão do MediaPipe.

Uso: python extrair_landmarks.py --dataset workspace/extraido
Gera workspace/landmarks_<nome>.npz com os pontos crus, letra, split e arquivo de origem.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from landmarks import ensure_hand_model
from treinar import WORK_DIR, class_name_from_folder, is_image

SPLIT_NAMES = {"train": "train", "training": "train", "val": "val", "valid": "val",
               "validation": "val", "test": "test", "testing": "test"}

_tracker = None
_upscale = 256


def _init_worker(upscale: int) -> None:
    global _tracker, _upscale
    os.environ.setdefault("GLOG_minloglevel", "2")
    from landmarks import HandTracker
    _tracker = HandTracker(video=False, min_confidence=0.3)
    _upscale = upscale


def _extract(path: str) -> tuple[str, np.ndarray | None]:
    import cv2

    image = cv2.imread(path)
    if image is None:
        return path, None
    height, width = image.shape[:2]
    if max(height, width) < _upscale:
        # O detector do MediaPipe acha bem menos mãos em imagens muito pequenas (64x64).
        factor = _upscale / max(height, width)
        image = cv2.resize(image, (round(width * factor), round(height * factor)), interpolation=cv2.INTER_CUBIC)
    return path, _tracker.detect(image)


def list_images(root: Path) -> list[tuple[Path, str, str]]:
    """Retorna (arquivo, letra, split). Aceita root/<split>/<LETRA>/ ou root/<LETRA>/."""
    splits = [(d, SPLIT_NAMES[d.name.lower()]) for d in sorted(root.iterdir())
              if d.is_dir() and d.name.lower() in SPLIT_NAMES]
    if not splits:
        splits = [(root, "train")]
    found = []
    for split_dir, split in splits:
        for class_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
            label = class_name_from_folder(class_dir.name)
            if label is None:
                continue
            found += [(p, label, split) for p in sorted(class_dir.iterdir()) if is_image(p)]
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="Extrai os pontos da mão de um dataset de imagens por letra.")
    parser.add_argument("--dataset", default="workspace/extraido", help="Pasta com train/ e test/ (ou só as letras).")
    parser.add_argument("--out", help="Arquivo .npz de saída (padrão: workspace/landmarks_<pasta>.npz).")
    parser.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--upscale", type=int, default=256, help="Amplia imagens menores que isso antes de detectar.")
    parser.add_argument("--force", action="store_true", help="Refaz mesmo se o arquivo de saída existir.")
    args = parser.parse_args()

    root = Path(args.dataset).expanduser().resolve()
    if not root.is_dir():
        print(f"Pasta não encontrada: {root}")
        return 2
    out = Path(args.out) if args.out else WORK_DIR / f"landmarks_{root.name}.npz"
    if out.is_file() and not args.force:
        print(f"Já existe: {out} (use --force para refazer).")
        return 0

    entries = list_images(root)
    if not entries:
        print("Nenhuma imagem encontrada em pastas por letra.")
        return 2

    # Duplicatas exatas viram o mesmo ponto; removê-las evita vazamento entre treino e teste.
    seen: dict[str, str] = {}
    unique = []
    for path, label, split in entries:
        digest = hashlib.sha1(path.read_bytes()).hexdigest()
        if digest in seen:
            if seen[digest] != label:
                print(f"Aviso: imagem idêntica com letras diferentes ignorada: {path}")
            continue
        seen[digest] = label
        unique.append((path, label, split))
    print(f"{len(entries)} imagens, {len(entries) - len(unique)} duplicatas exatas removidas.")

    ensure_hand_model()
    info = {str(path): (label, split) for path, label, split in unique}
    points, labels, splits, sources = [], [], [], []
    missed: Counter = Counter()
    total: Counter = Counter()
    with Pool(args.workers, initializer=_init_worker, initargs=(args.upscale,)) as pool:
        for index, (path, result) in enumerate(pool.imap_unordered(_extract, list(info), chunksize=64), 1):
            label, split = info[path]
            total[label] += 1
            if result is None:
                missed[label] += 1
            else:
                points.append(result)
                labels.append(label)
                splits.append(split)
                sources.append(str(Path(path).relative_to(root)))
            if index % 2000 == 0 or index == len(info):
                print(f"  {index}/{len(info)} imagens processadas", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, points=np.stack(points), labels=np.array(labels), splits=np.array(splits),
                        sources=np.array(sources))
    report = {label: {"imagens": total[label], "sem_mao": missed[label],
                      "deteccao": round(1 - missed[label] / total[label], 3)} for label in sorted(total)}
    out.with_suffix(".json").write_text(json.dumps(
        {"dataset": str(root), "amostras": len(points), "por_letra": report,
         "por_split": dict(Counter(splits))}, ensure_ascii=False, indent=2), encoding="utf-8")
    detected = len(points) / max(1, sum(total.values()))
    print(f"\n{len(points)} mãos encontradas ({detected:.1%}). Salvo em {out}")
    worst = sorted(report.items(), key=lambda item: item[1]["deteccao"])[:3]
    print("Letras com menor detecção: " + ", ".join(f"{k} {v['deteccao']:.0%}" for k, v in worst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
