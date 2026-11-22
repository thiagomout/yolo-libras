"""Treina os classificadores sobre pontos da mão.

- Estático (MLP): pontos extraídos do dataset de imagens + gravações estáticas de coleta/.
- Dinâmico (GRU): gravações de coleta/dinamico (H, J, K, X, Z, Ç e NADA).

Uso: python treinar_landmarks.py [--pessoa-teste ana]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from landmarks import COLLECTION_DIR, NOTHING_LABEL, SEQUENCE_LENGTH, MotionSegmenter, longest_motion, normalize_hand, resample, \
    sequence_features
from modelos_mao import DEFAULT_BUNDLE, SequenceGRU, StaticMLP, augment_points, save_bundle, time_warp

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_STATIC = PROJECT_DIR / "workspace" / "landmarks_extraido.npz"
DEFAULT_COLLECTED = COLLECTION_DIR
MOTION_CONFIG = {"start": 1.5, "still": 0.2, "window": 0.1, "pre_roll": 0.15, "min_duration": 0.25,
                 "max_duration": 3.0}


def load_recordings(folder: Path) -> list[dict]:
    """Lê as gravações; arquivos corrompidos ou incompletos são ignorados com aviso."""
    recordings = []
    for path in sorted(folder.glob("*/*.npz")):
        try:
            with np.load(path, allow_pickle=False) as data:
                recording = {"points": data["points"], "times": data["times"], "label": str(data["label"]),
                             "person": str(data["person"]), "path": path}
        except Exception as exc:  # zip incompleto, chave ausente etc.
            print(f"Aviso: ignorando {path.name}: {exc}")
            continue
        if len(recording["points"]) >= 2 and len(recording["points"]) == len(recording["times"]):
            recordings.append(recording)
    return recordings


def split_recordings(recordings: list[dict], test_person: str | None, val_fraction: float,
                     rng: random.Random) -> dict[str, list[dict]]:
    """Pessoa de teste nunca aparece no treino. O resto é dividido por gravação, por letra."""
    splits = {"train": [], "val": [], "test": []}
    by_label = defaultdict(list)
    for recording in recordings:
        if test_person and recording["person"] == test_person:
            splits["test"].append(recording)
        else:
            by_label[recording["label"]].append(recording)
    for items in by_label.values():
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_fraction)) if len(items) >= 4 else 0
        splits["val"] += items[:n_val]
        splits["train"] += items[n_val:]
    return splits


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device, batch: int = 2048) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            predictions.append(model(x[start:start + batch].to(device)).argmax(1).cpu())
    return torch.cat(predictions).numpy() if predictions else np.array([], dtype=int)


def report(name: str, predicted: np.ndarray, truth: np.ndarray, classes: list[str]) -> dict | None:
    if len(truth) == 0:
        return None
    accuracy = float((predicted == truth).mean())
    per_class = {}
    confusions = Counter()
    for index, label in enumerate(classes):
        mask = truth == index
        if mask.any():
            per_class[label] = round(float((predicted[mask] == index).mean()), 3)
    for p, t in zip(predicted, truth):
        if p != t:
            confusions[f"{classes[t]}->{classes[p]}"] += 1
    worst = sorted(per_class.items(), key=lambda item: item[1])[:5]
    print(f"  {name}: acurácia {accuracy:.1%} em {len(truth)} exemplos | piores: "
          + ", ".join(f"{k} {v:.0%}" for k, v in worst))
    if confusions:
        print("    confusões mais comuns: " + ", ".join(f"{k} ({v})" for k, v in confusions.most_common(5)))
    return {"acuracia": round(accuracy, 4), "exemplos": int(len(truth)), "por_letra": per_class,
            "confusoes": dict(confusions.most_common(10))}


def train_loop(model, make_batch, x_val, y_val, epochs, steps, device, lr, class_weights, patience):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs * steps)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
    best_state, best_accuracy, stale = None, -1.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for _ in range(steps):
            xb, yb = make_batch()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item()
        accuracy = float((evaluate(model, x_val, y_val, device) == y_val.numpy()).mean()) if len(y_val) else 0.0
        if accuracy > best_accuracy:
            best_accuracy, stale = accuracy, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  época {epoch:3d}  perda {total / steps:.3f}  val {accuracy:.1%}")
        if len(y_val) and stale >= patience:
            print(f"  parada antecipada na época {epoch} (melhor val {best_accuracy:.1%})")
            break
    if best_state is None or not len(y_val):
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best_state


def class_weights(y: np.ndarray, classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1) / classes
    return torch.tensor(np.sqrt(weights), dtype=torch.float32)  # raiz: compensa sem exagerar


def train_static(args, rng, generator, device, recordings) -> tuple[dict | None, dict]:
    points, labels, splits = [], [], []
    if args.static.is_file():
        with np.load(args.static) as data:
            points.append(data["points"])
            labels += data["labels"].tolist()
            ds_splits = data["splits"].tolist()
        # O dataset vem de poucos vídeos: a validação sai daqui só para parada antecipada.
        splits += ["dataset_" + ("val" if s == "train" and rng.random() < args.val_fraction else s) for s in ds_splits]
        print(f"Dataset de pontos: {len(ds_splits)} exemplos de {args.static}")
    else:
        print(f"Aviso: {args.static} não existe; rode extrair_landmarks.py para usar o dataset de imagens.")

    for split, items in split_recordings(recordings, args.test_person, args.val_fraction, rng).items():
        for recording in items:
            frames = recording["points"][::args.frame_stride]
            points.append(frames)
            labels += [recording["label"]] * len(frames)
            splits += [f"coleta_{split}"] * len(frames)
    if recordings:
        print(f"Coleta estática: {len(recordings)} gravações de {len({r['person'] for r in recordings})} pessoa(s)")
    if not labels:
        return None, {}

    all_points = np.concatenate(points)
    classes = sorted(set(labels))
    index = {label: i for i, label in enumerate(classes)}
    x = torch.from_numpy(np.stack([normalize_hand(p) for p in all_points]))  # (N, 21, 3)
    y = torch.tensor([index[label] for label in labels])
    splits = np.array(splits)
    train_mask = np.isin(splits, ["dataset_train", "coleta_train"])
    val_mask = np.isin(splits, ["dataset_val", "coleta_val"])
    x_train, y_train = x[train_mask], y[train_mask]
    x_val, y_val = x[val_mask].reshape(-1, 63), y[val_mask]
    print(f"Estático: {len(classes)} letras, treino {len(y_train)}, val {len(y_val)}")

    # Amostragem balanceada entre dataset e coleta: a coleta é pequena, mas representa a sua webcam.
    from_collected = torch.from_numpy(splits[train_mask] == "coleta_train")
    weights = torch.where(from_collected, 1.0 / max(1, int(from_collected.sum())),
                          1.0 / max(1, int((~from_collected).sum())))
    share = args.collected_share if from_collected.any() and (~from_collected).any() else None
    if share is not None:
        weights = torch.where(from_collected, weights * share, weights * (1 - share))

    def make_batch():
        chosen = torch.multinomial(weights, args.batch, replacement=True, generator=generator)
        return augment_points(x_train[chosen], generator).reshape(args.batch, -1), y_train[chosen]

    model = StaticMLP(len(classes))
    steps = max(20, min(400, len(y_train) // args.batch))
    print("Treinando classificador estático...")
    state = train_loop(model, make_batch, x_val, y_val, args.epochs, steps, device, 3e-3,
                       class_weights(y_train.numpy(), len(classes)), patience=8)
    metrics = {}
    for name in ("dataset_test", "coleta_val", "coleta_test"):
        mask = splits == name
        if mask.any():
            predicted = evaluate(model, x[mask].reshape(-1, 63), y[mask], device)
            metrics[name] = report(name, predicted, y[mask].numpy(), classes)
    return {"classes": classes, "state": state}, metrics


def train_dynamic(args, rng, generator, device, recordings) -> tuple[dict | None, dict]:
    labels = sorted({r["label"] for r in recordings})
    if len(labels) < 2:
        print("Dinâmico: precisa de gravações de pelo menos 2 letras em coleta/dinamico; pulando.")
        return None, {}
    segmenter = MotionSegmenter(**MOTION_CONFIG)
    untrimmed = 0
    for recording in recordings:
        segment = longest_motion(recording["points"], recording["times"], segmenter)
        if segment is None or len(segment) < 4:
            untrimmed += 1
            segment = recording["points"]
        recording["features"] = sequence_features(segment).reshape(len(segment), -1)
    if untrimmed:
        print(f"Aviso: em {untrimmed} gravações não achei movimento acima do limiar; usei a gravação inteira.")

    index = {label: i for i, label in enumerate(labels)}
    splits = split_recordings(recordings, args.test_person, args.val_fraction, rng)
    train = splits["train"]
    print(f"Dinâmico: {len(labels)} classes, treino {len(train)}, val {len(splits['val'])}, "
          f"teste {len(splits['test'])} | por classe: {dict(Counter(r['label'] for r in recordings))}")

    def fixed(items):
        if not items:
            return torch.zeros(0, SEQUENCE_LENGTH, 66), torch.zeros(0, dtype=torch.long)
        return (torch.from_numpy(np.stack([resample(r["features"]) for r in items])),
                torch.tensor([index[r["label"]] for r in items]))

    y_all = np.array([index[r["label"]] for r in train])
    np_rng = np.random.default_rng(args.seed)

    def make_batch():
        chosen = np_rng.integers(0, len(train), args.seq_batch)
        sequences = np.stack([time_warp(train[i]["features"], np_rng) for i in chosen])
        xb = torch.from_numpy(sequences).reshape(args.seq_batch, SEQUENCE_LENGTH, 22, 3)
        return augment_points(xb, generator, noise=0.02).reshape(args.seq_batch, SEQUENCE_LENGTH, -1), \
            torch.from_numpy(y_all[chosen])

    x_val, y_val = fixed(splits["val"])
    model = SequenceGRU(len(labels))
    print("Treinando classificador de movimento...")
    state = train_loop(model, make_batch, x_val, y_val, args.seq_epochs, max(10, len(train) // args.seq_batch * 2),
                       device, 2e-3, class_weights(y_all, len(labels)), patience=25)
    metrics = {}
    for name in ("val", "test"):
        x_eval, y_eval = fixed(splits[name])
        if len(y_eval):
            metrics[f"coleta_{name}"] = report(f"coleta_{name}", evaluate(model, x_eval, y_eval, device),
                                               y_eval.numpy(), labels)
    return {"classes": labels, "state": state}, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Treina classificadores de Libras sobre os pontos da mão.")
    parser.add_argument("--static", type=Path, default=DEFAULT_STATIC, help="Saída do extrair_landmarks.py.")
    parser.add_argument("--coleta", type=Path, default=DEFAULT_COLLECTED,
                        help="Pasta do coletar.py (padrão: variável LIBRAS_COLETA ou ./coleta).")
    parser.add_argument("--pessoa-teste", dest="test_person", help="Pessoa usada só no teste (mede generalização).")
    parser.add_argument("--por-pessoa", action="store_true",
                        help="Avalia deixando cada pessoa de fora por vez e depois treina o modelo final com todos.")
    parser.add_argument("--out", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seq-epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seq-batch", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--frame-stride", type=int, default=2, help="Usa 1 a cada N quadros das gravações estáticas.")
    parser.add_argument("--collected-share", type=float, default=0.3,
                        help="Fração de cada batch vinda da sua coleta, quando há coleta e dataset.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.test_person:
        args.test_person = args.test_person.strip().lower()
    print(f"Pasta da coleta: {args.coleta}")
    static_recordings = load_recordings(args.coleta / "estatico") if (args.coleta / "estatico").is_dir() else []
    dynamic_recordings = load_recordings(args.coleta / "dinamico") if (args.coleta / "dinamico").is_dir() else []
    people = sorted({r["person"] for r in static_recordings + dynamic_recordings})
    if people:
        print(f"Pessoas na coleta: {', '.join(people)}")
    if args.test_person and args.test_person not in people:
        print(f"Aviso: --pessoa-teste {args.test_person} não tem gravações.")

    if args.por_pessoa:
        if len(people) < 2:
            print("--por-pessoa precisa de gravações de pelo menos 2 pessoas.")
            return 2
        table = {}
        for person in people:
            print(f"\n===== Testando com {person} fora do treino =====")
            args.test_person = person
            _, static_metrics, dynamic_metrics = train_all(args, static_recordings, dynamic_recordings)
            table[person] = ((static_metrics.get("coleta_test") or {}).get("acuracia"),
                             (dynamic_metrics.get("coleta_test") or {}).get("acuracia"))
        print("\n===== Generalização para pessoas novas (cada uma fora do treino) =====")
        print(f"{'pessoa':15} {'estáticas':>10} {'movimento':>10}")
        for person, (static_accuracy, dynamic_accuracy) in table.items():
            show = lambda value: "-" if value is None else f"{value:.0%}"
            print(f"{person:15} {show(static_accuracy):>10} {show(dynamic_accuracy):>10}")
        print("\nTreinando o modelo final com todas as pessoas...")
        args.test_person = None

    (static, dynamic), static_metrics, dynamic_metrics = train_all(args, static_recordings, dynamic_recordings)
    if static is None and dynamic is None:
        print("Nada para treinar: rode extrair_landmarks.py e/ou coletar.py antes.")
        return 2

    config = {"sequence_length": SEQUENCE_LENGTH, "motion": MOTION_CONFIG, "nothing_label": NOTHING_LABEL}
    save_bundle(args.out, static, dynamic, config)
    metadata = {
        "estatico": {"classes": static["classes"] if static else [], "metricas": static_metrics},
        "dinamico": {"classes": dynamic["classes"] if dynamic else [], "metricas": dynamic_metrics},
        "pessoas": people, "pessoa_teste": args.test_person, "config": config, "seed": args.seed,
    }
    args.out.with_name(args.out.stem + "_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nModelo salvo em {args.out}. Próximo passo (Windows): iniciar_windows.ps1")
    return 0


def train_all(args, static_recordings, dynamic_recordings):
    """Treina os dois modelos com a mesma semente, para comparações entre rodadas serem justas."""
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    device = torch.device(args.device)
    static, static_metrics = train_static(args, rng, generator, device, static_recordings)
    dynamic, dynamic_metrics = train_dynamic(args, rng, generator, device, dynamic_recordings)
    return (static, dynamic), static_metrics, dynamic_metrics


if __name__ == "__main__":
    raise SystemExit(main())
