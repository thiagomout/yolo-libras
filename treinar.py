from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("Dependência ausente: instale com `pip install -r requirements.txt`.")
    raise

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PROJECT_DIR = Path(__file__).resolve().parent
WORK_DIR = PROJECT_DIR / "workspace"
MODEL_DIR = PROJECT_DIR / "model"


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    """Extrai o ZIP bloqueando caminhos que tentem sair da pasta de destino."""
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise ValueError(f"O ZIP contém um caminho inseguro: {member.filename}")
        archive.extractall(destination)


def find_detection_yaml(extracted_root: Path) -> tuple[Path, dict[str, Any]] | None:
    """Procura um YAML de dataset no formato de detecção da Ultralytics."""
    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    for path in extracted_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml"}:
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        if not isinstance(data, dict) or not all(key in data for key in ("train", "val", "names")):
            continue
        names = data.get("names")
        if not isinstance(names, (dict, list)) or not names:
            continue
        score = 10
        if path.name.lower() in {"data.yaml", "data.yml", "dataset.yaml", "dataset.yml"}:
            score += 5
        if any(p.name.lower() == "labels" for p in path.parent.rglob("labels")):
            score += 2
        candidates.append((score, path, data))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], len(str(item[1]))), reverse=True)
    _, yaml_path, data = candidates[0]
    return yaml_path, data


def names_to_list(names: dict | list) -> list[str]:
    if isinstance(names, list):
        return [str(x) for x in names]
    pairs = []
    for key, value in names.items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        pairs.append((idx, str(value)))
    pairs.sort(key=lambda pair: pair[0])
    return [name for _, name in pairs]


def resolve_existing_path(value: Any, root_hint: Path, yaml_parent: Path) -> Path | None:
    if not isinstance(value, str):
        return None
    raw = Path(value).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([root_hint / raw, yaml_parent / raw, Path.cwd() / raw])
        # Exportações podem deixar caminhos relativos ao diretório do projeto original.
        candidates.extend(parent / raw for parent in yaml_parent.parents)
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate.resolve()
        except OSError:
            pass
    return None


def prepare_detection_yaml(
    extracted_root: Path,
    yaml_path: Path,
    data: dict[str, Any],
    exclude_motion: bool,
) -> tuple[Path, list[str]]:
    names = names_to_list(data["names"])
    if not names:
        raise ValueError("Não consegui ler a lista de classes do YAML de detecção.")

    # Tenta encontrar a raiz configurada no próprio YAML e, depois, os caminhos dos splits.
    root_hint = yaml_path.parent
    configured_root = data.get("path")
    if isinstance(configured_root, str):
        configured = Path(configured_root).expanduser()
        root_candidates = []
        if configured.is_absolute():
            root_candidates.append(configured)
        else:
            root_candidates.extend([yaml_path.parent / configured, Path.cwd() / configured])
        for candidate in root_candidates:
            if candidate.exists():
                root_hint = candidate.resolve()
                break

    train_path = resolve_existing_path(data.get("train"), root_hint, yaml_path.parent)
    val_path = resolve_existing_path(data.get("val"), root_hint, yaml_path.parent)
    if train_path is None or val_path is None:
        raise FileNotFoundError(
            "Encontrei um YAML de detecção, mas não localizei os caminhos de treino/validação. "
            "Confira `path`, `train` e `val` no YAML do dataset."
        )

    # Para esta primeira etapa, J e Z são omitidas por serem sinais com movimento.
    excluded_ids: set[int] = set()
    if exclude_motion:
        excluded_ids = {i for i, name in enumerate(names) if name.strip().upper() in {"J", "Z"}}

    if excluded_ids:
        kept_old_ids = [i for i in range(len(names)) if i not in excluded_ids]
        remap = {old: new for new, old in enumerate(kept_old_ids)}
        # Edita apenas arquivos .txt dentro de pastas chamadas labels.
        for labels_dir in extracted_root.rglob("labels"):
            if not labels_dir.is_dir():
                continue
            for label_file in labels_dir.rglob("*.txt"):
                rewritten: list[str] = []
                try:
                    lines = label_file.read_text(encoding="utf-8", errors="ignore").splitlines()
                except OSError:
                    continue
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        old_id = int(float(parts[0]))
                    except ValueError:
                        rewritten.append(line)
                        continue
                    if old_id in excluded_ids:
                        continue
                    if old_id in remap:
                        parts[0] = str(remap[old_id])
                        rewritten.append(" ".join(parts))
                label_file.write_text("\n".join(rewritten) + ("\n" if rewritten else ""), encoding="utf-8")
        names = [names[i] for i in kept_old_ids]
        print("Classes de movimento removidas do dataset de detecção: J e/ou Z.")

    new_data = dict(data)
    new_data["path"] = str(extracted_root.resolve())
    new_data["train"] = str(train_path)
    new_data["val"] = str(val_path)
    new_data["names"] = {index: name for index, name in enumerate(names)}
    new_data["nc"] = len(names)

    # O test é opcional; se existir e o caminho for válido, mantemos seu caminho absoluto.
    if data.get("test"):
        test_path = resolve_existing_path(data.get("test"), root_hint, yaml_path.parent)
        if test_path is not None:
            new_data["test"] = str(test_path)

    prepared_yaml = WORK_DIR / "data_preparado.yaml"
    prepared_yaml.write_text(yaml.safe_dump(new_data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return prepared_yaml, names


def class_name_from_folder(folder_name: str) -> str | None:
    """Reconhece nomes como A, a, A_augmented, letter_A ou class-A."""
    name = folder_name.strip().upper()
    patterns = [
        r"^([A-Z])(?:$|[\s_(-].*)",
        r"^(?:LETTER|CLASS)[\s_-]*([A-Z])(?:$|[\s_(-].*)",
    ]
    for pattern in patterns:
        match = re.match(pattern, name)
        if match:
            return match.group(1)
    return None


def find_class_parent(extracted_root: Path) -> Path | None:
    """Encontra pastas em que cada subpasta representa uma classe e contém imagens."""
    candidates: list[tuple[int, int, int, Path]] = []
    ignored_names = {".git", "__macosx", "images", "labels", "annotations"}
    for parent in extracted_root.rglob("*"):
        if not parent.is_dir() or parent.name.lower() in ignored_names:
            continue
        subdirs = [d for d in parent.iterdir() if d.is_dir() and d.name.lower() not in ignored_names]
        class_dirs = [d for d in subdirs if any(is_image(f) for f in d.iterdir() if f.is_file())]
        # O alfabeto tem muitas classes; isso evita interpretar pastas genéricas como classes.
        letter_dirs = [d for d in class_dirs if class_name_from_folder(d.name)]
        if len(letter_dirs) < 10:
            continue
        total_images = sum(sum(1 for f in d.iterdir() if is_image(f)) for d in letter_dirs)
        train_bonus = 1 if parent.name.lower() in {"train", "training", "asl_alphabet_train"} else 0
        candidates.append((len(letter_dirs), total_images, train_bonus, parent))
    if not candidates:
        return None
    # Prioriza alfabeto com mais classes; desempata pela quantidade de imagens e pelo nome da pasta.
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return candidates[0][3]


def prepare_classification_dataset(
    class_parent: Path,
    output_dir: Path,
    exclude_motion: bool,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> list[str]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    rng = random.Random(seed)
    classes: dict[str, list[Path]] = defaultdict(list)
    for class_dir in sorted(d for d in class_parent.iterdir() if d.is_dir()):
        letter = class_name_from_folder(class_dir.name)
        if letter is None:
            continue
        if exclude_motion and letter in {"J", "Z"}:
            continue
        for image_path in sorted(class_dir.iterdir()):
            if is_image(image_path):
                classes[letter].append(image_path)

    classes = {label: files for label, files in classes.items() if files}
    if len(classes) < 10:
        raise ValueError(
            "Não consegui identificar pelo menos 10 classes de letras nas pastas. "
            "Este ZIP pode usar um formato diferente do esperado."
        )
    too_small = [label for label, files in classes.items() if len(files) < 2]
    if too_small:
        raise ValueError(
            "Cada letra precisa ter pelo menos 2 imagens para separar treino e validação. "
            f"Poucas imagens para: {', '.join(sorted(too_small))}."
        )

    print(f"Pasta de classes encontrada: {class_parent}")
    print(f"Classes usadas ({len(classes)}): {', '.join(sorted(classes))}")
    if exclude_motion:
        print("J e Z serão ignoradas nesta primeira versão por envolverem movimento.")

    for split in ("train", "val"):
        for label in classes:
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)

    for label, files in sorted(classes.items()):
        shuffled = files[:]
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * val_fraction)))
        val_count = min(val_count, len(shuffled) - 1)
        val_files = shuffled[:val_count]
        train_files = shuffled[val_count:]
        for split, split_files in (("train", train_files), ("val", val_files)):
            target_dir = output_dir / split / label
            for index, source in enumerate(split_files):
                target = target_dir / f"{label}_{index:06d}{source.suffix.lower()}"
                shutil.copy2(source, target)
    return sorted(classes)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Treina um YOLO para reconhecer letras estáticas de Libras a partir de um ZIP."
    )
    parser.add_argument("--zip", default="dataset.zip", help="Caminho do ZIP com o dataset (padrão: dataset.zip).")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas (padrão: 50).")
    parser.add_argument("--batch", type=int, default=16, help="Tamanho do batch (padrão: 16).")
    parser.add_argument("--include-j-z", action="store_true", help="Mantém J e Z, apesar de envolverem movimento.")
    parser.add_argument("--imgsz", type=int, default=None, help="Tamanho da imagem; padrão 224 para classificação e 640 para detecção.")
    args = parser.parse_args()

    zip_path = Path(args.zip).expanduser().resolve()
    if not zip_path.is_file():
        print(f"Não encontrei o ZIP: {zip_path}")
        print("Coloque o arquivo na pasta deste projeto ou informe o caminho: python treinar.py --zip caminho\\dataset.zip")
        return 2
    if args.epochs < 1 or args.batch < 1:
        print("--epochs e --batch precisam ser maiores que zero.")
        return 2

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    extracted_root = WORK_DIR / "extraido"
    extracted_root.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Extraindo {zip_path.name}...")
        safe_extract_zip(zip_path, extracted_root)
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        print(f"Erro ao extrair o ZIP: {exc}")
        return 2

    try:
        from ultralytics import YOLO
    except ImportError:
        print("A Ultralytics não está instalada. Execute: pip install -r requirements.txt")
        return 2

    exclude_motion = not args.include_j_z
    detection = find_detection_yaml(extracted_root)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        if detection:
            yaml_path, yaml_data = detection
            print(f"Dataset de detecção YOLO encontrado: {yaml_path.relative_to(extracted_root)}")
            prepared_yaml, classes = prepare_detection_yaml(
                extracted_root, yaml_path, yaml_data, exclude_motion=exclude_motion
            )
            image_size = args.imgsz or 640
            model = YOLO("yolo26n.pt")
            model.train(
                data=str(prepared_yaml),
                epochs=args.epochs,
                imgsz=image_size,
                batch=args.batch,
                project=str(PROJECT_DIR / "runs"),
                name="libras_deteccao",
                workers=4,
                plots=True,
            )
            task = "detect"
        else:
            class_parent = find_class_parent(extracted_root)
            if class_parent is None:
                print(
                    "Não consegui identificar o formato do ZIP. Eu esperava pastas por letra "
                    "(A/imagem.jpg, B/imagem.jpg, ...) ou um dataset de detecção YOLO com data.yaml.\n"
                    "Anexe o ZIP à conversa para eu adaptar o preparador ao formato real do seu dataset."
                )
                return 2
            class_dataset = WORK_DIR / "dataset_classificacao"
            classes = prepare_classification_dataset(
                class_parent, class_dataset, exclude_motion=exclude_motion
            )
            image_size = args.imgsz or 224
            model = YOLO("yolo26n-cls.pt")
            model.train(
                data=str(class_dataset),
                epochs=args.epochs,
                imgsz=image_size,
                batch=args.batch,
                project=str(PROJECT_DIR / "runs"),
                name="libras_classificacao",
                workers=4,
                plots=True,
            )
            task = "classify"

        trainer = getattr(model, "trainer", None)
        best = Path(trainer.best) if trainer is not None and getattr(trainer, "best", None) else None
        if best is None or not best.is_file():
            save_dir = Path(trainer.save_dir) if trainer is not None else PROJECT_DIR / "runs"
            best = save_dir / "weights" / "best.pt"
        if not best.is_file():
            raise FileNotFoundError(f"Treino terminou, mas não encontrei os pesos best.pt em {best}")

        target_model = MODEL_DIR / "best.pt"
        shutil.copy2(best, target_model)
        metadata = {
            "task": task,
            "classes": classes,
            "imgsz": image_size,
            "exclude_j_z": exclude_motion,
            "source_zip": zip_path.name,
        }
        (MODEL_DIR / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print("\nTreinamento concluído!")
        print(f"Modelo salvo em: {target_model}")
        print(f"Metadados salvos em: {MODEL_DIR / 'metadata.json'}")
        print("Próximo passo: python webcam.py")
        return 0
    except Exception as exc:
        print(f"\nOcorreu um erro durante a preparação ou o treinamento: {exc}")
        print("Confira se o ZIP contém imagens rotuladas e se há imagens suficientes por letra.")
        print("Se o erro continuar, envie o ZIP ou a mensagem completa do erro para ajustar o código.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
