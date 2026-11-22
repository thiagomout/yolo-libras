"""Mostra quantas gravações cada pessoa fez por letra e o que ainda falta.

Uso: python resumo_coleta.py [--coleta PASTA]
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from landmarks import COLLECTION_DIR, DYNAMIC_LETTERS, NOTHING_LABEL

STATIC_LETTERS = tuple("ABCDEFGILMNOPQRSTUVWY")
TARGETS = {"estatico": 15, "dinamico": 30}
NOTHING_TARGET = 40
FILENAME = re.compile(r"^(?P<person>.+)_\d{8}-\d{6}-\d+$")  # formato salvo pelo coletar.py


def count(folder: Path) -> Counter:
    """Conta por (tipo, letra, pessoa) pelo nome do arquivo: <pessoa>_<data>.npz."""
    counts: Counter = Counter()
    for path in folder.glob("*/*/*.npz"):
        match = FILENAME.match(path.stem)
        if match:  # ignora cópias ("arquivo (1).npz") e arquivos estranhos
            counts[(path.parent.parent.name, path.parent.name, match["person"])] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Resumo das gravações de coletar.py.")
    parser.add_argument("--coleta", type=Path, default=COLLECTION_DIR)
    args = parser.parse_args()
    if not args.coleta.is_dir():
        print(f"Pasta não encontrada: {args.coleta}")
        return 2
    counts = count(args.coleta)
    people = sorted({person for _, _, person in counts})
    if not people:
        print(f"Nenhuma gravação em {args.coleta}")
        return 0
    print(f"Pasta: {args.coleta}\nMeta: {TARGETS['estatico']} por letra estática, {TARGETS['dinamico']} por letra "
          f"com movimento, {NOTHING_TARGET} de {NOTHING_LABEL}. '*' = abaixo da meta.\n")
    sections = (("estatico", STATIC_LETTERS), ("dinamico", (*DYNAMIC_LETTERS, NOTHING_LABEL)))
    width = max(8, *(len(p) for p in people))
    for kind, letters in sections:
        extra = sorted({label for k, label, _ in counts if k == kind} - set(letters))
        letters = (*letters, *extra)
        print(("ESTÁTICAS" if kind == "estatico" else "COM MOVIMENTO"))
        print(f"{'letra':6}" + "".join(f"{p:>{width + 2}}" for p in people) + f"{'total':>8}")
        for label in letters:
            target = NOTHING_TARGET if label == NOTHING_LABEL else TARGETS[kind]
            cells = []
            for person in people:
                n = counts[(kind, label, person)]
                cells.append(f"{n}{'*' if n < target else ' '}".rjust(width + 2))
            total = sum(counts[(kind, label, p)] for p in people)
            print(f"{label:6}" + "".join(cells) + f"{total:>8}")
        print()
    for person in people:
        missing = [label for kind, letters in sections for label in letters
                   if counts[(kind, label, person)] < (NOTHING_TARGET if label == NOTHING_LABEL else TARGETS[kind])]
        print(f"{person}: " + ("completo" if not missing else f"falta {len(missing)}: {' '.join(missing)}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
