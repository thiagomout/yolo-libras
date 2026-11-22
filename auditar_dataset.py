"""Busca imagens visualmente semelhantes entre splits; não prova vazamento de pessoas/sessões."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from treinar import is_image


def perceptual_hash(path: Path) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'Imagem ilegível: {path}')
    small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coefficients = cv2.dct(small)[:8, :8].flatten()[1:]
    bits = coefficients > np.median(coefficients)
    return sum(int(bit) << index for index, bit in enumerate(bits))


def audit(root: Path, samples_per_class: int = 10, seed: int = 42) -> dict:
    train_paths = sorted(p for p in (root / 'train').rglob('*') if is_image(p))
    if not train_paths:
        raise ValueError('Esperava train/LETRA/imagem e val/LETRA/imagem ou test/LETRA/imagem.')
    print(f'Calculando hashes visuais de {len(train_paths)} imagens de treino...', flush=True)
    hashes = np.array([perceptual_hash(p) for p in train_paths], dtype=np.uint64)
    # Tabela de contagem de bits funciona também com versões antigas do NumPy.
    bit_counts = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    rng = np.random.default_rng(seed)
    report = {'train_images': len(train_paths), 'seed': seed, 'samples_per_class': samples_per_class,
              'note': 'Hash perceptual é uma triagem; proximidade não comprova duplicata nem sessão compartilhada.', 'splits': {}}
    for split in ('val', 'test'):
        rows = []
        for folder in sorted((root / split).glob('*')):
            if not folder.is_dir():
                continue
            paths = sorted(p for p in folder.iterdir() if is_image(p))
            if not paths:
                continue
            selected = rng.choice(len(paths), min(samples_per_class, len(paths)), replace=False)
            for index in selected:
                path = paths[int(index)]
                differences = np.bitwise_xor(hashes, np.uint64(perceptual_hash(path)))
                distances = bit_counts[differences.view(np.uint8).reshape(-1, 8)].sum(axis=1)
                nearest = int(distances.argmin())
                rows.append({'image': str(path), 'label': folder.name, 'nearest_train': str(train_paths[nearest]),
                             'nearest_label': train_paths[nearest].parent.name, 'distance_63_bits': int(distances[nearest])})
        if rows:
            report['splits'][split] = {'sampled': len(rows),
                                      'distance_at_most_4': sum(r['distance_63_bits'] <= 4 for r in rows),
                                      'nearest_same_label': sum(r['label'] == r['nearest_label'] for r in rows),
                                      'pairs': rows}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, help='Dataset preparado com train, val e/ou test.')
    parser.add_argument('--samples', type=int, default=10, help='Amostras por classe e split.')
    parser.add_argument('--output', default='workspace/auditoria_dataset.json')
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples deve ser positivo')
    report = audit(Path(args.dataset), args.samples)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for split, data in report['splits'].items():
        print(f"{split}: {data['distance_at_most_4']}/{data['sampled']} com hash próximo ao treino (distância <= 4); "
              f"vizinho da mesma classe: {data['nearest_same_label']}/{data['sampled']}.")
    print(f'Relatório: {output}. Inspecione os pares; essa triagem não prova vazamento.')


if __name__ == '__main__':
    main()
