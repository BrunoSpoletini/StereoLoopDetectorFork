"""Reduce los descriptores globales con PCA y los exporta en binario para demo_stereo.

La PCA se ajusta SOLO con las sesiones de FieldSAFE (train) y se aplica igual a RosarioV2.
  python export_global.py salad --dim 256
Salida: <CACHE>/<dataset>_<seq>_<model>_pca<dim>.f32  (int32 n, int32 d, float32 n*d, filas L2-normalizadas)
        resources/<model>_pca<dim>_fieldsafe.npz        (media y componentes de la PCA)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from datasets import CACHE, REPO, select


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--dim', type=int, default=256)
    a = ap.parse_args()

    pca_file = REPO / 'resources' / f'{a.model}_pca{a.dim}_fieldsafe.npz'
    if pca_file.exists():
        pca = np.load(pca_file)
        mu, comp = pca['mean'], pca['components']
    else:
        train = [np.load(CACHE / f'{s.dataset}_{s.seq}_{a.model}.npy')[::10].astype(np.float32)
                 for s in select('fieldsafe') if (CACHE / f'{s.dataset}_{s.seq}_{a.model}.npy').exists()]
        X = np.concatenate(train)
        mu = X.mean(0)
        _, _, vt = np.linalg.svd(X - mu, full_matrices=False)
        comp = vt[:a.dim]
        np.savez(pca_file, mean=mu, components=comp)
        print(f'PCA ajustada con {len(X)} descriptores de FieldSAFE -> {pca_file.name}')

    for s in select('all'):
        src = CACHE / f'{s.dataset}_{s.seq}_{a.model}.npy'
        if not src.exists():
            continue
        P = (np.load(src).astype(np.float32) - mu) @ comp.T
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        out = CACHE / f'{s.dataset}_{s.seq}_{a.model}_pca{a.dim}.f32'
        with open(out, 'wb') as f:
            np.array(P.shape, dtype=np.int32).tofile(f)
            P.astype(np.float32).tofile(f)
        print(f'{s.name}: {P.shape} -> {out.name}')


if __name__ == '__main__':
    main()
