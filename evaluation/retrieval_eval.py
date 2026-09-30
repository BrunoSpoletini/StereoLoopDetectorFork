"""Calidad de la recuperacion de candidatos (sin verificacion): recall@K sobre los frames revisita GT.

Para cada query revisita i la base son los frames con al menos S_M metros de camino detras
(misma exclusion por distancia que el detector con odometria). Acierto @K si alguno de los K
mas similares esta a < R_M metros.

  python retrieval_eval.py salad --sessions fieldsafe
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from datasets import CACHE, select
from sld_eval import R_M, S_M, load_gt

KS = (1, 5, 10, 25)


def recall_at_k(desc, gt, ks=KS, stride=1):
    desc = desc.astype(np.float32)
    hits = {k: 0 for k in ks}
    queries = np.where(gt.revisit)[0][::stride]
    for i in queries:
        n_db = np.searchsorted(gt.cum, gt.cum[i] - S_M, side='right')
        if n_db == 0:
            continue
        sim = desc[:n_db] @ desc[i]
        top = np.argsort(-sim)[:max(ks)]
        ok = np.linalg.norm(gt.xy[top] - gt.xy[i], axis=1) < R_M
        for k in ks:
            hits[k] += ok[:k].any()
    return {k: hits[k] / len(queries) for k in ks}, len(queries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--sessions', default='fieldsafe')
    ap.add_argument('--stride', type=int, default=5)
    a = ap.parse_args()
    tot, n_tot = {k: 0.0 for k in KS}, 0
    for s in select(a.sessions):
        f = CACHE / f'{s.dataset}_{s.seq}_{a.model}.npy'
        if not f.exists():
            continue
        gt = load_gt(s.poses)
        r, n = recall_at_k(np.load(f), gt, stride=a.stride)
        print(f'{s.name:14s} n={n:5d} ' + ' '.join(f'R@{k}={v:.3f}' for k, v in r.items()))
        for k in KS:
            tot[k] += r[k] * n
        n_tot += n
    print(f'{"TOTAL":14s} n={n_tot:5d} ' + ' '.join(f'R@{k}={tot[k] / n_tot:.3f}' for k in KS))


if __name__ == '__main__':
    main()
