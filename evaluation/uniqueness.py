"""Test de unicidad del lugar para loops aceptados (contra el aliasing entre surcos).

Para cada loop aceptado (q, m) busca los mejores candidatos SALAD de *otras pasadas* (frames con al menos
20 m de odometria detras de q y a mas de --gap frames del match; un candidato por ventana de --gap frames) y los
verifica con ALIKED + LightGlue igual que el match. Si el lugar es unico, ninguna alternativa deberia verificar
con un soporte comparable al del match. Guarda <out>/<sesion>_unique.csv con el soporte del match y de la mejor
alternativa (y, solo para evaluar, la distancia GT de cada uno a la query).

  lg_venv/bin/python uniqueness.py p6_odo --vo-run p3c5vo --sessions rof_1226_1339 --out p7_unique
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from datasets import CACHE, select
from learned_verify import RUNS, Features, calib, verify
from sld_eval import load_gt, load_run


def load_desc(s):
    f = CACHE / f'{s.dataset}_{s.seq}_salad_pca256.f32'
    with open(f, 'rb') as fh:
        n, d = np.fromfile(fh, np.int32, 2)
        return np.fromfile(fh, np.float32, n * d).reshape(n, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('loops_run', help='corrida con los loops aceptados (p. ej. p6_odo)')
    ap.add_argument('--vo-run', default='p3c5vo')
    ap.add_argument('--sessions', required=True)
    ap.add_argument('--out', default='p7_unique')
    ap.add_argument('--k', type=int, default=4, help='alternativas a verificar por loop')
    ap.add_argument('--gap', type=int, default=75, help='frames que separan pasadas distintas (5 s a 15 Hz)')
    ap.add_argument('--max-tp', type=int, default=0, help='submuestrear loops (para medir rapido); 0 = todos')
    a = ap.parse_args()
    out = RUNS / a.out
    out.mkdir(parents=True, exist_ok=True)
    fx = Features()
    for s in select(a.sessions):
        res, q, m, t, r = load_run(RUNS / a.loops_run / f'{s.name}_results.yml')
        if a.max_tp and len(q) > a.max_tp:
            keep = np.linspace(0, len(q) - 1, a.max_tp).astype(int)
            q, m = q[keep], m[keep]
        vo = pd.read_csv(RUNS / a.vo_run / f'{s.name}_queries.csv').set_index('query').sort_index()
        odo = vo.odometer.values
        D = load_desc(s)
        gt = load_gt(s.poses)
        K, fB = calib(s.calibration)
        L, R = s.left.read_text().split(), s.right.read_text().split()
        rows = []
        for n, (qi, mi) in enumerate(zip(q, m)):
            main = verify(fx, K, fB, dict(old_l=L[mi], old_r=R[mi], cur_l=L[qi]), None)
            n_db = np.searchsorted(odo, odo[qi] - 20.0, side='right')
            sims = D[:n_db] @ D[qi]
            order = np.argsort(-sims)
            alts = []
            for c in order:
                if abs(int(c) - int(mi)) <= a.gap or any(abs(int(c) - x) <= a.gap for x in alts):
                    continue
                alts.append(int(c))
                if len(alts) == a.k:
                    break
            best, best_c = -1, -1
            for c in alts:
                v = verify(fx, K, fB, dict(old_l=L[c], old_r=R[c], cur_l=L[qi]), None)
                sc = v['near'] + v['far'] if v['near'] >= 15 else 0
                if sc > best:
                    best, best_c = sc, c
            rows.append(dict(query=qi, match=mi, main=main['near'] + main['far'], main_near=main['near'],
                             alt=max(best, 0), alt_frame=best_c,
                             gd_match=float(np.linalg.norm(gt.xy[qi] - gt.xy[mi])),
                             gd_alt=float(np.linalg.norm(gt.xy[qi] - gt.xy[best_c])) if best_c >= 0 else np.nan))
            if n % 200 == 0:
                print(f'  {s.name}: {n}/{len(q)}', flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(out / f'{s.name}_unique.csv', index=False)
        print(f'{s.name}: {len(df)} loops evaluados', flush=True)


if __name__ == '__main__':
    main()
