"""Suma al pipeline los loops del generador de firma de hileras (interior del campo, ida y vuelta).

El generador (research/eval_rows.py, sin GT en las decisiones) consulta cada 2 m una ventana de 10 m de firma de
vegetacion contra los segmentos de seguimiento de hileras anteriores. Se aceptan los loops con margen >= --margin
y >= --rows filas emparejadas. Su pose sale de la firma: (pu, pw) = posicion de la query en el marco de hileras del
candidato (adelante, izquierda) y sentido relativo (flip = -1: sentido opuesto, yaw relativo pi).

Se escribe en la convencion de demo_stereo (x_query = R x_match + t): centro de la query en el frame del match
C = (-pw, 0, pu), R = Rqm^T con Rqm = diag(flip, 1, flip), t = -R C.

  python gen_loops.py p6_odo --out p9_gen --sessions rosariofr
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from datasets import select
from learned_verify import RUNS, write_results
from sld_eval import load_run

GEN = Path(__file__).parent / 'research' / 'out'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base_run')
    ap.add_argument('--out', default='p9_gen')
    ap.add_argument('--sessions', default='rosariofr')
    ap.add_argument('--margin', type=float, default=0.4)
    ap.add_argument('--rows', type=int, default=2)
    a = ap.parse_args()
    out = RUNS / a.out
    out.mkdir(parents=True, exist_ok=True)
    for s in select(a.sessions):
        res, q, m, t, r = load_run(RUNS / a.base_run / f'{s.name}_results.yml')
        recs = [dict(query=int(q[i]), candidate=int(m[i]), t=list(t[i]), r=list(r[i])) for i in range(len(q))]
        f = GEN / f"rows_{s.name.replace('rof_', '')}.json"
        n_gen = 0
        if f.exists():
            have = set(int(x) for x in q)
            for g in json.loads(f.read_text()).get('gen', []):
                if g['margin'] < a.margin or g['n_rows'] < a.rows or g['q'] in have:
                    continue
                flip = 1.0 if g['flip'] >= 0 else -1.0
                Rqm = np.diag([flip, 1.0, flip])
                C = np.array([-g['pw'], 0.0, g['pu']])
                R = Rqm.T
                tt = -R @ C
                recs.append(dict(query=int(g['q']), candidate=int(g['c']), t=list(tt),
                                 r=list(Rotation.from_matrix(R).as_rotvec())))
                n_gen += 1
        recs.sort(key=lambda x: x['query'])
        write_results(out / f'{s.name}_results.yml', res['num_images'], recs)
        print(f'{s.name}: {len(q)} loops del pipeline + {n_gen} del generador de firma')


if __name__ == '__main__':
    main()
