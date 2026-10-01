"""Filtro de consistencia con la odometria visual para loops verificados.

Para cada loop aceptado (q, m) compara el desplazamiento planar estimado por el loop (centro de la query en
el frame del match, -R^T t) con el que predice la odometria visual estereo integrada entre m y q (sin GPS).
Rechaza si ‖Δ_vo − Δ_loop‖ > alpha * camino_vo(m, q) + beta: la tolerancia crece con el camino recorrido
(drift de la odometria), asi que solo actua en loops "cercanos en el camino" — p. ej. el aliasing a lo largo
del mismo surco, donde la odometria sabe que el robot avanzo 20-35 m en linea recta. Mas alla de --max-path
metros de camino no se filtra: en una vuelta de ~700 m el drift de la odometria llega a ~25 %.

  python odo_filter.py p5_aliked --vo-run p3c5vo --out p6_odo --sessions fieldsafe
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from datasets import select
from learned_verify import RUNS, accept, write_results


def loop_displacement(t, r):
    """Centro de la query en el frame de la camara del match, en el plano (x derecha, z adelante)."""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_rotvec(r).as_matrix()
    c = -np.einsum('nji,nj->ni', R, t)
    return np.stack([c[:, 0], c[:, 2]], axis=1)


def vo_displacement(vo, q, m):
    """Desplazamiento de la odometria de m a q, expresado en el frame (planar) de la camara de m."""
    d = vo[['vo_x', 'vo_z']].values[q] - vo[['vo_x', 'vo_z']].values[m]
    yaw = vo.vo_yaw.values[m]       # heading del eje optico (atan2(R02, R22)): z_m = (sin, cos) en (x, z)
    s, c = np.sin(yaw), np.cos(yaw)
    return np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('verify_run')
    ap.add_argument('--vo-run', required=True, help='corrida de SLD con la pose de odometria en el CSV')
    ap.add_argument('--out', required=True)
    ap.add_argument('--sessions', default='fieldsafe')
    ap.add_argument('--alpha', type=float, default=0.2)
    ap.add_argument('--max-path', type=float, default=100.0,
                    help='solo se filtra si el camino VO entre m y q es menor (mas alla el drift domina)')
    ap.add_argument('--beta', type=float, default=1.0)
    ap.add_argument('--min-near', type=int, default=15)
    ap.add_argument('--min-total', type=int, default=60)
    a = ap.parse_args()
    out = RUNS / a.out
    out.mkdir(parents=True, exist_ok=True)
    for s in select(a.sessions):
        vf = RUNS / a.verify_run / f'{s.name}_verify.csv'
        qf = RUNS / a.vo_run / f'{s.name}_queries.csv'
        if not vf.exists() or not qf.exists():
            continue
        v = pd.read_csv(vf)
        if len(v) == 0:
            write_results(out / f'{s.name}_results.yml', len(s.left.read_text().split()), [])
            continue
        v = v[accept(v, a.min_near, a.min_total)].copy()
        vo = pd.read_csv(qf).set_index('query').sort_index()
        t = np.array([eval(x) for x in v.t]).reshape(-1, 3)
        r = np.array([eval(x) for x in v.r]).reshape(-1, 3)
        q, m = v['query'].values, v.candidate.values
        if len(v):
            incons = np.linalg.norm(vo_displacement(vo, q, m) - loop_displacement(t, r), axis=1)
            path = vo.odometer.values[q] - vo.odometer.values[m]
            v['ok'] = (path >= a.max_path) | (incons <= a.alpha * path + a.beta)
        else:
            v['ok'] = np.zeros(0, bool)
        acc = v[v.ok].assign(tot=lambda d: d.near + d.far, t=lambda d: d.t.map(eval), r=lambda d: d.r.map(eval))
        acc = acc.sort_values('tot', ascending=False).drop_duplicates('query').sort_values('query')
        write_results(out / f'{s.name}_results.yml', len(s.left.read_text().split()), acc.to_dict('records'))
        print(f'{s.name}: {len(v)} aceptados por geometria, {int(v.ok.sum())} consistentes con la odometria')


if __name__ == '__main__':
    main()
