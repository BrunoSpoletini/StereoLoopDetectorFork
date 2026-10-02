"""Impacto de los loops en el error de trayectoria (ATE) tras optimizar un grafo de poses 2D.

Nodos: un frame de cada `--stride` (pose SE(2) x, y, theta). Aristas:
  - odometria: odometria visual estereo integrada (sin GPS) entre nodos consecutivos;
  - loops: pose relativa estimada por cada loop aceptado (query respecto del match), llevada a los nodos mas
    cercanos componiendo con la odometria (tramo corto).
Se optimiza por minimos cuadrados (L2: los loops ya estan verificados; ver optimize) y se reporta el ATE: RMSE de posicion tras
alinear la trayectoria al GT con una transformacion rigida 2D (sin escala: el estereo es metrico).

  python ate_eval.py --runs baseline,p6_odo --vo-run p3c5vo --sessions rosariofr
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from datasets import REPO, select
from sld_eval import load_run, wrap

RUNS = REPO / 'evaluation/runs'
SIG_ODO = np.array([0.02, 0.02, np.radians(0.3)])   # por paso de nodo (x, y, theta)
SIG_LOOP = np.array([0.3, 0.3, np.radians(2.0)])


def se2_between(a, b):
    """Pose de b expresada en el frame de a (arrays (..., 3))."""
    d = b[..., :2] - a[..., :2]
    c, s = np.cos(a[..., 2]), np.sin(a[..., 2])
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1],
                     wrap(b[..., 2] - a[..., 2])], axis=-1)


def se2_compose(a, b):
    c, s = np.cos(a[..., 2]), np.sin(a[..., 2])
    return np.stack([a[..., 0] + c * b[..., 0] - s * b[..., 1], a[..., 1] + s * b[..., 0] + c * b[..., 1],
                     wrap(a[..., 2] + b[..., 2])], axis=-1)


def vo_poses(vo):
    """Odometria (ejes de camara del primer frame) -> SE(2) en (adelante, izquierda), theta antihorario."""
    return np.stack([vo.vo_z.values, -vo.vo_x.values, -vo.vo_yaw.values], axis=1)


def loop_measurements(t, r):
    """Pose de la query en el frame del match, en SE(2) (adelante, izquierda, theta)."""
    R = Rotation.from_rotvec(r).as_matrix()                  # x_cur = R x_old + t
    c = -np.einsum('nji,nj->ni', R, t)
    Rqm = np.transpose(R, (0, 2, 1))
    yaw = np.arctan2(Rqm[:, 0, 2], Rqm[:, 2, 2])
    return np.stack([c[:, 2], -c[:, 0], -yaw], axis=1)


def optimize(odo, loops, nodes, loss='linear'):
    """odo: (n_nodos, 3) poses iniciales; loops: lista de (i, j, medicion i->j)."""
    n = len(nodes)
    meas_odo = se2_between(odo[:-1], odo[1:])
    li = np.array([l[0] for l in loops], int)
    lj = np.array([l[1] for l in loops], int)
    lz = np.array([l[2] for l in loops]).reshape(-1, 3)

    def resid(x):
        p = x.reshape(n, 3)
        r_odo = (se2_between(p[:-1], p[1:]) - meas_odo)
        r_odo[:, 2] = wrap(r_odo[:, 2])
        r = [r_odo / SIG_ODO, ((p[0] - odo[0]) / 1e-3)[None]]
        if len(li):
            r_l = se2_between(p[li], p[lj]) - lz
            r_l[:, 2] = wrap(r_l[:, 2])
            r.append(r_l / SIG_LOOP)
        return np.concatenate([a.ravel() for a in r])

    m = 3 * (n - 1) + 3 + 3 * len(li)
    J = lil_matrix((m, 3 * n), dtype=int)
    for k in range(n - 1):
        J[3 * k:3 * k + 3, 3 * k:3 * k + 6] = 1
    J[3 * (n - 1):3 * n, 0:3] = 1
    base = 3 * n
    for k, (i, j) in enumerate(zip(li, lj)):
        J[base + 3 * k:base + 3 * k + 3, 3 * i:3 * i + 3] = 1
        J[base + 3 * k:base + 3 * k + 3, 3 * j:3 * j + 3] = 1
    # con drift de decenas de metros los residuos iniciales de los loops son enormes y una perdida robusta los
    # descarta a todos; los loops ya pasaron verificacion geometrica, asi que se usa L2 (sin robustez)
    sol = least_squares(resid, odo.ravel(), jac_sparsity=J.tocsr(), loss=loss, x_scale='jac', max_nfev=100)
    return sol.x.reshape(n, 3)


def ate(est_xy, gt_xy):
    """RMSE tras alinear con una transformacion rigida 2D (Umeyama sin escala)."""
    mu_e, mu_g = est_xy.mean(0), gt_xy.mean(0)
    H = (est_xy - mu_e).T @ (gt_xy - mu_g)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    aligned = (est_xy - mu_e) @ R.T + mu_g
    return float(np.sqrt(np.mean(np.sum((aligned - gt_xy) ** 2, axis=1)))), aligned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='baseline,p6_odo')
    ap.add_argument('--vo-run', default='p3c5vo')
    ap.add_argument('--sessions', default='rosariofr')
    ap.add_argument('--stride', type=int, default=5)
    ap.add_argument('--max-loops', type=int, default=3000, help='submuestreo uniforme de loops por sesion')
    a = ap.parse_args()
    rows = []
    for s in select(a.sessions):
        qf = RUNS / a.vo_run / f'{s.name}_queries.csv'
        if not qf.exists():
            continue
        vo = pd.read_csv(qf).set_index('query').sort_index()
        gt_xy = np.loadtxt(s.poses, delimiter=',', ndmin=2)[:, 1:3]
        full = vo_poses(vo)
        nodes = np.arange(0, len(full), a.stride)
        odo = full[nodes]
        node_of = np.clip(np.round(np.arange(len(full)) / a.stride).astype(int), 0, len(nodes) - 1)
        row = dict(session=s.name, path_m=float(np.linalg.norm(np.diff(gt_xy, axis=0), axis=1).sum()))
        row['odom'], _ = ate(odo[:, :2], gt_xy[nodes])
        for run in a.runs.split(','):
            f = RUNS / run / f'{s.name}_results.yml'
            if not f.exists():
                continue
            _, q, m, t, r = load_run(f)
            if r is None or not len(q):
                row[run], row[f'{run}_loops'] = row['odom'], 0
                continue
            if len(q) > a.max_loops:
                keep = np.linspace(0, len(q) - 1, a.max_loops).astype(int)
                q, m, t, r = q[keep], m[keep], t[keep], r[keep]
            z = loop_measurements(t, r)
            # llevar la medicion a los nodos mas cercanos: T_nm^nq = T_nm^m * T_m^q * T_q^nq (odometria corta)
            i, j = node_of[m], node_of[q]
            z = se2_compose(se2_compose(se2_between(full[nodes[i]], full[m]), z), se2_between(full[q], full[nodes[j]]))
            est = optimize(odo, list(zip(i, j, z)), nodes)
            row[run], _ = ate(est[:, :2], gt_xy[nodes])
            row[f'{run}_loops'] = len(q)
        rows.append(row)
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    df = pd.DataFrame(rows)
    out = RUNS / f'ate_{a.sessions}.csv'
    df.to_csv(out, index=False)
    print(df.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
