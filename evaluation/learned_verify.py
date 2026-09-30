"""Verificacion de candidatos con features aprendidas (ALIKED + LightGlue) y pose metrica estereo.

Toma los candidatos que SLD dejo pasar por la consistencia temporal (status 0 = loop, 7 = rechazado por
la geometria ORB) en una corrida, y los re-verifica:
  1. ALIKED + LightGlue entre izquierda vieja e izquierda actual (mascara del tractor en FieldSAFE).
  2. ALIKED + LightGlue entre izquierda y derecha viejas -> profundidad por disparidad (|dy| < 2 px).
  3. PnP RANSAC con los puntos de profundidad confiable (error relativo esperado <= 10 % con 0.5 px);
     los puntos lejanos se cuentan como rayos de profundidad desconocida (como hybrid_far en C++).
Guarda un puntaje por candidato (<out>/<sesion>_verify.csv) para barrer el umbral sin recalcular, y
escribe <out>/<sesion>_results.yml con los loops aceptados al umbral elegido (formato de demo_stereo).

  lg_venv/bin/python learned_verify.py p3_salad_c5 --out p5_aliked --sessions fieldsafe
"""
import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from datasets import REPO, select

DEV = 'cuda'
RUNS = REPO / 'evaluation/runs'


class Features:
    """ALIKED + LightGlue con cache LRU de features por imagen."""

    def __init__(self, max_kp=2048, cache=64):
        from lightglue import ALIKED, LightGlue
        self.ext = ALIKED(max_num_keypoints=max_kp).eval().to(DEV)
        self.lg = LightGlue(features='aliked').eval().to(DEV)
        self.cache = OrderedDict()
        self.size = cache

    @torch.no_grad()
    def get(self, path, mask=None):
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        im = cv2.imread(path, 0)
        t = torch.from_numpy(im).float()[None, None].to(DEV).repeat(1, 3, 1, 1) / 255
        f = self.ext.extract(t)
        if mask is not None:
            kp = f['keypoints'][0].long()
            keep = torch.from_numpy(mask).to(DEV)[kp[:, 1].clamp(0, mask.shape[0] - 1),
                                                   kp[:, 0].clamp(0, mask.shape[1] - 1)] > 0
            f = {k: (v[:, keep] if v.dim() > 1 and v.shape[1] == len(keep) else v) for k, v in f.items()}
        self.cache[path] = f
        if len(self.cache) > self.size:
            self.cache.popitem(last=False)
        return f

    @torch.no_grad()
    def match(self, f0, f1):
        m = self.lg({'image0': f0, 'image1': f1})['matches'][0].cpu().numpy()
        return (f0['keypoints'][0].cpu().numpy()[m[:, 0]], f1['keypoints'][0].cpu().numpy()[m[:, 1]], m[:, 0])


def calib(path):
    c = yaml.safe_load(open(path))
    P1 = np.array(c['left_projection_matrix']['data'], float).reshape(3, 4)
    P2 = np.array(c['right_projection_matrix']['data'], float).reshape(3, 4)
    return P1[:, :3], -P2[0, 3]


def far_inliers(b_old, p_cur, K, R, t, zmin, thr):
    """Rayos de profundidad desconocida >= zmin: minimo error de reproyeccion sobre rho in [0, 1/zmin]."""
    if len(b_old) == 0:
        return 0
    Rb = b_old @ R.T
    best = np.full(len(b_old), np.inf)
    for rho in np.linspace(0, 1 / zmin, 13):
        x = Rb + rho * t
        ok = x[:, 2] > 0
        uv = (x[:, :2] / np.where(ok, x[:, 2], 1)[:, None]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        e = np.where(ok, np.linalg.norm(uv - p_cur, axis=1), np.inf)
        best = np.minimum(best, e)
    return int((best < thr).sum())


def verify(fx, K, fB, paths, mask, rel_err=0.1, reproj=3.0, far_px=2.0):
    """Devuelve dict con matches, inliers cercanos/lejanos, inliers de la esencial y pose (rvec, tvec)."""
    f_ol = fx.get(paths['old_l'], mask)
    f_or = fx.get(paths['old_r'])
    f_q = fx.get(paths['cur_l'], mask)
    p_o, p_q, idx_o = fx.match(f_ol, f_q)
    out = dict(matches=len(p_o), ess=0, near=0, far=0, n3d=0, t=np.zeros(3), r=np.zeros(3))
    if len(p_o) < 8:
        return out
    E, em = cv2.findEssentialMat(p_o, p_q, K, cv2.RANSAC, 0.999, 1.0)
    out['ess'] = int(em.sum()) if em is not None else 0
    # profundidad de los keypoints viejos por matching estereo
    sl, sr, sidx = fx.match(f_ol, f_or)
    disp = sl[:, 0] - sr[:, 0]
    good = (np.abs(sl[:, 1] - sr[:, 1]) < 2.0) & (disp > 0.5)
    depth = dict(zip(sidx[good], fB / disp[good]))
    zmax = rel_err * fB / 0.5
    z = np.array([depth.get(i, np.inf) for i in idx_o])
    near = z <= zmax
    b = np.column_stack([(p_o[:, 0] - K[0, 2]) / K[0, 0], (p_o[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(p_o))])
    out['n3d'] = int(near.sum())
    if near.sum() < 6:
        return out
    X = b[near] * z[near, None]
    ok, rvec, tvec, inl = cv2.solvePnPRansac(X, p_q[near], K, None, iterationsCount=300,
                                             reprojectionError=reproj, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < 6:
        return out
    rvec, tvec = cv2.solvePnPRefineLM(X[inl[:, 0]], p_q[near][inl[:, 0]], K, None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    out.update(near=len(inl), far=far_inliers(b[~near], p_q[~near], K, R, tvec.ravel(), zmax, far_px),
               t=tvec.ravel(), r=rvec.ravel())
    return out


def write_results(path, n_images, rows):
    """Mismo formato que demo_stereo (YAML de OpenCV con listas planas)."""
    def lst(v):
        return '[ ' + ', '.join(str(x) for x in v) + ' ]' if len(v) else '[]'
    lines = ['%YAML:1.0', '---', f'num_images: {int(n_images)}', f'num_loops: {len(rows)}',
             f"loop_query_ids: {lst([int(r['query']) for r in rows])}",
             f"loop_match_ids: {lst([int(r['candidate']) for r in rows])}"]
    for i, ax in enumerate('xyz'):
        lines.append(f"loop_translation_{ax}: {lst([float(r['t'][i]) for r in rows])}")
    for i, ax in enumerate('xyz'):
        lines.append(f"loop_rotation_{ax}: {lst([float(r['r'][i]) for r in rows])}")
    Path(path).write_text('\n'.join(lines) + '\n')


def accept(df, min_near, min_total):
    return (df.near >= min_near) & (df.near + df.far >= min_total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', help='corrida de SLD de la que salen los candidatos')
    ap.add_argument('--out', required=True)
    ap.add_argument('--sessions', default='fieldsafe')
    ap.add_argument('--statuses', default='0,7', help='status de SLD a re-verificar (6 = sin consistencia temporal)')
    ap.add_argument('--min-near', type=int, default=15)
    ap.add_argument('--min-total', type=int, default=60)
    ap.add_argument('--limit', type=int, default=0, help='solo los primeros N candidatos por sesion (prueba)')
    a = ap.parse_args()

    out_dir = RUNS / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    fx = Features()
    mask = cv2.imread(str(REPO / 'resources/fs_left_mask.png'), 0)
    statuses = [int(s) for s in a.statuses.split(',')]
    for s in select(a.sessions):
        qf = RUNS / a.run / f'{s.name}_queries.csv'
        if not qf.exists():
            continue
        vf = out_dir / f'{s.name}_verify.csv'
        if vf.exists():
            ver = pd.read_csv(vf, converters={'t': lambda x: np.array(eval(x)), 'r': lambda x: np.array(eval(x))})
        else:
            q = pd.read_csv(qf)
            q = q[q.status.isin(statuses) & (q.candidate >= 0)]
            if a.limit:
                q = q.head(a.limit)
            K, fB = calib(s.calibration)
            L, Rr = s.left.read_text().split(), s.right.read_text().split()
            mk = mask if s.dataset == 'fieldsafe' else None
            rows = []
            for n, (qi, ci, st) in enumerate(zip(q['query'], q.candidate, q.status)):
                v = verify(fx, K, fB, dict(old_l=L[ci], old_r=Rr[ci], cur_l=L[qi]), mk)
                rows.append(dict(query=qi, candidate=ci, sld_status=st, **v))
                if n % 1000 == 0:
                    print(f'  {s.name}: {n}/{len(q)}', flush=True)
            ver = pd.DataFrame(rows)
            ver.assign(t=ver.t.map(list), r=ver.r.map(list)).to_csv(vf, index=False)
        acc = ver[accept(ver, a.min_near, a.min_total)]
        # un loop por query (el de mas inliers)
        acc = acc.assign(tot=acc.near + acc.far).sort_values('tot', ascending=False).drop_duplicates('query')
        acc = acc.sort_values('query')
        n_images = len(s.left.read_text().split())
        write_results(out_dir / f'{s.name}_results.yml', n_images, acc.to_dict('records'))
        print(f'{s.name}: {len(ver)} candidatos verificados, {len(acc)} aceptados')


if __name__ == '__main__':
    main()
