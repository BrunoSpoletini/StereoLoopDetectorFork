"""Experimento offline: ¿las features aprendidas separan mejor los candidatos correctos de los incorrectos?

Toma pares (query, candidato) de una corrida: positivos = candidato a < 3 m que la verificacion ORB
rechazo; negativos = candidato a > 10 m. Para cada par cuenta inliers de una matriz esencial RANSAC
(imagenes izquierdas rectificadas, mascara del tractor en FieldSAFE) con ORB y con extractores
aprendidos + LightGlue. Reporta la distribucion de inliers y el recall de positivos al umbral que deja
0 falsos positivos (y 1 %).

  lg_venv/bin/python verify_offline.py p3_salad --sessions fieldsafe --n 300
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from datasets import REPO, select
from sld_eval import load_gt

DEV = 'cuda'


def load_K(calib):
    import yaml
    P = np.array(yaml.safe_load(open(calib))['left_projection_matrix']['data'], float).reshape(3, 4)
    return P[:, :3]


def essential_inliers(p0, p1, K, thr=1.0):
    if len(p0) < 8:
        return 0
    E, mask = cv2.findEssentialMat(p0, p1, K, cv2.RANSAC, 0.999, thr)
    return int(mask.sum()) if mask is not None else 0


class Orb:
    name = 'orb'

    def __init__(self):
        self.orb = cv2.ORB_create(2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def __call__(self, im0, im1, mask):
        k0, d0 = self.orb.detectAndCompute(im0, mask)
        k1, d1 = self.orb.detectAndCompute(im1, mask)
        if d0 is None or d1 is None:
            return np.zeros((0, 2)), np.zeros((0, 2))
        good = [m for m, n in self.bf.knnMatch(d0, d1, k=2) if m.distance < 0.8 * n.distance]
        return (np.float32([k0[m.queryIdx].pt for m in good]), np.float32([k1[m.trainIdx].pt for m in good]))


class Learned:
    def __init__(self, name, max_kp=2048):
        from lightglue import ALIKED, DISK, LightGlue, SuperPoint
        ext = {'superpoint': SuperPoint, 'disk': DISK, 'aliked': ALIKED}[name]
        self.name = name
        self.ext = ext(max_num_keypoints=max_kp).eval().to(DEV)
        self.lg = LightGlue(features=name).eval().to(DEV)

    @torch.no_grad()
    def __call__(self, im0, im1, mask):
        def feats(im):
            t = torch.from_numpy(im).float()[None, None].to(DEV) / 255
            if self.name != 'superpoint':
                t = t.repeat(1, 3, 1, 1)
            f = self.ext.extract(t)
            if mask is not None:  # descartar keypoints sobre el tractor
                kp = f['keypoints'][0].long()
                keep = torch.from_numpy(mask).to(DEV)[kp[:, 1].clamp(0, mask.shape[0] - 1),
                                                       kp[:, 0].clamp(0, mask.shape[1] - 1)] > 0
                f = {k: (v[:, keep] if v.dim() > 1 and v.shape[1] == len(keep) else v) for k, v in f.items()}
            return f
        f0, f1 = feats(im0), feats(im1)
        m = self.lg({'image0': f0, 'image1': f1})['matches'][0].cpu().numpy()
        k0, k1 = f0['keypoints'][0].cpu().numpy(), f1['keypoints'][0].cpu().numpy()
        return k0[m[:, 0]], k1[m[:, 1]]


def sample_pairs(run, sessions, n, seed=0):
    rows = []
    for s in select(sessions):
        f = REPO / 'evaluation/runs' / run / f'{s.name}_queries.csv'
        if not f.exists():
            continue
        gt = load_gt(s.poses)
        q = pd.read_csv(f)
        q = q[(q.candidate >= 0) & gt.revisit[q['query'].values]]
        d = np.linalg.norm(gt.xy[q['query'].values] - gt.xy[q.candidate.values], axis=1)
        q = q.assign(dist=d, session=s.name)
        rows.append(q[(q.status == 7) & (q.dist < 3)].assign(label=1))
        rows.append(q[q.dist > 10].assign(label=0))
    df = pd.concat(rows)
    return pd.concat([g.sample(min(n, len(g)), random_state=seed) for _, g in df.groupby('label')])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--sessions', default='fieldsafe')
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--methods', default='orb,superpoint,disk,aliked')
    a = ap.parse_args()

    pairs = sample_pairs(a.run, a.sessions, a.n)
    sess = {s.name: s for s in select(a.sessions)}
    lists = {k: s.left.read_text().split() for k, s in sess.items()}
    mask = cv2.imread(str(REPO / 'resources/fs_left_mask.png'), 0)
    methods = [Orb() if m == 'orb' else Learned(m) for m in a.methods.split(',')]
    out = []
    for _, r in pairs.iterrows():
        s = sess[r.session]
        K = load_K(s.calibration)
        im0 = cv2.imread(lists[r.session][int(r.candidate)], 0)
        im1 = cv2.imread(lists[r.session][int(r['query'])], 0)
        mk = mask if s.dataset == 'fieldsafe' else None
        row = dict(session=r.session, label=r.label, dist=r.dist)
        for m in methods:
            p0, p1 = m(im0, im1, mk)
            row[f'{m.name}_matches'] = len(p0)
            row[f'{m.name}_inliers'] = essential_inliers(p0, p1, K)
        out.append(row)
    df = pd.DataFrame(out)
    df.to_csv(REPO / 'evaluation/runs' / a.run / 'verify_offline.csv', index=False)
    print(f'{(df.label == 1).sum()} positivos, {(df.label == 0).sum()} negativos')
    for m in methods:
        pos, neg = df[df.label == 1][f'{m.name}_inliers'], df[df.label == 0][f'{m.name}_inliers']
        thr0 = neg.max() + 1
        thr1 = np.percentile(neg, 99) + 1
        print(f'{m.name:10s} inliers med pos={pos.median():6.0f} neg={neg.median():5.0f}  '
              f'recall@0FP={(pos >= thr0).mean():.2f} (thr {thr0:.0f})  recall@1%FP={(pos >= thr1).mean():.2f}')


if __name__ == '__main__':
    main()
