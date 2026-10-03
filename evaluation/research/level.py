"""Calibracion de la orientacion de la camara respecto del suelo SIN GT: plano por RANSAC sobre puntos SGBM."""
import cv2
import numpy as np
from common import *


def level_from_stereo(imgs, rimgs, n=12, seed=0):
    sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=7, P1=8 * 49, P2=32 * 49,
                                 uniquenessRatio=10, speckleWindowSize=100, speckleRange=2)
    rng = np.random.default_rng(seed)
    normals, hs = [], []
    for k in np.linspace(len(imgs) * 0.05, len(imgs) * 0.95, n).astype(int):
        L = cv2.imread(imgs[k], 0); Rr = cv2.imread(rimgs[k], 0)
        d = sgbm.compute(L, Rr).astype(np.float32) / 16
        v, u = np.mgrid[0:H, 0:W]
        m = (d > 3) & (v > 450)
        Z = FX * BASE / d[m]; X = (u[m] - CX) * Z / FX; Y = (v[m] - CY) * Z / FY
        P = np.stack([X, Y, Z], 1)
        P = P[rng.choice(len(P), min(20000, len(P)), replace=False)]
        best = (0, None)
        for _ in range(300):
            a, b, c = P[rng.choice(len(P), 3, replace=False)]
            nn = np.cross(b - a, c - a); nrm = np.linalg.norm(nn)
            if nrm < 1e-9: continue
            nn /= nrm
            if nn[1] > 0: nn = -nn               # normal "hacia arriba" (y de camara apunta abajo)
            dist = (P - a) @ nn
            inl = np.abs(dist) < 0.05
            if inl.sum() > best[0]: best = (inl.sum(), nn)
        nn = best[1]
        h = -(P @ nn)                      # altura de cada punto bajo la camara (positiva)
        normals.append(nn); hs.append(np.percentile(h, 95))
    nrm = np.median(np.array(normals), 0); nrm /= np.linalg.norm(nrm)
    return nrm, float(np.median(hs))


def level_rotation(nrm):
    """Rotacion camara -> marco nivelado con ejes (x derecha, y adelante, z arriba)."""
    up = nrm
    fwd = np.array([0, 0, 1.0]) - up * up[2]; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    return np.stack([right, fwd, up], 0)


if __name__ == '__main__':
    import sys
    seq = sys.argv[1]
    t, pos, rot, imgs, rimgs = load_seq(seq)
    nrm, h = level_from_stereo(imgs, rimgs)
    pitch = np.degrees(np.arcsin(-nrm[2])) if False else np.degrees(np.arctan2(nrm[2], -nrm[1]))
    print('normal', np.round(nrm, 4), 'pitch(abajo) %.2f deg' % pitch, 'h_suelo %.3f' % h)
    # comparacion con GT (solo para verificar): angulo entre eje optico y horizontal
    zc = rot[:, :, 2]; print('GT: pitch medio %.2f deg' % np.degrees(np.arcsin(-zc[:, 2])).mean())
