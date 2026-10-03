"""Estima la altura de la camara sobre el suelo con estereo SGBM + orientacion GT (plano z=cte en mundo)."""
import sys
import cv2
import numpy as np
from common import *

seq = sys.argv[1] if len(sys.argv) > 1 else '2023-12-26-13-39-43'
t, pos, rot, imgs, rimgs = load_seq(seq)
sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=7, P1=8*49, P2=32*49,
                             uniquenessRatio=10, speckleWindowSize=100, speckleRange=2)
hs = []
for k in range(3000, len(imgs), 2500):
    L = cv2.imread(imgs[k], 0); Rr = cv2.imread(rimgs[k], 0)
    d = sgbm.compute(L, Rr).astype(np.float32) / 16
    v, u = np.mgrid[0:H, 0:W]
    m = (d > 2) & (v > 450)          # parte baja de la imagen: suelo/plantas cercanas
    Z = FX * BASE / d[m]; X = (u[m] - CX) * Z / FX; Y = (v[m] - CY) * Z / FY
    P = np.stack([X, Y, Z], 1) @ rot[k].T      # en ejes mundo, origen en la camara
    zz = P[:, 2]
    # el suelo es el percentil bajo de altura (las plantas sobresalen)
    hs.append((k, -np.percentile(zz, 5), -np.median(zz)))
    print(k, 'h_suelo(p5) %.3f  h_mediana %.3f  n=%d' % (hs[-1][1], hs[-1][2], m.sum()))
hs = np.array(hs)
print('altura camara (p5) media %.3f sd %.3f' % (hs[:, 1].mean(), hs[:, 1].std()))
