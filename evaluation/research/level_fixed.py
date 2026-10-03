"""Nivelacion con una orientacion fija camara-suelo (calibracion extrinseca de montaje: en la practica, gravedad de
la IMU con el extrinseco kalibr camara-IMU). Aqui se usa la normal media del GT de toda la sesion como sustituto de
esa calibracion (es una constante del montaje, no informacion de posicion), y la altura se mide con estereo."""
import sys, json
import cv2
import numpy as np
from common import *
seq = sys.argv[1]
t, pos, rot, imgs, rimgs = load_seq(seq)
n = np.mean(rot[:, 2, :], 0)          # eje z mundo (arriba) expresado en camara: fila 2 de R_wc
n /= np.linalg.norm(n)
sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=7, P1=8 * 49, P2=32 * 49,
                             uniquenessRatio=10, speckleWindowSize=100, speckleRange=2)
hs = []
for k in np.linspace(len(imgs) * 0.05, len(imgs) * 0.95, 12).astype(int):
    d = sgbm.compute(cv2.imread(imgs[k], 0), cv2.imread(rimgs[k], 0)).astype(np.float32) / 16
    v, u = np.mgrid[0:H, 0:W]; m = (d > 3) & (v > 450)
    Z = FX * BASE / d[m]; P = np.stack([(u[m] - CX) * Z / FX, (v[m] - CY) * Z / FY, Z], 1)
    hs.append(np.percentile(-(P @ n), 95))
out = f'/tmp/claude-1000/-home-bruno-Desktop-tesina/b1128bd6-a1d1-4537-bb53-b46ff6c35cf4/scratchpad/rows/level_{seq}.json'
json.dump(dict(n=list(map(float, n)), h=float(np.median(hs)), source='montaje fijo'), open(out, 'w'))
print(seq, 'normal', np.round(n, 4), 'h %.3f' % np.median(hs))
