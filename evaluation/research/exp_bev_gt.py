"""Factibilidad: BEV de vegetacion de dos pasadas por el interior de 13:39 (poses GT) y correlacion 2D."""
import sys
import cv2
import numpy as np
from common import *
from bev import *

seq = '2023-12-26-13-39-43'
t, pos, rot, imgs, _ = load_seq(seq)
src = ColorSource(seq, t)
OUT = Path(__file__).parent / 'out'; OUT.mkdir(exist_ok=True)

passes = {  # (frame_ini, frame_fin) de pasadas por el interior (x en [-150,-15] aprox.)
    'A+': (12182, 13669),   # +x, y~8.3
    'C-': (19858, 20221),   # -x, y~9.3
    'D-': (14188, 14938),   # -x, y~5.6
    'B+': (17752, 18371),   # +x, y~12.8
}
bounds = (-100, -15, 0, 18)
maps = {}
for name, (a, b) in passes.items():
    occ, cnt = build_bev(seq, range(a, b, 3), rot, pos, src, bounds)
    maps[name] = occ
    np.save(OUT / f'bev_{name}.npy', occ)
    print(name, 'celdas observadas %.1f m2' % (np.isfinite(occ).sum() * RES**2), 'frac veg %.2f' % np.nanmean(occ))

# imagen de control
vis = []
for name, occ in maps.items():
    im = np.nan_to_num(occ, nan=0.5)
    vis.append((im * 255).astype(np.uint8)[::-1])
cv2.imwrite(str(OUT / 'bev_all.png'), np.concatenate(vis, 0)[:, :])
