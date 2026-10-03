"""Firmas 1D por hilera (marco alineado a las hileras) y su alineamiento entre pasadas (13:39, poses GT)."""
import warnings
import numpy as np
from scipy.signal import find_peaks
from common import *
from bev import *
warnings.simplefilter('ignore')

seq = '2023-12-26-13-39-43'
THETA = np.radians(1.05)          # direccion de hileras estimada del BEV
t, pos, rot, imgs, _ = load_seq(seq)
src = ColorSource(seq, t)
OUT = Path(__file__).parent / 'out'
passes = {'A+': (12182, 13669), 'C-': (19858, 20221), 'D-': (14188, 14938), 'B+': (17752, 18371)}
U0, U1, W0, W1 = -100, -15, 0, 18
if __name__ == '__main__':
    for name, (a, b) in passes.items():
        occ, cnt = build_bev(seq, range(a, b, 2), rot, pos, src, (U0, U1, W0, W1), THETA)
        np.save(OUT / f'rbev_{name}.npy', occ); np.save(OUT / f'rcnt_{name}.npy', cnt)
        print(name, 'ok')
