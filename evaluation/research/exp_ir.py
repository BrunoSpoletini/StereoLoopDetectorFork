"""Variante solo-IR (camara del pipeline): mascara = intensidad suavizada < umbral (el suelo cercano satura en NIR)."""
import sys
import numpy as np
from common import *
import bev
from bev import build_bev, IRSource
import sig1d
from barcode import query_rows, barcode_match, summarize

seq = '2023-12-26-13-39-43'
t, pos, rot, imgs, _ = load_seq(seq)
THETA = np.radians(1.05)
SP = '/tmp/claude-1000/-home-bruno-Desktop-tesina/b1128bd6-a1d1-4537-bb53-b46ff6c35cf4/scratchpad/rows/'
sat = float(sys.argv[1]) if len(sys.argv) > 1 else 240
src = IRSource(imgs, step=4, v0=400, sat=sat, tex=-1)     # tex=-1: sin condicion de textura
for name, (a, b) in {'A+': (12182, 13669), 'C-': (19858, 20221)}.items():
    occ, _ = build_bev(seq, range(a, b, 2), rot, pos, src, (-100, -15, 0, 18), THETA)
    np.save(SP + f'rbev_IR{name}.npy', occ)
Q, M = sig1d.load('IRC-'), sig1d.load('IRA+')
print('frac veg IR A+ %.2f' % np.nanmean(M))
for ua in range(-63, -50, 2):
    wq = query_rows(Q, ua, ua + 10, 9.3)
    sc, wm, ms = barcode_match(Q, M, ua, ua + 10, wq)
    r = summarize(sc, ms)
    print('u=[%d,%d] filas=%d k=%+d du=%+.2f s=%.2f 2do=%.2f' % (ua, ua + 10, len(wq), r['k'], r['du'], r['score'], r['second']))
