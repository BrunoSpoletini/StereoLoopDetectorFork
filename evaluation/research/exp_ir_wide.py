"""IR con franja mas lejana (v>=V0) para pasadas separadas 2.7 m (D- vs A+) y 4.4 m (B+ vs A+)."""
import sys
import numpy as np
from common import *
from bev import build_bev, IRSource
import sig1d
from barcode import barcode_match, summarize
from sig1d import rows

seq = '2023-12-26-13-39-43'
t, pos, rot, imgs, _ = load_seq(seq)
THETA = np.radians(1.05)
SP = '/tmp/claude-1000/-home-bruno-Desktop-tesina/b1128bd6-a1d1-4537-bb53-b46ff6c35cf4/scratchpad/rows/'
V0 = int(sys.argv[1]); HALF = float(sys.argv[2])
src = IRSource(imgs, step=4, v0=V0, sat=240, tex=-1)
P = {'A+': ((12182, 13669), 8.3), 'D-': ((14188, 14938), 5.6), 'B+': ((17752, 18371), 12.75)}
for name, ((a, b), _) in P.items():
    occ, _ = build_bev(seq, range(a, b, 2), rot, pos, src, (-100, -15, 0, 18), THETA)
    np.save(SP + f'rbev_W{V0}{name}.npy', occ)
M = sig1d.load(f'W{V0}A+')
for qn, (u0, u1) in [('D-', (-71, -26)), ('B+', (-84, -46))]:
    Q = sig1d.load(f'W{V0}{qn}'); wt = P[qn][1]
    ok = n = 0
    for ua in np.arange(u0, u1 - 10, 3.0):
        wq = rows(Q, ua, ua + 10); wq = wq[(abs(wq - wt) <= HALF) & (abs(wq - 8.3) <= HALF)]
        if len(wq) < 2: continue
        sc, wm, ms = barcode_match(Q, M, ua, ua + 10, wq)
        if not sc: continue
        r = summarize(sc, ms); n += 1; ok += (r['k'] == 0 and abs(r['du']) < 1)
        print(qn, 'u=%.0f filas=%d k=%+d du=%+.2f s=%.2f 2do=%.2f' % (ua, len(wq), r['k'], r['du'], r['score'], r['second']))
    print(f'== {qn} V0={V0} half={HALF}: {ok}/{n} correctas')
