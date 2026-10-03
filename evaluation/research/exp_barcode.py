import sys
import numpy as np
from sig1d import load
from barcode import *

# pares (consulta, mapa, rango u de solape, w de la traza de la consulta)
PAIRS = [('C-', 'A+', (-63, -41), 9.3), ('D-', 'A+', (-71, -26), 5.6)]
for L in (5, 10, 15):
    print(f'=== ventana L={L} m')
    for qn, mn, (u0, u1), wt in PAIRS:
        Q, M = load(qn), load(mn)
        for ua in np.arange(u0, u1 - L + 0.01, 2.0):
            ub = ua + L
            wq = query_rows(Q, ua, ub, wt)
            sc, wm, ms = barcode_match(Q, M, ua, ub, wq)
            if not sc: print(qn, mn, ua, 'sin solape'); continue
            r = summarize(sc, ms)
            print('%s->%s u=[%.0f,%.0f] filas=%d  k*=%+d du*=%+.2f s=%.2f | 2do s=%.2f (k=%s du=%s) margen=%.2f' % (
                qn, mn, ua, ub, len(wq), r['k'], r['du'], r['score'], r['second'], r['second_k'],
                None if r['second_du'] is None else round(r['second_du'], 2), r['score'] - r['second']))
