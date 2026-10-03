"""Para cada fila de la pasada consulta (ventana de L m), busca la mejor (fila, desplazamiento) en la pasada mapa."""
import sys
import numpy as np
from sig1d import *

def run(qn, mn, ua, ub, search=8.0, rows_range=None, verbose=True):
    Q, M = load(qn), load(mn)
    wq = rows(Q, ua, ub); wm = rows(M, ua - search, ub + search)
    ms = int(search / R)
    res = []
    for w in wq:
        q = signature(Q, w, ua, ub)
        if np.isnan(q).mean() > 0.2: continue
        best = []
        for wmr in wm:
            m = signature(M, wmr, ua - search, ub + search)
            sc = ncc_scan(q, m, ms)
            if np.all(np.isnan(sc)): continue
            i = np.nanargmax(sc)
            # segundo mejor desplazamiento en la misma fila, fuera de +-0.3 m
            sc2 = sc.copy(); r = int(0.3 / R); sc2[max(0, i - r):i + r + 1] = np.nan
            best.append((sc[i], wmr, (i - ms) * R, np.nanmax(sc2) if np.isfinite(sc2).any() else np.nan))
        if not best: continue
        best.sort(reverse=True)
        top, second_row = best[0], (best[1] if len(best) > 1 else (np.nan,) * 4)
        res.append((w, top[1] - w, top[2], top[0], top[3], second_row[0], second_row[1] - w))
        if verbose:
            print('fila q w=%.2f -> dw=%+.2f du=%+.2f ncc=%.2f | 2do du misma fila %.2f | 2da fila ncc=%.2f (dw=%+.2f)'
                  % res[-1])
    return np.array(res)

if __name__ == '__main__':
    qn, mn, ua, ub = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    run(qn, mn, ua, ub)
