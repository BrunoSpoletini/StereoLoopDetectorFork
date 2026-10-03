"""Ablacion: la informacion esta en los huecos? Compara firma continua vs. firma binaria de huecos
(ocupacion suavizada < umbral) vs. firma sin huecos (recortada a [thr, 1])."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
import sig1d, barcode
from barcode import query_rows, barcode_match, summarize

orig = sig1d.signature


def make(mode, thr):
    def sig(occ, w, ua, ub, band=sig1d.BAND):
        s = orig(occ, w, ua, ub, band)
        f = np.isfinite(s); s2 = np.where(f, s, np.nanmean(s))
        s2 = gaussian_filter1d(s2, 2)          # 4 cm
        t = np.nanpercentile(s2, thr * 100) if thr < 0 or True else thr   # umbral relativo: percentil
        t = np.percentile(s2, abs(thr) * 100)
        if mode == 'gap':
            out = (s2 < t).astype(float)
        elif mode == 'nogap':
            out = np.maximum(s2, t)          # borra los huecos (todo lo que baja de thr queda en thr)
        else:
            out = s2
        out[~f] = np.nan
        return out
    return sig


for pre, (q, m) in {'color': ('C-', 'A+'), 'IR': ('IRC-', 'IRA+')}.items():
    Q, M = sig1d.load(q), sig1d.load(m)
    for mode, thr in [('cont', 0), ('gap', 0.10), ('gap', 0.20), ('nogap', 0.20), ('nogap', 0.40)]:
        barcode.signature = make(mode, thr)
        out = []
        for ua in range(-63, -50, 2):
            wq = query_rows(Q, ua, ua + 10, 9.3)
            sc, wm, ms = barcode_match(Q, M, ua, ua + 10, wq)
            r = summarize(sc, ms)
            out.append((r['k'] == 0 and abs(abs(r['du']) - 0.35) < 0.3, r['score'], r['second']))
        o = np.array(out, float)
        print(f'{pre:5s} {mode:5s} pct={thr:.2f}: correctas {int(o[:,0].sum())}/{len(o)}  score {o[:,1].mean():.2f}  2do {o[:,2].mean():.2f}  margen {np.mean(o[:,1]-o[:,2]):.2f}')
