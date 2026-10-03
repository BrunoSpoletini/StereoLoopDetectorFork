"""Matching conjunto de varias hileras ("codigo de barras 2D"): score(k, du) = media de NCC por fila,
con k = desplazamiento en indice de hilera y du = desplazamiento a lo largo de la hilera."""
import numpy as np
from sig1d import *


def query_rows(Q, ua, ub, w_track, half_width=1.6):
    w = rows(Q, ua, ub)
    return w[np.abs(w - w_track) <= half_width]


def barcode_match(Q, M, ua, ub, wq, search=10.0, min_rows=2):
    wm = rows(M, ua - search, ub + search)
    ms = int(search / R)
    sigs_q = [signature(Q, w, ua, ub) for w in wq]
    sigs_m = [signature(M, w, ua - search, ub + search) for w in wm]
    base = [int(np.argmin(np.abs(wm - w))) for w in wq]
    scores = {}
    for k in range(-len(wm), len(wm)):
        acc, n = np.zeros(2 * ms + 1), np.zeros(2 * ms + 1)
        for j, q in enumerate(sigs_q):
            i = base[j] + k
            if i < 0 or i >= len(wm) or np.isnan(q).mean() > 0.2: continue
            sc = ncc_scan(q, sigs_m[i], ms)
            ok = np.isfinite(sc); acc[ok] += sc[ok]; n[ok] += 1
        if n.max() >= min_rows:
            s = np.where(n >= max(min_rows, 0.6 * len(wq)), acc / np.maximum(n, 1), np.nan)
            if np.isfinite(s).any(): scores[k] = s
    return scores, wm, ms


def summarize(scores, ms, true_k=None):
    best = max(((np.nanmax(s), k, (np.nanargmax(s) - ms) * R) for k, s in scores.items() if np.isfinite(s).any()))
    sc, k, du = best
    # segundo: mejor de (otra k) o (misma k, |du - du*| > 0.3 m)
    cands = []
    for kk, s in scores.items():
        s2 = s.copy()
        if kk == k:
            i = int(round(du / R)) + ms; r = int(0.3 / R); s2[max(0, i - r):i + r + 1] = np.nan
        if np.isfinite(s2).any(): cands.append((np.nanmax(s2), kk, (np.nanargmax(s2) - ms) * R))
    sec = max(cands) if cands else (np.nan, None, None)
    return dict(score=sc, k=k, du=du, second=sec[0], second_k=sec[1], second_du=sec[2])
