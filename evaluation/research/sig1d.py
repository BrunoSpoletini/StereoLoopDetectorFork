"""Firmas 1D por hilera: ocupacion de vegetacion a lo largo de u en una banda de +-BAND alrededor del centro de la fila."""
import warnings
import numpy as np
from scipy.signal import find_peaks
warnings.simplefilter('ignore')
R = 0.02; U0 = -100; W0 = 0.0; BAND = 0.10


def load(n):
    return np.load(f'/tmp/claude-1000/-home-bruno-Desktop-tesina/b1128bd6-a1d1-4537-bb53-b46ff6c35cf4/scratchpad/rows/rbev_{n}.npy')


def rows(occ, ua, ub):
    p = np.nanmean(occ[:, int((ua - U0) / R):int((ub - U0) / R)], 1)
    pk, _ = find_peaks(np.nan_to_num(p), distance=15, prominence=0.15)
    return pk * R + W0


def signature(occ, w, ua, ub, band=BAND):
    r0, r1 = int((w - W0 - band) / R), int((w - W0 + band) / R) + 1
    s = np.nanmean(occ[r0:r1, int((ua - U0) / R):int((ub - U0) / R)], 0)
    return s


def ncc_scan(q, m, maxshift, min_valid=0.8):
    """q: firma consulta (len n), m: firma mapa (len n + 2*maxshift). NCC (con mascara de NaN) por desplazamiento."""
    n = len(q)
    from numpy.lib.stride_tricks import sliding_window_view
    Mw = sliding_window_view(m, n)[:2 * maxshift + 1]
    ok = np.isfinite(Mw) & np.isfinite(q)[None, :]
    cnt = ok.sum(1)
    qv = np.where(ok, q[None, :], 0.0); mv = np.where(ok, Mw, 0.0)
    c = np.maximum(cnt, 1)
    qm = qv.sum(1) / c; mm = mv.sum(1) / c
    qa = np.where(ok, qv - qm[:, None], 0); ma = np.where(ok, mv - mm[:, None], 0)
    num = (qa * ma).sum(1); den = np.sqrt((qa * qa).sum(1) * (ma * ma).sum(1)) + 1e-9
    out = num / den
    out[cnt < min_valid * n] = np.nan
    return out


def gap_stats(sig, thr=0.25, min_len=0.10):
    g = np.nan_to_num(sig, nan=1) < thr
    runs, cur = [], 0
    for v in g:
        if v: cur += 1
        else:
            if cur: runs.append(cur)
            cur = 0
    runs = [r * R for r in runs if r * R >= min_len]
    L = np.isfinite(sig).sum() * R
    return len(runs) / max(L, 1e-6), (np.sum(runs) / L if L else 0)
