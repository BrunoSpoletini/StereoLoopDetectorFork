"""Segmentos de "seguimiento de hileras" detectados SIN GT a partir del estado precomputado (rowstate.py), y sus
firmas por fila en el marco de hileras del segmento. Matching por FFT de una ventana consulta contra segmentos."""
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve, find_peaks
from scipy.ndimage import gaussian_filter1d, median_filter
from common import *
from level import level_rotation
from rowodo import FrameObs

SP = Path('/tmp/claude-1000/-home-bruno-Desktop-tesina/b1128bd6-a1d1-4537-bb53-b46ff6c35cf4/scratchpad/rows')
RES = 0.02
BAND = 0.10


class State:
    def __init__(self, seq, odo, qthr=0.25, psimax=8.0, min_len=12.0, gap=0.6):
        z = np.load(SP / f'state_{seq}.npz')
        self.frames = z['frames']; self.bits = z['bits']; self.nb = int(z['nb'])
        self.psi = z['psi']; self.phase = z['phase']; self.qual = z['qual']; self.spacing = float(z['spacing'])
        self.odo = odo[self.frames]
        t, pos, rot, imgs, _ = load_seq(seq)
        self.obs = FrameObs(imgs, level_rotation(z['n']), float(z['heff']))
        self.xy = self.obs.xy[self.obs.ok]
        qf = median_filter(self.qual, 9)
        if qthr < 0:          # umbral adaptativo: |qthr| * percentil 75 de la calidad de la sesion
            qthr = -qthr * np.percentile(qf, 75)
        self.qthr = qthr
        good = (qf >= qthr) & (np.abs(self.psi) < np.radians(psimax))
        self.good = good
        # segmentos: corridas de frames buenos, tolerando huecos cortos (< gap m)
        segs = []; i = 0; n = len(good)
        while i < n:
            if not good[i]: i += 1; continue
            j = i
            while j + 1 < n:
                if good[j + 1]: j += 1; continue
                k = j + 1
                while k < n and not good[k] and self.odo[k] - self.odo[j] < gap: k += 1
                if k < n and good[k] and self.odo[k] - self.odo[j] < gap: j = k; continue
                break
            if self.odo[j] - self.odo[i] >= min_len: segs.append((i, j))
            i = j + 1
        self.segs = segs

    def mask(self, i):
        return np.unpackbits(self.bits[i])[:self.nb].astype(np.float32)

    def track(self, i0, i1):
        U = [0.0]; Wl = [0.0]
        for i in range(i0 + 1, i1 + 1):
            U.append(U[-1] + (self.odo[i] - self.odo[i - 1]) * np.cos(self.psi[i]))
            dphi = self.phase[i] - self.phase[i - 1]
            dphi = (dphi + self.spacing / 2) % self.spacing - self.spacing / 2
            Wl.append(Wl[-1] - dphi)
        return np.array(U), np.array(Wl)

    def segment(self, si):
        """BEV del segmento y firmas por fila (filas a |w - traza| <= 1.6 m)."""
        i0, i1 = self.segs[si]
        U, Wl = self.track(i0, i1)
        u0, u1, w0, w1 = -1.0, U[-1] + 4.0, -4.0, 4.0
        nu, nw = int((u1 - u0) / RES), int((w1 - w0) / RES)
        acc = np.zeros((nw, nu), np.float32); cnt = np.zeros((nw, nu), np.float32)
        for j, i in enumerate(range(i0, i1 + 1)):
            a = self.psi[i]
            along = self.xy[:, 0] * np.sin(a) + self.xy[:, 1] * np.cos(a)
            lat = -self.xy[:, 0] * np.cos(a) + self.xy[:, 1] * np.sin(a)
            iu = ((U[j] + along - u0) / RES).astype(int); iw = ((Wl[j] + lat - w0) / RES).astype(int)
            g = (iu >= 0) & (iu < nu) & (iw >= 0) & (iw < nw)
            m = self.mask(i)
            np.add.at(acc, (iw[g], iu[g]), m[g]); np.add.at(cnt, (iw[g], iu[g]), 1)
        bev = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
        wt = np.median(Wl)
        prof = np.nanmean(bev, 1)
        pk, _ = find_peaks(np.nan_to_num(prof), distance=int(0.3 / RES), prominence=0.05)
        ws = w0 + pk * RES
        ws = ws[np.abs(ws - wt) <= 1.6]
        sigs = []
        for w in ws:
            r0, r1 = int((w - w0 - BAND) / RES), int((w - w0 + BAND) / RES) + 1
            s = np.nanmean(bev[r0:r1], 0)
            f = np.isfinite(s)
            s = np.where(f, s, np.nanmean(s) if f.any() else 0.0)
            s = gaussian_filter1d(s, 2)
            sigs.append(s.astype(np.float64))
        return dict(i0=i0, i1=i1, U=U, W=Wl, u0=u0, ws=np.array(ws), sigs=np.array(sigs) if sigs else np.zeros((0, nu)))


def ncc_all(q, m):
    """NCC de q (len n, sin NaN) contra todas las posiciones de m (len N >= n). Devuelve len N-n+1."""
    n = len(q)
    if len(m) < n: return np.zeros(0)
    qz = q - q.mean(); sq = np.sqrt((qz * qz).sum()) + 1e-9
    num = fftconvolve(m, qz[::-1], mode='valid')
    c1 = np.cumsum(np.r_[0, m]); c2 = np.cumsum(np.r_[0, m * m])
    s1 = c1[n:] - c1[:-n]; s2 = c2[n:] - c2[:-n]
    var = np.maximum(s2 - s1 * s1 / n, 1e-9)
    return num / (sq * np.sqrt(var))


def query_window(seg, iq, L):
    """Firma de las ultimas L m antes del frame iq (indice de estado) dentro del segmento."""
    j = iq - seg['i0']
    uq = seg['U'][j]
    if uq < L: return None
    a, b = int((uq - L - seg['u0']) / RES), int((uq - seg['u0']) / RES)
    wq = seg['W'][j]
    sel = np.abs(seg['ws'] - wq) <= 1.6
    if sel.sum() < 2: return None
    return dict(ws=seg['ws'][sel] - wq, sigs=seg['sigs'][sel][:, a:b], u_start=uq - L, uq=uq, wq=wq)


def match_segment(Q, S, spacing, min_rows=2):
    """Hipotesis (z, flip, k, du_pos(q en marco S), dw_pos) de la ventana Q contra el segmento S."""
    out = []
    if len(S['ws']) < min_rows: return out
    for s in (1, -1):
        wq = s * Q['ws']; sq = Q['sigs'] if s > 0 else Q['sigs'][:, ::-1]
        order = np.argsort(wq); wq = wq[order]; sq = sq[order]
        iq = np.round((wq - wq[0]) / spacing).astype(int)
        im = np.round((S['ws'] - S['ws'][0]) / spacing).astype(int)
        C = {}
        for j in range(len(wq)):
            for i in range(len(S['ws'])):
                C[(j, i)] = np.arctanh(np.clip(ncc_all(sq[j], S['sigs'][i]), -0.99, 0.99))
        nlen = len(next(iter(C.values())))
        if nlen == 0: continue
        for k in range(-iq.max() - 1, im.max() + 2):
            pairs = [(j, int(np.where(im == iq[j] + k)[0][0])) for j in range(len(iq)) if (iq[j] + k) in im]
            if len(pairs) < min_rows: continue
            z = sum(C[p] for p in pairs) / np.sqrt(len(pairs))
            idx = int(np.argmax(z))
            # posicion en S de la ventana: inicio en u = S.u0 + idx*RES; la consulta (extremo final) mapea a:
            # flip +: q -> inicio + L ; flip -: la ventana invertida empieza en el extremo q, asi que q -> inicio
            L = sq.shape[1] * RES
            u_q = S['u0'] + idx * RES + (L if s > 0 else 0.0)
            dw = np.mean([S['ws'][i] - wq[j] for j, i in pairs])   # ws de Q ya relativos a la traza en q
            out.append((float(z[idx]), s, k, u_q, dw, len(pairs), z, idx))
    return out


def best_and_margin(hyps):
    if not hyps: return None, np.nan
    hyps = sorted(hyps, key=lambda h: -h[0])
    b = hyps[0]
    sec = max([h[0] for h in hyps[1:]] + [-9])
    z = b[6].copy(); r = int(0.3 / RES); z[max(0, b[7] - r):b[7] + r + 1] = -9
    return b, b[0] - max(sec, float(z.max()))
