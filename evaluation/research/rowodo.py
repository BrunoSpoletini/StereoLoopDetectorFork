"""Odometria referida a las hileras, SIN GT: por frame se proyecta la mascara de vegetacion IR al plano del
canopeo con la orientacion nivelada por estereo, y se estiman (i) el angulo de las hileras respecto del rumbo
de la camara (psi) y (ii) la fase lateral de las hileras (posicion lateral modulo el espaciado). El avance a lo
largo sale del odometro de SLD (VO). Resultado: trayectoria (u, w, psi) en el marco de las hileras, local.
"""
import cv2
import numpy as np
from common import *
from bev import IRSource

ANG = np.radians(np.arange(-10, 10.01, 0.25))


class FrameObs:
    def __init__(self, imgs, Rlev, h_eff, sat_pct=None, step=4, v0=400):
        self.src = IRSource(imgs, step=step, v0=v0, sat=240, tex=-1)
        d = self.src.rays @ Rlev.T                # (x der, y adel, z arriba)
        s = -h_eff / d[:, 2]
        ok = (s > 0) & (s < 6)
        self.xy = (s[:, None] * d[:, :2])         # punto en el plano, marco nivelado de la camara
        self.ok = ok & (np.abs(self.xy[:, 0]) < (3.5 if v0 >= 400 else 5.0))
        self.sat_pct = sat_pct

    def mask(self, k):
        if not hasattr(self, '_cache'):
            from collections import OrderedDict
            self._cache = OrderedDict()
        if k in self._cache:
            self._cache.move_to_end(k); return self._cache[k]
        m = self._mask(k)
        self._cache[k] = m
        if len(self._cache) > 4000: self._cache.popitem(last=False)
        return m

    def _mask(self, k):
        if self.sat_pct is None:
            return self.src.mask(k)
        I = cv2.imread(self.src.imgs[k], 0).astype(np.float32)
        mu = cv2.GaussianBlur(I, (0, 0), 3)
        vals = mu[self.src.uv[:, 1], self.src.uv[:, 0]]
        if self.sat_pct > 0:     # suelo brillante (saturado): vegetacion = por debajo del percentil
            thr = np.percentile(vals, self.sat_pct)
            return (vals < thr).astype(np.float32)
        thr = np.percentile(vals, -self.sat_pct)   # suelo oscuro: vegetacion = por encima del percentil
        return (vals > thr).astype(np.float32)

    def phase_at(self, m, spacing, a):
        xy = self.xy[self.ok]; mv = m[self.ok]
        w = -xy[:, 0] * np.cos(a) + xy[:, 1] * np.sin(a)
        z = np.sum((mv - mv.mean()) * np.exp(1j * 2 * np.pi * w / spacing))
        return (np.angle(z) / (2 * np.pi)) * spacing, abs(z) / max(1, len(mv))

    def rows_angle_phase(self, m, spacing):
        """psi: angulo de las hileras (rad) en el marco de la camara; phi: fase lateral en [0, spacing)."""
        xy = self.xy[self.ok]; mv = m[self.ok]
        best = (-1, 0, 0)
        for a in ANG:
            # coordenada lateral perpendicular a las hileras rotadas a
            w = -xy[:, 0] * np.cos(a) + xy[:, 1] * np.sin(a)
            ph = 2 * np.pi * w / spacing
            z = np.sum((mv - mv.mean()) * np.exp(1j * ph))
            if abs(z) > best[0]: best = (abs(z), a, np.angle(z))
        _, a, ang = best
        return a, (ang / (2 * np.pi)) * spacing, best[0] / max(1, len(mv))


def row_track(obs, frames, odo, spacing, smooth=31, masks=None):
    """Integra (u, w, psi) en el marco de las hileras a lo largo de 'frames' (consecutivos, mismo sentido).
    Dos pasadas: angulo por frame -> mediana movil -> fase lateral evaluada con el angulo suavizado."""
    from scipy.ndimage import median_filter
    ms = [obs.mask(k) if masks is None else masks[k] for k in frames]
    raw = np.array([obs.rows_angle_phase(m, spacing)[0] for m in ms])
    psis = median_filter(raw, size=min(smooth, len(raw)), mode='nearest') if smooth > 1 else raw
    U, Wl, PS, Q = [], [], [], []
    u = 0.0; w = 0.0; prev_phi = None; prev = None
    for j, k in enumerate(frames):
        m = ms[j]; psi = psis[j]
        phi, q = obs.phase_at(m, spacing, psi)
        if prev is not None:
            u += (odo[k] - odo[prev]) * np.cos(psi)
            # fase de las hileras vista desde la camara: si el robot se mueve a la izquierda (+w), las hileras
            # se corren a la derecha (fase disminuye). w = -fase desenrollada.
            dphi = phi - prev_phi
            dphi = (dphi + spacing / 2) % spacing - spacing / 2
            w += -dphi
        prev_phi = phi; prev = k
        U.append(u); Wl.append(w); PS.append(psi); Q.append(q)
    return np.array(U), np.array(Wl), np.array(PS), np.array(Q)


def local_bev(obs, frames, U, Wl, PS, u_range, w_range, res=0.02, masks=None):
    """BEV en el marco de las hileras de la ventana (origen: primer frame de 'frames')."""
    u0, u1 = u_range; w0, w1 = w_range
    nu, nw = int((u1 - u0) / res), int((w1 - w0) / res)
    acc = np.zeros((nw, nu), np.float32); cnt = np.zeros((nw, nu), np.float32)
    for j, k in enumerate(frames):
        m = obs.mask(k) if masks is None else masks[k]
        xy = obs.xy[obs.ok]; mv = m[obs.ok]
        a = PS[j]
        along = xy[:, 0] * np.sin(a) + xy[:, 1] * np.cos(a)
        lat = -xy[:, 0] * np.cos(a) + xy[:, 1] * np.sin(a)      # + = izquierda
        uu = U[j] + along; ww = Wl[j] + lat
        iu = ((uu - u0) / res).astype(int); iw = ((ww - w0) / res).astype(int)
        g = (iu >= 0) & (iu < nu) & (iw >= 0) & (iw < nw)
        np.add.at(acc, (iw[g], iu[g]), mv[g]); np.add.at(cnt, (iw[g], iu[g]), 1)
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
