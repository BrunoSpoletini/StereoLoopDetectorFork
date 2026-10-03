"""Correlacion 2D (NCC) de un parche BEV de una pasada contra el BEV de otra. Verdad = desplazamiento (0,0)."""
import sys
import cv2
import numpy as np

RES = 0.02; X0, Y0 = -100, 0


def load(n):
    return np.load(f'out/bev_{n}.npy')


def fill(o):
    m = np.isfinite(o)
    f = np.where(m, o, np.nanmean(o))
    return f.astype(np.float32), m


def idx(x, y):
    return int((y - Y0) / RES), int((x - X0) / RES)


def corr(qname, mname, xq, yq, sx=12, sy=2.0, hp=False):
    q, qm = fill(load(qname)); M, Mm = fill(load(mname))
    if hp:   # quitar la estructura de baja frecuencia (filas continuas): resalta huecos/plantas
        q = q - cv2.GaussianBlur(q, (0, 0), 25); M = M - cv2.GaussianBlur(M, (0, 0), 25)
    (r0, c0), (r1, c1) = idx(xq[0], yq[0]), idx(xq[1], yq[1])
    T = q[r0:r1, c0:c1]
    assert qm[r0:r1, c0:c1].mean() > 0.9, qm[r0:r1, c0:c1].mean()
    dy, dx = int(sy / RES), int(sx / RES)
    S = M[r0 - dy:r1 + dy, c0 - dx:c1 + dx]
    cov = Mm[r0 - dy:r1 + dy, c0 - dx:c1 + dx]
    R = cv2.matchTemplate(S, T, cv2.TM_CCOEFF_NORMED)   # (2dy+1, 2dx+1)
    # invalidar desplazamientos donde el mapa no esta observado
    C = cv2.boxFilter(cov.astype(np.float32), -1, (T.shape[1], T.shape[0]), normalize=True,
                      anchor=(0, 0))[:R.shape[0], :R.shape[1]]
    R[C < 0.9] = -1
    iy, ix = np.unravel_index(np.argmax(R), R.shape)
    ey, ex = (iy - dy) * RES, (ix - dx) * RES
    best = R[iy, ix]
    # segundo pico fuera de un radio de 0.25 m
    R2 = R.copy(); r = int(0.25 / RES)
    R2[max(0, iy - r):iy + r + 1, max(0, ix - r):ix + r + 1] = -1
    j = np.unravel_index(np.argmax(R2), R2.shape)
    sec = R2[j]; sy2, sx2 = (j[0] - dy) * RES, (j[1] - dx) * RES
    true = R[dy, dx]
    return dict(peak=best, err_along=ex, err_lat=ey, second=sec, second_at=(round(sx2, 2), round(sy2, 2)),
                ratio=best / max(sec, 1e-3), score_at_truth=true, valid=(R > -1).mean())


if __name__ == '__main__':
    for hp in (False, True):
        print('--- high-pass' if hp else '--- crudo')
        for q, m, xq, yq in [('C-', 'A+', (-60, -45), (8.0, 10.5)),
                             ('C-', 'A+', (-58, -48), (8.3, 10.3)),
                             ('D-', 'A+', (-60, -40), (6.0, 8.0)),
                             ('D-', 'A+', (-40, -28), (6.0, 8.0)),
                             ('B+', 'A+', (-75, -55), (10.0, 11.0))]:
            try:
                r = corr(q, m, xq, yq, hp=hp)
                print(q, '->', m, xq, yq, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
            except AssertionError as e:
                print(q, m, xq, yq, 'parche no observado', e)
