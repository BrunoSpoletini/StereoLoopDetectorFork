"""Poses locales a partir de la odometria visual de SLD (vo_x, vo_z, vo_yaw), anclada en un frame de referencia."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

VO_DIR = '/home/bruno/Desktop/tesina/StereoLoopDetectorFork/evaluation/runs/p3c5vo'


def load_vo(tag):
    import csv
    d = np.genfromtxt(f'{VO_DIR}/rof_{tag}_queries.csv', delimiter=',', names=True)
    return np.c_[d['vo_x'], d['vo_z'], d['vo_yaw']], d['odometer']


def anchored_poses(vo, frames, s, pos_s, rot_s, yaw_sign=1.0):
    """Para cada frame i: pose camara->mundo usando el movimiento VO relativo a s y la pose (GT) en s."""
    xs, zs, ys = vo[s]
    c, si = np.cos(ys), np.sin(ys)
    fwd = rot_s[:, 2].copy(); fwd[2] = 0; fwd /= np.linalg.norm(fwd)
    right = np.array([fwd[1], -fwd[0], 0.0])   # derecha en mundo (z arriba)
    P, Rm = {}, {}
    for i in frames:
        dx, dz = vo[i, 0] - xs, vo[i, 1] - zs
        # expresar el desplazamiento en el marco de la camara en s (rotar por -yaw_s)
        lx = c * dx - si * dz; lz = si * dx + c * dz
        dyaw = yaw_sign * (vo[i, 2] - ys)
        P[i] = pos_s + lz * fwd + lx * right
        Rm[i] = Rot.from_rotvec([0, 0, dyaw]).as_matrix() @ rot_s
    return P, Rm


def odo_heading_poses(vo, odo, frames, s, pos_s, rot_s):
    """Dead reckoning: distancia del odometro de SLD + rumbo relativo de la VO, anclado en la pose de s."""
    frames = list(frames)
    fwd = rot_s[:, 2].copy(); fwd[2] = 0; fwd /= np.linalg.norm(fwd)
    h_s = np.arctan2(fwd[1], fwd[0])
    P, Rm = {}, {}
    p = pos_s.copy(); prev = s
    for i in frames:
        dyaw = -(vo[i, 2] - vo[s, 2])         # yaw VO = atan2(x, z): positivo hacia la derecha
        h = h_s + dyaw
        p = p + (odo[i] - odo[prev]) * np.array([np.cos(h), np.sin(h), 0.0]); prev = i
        P[i] = p.copy()
        Rm[i] = Rot.from_rotvec([0, 0, dyaw]).as_matrix() @ rot_s
    return P, Rm
