"""Mapa cenital (BEV) de vegetacion por pasada, proyectando la mascara de plantas al plano del suelo.

Mascara: ExG sobre la camara color (640x360, sincronizada por timestamp) o umbral en IR.
Pose: GT de la camara IR izquierda (camara color ~ coincidente; offset de ~1 cm ignorado) o VO.
"""
from pathlib import Path
import cv2
import numpy as np
from common import *

COLOR = Path('/mnt/datalake/datasets/rosariov2/sequences_extracted')
# intrinsecos color (kalibr, 1280x720) escalados a 640x360
KC = np.array([[890.4202450761666 / 2, 0, 633.5761943138773 / 2],
               [0, 895.5269973888081 / 2, 375.39479166983205 / 2], [0, 0, 1]])
DC = np.array([0.05112465195903441, -0.08817933012895941, -0.0011842812428454741, -0.00040533553962803756])
H_EFF = 1.30      # altura de la camara sobre el plano del canopeo bajo (1.44 m al suelo, ~0.15 m de planta)
RES = 0.02        # m por celda


class ColorSource:
    def __init__(self, seq, t_ir):
        d = COLOR / seq
        self.t2 = np.loadtxt(d / 'times_image_2.txt')
        self.dir = d / 'image_2'
        self.t_ir = t_ir
        v, u = np.mgrid[200:360:2, 0:640:2]     # filas de suelo cercano (~1.6-3.9 m adelante)
        self.uv = np.stack([u.ravel(), v.ravel()], 1).astype(np.float64)
        und = cv2.undistortPoints(self.uv.reshape(-1, 1, 2), KC, DC).reshape(-1, 2)
        self.rays = np.c_[und, np.ones(len(und))]   # en el marco de la camara (optico)

    def mask(self, k):
        j = int(np.argmin(np.abs(self.t2 - self.t_ir[k])))
        im = cv2.imread(str(self.dir / f'{j:06d}.png')).astype(np.float32)
        b, g, r = cv2.split(im)
        s = b + g + r + 1e-3
        exg = 2 * g / s - r / s - b / s
        exg = cv2.GaussianBlur(exg, (0, 0), 1.0)
        m = exg[self.uv[:, 1].astype(int), self.uv[:, 0].astype(int)] > 0.05
        return m.astype(np.float32)


def project(rays, Rwc, pwc, h=H_EFF):
    d = rays @ Rwc.T
    s = -h / d[:, 2]
    ok = (s > 0) & (s < 8)
    P = pwc[None, :2] + s[:, None] * d[:, :2]
    return P, ok


def build_bev(seq, frames, poses_R, poses_p, src, bounds, theta=0.0):
    """bounds = (xmin, xmax, ymin, ymax) en el marco rotado theta (rad) respecto del mundo:
    u = cos*x + sin*y (a lo largo de la hilera), w = -sin*x + cos*y (lateral). Devuelve (ocupacion, conteo)."""
    c_, s_ = np.cos(theta), np.sin(theta)
    xmin, xmax, ymin, ymax = bounds
    nx, ny = int((xmax - xmin) / RES), int((ymax - ymin) / RES)
    acc = np.zeros((ny, nx), np.float32); cnt = np.zeros((ny, nx), np.float32)
    for k in frames:
        m = src.mask(k)
        P, ok = project(src.rays, poses_R[k], poses_p[k])
        P = np.c_[c_ * P[:, 0] + s_ * P[:, 1], -s_ * P[:, 0] + c_ * P[:, 1]]
        ix = ((P[ok, 0] - xmin) / RES).astype(int); iy = ((P[ok, 1] - ymin) / RES).astype(int)
        good = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        np.add.at(acc, (iy[good], ix[good]), m[ok][good])
        np.add.at(cnt, (iy[good], ix[good]), 1)
    occ = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    return occ, cnt


class IRSource:
    """Mascara de vegetacion desde la IR izquierda (1280x720 rectificada): textura local alta y no saturada.
    En NIR el suelo arenoso satura (liso, ~255) y las hojas son brillantes pero texturadas; las sombras son oscuras."""
    def __init__(self, imgs, step=4, v0=400, sat=245, tex=6.0):
        self.imgs = imgs; self.sat = sat; self.tex = tex
        v, u = np.mgrid[v0:H:step, 0:W:step]
        self.uv = np.stack([u.ravel(), v.ravel()], 1)
        self.rays = np.c_[(self.uv[:, 0] - CX) / FX, (self.uv[:, 1] - CY) / FY, np.ones(len(self.uv))]

    def mask(self, k):
        I = cv2.imread(self.imgs[k], 0).astype(np.float32)
        mu = cv2.GaussianBlur(I, (0, 0), 3); sd = np.sqrt(np.maximum(cv2.GaussianBlur(I * I, (0, 0), 3) - mu * mu, 0))
        veg = (sd > self.tex) & (mu < self.sat) & (mu > 40)
        veg = cv2.morphologyEx(veg.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return veg[self.uv[:, 1], self.uv[:, 0]].astype(np.float32)
