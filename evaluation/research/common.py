"""Utilidades para el estudio exploratorio de firmas de hileras (huecos de cultivo) en RosarioV2 full-res."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path('/home/bruno/Desktop/tesina/datasets/rosariov2_fullres')
PREP = ROOT / 'prepared'
FX = FY = 647.00646972
CX, CY = 648.23236084, 350.12701416
BASE = 16.24343300 * 2 / FX   # m (P_right[0,3] = -fx*B en 640x360 -> fB=16.24 px.m a fx=323.5)
W, H = 1280, 720


def load_seq(seq):
    g = np.loadtxt(PREP / f'gt_{seq}.csv', delimiter=',')
    imgs = [l.strip() for l in open(PREP / f'left_{seq}.txt')]
    rimgs = [l.strip() for l in open(PREP / f'right_{seq}.txt')]
    assert len(imgs) == len(g)
    pos = g[:, 2:5]
    rot = R.from_quat(g[:, 5:9]).as_matrix()   # camara (optica: z adelante, y abajo) -> mundo
    return g[:, 1], pos, rot, imgs, rimgs
