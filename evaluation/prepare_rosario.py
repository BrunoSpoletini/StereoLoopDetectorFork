"""Prepara una secuencia de RosarioV2 para correr StereoLoopDetector.

Toma las imagenes ya extraidas (image_0/image_1 + times_image_0.txt) y las poses
ground-truth de mins_tum, y escribe en <outdir>:
  left_<seq>.txt / right_<seq>.txt   paths de las imagenes IR izq/der
  poses_<seq>.csv                    'id,x,y,0,0,0,0' (formato de demo_stereo)
  gt_<seq>.csv                       'id,t,x,y,z,qx,qy,qz,qw' pose completa por frame

Solo se conservan los frames cubiertos por el rango temporal del ground truth
(no se extrapola ni se clampea).
"""
import argparse
from pathlib import Path

import numpy as np

ROOT = Path('/mnt/datalake/datasets/rosariov2')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('seq')
    ap.add_argument('--outdir', default=str(ROOT / 'prepared'))
    ap.add_argument('--extracted', default=str(ROOT / 'sequences_extracted'),
                    help='raiz de las imagenes extraidas (p. ej. la version a resolucion completa)')
    a = ap.parse_args()

    ext = Path(a.extracted) / a.seq
    t_img = np.loadtxt(ext / 'times_image_0.txt')
    gt = np.loadtxt(ROOT / 'sequences_plain' / 'mins_tum' / f'{a.seq}_mins_tum.csv')
    gt = gt[np.argsort(gt[:, 0])]

    keep = np.where((t_img >= gt[0, 0]) & (t_img <= gt[-1, 0]))[0]
    t = t_img[keep]
    pose = np.stack([np.interp(t, gt[:, 0], gt[:, c]) for c in range(1, 8)], axis=1)
    q = pose[:, 3:]
    pose[:, 3:] = q / np.linalg.norm(q, axis=1, keepdims=True)

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f'left_{a.seq}.txt', 'w') as fl, open(out / f'right_{a.seq}.txt', 'w') as fr:
        for i in keep:
            fl.write(f'{ext}/image_0/{i:06d}.png\n')
            fr.write(f'{ext}/image_1/{i:06d}.png\n')
    ids = np.arange(len(keep))
    np.savetxt(out / f'poses_{a.seq}.csv',
               np.column_stack([ids, pose[:, 0], pose[:, 1], np.zeros((len(ids), 4))]),
               delimiter=',', fmt=['%d', '%.6f', '%.6f', '%d', '%d', '%d', '%d'])
    np.savetxt(out / f'gt_{a.seq}.csv', np.column_stack([ids, t, pose]), delimiter=',',
               fmt=['%d', '%.6f'] + ['%.6f'] * 7)
    path = np.linalg.norm(np.diff(pose[:, :2], axis=0), axis=1).sum()
    print(f'{a.seq}: {len(keep)}/{len(t_img)} frames con GT, recorrido {path:.0f} m')


if __name__ == '__main__':
    main()
