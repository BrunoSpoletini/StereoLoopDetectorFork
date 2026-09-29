"""Descriptores globales por imagen (izquierda) para recuperacion de candidatos.

  python global_desc.py --model salad --sessions fieldsafe

Guarda <CACHE>/<dataset>_<seq>_<model>.npy: float16 (n_imagenes, D), normalizado L2.
Modelos:
  dinov2_gem  DINOv2 ViT-B/14 sin ajuste, GeM sobre los tokens de parche (estilo AnyLoc sin vocabulario)
  salad       DINOv2 + SALAD (Izquierdo & Civera, CVPR 2024), entrenado para VPR
Correr con el python del entorno conda mast3r-slam (tiene torch + CUDA).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "shims"))  # pytorch_lightning para SALAD
from datasets import CACHE, select

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_model(name):
    if name == 'dinov2_gem':
        m = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

        def fwd(x):
            tok = m.forward_features(x)['x_norm_patchtokens']       # (b, n, c)
            g = tok.clamp(min=1e-6).pow(3).mean(1).pow(1 / 3)       # GeM p=3
            return F.normalize(g, dim=-1)
        return m, fwd
    if name == 'salad':
        m = torch.hub.load('serizba/salad', 'dinov2_salad')
        return m, lambda x: F.normalize(m(x), dim=-1)
    raise ValueError(name)


def load_batch(paths, size):
    ims = []
    for p in paths:
        im = Image.open(p).convert('RGB').resize(size, Image.BILINEAR)
        ims.append(torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255)
    return (torch.stack(ims) - MEAN) / STD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='salad')
    ap.add_argument('--sessions', default='fieldsafe')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--width', type=int, default=518)   # multiplos de 14
    ap.add_argument('--height', type=int, default=280)
    a = ap.parse_args()

    model, fwd = load_model(a.model)
    model = model.eval().cuda()
    for s in select(a.sessions):
        out = CACHE / f'{s.dataset}_{s.seq}_{a.model}.npy'
        if out.exists():
            print(f'{s.name}: ya existe'); continue
        paths = s.left.read_text().split()
        descs = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            for i in range(0, len(paths), a.batch):
                x = load_batch(paths[i:i + a.batch], (a.width, a.height)).cuda()
                descs.append(fwd(x).float().cpu().numpy().astype(np.float16))
                if i % (a.batch * 50) == 0:
                    print(f'  {s.name}: {i}/{len(paths)}', flush=True)
        np.save(out, np.concatenate(descs))
        print(f'{s.name}: {out}')


if __name__ == '__main__':
    main()
