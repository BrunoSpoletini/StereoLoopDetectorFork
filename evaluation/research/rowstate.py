"""Precomputa, para TODA una secuencia (cada 2 frames), el estado de hileras observado por la camara IR, SIN GT:
mascara de vegetacion (empaquetada en bits), angulo de hileras psi, fase lateral y calidad intrinseca
(coherencia de fase normalizada en [0,1]). Polaridad de la mascara y espaciado elegidos por coherencia.

  python3 rowstate.py <seq>     -> scratchpad/rows/state_<seq>.npz
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter
from common import *
from level import level_rotation
from rowodo import FrameObs

SP = Path('/tmp/claude-1000/-home-bruno-Desktop-tesina/b1128bd6-a1d1-4537-bb53-b46ff6c35cf4/scratchpad/rows')
LEVEL = Path(__file__).parent / 'out'
COARSE = np.radians(np.arange(-12, 12.01, 1.0))


def coh(xy, mv, a, sp):
    w = -xy[:, 0] * np.cos(a) + xy[:, 1] * np.sin(a)
    mc = mv - mv.mean()
    z = np.sum(mc * np.exp(2j * np.pi * w / sp))
    return abs(z) / (np.sqrt(np.sum(mc * mc) * len(mc)) + 1e-9), np.angle(z) / (2 * np.pi) * sp


def best_angle(xy, mv, sp):
    c = [coh(xy, mv, a, sp)[0] for a in COARSE]
    a0 = COARSE[int(np.argmax(c))]
    fine = a0 + np.radians(np.arange(-0.9, 0.91, 0.1))
    c2 = [coh(xy, mv, a, sp)[0] for a in fine]
    return fine[int(np.argmax(c2))], max(c2)


if __name__ == '__main__':
    seq = sys.argv[1]
    t, pos, rot, imgs, _ = load_seq(seq)
    d = json.load(open(LEVEL / f'level_{seq}.json'))
    Rlev = level_rotation(np.array(d['n'])); heff = d['h'] - 0.14
    sample = np.linspace(len(imgs) * .1, len(imgs) * .9, 16).astype(int)
    best = None
    for sat in (None, -50.0):
        obs = FrameObs(imgs, Rlev, heff, sat_pct=sat)
        ms = [obs.mask(k) for k in sample]
        for sp in np.arange(0.40, 0.80, 0.01):
            q = np.median([coh(obs.xy[obs.ok], m[obs.ok], 0.0, sp)[0] for m in ms])
            if best is None or q > best[0]: best = (q, sat, sp)
    _, sat, spacing = best
    obs = FrameObs(imgs, Rlev, heff, sat_pct=sat)
    # refinar el espaciado con angulo libre
    ms = [obs.mask(k) for k in sample]
    spacing = max(np.arange(spacing - 0.03, spacing + 0.031, 0.005),
                  key=lambda sp: np.median([best_angle(obs.xy[obs.ok], m[obs.ok], sp)[1] for m in ms]))
    print(f'{seq}: polaridad {"suelo saturado" if sat is None else "suelo oscuro"}, espaciado {spacing:.3f}', flush=True)
    frames = np.arange(0, len(imgs), 2)
    nb = int(obs.ok.sum())
    bits = np.zeros((len(frames), (nb + 7) // 8), np.uint8)
    psi_raw = np.zeros(len(frames)); qual = np.zeros(len(frames))
    xy = obs.xy[obs.ok]
    for i, k in enumerate(frames):
        m = obs._mask(k)[obs.ok]
        bits[i] = np.packbits(m > 0.5)
        psi_raw[i], qual[i] = best_angle(xy, m, spacing)
        if i % 2000 == 0: print(f'  {i}/{len(frames)}', flush=True)
    psi = median_filter(psi_raw, size=31, mode='nearest')
    phase = np.zeros(len(frames)); qual_s = np.zeros(len(frames))
    for i, k in enumerate(frames):
        m = np.unpackbits(bits[i])[:nb].astype(np.float32)
        qual_s[i], phase[i] = coh(xy, m, psi[i], spacing)
    np.savez_compressed(SP / f'state_{seq}.npz', frames=frames, bits=bits, nb=nb, psi=psi, psi_raw=psi_raw,
                        phase=phase, qual=qual_s, spacing=spacing, sat=np.nan if sat is None else sat,
                        n=np.array(d['n']), heff=heff)
    print(f'{seq}: listo; calidad mediana {np.median(qual_s):.3f}', flush=True)
