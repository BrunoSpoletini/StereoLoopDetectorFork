"""Precision de la odometria referida a hileras (sin GT) vs GT en ventanas de 15 m del interior; y costo por frame."""
import sys, time, json
import numpy as np
from common import *; from level import *; from rowodo import *; from vo import load_vo
seq, tag = sys.argv[1], sys.argv[2]
t, pos, rot, imgs, rimgs = load_seq(seq)
nrm, hg = level_from_stereo(imgs, rimgs)
obs = FrameObs(imgs, level_rotation(nrm), hg - 0.14)
vo, odo = load_vo(tag)
sp = float(sys.argv[3])
v = np.gradient(pos[:, :2], axis=0) * 15; ang = np.arctan2(v[:, 1], v[:, 0]); mv = np.linalg.norm(v, axis=1) > 0.3
th = np.angle(np.mean(np.exp(2j * ang[mv]))) / 2; d = np.array([np.cos(th), np.sin(th)])
rng = np.random.default_rng(1); out = []
tm = []
starts = [s for s in rng.integers(1000, len(imgs) - 400, 3000)
          if np.all(np.abs(np.cos(ang[s:s + 300] - th)) > 0.97) and mv[s:s + 300].all()][:15]
for s in starts:
    fr = list(range(s, s + 300, 2))
    t0 = time.time(); U, Wl, PS, Q = row_track(obs, fr, odo, sp); tm.append((time.time() - t0) / len(fr))
    dd = d if (pos[fr[-1], :2] - pos[fr[0], :2]) @ d > 0 else -d; ll = np.array([-dd[1], dd[0]])
    gu = (pos[fr, :2] - pos[fr[0], :2]) @ dd; gw = (pos[fr, :2] - pos[fr[0], :2]) @ ll
    out.append((gu[-1], U[-1] - gu[-1], Wl[-1] - gw[-1], np.abs(Wl - gw).max()))
o = np.array(out)
print(f'{seq}: n={len(o)} ventanas ~{o[:,0].mean():.1f} m; err u final: media {o[:,1].mean():+.2f} m ({100*np.mean(o[:,1]/o[:,0]):+.1f}%), '
      f'|err w| final mediana {np.median(np.abs(o[:,2])):.3f} p90 {np.percentile(np.abs(o[:,2]),90):.3f}; max|err w| mediana {np.median(o[:,3]):.3f}; '
      f'tiempo {1000*np.mean(tm):.1f} ms/frame (mascara+angulo+fase, CPU)')
