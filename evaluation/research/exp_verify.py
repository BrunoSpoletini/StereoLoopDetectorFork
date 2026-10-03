"""La firma de hileras como verificador de los loops del pipeline (SALAD+ALIKED+PnP, runs/p5_aliked) en el interior.

Para cada loop (q, c) del interior: firma de los ultimos L m antes de q contra una ventana de +-(L+5) m
alrededor de c (frames contiguos en el tiempo; sin GT). Se reporta la mejor hipotesis (sentido, du, dw) y su
margen. Evaluacion con GT: real = |p_q - p_c| < 3 m.
"""
import sys, json
import numpy as np
import cv2
from common import *
from level import level_rotation
from rowodo import FrameObs, row_track, local_bev
from vo import load_vo
from exp_vo import rows_and_sigs, match, second_best, RESM

seq, tag, run = sys.argv[1], sys.argv[2], sys.argv[3]
sat = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != 'none' else None
nmax = int(sys.argv[5]) if len(sys.argv) > 5 else 40
L = 10.0
t, pos, rot, imgs, rimgs = load_seq(seq)
vo, odo = load_vo(tag)
dl = json.load(open(f'/tmp/claude-1000/-home-bruno-Desktop-tesina/b1128bd6-a1d1-4537-bb53-b46ff6c35cf4/scratchpad/rows/level_{seq}.json'))
obs = FrameObs(imgs, level_rotation(np.array(dl['n'])), dl['h'] - 0.14, sat_pct=sat)
import re
_txt = open(f'/home/bruno/Desktop/tesina/StereoLoopDetectorFork/evaluation/runs/{run}/rof_{tag}_results.yml').read()


def _seq(key):
    m = re.search(key + r':\s*\[(.*?)\]', _txt, re.S)
    return np.array([float(x) for x in m.group(1).replace('\n', ' ').split(',') if x.strip()]) if m else np.array([])


Q = _seq('loop_query_ids').astype(int); M = _seq('loop_match_ids').astype(int)
# interior (GT solo para seleccionar y evaluar)
v = np.gradient(pos[:, :2], axis=0) * 15; sp = np.linalg.norm(v, axis=1); ang = np.arctan2(v[:, 1], v[:, 0]); mov = sp > 0.3
th = np.angle(np.mean(np.exp(2j * ang[mov]))) / 2; dd = np.array([np.cos(th), np.sin(th)]); ll = np.array([-dd[1], dd[0]])
U = pos[:, :2] @ dd
al = mov & (np.abs(np.cos(ang - th)) > 0.97)
lo, hi = np.percentile(U[al], 0.5) + 15, np.percentile(U[al], 99.5) - 15
inter = [(q, c) for q, c in zip(Q, M) if lo < U[q] < hi and lo < U[c] < hi and al[q] and al[c]]
dist = np.array([np.linalg.norm(pos[q, :2] - pos[c, :2]) for q, c in inter])
real = [p for p, d in zip(inter, dist) if d < 3]; fake = [p for p, d in zip(inter, dist) if d >= 3]
print(f'{seq}: loops {len(Q)}, interior {len(inter)} (reales {len(real)}, falsos {len(fake)})', flush=True)
rng = np.random.default_rng(0)
sel = [(p, True) for p in (real if len(real) <= nmax else [real[i] for i in rng.choice(len(real), nmax, replace=False)])] + \
      [(p, False) for p in (fake if len(fake) <= nmax else [fake[i] for i in rng.choice(len(fake), nmax, replace=False)])]
spacing = float(sys.argv[6]) if len(sys.argv) > 6 else 0.505


def contiguous(k, back, fwd):
    a = k
    while a > 0 and odo[k] - odo[a - 1] <= back and al[a - 1]: a -= 1
    b = k
    while b < len(odo) - 1 and odo[b + 1] - odo[k] <= fwd and al[b + 1]: b += 1
    return list(range(a, b + 1, 2))


out = []
for (q, c), is_real in sel:
    fq = contiguous(q, L, 0); fm = contiguous(c, L + 5, L + 5)
    if len(fq) < 30 or len(fm) < 30:
        continue
    Uq, Wq, Pq, _ = row_track(obs, fq, odo, spacing); Um, Wm, Pm, _ = row_track(obs, fm, odo, spacing)
    bq = local_bev(obs, fq, Uq, Wq, Pq, (-1, Uq[-1] + 6), (-6, 6)); bm = local_bev(obs, fm, Um, Wm, Pm, (-1, Um[-1] + 6), (-6, 6))
    cols = np.where(np.isfinite(bq).mean(0) > 0.3)[0]
    if len(cols) < 20: continue
    bq = bq[:, cols[0]:cols[-1] + 1]; uq0 = -1 + cols[0] * RESM
    wsq, sq = rows_and_sigs(bq, uq0, -6, np.median(Wq)); wsm, sm = rows_and_sigs(bm, -1, -6, np.median(Wm))
    H = match(dict(ws=wsq, sigs=sq, u0=uq0), dict(ws=wsm, sigs=sm, u0=-1), spacing)
    if not H: continue
    z, s, k, tu, tw = H[0][:5]; mg = z - second_best(H)
    jc = int(np.argmin(np.abs(np.array(fm) - c)))
    pu = s * Uq[-1] + tu - Um[jc]; pw = s * Wq[-1] + tw - Wm[jc]
    # GT de q respecto de c en el marco de hileras orientado con la marcha en c
    dirc = np.sign((pos[min(c + 15, len(pos) - 1), :2] - pos[c, :2]) @ dd) or 1
    gu = (pos[q, :2] - pos[c, :2]) @ (dirc * dd); gw = (pos[q, :2] - pos[c, :2]) @ (dirc * ll)
    out.append(dict(q=int(q), c=int(c), real=is_real, dist=float(np.linalg.norm(pos[q, :2] - pos[c, :2])),
                    gu=round(float(gu), 2), gw=round(float(gw), 2), z=round(z, 3), margin=round(float(mg), 3),
                    flip=int(s), pu=round(float(pu), 2), pw=round(float(pw), 2)))
    print(out[-1], flush=True)
json.dump(out, open(f'out/verify_{tag}.json', 'w'))
o = out
for thr in (0.2, 0.3, 0.4):
    conf = lambda r: r['margin'] >= thr and abs(r['pu']) < 2.0 and abs(r['pw']) < 0.6   # "la firma confirma q ~ c"
    R_ = [r for r in o if r['real']]; F_ = [r for r in o if not r['real']]
    print(f'margen>={thr}: reales confirmados {sum(map(conf, R_))}/{len(R_)}  falsos confirmados {sum(map(conf, F_))}/{len(F_)} '
          f'-> como veto: elimina {len(F_) - sum(map(conf, F_))}/{len(F_)} falsos y {len(R_) - sum(map(conf, R_))}/{len(R_)} reales')
# la firma corrige? (pose de la firma vs GT cuando el margen es alto)
good = [r for r in o if r['margin'] >= 0.3]
if good:
    e = np.array([(abs(r['pu'] - r['gu']), abs(r['pw'] - r['gw'])) for r in good])
    print(f'margen>=0.3: {len(good)} casos; pose de la firma vs GT: |e_u| mediana {np.median(e[:,0]):.2f}, |e_w| mediana {np.median(e[:,1]):.2f}; '
          f'correctas (|e_u|<1, |e_w|<0.25): {int(np.sum((e[:,0]<1)&(e[:,1]<0.25)))}')
