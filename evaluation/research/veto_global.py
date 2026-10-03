"""Veto "por ubicacion confiable": la firma de la ventana de q se busca contra TODOS los segmentos anteriores (como el
generador). Si la firma ubica a q con confianza (margen >= TAU) en otro lugar distinto al que dice el loop (m), se
veta; si lo ubica en el mismo lugar o no tiene confianza, se conserva. Sin GT en la decision.
  python3 veto_global.py <seq> <tag> [--qthr -0.7]
"""
import argparse, json, re
import numpy as np
from common import *
from vo import load_vo
from segsig import State, query_window, match_segment, best_and_margin

ap = argparse.ArgumentParser()
ap.add_argument('seq'); ap.add_argument('tag'); ap.add_argument('--qthr', type=float, default=-0.7)
ap.add_argument('--L', type=float, default=10.0)
a = ap.parse_args()
t, pos, rot, imgs, _ = load_seq(a.seq)
_, odo = load_vo(a.tag)
st = State(a.seq, odo, qthr=a.qthr); F = st.frames
SEG = [st.segment(si) for si in range(len(st.segs))]
seg_of = np.full(len(F), -1)
for si, (i0, i1) in enumerate(st.segs): seg_of[i0:i1 + 1] = si
txt = open(f'/home/bruno/Desktop/tesina/StereoLoopDetectorFork/evaluation/runs/p6_odo/rof_{a.tag}_results.yml').read()
sq = lambda k: np.array([float(x) for x in re.search(k + r':\s*\[(.*?)\]', txt, re.S).group(1).split(',') if x.strip()])
Ql, Ml, tx, tz = sq('loop_query_ids').astype(int), sq('loop_match_ids').astype(int), sq('loop_translation_x'), sq('loop_translation_z')
idx = {int(f): i for i, f in enumerate(F)}
cache = {}
out = []
for q, m, x, z in zip(Ql, Ml, tx, tz):
    iq = idx[q - q % 2]; im = idx[m - m % 2]
    rec = dict(q=int(q), m=int(m), tx=float(x), tz=float(z), dist=float(np.linalg.norm(pos[q, :2] - pos[m, :2])), loc=False)
    si = seg_of[iq]
    if si >= 0 and st.odo[iq] - st.odo[st.segs[si][0]] >= a.L:
        if iq not in cache:
            Q = query_window(SEG[si], iq, a.L)
            earlier = [sj for sj, (j0, j1) in enumerate(st.segs) if j1 < st.segs[si][0]]
            H = []
            if Q is not None:
                for sj in earlier:
                    H += [h + (sj,) for h in match_segment(Q, SEG[sj], st.spacing)]
            cache[iq] = best_and_margin(H) if H else (None, np.nan)
        b, mg = cache[iq]
        if b is not None:
            sj = b[8]; S = SEG[sj]
            rec.update(loc=True, margin=float(mg), seg=int(sj), m_in_seg=bool(seg_of[im] == sj))
            if seg_of[im] == sj:
                jm = im - S['i0']
                rec.update(pu=float(b[3] - S['U'][jm]), pw=float(b[4] - S['W'][jm]))
    out.append(rec)
json.dump(out, open(f'out/vglobal_{a.tag}.json', 'w'))
real = lambda r: r['dist'] < 3
cons = lambda r: r.get('m_in_seg') and abs(r['pw'] - r['tx']) <= 0.3 and abs(r['pu'] + r['tz']) <= 1.0
nT = sum(map(real, out)); nF = len(out) - nT
print(f'{a.seq}: {len(out)} loops (TP {nT}, FP {nF}); firma con ubicacion propia en {sum(r["loc"] for r in out)}')
for tau in (0.3, 0.4, 0.5):
    veto = lambda r: r['loc'] and r['margin'] >= tau and not cons(r)
    vt = sum(1 for r in out if veto(r) and real(r)); vf = sum(1 for r in out if veto(r) and not real(r))
    print(f'  TAU={tau}: veta TP {vt}/{nT}, FP {vf}/{nF}')
