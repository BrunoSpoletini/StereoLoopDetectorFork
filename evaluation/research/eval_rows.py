"""Evalua sobre la secuencia COMPLETA (sin GT en las decisiones; GT solo para medir):
  veto: reglas de veto/confirmacion de la firma sobre todos los loops de runs/<run>/rof_<tag>_results.yml
  gen : generador de loops del interior (consulta cada STEP m contra todos los segmentos anteriores)
  python3 eval_rows.py <seq> <tag> [--run p6_odo] [--L 10] [--step 2]
"""
import argparse, json, re
import numpy as np
from common import *
from vo import load_vo
from segsig import State, query_window, match_segment, best_and_margin, RES

ap = argparse.ArgumentParser()
ap.add_argument('seq'); ap.add_argument('tag')
ap.add_argument('--run', default='p6_odo'); ap.add_argument('--L', type=float, default=10.0)
ap.add_argument('--step', type=float, default=2.0); ap.add_argument('--no-gen', action='store_true'); ap.add_argument('--tagout', default=''); ap.add_argument('--qthr', type=float, default=0.25)
a = ap.parse_args()
t, pos, rot, imgs, _ = load_seq(a.seq)
_, odo = load_vo(a.tag)
st = State(a.seq, odo, qthr=a.qthr)
F = st.frames
print(f'{a.seq}: espaciado {st.spacing:.3f}, frames con hileras {100*st.good.mean():.0f}%, segmentos {len(st.segs)}, '
      f'largo total {sum(st.odo[j]-st.odo[i] for i,j in st.segs):.0f} m', flush=True)
SEG = [st.segment(si) for si in range(len(st.segs))]
seg_of = np.full(len(F), -1)
for si, (i0, i1) in enumerate(st.segs): seg_of[i0:i1 + 1] = si

# --- GT (solo evaluacion)
v = np.gradient(pos[:, :2], axis=0) * 15; spd = np.linalg.norm(v, axis=1); ang = np.arctan2(v[:, 1], v[:, 0]); mov = spd > 0.3
th = np.angle(np.mean(np.exp(2j * ang[mov]))) / 2; dd = np.array([np.cos(th), np.sin(th)]); ll = np.array([-dd[1], dd[0]])
Ug = pos[:, :2] @ dd; al = mov & (np.abs(np.cos(ang - th)) > 0.97)
lo, hi = np.percentile(Ug[al], 0.5), np.percentile(Ug[al], 99.5)


def gt_rel(q, c):
    dirc = np.sign((pos[min(c + 15, len(pos) - 1), :2] - pos[max(c - 15, 0), :2]) @ dd) or 1
    return (pos[q, :2] - pos[c, :2]) @ (dirc * dd), (pos[q, :2] - pos[c, :2]) @ (dirc * ll)


def run_query(iq, candidates):
    si = seg_of[iq]
    Q = query_window(SEG[si], iq, a.L)
    if Q is None: return None
    H = []
    for sj in candidates:
        for h in match_segment(Q, SEG[sj], st.spacing):
            H.append(h + (sj,))
    if not H: return None
    b, mg = best_and_margin(H)
    sj = b[8]; S = SEG[sj]
    jc = int(np.argmin(np.abs(S['U'] - b[3])))
    c = int(F[S['i0'] + jc])
    return dict(q=int(F[iq]), c=c, z=b[0], margin=float(mg), flip=b[1], pu=float(b[3] - S['U'][jc]),
                pw=float(b[4] - S['W'][jc]), n_rows=b[5], cand_seg=int(sj))


res = dict(seq=a.seq, segs=[(int(F[i]), int(F[j])) for i, j in st.segs])
# ---------------- generador ----------------
gen = []
for si, (i0, i1) in (enumerate(st.segs) if not a.no_gen else []):
    earlier = [sj for sj, (j0, j1) in enumerate(st.segs) if j1 < i0 and st.odo[i0] - st.odo[j1] > 5]
    if not earlier: continue
    last = -1e9
    for iq in range(i0, i1 + 1):
        if st.odo[iq] - last < a.step: continue
        if st.odo[iq] - st.odo[i0] < a.L: continue
        last = st.odo[iq]
        r = run_query(iq, earlier)
        if r is None: continue
        q, c = r['q'], r['c']
        gu, gw = gt_rel(q, c)
        r.update(gu=round(float(gu), 2), gw=round(float(gw), 2), dist=float(np.linalg.norm(pos[q, :2] - pos[c, :2])),
                 interior=bool(lo + 15 < Ug[q] < hi - 15), ok=bool(abs(r['pu'] - gu) < 1.0 and abs(r['pw'] - gw) < 0.3),
                 Ugq=float(Ug[q]))
        gen.append(r)
print(f'generador: {len(gen)} consultas', flush=True)
res['gen'] = gen
# ---------------- veto ----------------
txt = open(f'/home/bruno/Desktop/tesina/StereoLoopDetectorFork/evaluation/runs/{a.run}/rof_{a.tag}_results.yml').read()
seqv = lambda k: np.array([float(x) for x in re.search(k + r':\s*\[(.*?)\]', txt, re.S).group(1).split(',') if x.strip()]) \
    if re.search(k + r':\s*\[(.*?)\]', txt, re.S) else np.array([])
Ql, Ml, tx = seqv('loop_query_ids').astype(int), seqv('loop_match_ids').astype(int), seqv('loop_translation_x')
idx_of = {int(f): i for i, f in enumerate(F)}
veto = []
for q, m, x in zip(Ql, Ml, tx):
    iq = idx_of.get(q - q % 2); im = idx_of.get(m - m % 2)
    rec = dict(q=int(q), m=int(m), tx=float(x), dist=float(np.linalg.norm(pos[q, :2] - pos[m, :2])),
               interior=bool(lo + 15 < Ug[q] < hi - 15))
    gu, gw = gt_rel(q, m); rec.update(gu=float(gu), gw=float(gw))
    sq_ = seg_of[iq] if iq is not None else -1; sm_ = seg_of[im] if im is not None else -1
    rec['applicable'] = bool(sq_ >= 0 and sm_ >= 0 and sq_ != sm_ and st.odo[iq] - st.odo[st.segs[sq_][0]] >= a.L)
    if iq is not None:
        wq = [i for i in range(max(0, iq - 200), iq + 1) if st.odo[iq] - st.odo[i] <= a.L]
        rec['qual_q'] = float(np.mean(st.qual[wq])); rec['psi_q'] = float(np.degrees(np.max(np.abs(st.psi[wq]))))
    if im is not None:
        wm = [i for i in range(max(0, im - 200), min(len(F), im + 200)) if abs(st.odo[im] - st.odo[i]) <= 5]
        rec['qual_m'] = float(np.mean(st.qual[wm]))
    rec['qthr'] = float(st.qthr)
    if rec['applicable']:
        Q = query_window(SEG[sq_], iq, a.L)
        H = [h + (sm_,) for h in match_segment(Q, SEG[sm_], st.spacing)] if Q is not None else []
        if H:
            b, mg = best_and_margin(H)
            S = SEG[sm_]; jm = im - S['i0']
            rec.update(z=b[0], margin=float(mg), pu=float(b[3] - S['U'][jm]), pw=float(b[4] - S['W'][jm]), flip=b[1])
        else:
            rec['applicable'] = False
    veto.append(rec)
res['veto'] = veto
json.dump(res, open(f'out/rows_{a.tag}{a.tagout}.json', 'w'))
print(f'veto: {len(veto)} loops, aplicable en {sum(r["applicable"] for r in veto)}', flush=True)
