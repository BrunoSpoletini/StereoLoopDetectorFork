"""TEST DE REALISMO: firma de hileras construida SOLO con odometria (sin GT) en ventanas locales.

  python3 exp_vo.py <seq> <tag> [--L 5 10 15] [--offset 6] [--pairs auto]

- Orientacion de la camara respecto del suelo: plano RANSAC sobre estereo (level.py), sin GT.
- Trayectoria local en el marco de las hileras: avance = odometro de SLD (VO); rumbo relativo a las hileras y
  posicion lateral = angulo y fase de las hileras observadas en cada frame (rowodo.py).
- Consulta: ventana de L m de odometro que termina en q (solo pasado). Mapa: ventana de L+2*SEARCH m alrededor
  de un candidato c de la pasada anterior. c = frame GT mas cercano a q desplazado un error aleatorio de
  +-offset m a lo largo (simula el error del candidato de SALAD). El GT solo se usa para elegir q, c y evaluar.
- Matching: (flip, k, du); flip = +-1 (sentido relativo), k = corrimiento entero de fila, du por NCC.
- Correcto: flip correcto, error lateral < media hilera (0.29 m) y error a lo largo < 1 m.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from common import *
from level import level_from_stereo, level_rotation
from rowodo import FrameObs, row_track, local_bev
from vo import load_vo

RESM = 0.02
SEARCH = 10.0
BAND = 0.10


def estimate_spacing(obs, frames):
    best = (0, 0)
    for sp in np.arange(0.40, 0.80, 0.005):
        q = np.mean([obs.rows_angle_phase(obs.mask(k), sp)[2] for k in frames])
        if q > best[0]: best = (q, sp)
    return best[1]


def rows_and_sigs(bev, u0, w0, w_track, half=1.6):
    prof = np.nanmean(bev, 1)
    pk, _ = find_peaks(np.nan_to_num(prof), distance=int(0.3 / RESM), prominence=0.1)
    ws = w0 + pk * RESM
    ws = ws[np.abs(ws - w_track) <= half]
    sigs = []
    for w in ws:
        r0, r1 = int((w - w0 - BAND) / RESM), int((w - w0 + BAND) / RESM) + 1
        s = np.nanmean(bev[r0:r1], 0)
        f = np.isfinite(s); s2 = np.where(f, s, np.nanmean(s) if f.any() else 0)
        s2 = gaussian_filter1d(s2, 2); s2[~f] = np.nan
        sigs.append(s2)
    return ws, sigs


def ncc_full(q, m, min_valid=0.8):
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(q)
    if len(m) < n: return np.full(0, np.nan)
    Mw = sliding_window_view(m, n)
    ok = np.isfinite(Mw) & np.isfinite(q)[None, :]
    cnt = ok.sum(1); c = np.maximum(cnt, 1)
    qv = np.where(ok, q[None, :], 0.0); mv = np.where(ok, Mw, 0.0)
    qa = np.where(ok, qv - (qv.sum(1) / c)[:, None], 0); ma = np.where(ok, mv - (mv.sum(1) / c)[:, None], 0)
    out = (qa * ma).sum(1) / (np.sqrt((qa * qa).sum(1) * (ma * ma).sum(1)) + 1e-9)
    out[cnt < min_valid * n] = np.nan
    return out


def match(Q, M, spacing, min_rows=2):
    """Q, M: dict(ws, sigs, u0). Devuelve lista de hipotesis (score, flip, k, t_u, t_w) ordenada."""
    hyps = []
    for s in (+1, -1):
        wq = s * Q['ws']
        sq = [x if s > 0 else x[::-1] for x in Q['sigs']]
        uq0 = Q['u0'] if s > 0 else -(Q['u0'] + len(Q['sigs'][0]) * RESM) if Q['sigs'] else 0
        order = np.argsort(wq); wq = wq[order]; sq = [sq[i] for i in order]
        if len(wq) < min_rows or len(M['ws']) < min_rows: continue
        iq = np.round((wq - wq[0]) / spacing).astype(int)
        im = np.round((M['ws'] - M['ws'][0]) / spacing).astype(int)
        for k in range(-iq.max() - 1, im.max() + 2):
            pairs = [(j, int(np.where(im == iq[j] + k)[0][0])) for j in range(len(iq)) if (iq[j] + k) in im]
            if len(pairs) < min_rows: continue
            zs = None; sc = None; n = 0
            for j, i in pairs:
                c = ncc_full(sq[j], M['sigs'][i])
                if c.size == 0: continue
                c = np.nan_to_num(c, nan=0.0)
                z = np.arctanh(np.clip(c, -0.99, 0.99))
                sc = c if sc is None else sc + c
                zs = z if zs is None else zs + z
                n += 1
            if sc is None or n < min_rows: continue
            sc = sc / n; zs = zs / np.sqrt(n)     # Stouffer: mas filas coherentes => mas evidencia
            idx = int(np.argmax(zs))
            t_u = M['u0'] + idx * RESM - uq0
            t_w = np.mean([M['ws'][i] - wq[j] for j, i in pairs])
            hyps.append((float(zs[idx]), s, k, float(t_u), float(t_w), len(pairs), zs, uq0, float(sc[idx])))
    hyps.sort(key=lambda h: -h[0])
    return hyps


def second_best(hyps):
    """Mejor score de una hipotesis distinta: otro (flip,k), o el mismo con |du| > 0.3 m."""
    b = hyps[0]
    best2 = -1
    for h in hyps[1:]:
        best2 = max(best2, h[0])
    sc = b[6].copy(); i = int(round((b[3] - (hyps[0][3])) / RESM))
    i0 = int(round((b[3] + b[7] - 0) / RESM))  # no usado
    ib = int(np.argmax(sc)); r = int(0.3 / RESM)
    sc[max(0, ib - r):ib + r + 1] = -1
    return max(best2, float(sc.max()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('seq'); ap.add_argument('tag')
    ap.add_argument('--L', type=float, nargs='+', default=[5, 10, 15])
    ap.add_argument('--offset', type=float, default=6.0)
    ap.add_argument('--every', type=float, default=5.0)
    ap.add_argument('--maxdw', type=float, default=3.5)
    ap.add_argument('--half', type=float, default=1.6)
    ap.add_argument('--sat_pct', type=float, default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--v0', type=int, default=400)
    ap.add_argument('--mindw', type=float, default=0.0)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    t, pos, rot, imgs, rimgs = load_seq(a.seq)
    vo, odo = load_vo(a.tag)
    cache = Path('/tmp/claude-1000/-home-bruno-Desktop-tesina/b1128bd6-a1d1-4537-bb53-b46ff6c35cf4/scratchpad/rows') / f'level_{a.seq}.json'
    if cache.exists():
        dl = json.load(open(cache)); nrm, hg = np.array(dl['n']), dl['h']
    else:
        nrm, hg = level_from_stereo(imgs, rimgs)
        json.dump(dict(n=list(map(float, nrm)), h=float(hg)), open(cache, 'w'))
    obs = FrameObs(imgs, level_rotation(nrm), hg - 0.14, sat_pct=a.sat_pct, v0=a.v0)
    # --- pasadas (GT solo para elegir) ---
    v = np.gradient(pos[:, :2], axis=0) * 15; sp = np.linalg.norm(v, axis=1); ang = np.arctan2(v[:, 1], v[:, 0])
    mov = sp > 0.3; theta = np.angle(np.mean(np.exp(2j * ang[mov]))) / 2
    dd = np.array([np.cos(theta), np.sin(theta)]); ll = np.array([-dd[1], dd[0]])
    U = pos[:, :2] @ dd; Wt = pos[:, :2] @ ll
    along = mov & (np.abs(np.cos(ang - theta)) > 0.97); dsg = np.where(along, np.sign(np.gradient(U)), 0)
    passes = []; st = None
    for i, x in enumerate(dsg):
        if st is None and x != 0: st = (i, x)
        elif st is not None and x != st[1]:
            if i - st[0] > 150: passes.append((st[0], i - 1, int(st[1])))
            st = (i, x) if x != 0 else None
    lo, hi = np.percentile(U[along], 0.5) + 15, np.percentile(U[along], 99.5) - 15
    spacing = estimate_spacing(obs, list(range(passes[0][0], passes[0][0] + 200, 20)))
    print(f'{a.seq}: pitch normal {np.round(nrm,3)}, h={hg:.2f}, espaciado={spacing:.3f} m, pasadas={len(passes)}')
    pinfo = [dict(s0=p[0], s1=p[1], dir=p[2], w=float(np.median(Wt[p[0]:p[1]]))) for p in passes]

    def window(k_end, back, fwd, p):
        fr = [k for k in range(p['s0'], p['s1'] + 1, 2) if odo[k_end] - back <= odo[k] <= odo[k_end] + fwd]
        return fr

    track_cache = {}

    def pass_track(pi):
        if pi not in track_cache:
            p = pinfo[pi]
            fr = list(range(p['s0'], p['s1'] + 1, 2))
            Uall, Wall, Pall, _ = row_track(obs, fr, odo, spacing)
            track_cache[pi] = {k: (Uall[j], Wall[j], Pall[j]) for j, k in enumerate(fr)}
        return track_cache[pi]

    def sliced(pi, fr):
        tr = pass_track(pi)
        Uw = np.array([tr[k][0] for k in fr]); Ww = np.array([tr[k][1] for k in fr]); Pw = np.array([tr[k][2] for k in fr])
        return Uw - Uw[0], Ww - Ww[0], Pw, None

    results = []
    for qi, qp in enumerate(pinfo):
        for mi, mp in enumerate(pinfo):
            if mp['s0'] >= qp['s0']: continue
            dw = abs(qp['w'] - mp['w'])
            if dw > a.maxdw or dw < a.mindw: continue
            # q a lo largo de la pasada consulta, en el interior y dentro del solape
            qs = [k for k in range(qp['s0'], qp['s1'], 2) if lo + 15 < U[k] < hi - 15 and
                  min(U[mp['s0']], U[mp['s1']]) + 15 < U[k] < max(U[mp['s0']], U[mp['s1']]) - 15]
            if not qs: continue
            last = -1e9
            for q in qs:
                if abs(odo[q] - last) < a.every: continue
                last = odo[q]
                mframes = np.arange(mp['s0'], mp['s1'])
                c_true = int(mframes[np.argmin(np.abs(U[mframes] - U[q]))])
                off = rng.uniform(-a.offset, a.offset)
                cands = mframes[np.argmin(np.abs(odo[mframes] - (odo[c_true] + off)))]
                c = int(cands)
                for L in a.L:
                    fq = window(q, L, 0, qp); fm = window(c, L + SEARCH, SEARCH, mp)
                    if len(fq) < 10 or len(fm) < 10: continue
                    Uq, Wq, Pq, _ = sliced(qi, fq)
                    Um, Wm, Pm, _ = sliced(mi, fm)
                    bq = local_bev(obs, fq, Uq, Wq, Pq, (-1, Uq[-1] + 6), (-6, 6))
                    bm = local_bev(obs, fm, Um, Wm, Pm, (-1, Um[-1] + 6), (-6, 6))
                    # recortar la consulta a la zona efectivamente observada en u
                    colsq = np.where(np.isfinite(bq).mean(0) > 0.3)[0]
                    if len(colsq) < 20: continue
                    bq = bq[:, colsq[0]:colsq[-1] + 1]; uq0 = -1 + colsq[0] * RESM
                    wq_track = np.median(Wq); wm_track = np.median(Wm)
                    wsq, sq = rows_and_sigs(bq, uq0, -6, wq_track, a.half)
                    wsm, sm = rows_and_sigs(bm, -1, -6, wm_track, a.half)
                    Qd = dict(ws=wsq, sigs=sq, u0=uq0); Md = dict(ws=wsm, sigs=sm, u0=-1)
                    hyps = match(Qd, Md, spacing)
                    if not hyps:
                        results.append(dict(q=q, c=c, L=L, dw=dw, same_dir=qp['dir'] == mp['dir'], ok=False, score=np.nan)); continue
                    sc, s, k, tu, tw = hyps[0][:5]
                    sec = second_best(hyps)
                    # posicion de q en el marco del mapa (origen = primer frame de fm)
                    jq = len(fq) - 1; jc = fm.index(c) if c in fm else int(np.argmin(np.abs(np.array(fm) - c)))
                    pu = s * Uq[jq] + tu - Um[jc]; pw = s * Wq[jq] + tw - Wm[jc]
                    # verdad GT: q respecto de c en el marco de hileras orientado con la marcha de la pasada mapa
                    dmap = mp['dir'] * dd; lmap = np.array([-dmap[1], dmap[0]])
                    gu = (pos[q, :2] - pos[c, :2]) @ dmap; gw = (pos[q, :2] - pos[c, :2]) @ lmap
                    flip_true = 1 if qp['dir'] == mp['dir'] else -1
                    eu, ew = pu - gu, pw - gw
                    ok = (s == flip_true) and abs(ew) < spacing / 2 and abs(eu) < 1.0
                    results.append(dict(q=q, c=c, L=L, dw=round(dw, 2), same_dir=bool(qp['dir'] == mp['dir']),
                                        off=round(off, 2), score=round(sc, 3), second=round(sec, 3), ncc=round(hyps[0][8], 3), flip=s, k=k,
                                        n_rows=hyps[0][5], err_u=round(eu, 2), err_w=round(ew, 2), ok=bool(ok)))
                    print(results[-1], flush=True)
    out = a.out or f'out/vo_{a.tag}.json'
    json.dump(results, open(out, 'w'))
    for L in a.L:
        for same in (False, True):
            r = [x for x in results if x['L'] == L and x['same_dir'] == same]
            if not r: continue
            for lo_dw, hi_dw in ((0, 1.6), (1.6, 99)):
                rr = [x for x in r if lo_dw <= x['dw'] < hi_dw]
                if not rr: continue
                okk = sum(x['ok'] for x in rr)
                scs = np.array([x['score'] for x in rr if np.isfinite(x.get('score', np.nan))])
                mg = np.array([x['score'] - x['second'] for x in rr if 'second' in x])
                print(f'L={L:>4} {"mismo" if same else "opuesto":7s} dw[{lo_dw},{hi_dw}): correctas {okk}/{len(rr)}'
                      f'  score {scs.mean():.2f}  margen {mg.mean() if len(mg) else np.nan:.2f}')
