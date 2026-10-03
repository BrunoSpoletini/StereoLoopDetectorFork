"""(solo matching, reutiliza los BEV guardados) Experimento sistematico: firmas de hileras ("codigo de barras") entre todas las pasadas del interior de una sesion.

  python3 exp_all.py <seq> <tag_vo> [--odo]

- Segmenta la trayectoria GT en pasadas rectas a lo largo de las hileras; descarta 15 m en cada cabecera.
- BEV de vegetacion (ExG, camara color) por pasada en el marco de las hileras (u a lo largo, w lateral).
  Con --odo, la coordenada u de cada frame se reemplaza por la del odometro de SLD (sin GPS), anclada
  localmente cada 10 m (simula construir el mapa sin GT: error de escala/odometro dentro de la ventana).
- Para cada par de pasadas con >= 2 hileras observadas por ambas y ventana de 10 m: matching conjunto
  (k, du). Tambien un "impostor" (misma consulta contra el mapa desplazado 40 m) para la distribucion de
  scores de lugares equivocados.
"""
import argparse
import json
import numpy as np
from common import *
from bev import *
from bev import RES
from vo import load_vo
from barcode import barcode_match, summarize
import sig1d
from sig1d import rows

ap = argparse.ArgumentParser()
ap.add_argument('seq'); ap.add_argument('tag'); ap.add_argument('--odo', action='store_true')
ap.add_argument('--L', type=float, default=10.0); ap.add_argument('--half', type=float, default=1.6)
a = ap.parse_args()
OUT = Path('/tmp/claude-1000/-home-bruno-Desktop-tesina/b1128bd6-a1d1-4537-bb53-b46ff6c35cf4/scratchpad/rows') / a.seq
OUT.mkdir(parents=True, exist_ok=True)
RES_DIR = Path(__file__).parent / 'out' / a.seq; RES_DIR.mkdir(parents=True, exist_ok=True)
t, pos, rot, imgs, _ = load_seq(a.seq)
src = ColorSource(a.seq, t)
vo, odo = load_vo(a.tag)

# --- direccion de hileras: direccion media de movimiento en tramos rectos (refinada con el BEV abajo)
v = np.gradient(pos[:, :2], axis=0) * 15; sp = np.linalg.norm(v, axis=1)
ang = np.arctan2(v[:, 1], v[:, 0])
mov = sp > 0.3
a2 = np.angle(np.mean(np.exp(2j * ang[mov]))) / 2          # direccion (mod 180)
theta = a2
c_, s_ = np.cos(theta), np.sin(theta)
U = pos[:, 0] * c_ + pos[:, 1] * s_; Wt = -pos[:, 0] * s_ + pos[:, 1] * c_
du_dt = np.gradient(U) * 15
along = mov & (np.abs(np.cos(ang - theta)) > 0.97)
d = np.where(along, np.sign(du_dt), 0)
passes = []; st = None
for i, x in enumerate(d):
    if st is None and x != 0: st = (i, x)
    elif st is not None and x != st[1]:
        if i - st[0] > 150: passes.append((st[0], i - 1, int(st[1])))
        st = (i, x) if x != 0 else None
umin, umax = np.percentile(U[along], 0.5), np.percentile(U[along], 99.5)
lo, hi = umin + 15, umax - 15     # interior
print(f'theta={np.degrees(theta):.2f} deg, u in [{umin:.1f},{umax:.1f}], interior [{lo:.1f},{hi:.1f}], pasadas={len(passes)}')

# --- refinar theta con la varianza del perfil lateral del BEV de la pasada mas larga
# (se omite: el angulo de movimiento ya coincide con el de hileras a ~0.1 deg en 13:39)

bounds = tuple(json.load(open(OUT / 'info.json'))['bounds'])
sig1d.U0 = bounds[0]; sig1d.W0 = bounds[2]
info = json.load(open(OUT / 'info.json'))['passes']
# --- matching entre pares
res = []
from functools import lru_cache


@lru_cache(maxsize=8)
def getmap(i):
    return np.load(OUT / f'p{i}{"_odo" if a.odo else ""}.npy').astype(np.float32)
for qi in info:
    for mi in info:
        if mi['s0'] >= qi['s0']: continue        # mapa = pasada anterior
        dw = abs(qi['w'] - mi['w'])
        if dw > 2 * a.half: continue
        u0 = max(qi['u0'], mi['u0'], lo) + 1; u1 = min(qi['u1'], mi['u1'], hi) - 1
        Q, M = getmap(qi['i']), getmap(mi['i'])
        for ua in np.arange(u0, u1 - a.L + 1e-6, 5.0):
            ub = ua + a.L
            wq = rows(Q, ua, ub)
            wq = wq[(np.abs(wq - qi['w']) <= a.half) & (np.abs(wq - mi['w']) <= a.half)]
            if len(wq) < 2: continue
            sc, wm, ms = barcode_match(Q, M, ua, ub, wq)
            if not sc: continue
            r = summarize(sc, ms)
            # impostor: mapa desplazado 40 m (si existe)
            imp = np.nan
            for off in (40, -40):
                if mi['u0'] + 2 < ua + off - 10 and ua + off + a.L + 10 < mi['u1'] - 2 and lo < ua + off and ua + off + a.L < hi:
                    Mi = np.roll(M, -int(off / RES), axis=1)
                    sci, _, msi = barcode_match(Q, Mi, ua, ub, wq)
                    if sci: imp = summarize(sci, msi)['score']
                    break
            res.append(dict(q=qi['i'], m=mi['i'], qdir=qi['dir'], mdir=mi['dir'], dw=dw, ua=float(ua),
                            n_rows=int(len(wq)), score=float(r['score']), k=int(r['k']), du=float(r['du']),
                            second=float(r['second']), imp=float(imp)))
json.dump(res, open(RES_DIR / f'res{"_odo" if a.odo else ""}.json', 'w'), indent=0)
R_ = res
ok = [x for x in R_ if x['k'] == 0 and abs(x['du']) < 1.0]
print(f'ventanas={len(R_)}  correctas(k=0,|du|<1m)={len(ok)} ({100*len(ok)/max(1,len(R_)):.0f}%)')
if R_:
    sc = np.array([x['score'] for x in R_]); mg = np.array([x['score'] - x['second'] for x in R_])
    imp = np.array([x['imp'] for x in R_]); imp = imp[np.isfinite(imp)]
    print('score medio %.2f, margen medio %.2f; impostor: n=%d media %.2f p95 %.2f max %.2f' % (
        sc.mean(), mg.mean(), len(imp), imp.mean() if len(imp) else np.nan,
        np.percentile(imp, 95) if len(imp) else np.nan, imp.max() if len(imp) else np.nan))
