"""Conteo de hileras en las cabeceras: el desplazamiento lateral entre el final de una pasada y el inicio de la
siguiente, medido con la VO (sin GT), permite asignar el indice de hilera de la nueva pasada?"""
import sys
import numpy as np
from common import *
from vo import load_vo
seq, tag, sp = sys.argv[1], sys.argv[2], float(sys.argv[3])
t, pos, rot, imgs, _ = load_seq(seq)
vo, odo = load_vo(tag)
v = np.gradient(pos[:, :2], axis=0) * 15; spd = np.linalg.norm(v, axis=1); ang = np.arctan2(v[:, 1], v[:, 0])
mov = spd > 0.3; th = np.angle(np.mean(np.exp(2j * ang[mov]))) / 2
dd = np.array([np.cos(th), np.sin(th)]); ll = np.array([-dd[1], dd[0]])
U = pos[:, :2] @ dd; Wt = pos[:, :2] @ ll
along = mov & (np.abs(np.cos(ang - th)) > 0.97); dsg = np.where(along, np.sign(np.gradient(U)), 0)
passes = []; st = None
for i, x in enumerate(dsg):
    if st is None and x != 0: st = (i, x)
    elif st is not None and x != st[1]:
        if i - st[0] > 150: passes.append((st[0], i - 1, int(st[1])))
        st = (i, x) if x != 0 else None
# fusionar fragmentos contiguos del mismo sentido; quedarse con transiciones con cambio de sentido (cabeceras)
P = [list(passes[0])]
for p in passes[1:]:
    if p[2] == P[-1][2] and p[0] - P[-1][1] < 150: P[-1][1] = p[1]
    else: P.append(list(p))
P3 = np.c_[vo[:, 0], vo[:, 1]]                       # posicion VO (x der, z adel del primer frame)
errs = []
for a, b in zip(P[:-1], P[1:]):
    if a[2] == b[2]: continue
    e, s = a[1], b[0]
    # direccion de hileras en el marco VO: direccion de movimiento VO en los ultimos 10 m de la pasada a
    k0 = [k for k in range(a[0], e) if odo[e] - odo[k] < 10][0]
    dv = P3[e] - P3[k0]; dv /= np.linalg.norm(dv); nv = np.array([-dv[1], dv[0]])
    k1 = [k for k in range(s, b[1]) if odo[k] - odo[s] > 3][0]     # 3 m dentro de la nueva pasada
    dw_vo = (P3[k1] - P3[e]) @ nv
    dw_gt = (pos[k1, :2] - pos[e, :2]) @ (ll * a[2])               # mismo sentido de normal (izq. de la pasada a)
    # el signo de la normal VO depende de la convencion de ejes; se alinea con el GT por el signo de la mediana
    errs.append((dw_gt, dw_vo, odo[k1] - odo[e]))
E = np.array(errs)
sgn = np.sign(np.median(E[:, 0] * E[:, 1]))
err = sgn * E[:, 1] - E[:, 0]
print(f'{seq}: cabeceras={len(E)}  |dw| GT mediana {np.median(np.abs(E[:,0])):.2f} m, recorrido en la cabecera mediana {np.median(E[:,2]):.1f} m')
print(f'  error lateral VO: mediana |e| {np.median(np.abs(err)):.2f} m, p90 {np.percentile(np.abs(err),90):.2f} m, max {np.abs(err).max():.2f} m')
print(f'  indice de hilera correcto (|e| < {sp/2:.2f} m): {np.sum(np.abs(err) < sp/2)}/{len(err)}')
