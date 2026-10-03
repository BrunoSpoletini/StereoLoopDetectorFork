"""Diagnostico (con GT, solo para analizar) de las decisiones de sig_veto: por que los reales no dan margen."""
import re, sys
import numpy as np
sys.path.insert(0, 'research')
from common import load_seq
SEQ = {'1226_1339': '2023-12-26-13-39-43', '1222_1429': '2023-12-22-14-29-43', '1222_1631': '2023-12-22-16-31-08',
       '1222_1314': '2023-12-22-13-14-16', '1226_1510': '2023-12-26-15-10-15', '1226_1548': '2023-12-26-15-48-38'}


def load_csv(f):
    rows = []
    for l in open(f).read().splitlines()[1:]:
        q, m, keep, info = l.split(',', 3)
        rows.append((int(q), int(m), keep == 'True', info))
    return rows


def geom(tag):
    t, pos, rot, imgs, _ = load_seq(SEQ[tag])
    v = np.gradient(pos[:, :2], axis=0) * 15; sp = np.linalg.norm(v, axis=1); ang = np.arctan2(v[:, 1], v[:, 0]); mov = sp > 0.3
    th = np.angle(np.mean(np.exp(2j * ang[mov]))) / 2; dd = np.array([np.cos(th), np.sin(th)]); ll = np.array([-dd[1], dd[0]])
    U = pos[:, :2] @ dd
    al = mov & (np.abs(np.cos(ang - th)) > 0.97)
    lo, hi = np.percentile(U[al], 0.5), np.percentile(U[al], 99.5)
    return pos, dd, ll, U, lo, hi, al


if __name__ == '__main__':
    for tag in sys.argv[1:]:
        rows = load_csv(f'runs/p8_sig/rof_{tag}_veto.csv')
        pos, dd, ll, U, lo, hi, al = geom(tag)
        F = []
        for q, m, keep, info in rows:
            if not info.startswith('firma'): continue
            mg, pu, pw = [float(x) for x in re.findall(r'-?\d+\.\d+', info)]
            d = np.linalg.norm(pos[q, :2] - pos[m, :2])
            dirm = np.sign((pos[min(m + 15, len(pos) - 1), :2] - pos[max(m - 15, 0), :2]) @ dd) or 1
            dirq = np.sign((pos[min(q + 15, len(pos) - 1), :2] - pos[max(q - 15, 0), :2]) @ dd) or 1
            gu = (pos[q, :2] - pos[m, :2]) @ (dirm * dd); gw = (pos[q, :2] - pos[m, :2]) @ (dirm * ll)
            edge = min(U[q] - lo, hi - U[q])
            F.append(dict(q=q, m=m, keep=keep, mg=mg, pu=pu, pw=pw, d=d, gu=gu, gw=gw, same=dirq == dirm, edge=edge, al=al[q] and al[m]))
        R = [f for f in F if f['d'] < 3]; X = [f for f in F if f['d'] >= 3]
        print(f'== {tag}: firma evaluada en {len(F)} loops (reales<3m {len(R)}, falsos {len(X)})')
        if not R: continue
        a = lambda key, S: np.array([s[key] for s in S])
        print('  reales: |gw| mediana %.2f (p10 %.2f p90 %.2f); |gu| mediana %.2f; distancia a cabecera mediana %.1f m (frac < 15 m: %.0f%%); mismo sentido %.0f%%; '
              'en recta (GT) %.0f%%; margen mediano %.2f' % (
                  np.median(np.abs(a('gw', R))), np.percentile(np.abs(a('gw', R)), 10), np.percentile(np.abs(a('gw', R)), 90),
                  np.median(np.abs(a('gu', R))), np.median(a('edge', R)), 100 * np.mean(a('edge', R) < 15),
                  100 * np.mean(a('same', R)), 100 * np.mean(a('al', R)), np.median(a('mg', R))))
        for name, sel in [('cabecera (<15 m)', lambda f: f['edge'] < 15), ('interior', lambda f: f['edge'] >= 15)]:
            S = [f for f in R if sel(f)]
            if not S: continue
            for lab, s2 in [('|gw|<0.8', lambda f: abs(f['gw']) < 0.8), ('|gw|>=0.8', lambda f: abs(f['gw']) >= 0.8)]:
                T = [f for f in S if s2(f)]
                if not T: continue
                ok = [abs(f['pu'] - f['gu']) < 1 and abs(f['pw'] - f['gw']) < 0.3 for f in T]
                print(f'    reales {name:16s} {lab:9s}: n={len(T):4d} margen mediano {np.median(a("mg", T)):.2f} '
                      f'margen>=0.2 {100*np.mean(a("mg", T) >= 0.2):.0f}%  firma correcta (pose) {100*np.mean(ok):.0f}%')
