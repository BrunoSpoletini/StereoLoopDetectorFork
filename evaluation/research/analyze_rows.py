"""Resumen de out/rows_<tag>.json: reglas de veto y generador del interior."""
import json, sys
import numpy as np
from common import load_seq
from vo import load_vo
SEQ = {'1226_1339': '2023-12-26-13-39-43', '1222_1429': '2023-12-22-14-29-43', '1222_1631': '2023-12-22-16-31-08',
       '1222_1314': '2023-12-22-13-14-16', '1226_1510': '2023-12-26-15-10-15', '1226_1548': '2023-12-26-15-48-38'}
TAU = [0.3, 0.4, 0.5]
tot = {}
for tag in sys.argv[1:]:
    R = json.load(open(f'out/rows_{tag}.json'))
    V = R['veto']; real = lambda r: r['dist'] < 3
    import re
    txt = open(f'/home/bruno/Desktop/tesina/StereoLoopDetectorFork/evaluation/runs/p6_odo/rof_{tag}_results.yml').read()
    sq = lambda k: np.array([float(x) for x in re.search(k + r':\s*\[(.*?)\]', txt, re.S).group(1).split(',') if x.strip()])
    TT = {(a_, b_): z_ for a_, b_, z_ in zip(sq('loop_query_ids').astype(int), sq('loop_match_ids').astype(int), sq('loop_translation_z'))}
    for r in V: r['tz'] = TT[(r['q'], r['m'])]
    # consistencia firma-PnP (convencion medida en 15:10: lateral = t_x, a lo largo = -t_z)
    cons = lambda r: abs(r['pw'] - r['tx']) <= 0.3 and abs(r['pu'] + r['tz']) <= 1.0
    nT = sum(map(real, V)); nF = len(V) - nT
    rules = {
        'sin veto': lambda r: True,
        'confirmar si PnP<0.8 y aplicable': lambda r: not (r['applicable'] and abs(r['tx']) < 0.8) or
            (r['margin'] >= 0.2 and abs(r['pu']) < 2 and abs(r['pw']) < 0.6),
        'confirmar (consistencia PnP) m>=0.2': lambda r: not r['applicable'] or (r['margin'] >= 0.2 and cons(r)),
        'vetar si contradice al PnP m>=0.3': lambda r: not (r['applicable'] and r['margin'] >= 0.3 and not cons(r)),
        'vetar si contradice al PnP m>=0.4': lambda r: not (r['applicable'] and r['margin'] >= 0.4 and not cons(r)),
    }
    print(f'== {tag}: loops {len(V)} (TP<3m {nT}, FP {nF}); firma aplicable (intrinseca) en {sum(r["applicable"] for r in V)} '
          f'(TP {sum(r["applicable"] and real(r) for r in V)}, FP {sum(r["applicable"] and not real(r) for r in V)})')
    for name, f in rules.items():
        k = [f(r) for r in V]
        kt = sum(1 for r, x in zip(V, k) if x and real(r)); kf = sum(1 for r, x in zip(V, k) if x and not real(r))
        print(f'   {name:34s}: conserva TP {kt}/{nT}, FP {kf}/{nF}, precision {kt/max(1,kt+kf):.3f}')
        tot.setdefault(name, [0, 0, 0, 0]); tot[name][0] += kt; tot[name][1] += nT; tot[name][2] += kf; tot[name][3] += nF
    # generador
    G = R['gen']
    t, pos, rot, imgs, _ = load_seq(SEQ[tag]); _, odo = load_vo(tag)
    for tau in TAU:
        A = [g for g in G if g['margin'] >= tau]
        AI = [g for g in A if g['interior']]
        ok = sum(g['ok'] for g in A); oki = sum(g['ok'] for g in AI)
        opp = sum(g['ok'] and g['flip'] < 0 for g in A)
        # cobertura del interior: tramos de 10 m de odometro (interior) con >=1 loop correcto del generador
        tr = lambda g: int(odo[g['q']] // 10)
        cov = {tr(g) for g in AI if g['ok']}
        print(f'   generador margen>={tau}: aceptados {len(A)} (interior {len(AI)}), correctos {ok} ({100*ok/max(1,len(A)):.0f}%), '
              f'interior correctos {oki}/{len(AI)}, sentido opuesto {opp}; tramos de 10 m del interior cubiertos: {len(cov)}')
        tot.setdefault(f'gen{tau}', [0, 0, 0]); tot[f'gen{tau}'][0] += len(A); tot[f'gen{tau}'][1] += ok; tot[f'gen{tau}'][2] += len(cov)
    allint = {int(odo[g['q']] // 10) for g in G if g['interior']}
    # cobertura del interior por los loops del pipeline (TP < 3 m) y union con el generador (margen >= 0.4)
    pipe = {int(odo[r['q']] // 10) for r in V if r['interior'] and real(r)}
    genc = {int(odo[g['q']] // 10) for g in G if g['interior'] and g['ok'] and g['margin'] >= 0.4}
    print(f'   tramos interior cubiertos: pipeline {len(pipe)}, generador {len(genc)}, union {len(pipe | genc)} (nuevos {len(genc - pipe)})')
    tot.setdefault('cov', [0, 0, 0]); tot['cov'][0] += len(pipe); tot['cov'][1] += len(genc - pipe); tot['cov'][2] += len(pipe | genc)
    print(f'   (consultas del generador: {len(G)}; tramos de 10 m del interior consultados {len(allint)})')
print('== TOTAL')
for k, v in tot.items():
    if k == 'cov':
        print(f'   tramos interior: pipeline {v[0]}, nuevos por el generador {v[1]}, union {v[2]}'); continue
    if k.startswith('gen'):
        print(f'   {k}: aceptados {v[0]}, correctos {v[1]} ({100*v[1]/max(1,v[0]):.0f}%), tramos interior cubiertos {v[2]}')
    else:
        print(f'   {k:34s}: TP {v[0]}/{v[1]}  FP {v[2]}/{v[3]}  precision {v[0]/max(1,v[0]+v[2]):.3f}')
