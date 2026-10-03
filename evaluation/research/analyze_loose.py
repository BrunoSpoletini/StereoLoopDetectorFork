"""Reglas de veto con aplicabilidad mas amplia (segmentos con umbral 0.45*p75) y calidad de ventana."""
import json, re, sys
import numpy as np
rules = {}
tot = {}
for tag in sys.argv[1:]:
    V = json.load(open(f'out/rows_{tag}_loose.json'))['veto']
    txt = open(f'/home/bruno/Desktop/tesina/StereoLoopDetectorFork/evaluation/runs/p6_odo/rof_{tag}_results.yml').read()
    sq = lambda k: np.array([float(x) for x in re.search(k + r':\s*\[(.*?)\]', txt, re.S).group(1).split(',') if x.strip()])
    TT = {(a, b): z for a, b, z in zip(sq('loop_query_ids').astype(int), sq('loop_match_ids').astype(int), sq('loop_translation_z'))}
    for r in V: r['tz'] = TT[(r['q'], r['m'])]
    real = lambda r: r['dist'] < 3
    cons = lambda r: abs(r['pw'] - r['tx']) <= 0.3 and abs(r['pu'] + r['tz']) <= 1.0
    R = {
        'sin veto': lambda r: True,
        'confirmar si aplicable (m>=0.2, consistente)': lambda r: not r['applicable'] or (r['margin'] >= 0.2 and cons(r)),
        'confirmar si aplicable y PnP<0.8': lambda r: not (r['applicable'] and abs(r['tx']) < 0.8) or (r['margin'] >= 0.2 and cons(r)),
        'contradice (aplicable, m>=0.3)': lambda r: not (r['applicable'] and r['margin'] >= 0.3 and not cons(r)),
        'contradice (aplicable, m>=0.2)': lambda r: not (r['applicable'] and r['margin'] >= 0.2 and not cons(r)),
    }
    nT = sum(map(real, V)); nF = len(V) - nT
    A = [r for r in V if r['applicable']]
    print(f'== {tag}: {len(V)} loops (TP {nT}, FP {nF}); aplicable {len(A)} (TP {sum(map(real, A))}, FP {len(A)-sum(map(real, A))})')
    for name, f in R.items():
        kt = sum(1 for r in V if f(r) and real(r)); kf = sum(1 for r in V if f(r) and not real(r))
        print(f'   {name:46s}: TP {kt}/{nT}  FP {kf}/{nF}')
        t = tot.setdefault(name, [0, 0, 0, 0]); t[0] += kt; t[1] += nT; t[2] += kf; t[3] += nF
    # de los aplicables: margen y consistencia por clase
    for lab, S in (('TP', [r for r in A if real(r)]), ('FP', [r for r in A if not real(r)])):
        if S:
            mg = np.array([r['margin'] for r in S]); c = np.array([cons(r) for r in S])
            print(f'   aplicables {lab}: margen mediano {np.median(mg):.2f}; margen>=0.3 {100*np.mean(mg>=0.3):.0f}%; consistente con PnP {100*np.mean(c):.0f}%; '
                  f'm>=0.3 y NO consistente {int(np.sum((mg>=0.3)&~c))}')
print('== TOTAL')
for k, v in tot.items():
    print(f'   {k:46s}: TP {v[0]}/{v[1]}  FP {v[2]}/{v[3]}  precision {v[0]/max(1,v[0]+v[2]):.4f}')
