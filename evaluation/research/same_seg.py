"""Regla intrinseca: un loop cuyo q y m caen en el MISMO segmento continuo de seguimiento de hileras es imposible
(el robot avanzo en linea recta por la hilera entre ambos). Cuenta TP/FP afectados (GT solo para medir)."""
import json, sys
import numpy as np
from vo import load_vo
for tag in sys.argv[1:]:
    R = json.load(open(f'out/rows_{tag}.json'))
    segs = R['segs']
    def seg(f):
        for i, (a, b) in enumerate(segs):
            if a <= f <= b: return i
        return -1
    V = R['veto']
    same = [r for r in V if seg(r['q']) >= 0 and seg(r['q']) == seg(r['m'])]
    tp = sum(r['dist'] < 3 for r in same); fp = len(same) - tp
    nT = sum(r['dist'] < 3 for r in V); nF = len(V) - nT
    print(f'{tag}: mismo segmento: {len(same)} loops (TP {tp}/{nT}, FP {fp}/{nF}); '
          f'precision {nT/max(1,len(V)):.3f} -> {(nT-tp)/max(1,len(V)-len(same)):.3f}')
    for r in [r for r in same if r['dist'] < 3][:3]: print('   TP afectado', r['q'], r['m'], round(r['dist'], 2), round(r['gu'], 2))
