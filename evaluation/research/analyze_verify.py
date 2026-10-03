"""Regla de veto condicionada al PnP: la firma solo se exige cuando el PnP afirma que q y c estan a < W_PNP m
laterales (regimen donde la firma es informativa)."""
import json, re, sys
import numpy as np
for tag in sys.argv[1:]:
    txt = open(f'/home/bruno/Desktop/tesina/StereoLoopDetectorFork/evaluation/runs/p5_aliked/rof_{tag}_results.yml').read()
    seq = lambda k: np.array([float(x) for x in re.search(k + r':\s*\[(.*?)\]', txt, re.S).group(1).split(',') if x.strip()])
    Q, M, tx, tz = seq('loop_query_ids').astype(int), seq('loop_match_ids').astype(int), seq('loop_translation_x'), seq('loop_translation_z')
    T = {(q, c): (x, z) for q, c, x, z in zip(Q, M, tx, tz)}
    o = json.load(open(f'out/verify_{tag}.json'))
    for r in o:
        r['tx'], r['tz'] = T[(r['q'], r['c'])]
    R_ = [r for r in o if r['real']]; F_ = [r for r in o if not r['real']]
    print(f'== {tag}: reales {len(R_)}, falsos {len(F_)}')
    print('   |t_x PnP| reales: mediana %.2f (GT |gw| %.2f) | falsos: mediana %.2f (GT |gw| %.2f)' % (
        np.median([abs(r['tx']) for r in R_]) if R_ else np.nan, np.median([abs(r['gw']) for r in R_]) if R_ else np.nan,
        np.median([abs(r['tx']) for r in F_]), np.median([abs(r['gw']) for r in F_])))
    for wp in (0.8, 1.2):
        for thr in (0.2, 0.3):
            def keep(r):
                if abs(r['tx']) >= wp: return True               # PnP dice "pasada vecina": la firma no aplica
                return r['margin'] >= thr and abs(r['pu']) < 2 and abs(r['pw']) < 0.6
            kr = sum(map(keep, R_)); kf = sum(map(keep, F_))
            print(f'   W_PNP={wp} margen>={thr}: conserva reales {kr}/{len(R_)}, falsos {kf}/{len(F_)}')
