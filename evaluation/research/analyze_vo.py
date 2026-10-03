"""Resume out/vo_*.json: correctas por L / sentido / separacion lateral, y precision-recall al aceptar por margen."""
import json, sys
import numpy as np
for f in sys.argv[1:]:
    R = [r for r in json.load(open(f)) if 'second' in r]
    print(f'== {f}: {len(R)} pruebas')
    for L in sorted({r['L'] for r in R}):
        for same in (False, True):
            for lo, hi in ((0, 1.0), (1.0, 1.6), (1.6, 2.6), (2.6, 99)):
                rr = [r for r in R if r['L'] == L and r['same_dir'] == same and lo <= r['dw'] < hi]
                if not rr: continue
                ok = np.array([r['ok'] for r in rr]); z = np.array([r['score'] for r in rr])
                mg = np.array([r['score'] - r['second'] for r in rr]); ncc = np.array([r['ncc'] for r in rr])
                eu = np.array([abs(r['err_u']) for r in rr if r['ok']]); ew = np.array([abs(r['err_w']) for r in rr if r['ok']])
                print(f'L={L:>4} {"mismo  " if same else "opuesto"} dw[{lo},{hi}): {ok.sum()}/{len(ok)} correctas | '
                      f'z {z.mean():.2f} ncc {ncc.mean():.2f} margen(ok) {mg[ok].mean() if ok.any() else np.nan:.2f} '
                      f'margen(mal) {mg[~ok].mean() if (~ok).any() else np.nan:.2f} | err ok: |u| {np.median(eu) if len(eu) else np.nan:.2f} |w| {np.median(ew) if len(ew) else np.nan:.2f}')
        rr = [r for r in R if r['L'] == L]
        ok = np.array([r['ok'] for r in rr]); mg = np.array([r['score'] - r['second'] for r in rr])
        for th in (0.1, 0.2, 0.3, 0.4):
            acc = mg >= th
            print(f'   L={L} aceptar si margen>={th}: aceptadas {acc.sum()}, correctas {int((acc & ok).sum())} '
                  f'(precision {100*(acc & ok).sum()/max(1,acc.sum()):.0f}%), recall sobre correctas posibles {100*(acc & ok).sum()/max(1,ok.sum()):.0f}%')
