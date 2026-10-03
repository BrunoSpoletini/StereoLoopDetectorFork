"""Veto de loops por firma de hileras (huecos de cultivo), condicionado al PnP. Sin GT.

Para cada loop aceptado (q, m):
  - si el PnP afirma una separacion lateral >= --w-pnp (pasada vecina) o el robot no va recto siguiendo
    hileras en q o en m (cabecera, giro), el loop se conserva: la firma no es informativa ahi;
  - si no, se construye la firma de vegetacion de los ultimos L m antes de q y de +-(L+5) m alrededor de m
    (odometro de SLD + angulo/fase de hileras medidos en cada frame; ver research/rowodo.py) y se exige que
    confirme q ~ m: margen >= --margin y pose de la firma a < 2 m a lo largo y < 0.6 m lateral.
Polaridad de la mascara IR (suelo saturado vs suelo oscuro) elegida automaticamente por coherencia de fase.

  python sig_veto.py p6_odo --out p8_sig --sessions rosariofr
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'research'))
from datasets import select                                         # noqa: E402
from learned_verify import RUNS, write_results                       # noqa: E402
from sld_eval import load_run                                        # noqa: E402
from level import level_from_stereo, level_rotation                  # noqa: E402
from rowodo import FrameObs, row_track, local_bev                    # noqa: E402
from exp_vo import rows_and_sigs, match, second_best, RESM, estimate_spacing   # noqa: E402

LEVEL_DIR = HERE / 'research' / 'out'


def get_level(s):
    f = LEVEL_DIR / f'level_{s.seq}.json'
    if f.exists():
        d = json.loads(f.read_text())
    else:
        imgs, rimgs = s.left.read_text().split(), s.right.read_text().split()
        n, h = level_from_stereo(imgs, rimgs)
        d = dict(n=[float(x) for x in n], h=h)
        f.write_text(json.dumps(d))
    return np.array(d['n']), d['h']


def choose_polarity(imgs, Rlev, h_eff, spacing=0.52, n=12):
    """Polaridad con mayor coherencia de fase de hileras en frames de muestra (None = suelo saturado)."""
    best = (None, -1)
    for sat in (None, -50.0):
        obs = FrameObs(imgs, Rlev, h_eff, sat_pct=sat)
        coh = np.median([obs.rows_angle_phase(obs.mask(k), spacing)[2]
                         for k in np.linspace(len(imgs) * .1, len(imgs) * .9, n).astype(int)])
        if coh > best[1]:
            best = (sat, coh)
    return best[0]


def straight_mask(vo_yaw, odo, win=2.0, tol_deg=6.0):
    """Frames donde el rumbo de la odometria no cambia mas de tol en +-win metros (siguiendo una hilera)."""
    yaw = np.unwrap(vo_yaw)
    lo = np.searchsorted(odo, odo - win); hi = np.clip(np.searchsorted(odo, odo + win), 0, len(odo) - 1)
    return np.abs(yaw[hi] - yaw[lo]) < np.radians(tol_deg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('loops_run')
    ap.add_argument('--vo-run', default='p3c5vo')
    ap.add_argument('--out', default='p8_sig')
    ap.add_argument('--sessions', default='rosariofr')
    ap.add_argument('--L', type=float, default=10.0)
    ap.add_argument('--w-pnp', type=float, default=0.8)
    ap.add_argument('--margin', type=float, default=0.2)
    a = ap.parse_args()
    out = RUNS / a.out
    out.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    for s in select(a.sessions):
        res, q, m, t, r = load_run(RUNS / a.loops_run / f'{s.name}_results.yml')
        vo = pd.read_csv(RUNS / a.vo_run / f'{s.name}_queries.csv').set_index('query').sort_index()
        odo = vo.odometer.values
        straight = straight_mask(vo.vo_yaw.values, odo)
        imgs = s.left.read_text().split()
        nrm, h = get_level(s)
        Rlev = level_rotation(nrm)
        sat = choose_polarity(imgs, Rlev, h - 0.14)
        obs = FrameObs(imgs, Rlev, h - 0.14, sat_pct=sat)
        # memoizar el angulo/fase de hileras por mascara: los loops consecutivos comparten casi todos los frames
        _rap, _memo = obs.rows_angle_phase, {}

        def rows_angle_phase(mk, sp, _rap=_rap, _memo=_memo):
            key = (id(mk), sp)
            if key not in _memo:
                _memo[key] = (_rap(mk, sp), mk)     # se guarda la mascara para que su id no se reutilice
            return _memo[key][0]
        obs.rows_angle_phase = rows_angle_phase
        spacing = estimate_spacing(obs, np.linspace(len(imgs) * .2, len(imgs) * .8, 20).astype(int))

        def window(k, back, fwd):
            i = k
            while i > 0 and odo[k] - odo[i - 1] <= back and straight[i - 1]: i -= 1
            j = k
            while j < len(odo) - 1 and odo[j + 1] - odo[k] <= fwd and straight[j + 1]: j += 1
            return list(range(i, j + 1, 2))

        keep, rows = [], []
        for k, (qi, mi) in enumerate(zip(q, m)):
            decision, info = True, 'pnp_lateral'
            if abs(t[k, 0]) < a.w_pnp:
                info = 'curva'
                if straight[qi] and straight[mi]:
                    fq, fm = window(qi, a.L, 0), window(mi, a.L + 5, a.L + 5)
                    info = 'ventana_corta'
                    if len(fq) >= 30 and len(fm) >= 30:
                        Uq, Wq, Pq, _ = row_track(obs, fq, odo, spacing)
                        Um, Wm, Pm, _ = row_track(obs, fm, odo, spacing)
                        bq = local_bev(obs, fq, Uq, Wq, Pq, (-1, Uq[-1] + 6), (-6, 6))
                        bm = local_bev(obs, fm, Um, Wm, Pm, (-1, Um[-1] + 6), (-6, 6))
                        cols = np.where(np.isfinite(bq).mean(0) > 0.3)[0]
                        info = 'sin_firma'
                        decision = False
                        if len(cols) >= 20:
                            bq = bq[:, cols[0]:cols[-1] + 1]; uq0 = -1 + cols[0] * RESM
                            wsq, sq = rows_and_sigs(bq, uq0, -6, np.median(Wq))
                            wsm, sm = rows_and_sigs(bm, -1, -6, np.median(Wm))
                            H = match(dict(ws=wsq, sigs=sq, u0=uq0), dict(ws=wsm, sigs=sm, u0=-1), spacing)
                            if H:
                                z, sgn, kk, tu, tw = H[0][:5]
                                mg = z - second_best(H)
                                jc = int(np.argmin(np.abs(np.array(fm) - mi)))
                                pu = sgn * Uq[-1] + tu - Um[jc]; pw = sgn * Wq[-1] + tw - Wm[jc]
                                decision = mg >= a.margin and abs(pu) < 2.0 and abs(pw) < 0.6
                                info = f'firma margen {mg:.2f} pu {pu:.2f} pw {pw:.2f}'
            keep.append(decision)
            rows.append(dict(query=int(qi), match=int(mi), keep=decision, info=info))
            if k % 200 == 0:
                print(f'  {s.name}: {k}/{len(q)}', flush=True)
        keep = np.array(keep, bool)
        pd.DataFrame(rows).to_csv(out / f'{s.name}_veto.csv', index=False)
        recs = [dict(query=int(q[i]), candidate=int(m[i]), t=list(t[i]), r=list(r[i])) for i in np.where(keep)[0]]
        write_results(out / f'{s.name}_results.yml', res['num_images'], recs)
        print(f'{s.name}: polaridad {sat}, espaciado {spacing:.3f} m, {int(keep.sum())}/{len(q)} loops conservados',
              flush=True)


if __name__ == '__main__':
    main()
