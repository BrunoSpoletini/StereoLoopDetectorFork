"""Embudo por etapa sobre los frames revisita GT: donde se pierde cada revisita y si el candidato era correcto."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from datasets import select
from sld_eval import R_M, load_gt

STATUS = {0: 'loop', 1: 'recent', 2: 'no_db', 3: 'low_nss', 4: 'low_score', 5: 'no_groups',
          6: 'no_temporal', 7: 'no_geom', -1: 'skipped'}


def funnel(run, sessions='fieldsafe'):
    rows = []
    for s in select(sessions):
        gt = load_gt(s.poses)
        f = Path(__file__).parent / 'runs' / run / f'{s.name}_queries.csv'
        if not f.exists() or f.stat().st_size == 0 or 'num_images' not in f.with_name(f'{s.name}_results.yml').read_text():
            continue
        q = pd.read_csv(f)
        q = q[gt.revisit[q['query'].values]]
        cand = q.candidate.values
        ok = cand >= 0
        correct = np.zeros(len(q), bool)
        correct[ok] = np.linalg.norm(gt.xy[q['query'].values[ok]] - gt.xy[cand[ok]], axis=1) < R_M
        rows.append(pd.DataFrame(dict(session=s.name, status=q.status.map(STATUS), correct=correct,
                                      inliers=q.geom_inliers, matches=q.geom_matches)))
    df = pd.concat(rows)
    t = df.groupby('status').agg(n=('correct', 'size'), cand_correcto=('correct', 'sum'))
    t['frac'] = t.n / t.n.sum()
    return df, t


if __name__ == '__main__':
    df, t = funnel(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'fieldsafe')
    print(t.sort_values('n', ascending=False).to_string(float_format='{:.3f}'.format))
    g = df[(df.status == 'no_geom')]
    print('no_geom: matches med (cand correcto / incorrecto):',
          g[g.correct].matches.median(), g[~g.correct].matches.median(),
          ' inliers med:', g[g.correct].inliers.median(), g[~g.correct].inliers.median())
