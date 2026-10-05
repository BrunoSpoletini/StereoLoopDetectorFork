"""Carga de corridas para el visor: una tabla con una fila por loop detectado.

Las corridas viven en evaluation/finalRuns/<corrida>/ (versionadas, van al paper) y evaluation/testRuns/<corrida>/
(pruebas, no versionadas). Cada carpeta tiene un <sesion>_results.yml por sesion, con <sesion> uno de los nombres
de SESSIONS (fs_dynamic1, ro_1222_1314, rof_1222_1314, ...) y, opcionalmente, <sesion>_queries.csv y README.md.

El ground truth es el csv de poses `id,x,y,...` de cada sesion (el mismo que usa demo_stereo) en la carpeta
`prepared` de su dataset. Las carpetas se pueden cambiar con las variables de entorno SLD_FIELDSAFE, SLD_ROSARIO
y SLD_ROSARIO_FR o desde la barra lateral del visor.
"""
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EVAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL))
from commons import STOPPED_PATH_M, load_ground_truth, load_results, translation_errors  # noqa: E402

SOURCES = {'finalRuns': EVAL / 'finalRuns', 'testRuns': EVAL / 'testRuns'}
FAR_M = 5.0  # query y match a mas de esto segun el GT: no son el mismo lugar

DATASETS = {  # dataset -> (nombre para mostrar, variable de entorno, carpeta prepared por defecto)
    'fieldsafe': ('FieldSAFE', 'SLD_FIELDSAFE', '/mnt/datalake/datasets/fieldsafe/prepared'),
    'rosario': ('RosarioV2 640×360', 'SLD_ROSARIO', '/mnt/datalake/datasets/rosariov2/prepared'),
    'rosariofr': ('RosarioV2 1280×720', 'SLD_ROSARIO_FR', '/home/bruno/Desktop/tesina/datasets/rosariov2_fullres/prepared'),
}
CLASSES = ['en movimiento', 'detenido', 'lejano']


@dataclass
class Session:
    dataset: str
    seq: str
    label: str
    frequency: float


SESSIONS = {
    'fs_static1': Session('fieldsafe', '2016-10-25-11-09-42', 'Estática #1', 10.0),
    'fs_11-34': Session('fieldsafe', '2016-10-25-11-34-25', '11:34', 10.0),
    'fs_dynamic1': Session('fieldsafe', '2016-10-25-11-41-21', 'Dinámica #1', 10.0),
    'fs_dynamic2': Session('fieldsafe', '2016-10-25-12-07-22', 'Dinámica #2', 10.0),
    'fs_12-37': Session('fieldsafe', '2016-10-25-12-37-57', '12:37', 10.0),
}
for _seq in ['2023-12-22-13-14-16', '2023-12-22-14-29-43', '2023-12-22-16-31-08',
             '2023-12-26-13-39-43', '2023-12-26-15-10-15', '2023-12-26-15-48-38']:
    _short = f'{_seq[5:7]}{_seq[8:10]}_{_seq[11:13]}{_seq[14:16]}'
    _label = f'{_seq[8:10]}/{_seq[5:7]} {_seq[11:13]}:{_seq[14:16]}'
    SESSIONS[f'ro_{_short}'] = Session('rosario', _seq, _label, 15.0)
    SESSIONS[f'rof_{_short}'] = Session('rosariofr', _seq, _label, 15.0)


def default_roots():
    return {ds: os.environ.get(env, default) for ds, (_, env, default) in DATASETS.items()}


def session_label(name):
    """'fs_dynamic1' -> 'FS · Dinámica #1', 'rof_1222_1314' -> 'ROF · 22/12 13:14'."""
    s = SESSIONS.get(name)
    return f"{name.split('_')[0].upper()} · {s.label}" if s else name


@dataclass
class Run:
    id: str           # '<fuente>/<carpeta>', p. ej. 'finalRuns/03_distancia_minima_20m'
    source: str
    name: str
    path: Path
    sessions: list
    description: str


def _description(readme):
    """Primera linea del README: '# 03_x — descripcion' -> 'descripcion'."""
    if not readme.exists():
        return ''
    first = readme.read_text(encoding='utf-8').strip().splitlines()[0].lstrip('# ').strip()
    return re.split(r'\s+[—–-]\s+', first, maxsplit=1)[-1]


def list_runs(sources=SOURCES):
    """Corridas de cada fuente: subcarpetas con al menos un <sesion>_results.yml, en orden de nombre."""
    runs = []
    for source, root in sources.items():
        if not Path(root).is_dir():
            continue
        for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
            sessions = sorted(p.name[:-len('_results.yml')] for p in d.glob('*_results.yml'))
            if sessions:
                runs.append(Run(f'{source}/{d.name}', source, d.name, d, sessions, _description(d / 'README.md')))
    return runs


def poses_path(session, roots):
    s = SESSIONS[session]
    return Path(roots[s.dataset]) / f'poses_{s.seq}.csv'


def frame_times(session, roots, n):
    """Tiempo [s] de cada frame desde el inicio: la columna t de gt_<seq>.csv si existe (Rosario), si no
    id / frecuencia del dataset (FieldSAFE)."""
    s = SESSIONS[session]
    g = Path(roots[s.dataset]) / f'gt_{s.seq}.csv'
    if g.exists():
        t = np.loadtxt(g, delimiter=',', ndmin=2, usecols=1).ravel()
        if len(t) == n:
            return t - t[0]
    return np.arange(n) / s.frequency


def load_trajectory(session, roots):
    """(xy, camino acumulado, tiempos) de una sesion."""
    xy = load_ground_truth(poses_path(session, roots))
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    return xy, cum, frame_times(session, roots, len(xy))


def loops_table(results_yml, session, run_id, traj, far_m=FAR_M, stopped_m=STOPPED_PATH_M):
    """Una fila por loop detectado, con el error de traslacion de commons.translation_errors.

    Clase de cada loop: 'detenido' si el robot recorrio menos de stopped_m entre match y query, 'lejano' si
    query y match estan a mas de far_m segun el GT, 'en movimiento' en otro caso.
    """
    xy, cum, times = traj
    res, q, m, t = load_results(results_yml)
    if res['num_images'] != len(xy):
        raise ValueError(f"{res['num_images']} imágenes en el .yml vs {len(xy)} poses")
    err, stopped = translation_errors(xy, q, m, t, stopped_m)
    gd = np.linalg.norm(xy[q] - xy[m], axis=1)
    cls = np.where(stopped, 'detenido', np.where(gd > far_m, 'lejano', 'en movimiento'))
    df = pd.DataFrame(dict(
        run=run_id, session=session, query=q, match=m, t_query=times[q], t_match=times[m],
        x_query=xy[q, 0], y_query=xy[q, 1], x_match=xy[m, 0], y_match=xy[m, 1],
        gt_dist=gd, est_dist=np.linalg.norm(t, axis=1), path_sep=cum[q] - cum[m], clase=cls, error=err))
    qcsv = Path(str(results_yml).replace('_results.yml', '_queries.csv'))
    if qcsv.exists() and len(df):
        diag = pd.read_csv(qcsv)
        cols = [c for c in ('geom_inliers', 'best_score') if c in diag]
        if 'query' in diag and cols:
            df = df.merge(diag[['query'] + cols], on='query', how='left')
    return df, res


def summary(df, info):
    """Resumen por (corrida, sesion). `info` es {(corrida, sesion): dict del .yml}."""
    rows = []
    for (run, session), res in info.items():
        d = df[(df.run == run) & (df.session == session)]
        mov = d[d.clase == 'en movimiento'].error
        rows.append({
            'corrida': run, 'sesión': session, 'imágenes': res.get('num_images'), 'loops': len(d),
            'en movimiento': int((d.clase == 'en movimiento').sum()), 'detenido': int((d.clase == 'detenido').sum()),
            'lejano': int((d.clase == 'lejano').sum()),
            'error mediano [m]': mov.median() if len(mov) else np.nan,
            'error p90 [m]': mov.quantile(0.9) if len(mov) else np.nan,
            'error máx [m]': mov.max() if len(mov) else np.nan,
            'distancia mínima [m]': res.get('min_travel_distance', np.nan),
            'detección [ms]': res.get('loop_detection_time_ms', np.nan),
        })
    return pd.DataFrame(rows)
