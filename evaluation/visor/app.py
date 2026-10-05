"""Visor interactivo de corridas de StereoLoopDetector (FieldSAFE / RosarioV2).

  pip install streamlit plotly pandas pyyaml
  streamlit run evaluation/visor/app.py

Lee las corridas de evaluation/finalRuns y evaluation/testRuns (ver data.py). Se eligen una o varias corridas,
las sesiones y la vista; varias corridas se superponen con un color por corrida para compararlas.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import charts  # noqa: E402
import data  # noqa: E402

VIEWS = ['Resumen', 'Error de traslación', 'Dispersión', 'Trayectoria 3D', 'Mapa 2D']
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { min-width: 360px; }
.block-container { padding-top: 2.2rem; max-width: 1400px; }
[data-testid="stAppDeployButton"] { display: none; }
h1 { font-weight: 700; letter-spacing: -0.02em; }
.run-head { display: flex; align-items: center; gap: .5rem; font-weight: 600; font-size: .98rem; line-height: 1.3; }
.run-dot { width: .75rem; height: .75rem; border-radius: 50%; flex: none; }
.badge { font-size: .7rem; font-weight: 600; padding: .1rem .45rem; border-radius: .6rem;
         background: rgba(128,128,128,.16); text-transform: uppercase; letter-spacing: .04em; }
.run-desc { opacity: .72; font-size: .85rem; margin: .35rem 0 .6rem; min-height: 1.2rem; }
.run-stats { display: grid; grid-template-columns: repeat(4, auto); gap: 1rem; justify-content: start; }
.run-stats div { display: flex; flex-direction: column; }
.run-stats b { font-size: 1.25rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.run-stats span { font-size: .66rem; white-space: nowrap; opacity: .65; text-transform: uppercase; letter-spacing: .04em; }
</style>
"""


@st.cache_data(show_spinner='Cargando ground truth…')
def trajectory(session, roots_items, mtime):
    return data.load_trajectory(session, dict(roots_items))


@st.cache_data(show_spinner='Cargando loops…')
def loops(yml, mtime, session, run_id, roots_items, poses_mtime, far_m, stopped_m):
    traj = trajectory(session, roots_items, poses_mtime)
    return data.loops_table(yml, session, run_id, traj, far_m, stopped_m)


def load_selection(runs, sessions, roots, far_m, stopped_m):
    dfs, info, problems = [], {}, []
    roots_items = tuple(sorted(roots.items()))
    for run in runs:
        for s in sessions:
            yml = run.path / f'{s}_results.yml'
            if not yml.exists():
                continue
            if s not in data.SESSIONS:
                problems.append(f'**{run.id}/{s}**: sesión desconocida (agregala a `SESSIONS` en `visor/data.py`).')
                continue
            poses = data.poses_path(s, roots)
            if not poses.exists():
                problems.append(f'**{run.id}/{s}**: no se encuentra `{poses}`.')
                continue
            try:
                df, res = loops(str(yml), yml.stat().st_mtime, s, run.id, roots_items, poses.stat().st_mtime,
                                far_m, stopped_m)
            except Exception as e:  # un .yml roto no debe tirar abajo el resto
                problems.append(f'**{run.id}/{s}**: {e}')
                continue
            dfs.append(df)
            info[(run.id, s)] = res
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=['run', 'session', 'clase', 'error'])
    return df, info, problems


def segmented(label, options, default, key):
    """Control segmentado si la version de Streamlit lo tiene; si no, radio horizontal."""
    if hasattr(st, 'segmented_control'):
        v = st.segmented_control(label, options, default=default, key=key, label_visibility='collapsed')
        return v or default
    return st.radio(label, options, index=options.index(default), horizontal=True, key=key,
                    label_visibility='collapsed')


def class_filter(key, default):
    return st.multiselect('Clases de loop', data.CLASSES, default=default, key=key,
                          help='detenido: el robot recorrió menos de la distancia de "detenido" entre match y query; '
                               'falso positivo: query y match a más de la distancia de "falso positivo" según el GT')


def run_cards(runs, df, colors):
    cols = st.columns(min(len(runs), 3))
    for i, run in enumerate(runs):
        d = df[df.run == run.id]
        mov = d[d.clase == 'en movimiento'].error
        med = f'{mov.median():.3f}' if len(mov) else '—'
        with cols[i % len(cols)].container(border=True):
            st.markdown(
                f'<div class="run-head"><span class="run-dot" style="background:{colors[run.id]}"></span>'
                f'{run.name}<span class="badge">{"final" if run.source == "finalRuns" else "test"}</span></div>'
                f'<div class="run-desc">{run.description}</div>'
                f'<div class="run-stats"><div><b>{len(d)}</b><span>loops</span></div>'
                f'<div><b>{(d.clase == "en movimiento").sum()}</b><span>en movimiento</span></div>'
                f'<div><b>{(d.clase == "detenido").sum()}</b><span>detenido</span></div>'
                f'<div><b>{med}</b><span>error mediano [m]</span></div></div>',
                unsafe_allow_html=True)


def subsample(d, max_loops):
    """Como mucho max_loops por (corrida, clase), uniformemente, para que los graficos sigan fluidos."""
    if d.empty:
        return d
    g = d.groupby(['run', 'clase'])
    step = (-(-g['query'].transform('size') // max_loops)).clip(lower=1)
    return d[g.cumcount() % step == 0]


def main():
    st.set_page_config(page_title='Visor de loops', page_icon='🔁', layout='wide')
    st.markdown(CSS, unsafe_allow_html=True)

    all_runs = data.list_runs()
    with st.sidebar:
        st.header('Corridas')
        sources = st.multiselect('Fuentes', list(data.SOURCES), default=list(data.SOURCES),
                                 help='finalRuns: corridas del paper (versionadas). testRuns: pruebas (no versionadas).')
        available = [r for r in all_runs if r.source in sources]
        if not available:
            st.warning('No hay corridas con `*_results.yml` en ' + ', '.join(f'`{data.SOURCES[s]}`' for s in sources))
            st.stop()
        by_id = {r.id: r for r in available}
        finals = [r.id for r in available if r.source == 'finalRuns']
        run_ids = st.multiselect('Corridas', list(by_id), default=(finals or list(by_id))[-1:],
                                 format_func=charts.short,
                                 help='Elegí varias para compararlas: cada corrida tiene un color fijo.')
        runs = [by_id[i] for i in run_ids]
        all_sessions = sorted({s for r in runs for s in r.sessions})
        sessions = st.multiselect('Sesiones', all_sessions, default=all_sessions, format_func=data.session_label)

        with st.expander('Clasificación y datos'):
            far_m = st.number_input('Falso positivo: distancia GT query–match mayor a [m]', 0.5, 100.0, data.FAR_M, 0.5)
            stopped_m = st.number_input('Detenido: camino entre match y query menor a [m]', 0.0, 100.0,
                                        float(data.STOPPED_PATH_M), 0.5)
            roots = {}
            for ds, (name, env, _) in data.DATASETS.items():
                roots[ds] = st.text_input(f'Carpeta prepared de {name}', data.default_roots()[ds],
                                          help=f'También se puede fijar con la variable de entorno {env}')

    st.title('Visor de cierres de ciclo')
    st.caption('StereoLoopDetector en FieldSAFE y RosarioV2 · loops detectados contra el ground truth de cada sesión')

    if not runs or not sessions:
        st.info('Elegí al menos una corrida y una sesión en la barra lateral.')
        return

    df, info, problems = load_selection(runs, sessions, roots, far_m, stopped_m)
    colors = charts.run_colors(run_ids)
    run_cards(runs, df, colors)
    for p in problems:
        st.warning(p, icon='⚠️')
    if not info:
        return

    view = segmented('Vista', VIEWS, VIEWS[0], 'view')
    labels = data.session_label

    if view == 'Resumen':
        sdf = data.summary(df, info)
        numeric = [c for c in sdf.columns if c not in ('corrida', 'sesión')]
        metric = st.selectbox('Métrica', numeric, index=numeric.index('en movimiento'))
        with st.container(border=True):
            st.plotly_chart(charts.summary_bars(sdf, metric, run_ids, labels))
        st.dataframe(sdf.assign(corrida=sdf['corrida'].map(charts.short), **{'sesión': sdf['sesión'].map(labels)}),
                     hide_index=True, column_config={c: st.column_config.NumberColumn(format='%.3f')
                                                     for c in sdf.columns if '[m]' in c})

    elif view == 'Error de traslación':
        cls = class_filter('box_cls', ['en movimiento'])
        d = df[df.clase.isin(cls)]
        with st.container(border=True):
            st.plotly_chart(charts.boxplot(d, run_ids, labels))
        stats = d.groupby(['run', 'session']).error.describe(percentiles=[.5, .9])[['count', '50%', '90%', 'max']]
        st.dataframe(stats.rename(columns={'count': 'n', '50%': 'mediana [m]', '90%': 'p90 [m]', 'max': 'máx [m]'})
                     .reset_index().assign(run=lambda x: x.run.map(charts.short),
                                           session=lambda x: x.session.map(labels)),
                     hide_index=True)

    elif view == 'Dispersión':
        cols = [c for c in ['gt_dist', 'est_dist', 'error', 'path_sep', 'geom_inliers', 'best_score'] if c in df]
        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
        x = c1.selectbox('Eje x', cols, index=0, format_func=lambda c: charts.LABEL.get(c, c))
        y = c2.selectbox('Eje y', cols, index=1, format_func=lambda c: charts.LABEL.get(c, c))
        by = c3.radio('Color por', ['corrida', 'clase'], horizontal=True)
        with c4:
            cls = class_filter('sc_cls', data.CLASSES)
        with st.container(border=True):
            st.plotly_chart(charts.scatter(df[df.clase.isin(cls)], x, y, run_ids,
                                           by='run' if by == 'corrida' else 'clase'))

    else:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            cls = class_filter('traj_cls', ['en movimiento', 'falso positivo'])
        max_loops = c2.number_input('Máx. loops por corrida y clase', 50, 50000, 3000, step=500,
                                    help='Se submuestrea uniformemente para que el gráfico siga fluido')
        color_time = view == 'Trayectoria 3D' and c3.toggle('Color por tiempo', True)
        roots_items = tuple(sorted(roots.items()))
        for s in sessions:
            if not any(k[1] == s for k in info):
                continue
            traj = trajectory(s, roots_items, data.poses_path(s, roots).stat().st_mtime)
            d = subsample(df[(df.session == s) & df.clase.isin(cls)], max_loops)
            with st.container(border=True):
                st.subheader(labels(s))
                fig = charts.trajectory_3d(traj, d, run_ids, color_time) if view == 'Trayectoria 3D' \
                    else charts.map_2d(traj, d, run_ids)
                st.plotly_chart(fig)

    with st.expander(f'Tabla de loops ({len(df)})'):
        st.dataframe(df, hide_index=True)
        st.download_button('Descargar CSV', df.to_csv(index=False), 'loops.csv', 'text/csv')


main()
