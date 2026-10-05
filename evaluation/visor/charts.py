"""Graficos interactivos (Plotly) del visor. Cada funcion recibe tablas de data.py y devuelve una figura.

No se fija template ni colores de fondo: st.plotly_chart aplica el tema de Streamlit (claro u oscuro) y
respeta los colores de las trazas.
"""
import numpy as np
import plotly.graph_objects as go

CLASS_COLOR = {'en movimiento': '#2a78d6', 'detenido': '#eb6834', 'falso positivo': '#e34948'}
CLASS_DASH = {'en movimiento': 'solid', 'detenido': 'dot', 'falso positivo': 'dash'}
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
OTHER = '#8a8984'
TRACK = '#a8a7a1'

LABEL = {
    'error': 'Error de traslación |‖t_est‖ − ‖Δp_gt‖| [m]',
    'gt_dist': 'Distancia GT query–match [m]',
    'est_dist': 'Traslación estimada ‖t_est‖ [m]',
    'path_sep': 'Camino recorrido entre match y query [m]',
    'geom_inliers': 'Inliers del PnP',
    'best_score': 'Score BoW',
}

HOVER = ('<b>%{customdata[0]}</b><br>%{customdata[1]}<br>query %{customdata[2]} ↔ match %{customdata[3]}'
         ' · %{customdata[4]}<br>dist GT %{customdata[5]:.2f} m · ‖t‖ %{customdata[6]:.2f} m'
         '<br>error %{customdata[7]:.3f} m · camino %{customdata[8]:.0f} m<extra></extra>')
HOVER_COLS = ['run', 'session', 'query', 'match', 'clase', 'gt_dist', 'est_dist', 'error', 'path_sep']


def run_colors(runs):
    """Color fijo por corrida segun su orden de seleccion; de la novena en adelante, gris."""
    return {r: SERIES[i] if i < len(SERIES) else OTHER for i, r in enumerate(runs)}


def short(run):
    """'finalRuns/03_distancia_minima_20m' -> '03_distancia_minima_20m' (test: con prefijo)."""
    src, name = run.split('/', 1)
    return name if src == 'finalRuns' else f'test · {name}'


def _layout(fig, height=520, **kw):
    fig.update_layout(height=height, legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0, title=None),
                      margin=dict(t=40, l=10, r=10, b=10), hoverlabel=dict(font_size=12), **kw)
    return fig


def summary_bars(sdf, metric, runs, labels):
    """Una columna del resumen por sesion, una barra por corrida."""
    colors = run_colors(runs)
    fig = go.Figure()
    for r in runs:
        d = sdf[sdf['corrida'] == r]
        fig.add_trace(go.Bar(x=[labels(s) for s in d['sesión']], y=d[metric], name=short(r),
                             marker=dict(color=colors[r], cornerradius=4),
                             hovertemplate=f'<b>{short(r)}</b><br>%{{x}}<br>{metric}: %{{y:.3~f}}<extra></extra>'))
    fig.update_layout(barmode='group', bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(title=metric)
    return _layout(fig, height=420)


def boxplot(df, runs, labels):
    """Error de traslacion por sesion; una caja por corrida, con todos los loops como puntos."""
    colors = run_colors(runs)
    fig = go.Figure()
    for r in runs:
        d = df[df.run == r]
        fig.add_trace(go.Box(x=[labels(s) for s in d.session], y=d.error, name=f'{short(r)} ({len(d)})',
                             marker=dict(color=colors[r], size=5, opacity=0.5), line=dict(width=1.5),
                             boxpoints='all', jitter=0.35, pointpos=0, customdata=d[HOVER_COLS],
                             hovertemplate=HOVER))
    fig.update_layout(boxmode='group')
    fig.update_yaxes(title=LABEL['error'], rangemode='tozero')
    return _layout(fig, height=560)


def scatter(df, x, y, runs, by='run'):
    """Dispersion de dos columnas por loop. Con x=gt_dist e y=est_dist se dibuja la recta y = x."""
    fig = go.Figure()
    if by == 'run':
        colors = run_colors(runs)
        groups = [(short(r), df[df.run == r], colors[r]) for r in runs]
    else:
        groups = [(c, df[df.clase == c], CLASS_COLOR[c]) for c in CLASS_COLOR]
    for name, d, c in groups:
        if len(d):
            fig.add_trace(go.Scattergl(x=d[x], y=d[y], mode='markers', name=f'{name} ({len(d)})',
                                       marker=dict(color=c, size=8, opacity=0.75, line=dict(color='white', width=1)),
                                       customdata=d[HOVER_COLS], hovertemplate=HOVER))
    if {x, y} == {'gt_dist', 'est_dist'} and len(df):
        hi = float(np.nanmax(df[[x, y]].to_numpy()))
        fig.add_trace(go.Scatter(x=[0, hi], y=[0, hi], mode='lines', name='y = x',
                                 line=dict(color=OTHER, width=1, dash='dash'), hoverinfo='skip'))
    fig.update_xaxes(title=LABEL.get(x, x), rangemode='tozero')
    fig.update_yaxes(title=LABEL.get(y, y), rangemode='tozero')
    return _layout(fig, height=600)


def _segments(d, cols_a, cols_b):
    """Coordenadas de segmentos query–match separados por NaN (una sola traza para miles de loops)."""
    n = len(d)
    out = []
    for a, b in zip(cols_a, cols_b):
        v = np.full(3 * n, np.nan)
        v[0::3], v[1::3] = d[a].to_numpy(), d[b].to_numpy()
        out.append(v)
    return out, np.repeat(d[HOVER_COLS].to_numpy(), 3, axis=0)


def _loop_traces(df, runs, three_d):
    """Una traza por (corrida, clase). Una corrida: color por clase; varias: color por corrida, trazo por clase."""
    colors = run_colors(runs)
    single = len(runs) == 1
    a = ['x_query', 'y_query', 't_query'] if three_d else ['x_query', 'y_query']
    b = ['x_match', 'y_match', 't_match'] if three_d else ['x_match', 'y_match']
    traces = []
    for r in runs:
        for c in CLASS_COLOR:
            d = df[(df.run == r) & (df.clase == c)]
            if not len(d):
                continue
            xyz, custom = _segments(d, a, b)
            color = CLASS_COLOR[c] if single else colors[r]
            kw = dict(mode='lines+markers', name=f'{c} ({len(d)})' if single else f'{short(r)} · {c} ({len(d)})',
                      line=dict(color=color, width=3 if three_d else 1.5, dash='solid' if single else CLASS_DASH[c]),
                      marker=dict(size=2.5 if three_d else 5, color=color), customdata=custom, hovertemplate=HOVER)
            traces.append(go.Scatter3d(x=xyz[0], y=xyz[1], z=xyz[2], **kw) if three_d
                          else go.Scattergl(x=xyz[0], y=xyz[1], **kw))
    return traces


def trajectory_3d(traj, df, runs, color_time=True):
    """Pose del robot (x, y) en funcion del tiempo (eje z) con los loops como segmentos query–match."""
    xy, _, t = traj
    step = max(1, len(xy) // 20000)  # la trayectoria completa puede tener >30k frames
    xy, t = xy[::step], t[::step]
    line = dict(color=t, colorscale=[[0, '#d6d5d0'], [1, '#3d3c38']], width=4, colorbar=dict(title='t [s]', thickness=10, len=0.5)) \
        if color_time else dict(color=TRACK, width=4)
    fig = go.Figure([go.Scatter3d(x=xy[:, 0], y=xy[:, 1], z=t, mode='lines', name='trayectoria GT', line=line,
                                  hovertemplate='x %{x:.1f} m · y %{y:.1f} m<br>t %{z:.1f} s<extra></extra>')])
    fig.add_traces(_loop_traces(df, runs, three_d=True))
    fig.update_layout(scene=dict(xaxis_title='x [m]', yaxis_title='y [m]', zaxis_title='tiempo [s]',
                                 aspectmode='manual', aspectratio=dict(x=1, y=1, z=1.1)))
    return _layout(fig, height=720)


def map_2d(traj, df, runs):
    """Vista en planta: trayectoria GT y loops como segmentos query–match."""
    xy = traj[0]
    fig = go.Figure([go.Scattergl(x=xy[:, 0], y=xy[:, 1], mode='lines', name='trayectoria GT',
                                  line=dict(color=TRACK, width=1), hoverinfo='skip')])
    fig.add_traces(_loop_traces(df, runs, three_d=False))
    fig.update_xaxes(title='x [m]')
    fig.update_yaxes(title='y [m]', scaleanchor='x', scaleratio=1)
    return _layout(fig, height=640)
