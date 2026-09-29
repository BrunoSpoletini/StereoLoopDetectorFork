"""Graficos de las corridas (PNG estaticos para la bitacora/tesis).

  python plots.py map <run> [--sessions fieldsafe]     mapa de loops por sesion
  python plots.py compare <run1> <run2> ...            comparacion de configuraciones
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from datasets import REPO, select
from sld_eval import evaluate, load_gt

RUNS = REPO / 'evaluation/runs'
FIGS = REPO / 'evaluation/figs'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e5e4df'
TRACK, REVISIT = '#c9c8c2', '#d9e6f7'
C_TP, C_TRIV, C_FP = '#2a78d6', '#eb6834', '#e34948'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_map(run, sessions):
    ss = select(sessions)
    ncol = min(len(ss), 4)
    nrow = int(np.ceil(len(ss) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.4 * nrow), dpi=130, squeeze=False)
    for ax in axs.flat[len(ss):]:
        ax.axis('off')
    for ax, s in zip(axs.flat, ss):
        gt = load_gt(s.poses)
        m, d = evaluate(RUNS / run / f'{s.name}_results.yml', gt)
        ax.plot(*gt.xy.T, color=TRACK, lw=0.7, zorder=1)
        ax.scatter(*gt.xy[gt.revisit].T, s=3, color=REVISIT, lw=0, zorder=1, label='revisita GT')
        for mask, c, lab in ((d['trivial'], C_TRIV, 'trivial'), (d['fp'], C_FP, 'FP'), (d['tp'], C_TP, 'TP')):
            ax.scatter(*gt.xy[d['q'][mask]].T, s=10, color=c, lw=0.4, edgecolor='white', zorder=3,
                       label=f'{lab} ({mask.sum()})')
        ax.set_title(f"{s.name}   cobertura {m['coverage']:.0%}   e_vec {m.get('e_vec_med', np.nan):.2f} m",
                     fontsize=9, color=INK)
        ax.set_aspect('equal', 'datalim')
        style(ax)
        ax.legend(fontsize=7, frameon=False, loc='best', markerscale=1.5)
    fig.suptitle(f'Loops detectados — {run}', color=INK, fontsize=11)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    out = FIGS / f'map_{run}_{sessions}.png'
    fig.savefig(out)
    print(out)


def plot_compare(runs, dataset='fieldsafe', out_name=None):
    """Small multiples: una metrica por panel, una barra por configuracion."""
    rows = []
    for r in runs:
        df = pd.read_csv(RUNS / r / 'metrics.csv')
        tot = df[df.session == f'TOTAL_{dataset}']
        if len(tot):
            rows.append(tot.iloc[0])
    df = pd.DataFrame(rows)
    metrics = [('tp', 'Loops TP'), ('coverage', 'Cobertura (tramos de 10 m)'),
               ('precision', 'Precisión'), ('e_vec_med', 'Error traslación e_vec (m, mediana)')]
    metrics = [(k, t) for k, t in metrics if k in df]
    fig, axs = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 0.45 * len(df) + 1.4), dpi=130,
                            sharey=True)
    y = np.arange(len(df))
    for ax, (k, title) in zip(axs, metrics):
        v = df[k].astype(float).values
        ax.barh(y, np.nan_to_num(v), color=C_TP, height=0.62)
        for yi, vi in zip(y, v):
            txt = '—' if np.isnan(vi) else (f'{vi:.0f}' if k == 'tp' else f'{vi:.2f}')
            ax.text(np.nan_to_num(vi), yi, ' ' + txt, va='center', fontsize=7.5, color=INK)
        ax.set_title(title, fontsize=9, color=INK)
        style(ax)
        ax.grid(axis='y', visible=False)
    axs[0].set_yticks(y, df.config.values)
    axs[0].invert_yaxis()
    fig.suptitle(f'{dataset}: comparación de configuraciones', fontsize=10.5, color=INK)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    out = FIGS / (out_name or f'compare_{dataset}.png')
    fig.savefig(out)
    print(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['map', 'compare'])
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--sessions', default='fieldsafe')
    ap.add_argument('--dataset', default='fieldsafe')
    ap.add_argument('--out')
    a = ap.parse_args()
    if a.cmd == 'map':
        plot_map(a.runs[0], a.sessions)
    else:
        plot_compare(a.runs, a.dataset, a.out)
