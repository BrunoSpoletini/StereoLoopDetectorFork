"""Evaluacion de corridas de StereoLoopDetector contra el ground truth.

Definiciones (todas en funcion de la trayectoria GT planar xy):
  - sep(q, m): camino recorrido entre m y q (arc length), no distancia recta.
  - Revisita GT: el frame i tiene algun frame anterior j con ‖xy_i − xy_j‖ < R y
    sep(i, j) ≥ S. Es "misma direccion" si ademas |Δyaw| < MAX_DYAW (una camara
    frontal no puede cerrar un loop recorriendo el surco en sentido contrario).
  - Cobertura: el trayecto se divide en tramos de BIN_M metros de camino; un tramo
    con frames revisita esta cubierto si algun loop TP tiene su query en el tramo.
Cada loop detectado (q, m) se clasifica como:
  - TP:     ‖xy_q − xy_m‖ < R y sep ≥ S
  - trivial: sep < S  (p. ej. robot detenido: correcto pero inutil para SLAM)
  - FP:     ‖xy_q − xy_m‖ ≥ R y sep ≥ S
Error de pose de los TP:
  - e_mag = | ‖t_est‖ − ‖Δxy‖ |             (lo que se usaba hasta ahora)
  - e_vec = ‖Δ_est − Δ_gt‖ en el plano del frame del match (adelante/izquierda),
    con Δ_est = −Rᵀt (centro de la camara query en el frame de la camara match) y
    el heading GT derivado de la trayectoria.
  - e_yaw = |yaw_est − Δyaw_gt|
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from commons import load_opencv_yml

R_M = 3.0          # radio de revisita [m]
S_M = 20.0         # separacion minima de camino [m]
MAX_DYAW = np.radians(60)
POSE_OK_M = 1.0    # un loop tiene pose correcta si e_vec < 1 m ...
POSE_OK_DEG = 10.0  # ... y e_yaw < 10 grados
BIN_M = 10.0       # largo de los tramos para la cobertura [m]


@dataclass
class GroundTruth:
    xy: np.ndarray       # (n, 2)
    cum: np.ndarray      # (n,) arc length
    yaw: np.ndarray      # (n,) heading de la trayectoria [rad]
    revisit: np.ndarray  # (n,) bool, revisita en misma direccion
    revisit_any: np.ndarray  # (n,) bool, revisita en cualquier direccion
    bins: np.ndarray     # (n,) tramo de BIN_M metros al que pertenece cada frame


def trajectory_heading(xy, cum, half_window=2.0):
    """Heading de la trayectoria: direccion entre los puntos a ±half_window metros de camino."""
    hi = np.clip(np.searchsorted(cum, cum + half_window), 0, len(xy) - 1)
    lo = np.clip(np.searchsorted(cum, cum - half_window) - 1, 0, len(xy) - 1)
    d = xy[hi] - xy[lo]
    return np.arctan2(d[:, 1], d[:, 0])


def sensor_heading(poses_csv):
    """Yaw GT de un sensor de orientacion si existe junto al csv de poses: heading_<seq>.csv (FieldSAFE,
    GPS de doble antena) o gt_<seq>.csv (Rosario, cuaternion 6DoF). Solo se usan diferencias de yaw,
    El offset constante de montaje se estima en load_gt."""
    seq = poses_csv.stem.replace('poses_', '')
    h = poses_csv.with_name(f'heading_{seq}.csv')
    if h.exists():
        return np.loadtxt(h, delimiter=',', ndmin=2)[:, 1]
    g = poses_csv.with_name(f'gt_{seq}.csv')
    if g.exists():
        q = np.loadtxt(g, delimiter=',', ndmin=2)[:, 5:9]           # qx qy qz qw
        # el GT de Rosario es la orientacion de la camara (z = eje optico, ~19 grados hacia abajo):
        # el yaw es la direccion del eje optico proyectado en el plano
        R = Rotation.from_quat(q).as_matrix()
        return np.arctan2(R[:, 1, 2], R[:, 0, 2])
    return None


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def load_gt(poses_csv, r=R_M, s=S_M):
    poses = np.loadtxt(poses_csv, delimiter=',', ndmin=2)
    xy = poses[:, 1:3]
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    yaw = trajectory_heading(xy, cum)
    sensor = sensor_heading(Path(poses_csv))
    if sensor is not None and len(sensor) == len(xy):
        # offset de montaje del sensor: mediana de la diferencia con el heading de la trayectoria en movimiento
        moving = np.gradient(cum) > 0.02
        offset = np.median(wrap(sensor - yaw)[moving]) if moving.any() else 0.0
        yaw = wrap(sensor - offset)
    tree = cKDTree(xy)
    revisit = np.zeros(len(xy), bool)
    revisit_any = np.zeros(len(xy), bool)
    for i, nb in enumerate(tree.query_ball_point(xy, r)):
        nb = np.asarray(nb, dtype=int)
        nb = nb[(nb < i)]
        nb = nb[cum[i] - cum[nb] >= s]
        if len(nb):
            revisit_any[i] = True
            revisit[i] = (np.abs(wrap(yaw[i] - yaw[nb])) < MAX_DYAW).any()
    bins = (cum // BIN_M).astype(int)
    return GroundTruth(xy, cum, yaw, revisit, revisit_any, bins)


def load_run(results_yml):
    res = load_opencv_yml(results_yml)
    q = np.array(res.get('loop_query_ids') or [], dtype=int)
    m = np.array(res.get('loop_match_ids') or [], dtype=int)
    t = np.array([res.get(f'loop_translation_{a}') or [] for a in 'xyz'], float).T.reshape(-1, 3)
    if 'loop_rotation_x' in res:
        r = np.array([res.get(f'loop_rotation_{a}') or [] for a in 'xyz'], float).T.reshape(-1, 3)
    else:
        r = None
    return res, q, m, t, r


def pose_errors(gt, q, m, t, rv):
    """Error del vector de traslacion planar [m] y de yaw [grados] de cada loop."""
    R = Rotation.from_rotvec(rv).as_matrix()               # x_cur = R x_old + t
    c = -np.einsum('nji,nj->ni', R, t)                       # centro de la query en el frame del match
    est = np.stack([c[:, 2], -c[:, 0]], axis=1)              # (adelante, izquierda)
    psi = gt.yaw[m]
    d = gt.xy[q] - gt.xy[m]
    gtv = np.stack([np.cos(psi) * d[:, 0] + np.sin(psi) * d[:, 1],
                    -np.sin(psi) * d[:, 0] + np.cos(psi) * d[:, 1]], axis=1)
    e_vec = np.linalg.norm(est - gtv, axis=1)
    # yaw relativo de la query respecto del match: rotacion alrededor del eje y (abajo) de la camara
    Rqm = np.transpose(R, (0, 2, 1))                          # orientacion de la query en el frame del match
    yaw_est = np.arctan2(Rqm[:, 0, 2], Rqm[:, 2, 2])          # angulo del eje z de la query en el plano x-z
    yaw_gt = wrap(gt.yaw[q] - psi)
    e_yaw = np.degrees(np.abs(wrap(-yaw_est - yaw_gt)))
    return e_vec, e_yaw


def evaluate(results_yml, gt: GroundTruth, r=R_M, s=S_M):
    res, q, m, t, rv = load_run(results_yml)
    n = len(gt.xy)
    assert res['num_images'] == n, f"{results_yml}: {res['num_images']} imagenes vs {n} poses"
    gd = np.linalg.norm(gt.xy[q] - gt.xy[m], axis=1)
    sep = gt.cum[q] - gt.cum[m]
    tp = (gd < r) & (sep >= s)
    trivial = sep < s
    fp = (gd >= r) & (sep >= s)

    revisit_bins = np.unique(gt.bins[gt.revisit])
    out = dict(images=n, path_m=gt.cum[-1], gt_revisit_frames=int(gt.revisit.sum()),
               gt_bins=len(revisit_bins), loops=len(q), tp=int(tp.sum()), trivial=int(trivial.sum()),
               fp=int(fp.sum()))
    out['precision'] = out['tp'] / (out['tp'] + out['fp']) if out['tp'] + out['fp'] else np.nan
    out['recall'] = np.isin(np.where(gt.revisit)[0], q[tp]).sum() / max(gt.revisit.sum(), 1)
    out['bins_covered'] = int(np.isin(revisit_bins, gt.bins[q[tp]]).sum())
    out['coverage'] = out['bins_covered'] / max(len(revisit_bins), 1)

    # error de pose de los TP
    e_mag = np.abs(np.linalg.norm(t, axis=1) - gd)[tp]
    out['e_mag_med'] = float(np.median(e_mag)) if tp.any() else np.nan
    out['e_mag_p90'] = float(np.percentile(e_mag, 90)) if tp.any() else np.nan
    if rv is not None and tp.any():
        e_vec, e_yaw = pose_errors(gt, q, m, t, rv)
        out['e_vec_med'] = float(np.median(e_vec[tp]))
        out['e_vec_p90'] = float(np.percentile(e_vec[tp], 90))
        out['e_yaw_med'] = float(np.median(e_yaw[tp]))
        out['e_yaw_p90'] = float(np.percentile(e_yaw[tp], 90))
        # loops no triviales con pose relativa correcta, a cualquier distancia (lo que usa un SLAM)
        pose_ok = ~trivial & (e_vec < POSE_OK_M) & (e_yaw < POSE_OK_DEG)
        out['pose_ok'] = int(pose_ok.sum())
        out['pose_ok_frac'] = float(pose_ok.sum() / max((~trivial).sum(), 1))
    out['feature_ms'] = res.get('feature_time_ms')
    out['detection_ms'] = res.get('loop_detection_time_ms')
    return out, dict(q=q, m=m, t=t, gd=gd, sep=sep, tp=tp, trivial=trivial, fp=fp)


def summarize(rows):
    """Agrega metricas de varias sesiones (suma de conteos, micro-promedio de tasas)."""
    df = pd.DataFrame(rows)
    tot = dict(images=df.images.sum(), gt_revisit_frames=df.gt_revisit_frames.sum(),
               gt_bins=df.gt_bins.sum(), loops=df.loops.sum(), tp=df.tp.sum(),
               trivial=df.trivial.sum(), fp=df.fp.sum(), bins_covered=df.bins_covered.sum())
    if 'pose_ok' in df:
        tot['pose_ok'] = df.pose_ok.fillna(0).sum()
        nontriv = (df.loops - df.trivial).sum()
        tot['pose_ok_frac'] = tot['pose_ok'] / max(nontriv, 1)
    tot['precision'] = tot['tp'] / (tot['tp'] + tot['fp']) if tot['tp'] + tot['fp'] else np.nan
    tot['recall'] = (df.recall * df.gt_revisit_frames).sum() / max(tot['gt_revisit_frames'], 1)
    tot['coverage'] = tot['bins_covered'] / max(tot['gt_bins'], 1)
    for k in ('e_mag_med', 'e_vec_med', 'e_yaw_med'):
        if k in df:
            w = df.tp.where(df[k].notna(), 0)
            tot[k] = float((df[k].fillna(0) * w).sum() / w.sum()) if w.sum() else np.nan
    return tot
