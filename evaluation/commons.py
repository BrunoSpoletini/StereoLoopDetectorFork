"""Funciones comunes para evaluar las corridas de StereoLoopDetector.

- Carga de resultados: el .yml que escribe el detector (formato OpenCV FileStorage).
- Carga de ground truth: csv de poses `id,x,y,ow,ox,oy,oz` sin header, una fila por imagen.
- Cálculo de error: error de traslación de cada loop contra el ground truth.
"""
from pathlib import Path

import numpy as np
import yaml

STOPPED_PATH_M = 1.0  # si el robot recorrió menos que esto entre match y query, el loop es "con robot detenido"


def load_opencv_yml(path):
    return yaml.safe_load(Path(path).read_text().split('---', 1)[1])


def load_results(path):
    """Carga los resultados del detector.

    Devuelve (res, q, m, t_est): el dict crudo del .yml, los ids de query y match de cada loop,
    y la traslación estimada por PnP de cada loop como array (n_loops, 3).
    """
    res = load_opencv_yml(path)
    q = np.array(res.get('loop_query_ids') or [], dtype=int)
    m = np.array(res.get('loop_match_ids') or [], dtype=int)
    keys = [f'loop_translation_{axis}' for axis in 'xyz']
    missing = [k for k in keys if k not in res]
    assert not missing, f'{path}: faltan {missing}'
    t_est = np.array([res[k] or [] for k in keys], dtype=float).T.reshape(-1, 3)
    return res, q, m, t_est


def load_ground_truth(path, num_images=None):
    """Carga el csv de poses y devuelve las posiciones planares xy (n_imagenes, 2), indexadas por id de imagen.

    Si se pasa `num_images`, verifica que haya una pose por imagen.
    """
    poses = np.loadtxt(path, delimiter=',', ndmin=2)
    assert (poses[:, 0] == np.arange(len(poses))).all(), 'se esperaba que el id de la pose coincida con la fila'
    if num_images is not None:
        assert len(poses) == num_images, f'{len(poses)} poses para {num_images} imágenes'
    return poses[:, 1:3]


def translation_errors(xy, q, m, t_est, stopped_path_m=STOPPED_PATH_M):
    """Error de traslación de cada loop: e = | ‖t_est‖ − ‖xy[q] − xy[m]‖ |.

    Se compara solo la magnitud porque el ground truth no tiene orientación y el detector descarta la
    rotación de PnP. Devuelve (err, stopped), donde `stopped` marca los loops en los que el robot recorrió
    menos de `stopped_path_m` metros entre match y query.
    """
    gt_dist = np.linalg.norm(xy[q] - xy[m], axis=1)
    err = np.abs(np.linalg.norm(t_est, axis=1) - gt_dist)
    cum_path = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    stopped = (cum_path[q] - cum_path[m]) < stopped_path_m
    return err, stopped
