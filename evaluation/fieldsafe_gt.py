"""GT de FieldSAFE (posicion y heading) re-sincronizado, interpolado a cada par estereo.

Posicion: $GPGGA; heading: GPS de doble antena ($GPHDT).
Los pares (y por lo tanto los ids de imagen y las caches) se eligen exactamente como en prepare_session.py
con --time-offset 19.25; la pose de cada par se evalua en t + time_offset + extra_offset. El extra de
+1.5 s sale de dos estimaciones independientes: el retardo de la velocidad GPS respecto de la odometria
visual (1.4-1.7 s en 3 de 4 sesiones) y el maximo de loops con pose correcta (lag de 15 frames en las 4).
Las poses viejas se guardan como poses_<sesion>_offset19.25.csv.

Recalcula los mismos pares y tiempos que prepare_session.py y escribe
<prepared>/heading_<sesion>.csv con 'id,yaw' (yaw ENU en radianes, antihorario desde el este).
El evaluador lo usa en lugar del heading derivado de la trayectoria cuando existe.

  python fieldsafe_heading.py 2016-10-25-11-41-21
"""
import argparse
import datetime
import os
import re
import sys

import numpy as np

FS = '/mnt/datalake/datasets/fieldsafe'
sys.path.insert(0, FS)
from prepare_session import pair_stereo  # noqa: E402


def parse_hdt(path, date):
    """(epoch, heading_deg) de cada $GPHDT, con la hora del $GPGGA que lo precede."""
    out, t = [], None
    pat = re.compile(rb'\$GP(GGA|HDT),([\d.]+)')
    with open(path, 'rb') as f:
        for m in pat.finditer(f.read()):
            kind, val = m.groups()
            if kind == b'GGA':
                s = val.decode()
                h, mi, sec = int(s[0:2]), int(s[2:4]), float(s[4:])
                t = datetime.datetime(date.year, date.month, date.day, h, mi, int(sec),
                                      int((sec % 1) * 1e6), tzinfo=datetime.timezone.utc).timestamp()
            elif t is not None:
                out.append((t, float(val)))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--max-dt', type=float, default=0.02)
    ap.add_argument('--time-offset', type=float, default=19.25)
    ap.add_argument('--extra-offset', type=float, default=1.5)
    a = ap.parse_args()
    ext = os.path.join(FS, 'extracted', a.session)
    ts_l = np.loadtxt(os.path.join(ext, 'timestamps_left.txt'))
    ts_r = np.loadtxt(os.path.join(ext, 'timestamps_right.txt'))
    date = datetime.datetime.strptime(a.session[:10], '%Y-%m-%d').date()
    hdt = parse_hdt(os.path.join(ext, 'gps_raw.txt'), date)
    pairs = pair_stereo(ts_l, ts_r, a.max_dt)
    t_pair = np.array([ts_l[i] for i, _, _ in pairs]) + a.time_offset
    # mismo recorte que prepare_session (rango del GPS GGA ~ rango del HDT)
    sys.path.insert(0, FS)
    from prepare_session import latlon_to_local, parse_gpgga
    gps = parse_gpgga(os.path.join(ext, 'gps_raw.txt'), date)
    t_pair = t_pair[(t_pair >= gps[0, 0]) & (t_pair <= gps[-1, 0])]   # mismos pares que prepare_session
    t_q = t_pair + a.extra_offset
    gx, gy = latlon_to_local(gps[:, 1], gps[:, 2], gps[0][1], gps[0][2])
    px, py = np.interp(t_q, gps[:, 0], gx), np.interp(t_q, gps[:, 0], gy)
    yaw_enu = np.unwrap(np.radians(90.0 - hdt[:, 1]))
    yaw = np.interp(t_q, hdt[:, 0], yaw_enu)
    poses = os.path.join(FS, 'prepared', f'poses_{a.session}.csv')
    backup = os.path.join(FS, 'prepared', f'poses_{a.session}_offset19.25.csv')
    if not os.path.exists(backup):
        os.rename(poses, backup)
    old = np.loadtxt(backup, delimiter=',', ndmin=2)
    assert len(old) == len(yaw), f'{len(yaw)} pares vs {len(old)} poses'
    ids = np.arange(len(yaw))
    np.savetxt(poses, np.column_stack([ids, px, py, np.zeros((len(ids), 4))]), delimiter=',',
               fmt=['%d', '%.6f', '%.6f', '%d', '%d', '%d', '%d'])
    out = os.path.join(FS, 'prepared', f'heading_{a.session}.csv')
    np.savetxt(out, np.column_stack([ids, yaw]), delimiter=',', fmt=['%d', '%.6f'])
    print(f'{a.session}: {len(yaw)} pares, extra offset {a.extra_offset:+.2f} s, '
          f'desplazamiento medio vs GT viejo {np.linalg.norm(np.column_stack([px, py]) - old[:, 1:3], axis=1).mean():.2f} m')


if __name__ == '__main__':
    main()
