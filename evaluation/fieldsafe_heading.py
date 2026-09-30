"""Heading GT de FieldSAFE desde el GPS de doble antena ($GPHDT), interpolado a cada par estereo.

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
    a = ap.parse_args()
    ext = os.path.join(FS, 'extracted', a.session)
    ts_l = np.loadtxt(os.path.join(ext, 'timestamps_left.txt'))
    ts_r = np.loadtxt(os.path.join(ext, 'timestamps_right.txt'))
    date = datetime.datetime.strptime(a.session[:10], '%Y-%m-%d').date()
    hdt = parse_hdt(os.path.join(ext, 'gps_raw.txt'), date)
    pairs = pair_stereo(ts_l, ts_r, a.max_dt)
    t_pair = np.array([ts_l[i] for i, _, _ in pairs]) + a.time_offset
    # mismo recorte que prepare_session (rango del GPS GGA ~ rango del HDT)
    t_pair = t_pair[(t_pair >= hdt[0, 0]) & (t_pair <= hdt[-1, 0])]
    yaw_enu = np.unwrap(np.radians(90.0 - hdt[:, 1]))
    yaw = np.interp(t_pair, hdt[:, 0], yaw_enu)
    n_poses = sum(1 for _ in open(os.path.join(FS, 'prepared', f'poses_{a.session}.csv')))
    assert len(yaw) == n_poses, f'{len(yaw)} headings vs {n_poses} poses'
    out = os.path.join(FS, 'prepared', f'heading_{a.session}.csv')
    np.savetxt(out, np.column_stack([np.arange(len(yaw)), yaw]), delimiter=',', fmt=['%d', '%.6f'])
    print(f'{a.session}: {len(yaw)} headings -> {out}')


if __name__ == '__main__':
    main()
