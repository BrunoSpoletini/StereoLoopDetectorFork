"""Sesiones usadas en los experimentos: FieldSAFE = train, RosarioV2 = validacion."""
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FS = Path('/mnt/datalake/datasets/fieldsafe/prepared')
RO = Path('/mnt/datalake/datasets/rosariov2/prepared')
CACHE = Path('/home/bruno/Desktop/tesina/sld_cache')


@dataclass
class Session:
    name: str
    dataset: str
    seq: str
    prepared: Path
    calibration: Path
    frequency: float

    @property
    def left(self):
        return self.prepared / f'left_{self.seq}.txt'

    @property
    def right(self):
        return self.prepared / f'right_{self.seq}.txt'

    @property
    def poses(self):
        return self.prepared / f'poses_{self.seq}.csv'

    def available(self):
        return self.left.exists() and self.right.exists() and self.poses.exists()


def _fs(name, seq):
    return Session(name, 'fieldsafe', seq, FS, REPO / 'resources/fs_stereo_parameters.yaml', 10.0)


def _ro(name, seq):
    return Session(name, 'rosario', seq, RO, REPO / 'resources/rosario_stereo_parameters.yaml', 15.0)


SESSIONS = [
    _fs('fs_static1', '2016-10-25-11-09-42'),
    _fs('fs_11-34', '2016-10-25-11-34-25'),
    _fs('fs_dynamic1', '2016-10-25-11-41-21'),
    _fs('fs_dynamic2', '2016-10-25-12-07-22'),
    _fs('fs_12-37', '2016-10-25-12-37-57'),
    _ro('ro_1222_1314', '2023-12-22-13-14-16'),
    _ro('ro_1222_1429', '2023-12-22-14-29-43'),
    _ro('ro_1222_1631', '2023-12-22-16-31-08'),
    _ro('ro_1226_1339', '2023-12-26-13-39-43'),
    _ro('ro_1226_1510', '2023-12-26-15-10-15'),
    _ro('ro_1226_1548', '2023-12-26-15-48-38'),
]


def select(which):
    """which: 'fieldsafe', 'rosario', 'all' o lista de nombres separados por coma."""
    if which in ('fieldsafe', 'rosario'):
        out = [s for s in SESSIONS if s.dataset == which]
    elif which == 'all':
        out = list(SESSIONS)
    else:
        names = which.split(',')
        out = [s for s in SESSIONS if s.name in names]
    return [s for s in out if s.available()]
