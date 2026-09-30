"""Corre demo_stereo con una configuracion sobre un conjunto de sesiones y evalua.

Uso:
  python run_sld.py --config configs/baseline.yaml --sessions fieldsafe [--jobs 3]

Salida en evaluation/runs/<nombre_config>/:
  <sesion>_results.yml, <sesion>_queries.csv   (salida cruda del detector, no versionada)
  metrics.csv                                   (una fila por sesion + TOTAL)
"""
import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from datasets import CACHE, REPO, select
from sld_eval import evaluate, load_gt, summarize

BIN = REPO / 'build/demo_stereo'
VOC = REPO / 'resources/ORBvoc.yml.gz'
ENV = dict(os.environ, LD_LIBRARY_PATH='/home/bruno/Desktop/tesina/opencv_build/install/lib')


def cache_path(session, config):
    nfeat = config.get('orb_nfeatures', 2000)
    return CACHE / f'{session.dataset}_{session.seq}_orb{nfeat}.bin'


def run_one(session, config_path, config, outdir, force, binary):
    prefix = outdir / session.name
    done = Path(f'{prefix}_results.yml')
    if done.exists() and 'num_images' in done.read_text() and not force:
        return session.name, 'cached'
    CACHE.mkdir(parents=True, exist_ok=True)
    freq = config.get('frequency', session.frequency)
    cmd = [str(binary), '--input-left', str(session.left), '--input-right', str(session.right),
           '--calibration', str(config.get(f'calibration_{session.dataset}', session.calibration)),
           '--poses-file', str(session.poses), '--type', 'ORB', '--voc', str(VOC),
           '--frequency', str(freq), '--config', str(config_path), '--output', str(prefix),
           '--cache', str(cache_path(session, config))]
    if config.get('global_retrieval'):
        cmd += ['--global-desc', str(CACHE / f"{session.dataset}_{session.seq}_{config['global_desc']}.f32")]
    with open(f'{prefix}.log', 'w') as log:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=ENV, text=True)
        # el log completo es enorme (varias lineas por imagen): guardar solo la cola
        log.write('\n'.join(p.stdout.splitlines()[-40:]))
    return session.name, f'exit {p.returncode}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--sessions', default='fieldsafe')
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--name', help='nombre de la corrida (default: nombre del yaml)')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    config_path = Path(a.config).resolve()
    config = yaml.safe_load(config_path.read_text()) or {}
    name = a.name or config_path.stem
    outdir = REPO / 'evaluation/runs' / name
    outdir.mkdir(parents=True, exist_ok=True)
    sessions = select(a.sessions)

    # copia congelada del binario (se puede recompilar mientras corre) + commit usado
    binary = outdir / 'demo_stereo'
    shutil.copy2(BIN, binary)
    commit = subprocess.run(['git', 'describe', '--always', '--dirty'], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    (outdir / 'info.txt').write_text(f'commit: {commit}\nconfig: {config_path.name}\n{config_path.read_text()}')

    with ThreadPoolExecutor(a.jobs) as ex:
        for s, status in ex.map(lambda s: run_one(s, config_path, config, outdir, a.force, binary), sessions):
            print(f'  {s}: {status}', flush=True)

    rows = []
    for s in sessions:
        yml = outdir / f'{s.name}_results.yml'
        if not yml.exists():
            print(f'  {s.name}: sin resultados'); continue
        metrics, _ = evaluate(yml, load_gt(s.poses))
        rows.append(dict(config=name, session=s.name, dataset=s.dataset, **metrics))
    df = pd.DataFrame(rows)
    for ds, g in df.groupby('dataset'):
        rows.append(dict(config=name, session=f'TOTAL_{ds}', dataset=ds, **summarize(g.to_dict('records'))))
    df = pd.DataFrame(rows)
    df.to_csv(outdir / 'metrics.csv', index=False)
    cols = ['session', 'loops', 'tp', 'trivial', 'fp', 'precision', 'recall', 'coverage', 'pose_ok',
            'e_mag_med', 'e_vec_med', 'e_yaw_med']
    with pd.option_context('display.width', 200, 'display.float_format', '{:.3f}'.format):
        print(df[[c for c in cols if c in df]].to_string(index=False))


if __name__ == '__main__':
    main()
