#!/bin/bash
# Validacion en Rosario: candidatos SALAD (+ pose de odometria) -> ALIKED+LightGlue -> filtro de odometria.
cd /home/bruno/Desktop/tesina/StereoLoopDetectorFork/evaluation
PY=../.venv/bin/python; LG=../../lg_venv/bin/python
echo "== p3c5vo $(date +%H:%M)"
# en tandas: cada demo_stereo guarda todas las features en RAM (~0.3 GB por cada 1000 frames)
$PY run_sld.py --config configs/p3_salad_c5.yaml --name p3c5vo --sessions ro_1222_1314,ro_1222_1429 --jobs 2 | grep -E "exit"
$PY run_sld.py --config configs/p3_salad_c5.yaml --name p3c5vo --sessions ro_1222_1631,ro_1226_1510 --jobs 2 | grep -E "exit"
$PY run_sld.py --config configs/p3_salad_c5.yaml --name p3c5vo --sessions ro_1226_1339,ro_1226_1548 --jobs 2 | grep -E "exit"
$PY run_sld.py --config configs/p3_salad_c5.yaml --name p3c5vo --sessions rosario --eval-only | grep TOTAL
echo "== learned_verify $(date +%H:%M)"
$LG learned_verify.py p3c5vo --out p5_aliked --sessions ro_1226_1548,ro_1226_1339 2>&1 | grep -E "verificados"
$PY run_sld.py --config configs/baseline.yaml --name p5_aliked --sessions rosario --eval-only
echo "== odo_filter $(date +%H:%M)"
$LG odo_filter.py p5_aliked --vo-run p3c5vo --out p6_odo --sessions rosario 2>&1 | grep -v -i warn
$PY run_sld.py --config configs/baseline.yaml --name p6_odo --sessions rosario --eval-only
$PY run_sld.py --config configs/baseline.yaml --name baseline --sessions rosario --eval-only
echo CHAIN_DONE
