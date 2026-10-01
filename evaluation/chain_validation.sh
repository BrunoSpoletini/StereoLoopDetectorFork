#!/bin/bash
# Validacion en Rosario: candidatos SALAD (+ pose de odometria) -> ALIKED+LightGlue -> filtro de odometria.
cd /home/bruno/Desktop/tesina/StereoLoopDetectorFork/evaluation
PY=../.venv/bin/python; LG=../../lg_venv/bin/python
echo "== p3c5vo $(date +%H:%M)"
$PY run_sld.py --config configs/p3_salad_c5.yaml --name p3c5vo --sessions rosario --jobs 6 | grep -E "TOTAL|exit [^0]"
echo "== learned_verify $(date +%H:%M)"
$LG learned_verify.py p3c5vo --out p5_aliked --sessions ro_1226_1548,ro_1226_1339 2>&1 | grep -E "verificados"
$PY run_sld.py --config configs/baseline.yaml --name p5_aliked --sessions rosario --eval-only
echo "== odo_filter $(date +%H:%M)"
$LG odo_filter.py p5_aliked --vo-run p3c5vo --out p6_odo --sessions rosario 2>&1 | grep -v -i warn
$PY run_sld.py --config configs/baseline.yaml --name p6_odo --sessions rosario --eval-only
$PY run_sld.py --config configs/baseline.yaml --name baseline --sessions rosario --eval-only
echo CHAIN_DONE
