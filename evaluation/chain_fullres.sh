#!/bin/bash
# Rosario 1280x720: misma cadena que la validacion (SALAD reutilizado: mismos frames, SALAD redimensiona igual)
cd /home/bruno/Desktop/tesina/StereoLoopDetectorFork/evaluation
PY=../.venv/bin/python; LG=../../lg_venv/bin/python
S=${1:-rof_1222_1631}
echo "== p3c5vo $S $(date +%H:%M)"
# tandas de a 2 por memoria
for B in ${S//;/ }; do $PY run_sld.py --config configs/p3_salad_c5.yaml --name p3c5vo --sessions $B --jobs 2 | grep -E "exit"; done
S=${S//;/,}
echo "== learned_verify $(date +%H:%M)"
$LG learned_verify.py p3c5vo --out p5_aliked --sessions $S 2>&1 | grep -E "verificados"
$LG odo_filter.py p5_aliked --vo-run p3c5vo --out p6_odo --sessions $S 2>&1 | grep -v -i warn
for r in p5_aliked p6_odo; do $PY run_sld.py --config configs/baseline.yaml --name $r --sessions $S --eval-only | grep rof_ ; done
echo CHAIN_DONE
