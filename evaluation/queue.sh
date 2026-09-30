#!/bin/bash
# Cola de corridas en serie (sobrevive a reinicios de la sesion: lanzar con setsid).
cd /home/bruno/Desktop/tesina/StereoLoopDetectorFork/evaluation
PY=../.venv/bin/python
for c in "$@"; do
  name=${c%%:*}; sess=${c#*:}; [ "$sess" = "$c" ] && sess=fieldsafe
  echo "== $name ($sess) $(date +%H:%M)"
  $PY run_sld.py --config configs/$name.yaml --sessions $sess --jobs 5 | grep -E "TOTAL|exit [^0]"
done
echo QUEUE_DONE
