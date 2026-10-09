#!/bin/sh
# quant-lab: run the self-checks, then the full story of eight experiments.
# The first run trains the model and caches it in assets/base.npz.
set -e
cd "$(dirname "$0")"

if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
good() { printf '\033[32m%s\033[0m\n' "$1"; }
bad()  { printf '\033[31m%s\033[0m\n' "$1"; }

bold 'quant-lab'
echo "  python : $($PY --version 2>&1)"
if ! "$PY" -c 'import numpy' 2>/dev/null; then
  bad '  numpy  : missing - install it with: pip install numpy'
  exit 1
fi
echo '  numpy  : ok'

START=$(date +%s)

bold '-- self-checks'
"$PY" selfcheck.py || { bad 'checks failed'; exit 1; }

bold '-- experiments'
"$PY" experiments.py all || { bad 'an experiment failed'; exit 1; }

END=$(date +%s)
good "-- all done in $((END - START)) s"
