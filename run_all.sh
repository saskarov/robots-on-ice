#!/usr/bin/env bash
# Reproduce every result and figure in results/ from scratch:
#   tests -> tuning (learning rate, then batch and initial sigma) -> m x C grid -> figures.
#
#   bash run_all.sh                       # uses `python` from the active environment
#   PYTHON=.venv/Scripts/python.exe bash run_all.sh   # or point to an interpreter
#
# Everything is seeded, so a rerun on the same setup gives the same numbers.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
WORKERS="${WORKERS:-4}"

step() { echo; echo "=== $1  ($(date +%H:%M:%S))"; }

step "1/5 tests";                 "$PY" -m pytest -q
step "2/5 tuning: learning rate"; "$PY" experiments/tune.py --stage lr --workers "$WORKERS"
step "3/5 tuning: batch, init σ"; "$PY" experiments/tune.py --stage other --lr 0.01 --workers "$WORKERS"
step "4/5 m x C grid";            "$PY" experiments/run_grid.py --workers "$WORKERS" --force
step "5/5 figures";               "$PY" experiments/make_plots.py
step "done"
