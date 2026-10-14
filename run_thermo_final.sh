#!/usr/bin/env bash
# Thermometer final run (docs/m1c_thermo_final_prespec.md), stages in order, one job at a time. Stops at the first failing stage.
set -e
cd "$(dirname "$0")"
PY=$HOME/.venvs/m1/bin/python
for s in 42 123 7 456 789; do $PY src/m1c_thermo_final_sim.py --seed $s; done
for s in 42 123 7 456 789; do $PY src/m1c_thermo_final_real.py --seed $s; done
$PY src/m1c_thermo_final_locked.py
for s in 42 123 7 456 789; do $PY src/m1c_thermo_final_near.py --seed $s; done
$PY src/m1c_thermo_final_summarize.py
echo "ALL STAGES DONE"
