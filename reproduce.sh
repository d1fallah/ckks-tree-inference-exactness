#!/usr/bin/env bash
# Single-command reproduction of M1's Track A headline results, per
# CLAUDE.md Sec 1.3 and guide v4.1 Sec 7's Experiment Design Checklist
# ("Scripts committed: ... reproduce.sh"). Thin wrapper around
# experiments/run_reproducible_m1.py -- see that script's docstring for
# what each stage does and which venv it needs.
#
# Usage:
#   ./reproduce.sh                 # from the committed cohort/features parquets
#   ./reproduce.sh --from-scratch  # also rebuilds the MIMIC cohort/crosswalk
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PY="${PYTHON_FOR_REPRODUCE:-$HOME/.venvs/m1/bin/python}"
if [ ! -x "$PY" ]; then
  echo "error: $PY not found. See environment_snapshot.md for how the venv was built," >&2
  echo "or set PYTHON_FOR_REPRODUCE to a python with pyyaml installed." >&2
  exit 1
fi

exec "$PY" experiments/run_reproducible_m1.py --config configs/paper_track_m1.yaml "$@"
