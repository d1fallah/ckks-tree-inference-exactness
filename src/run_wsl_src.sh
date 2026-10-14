#!/bin/bash
# Run an M1 research script from src/ inside WSL2 with the module's TenSEAL venv.
# usage (from Windows): wsl -e bash "/mnt/c/.../module 1/src/run_wsl_src.sh" <script.py>
cd "$(dirname "$0")/.." || exit 1
exec "$HOME/.venvs/m1/bin/python" "src/$1"
