#!/usr/bin/env bash
# Automated setup: creates a venv (Python 3.11 — vizdoom's prebuilt Apple
# Silicon wheels lag behind the newest CPython, so this pins to 3.11 rather
# than whatever `python3` resolves to) and installs requirements.txt.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3.11}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: $PYTHON_BIN not found." >&2
  echo "Install it with: brew install python@3.11" >&2
  echo "(or set PYTHON_BIN=/path/to/python3.11 and re-run this script)" >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  uv venv --python "$PYTHON_BIN" .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install -r requirements.txt
else
  "$PYTHON_BIN" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
fi

echo
echo "Setup complete."
echo "Activate the venv in new shells with: source .venv/bin/activate"
echo "Freedoom (bundled inside the vizdoom package) is used automatically — no commercial Doom files needed."
echo "First 'laya' controller run downloads the convaiinnovations/laya checkpoint from Hugging Face"
echo "(~800MB; see README for measured download/load time on this machine)."
echo
echo "Try:"
echo "  python -m experiments.run --controller random --episodes 3 --scenario basic"
