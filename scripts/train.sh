#!/usr/bin/env bash
# Thin wrapper (Mask3D-style). Run from anywhere; pass Hydra overrides.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec python -m shadowcorr.train_app "$@"
