#!/usr/bin/env bash
# Evaluator entry point: one EmbodiedSWE grading rollout per seed of the split (see evaluate.py).
set -euo pipefail
exec "${RSIB_PYTHON:-python3}" "$(dirname "$0")/evaluate.py"
