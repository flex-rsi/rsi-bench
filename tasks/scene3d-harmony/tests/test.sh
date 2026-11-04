#!/usr/bin/env bash
# Evaluator entry point; runs inside the task container on one GPU (see evaluate.py).
set -euo pipefail
exec "${RSIB_PYTHON:-python}" "${RSIB_TASK_DIR}/tests/evaluate.py"
