#!/usr/bin/env bash
# Evaluator entry point; runs inside the task container ("pytorch") on a GPU node.
set -euo pipefail
exec "${RSIB_PYTHON:-python}" "${RSIB_TASK_DIR}/tests/evaluate.py"
