#!/usr/bin/env bash
# Evaluator entry point (see docs/task-authoring.md, "The evaluator contract").
set -euo pipefail
exec "${RSIB_PYTHON:-python3}" "${RSIB_TASK_DIR}/tests/evaluate.py"
