#!/usr/bin/env bash
set -euo pipefail
exec "${RSIB_PYTHON:-python3}" "${RSIB_TASK_DIR}/tests/evaluate.py"
