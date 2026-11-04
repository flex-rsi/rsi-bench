#!/usr/bin/env bash
# Evaluator entry point (see src/rsibench/orchestrator/evaluator.py for the RSIB_* contract).
set -euo pipefail
exec "${RSIB_PYTHON:-python3}" "${RSIB_TASK_DIR}/tests/evaluate.py"
