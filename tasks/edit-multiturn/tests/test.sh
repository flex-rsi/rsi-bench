#!/usr/bin/env bash
# Evaluator entry point (see docs/task-authoring.md, "The evaluator contract"). Needs
# EDIT_MULTITURN_MODELS and EDIT_MULTITURN_JUDGE_HOME, passed with RSIB_PASS_ENV.
set -euo pipefail
exec "${RSIB_PYTHON:-python3}" "${RSIB_TASK_DIR}/tests/evaluate.py"
