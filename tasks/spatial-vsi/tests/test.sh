#!/usr/bin/env bash
# Evaluator entry point: the solver answers the split's questions in a child process, then
# evaluate.py scores the answers with VSI-Bench's official metric.
set -euo pipefail
exec "${RSIB_PYTHON:-python3}" "$(dirname "$0")/evaluate.py"
