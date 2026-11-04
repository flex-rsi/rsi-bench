#!/usr/bin/env bash
# Evaluator. Environment (see src/rsibench/orchestrator/evaluator.py):
#   RSIB_SPLIT, RSIB_DATA_DIR, RSIB_TRAIN_DIR, RSIB_WORKSPACE, RSIB_OUTPUT_DIR, RSIB_TASK_DIR, RSIB_PYTHON,
#   RSIB_SANDBOX
#   RSIB_GATEWAY_URL, RSIB_GATEWAY_TOKEN, only with [rsi.resources] model_access: the gateway and
#   a token for /v1/complete until the evaluation ends; hand them to the solution
# Must write $RSIB_OUTPUT_DIR/reward.json with the primary metric as a finite number, and exit 0.
# Run the submitted solution in a child process that sees only the split's inputs, then score
# its predictions here, where the labels are. When $RSIB_SANDBOX is a non-empty JSON list, start
# the child as  prefix + ["--rw", io_dir, "--cwd", io_dir, "--"] + command  so that it cannot read
# anything else (see solution_cmd() in tasks/toy-regression/tests/evaluate.py).
set -euo pipefail
echo "fill in tests/test.sh" >&2
exit 1
