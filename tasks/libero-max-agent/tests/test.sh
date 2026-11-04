#!/usr/bin/env bash
set -euo pipefail
exec "${RSIB_PYTHON:-python3}" "$(dirname "$0")/evaluate.py"
