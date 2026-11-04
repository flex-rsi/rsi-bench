"""Entry point the orchestrator calls for every harness phase.

    python harness/run_phase.py <executor|improver|meta|accept> --context ctx.json --output out.json

The context file describes the task, the round, the history and the budget. The output file
must contain {"status": ..., "summary": ..., "steps": ...} (accept: {"accept": bool, "reason"}).
This file is part of scope E: the improver may change it, but if it stops working the round is
lost and repeated crashes roll the harness back.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["executor", "improver", "meta", "accept"])
    parser.add_argument("--context", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    ctx = json.loads(Path(args.context).read_text())
    out_path = Path(args.output)
    try:
        if args.phase == "accept":
            from harness.improver.accept import decide
            result = {"status": "ok", **decide(ctx["accept_input"])}
        else:
            from harness.core.gateway_client import Gateway
            gateway = Gateway()
            if args.phase == "executor":
                from harness.executor.phase import run
            elif args.phase == "improver":
                from harness.improver.phase import run
            else:
                from harness.improver.phase import run_meta as run
            result = run(ctx, gateway)
        out_path.write_text(json.dumps(result, default=str))
        return 0
    except Exception:
        out_path.write_text(json.dumps({"status": "crash", "error": traceback.format_exc()}))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
