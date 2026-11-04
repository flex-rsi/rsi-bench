"""Runs the workspace's solver in its own process: it sees the split's questions and videos,
never the labels."""

import json
import os
import sys
from pathlib import Path

io = Path(os.environ["RSIB_IO_DIR"])
sys.path.insert(0, os.environ["RSIB_SOLVER_DIR"])
import solver  # noqa: E402

questions = json.loads((io / "questions.json").read_text())
for q in questions:
    q["video"] = str(io / q["video"])
answers = solver.answer_all(questions, io)
(io / "answers.json").write_text(json.dumps({str(k): str(v) for k, v in answers.items()}))
