"""Runs the submitted solver in a separate process that only sees the training set and the
query inputs (never the labels being scored)."""

import json
import os
import sys
from pathlib import Path

io = Path(os.environ["RSIB_IO_DIR"])
sys.path.insert(0, os.environ["RSIB_SOLVER_DIR"])
import solver  # noqa: E402

train_x = json.loads((io / "train_inputs.json").read_text())["x"]
train_y = json.loads((io / "train_labels.json").read_text())["y"]
query_x = json.loads((io / "query_inputs.json").read_text())["x"]
preds = solver.fit_predict(list(train_x), list(train_y), list(query_x))
(io / "pred.json").write_text(json.dumps([float(p) for p in preds]))
