"""Runs the submitted classifier in a separate process that never sees the scored labels."""

import json
import os
import sys
from pathlib import Path

io = Path(os.environ["RSIB_IO_DIR"])
sys.path.insert(0, os.environ["RSIB_SOLVER_DIR"])
import classifier  # noqa: E402

train_x = json.loads((io / "train_inputs.json").read_text())["x"]
train_y = json.loads((io / "train_labels.json").read_text())["y"]
x = json.loads((io / "query_inputs.json").read_text())["x"]
preds = classifier.fit_predict(train_x, train_y, x)
(io / "pred.json").write_text(json.dumps([int(p) for p in list(preds)]))
