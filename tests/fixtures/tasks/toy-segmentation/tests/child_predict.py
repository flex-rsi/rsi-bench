"""Runs the submitted segmenter in a separate process that never sees the scored masks."""

import json
import os
import sys
from pathlib import Path

io = Path(os.environ["RSIB_IO_DIR"])
sys.path.insert(0, os.environ["RSIB_SOLVER_DIR"])
import segmenter  # noqa: E402

train_images = json.loads((io / "train_inputs.json").read_text())["images"]
train_masks = json.loads((io / "train_labels.json").read_text())["masks"]
images = json.loads((io / "query_inputs.json").read_text())["images"]
masks = segmenter.predict(train_images, train_masks, images)
(io / "pred.json").write_text(json.dumps([[[int(v) for v in row] for row in m] for m in masks]))
