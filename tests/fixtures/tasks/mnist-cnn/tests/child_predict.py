"""Trains and predicts with the submitted model in a separate process (no access to the labels
being scored). Runs inside the task container on a GPU node."""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

io = Path(os.environ["RSIB_IO_DIR"])
sys.path.insert(0, os.environ["RSIB_SOLVER_DIR"])
import model  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
train_x = np.load(io / "train_images.npy")
train_y = np.load(io / "train_labels.npy")
x = np.load(io / "query_images.npy")
start = time.time()
preds = np.asarray(model.fit_predict(train_x, train_y, x, device=device)).astype(np.int64)
np.save(io / "pred.npy", preds)
print(f"device={device} fit_predict_seconds={time.time() - start:.1f}")
