# Cross-view instance tracking (IGGT 3D tracking benchmark)

Each clip is 6-8 RGB frames (448 x 592) of one indoor scene, taken from widely different
viewpoints. Segment the object instances and give each object the same id in every frame where it
appears. No camera poses, depth or intrinsics are given; only the RGB frames.

`tracker.py` must define

```python
def predict(frames_dir: str) -> dict:   # {"idmap": int32 array (S, H, W), "scores": array}
```

`frames_dir` contains `000.png`, `001.png`, ... `idmap[f, y, x]` is the instance id of that pixel
in frame f (0 = background); ids must be consistent across frames. `scores` holds one confidence
per instance (not used by the metric).

## Evaluation

The evaluator calls `predict` once per clip of the split, in a separate process, on one GPU inside
the task container (Python 3.12 with PyTorch, numpy, scipy, Pillow), with no network access. A
whole split must finish within 45 minutes. Predicted tracks are matched one-to-one to ground-truth
object tracks by temporal IoU (greedy); unmatched ground-truth tracks score 0, unmatched predictions
are not penalized.

- Primary metric: **T-mIoU**, the mean over ground-truth tracks of the temporal IoU
  (sum over frames of |pred ∩ gt| / sum over frames of |pred ∪ gt|), averaged over clips. Higher is better.
- Also reported: T-SR (share of tracks with per-frame IoU > 0.5 in every frame where they appear),
  J (region IoU), F (boundary F-measure), J&F.

The ground truth labels a subset of object instances per clip (typically 4-8 objects such as
furniture, lamps, decorations); walls, floors and ceilings are not labelled.

## Data and rules

- The dev data directory has `inputs/<clip>/frames/*.png` and `labels/<clip>/gt.npz`
  (`inst_ids`, (S, H, W) int32, 0 = background) for 100 clips built the same way from other scenes.
- No training of models is required; you may use any publicly released pretrained model or
  library and combine their outputs as you see fit.
- Your code only sees the worktree, the dev data and the run cache `$RSIB_CACHE`. Download weights,
  code and Python packages into `$RSIB_CACHE` while you work (`HF_HOME`, `TORCH_HOME` and
  `PIP_CACHE_DIR` already point there); the evaluator reads it but has no network, so nothing can
  be downloaded during evaluation.
- If your solution needs a different Python interpreter, it must live inside `$RSIB_CACHE`; write
  its absolute path in `runtime.txt` in the workspace and the evaluator runs `predict` with it.
- Files larger than 5 MB are never kept with the solution; keep large weights in `$RSIB_CACHE`.
- Use only the dev data and pretrained resources; do not look for the validation or test data.
