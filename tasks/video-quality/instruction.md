# Video quality assessment

Judge how good videos look to people, in two kinds of items (`kind`):

- **`rate`**: one real user-generated video (16.6 to 20 seconds, 360p to 1080p, from 14 content
  categories such as gaming, sports, lectures, vlogs, vertical phone videos and HDR). Give it a
  quality score. People rated each video from 1 (bad) to 5 (excellent); the label is the mean of
  105 to 154 ratings (the MOS). Only the order of your scores within a split matters.
- **`compare`**: two AI-generated clips, A and B (5 to 6 seconds), made by two text-to-video models
  from the same prompt. Answer three questions, each with `A` or `B`: **`alignment`**, which clip
  matches the prompt better; **`artifacts`**, which clip has fewer visual artifacts and glitches
  (looks less AI-generated); **`preference`**, which clip looks better overall. The labels are the
  majority of 15 weighted votes per question; a question where neither clip got more than 55% of
  the votes is a tie and is not scored. Raters of `artifacts` and `preference` did not see the prompt.

**The frame budget.** The model under test sees at most 8 x 1024 x 576 = 4,718,592 pixels per
call: the images of one call may hold that many pixels together (8 frames of 1024 x 576, or more
frames at a smaller size), and the evaluator refuses a call with more (HTTP 400, error
`frame_budget`). What the model sees decides what it can judge: which moments, how many frames,
at what size or crop, in one call or several.

## The interface: `solver.py`

`solver.py` must define

```python
def rate(item: dict) -> float:     # a quality score; higher is better
def compare(item: dict) -> dict:   # {"alignment": "A" or "B", "artifacts": ..., "preference": ...}
```

A `rate` item holds `id`, `kind`, `video` (an opaque id), `category` and the clip's fields:
`path` (the MP4, H.264, no audio), `frames` (a directory of JPEG frames at full resolution, taken
`frames_per_second` = 2 times a second: `000000.jpg` is at 0 s, `000001.jpg` at 0.5 s, and so on),
`duration_s`, `width`, `height` and `fps`. A `compare` item holds `id`, `kind`, `prompt` and
`videos`: the two clips, A first, each with the same fields.

The evaluator calls `rate` or `compare` once per item of a split, in a separate process, with
`WORKERS` items at a time (a module constant of `solver.py`, default 8, at most 16), so both must be
safe to call from several threads. A rated video whose `rate` raises or returns no number gets the
median of the split's other scores; a question `compare` does not answer with `A` or `B` counts as
wrong. A whole split must finish within 60 minutes.

## Model access

The solution calls the model under test while it is evaluated: `model.py` in the workspace sends one
request through the evaluator's gateway (`$RSIB_GATEWAY_URL`, `$RSIB_GATEWAY_TOKEN`), and its
`image_part` shrinks an image to a pixel budget; you may change or replace both. A request is the
gateway's `/v1/complete` body: `system`, `messages` (a whole conversation, so an item may be a
multi-turn exchange), `tools`, `max_tokens`. **The call budget:** a split of n items may make at
most 3n model calls in total, 3 per item on average (`MAX_CALLS` in `model.py`, from
`$VIDEO_QUALITY_MAX_CALLS`); the evaluator refuses every call after that (HTTP 400, error
`call_budget`), so spend calls where they help. Model calls on dev are charged to your budget; on
val and test they are metered only.

The starter is the direct pipeline: 8 frames of the video (6 of each clip for `compare`), evenly
spaced and shrunk to fit the frame budget together, each labeled with its time; then the question;
one call per item.

The evaluation runs on one GPU inside a PyTorch container (Python 3.12, numpy, Pillow, PyTorch);
a video decoder is not guaranteed, which is why the frames are provided. Open models, code and
Python packages you need at evaluation time must be downloaded into `$RSIB_CACHE` while you work
(`HF_HOME`, `TORCH_HOME` and `PIP_CACHE_DIR` point there); the evaluated solution must not use the
network. If your solution needs another Python interpreter, it must live in `$RSIB_CACHE`; write
its absolute path in `runtime.txt` in the workspace.

## Metric

`score` = (`srcc` + `pair_accuracy`) / 2, higher is better: `srcc` is the Spearman rank correlation
of the rated videos' scores with their MOS, `pair_accuracy` the share of the pairs' non-tied
questions answered right. `plcc` (the Pearson correlation of the scores with the MOS),
`acc_alignment`, `acc_artifacts`, `acc_preference`, `model_calls` (calls the evaluation forwarded)
and `refused_calls` (frame or call budget) are reported too.

## Data and rules

The dev data directory holds `videos/<video>.mp4`, `frames/<video>/`, `videos.jsonl` (duration, size,
frame rate and frame count of each clip), `items.jsonl` (the items above, without the paths),
`labels.jsonl` (`{"id", "mos"}` for a rated video; `{"id", "alignment", "artifacts", "preference"}`,
each `A`, `B` or `tie`, for a pair) and `details.jsonl` (the spread and number of the ratings and
the MOS of three 10-second windows of each rated video; the vote shares of each pair, A first, with
`artifacts` the share that found the clip the more glitchy one): 30 rated videos and 15 pairs. The
hidden val split (30 videos, 15 pairs) and test split (40 videos, 20 pairs) are drawn the same way:
every split covers every content category and the whole range of ratings, and the pairs of every
split pit the same text-to-video models against each other. `evaluate_dev` returns the score and
`per_item.json` with every answer.

The rated videos and their ratings come from YouTube-UGC, the pairs and their votes from Rapidata's
public text-to-video preference data. Do not download, search for or use these datasets, their
ratings or votes or other copies of their videos from anywhere, and do not look for the validation
or test data. Use only the dev data, your own work and publicly released models and libraries.
