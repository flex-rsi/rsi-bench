# Quantitative kinematics from video with a fixed open VLM (QuantiPhy)

Each question asks for one physical quantity of an object in a short video (a few seconds): a
size, a distance, a velocity or an acceleration, in the units the question names (cm, m, cm/s,
m/s, m/s², ...). The question comes with one **prior** in physical units: a known length ("ruler
calibre = 1 cm"), a known speed ("walking velocity = 1.25 m/s") or a known acceleration, and for
some videos depths (`depth_info`). The prior fixes the scale between pixels and the world; the
answer follows from measuring the object in pixels (and over time, with the fps) and applying it.
Videos are simulations, lab recordings, internet clips and segmented renderings.

`inference_type` names the kind of question: the first letter is the prior (`S` static: a size;
`D` dynamic: a speed or an acceleration), the second the target (`S` a size or a distance; `D` a
speed or an acceleration). Questions about a moment name it ("at 3.0s").

## The interface: `solver.py`

`solver.py` must define

```python
def answer(item: dict) -> float | str:   # the value in the question's unit, e.g. 4.8 or "4.8"
```

`item` holds `id`, `video` (absolute path of the mp4), `video_id`, `source` (`simulation`, `lab`,
`internet`, `segmentation`), `video_type`, `fps`, `inference_type`, `question`, `prior` and
`depth_info` (a string or null). A string reply is read as its first number.

The evaluator calls `answer` once per question of a split, in a separate process, with `WORKERS`
questions at a time (a module constant of `solver.py`, default 8, at most 16), so `answer` must be
safe to call from several threads. A question whose `answer` raises or returns no number scores 0.
A whole split must finish within 45 minutes.

## The fixed backbone

This task fixes the vision-language model: **Qwen3.8-27B** (open weights), served by vLLM on the
run's machine. It is the only language or vision-language model a solution may use, and a solution
reaches it only through `model.py`, which calls the evaluator's proxy (`$VLM_URL`, `$VLM_KEY`,
OpenAI chat-completions format, images as base64 JPEG parts; thinking on or off per call). The
proxy limits each split: at most 32 images and 32 x 640 x 360 pixels per call, and at most 8 calls
per question on average (`MAX_CALLS` in `model.py`); it refuses anything beyond (HTTP 400).
Backbone calls are not charged to your budget.

While you work, the same server is at `$FIXED_VLM_URL` (model name `$FIXED_VLM_MODEL`), without the
proxy's limits, for your experiments; the evaluated solution must not use that address.

Besides the backbone, a solution may run **open non-language vision models and classical code**
on one GPU while it is evaluated: point trackers, optical flow, detectors, segmenters, depth or
pose estimators, OpenCV, numpy/scipy. Weights and packages must be downloaded into `$RSIB_CACHE`
while you work (`HF_HOME`, `TORCH_HOME` and `PIP_CACHE_DIR` point there); the evaluated solution
must not use the network. If your solution needs another Python interpreter, it must live in
`$RSIB_CACHE`; write its absolute path in `runtime.txt` in the workspace. No other LLM or VLM
(API or open weights) may be used, in the solution or to label data.

The starter sends four evenly spaced frames, the question and the prior in one call, with
thinking off, and reads the first number of the reply.

## Metric

`mra`, QuantiPhy's mean relative accuracy (higher is better): a prediction p of a true value y is
correct at threshold t when |p − y| / |y| < 1 − t; a question's score is the share of the
thresholds t in {0.1, 0.2, ..., 0.9, 0.95} at which it is correct, and `mra` is the mean over the
split's questions. `mra_ss`, `mra_sd`, `mra_ds`, `mra_dd` (by inference type), `answered`,
`model_calls` (backbone calls forwarded) and `refused_calls` are reported too.

## Data and rules

The dev data directory holds `videos/`, `questions.jsonl` and `labels.jsonl` (`{"id",
"answer"}`): 39 questions on 6 videos. The hidden val split has 52 questions on 8 videos and the
test split 67 on 10; no video is in two splits, and all three mix the four sources and inference
types. The splits are small, so prefer general methods (measure in pixels, track over time, apply
the prior) over tuning to dev.
`evaluate_dev` returns the MRA and `per_question.json` with every answer.

QuantiPhy is a public benchmark. Do not download, search for or use QuantiPhy, its videos, its
annotations or its answers from anywhere (Hugging Face, GitHub, the project site, papers, other
copies), and do not look for the validation or test data. Use only the dev data, your own work
and publicly released models and libraries within the rules above.
