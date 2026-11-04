# Product weight from photos and dimensions with a fixed open VLM (Amazon Berkeley Objects)

Each item is a product from an online catalog: furniture, lamps, rugs, shoes, luggage, kitchen
and office items, decor, hardware and more (about 130 product types per split). You get its
catalog photos (the main photo and up to two more, longer side at most 1024 px), its coarse
product type and its dimensions in cm, and must estimate its **weight in kilograms**. The name,
brand and description are not given: what it is made of, how solid or hollow it is and how its
volume is filled must come from the photos. Weights range from 10 g to 150 kg (median about 1 kg).

The dimensions are the listing's: usually the item's bounding box, in an order (length, width,
height) that listings do not always follow.

## The interface: `solver.py`

`solver.py` must define

```python
def answer(item: dict) -> float | str:   # the weight in kg, e.g. 12.5 or "12.5"
```

`item` holds `id`, `images` (absolute paths of the photos, the main one first), `product_type`
(e.g. `chair`, `home furniture and decor`) and `dimensions_cm` (`{"length", "width",
"height"}`). A string reply is read as its first number.

The evaluator calls `answer` once per item of a split, in a separate process, with `WORKERS`
items at a time (a module constant of `solver.py`, default 8, at most 16), so `answer` must be
safe to call from several threads. An item whose `answer` raises or returns no number scores 0. A
whole split must finish within 45 minutes.

## The fixed backbone

This task fixes the vision-language model: **Qwen3.8-27B** (open weights), served by vLLM on the
run's machine. It is the only language or vision-language model a solution may use, and a solution
reaches it only through `model.py`, which calls the evaluator's proxy (`$VLM_URL`, `$VLM_KEY`,
OpenAI chat-completions format, images as base64 JPEG parts; thinking on or off per call). The
proxy limits each split: at most 8 images and 8 x 1024 x 1024 pixels per call, and at most 4 calls
per item on average (`MAX_CALLS` in `model.py`); it refuses anything beyond (HTTP 400). Backbone
calls are not charged to your budget.

While you work, the same server is at `$FIXED_VLM_URL` (model name `$FIXED_VLM_MODEL`), without the
proxy's limits, for your experiments; the evaluated solution must not use that address.

Besides the backbone, a solution may run **open non-language models and classical code** on one
GPU while it is evaluated: segmenters, depth estimators, image encoders (e.g. DINO or CLIP image
features), regressors or small heads you train on dev, numpy/scipy/scikit-learn. Weights and
packages must be downloaded into `$RSIB_CACHE` while you work (`HF_HOME`, `TORCH_HOME` and
`PIP_CACHE_DIR` point there); the evaluated solution must not use the network. If your solution
needs another Python interpreter, it must live in `$RSIB_CACHE`; write its absolute path in
`runtime.txt` in the workspace. No other LLM or VLM (API or open weights) may be used, in the
solution or to label data, and no text encoder may read data other than what `item` holds.

The starter sends the main photo, the product type and the dimensions in one call, with thinking
off, and reads the first number of the reply.

## Metric

`mra`, mean relative accuracy (higher is better), as in QuantiPhy: a prediction p of a true weight
y is correct at threshold t when |p − y| / |y| < 1 − t; an item's score is the share of the
thresholds t in {0.1, 0.2, ..., 0.9, 0.95} at which it is correct, and `mra` is the mean over the
split's items. `mra_light` (< 0.5 kg), `mra_medium` (0.5-5 kg), `mra_heavy` (> 5 kg),
`median_abs_log_ratio`, `answered`, `model_calls` and `refused_calls` are reported too.

## Data and rules

The dev data directory holds `images/`, `questions.jsonl`, `labels.jsonl` (`{"id", "answer"}`,
kg) and `items.jsonl` (`{"id", "name", "item_id"}`, for your analysis only: the evaluated
solution never gets names): 300 items, drawn the same way as the hidden val split (300 items) and
test split (600). Variants of one product (colors, sizes) are in one split only, and no product
type makes more than 3% of a split.
`evaluate_dev` returns the MRA and `per_question.json` with every answer.

Amazon Berkeley Objects is a public dataset. Do not download, search for or use ABO (its
listings, photos, 3D models or metadata), Amazon product pages or any other source of these
items' weights, and do not look for the validation or test data. Use only the dev data, your own
work and publicly released models and libraries within the rules above.
