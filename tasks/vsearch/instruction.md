# Visual search with a limited field of view (V*Bench + HR-Bench 8K)

Each question is about one or two small objects in a large photo: about 2K (V*Bench) or about 8K
(HR-Bench 8K, up to 59 megapixels). The objects are tiny; on V* half of them cover less than 0.07%
of the photo. The questions are multiple choice: four options (A-D), or two (A-B) for V* questions
about left and right.

**The field of view.** The model under test sees at most 1024 x 1024 = 1,048,576 pixels per call:
the images of one call may hold that many pixels together, and the evaluator refuses a call with more
(HTTP 400, error `field_of_view`). A whole photo therefore reaches the model shrunk, by up to 7x on
each side for HR-Bench, and the objects may become a few pixels. You choose what the model looks at:
the whole photo shrunk, a full-resolution crop, several smaller crops in one call, several calls.

## The interface: `solver.py`

`solver.py` must define

```python
def answer(item: dict) -> str:   # the letter of the chosen option, e.g. "B"
```

`item` holds `id`, `image` (absolute path of the original photo, JPEG or PNG), `source` (`vstar` or
`hrbench8k`), `category` (V*: `direct_attributes`, `relative_position`; HR-Bench: `single`,
`cross`), `question`, `options` (the option texts, in order A, B, ...) and `prompt` (the question, the
lettered options and an answer-format line).

The evaluator calls `answer` once per question of a split, in a separate process, with `WORKERS`
questions at a time (a module constant of `solver.py`, default 8, at most 16), so `answer` must be
safe to call from several threads. A question whose `answer` raises or returns no valid letter scores
0. A whole split must finish within 45 minutes.

## Model access

The solution calls the model under test while it is evaluated: `model.py` in the workspace sends one
request through the evaluator's gateway (`$RSIB_GATEWAY_URL`, `$RSIB_GATEWAY_TOKEN`), and its
`image_part` shrinks an image to the field of view; you may change or replace both. A request is the
gateway's `/v1/complete` body: `system`, `messages` (a whole conversation, so a question may be a
multi-turn exchange), `tools`, `max_tokens`. **The call budget:** a split of n questions may make
at most 5n model calls in total, 5 per question on average (`MAX_CALLS` in `model.py`, from
`$VSEARCH_MAX_CALLS`); the evaluator refuses every call after that (HTTP 400, error `call_budget`),
so spend calls where they help. Model calls on dev are charged to your budget; on val and test they
are metered only.

The starter is the direct pipeline: the whole photo shrunk to the field of view and the prompt, one
call per question.

The evaluation runs on one GPU with Python, numpy, scipy, Pillow and PyTorch. Open models, code and
Python packages you need at evaluation time must be downloaded into `$RSIB_CACHE` while you work
(`HF_HOME`, `TORCH_HOME` and `PIP_CACHE_DIR` point there); the evaluated solution must not use the
network. If your solution needs another Python interpreter, it must live in `$RSIB_CACHE`; write its
absolute path in `runtime.txt` in the workspace.

## Metric

`accuracy` over the split's questions (higher is better). `acc_vstar`, `acc_hrbench8k`,
`model_calls` (calls the evaluation forwarded) and `refused_calls` (field of view or call budget)
are reported too.

## Data and rules

The dev data directory holds `images/`, `questions.jsonl`, `labels.jsonl` (`{"id", "answer"}`)
and `targets.jsonl` (for V* questions, the objects each is about, `{"name", "box": [x, y, w, h]}` in
pixels of the original photo): 70 questions (33 V*, 37 HR-Bench) drawn the same way as the hidden
val split (80 questions: 40 V*, 40 HR-Bench) and test split (140: 70 and 70); no photo is in two
splits.
`evaluate_dev` returns the accuracy and `per_question.json` with every answer.

V*Bench and HR-Bench are public benchmarks. Do not download, search for or use either benchmark,
its annotations or its answers from anywhere (Hugging Face, GitHub, papers, other copies), and do not
look for the validation or test data. Use only the dev data, your own work and publicly released
models and libraries.
