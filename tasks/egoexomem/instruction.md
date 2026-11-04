# Memory over synchronized egocentric and exocentric video (EgoExoMem)

Each question is about one recorded activity, cooking or bike repair, filmed at the same time by
two cameras: the **ego** view from glasses the person doing the task wears (448 x 448) and the
**exo** view from a fixed camera in the room (796 x 448). The videos last from 20 seconds to
23 minutes (about 6 minutes typically). The questions ask what the videos show at some moment or
over time, so the answer has to be found and remembered across the video, often in only one of the
two views. They are multiple choice with four options (A-D), in seven kinds (`category`):

| category | asks about |
|---|---|
| `habitual_location` | where an object usually is |
| `instantaneous_position` | where an object is at a moment the question describes |
| `resulting_location` | where an object ends up after an action |
| `egocentric_direction` | where something is relative to the camera wearer |
| `object_state` | the state of an object at a moment |
| `allocentric_relation` | where one object is relative to another |
| `temporal_ordering` | the order of a sequence of steps |

**The frame budget.** The model under test sees at most 16 x 448 x 448 = 3,211,264 pixels per
call: the images of one call may hold that many pixels together (16 ego frames at full size, or
more frames at a smaller size), and the evaluator refuses a call with more (HTTP 400, error
`frame_budget`). A video of several minutes has hundreds of seconds of footage in two views, so
you choose what the model looks at: which moments, which view, how large, in one call or several.

## The interface: `solver.py`

`solver.py` must define

```python
def answer(item: dict) -> str:   # the letter of the chosen option, e.g. "B"
```

`item` holds `id`, `video` (an opaque id; the questions of one video share it), `family`
(`cooking` or `bike`), `scenario` (e.g. "Cooking Noodles"), `category`, `question`, `options` (the
option texts, in order A, B, ...), `prompt` (the question, the lettered options and an answer-format
line), `duration_s`, and the paths of the video:

- `ego_frames`, `exo_frames`: directories of JPEG frames taken once per second (`frames_per_second`
  is 1): `000000.jpg` is at 0 s, `000001.jpg` at 1 s, and so on; the two views are synchronized,
  so the same file name is the same moment;
- `ego_video`, `exo_video`: the 30 fps MP4 videos (H.264), for finer timing.

The evaluator calls `answer` once per question of a split, in a separate process, with `WORKERS`
questions at a time (a module constant of `solver.py`, default 8, at most 16), so `answer` must be
safe to call from several threads. A question whose `answer` raises or returns no valid letter scores
0. A whole split must finish within 60 minutes.

## Model access

The solution calls the model under test while it is evaluated: `model.py` in the workspace sends one
request through the evaluator's gateway (`$RSIB_GATEWAY_URL`, `$RSIB_GATEWAY_TOKEN`), and its
`image_part` shrinks an image to a pixel budget; you may change or replace both. A request is the
gateway's `/v1/complete` body: `system`, `messages` (a whole conversation, so a question may be a
multi-turn exchange), `tools`, `max_tokens`. **The call budget:** a split of n questions may make
at most 3n model calls in total, 3 per question on average (`MAX_CALLS` in `model.py`, from
`$EGOEXOMEM_MAX_CALLS`); the evaluator refuses every call after that (HTTP 400, error
`call_budget`), so spend calls where they help. Model calls on dev are charged to your budget; on
val and test they are metered only.

The starter is the direct pipeline: 8 frames of each view, evenly spaced over the video and shrunk
to fit the frame budget together, each labeled with its view and time, then the prompt; one call per
question.

The evaluation runs on one GPU inside a PyTorch container (Python 3.12, numpy, Pillow, PyTorch);
a video decoder is not guaranteed, which is why the frames are provided. Open models (for example to
retrieve the frames a question needs), code and Python packages you need at evaluation time must be
downloaded into `$RSIB_CACHE` while you work (`HF_HOME`, `TORCH_HOME` and `PIP_CACHE_DIR` point
there); the evaluated solution must not use the network. If your solution needs another Python
interpreter, it must live in `$RSIB_CACHE`; write its absolute path in `runtime.txt` in the
workspace.

## Metric

`accuracy` over the split's questions (higher is better). `acc_cooking`, `acc_bike`, the accuracy
of every category (`acc_<category>`), `model_calls` (calls the evaluation forwarded) and
`refused_calls` (frame or call budget) are reported too.

## Data and rules

The dev data directory holds `videos/<video>/{ego,exo}.mp4`, `frames/<video>/{ego,exo}/`,
`videos.jsonl` (duration, size and frame counts of each video), `questions.jsonl` (the items above,
without the paths), `labels.jsonl` (`{"id", "answer"}`) and `evidence.jsonl`
(`{"id", "timestamps", "view", "note"}`: the seconds at which the answer can be seen, the view that
shows it (`ego`, `exo` or `either`) and the annotators' note): 80 questions on 38 videos. The hidden
val split (80 questions on 33 videos) and test split (120 questions on 47 videos) are drawn the same
way; no kitchen or workshop session is in two splits.
`evaluate_dev` returns the accuracy and `per_question.json` with every answer.

The questions come from EgoExoMem, a public benchmark, and the videos from EgoExo4D. Do not
download, search for or use EgoExoMem, its annotations or its answers from anywhere (GitHub,
Hugging Face, papers, other copies), do not use EgoExo4D's own annotations or other copies of its
videos, and do not look for the validation or test data. Use only the dev data, your own work and
publicly released models and libraries.
