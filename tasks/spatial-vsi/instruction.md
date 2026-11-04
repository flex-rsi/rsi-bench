# VSI-Bench: spatial reasoning from indoor video

You are improving a solver that answers spatial questions about egocentric videos of real indoor
rooms (VSI-Bench, "Thinking in Space", arXiv:2412.14171). Nothing is trained: the solver is
`solver.py`, which calls the model under test when it is evaluated.

## The questions

Each question is about one video (640x480, 24-30 fps, 1-3 minutes, a camera walking through a
room from ScanNet, ScanNet++ or ARKitScenes). There are ten question types in eight groups:

| Group | Answer | Example |
|---|---|---|
| `object_counting` | number | How many chair(s) are in this room? |
| `object_abs_distance` | number (meters) | the distance between two objects |
| `object_size_estimation` | number (centimeters) | the longest dimension of an object |
| `room_size_estimation` | number (square meters) | the size of the room |
| `object_rel_distance` | option letter | which object is closest to X |
| `object_rel_direction` (easy, medium, hard) | option letter | standing by X facing Y, is Z to my left or right |
| `route_planning` | option letter | fill in the turns of a route |
| `obj_appearance_order` | option letter | the order in which objects first appear |

## What is scored

The official VSI-Bench metric. A multiple-choice answer counts if its first word (a trailing
"." dropped) is the option letter. A numeric answer gets MRA: the share of the thresholds
t = 0.50, 0.55, ..., 0.95 with |answer - truth| / truth <= 1 - t. The first word of a numeric
answer must parse as a number, or it scores 0. `score` is the mean over the 8 groups (the three
direction levels are first averaged into one), in [0, 1]. `debiased_score` is the same on the
questions VSI-Bench-Debiased keeps (questions answerable without the video removed). It is
reported; it is not the target.

| Split | Questions | Who sees what |
|---|---|---|
| dev | 80, 46 scenes | you: questions, answers, videos, per-question results |
| val | 480, 84 other scenes | the orchestrator; you see the score only |
| test | 320, 100 other scenes | nobody during the run |

No scene is in two splits. The question types are balanced the same way in every split.

## The interface: `solver.py`

The evaluator imports `solver.py` from your workspace and calls

```python
answer_all(questions: list[dict], io_dir: Path) -> dict   # {question id: answer string}
```

once per evaluation. Each question has `id`, `video` (an absolute path to the mp4),
`question_type`, `question` and `options` (a list for multiple choice, else `None`). Files
written under `io_dir/trace/` come back with dev results. One evaluation has 4 hours.

`$RSIB_DEV_DATA/questions.json` lists the dev questions with their `answer` (and `debiased`);
the videos are under `$RSIB_DEV_DATA/videos/`.

## Calling the model

The solver calls the run's model through the gateway, as the starter does: `POST
$RSIB_GATEWAY_URL/v1/complete` with `Authorization: Bearer $RSIB_GATEWAY_TOKEN` and the JSON body
`{"system", "messages", "tools", "max_tokens", "session"}` (message content is a list of
`{"type": "text"}` and `{"type": "image", "mime", "data"}` parts). The reply has `text`,
`tool_calls` and `usage`.

- One evaluation may make at most **2 model calls per question** of its split (dev 160, val 960,
  test 640), counted over the whole split: the solver may spend them unevenly across questions.
  Past the limit a call gets HTTP 402 `split_call_limit`, and the questions left get no answer.
- Dev evaluations are charged to your budget, model calls included. Val and test calls are not.

## Rules

This is the **fixed-backbone** variant (the open variant is `spatial-vsi-open`).

- **Models.** The solver may call the run's model through the gateway, as often as the call
  limit allows, and no other model: no other model API, and no downloaded model weights (no
  depth, detection, segmentation or 3D reconstruction networks).
- **Everything else is yours to choose:** which frames and how many, at what resolution, crops,
  what the model is asked and in how many steps, tools it calls (code you write: geometry,
  classical vision with OpenCV, counting, voting), verification, and per-type strategies. What you
  learn on dev (priors, calibration, what works) may go into the solver.
- The answers must come from the video and the question. Do not look the scenes up: the solver
  must not read ScanNet, ScanNet++ or ARKitScenes annotations, VSI-Bench files other than the
  questions it is given, or anything on the network besides the gateway.

## Where the solver runs

In the evaluator's Python (3.12, with numpy, opencv-python-headless, Pillow). In a sandboxed run
it sees the workspace snapshot, the run cache and its own I/O directory (the split's questions and
videos), and nothing else.

## The starter

`solver.py` is the direct pipeline: 16 frames sampled evenly from the video (512 px wide) and
the official lmms-eval prompt, one model call per question (answer-only system prompt, markdown
stripped from the reply). Its val score is the baseline of the run.
