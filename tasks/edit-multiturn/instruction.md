# Multi-turn image editing (fixed backbone)

Improve a multi-turn image-editing system. A session is a source image and five editing
instructions applied one after another, each to the result of the previous turn, e.g. "Remove
the person from the chair." → "Add a cat sitting on the chair." → "Change the chair color to
vibrant pink." → ... The instructions add, remove and replace objects, change attributes,
backgrounds, posture, expression, orientation, position, actions and interactions, restyle the
whole image or move the camera (dolly in or out, pan, tilt, orbit). A turn's instruction is only
known when it is applied: the next one arrives after your system has returned the current turn's
image. Every turn's image is scored.

`editor.py` must define

```python
from PIL import Image

class SessionEditor:
    def __init__(self, device: str = "cuda"): ...
    def reset(self, image: Image.Image) -> None: ...      # a new session starts from this image
    def step(self, instruction: str) -> Image.Image: ...  # this turn's edited image
```

The evaluator starts one process per GPU (4 GPUs), constructs `SessionEditor` once per process
and runs that GPU's share of the sessions one after another: `reset(source)` with the source as
an RGB image at its original size (sizes vary a lot; many are small), then `step(instruction)`
for turns 1 to 5. The object may keep anything across the turns of a session (the source, earlier
images and instructions); `reset` starts the next session. `step` returns a PIL image, saved as
an RGB PNG, at least 16 px a side and with an aspect ratio within 5% of the source's. Every turn
is judged at the backbone's output size for the source: W × H with W = sqrt(1024² · a) and
H = W / a, both rounded to multiples of 32 (a = source width / height; about 1 MP). An image more
than 32 px off W or H is resized (Lanczos) to W × H first; return that size to avoid resampling.
A turn fails if `step` raises, returns no image, changes the aspect ratio by more than 5% or does
not finish within the time budget; a failed turn and the later turns of its session score 0. A
process that dies is started again (twice at most) for the next session, within the same budget.
During evaluation the workspace is read-only; temporary files go to the working directory.

## Rules

- The editing backbone is fixed: **Qwen-Image-Edit** (`Qwen/Qwen-Image-Edit`, the August 2025
  release). You may change everything around it: prompt rewriting, sampler settings, step
  count, guidance, seeds, the LoRAs listed below, best-of-N with self-verification, masking and
  compositing, using the session's earlier images (the source included), multi-pass editing,
  pre- and post-processing and image operations with the listed packages. Do not train or
  fine-tune the backbone.
- Models available offline (read-only Hugging Face cache,
  `cache_dir=os.environ["EDIT_MULTITURN_MODELS"] + "/hub"`): `Qwen/Qwen-Image-Edit`,
  `lightx2v/Qwen-Image-Lightning` (its Qwen-Image-Edit LoRAs: 4 and 8 steps, bf16) and
  `Qwen/Qwen2.5-VL-7B-Instruct` (usable for instruction rewriting, planning, grounding or
  checking candidate images). Nothing else may be downloaded or called over the network;
  evaluation runs with `HF_HUB_OFFLINE=1`.
- The sessions come from MSE-Bench, a public benchmark. Do not download, search for or use
  MSE-Bench, its published results (MSE-Bench-results), VINCIE's outputs or any other copy of its
  images, instructions or annotations from anywhere (Hugging Face, GitHub, the project site,
  papers, other copies), and do not look for the validation or test data. Use only the dev data,
  your own work and the models and packages listed here.
- Time budget per GPU: 300 s for start-up plus on average 60 s per turn of that GPU's sessions,
  pooled over them. When it runs out the GPU's process is stopped, and its unfinished turns score
  0. The starter takes about 9 s per turn. Each process has one GPU of at least 80 GB to itself;
  the backbone in bf16 takes about 59 GB of it.
- Available packages: PyTorch, diffusers, transformers, peft, accelerate, vLLM, PIL, numpy, scipy,
  scikit-image, scikit-learn, opencv.

## Scoring

Every turn is scored by a frozen vision-language judge (a different model family from the ones
you can use) with two requests:

- **PF** prompt following and **CONS** consistency (0-10), from MSE-Bench's own evaluation prompt.
  For turn k the judge gets the instructions of turns 1 to k and the images [source, turn 1, ...,
  turn k], and rates turn k: PF, whether its image does what its instruction asks; CONS, whether
  the parts the instruction leaves alone are kept as in the previous image (10 = a minimal yet
  effective edit, 0 = a different scene).
- **PQ** perceptual quality (0-10) of turn k's image alone, judged strictly: noise, colour
  speckles, blocky or tiled patterns, banding, over-saturated or shifted colours,
  over-sharpening, halos, blur, smeared or plastic texture, malformed objects, faces, hands or
  text, seams and inconsistent lighting all count.

The judge sees each turn's image at that size (and the source resized to the size of the turn-1
image), and nothing else about the session.

- Turn score = sqrt(PF*PQ)/10 * (0.5 + 0.5*CONS/10), 0 if PF < 3. Session score = mean of its
  five turn scores. `score` = mean over sessions.
- Also reported: `success_t1` ... `success_t5`, MSE-Bench's cumulative success rate (a session
  succeeds at turn k when every turn up to k has PF > 6 and CONS > 6), `score_t1` ... `score_t5`,
  `pf`, `cons`, `pq`, `failed_turns` and `edit_s_per_turn` (mean seconds per finished turn).

The dev split (30 sessions) is in the dev data directory (`$RSIB_DEV_DATA`): `sessions.jsonl`
(id, image, width, height and the five instructions), `images/`, and `labels.jsonl` with
MSE-Bench's annotations, for your analysis only (per turn the operation type and a description of
the intended result, plus a description of the source and a summary). Your solution never
receives these files: the evaluator hands it the source image and one instruction at a time.
Dev evaluations return every turn's PF, CONS and PQ with the judge's reasons, errors and timings,
means per operation type, and a contact sheet per session (the source and its five turns). On val
and test you only see `score`, `n` and `success_t1` ... `success_t5`.
