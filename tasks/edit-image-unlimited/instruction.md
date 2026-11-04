# Instruction-based image editing (unlimited: any model, any tool)

Improve an image-editing system. Each sample is a source image plus a natural-language editing
instruction (English or Chinese), e.g. "Remove all the seagulls", "把路牌上的英文改成 Maple St",
"Show what these candles look like after burning for three hours". Your system must return the
edited image.

`editor.py` must define

```python
GPUS_PER_WORKER = 1   # optional module-level integer, default 1
class Editor:
    def __init__(self, device: str = "cuda"): ...
    def edit(self, image_path: str, instruction: str, out_path: str, meta: dict) -> None: ...
```

`edit` writes the edited image (PNG, RGB) to `out_path`. `meta` holds `id` and `lang`
(`"en"` or `"zh"`). The evaluator has 4 GPUs (80 GB each). It starts one worker process per group
of `GPUS_PER_WORKER` GPUs (1: four workers with one GPU each; 2: two workers with two GPUs; 4: one
worker that sees all four as `cuda:0`..`cuda:3`), constructs `Editor` once per worker and calls
`edit` for that worker's share of the samples. A sample whose `edit` raises or writes nothing
scores 0. Resolution rule: input images have at most 1536 × 1024 pixels (1.57 MP, aspect ratio
preserved, width and height multiples of 16). Outputs are scored at the input's resolution: an
output of another size is resized (Lanczos) to the input's exact size before any check, or, for
outpainting instructions, keeps its own aspect ratio and is resized to the input's pixel count
(the aspect ratio of outpainting outputs is checked). Return the input size to avoid resampling.

## Rules

- **Any editing model, any tool.** You choose the system: any open-weight image-editing or
  generation model, any number of models, vision-language models for planning, grounding or
  verification, segmentation/detection/inpainting/super-resolution/matting models, classical image
  processing, any open-source Python package, LoRAs, your own training or fine-tuning on the dev
  split. Combine them however you like (routing by edit type, best-of-N with self-verification,
  masking and compositing, multi-pass editing, ...). The starter uses Qwen-Image-Edit-2509 with
  an 8-step Lightning LoRA only because it is a reasonable baseline; replace it if you find
  something better.
- Exceptions: **no model of the judge's family** (Zhipu/zai-org GLM vision models such as
  GLM-4.5V/GLM-4.6V, in any form), **no remote services** (hosted APIs, web requests) when the
  solution runs, and nothing that reads labels or val/test data.
- **Getting models and packages.** Your phases and experiments have internet access to the
  Hugging Face Hub and PyPI (no Hugging Face token: gated repositories cannot be downloaded).
  Download weights into `$RSIB_CACHE` (`HF_HOME` already points there, so
  `from_pretrained("org/name")` caches them there) and install extra packages with
  `pip install --target $RSIB_CACHE/pylib ...`, then put that directory on `sys.path` at the top of
  `editor.py`. Download with `run_experiment` (it has a longer time limit than a shell command) and
  mind the run's wall-clock budget: a 50 GB model takes minutes, a 170 GB one much longer.
- At evaluation the solution sees the workspace, `$RSIB_CACHE` (read-only, with `HF_HOME`,
  `TORCH_HOME` and `RSIB_CACHE` set as in the phases) and two read-only Hugging Face caches,
  and runs **offline** (`HF_HUB_OFFLINE=1`):
  (the evaluator sets `EDIT_IMAGE_MODELS` and `EDIT_IMAGE_UNLIMITED_MODELS`; phases and experiments do
  not, so fall back to the paths given here)
  - `os.environ["EDIT_IMAGE_MODELS"] + "/hub"` (`/models/edit-image/hf_models/hub`): `Qwen/Qwen-Image-Edit-2509`,
    `lightx2v/Qwen-Image-Lightning`, `Qwen/Qwen2.5-VL-7B-Instruct`;
  - `os.environ["EDIT_IMAGE_UNLIMITED_MODELS"] + "/hub"` (`/models/edit-image/hf_open/hub`, pre-staged so you need not spend budget on
    downloading them): `Qwen/Qwen-Image-2.1` (unified generation + editing, top open-weight model
    on the Artificial Analysis editing leaderboard as of 2026-10), `Qwen/Qwen-Image-Edit-2511`,
    `lightx2v/Qwen-Image-Edit-2511-Lightning` (4/8-step bf16 LoRAs), and
    `tencent/HunyuanImage-3.0-Instruct` (80B MoE, needs several GPUs).
  Load from these with `cache_dir=...` or by pointing `HF_HUB_CACHE` at them. A model that is in
  none of these places cannot be loaded at evaluation. `Qwen-Image-2.1` needs a newer diffusers
  (`QwenImage21Pipeline`) than the installed 0.41: one built from diffusers' main branch is in
  `os.environ["EDIT_IMAGE_UNLIMITED_MODELS"] + "/pylib"`; put it first on `sys.path` before importing
  diffusers. HunyuanImage-3.0 loads with `trust_remote_code=True` (see its model card) from its
  snapshot directory (the repo id contains a dot), but its code needs transformers 4.x: a
  compatible set (transformers 4.57.1, tokenizers 0.22.1, huggingface_hub 0.36.0, diffusers
  0.35.2) is in `os.environ["EDIT_IMAGE_UNLIMITED_MODELS"] + "/pylib_hunyuan"`; put it first on
  `sys.path` in the process that runs HunyuanImage (it cannot share a process with the main-branch
  diffusers of `pylib/`), and set `model.config.model_version = None` before
  `model.load_tokenizer(path)` (the attribute is missing from its config). Measured on 4 GPUs:
  about 40 s to load (about 41 GB per GPU), about 140 s per edit with `bot_task="think_recaption"`
  and 50 diffusion steps.
- Time budget: on average 150 s per sample per GPU, plus 600 s for start-up; the whole call is
  killed when the budget runs out (samples not finished score 0). With `GPUS_PER_WORKER = 4` a
  worker therefore has about 4 x 150 s per sample.
- Installed packages: PyTorch, diffusers, transformers, vLLM, PIL, numpy, scipy, scikit-image,
  opencv (plus whatever you install into `$RSIB_CACHE`).

## Scoring

Each output is scored by a frozen vision-language judge (a model family you may not use) plus
programmatic checks:

- **SC** instruction compliance (0-10), **PR** preservation of what should be kept (0-10, judged
  down to fine texture: smoothed, blurred or redrawn detail counts as a change), **PQ**
  perceptual quality of the output alone (0-10; softness, upscaling look, waxy texture and
  artifacts count). The judge is told to be strict and sees images at up to 1536 px.
- **Verification protocol by edit type.** What "should be kept" depends on the edit, and so do
  the measurements:

  | protocol | sub-categories | judge PR is about | geometry | preserve | detail |
  |---|---|---|---|---|---|
  | local | A1-A4, A6, C1-C3, E3, G2 | everything outside the edited object/region, down to texture | yes | outside region | outside region |
  | motion | A5, A7 | everything but the subject's pose/position | with mask | with mask | with mask |
  | style | B1 | content, shapes, positions, composition (style, texture, colours are meant to change) | no | with mask | with mask |
  | background | B2 | the kept subjects: detail, size, position | yes | no | no |
  | environment | B3 | objects, layout, composition (light/weather/season are meant to change) | yes | with mask | with mask |
  | tone | E1 | content and fine detail (tone is meant to change) | yes | no | frame median |
  | restore | E2 | content not invented or redrawn beyond the repair | yes | no | no |
  | reframe | E4 | the original picture inside the new canvas | yes (beyond scale+offset) | source frame | source frame |
  | reasoning, compound | F1-F4, G1 | everything the intended result does not require to change | yes | with mask | with mask |

  The edit region is the sample's edit mask; for unmasked local edits the judge grounds the
  objects that appear, disappear or are edited (with their shadows, reflections, steam...) and
  the boxes, padded, form the region. Pixel checks are skipped when the region leaves less than
  15% of the frame; an unmasked local edit whose objects cannot be grounded gets only the frame
  median of detail.
- **Global deformation** is measured, not judged: a SIFT + RANSAC homography H is fitted from
  input to output (keypoints outside the edit region, the whole frame if too few remain) and its
  displacement |H(x) - x| is taken over the area covered by the matched keypoints. Its 90th
  percentile p90, in px at a 512-px short side, gives g = exp(-max(0, p90 - 0.5) / 6) (1 px ->
  0.92, 5 px -> 0.47, 10 px -> 0.21), which is deducted from the judge's preservation:
  PR_eff = max(0, PR - 5 * (1 - g)).
- Judge score = sqrt(SC*PQ)/10 * (0.5 + 0.5*PR_eff/10).
- Programmatic checks where applicable: object presence/absence and object counts (VQA), OCR of
  requested text (character error rate) and of text that must disappear, output aspect ratio
  for outpainting, **preserve** (the output is aligned back onto the input with H and the mean
  pixel difference outside the edit region is scored) and **detail** (the aligned output is
  compared with the input at a 1024-px short side by local contrast and structure, the SSIM
  contrast-structure term, so smeared, blurred or redrawn fine texture is penalised).
- Sample score = 0.5 * judge + 0.5 * programmatic (judge only if no check applies); 0 if SC < 3.
- `overall` = mean over the six edit families (A local semantic edits, B global edits/style,
  C text editing, E photo retouching/restoration, F reasoning-based edits, G compositional and
  constrained instructions), so each family weighs the same.

The dev split (`samples.jsonl`, images, edit masks and check targets) is in the dev data
directory; dev evaluations also return per-sample judge reasons and side-by-side comparison
images. On val and test you only see the aggregate score.
