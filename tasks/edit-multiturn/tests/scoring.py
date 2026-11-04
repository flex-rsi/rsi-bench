"""Judge prompts, answer parsing and scores of edit-multiturn. Part of the task version: do not
change between runs.

For turn k of a session the judge answers two requests:
  mse  MSE-Bench's own evaluation prompt (VINCIE) with the instructions of turns 1..k and the images
       [source, turn 1, ..., turn k]: prompt_following (PF) and consistency (CONS) of turn k, 0-10
  pq   the turn-k image alone: perceptual quality (PQ), 0-10
"""

import ast
import json
import math
import re

from PIL import Image

# Verbatim from VINCIE (github.com/ByteDance-Seed/VINCIE, evaluation/compute_score.py at commit
# be1c661, Apache-2.0), written line by line so that its trailing spaces survive editors.
MSE_PROMPT = "\n".join([
    " Assume you are an expert in evaluating multi-turn image editing. In this task, a user interacts with an image editing system across multiple turns. At the first turn, the user provides a source image and an editing prompt. The system returns the edited image. In each subsequent turn, the user supplies a new prompt, and the system generates a new image based on the output from the previous turn.",
    "Your goal is to evaluate how successfully the editing instruction of the LAST turn (turn-{num_prompts}) has been executed. ",
    "",
    "You will be given {num_prompts} user editing prompts and {num_images} images: the first image is the original source image, and the next are the edited results from each turn for each prompt.",
    "You should focus more on the last prompt and the last edited image, but you may also consider the previous prompts and images as context. ",
    "",
    "The {num_prompts} user editing prompts are: {editing_prompt}",
    "",
    "Please follow these evaluation rules:",
    "1) Last-turn Evaluation: For the last turn, you should first assess the result based on two criteria by giving a reason: 1) prompt_following, does the last edited image fulfill the last user’s editing prompt? 2) consistency: Are the untouched parts of the last result image consistent with the input reference (the source image at the first turn, or the result image at the previous turn)?",
    '2) Scoring: Based on the reason, you assign scores for "prompt_following" and "consistency". ',
    "From scale 0 to 10: ",
    'A "prompt_following" score from 0 to 10 will be given based on the editing success of prompt following. (0 indicates that the scene in the last edited image does not follow the last editing instruction at all. 10 indicates that the scene in the last edited image follow the last editing instruction perfectly.)',
    'A "consistency" score from 0 to 10 will rate the degree of overediting in the last edited image. (0 indicates that the scene in the last edited image is completely different from the original. 10 indicates that the last edited image can be recognized as a minimal edited yet effective version of original.)',
    "",
    "3) Return your results in a JSON structure, following this format:",
    '{{"reason": "...", "prompt_following": int, "consistency": int}}',
    "",
])
MSE_SUCCESS = 6  # MSE-Bench: a turn succeeds when both PF and CONS are above 6

PQ_PROMPT = """You are a strict judge of the visual quality of an image produced by an AI editing system after
several rounds of editing. Ignore whether it matches any instruction.
Inspect the whole image at full resolution. Penalise the artifacts that build up over repeated
editing: noise, colour speckles or grain, blocky or tiled patterns, banding, over-saturated or
shifted colours, over-sharpening, halos or crunchy texture; also blur, smeared or plastic
textures, warped or malformed objects, hands, faces or text, seams and inconsistent lighting.
Rate PQ (perceptual quality) on an integer scale from 0 to 10:
10 = indistinguishable from a high-quality real photograph (or a well-made artwork in its medium);
8 = minor flaws visible only on close inspection;
6 = noticeable noise, softness, colour shift or small artifacts;
4 = obvious artifacts over large areas;
2 = severe noise, blocky or broken content; 0 = unrecognizable.
For paintings, drawings or old photographs, judge consistency with the medium, not photographic sharpness.
Answer with JSON only: {"reason": "<one or two sentences>", "PQ": <int>}"""

GATE = 3  # a turn whose PF is below this scores 0


def mse_prompt(instructions: list[str]) -> str:
    """The MSE-Bench prompt for the last of `instructions` (turns 1..k), formatted as VINCIE does."""
    editing_prompt = {f"turn{i + 1}": p for i, p in enumerate(instructions)}
    return MSE_PROMPT.format(editing_prompt=str(editing_prompt), num_prompts=len(instructions),
                             num_images=len(instructions) + 1)


def judge_source(source: str, turn1: str, dest: str) -> str:
    """The source image as MSE-Bench shows it to the judge: resized to the size of the turn-1 result."""
    with Image.open(turn1) as t1:
        size = t1.size
    Image.open(source).convert("RGB").resize(size).save(dest)
    return dest


def parse(text: str | None, keys: tuple[str, ...]) -> dict:
    """The scores (0-10, None when missing) and the reason in a judge answer."""
    text = (text or "").replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
    obj = None
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        for load in (json.loads, ast.literal_eval):
            try:
                obj = load(m.group(0))
                break
            except (ValueError, SyntaxError):
                pass
    out = {}
    for k in keys:
        v = obj.get(k) if isinstance(obj, dict) else None
        if v is None:
            mk = re.search(r'"?%s"?\s*[:=]\s*"?(\d+(?:\.\d+)?)' % k, text)
            v = mk.group(1) if mk else None
        try:
            out[k] = min(10.0, max(0.0, float(v))) if v is not None else None
        except (TypeError, ValueError):
            out[k] = None
    out["reason"] = str(obj.get("reason", "")) if isinstance(obj, dict) else text[:300]
    return out


def turn_score(pf, cons, pq) -> float:
    """Score of one turn in [0, 1]."""
    if pf is None or cons is None or pq is None or pf < GATE:
        return 0.0
    return math.sqrt(pf * pq) / 10 * (0.5 + 0.5 * cons / 10)


def cumulative_success(turns: list[dict]) -> list[int]:
    """MSE-Bench success at turns 1..n: 1 while every turn so far has PF and CONS above MSE_SUCCESS."""
    out, ok = [], True
    for t in turns:
        ok = ok and (t.get("pf") or 0) > MSE_SUCCESS and (t.get("cons") or 0) > MSE_SUCCESS
        out.append(int(ok))
    return out
