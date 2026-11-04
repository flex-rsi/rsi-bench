"""Starter solution: the direct "look and judge" pipeline, one model call per item.

rate: 8 frames evenly spaced over the video, shrunk to fit the frame budget together; the model
gives a quality score from 1 (bad) to 5 (excellent).
compare: 6 frames of each clip, the same way; the model answers the three questions with A or B."""

import os
import re

from model import MAX_PIXELS, complete, image_part

WORKERS = 8           # items judged at once
RATE_FRAMES = 8
COMPARE_FRAMES = 6    # per clip


def evenly(clip: dict, n: int) -> list[tuple[float, str]]:
    """n frames evenly spaced over the clip: (seconds, path)."""
    files = sorted(os.listdir(clip["frames"]))
    picks = sorted({round(i * (len(files) - 1) / max(1, n - 1)) for i in range(n)})
    fps = clip["frames_per_second"]
    return [(int(files[i][:-4]) / fps, os.path.join(clip["frames"], files[i])) for i in picks]


def rate(item: dict) -> float | None:
    per_image = MAX_PIXELS // RATE_FRAMES
    content = []
    for t, path in evenly(item, RATE_FRAMES):
        content += [{"type": "text", "text": f"[{t:.1f} s]"}, image_part(path, per_image)]
    content.append({"type": "text", "text":
                    f"These are frames, in order, of one video ({item['width']} x {item['height']}, "
                    f"{item['duration_s']:.1f} s). Rate the video's overall perceptual quality as a "
                    "viewer would: 1 = bad, 2 = poor, 3 = fair, 4 = good, 5 = excellent. "
                    "Answer with one number from 1 to 5 (decimals allowed)."})
    m = re.search(r"\d+(?:\.\d+)?", complete(content))
    return float(m.group()) if m else None


def compare(item: dict) -> dict:
    per_image = MAX_PIXELS // (2 * COMPARE_FRAMES)
    content = [{"type": "text", "text": "Two videos, A and B, were generated from the same text "
                f"prompt:\n\"{item['prompt']}\""}]
    for name, clip in zip("AB", item["videos"]):
        for t, path in evenly(clip, COMPARE_FRAMES):
            content += [{"type": "text", "text": f"[video {name}, {t:.1f} s]"}, image_part(path, per_image)]
    content.append({"type": "text", "text":
                    "Answer three questions:\n"
                    "alignment: which video matches the text prompt better?\n"
                    "artifacts: which video has fewer visual artifacts and glitches (looks less "
                    "AI-generated)?\n"
                    "preference: which video looks better overall?\n"
                    "Reply with exactly three lines: 'alignment: A' or 'alignment: B', then "
                    "'artifacts: ...', then 'preference: ...'."})
    reply = complete(content)
    out = {}
    for c in ("alignment", "artifacts", "preference"):
        m = re.search(rf"{c}\W*(?:video\s*)?([AB])\b", reply, re.I)
        out[c] = m.group(1).upper() if m else None
    return out
