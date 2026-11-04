"""Starter solution: the direct "look and answer" pipeline. Show the model 8 frames of each view,
evenly spaced over the video, shrunk so that all 16 fit the frame budget, each labeled with its view
and time; then the question's prompt; read the option letter it answers with. One call per question."""

import os
import re

from model import MAX_PIXELS, complete, image_part

WORKERS = 8      # questions answered at once
PER_VIEW = 8     # frames per view
VIEWS = (("ego", "Ego view (the camera wearer)"), ("exo", "Exo view (a fixed camera)"))


def evenly(frame_dir: str, n: int) -> list[tuple[int, str]]:
    """n frames evenly spaced over the video: (second, path). Frame k is at k seconds."""
    files = sorted(os.listdir(frame_dir))
    picks = sorted({round(i * (len(files) - 1) / max(1, n - 1)) for i in range(n)})
    return [(int(files[i][:-4]), os.path.join(frame_dir, files[i])) for i in picks]


def answer(item: dict) -> str | None:
    per_image = MAX_PIXELS // (len(VIEWS) * PER_VIEW)
    content = []
    for view, name in VIEWS:
        for t, path in evenly(item[f"{view}_frames"], PER_VIEW):
            content += [{"type": "text", "text": f"[{name}, {t} s]"}, image_part(path, per_image)]
    content.append({"type": "text", "text": "The two views are synchronized recordings of the same "
                    "activity.\n\n" + item["prompt"]})
    reply = complete(content)
    m = re.search(r"\b([A-D])\b", reply)
    return m.group(1) if m else None
