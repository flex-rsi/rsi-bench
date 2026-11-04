"""Starter: four evenly spaced frames, the question and the prior, one backbone call."""

import cv2

from model import ask, image_part

WORKERS = 8


def frames(path: str, k: int = 4) -> list:
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in range(k):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * (n - 1) / max(1, k - 1)))
        ok, frame = cap.read()
        if ok:
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def answer(item: dict) -> str:
    text = f"{item['question']}\nKnown: {item['prior']}."
    if item.get("depth_info"):
        text += f"\nDepth: {item['depth_info']}."
    text += "\nAnswer with a single number only."
    return ask([*(image_part(f) for f in frames(item["video"])), {"type": "text", "text": text}],
               max_tokens=64)
