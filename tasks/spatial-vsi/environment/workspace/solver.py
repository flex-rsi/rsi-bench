"""Starter: the direct pipeline. For each question, 16 frames sampled evenly from the video and
the official VSI-Bench prompt go to the model in one call; the reply is the answer."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

cv2.setNumThreads(1)  # OpenCV's own pool: one per call is enough, the workers run in parallel

FRAMES = 16
WIDTH = 512  # frames are downscaled to this width before encoding
WORKERS = 8
MAX_TOKENS = 4000  # the model thinks before it answers: 64 left most replies empty
MCA = {"object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard",
       "object_rel_distance", "route_planning", "obj_appearance_order"}


def frames(path: str, n: int = FRAMES) -> list[bytes]:
    # one decode thread: under a process limit ffmpeg's threaded scaler fails and returns black
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((i + 0.5) * total / n))
        ok, img = cap.read()
        if not ok or img.mean() < 2:  # unreadable or black: a decode failure, not content
            print(f"frame {i} of {path}: decode failed", flush=True)
            continue
        h, w = img.shape[:2]
        img = cv2.resize(img, (WIDTH, round(h * WIDTH / w)))
        out.append(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
    cap.release()
    return out


def prompt(q: dict) -> str:
    """The official lmms-eval prompt of VSI-Bench."""
    pre = "These are frames of a video."
    if q["question_type"] in MCA:
        return "\n".join([pre, q["question"], "Options:\n" + "\n".join(q["options"]),
                          "Answer with the option's letter from the given choices directly."])
    return pre + "\n" + q["question"] + "\nPlease answer the question using a single word or phrase."


def complete(content: list, session: str) -> str:
    url = os.environ["RSIB_GATEWAY_URL"].rstrip("/") + "/v1/complete"
    body = {"system": "You answer questions about indoor videos. Reply with the answer only: "
                      "no markdown, no explanation.",
            "messages": [{"role": "user", "content": content}], "tools": [],
            "max_tokens": MAX_TOKENS, "session": session}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('RSIB_GATEWAY_TOKEN', '')}"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())["text"].replace("*", "").strip()


def answer_one(q: dict) -> str:
    content = [{"type": "image", "mime": "image/jpeg", "data": base64.b64encode(f).decode()}
               for f in frames(q["video"])]
    content.append({"type": "text", "text": prompt(q)})
    try:
        return complete(content, session=f"q{q['id']}")
    except (urllib.error.URLError, KeyError, TimeoutError) as exc:
        print(f"q{q['id']}: {exc}", flush=True)
        return ""


def answer_all(questions: list[dict], io_dir: Path) -> dict:
    """{question id: answer string}. io_dir is writable; files under io_dir/trace/ come back with
    dev results."""
    with ThreadPoolExecutor(WORKERS) as pool:
        replies = list(pool.map(answer_one, questions))
    return {q["id"]: r for q, r in zip(questions, replies)}
