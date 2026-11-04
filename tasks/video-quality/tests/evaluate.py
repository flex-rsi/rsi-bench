"""Evaluator: judge the split's items in a child process that sees the videos, their frames, the
items and a budget-limited gateway of the model under test, never the labels; score here.

score = (SRCC of the rated videos' scores against their MOS + accuracy of the pairwise judgments) / 2.
A rated video without a usable score gets the median of the split's other scores; a judgment the
source calls a tie is left out, a missing judgment counts as wrong.

The frame budget: the child's RSIB_GATEWAY_URL is a proxy in this process. It forwards a model call
to the run's gateway only if the images of the call hold at most MAX_PIXELS pixels together (8
frames of 1024 x 576), and refuses it otherwise (HTTP 400, error "frame_budget"); it also refuses
every call after the split's first CALLS_PER_ITEM x n (error "call_budget"). The child never sees
the gateway's own token, so every call it makes passes through the proxy. The split's videos and
frames are mounted read-only into the child's sandbox; its labels are not."""

import base64
import io
import json
import math
import os
import secrets
import statistics
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

TIME_LIMIT_S = 3600              # the whole split
MAX_PIXELS = 8 * 1024 * 576      # all images of one model call together
CALLS_PER_ITEM = 3               # on average: a split of n items may make 3n model calls
FRAMES_PER_SECOND = 2
CRITERIA = ("alignment", "artifacts", "preference")


def solution_cmd(python, io_dir, readable):
    """The solver's process. When the orchestrator gives a sandbox launcher ($RSIB_SANDBOX), the
    solver runs through it and sees its own code, the run cache, io_dir and the split's videos and
    frames (read-only), never the labels."""
    child = Path(io_dir) / "_child_answer.py"
    child.write_text(Path(__file__).with_name("child_answer.py").read_text())
    cmd = [python, str(child)]
    prefix = json.loads(os.environ.get("RSIB_SANDBOX") or "[]")
    if not prefix:
        return cmd
    ro = [a for p in readable for a in ("--ro", str(p))]
    return prefix + ro + ["--rw", str(io_dir), "--cwd", str(io_dir), "--"] + cmd


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def image_pixels(payload: dict) -> int:
    """Pixels of every image part in the messages of a /v1/complete request."""
    total = 0
    for msg in payload.get("messages") or []:
        contents = [msg.get("content")] if msg.get("role") == "user" else \
            [r.get("content") for r in msg.get("results") or []]
        for content in contents:
            for part in content if isinstance(content, list) else []:
                if isinstance(part, dict) and part.get("type") == "image":
                    w, h = Image.open(io.BytesIO(base64.b64decode(part["data"]))).size
                    total += w * h
    return total


class Proxy:
    """The child's gateway: checks the frame budget, then forwards with the evaluation's token."""

    def __init__(self, upstream: str, token: str, max_calls: int):
        self.upstream, self.token, self.key = upstream.rstrip("/"), token, secrets.token_hex(16)
        self.max_calls = max_calls
        self.calls = self.refused = 0
        self.lock = threading.Lock()
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def reply(self, status: int, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):  # noqa: N802
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if self.headers.get("Authorization") != f"Bearer {proxy.key}":
                    return self.reply(401, b'{"error": "unauthorized"}')
                if self.path == "/v1/complete":
                    try:
                        pixels = image_pixels(json.loads(raw or b"{}"))
                    except Exception as exc:  # unreadable image: the gateway would refuse it too
                        return self.reply(400, json.dumps({"error": "bad_request",
                                                           "message": str(exc)[:300]}).encode())
                    if pixels > MAX_PIXELS:
                        with proxy.lock:
                            proxy.refused += 1
                        return self.reply(400, json.dumps({
                            "error": "frame_budget", "message": f"the images of one call may hold "
                            f"at most {MAX_PIXELS} pixels together; this call has {pixels}"}).encode())
                    with proxy.lock:
                        over = proxy.calls >= proxy.max_calls
                        if over:
                            proxy.refused += 1
                        else:
                            proxy.calls += 1
                    if over:
                        return self.reply(400, json.dumps({
                            "error": "call_budget", "message": f"this split may make at most "
                            f"{proxy.max_calls} model calls ({CALLS_PER_ITEM} per item)"}).encode())
                req = urllib.request.Request(proxy.upstream + self.path, data=raw, method="POST",
                                             headers={"Content-Type": "application/json",
                                                      "Authorization": f"Bearer {proxy.token}"})
                try:
                    with urllib.request.urlopen(req, timeout=3600) as resp:
                        return self.reply(resp.status, resp.read())
                except urllib.error.HTTPError as exc:
                    return self.reply(exc.code, exc.read())
                except Exception as exc:
                    return self.reply(502, json.dumps({"error": "unreachable",
                                                       "message": str(exc)[:300]}).encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"


def ranks(values: list[float]) -> list[float]:
    """Ranks from 1, ties sharing the mean of their positions."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return 0.0
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx, syy = sum((a - mx) ** 2 for a in x), sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else 0.0  # constant scores: no order


def number(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


split_dir = Path(os.environ["RSIB_DATA_DIR"])
workspace = Path(os.environ["RSIB_WORKSPACE"])
out_dir = Path(os.environ["RSIB_OUTPUT_DIR"])
python = os.environ.get("RSIB_PYTHON", sys.executable)
runtime = workspace / "runtime.txt"  # optional: another interpreter (inside $RSIB_CACHE when sandboxed)
if runtime.exists() and runtime.read_text().strip():
    python = runtime.read_text().strip().splitlines()[0].strip()

items = read_jsonl(split_dir / "items.jsonl")
videos = {v["video"]: v for v in read_jsonl(split_dir / "videos.jsonl")}
max_calls = CALLS_PER_ITEM * len(items)
proxy = None
if os.environ.get("RSIB_GATEWAY_URL"):
    proxy = Proxy(os.environ["RSIB_GATEWAY_URL"], os.environ["RSIB_GATEWAY_TOKEN"], max_calls)
frames, video_dir = split_dir / "frames", split_dir / "videos"


def clip(vid: str) -> dict:
    v = videos[vid]
    return {"video": vid, "path": str(video_dir / f"{vid}.mp4"), "frames": str(frames / vid),
            "frames_per_second": FRAMES_PER_SECOND, "duration_s": v["duration_s"],
            "width": v["width"], "height": v["height"], "fps": v["fps"]}


with tempfile.TemporaryDirectory(prefix="video-quality-", dir=out_dir) as tmp:
    io_dir = Path(tmp)
    full = [{**it, **clip(it["video"])} if it["kind"] == "rate" else
            {**it, "videos": [clip(v) for v in it["videos"]]} for it in items]
    (io_dir / "items.jsonl").write_text("".join(json.dumps(it) + "\n" for it in full))
    env = {k: v for k, v in os.environ.items() if not k.startswith("RSIB_")}
    env.update({"RSIB_IO_DIR": tmp, "RSIB_SOLVER_DIR": str(workspace),
                "VIDEO_QUALITY_MAX_PIXELS": str(MAX_PIXELS), "VIDEO_QUALITY_MAX_CALLS": str(max_calls),
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"})
    if proxy:
        env.update({"RSIB_GATEWAY_URL": proxy.url, "RSIB_GATEWAY_TOKEN": proxy.key})
    try:
        proc = subprocess.run(solution_cmd(python, io_dir, [frames, video_dir]), env=env, cwd=tmp,
                              capture_output=True, text=True, timeout=TIME_LIMIT_S)
    except subprocess.TimeoutExpired:
        sys.exit(f"the solver exceeded {TIME_LIMIT_S}s on this split")
    print(proc.stdout[-4000:])
    if proc.stderr.strip():  # tracebacks of items that failed, also when the others did not
        print("[solver stderr]\n" + proc.stderr[-6000:])
    if proc.returncode != 0:
        sys.exit("solver failed")
    predictions = json.loads((io_dir / "predictions.json").read_text())

labels = {row["id"]: row for row in read_jsonl(split_dir / "labels.jsonl")}
rated = [it for it in items if it["kind"] == "rate"]
given = {it["id"]: number(predictions.get(it["id"])) for it in rated}
known = [v for v in given.values() if v is not None]
fill = statistics.median(known) if known else 0.0
scores = [given[it["id"]] if given[it["id"]] is not None else fill for it in rated]
mos = [labels[it["id"]]["mos"] for it in rated]
srcc = pearson(ranks(scores), ranks(mos))
plcc = pearson(scores, mos)

results, judgments = [], []
for it in items:
    lab, pred = labels[it["id"]], predictions.get(it["id"])
    if it["kind"] == "rate":
        results.append({"id": it["id"], "kind": "rate", "video": it["video"], "category": it["category"],
                        "score": given[it["id"]], "mos": lab["mos"]})
        continue
    pred = pred if isinstance(pred, dict) else {}
    row = {"id": it["id"], "kind": "compare", "videos": it["videos"]}
    for c in CRITERIA:
        p = str(pred.get(c) or "").strip().upper()[:1] or None
        row[c] = {"answer": p, "label": lab[c], "correct": None if lab[c] == "tie" else p == lab[c]}
        if lab[c] != "tie":
            judgments.append((c, p == lab[c]))
    results.append(row)


def accuracy(rows: list[bool]) -> float:
    return sum(rows) / len(rows) if rows else 0.0


pair_accuracy = accuracy([ok for _, ok in judgments])
reward = {"score": (srcc + pair_accuracy) / 2, "srcc": srcc, "plcc": plcc, "pair_accuracy": pair_accuracy,
          **{f"acc_{c}": accuracy([ok for k, ok in judgments if k == c]) for c in CRITERIA},
          "n_videos": len(rated), "n_pairs": len(items) - len(rated), "n_judgments": len(judgments),
          "answered": len(known) + sum(isinstance(predictions.get(it["id"]), dict)
                                       for it in items if it["kind"] == "compare"),
          "model_calls": proxy.calls if proxy else 0,
          "refused_calls": proxy.refused if proxy else 0,
          "artifacts": [{"path": "per_item.json", "role": "diagnosis",
                         "note": "score and MOS of every rated video; answers and labels of every pair"}]}
(out_dir / "per_item.json").write_text(json.dumps(results, indent=1))
(out_dir / "reward.json").write_text(json.dumps(reward))
print(json.dumps({k: v for k, v in reward.items() if k != "artifacts"}))
