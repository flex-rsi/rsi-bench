"""Evaluator: answer the split's questions in a child process that sees the images, the questions
and a view-limited gateway of the model under test, never the answers; score accuracy here.

The field of view: the child's RSIB_GATEWAY_URL is a proxy in this process. It forwards a model
call to the run's gateway only if the images of the call hold at most MAX_PIXELS pixels together,
and refuses it otherwise (HTTP 400, error "field_of_view"); it also refuses every call after the
split's first CALLS_PER_QUESTION x n (error "call_budget"). The child never sees the gateway's own
token, so every call it makes passes through the proxy."""

import base64
import io
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

TIME_LIMIT_S = 2700   # the whole split
MAX_PIXELS = 1024 * 1024  # all images of one model call together
CALLS_PER_QUESTION = 5    # on average: a split of n questions may make 5n model calls


def solution_cmd(python, io_dir):
    """The solver's process. When the orchestrator gives a sandbox launcher ($RSIB_SANDBOX), the
    solver runs through it and sees its own code, the run cache and io_dir, never the labels."""
    child = Path(io_dir) / "_child_answer.py"
    shutil.copy(Path(__file__).with_name("child_answer.py"), child)
    cmd = [python, str(child)]
    prefix = json.loads(os.environ.get("RSIB_SANDBOX") or "[]")
    return prefix + ["--rw", str(io_dir), "--cwd", str(io_dir), "--"] + cmd if prefix else cmd


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
    """The child's gateway: checks the field of view, then forwards with the evaluation's token."""

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
                            "error": "field_of_view", "message": f"the images of one call may hold "
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
                            f"{proxy.max_calls} model calls ({CALLS_PER_QUESTION} per question)"}).encode())
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


split_dir = Path(os.environ["RSIB_DATA_DIR"])
workspace = Path(os.environ["RSIB_WORKSPACE"])
out_dir = Path(os.environ["RSIB_OUTPUT_DIR"])
python = os.environ.get("RSIB_PYTHON", sys.executable)
runtime = workspace / "runtime.txt"  # optional: another interpreter (inside $RSIB_CACHE when sandboxed)
if runtime.exists() and runtime.read_text().strip():
    python = runtime.read_text().strip().splitlines()[0].strip()

questions = read_jsonl(split_dir / "questions.jsonl")
max_calls = CALLS_PER_QUESTION * len(questions)
proxy = None
if os.environ.get("RSIB_GATEWAY_URL"):
    proxy = Proxy(os.environ["RSIB_GATEWAY_URL"], os.environ["RSIB_GATEWAY_TOKEN"], max_calls)
with tempfile.TemporaryDirectory(prefix="vsearch-", dir=out_dir) as tmp:
    io_dir = Path(tmp)
    shutil.copytree(split_dir / "images", io_dir / "images")
    items = [{**q, "image": str(io_dir / q["image"])} for q in questions]
    (io_dir / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in items))
    env = {k: v for k, v in os.environ.items() if not k.startswith("RSIB_")}
    env.update({"RSIB_IO_DIR": tmp, "RSIB_SOLVER_DIR": str(workspace),
                "VSEARCH_MAX_PIXELS": str(MAX_PIXELS), "VSEARCH_MAX_CALLS": str(max_calls),
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"})
    if proxy:
        env.update({"RSIB_GATEWAY_URL": proxy.url, "RSIB_GATEWAY_TOKEN": proxy.key})
    try:
        proc = subprocess.run(solution_cmd(python, io_dir), env=env, cwd=tmp, capture_output=True,
                              text=True, timeout=TIME_LIMIT_S)
    except subprocess.TimeoutExpired:
        sys.exit(f"the solver exceeded {TIME_LIMIT_S}s on this split")
    print(proc.stdout[-4000:])
    if proc.stderr.strip():  # tracebacks of questions that failed, also when the others did not
        print("[solver stderr]\n" + proc.stderr[-6000:])
    if proc.returncode != 0:
        sys.exit("solver failed")
    predictions = json.loads((io_dir / "predictions.json").read_text())

labels = {row["id"]: row["answer"] for row in read_jsonl(split_dir / "labels.jsonl")}
results = [{"id": q["id"], "source": q["source"], "category": q["category"],
            "answer": predictions.get(q["id"]), "label": labels[q["id"]],
            "correct": predictions.get(q["id"]) == labels[q["id"]]} for q in questions]


def accuracy(rows: list[dict]) -> float:
    return sum(r["correct"] for r in rows) / len(rows) if rows else 0.0


reward = {"accuracy": accuracy(results),
          "acc_vstar": accuracy([r for r in results if r["source"] == "vstar"]),
          "acc_hrbench8k": accuracy([r for r in results if r["source"] == "hrbench8k"]),
          "n": len(results), "answered": sum(r["answer"] is not None for r in results),
          "model_calls": proxy.calls if proxy else 0,
          "refused_calls": proxy.refused if proxy else 0,
          "artifacts": [{"path": "per_question.json", "role": "diagnosis",
                         "note": "answer, label and correctness of every question"}]}
(out_dir / "per_question.json").write_text(json.dumps(results, indent=1))
(out_dir / "reward.json").write_text(json.dumps(reward))
print(json.dumps({k: v for k, v in reward.items() if k != "artifacts"}))
