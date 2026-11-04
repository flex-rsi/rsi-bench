"""Evaluator of spatial-vsi: run the workspace's solver on the split's questions in a child
process that sees the questions and the videos but never the labels, then score its answers with
VSI-Bench's official metric (vision-x-nyu/thinking-in-space, lmms_eval/tasks/vsibench/utils.py):

    multiple choice  exact match of the first word of the answer (trailing "." dropped) with the
                     option letter
    numeric          MRA, mean over thresholds 0.50, 0.55, ... 0.95 of |pred - gt| / gt <= 1 - t
    score            macro mean over the 8 groups (the three relative-direction levels are first
                     averaged into one), in [0, 1]

The child reaches the model through a relay here that forwards to the evaluation's gateway grant
and allows CALLS_PER_QUESTION calls per question of the split (HTTP 402 past that), so a dev
evaluation costs what a val or test one does per question. Sandboxed, the child sees its workspace
snapshot, the run cache and its I/O directory (the questions and video links of this split) only.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ["RSIB_DATA_DIR"])
WORKSPACE = Path(os.environ["RSIB_WORKSPACE"]).resolve()
OUT = Path(os.environ["RSIB_OUTPUT_DIR"]).resolve()
PYTHON = os.environ.get("RSIB_PYTHON", sys.executable)
TIMEOUT_S = 14400
CALLS_PER_QUESTION = 2

MCA = {"object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard",
       "object_rel_distance", "route_planning", "obj_appearance_order"}
NA = {"object_abs_distance", "object_counting", "object_size_estimation", "room_size_estimation"}


def group(qtype: str) -> str:
    return "object_rel_direction" if qtype.startswith("object_rel_direction") else qtype


def first_word(pred) -> str:
    return str(pred).split(" ")[0].rstrip(".").strip()


def mra(pred: float, gt: float) -> float:
    ts = [0.5 + 0.05 * i for i in range(10)]
    return sum(abs(pred - gt) / gt <= 1 - t for t in ts) / len(ts)


def item_score(qtype: str, pred, gt: str) -> float:
    p = first_word(pred)
    if qtype in MCA:
        return float(p.lower() == gt.lower())
    try:
        v = float(p)
    except ValueError:
        return 0.0
    return mra(v, float(gt)) if math.isfinite(v) else 0.0


def overall(rows: list[dict]) -> tuple[float, dict]:
    """The official aggregation: mean per question type, the three direction levels averaged
    into one, then the mean over the 8 groups. Also returned per group."""
    per_type = defaultdict(list)
    for r in rows:
        per_type[r["question_type"]].append(r["score"])
    means = {t: sum(v) / len(v) for t, v in per_type.items()}
    dirs = [means.pop(t) for t in sorted(means) if t.startswith("object_rel_direction")]
    if dirs:
        means["object_rel_direction"] = sum(dirs) / len(dirs)
    return (sum(means.values()) / len(means) if means else 0.0), means


class Relay:
    """Forwards /v1/complete to the evaluation's grant, at most `cap` calls in all."""

    def __init__(self, url: str, token: str, cap: int):
        self.url, self.token, self.cap = url.rstrip("/"), token, cap
        self.calls = self.refused = 0
        self.lock = threading.Lock()
        relay = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                status, body = relay.handle(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.address = f"http://127.0.0.1:{self.server.server_address[1]}"

    def handle(self, data: bytes) -> tuple[int, bytes]:
        with self.lock:
            if self.calls >= self.cap:
                self.refused += 1
                return 402, json.dumps({"error": "split_call_limit", "message":
                                        f"{self.cap} model calls per evaluation of this split"}).encode()
            self.calls += 1
        req = urllib.request.Request(self.url + "/v1/complete", data=data, method="POST", headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.token}"})
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return 502, json.dumps({"error": "unreachable", "message": str(exc)}).encode()


def solver_cmd(io: Path) -> list[str]:
    child = io / "_child_answer.py"
    shutil.copy(HERE / "child_answer.py", child)
    cmd = [PYTHON, str(child)]
    prefix = json.loads(os.environ.get("RSIB_SANDBOX") or "[]")
    return prefix + ["--rw", str(io), "--cwd", str(io), "--"] + cmd if prefix else cmd


def main() -> int:
    questions = json.loads((DATA / "questions.json").read_text())
    labels = json.loads((DATA / "labels.json").read_text())
    with tempfile.TemporaryDirectory(prefix="vsi-") as tmp:
        io = Path(tmp)
        (io / "videos").mkdir()
        for q in questions:  # resolved copies of the links: the child sees the files, not DATA
            dst = io / q["video"]
            if not dst.exists():
                os.link(os.path.realpath(DATA / q["video"]), dst) if same_fs(DATA, io) else \
                    shutil.copy(DATA / q["video"], dst)
        public = [{k: v for k, v in q.items() if k not in ("answer", "debiased")} for q in questions]
        (io / "questions.json").write_text(json.dumps(public))
        relay = None
        if os.environ.get("RSIB_GATEWAY_URL") and os.environ.get("RSIB_GATEWAY_TOKEN"):
            relay = Relay(os.environ["RSIB_GATEWAY_URL"], os.environ["RSIB_GATEWAY_TOKEN"],
                          CALLS_PER_QUESTION * len(questions))
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": tmp, "TMPDIR": tmp,
               "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1", "RSIB_IO_DIR": tmp,
               "RSIB_SOLVER_DIR": str(WORKSPACE),
               **({"RSIB_GATEWAY_URL": relay.address, "RSIB_GATEWAY_TOKEN": "split"} if relay else {}),
               **{k: os.environ[k] for k in ("CUDA_VISIBLE_DEVICES",) if k in os.environ}}
        proc = subprocess.run(solver_cmd(io), env=env, cwd=tmp, capture_output=True, text=True,
                              timeout=TIMEOUT_S)
        (OUT / "solver.log").write_text(proc.stdout[-200000:] + "\n--- stderr ---\n"
                                        + proc.stderr[-200000:])
        if proc.returncode != 0:
            print(proc.stderr[-4000:], file=sys.stderr)
            return 1
        answers = json.loads((io / "answers.json").read_text())
        if (io / "trace").is_dir():  # whatever the solver logs per question comes back on dev
            shutil.copytree(io / "trace", OUT / "trace", dirs_exist_ok=True)
    rows = []
    for q in questions:
        gt = labels[str(q["id"])]
        pred = answers.get(str(q["id"]), "")
        rows.append({"id": q["id"], "question_type": q["question_type"],
                     "debiased": gt["debiased"], "prediction": str(pred)[:200],
                     "answer": gt["answer"], "score": item_score(q["question_type"], pred,
                                                                 gt["answer"])})
    score, groups = overall(rows)
    deb, _ = overall([r for r in rows if r["debiased"]])
    reward = {"score": score, "debiased_score": deb, "questions": len(rows),
              "answered": sum(str(q["id"]) in answers for q in questions),
              "model_calls": relay.calls if relay else 0,
              "refused_calls": relay.refused if relay else 0,
              "metrics": {"per_group": groups},
              "info": {"items": rows}}
    assert math.isfinite(score)
    (OUT / "reward.json").write_text(json.dumps(reward, indent=1))
    print(json.dumps({k: reward[k] for k in ("score", "debiased_score", "questions", "answered")}))
    return 0


def same_fs(a: Path, b: Path) -> bool:
    return os.stat(os.path.realpath(a)).st_dev == os.stat(b).st_dev


if __name__ == "__main__":
    sys.exit(main())
