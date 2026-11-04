"""Evaluator: run the solution's search policy on every episode of a split and score it here.

The solution runs in a child process that never sees the panoramas or the targets. For each
episode it gets the instruction; every view, the first one included, comes from this process,
which renders it from the panorama when the episode starts and when the policy turns the camera.
The child knows the episodes by opaque ids (e000, e001, ...), and at most MAX_ACTIVE episodes may
be in progress at once: episodes are solved one by one, each on its own views. Both the environment and the model of the run are reached over
one local HTTP server in this process:

    POST {RSIB_ENV_URL}/env/act       {"id", "action": "start" | "rotate" | "submit" | "end",
                                       "yaw", "pitch"}
    POST {RSIB_GATEWAY_URL}/v1/...    forwarded to the run's gateway with the evaluation's token,
                                      at most CALLS_PER_EPISODE x n model calls per split

Environment (as in H*Bench): a 100-degree field of view rendered at 960 x 720 with a green cross at
the center; an episode starts at the given yaw and pitch 0; rotate(dyaw, dpitch) turns relative to
the current direction (pitch clamped to [-90, 90], yaw > 0 turns right, pitch > 0 looks up);
submit(yaw, pitch) ends the episode with that absolute direction. At most MAX_STEPS actions; an
episode that has not submitted by then fails. Success: object search (HOS) needs the submitted yaw
and pitch inside the target ranges, widened to at least +-30 and +-20 degrees around their centers
(when the yaw range crosses 0/360 degrees, the yaw alone, as H*Bench's environment checks it);
path search (HPS) needs the yaw inside the target range, widened to at least +-10 degrees.

Score: success weighted by steps (as SPL weights success by path length). An episode scores
success x fewest / max(fewest, steps), where steps counts its actions (turns and the submission)
and fewest is the fewest actions that can succeed: 1 when the target is in the first view (its
center within half the field of view of the starting yaw), else 2 (one turn, then submit). The
primary metric `spl` is the mean of the object-search and path-search averages; the plain success
rates and step counts are reported too.
"""

import json
import math
import os
import random
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

TIME_LIMIT_S = 2700        # the whole split
MAX_STEPS = 10             # actions per episode
CALLS_PER_EPISODE = 20     # model calls: a split of n episodes may make 20n
MAX_ACTIVE = 16            # episodes in progress at once (the child runs at most 16 workers)
FOV, WIDTH, HEIGHT = 100.0, 960, 720
TOLERANCE = {"hos": (30.0, 20.0), "hps": (10.0, None)}  # yaw, pitch (degrees)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def norm(angle: float) -> float:
    return (angle + 360.0) % 360.0


# ---------------------------------------------------------------------------- rendering
_RAYS = {}


def _rays(fov: float, w: int, h: int) -> np.ndarray:
    """Unit view rays of a pinhole camera looking along +z, one per output pixel (h, w, 3)."""
    key = (fov, w, h)
    if key not in _RAYS:
        f = w / (2 * math.tan(math.radians(fov) / 2))
        i, j = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        x, y = (i - w / 2) / f, (j - h / 2) / f
        r = np.sqrt(x * x + y * y + 1)
        _RAYS[key] = np.stack((x / r, y / r, 1 / r), -1)
    return _RAYS[key]


def render(pano: np.ndarray, yaw: float, pitch: float, fov: float = FOV, w: int = WIDTH,
           h: int = HEIGHT) -> np.ndarray:
    """Perspective view of an equirectangular panorama (RGB uint8), H*Bench's geometry
    (R = Ry(yaw) Rx(pitch), lon/lat lookup, bilinear, wrapping in longitude) with its center cross."""
    a, b = math.radians(yaw), math.radians(pitch)
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    d = _rays(fov, w, h) @ (ry @ rx).T
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
    H, W = pano.shape[:2]
    mx = (lon / np.pi + 1) / 2 * W
    my = (lat / np.pi + 0.5) * H
    x0, y0 = np.floor(mx).astype(np.int64), np.floor(my).astype(np.int64)
    fx, fy = (mx - x0)[..., None], (my - y0)[..., None]
    x1 = (x0 + 1) % W
    x0 = x0 % W
    y0c, y1c = np.clip(y0, 0, H - 1), np.clip(y0 + 1, 0, H - 1)
    p = pano.astype(np.float32)
    out = (p[y0c, x0] * (1 - fx) * (1 - fy) + p[y0c, x1] * fx * (1 - fy)
           + p[y1c, x0] * (1 - fx) * fy + p[y1c, x1] * fx * fy)
    out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    cx, cy, n = w // 2, h // 2, 10
    out[cy, cx - n:cx + n + 1] = (0, 255, 0)
    out[cy - n:cy + n + 1, cx] = (0, 255, 0)
    return out


# ---------------------------------------------------------------------------- scoring
def widened(lo: float, hi: float, tol: float, wrap: bool) -> tuple[float, float]:
    """A target range of at least 2 x tol, centered on the annotated one (H*Bench's rule)."""
    if wrap and lo > hi:
        hi += 360.0
    if hi - lo < 2 * tol:
        c = (lo + hi) / 2
        lo, hi = c - tol, c + tol
    return (norm(lo), norm(hi)) if wrap else (max(-90.0, lo), min(90.0, hi))


def inside_yaw(yaw: float, lo: float, hi: float) -> bool:
    yaw = norm(yaw)
    return lo <= yaw <= hi if lo <= hi else (yaw >= lo or yaw <= hi)


def success(label: dict, yaw: float, pitch: float) -> bool:
    """H*Bench's rule, as its environment applies it: when the (widened) yaw range crosses 0/360
    degrees only the yaw is checked; otherwise object search also needs the pitch in range."""
    tol_yaw, tol_pitch = TOLERANCE[label["kind"]]
    lo, hi = widened(*label["yaw"], tol_yaw, wrap=True)
    if lo > hi:
        return inside_yaw(yaw, lo, hi)
    if not lo <= norm(yaw) <= hi:
        return False
    if label["kind"] == "hos":
        plo, phi = widened(*label["pitch"], tol_pitch, wrap=False)
        return plo <= pitch <= phi
    return True


# ---------------------------------------------------------------------------- environment
class Episode:
    def __init__(self, key: str, ep: dict, label: dict, pano_path: Path, view_dir: Path):
        self.key, self.ep, self.label, self.pano_path, self.view_dir = key, ep, label, pano_path, view_dir
        self.yaw, self.pitch = norm(float(ep["start_yaw"])), 0.0
        self.steps, self.done, self.won, self.submitted = 0, False, False, None
        self.started = False
        self.actions: list[str] = []
        self.lock = threading.Lock()
        self._pano = None

    def pano(self) -> np.ndarray:
        if self._pano is None:
            self._pano = np.asarray(Image.open(self.pano_path).convert("RGB"))
        return self._pano

    def view(self) -> str:
        path = self.view_dir / f"{self.key}_{self.steps:02d}.jpg"
        Image.fromarray(render(self.pano(), self.yaw, self.pitch)).save(path, quality=92)
        return str(path)

    def state(self, view: str | None) -> dict:
        return {"id": self.key, "yaw": round(self.yaw) % 360, "pitch": round(self.pitch),
                "step": self.steps, "steps_left": MAX_STEPS - self.steps, "done": self.done,
                "view": view}

    def fewest_steps(self) -> int:
        """1 when the target's center is in the first view, else 2 (turn, then submit)."""
        lo, hi = self.label["yaw"]
        center = (lo + hi) / 2 if lo <= hi else norm((lo + hi + 360.0) / 2)
        off = abs((norm(float(self.ep["start_yaw"])) - center + 180.0) % 360.0 - 180.0)
        return 1 if off <= FOV / 2 else 2

    def act(self, req: dict, active) -> dict:
        with self.lock:
            if self.done:
                return {"error": "done", "message": "this episode has ended", **self.state(None)}
            if req.get("action") == "start":
                if self.started:
                    return {"error": "bad_request", "message": "this episode has already started"}
                if not active(self):  # marks the episode started when there is room
                    return {"error": "bad_request", "message": f"at most {MAX_ACTIVE} episodes "
                            "may be in progress at once"}
                return self.state(self.view())
            if not self.started:
                return {"error": "bad_request", "message": "start the episode first"}
            if req.get("action") == "end":  # the policy returned without submitting
                self.done = True
                self._pano = None
                return self.state(None)
            try:
                a, b = float(req.get("yaw", 0)), float(req.get("pitch", 0))
            except (TypeError, ValueError):
                return {"error": "bad_request", "message": "yaw and pitch must be numbers"}
            if not (math.isfinite(a) and math.isfinite(b)):
                return {"error": "bad_request", "message": "yaw and pitch must be finite"}
            self.steps += 1
            if req.get("action") == "submit":
                self.actions.append(f"submit({round(a)},{round(b)})")
                self.submitted = (norm(round(a)), float(round(b)))
                self.won = success(self.label, *self.submitted)
                self.done = True
                self._pano = None
                return self.state(None)
            if req.get("action") != "rotate":
                self.steps -= 1
                return {"error": "bad_request", "message": "action is 'rotate' or 'submit'"}
            self.actions.append(f"rotate({round(a)},{round(b)})")
            self.yaw = norm(self.yaw + round(a))
            self.pitch = max(-90.0, min(90.0, self.pitch + round(b)))
            if self.steps >= MAX_STEPS:  # out of actions without a submission
                self.done = True
                self._pano = None
                return self.state(None)
            return self.state(self.view())


class Server:
    """The child's gateway and environment: forwards model calls (within the split's call budget)
    and runs the episodes' actions."""

    def __init__(self, episodes: dict, upstream: str | None, token: str | None, max_calls: int):
        self.episodes, self.max_calls = episodes, max_calls
        self.upstream, self.token = (upstream or "").rstrip("/"), token
        self.key = secrets.token_hex(16)
        self.calls = self.refused = 0
        self.lock = threading.Lock()
        self.admit_lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def reply(self, status: int, body) -> None:
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):  # noqa: N802
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if self.headers.get("Authorization") != f"Bearer {server.key}":
                    return self.reply(401, {"error": "unauthorized"})
                if self.path == "/env/act":
                    try:
                        req = json.loads(raw or b"{}")
                        episode = server.episodes[req["id"]]
                    except Exception:
                        return self.reply(400, {"error": "bad_request", "message": "unknown episode"})
                    out = episode.act(req, server.admit)
                    return self.reply(400 if out.get("error") == "bad_request" else 200, out)
                if not server.upstream:
                    return self.reply(503, {"error": "no_model", "message": "no model in this run"})
                if self.path == "/v1/complete":
                    with server.lock:
                        over = server.calls >= server.max_calls
                        server.refused += over
                        server.calls += not over
                    if over:
                        return self.reply(400, {"error": "call_budget", "message":
                                                f"this split may make at most {server.max_calls} "
                                                f"model calls ({CALLS_PER_EPISODE} per episode)"})
                req = urllib.request.Request(server.upstream + self.path, data=raw, method="POST",
                                             headers={"Content-Type": "application/json",
                                                      "Authorization": f"Bearer {server.token}"})
                try:
                    with urllib.request.urlopen(req, timeout=3600) as resp:
                        return self.reply(resp.status, resp.read())
                except urllib.error.HTTPError as exc:
                    return self.reply(exc.code, exc.read())
                except Exception as exc:
                    return self.reply(502, {"error": "unreachable", "message": str(exc)[:300]})

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.http.daemon_threads = True
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.http.server_address[1]}"


    def admit(self, episode) -> bool:
        """Start `episode` if fewer than MAX_ACTIVE others are in progress (called with that
        episode's lock held)."""
        with self.admit_lock:
            if sum(e.started and not e.done for e in self.episodes.values()) >= MAX_ACTIVE:
                return False
            episode.started = True
            return True


def solution_cmd(python: str, io_dir: Path) -> list[str]:
    """The policy's process: through the sandbox launcher when the orchestrator gives one, so it
    sees its own code, the run cache and io_dir, never the panoramas or labels."""
    child = io_dir / "_child_search.py"
    shutil.copy(Path(__file__).with_name("child_search.py"), child)
    cmd = [python, str(child)]
    prefix = json.loads(os.environ.get("RSIB_SANDBOX") or "[]")
    return prefix + ["--rw", str(io_dir), "--cwd", str(io_dir), "--"] + cmd if prefix else cmd


def rate(rows: list[dict], key: str = "success") -> float | None:
    return sum(r[key] for r in rows) / len(rows) if rows else None


def main() -> None:
    split_dir = Path(os.environ["RSIB_DATA_DIR"])
    workspace = Path(os.environ["RSIB_WORKSPACE"])
    out_dir = Path(os.environ["RSIB_OUTPUT_DIR"])
    python = os.environ.get("RSIB_PYTHON", sys.executable)
    runtime = workspace / "runtime.txt"  # optional: another interpreter (inside $RSIB_CACHE)
    if runtime.exists() and runtime.read_text().strip():
        python = runtime.read_text().strip().splitlines()[0].strip()

    eps = read_jsonl(split_dir / "episodes.jsonl")
    labels = {row["id"]: row for row in read_jsonl(split_dir / "labels.jsonl")}
    with tempfile.TemporaryDirectory(prefix="hstar-", dir=out_dir) as tmp:
        io_dir = Path(tmp)
        (io_dir / "views").mkdir()
        # the child sees opaque ids, not the episode ids (which name their scenes)
        episodes = {}
        order = eps[:]
        random.Random(0).shuffle(order)  # so that neighbouring keys are not the same scene
        for i, e in enumerate(order):
            key = f"e{i:03d}"
            episodes[key] = Episode(key, e, labels[e["id"]], split_dir / "panoramas" / e["panorama"],
                                    io_dir / "views")
        items = [{"id": key, "kind": ep.ep["kind"], "task": ep.ep["task"], "max_steps": MAX_STEPS,
                  "fov": FOV} for key, ep in episodes.items()]
        (io_dir / "episodes.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
        server = Server(episodes, os.environ.get("RSIB_GATEWAY_URL"),
                        os.environ.get("RSIB_GATEWAY_TOKEN"), CALLS_PER_EPISODE * len(eps))
        env = {k: v for k, v in os.environ.items() if not k.startswith("RSIB_")}
        env.update({"RSIB_IO_DIR": tmp, "RSIB_SOLVER_DIR": str(workspace),
                    "RSIB_ENV_URL": server.url, "RSIB_GATEWAY_URL": server.url,
                    "RSIB_GATEWAY_TOKEN": server.key, "HSTAR_MAX_CALLS": str(server.max_calls),
                    "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"})
        t0 = time.time()
        try:
            proc = subprocess.run(solution_cmd(python, io_dir), env=env, cwd=tmp,
                                  capture_output=True, text=True, timeout=TIME_LIMIT_S)
        except subprocess.TimeoutExpired:
            sys.exit(f"the policy exceeded {TIME_LIMIT_S}s on this split")
        print(proc.stdout[-4000:])
        if proc.stderr.strip():
            print("[policy stderr]\n" + proc.stderr[-6000:])
        if proc.returncode != 0:
            sys.exit("policy failed")

    rows = []
    for ep in episodes.values():
        e = ep.ep
        fewest = ep.fewest_steps()
        rows.append({"id": e["id"], "kind": e["kind"], "level": ep.label.get("level"),
                     "task": e["task"], "start_yaw": e["start_yaw"], "actions": ep.actions,
                     "steps": ep.steps, "fewest_steps": fewest, "submitted": ep.submitted,
                     "success": bool(ep.won),
                     "spl": fewest / max(fewest, ep.steps) if ep.won else 0.0})
    hos = [r for r in rows if r["kind"] == "hos"]
    hps = [r for r in rows if r["kind"] == "hps"]

    def mean_of_kinds(key: str) -> float:
        vals = [x for x in (rate(hos, key), rate(hps, key)) if x is not None]
        return sum(vals) / len(vals) if vals else 0.0

    won = [r for r in rows if r["success"]]
    reward = {"spl": mean_of_kinds("spl"), "success": mean_of_kinds("success"),
              "hos_spl": rate(hos, "spl"), "hps_spl": rate(hps, "spl"),
              "hos_sr": rate(hos), "hps_sr": rate(hps), "n": len(rows),
              "submitted": sum(r["submitted"] is not None for r in rows) / max(1, len(rows)),
              "avg_steps": rate(rows, "steps"), "hos_avg_steps": rate(hos, "steps"),
              "hps_avg_steps": rate(hps, "steps"), "avg_steps_success": rate(won, "steps"),
              "model_calls": server.calls, "refused_calls": server.refused,
              "seconds": round(time.time() - t0, 1)}
    for kind, group in (("hos", hos), ("hps", hps)):
        for level in sorted({r["level"] for r in group if r["level"] is not None}):
            reward[f"{kind}_sr_l{level}"] = rate([r for r in group if r["level"] == level])
    reward = {k: v for k, v in reward.items() if v is not None}
    reward["artifacts"] = [{"path": "per_episode.json", "role": "diagnosis",
                            "note": "actions, submitted direction, success and step-weighted "
                                    "score of every episode"}]
    (out_dir / "per_episode.json").write_text(json.dumps(rows, indent=1))
    (out_dir / "reward.json").write_text(json.dumps(reward))
    print(json.dumps({k: v for k, v in reward.items() if k != "artifacts"}))


if __name__ == "__main__":
    main()
