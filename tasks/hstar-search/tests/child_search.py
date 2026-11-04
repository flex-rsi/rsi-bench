"""Runs the submitted policy on every episode of one split, several at a time. It sees the
instructions and the views the environment renders for it; never the panoramas or the targets."""

import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

io = Path(os.environ["RSIB_IO_DIR"])
sys.path.insert(0, os.environ["RSIB_SOLVER_DIR"])
import policy  # noqa: E402


class Env:
    """One episode. `view` is the path of the current view (JPEG, 960 x 720, 100-degree field of
    view, green cross at the center); `yaw`, `pitch` the current direction in degrees."""

    def __init__(self, item: dict):
        self.id, self.kind, self.task = item["id"], item["kind"], item["task"]
        self.max_steps, self.fov = item["max_steps"], item["fov"]
        self._act("start", 0, 0)  # the first view: yaw, pitch, view, step, steps_left, done

    def _act(self, action: str, yaw, pitch) -> dict:
        req = urllib.request.Request(
            os.environ["RSIB_ENV_URL"].rstrip("/") + "/env/act", method="POST",
            data=json.dumps({"id": self.id, "action": action, "yaw": yaw, "pitch": pitch}).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ['RSIB_GATEWAY_TOKEN']}"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                out = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise ValueError(json.loads(exc.read()).get("message", str(exc))) from None
        self.yaw, self.pitch, self.step = out["yaw"], out["pitch"], out["step"]
        self.steps_left, self.done = out["steps_left"], out["done"]
        self.view = out.get("view")
        return out

    def rotate(self, dyaw, dpitch) -> str | None:
        """Turn by (dyaw, dpitch) degrees relative to the current direction (yaw > 0: right,
        pitch > 0: up). Returns the path of the new view, or None when that was the last step."""
        if self.done:
            raise RuntimeError("the episode has ended")
        return self._act("rotate", dyaw, dpitch)["view"]

    def submit(self, yaw, pitch) -> None:
        """End the episode with this absolute direction (degrees)."""
        if self.done:
            raise RuntimeError("the episode has ended")
        self._act("submit", yaw, pitch)


def run(item: dict) -> None:
    t0 = time.time()
    try:
        env = Env(item)
    except Exception:  # the episode could not start (it scores 0)
        traceback.print_exc()
        return
    try:
        policy.search(env)
    except Exception:  # one failed episode scores 0; the others keep their results
        traceback.print_exc()
    finally:
        if not env.done:  # frees its place for the next episode
            env._act("end", 0, 0)
    print(f"{env.id}: steps {env.step} done {env.done} {time.time() - t0:.1f}s", flush=True)


items = [json.loads(line) for line in (io / "episodes.jsonl").read_text().splitlines() if line]
workers = max(1, min(int(getattr(policy, "WORKERS", 8)), 16))  # the policy may set WORKERS
with ThreadPoolExecutor(workers) as pool:
    list(pool.map(run, items))
