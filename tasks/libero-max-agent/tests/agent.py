"""Agent-as-policy additions to the trusted runner: camera geometry, depth and the model relay.

The controller sees two RGB images exactly as before, plus per-camera calibration and metric
depth in the same array orientation, and it may call the run's model through a per-episode
relay. The relay caps calls per episode and, for a MAX Dynamic arm, replays the Base arm's
replies while the evaluator reconstructs the shared prefix, so a model-calling controller stays
deterministic across the pair.
"""
from __future__ import annotations

import json
import math
import secrets
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

CAMERA_NAMES = {"agentview": "agentview", "wrist": "robot0_eye_in_hand"}
MAX_REQUEST_BYTES = 32 << 20
UPSTREAM_TIMEOUT_S = 900


def find_sim(env):
    """The MuJoCo sim behind LIBERO's env and its PRO/MAX wrappers."""
    seen, current = set(), env
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        sim = getattr(current, "sim", None)
        if sim is not None and hasattr(sim, "model"):
            return sim
        current = getattr(current, "env", None)
    raise RuntimeError("no MuJoCo sim behind the environment")


def camera_info(sim, camera: str, width: int, height: int) -> dict:
    """Pinhole model in the OpenCV convention (x right, y down, z forward).

    K maps camera-frame points to (u, v) of the upright camera image, which is the public array
    flipped vertically (agent.raw_rows). camera_to_world is the camera pose in the world frame.
    """
    cam_id = sim.model.camera_name2id(CAMERA_NAMES[camera])
    fovy = math.radians(float(sim.model.cam_fovy[cam_id]))
    f = 0.5 * height / math.tan(0.5 * fovy)
    K = np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])
    rotation = np.asarray(sim.data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    pose = np.eye(4)
    pose[:3, :3] = rotation @ np.diag([1.0, -1.0, -1.0])  # MuJoCo looks along -z with y up
    pose[:3, 3] = np.asarray(sim.data.cam_xpos[cam_id], dtype=np.float64)
    return {"K": K, "camera_to_world": pose, "width": width, "height": height}


def render_depth(sim, camera: str, width: int, height: int) -> np.ndarray:
    """Metric depth (distance along the optical axis, meters) in the public array orientation."""
    _, depth = sim.render(width=width, height=height, camera_name=CAMERA_NAMES[camera], depth=True)
    extent = float(sim.model.stat.extent)
    near = float(sim.model.vis.map.znear) * extent
    far = float(sim.model.vis.map.zfar) * extent
    depth = np.asarray(depth, dtype=np.float64)
    metric = near / (1.0 - depth * (1.0 - near / far))
    return np.ascontiguousarray(metric, dtype=np.float32)


class EpisodeGeometry:
    """Agentview calibration is the nominal pose measured at the first step of the episode (a
    camera_shift event moves the camera but not this calibration); the wrist camera follows the
    arm and is measured every step."""

    def __init__(self):
        self.nominal = None

    def observe(self, env, raw) -> tuple[dict, dict]:
        sim = find_sim(env)
        height, width = np.asarray(raw["agentview_image"]).shape[:2]
        if self.nominal is None:
            self.nominal = camera_info(sim, "agentview", width, height)
        wrist_h, wrist_w = np.asarray(raw["robot0_eye_in_hand_image"]).shape[:2]
        cameras = {"agentview": self.nominal,
                   "wrist": camera_info(sim, "wrist", wrist_w, wrist_h)}
        depth = {"agentview": render_depth(sim, "agentview", width, height),
                 "wrist": render_depth(sim, "wrist", wrist_w, wrist_h)}
        return cameras, depth


class Relay:
    """Per-episode proxy to the evaluation's gateway grant (model calls only).

    Each episode gets its own URL. `cap` limits the calls an episode makes, counting replayed
    calls the same way as live ones. An episode opened with `replay` serves those recorded
    (request, status, body) entries in order while `replaying(key)` is true and checks that each
    request equals the recorded one."""

    def __init__(self, url: str | None, token: str | None, cap: int):
        self.url, self.token, self.cap = (url or "").rstrip("/"), token, cap
        self.lock = threading.Lock()
        self.episodes: dict[str, dict] = {}
        relay = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):  # noqa: N802
                status, body = relay.handle(self.path, self.headers, self.rfile)
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def open(self, label: str, replay: list | None = None) -> str:
        key = secrets.token_hex(12)
        with self.lock:
            self.episodes[key] = {"label": label, "calls": 0, "live_calls": 0, "replayed_calls": 0,
                                  "refused": 0, "errors": 0, "model_failures": 0,
                                  "input_tokens": 0, "output_tokens": 0, "record": [],
                                  "replay": list(replay or []), "replaying": bool(replay)}
        return f"http://127.0.0.1:{self.server.server_address[1]}/{key}"

    @staticmethod
    def key(url: str) -> str:
        return url.rsplit("/", 1)[1]

    def set_replaying(self, url: str, replaying: bool) -> None:
        with self.lock:
            self.episodes[self.key(url)]["replaying"] = replaying

    def close(self, url: str) -> dict:
        """Episode statistics plus its recorded exchanges; frees the episode."""
        with self.lock:
            ep = self.episodes.pop(self.key(url))
        if ep["replaying"] and ep["replay"]:
            ep["unused_replay"] = len(ep["replay"])
        return ep

    @staticmethod
    def stats(ep: dict) -> dict:
        return {k: ep[k] for k in ("calls", "live_calls", "replayed_calls", "refused", "errors",
                                   "model_failures", "input_tokens", "output_tokens")}

    def handle(self, path: str, headers, rfile) -> tuple[int, dict]:
        key, _, route = path.lstrip("/").partition("/")
        ep = self.episodes.get(key)
        if ep is None:
            return 401, {"error": "unauthorized", "message": "unknown episode"}
        if route != "v1/complete":
            return 403, {"error": "forbidden", "message": "the policy may call /v1/complete only"}
        size = int(headers.get("Content-Length") or 0)
        if size > MAX_REQUEST_BYTES:
            return 413, {"error": "too_large", "message": f"request over {MAX_REQUEST_BYTES} bytes"}
        raw = rfile.read(size)
        try:
            payload = json.loads(raw)
            assert isinstance(payload, dict)
        except (ValueError, AssertionError):
            return 400, {"error": "bad_request", "message": "body is not a JSON object"}
        request = json.dumps(payload, sort_keys=True)
        with self.lock:
            replaying = ep["replaying"]
        if replaying:
            return self._replay(ep, request)
        status, body = self._live(ep, payload)
        with self.lock:
            ep["record"].append((request, status, body))
        return status, body

    def _replay(self, ep: dict, request: str) -> tuple[int, dict]:
        with self.lock:
            if not ep["replay"]:
                return 409, {"error": "replay_mismatch",
                             "message": "the controller made more model calls than in the Base prefix"}
            expected, status, body = ep["replay"].pop(0)
            if request != expected:
                return 409, {"error": "replay_mismatch",
                             "message": "a prefix model request differs from the Base arm's request"}
            ep["calls"] += 1
            ep["replayed_calls"] += 1
            ep["record"].append((request, status, body))
        return status, body

    def _live(self, ep: dict, payload: dict) -> tuple[int, dict]:
        with self.lock:
            if ep["calls"] >= self.cap:
                ep["refused"] += 1
                return 402, {"error": "episode_call_limit",
                             "message": f"the episode reached its limit of {self.cap} model calls"}
            ep["calls"] += 1
            ep["live_calls"] += 1
        if not self.url or not self.token:
            return 503, {"error": "no_model_access", "message": "this evaluation has no gateway grant"}
        label = ep["label"].replace("/", "-")
        payload["session"] = f"{label}/{payload.get('session', '0')}"
        req = urllib.request.Request(
            self.url + "/v1/complete", data=json.dumps(payload).encode(), method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"})
        try:
            with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT_S) as resp:
                body = json.loads(resp.read())
                usage = body.get("usage") or {}
                with self.lock:
                    ep["input_tokens"] += sum(int(usage.get(k) or 0) for k in (
                        "input_tokens", "cache_read_tokens", "cache_write_tokens"))
                    ep["output_tokens"] += int(usage.get("output_tokens") or 0)
                return resp.status, body
        except urllib.error.HTTPError as exc:
            try:
                body = json.loads(exc.read())
            except ValueError:
                body = {"error": "http_error", "message": str(exc)}
            self._failed(ep, exc.code, body)
            return exc.code, body
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            body = {"error": "unreachable", "message": str(exc)}
            self._failed(ep, 502, body)
            return 502, body

    def _failed(self, ep: dict, status: int, body: dict) -> None:
        with self.lock:
            ep["errors"] += 1
            if status >= 500:
                ep["model_failures"] += 1
                ep["model_failure"] = str(body.get("message", ""))[:300]
