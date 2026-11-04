#!/usr/bin/env python3
"""Frozen X-VLA adapter; workspace code receives image tensors only."""

from __future__ import annotations

import argparse
import collections
import contextlib
import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

IMAGE_KEYS = ("observation.images.image", "observation.images.image2")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def transform_command(python: str, io_dir: Path, workspace: Path) -> list[str]:
    child = io_dir / "image_transform.py"
    command = [python, "-u", str(child), "--transform-worker", "--workspace", str(workspace)]
    prefix = json.loads(os.environ.get("RSIB_SANDBOX") or "[]")
    if not prefix:
        return command
    roots = {str(Path(python).parent.parent), str(Path(os.path.realpath(python)).parent.parent)}
    extra_ro = [item for root in sorted(roots) for item in ("--ro", root)]
    return prefix + extra_ro + ["--rw", str(io_dir), "--cwd", str(io_dir), "--"] + command


class ImageTransform:
    """One persistent CPU child, using the evaluator's existing solution sandbox."""

    def __init__(self, python: str, workspace: Path, io_dir: Path):
        io_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__), io_dir / "image_transform.py")
        self.log = (io_dir / "transform.log").open("w")
        self.process = subprocess.Popen(
            transform_command(python, io_dir, workspace), cwd=io_dir,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            env={"PATH": str(Path(python).parent) + ":/usr/bin:/bin", "HOME": str(io_dir),
                 "LANG": "C.UTF-8", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                 "CUDA_VISIBLE_DEVICES": "",
                 "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})

    def __call__(self, images: dict) -> dict:
        import torch

        try:
            # Requests come from this evaluator; responses come from agent code.
            pickle.dump(images, self.process.stdin, protocol=pickle.HIGHEST_PROTOCOL)
            self.process.stdin.flush()
            header = self.process.stdout.readline(4096)
            if not header:
                raise EOFError
            response = json.loads(header)
            if not isinstance(response, dict):
                raise ValueError("expected a JSON object")
            if "error" in response:
                raise RuntimeError("image transform failed: " + str(response["error"]))
            shapes = {key: list(images[key].shape) for key in IMAGE_KEYS}
            if response != {"shapes": shapes}:
                raise ValueError("response must preserve the two image keys and shapes")
            transformed = {}
            for key in IMAGE_KEYS:
                # Only the trusted input determines how many bytes to read.
                size = images[key].numel() * 4
                payload = self.process.stdout.read(size)
                if len(payload) != size:
                    raise EOFError
                value = torch.frombuffer(bytearray(payload), dtype=torch.float32).reshape(images[key].shape)
                if not bool(torch.isfinite(value).all()) or float(value.min()) < 0 or float(value.max()) > 1:
                    raise ValueError(f"{key} must contain finite pixels in [0, 1]")
                transformed[key] = value
        except (BrokenPipeError, EOFError) as exc:
            raise RuntimeError("image transform child stopped; see transform.log") from exc
        except ValueError as exc:
            raise RuntimeError("invalid image transform response: " + str(exc)) from exc
        return transformed

    def close(self) -> None:
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()
        self.log.close()


def transform_worker(workspace: Path) -> int:
    import torch

    source, target = sys.stdin.buffer, sys.stdout.buffer
    sys.path.insert(0, str(workspace))
    while True:
        try:
            images = pickle.load(source)
        except EOFError:
            return 0
        try:
            # A fresh solution module on each query makes the exported function
            # stateless without restarting the tensor runtime on every call.
            with contextlib.redirect_stdout(sys.stderr):
                solution = load_module(workspace / "solution.py", "rsib_solution")
                transformed = solution.transform_images(images)
            if not isinstance(transformed, dict) or set(transformed) != set(IMAGE_KEYS):
                raise ValueError("transform_images must preserve the two image keys")
            for key, value in transformed.items():
                if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                        or value.dtype != torch.float32 or value.shape != images[key].shape):
                    raise ValueError(f"{key} must preserve CPU float32 tensor shape")
                if not bool(torch.isfinite(value).all()) or float(value.min()) < 0 or float(value.max()) > 1:
                    raise ValueError(f"{key} must contain finite pixels in [0, 1]")
            response = {"shapes": {key: list(transformed[key].shape) for key in IMAGE_KEYS}}
            payloads = [transformed[key].detach().contiguous().numpy().tobytes() for key in IMAGE_KEYS]
        except Exception as exc:
            response = {"error": f"{type(exc).__name__}: {exc}"}
            payloads = []
        target.write(json.dumps(response).encode("utf-8") + b"\n")
        for payload in payloads:
            target.write(payload)
        target.flush()


def install_image_hook(native, transform, capture=None) -> None:
    """Apply the solution after native formatting and before policy preprocessing."""
    original = native._format_observation

    def format_observation(raw_obs, env):
        observation = original(raw_obs, env)
        images = {key: observation[key] for key in IMAGE_KEYS}
        transformed = transform(images)
        if not isinstance(transformed, dict) or set(transformed) != set(images):
            raise ValueError("transform_images must preserve the two image keys")
        observation.update(transformed)
        if capture is not None:
            capture(images, transformed, raw_obs=raw_obs)
        return observation

    native._format_observation = format_observation


class FrameCapture:
    """Dev-only actual-query statistics and bounded public temporal previews.

    Events may occur inside an action chunk. Event frames observe native step
    results; processed images exist only at actual model queries.
    """

    def __init__(self, root: Path):
        self.root, self.case_id, self.active = root, None, None
        self.summaries = []

    def start_case(self, case_id: str, arm: str | None = None) -> None:
        self.finish_arm()
        self.case_id = case_id
        if arm is not None:
            self.start_arm(arm)

    def start_arm(self, arm: str) -> None:
        self.finish_arm()
        self.active = {"case_id": self.case_id, "arm": arm, "query_count": 0,
                       "cameras": {key: {"raw_mean": 0.0, "processed_mean": 0.0,
                                         "raw_robust_sigma": 0.0, "processed_robust_sigma": 0.0,
                                         "pixel_mae": 0.0, "max_pixel_mae": 0.0}
                                   for key in IMAGE_KEYS}}
        self.first_query = self.last_query = None
        self.recent_steps = collections.deque(maxlen=2)
        self.event_frames = []
        self.event_step = None
        self.query_steps = []

    @staticmethod
    def _policy_step(env) -> int:
        return max(0, env.total_env_steps - env.warmup_steps)

    @staticmethod
    def _preview(image):
        import numpy as np
        from PIL import Image

        pixels = np.clip(np.rint(image.transpose(1, 2, 0) * 255), 0, 255).astype(np.uint8)
        return Image.fromarray(pixels).resize((180, 180))

    def __call__(self, images: dict, transformed: dict, *, raw_obs=None) -> None:
        metadata = {"policy_step": (self.recent_steps[-1][0]["policy_step"]
                                    if self.recent_steps else 0)}
        if raw_obs is not None and "robot0_eef_pos" in raw_obs:
            metadata["eef_position_m"] = raw_obs["robot0_eef_pos"].tolist()
        if self.recent_steps:
            metadata["action"] = self.recent_steps[-1][0].get("action")
        self.record({key: value[0].detach().numpy() for key, value in images.items()},
                    {key: value[0].detach().numpy() for key, value in transformed.items()},
                    **metadata)

    def observe_step(self, raw_obs: dict, env, action, event=None) -> None:
        """Observe native results without changing events, actions or query cadence."""
        import numpy as np

        if self.active is None or env.total_env_steps <= env.warmup_steps:
            return
        step = self._policy_step(env)
        metadata = {"policy_step": step, "model_query": False,
                    "action": np.asarray(action).tolist(),
                    "eef_position_m": (np.asarray(raw_obs["robot0_eef_pos"]).tolist()
                                       if "robot0_eef_pos" in raw_obs else None)}
        previews = []
        for key in ("agentview_image", "robot0_eye_in_hand_image"):
            raw = np.asarray(raw_obs[key]).transpose(2, 0, 1) / 255.0
            previews.extend((self._preview(raw), None))
        frame = (metadata, previews)
        if event is not None:
            # Legacy native name denotes the event step, not a model-query boundary.
            self.event_step = event["cosmos_query_boundary_step"]
            self.event_frames = list(self.recent_steps)[-1:] + [frame]
            metadata["event"] = True
        elif self.event_step is not None and step == self.event_step + 1:
            self.event_frames.append(frame)
        self.recent_steps.append(frame)

    def record(self, images: dict, transformed: dict, *, policy_step=None,
               action=None, eef_position_m=None) -> None:
        """Consume CPU C,H,W arrays without retaining full-resolution queries."""
        import numpy as np

        if self.active is None:
            return
        previews = []
        for key in IMAGE_KEYS:
            raw, processed = images[key], transformed[key]
            stats = self.active["cameras"][key]
            stats["raw_mean"] += float(raw.mean())
            stats["processed_mean"] += float(processed.mean())
            stats["raw_robust_sigma"] += robust_sigma(raw)
            stats["processed_robust_sigma"] += robust_sigma(processed)
            mae = float(np.abs(processed - raw).mean())
            stats["pixel_mae"] += mae
            stats["max_pixel_mae"] = max(stats["max_pixel_mae"], mae)
            for image in (raw, processed):
                previews.append(self._preview(image))
        self.active["query_count"] += 1
        metadata = {"policy_step": policy_step, "model_query": True,
                    "action": action, "eef_position_m": eef_position_m}
        if self.event_step is not None and policy_step == self.event_step:
            metadata["event"] = True
        frame = (metadata, previews)
        if self.first_query is None:
            self.first_query = frame
        self.last_query = frame
        if policy_step is not None:
            self.query_steps.append(policy_step)
            if (self.event_step is not None and policy_step >= self.event_step
                    and not any(item[0].get("model_query") for item in self.event_frames)):
                self.event_frames.append(frame)

    def _finish_sequence(self) -> None:
        from PIL import Image, ImageDraw

        frames = [item for item in (self.first_query, *self.event_frames, self.last_query)
                  if item is not None]
        # At a shared step, the query frame adds processed pixels to the same real observation.
        indexed = {}
        for metadata, previews in frames:
            key = metadata["policy_step"]
            if key not in indexed or metadata["model_query"]:
                indexed[key] = (metadata, previews)
        frames = [indexed[key] for key in sorted(indexed, key=lambda step: -1 if step is None else step)]
        if not frames:
            return
        folder = self.root / self.case_id
        folder.mkdir(parents=True, exist_ok=True)
        stem = self.active["arm"] + "_sequence"
        sheet = Image.new("RGB", (720, 34 + 224 * len(frames)), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((4, 2), f"{self.case_id} / {self.active['arm']} / actual public observations", fill="black")
        draw.text((4, 18), "agentview raw | processed at query | wrist raw | processed at query", fill="black")
        for row, (metadata, previews) in enumerate(frames):
            top = 34 + row * 224
            for column, preview in enumerate(previews):
                if preview is not None:
                    sheet.paste(preview, (180 * column, top))
            label = f"step {metadata['policy_step']} / model_query={metadata['model_query']}"
            if metadata.get("event"):
                label += " / event"
            draw.text((4, top + 180), label, fill="black")
            for offset, key in ((194, "action"), (208, "eef_position_m")):
                value = metadata.get(key)
                text = None if value is None else [round(float(item), 3) for item in value]
                draw.text((4, top + offset), f"{key}: {text}", fill="black")
        sheet.save(folder / (stem + ".png"))
        sequence = {"image_file": stem + ".png",
                    "event_step": self.event_step, "actual_query_steps": self.query_steps,
                    "frames": [metadata for metadata, _ in frames]}
        (folder / (stem + ".json")).write_text(json.dumps(sequence, indent=2))
        self.active["frame_sequence_path"] = str((folder / (stem + ".json")).relative_to(self.root.parent))
        self.active["event_step"] = self.event_step
        self.active["actual_query_steps"] = self.query_steps

    def finish_arm(self) -> None:
        if self.active is None:
            return
        count = self.active["query_count"]
        for stats in self.active["cameras"].values():
            for key in stats:
                if not count:
                    stats[key] = None
                elif key != "max_pixel_mae":
                    stats[key] /= count
        if count:
            from PIL import Image, ImageDraw

            sheet = Image.new("RGB", (720, 414), "white")
            draw = ImageDraw.Draw(sheet)
            draw.text((4, 2), f"{self.case_id} / {self.active['arm']} / {count} model queries", fill="black")
            labels = ("agentview raw", "agentview processed", "wrist raw", "wrist processed")
            for column, label in enumerate(labels):
                draw.text((180 * column + 4, 18), label, fill="black")
            for row, frames in enumerate((self.first_query[1], self.last_query[1])):
                top = 34 + row * 190
                for column, frame in enumerate(frames):
                    sheet.paste(frame, (180 * column, top))
                draw.text((4, top + 180), "first model query" if row == 0 else "last model query", fill="black")
            folder = self.root / self.case_id
            folder.mkdir(parents=True, exist_ok=True)
            sheet.save(folder / (self.active["arm"] + ".png"))
        self._finish_sequence()
        self.summaries.append(self.active)
        self.active = None

    def finish(self) -> None:
        self.finish_arm()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "image_diagnostics.json").write_text(json.dumps(self.summaries, indent=2))


def robust_sigma(image) -> float:
    """Current-image noise estimate from median absolute mixed differences."""
    import numpy as np

    mixed = image[:, 1:, 1:] - image[:, :-1, 1:] - image[:, 1:, :-1] + image[:, :-1, :-1]
    return float(np.median(np.abs(mixed)) / (2 * 0.67448975))


def native_runner(runtime: dict):
    scripts = Path(runtime["max_root"]) / "scripts"
    sys.path.insert(0, str(scripts))
    sys.path.insert(0, str(Path(runtime["max_root"]) / "src"))
    return load_module(scripts / "run_xvla_persistent_shard.py", "rsib_native_xvla")


def run_max(native, args, runtime: dict, capture) -> int:
    if capture is not None:
        original_loader = native.load_case_task
        original_wrapper = native.CosmosInterventionEnv

        def load_case_task(case, *other):
            capture.start_case(case["case_id"])
            return original_loader(case, *other)

        def intervention_env(*other, **kwargs):
            # The official runner constructs each arm after loading one case.
            # Statistics use model queries; temporal previews observe native steps too.
            capture.start_arm(kwargs["arm"])
            wrapped = original_wrapper(*other, **kwargs)
            original_step = wrapped.step

            def step(action):
                result = original_step(action)
                capture.observe_step(result[0], wrapped, action,
                                     (result[3] or {}).get("libero_max_event"))
                return result

            wrapped.step = step
            return wrapped

        native.load_case_task = load_case_task
        native.CosmosInterventionEnv = intervention_env
    previous = sys.argv
    sys.argv = ["run_xvla_persistent_shard.py", str(args.manifest),
                "--output-root", str(args.output_root), "--shard-index", "0", "--num-shards", "1",
                "--lerobot-root", runtime["lerobot_root"], "--checkpoint", runtime["checkpoint"],
                "--query-interval", "30"]
    try:
        return native.main()
    finally:
        sys.argv = previous
        if capture is not None:
            native.load_case_task = original_loader
            native.CosmosInterventionEnv = original_wrapper


def run_pro_episode(native, case: dict, policy_parts: dict, dependencies: dict, capture=None) -> dict:
    """A PRO episode using the same native X-VLA controls and saved initial state."""
    import numpy as np

    seed = int(case["policy_seed"])
    dependencies["reseed"](seed)
    task, initial_states = native.load_case_task(case, dependencies["benchmark"], dependencies["suite_cache"])
    env, instruction = native.create_libero_env_with_retry(
        lambda: native._create_libero_env(
            task, 360, seed, bddl_root=dependencies["get_libero_path"]("bddl_files"),
            env_type=dependencies["env_type"]),
        policy_seed=seed, reseed=dependencies["reseed"])
    env = native.wrap_case_env(env, case)
    started = time.monotonic()
    success, steps, queries = False, 0, []
    if capture is not None:
        capture.start_case(case["case_id"], "pro")
    try:
        env.seed(seed)
        env.reset()
        render_stats = dependencies["install_stable_camera_render"](env)
        observation = env.set_init_state(initial_states[case["init_state_index"]])
        render_qa = dependencies["prime_offscreen_renderer"](env)
        if render_qa["status"] != "passed":
            raise RuntimeError("PRO offscreen renderer did not pass initialization")
        render_qa["stable_camera_render"] = render_stats
        render_qa["fallbacks_before_policy"] = int(render_stats.get("fallbacks", 0))
        for _ in range(10):
            observation, _, _, _ = env.step(dependencies["dummy_action"]())
        for robot in env.robots:
            robot.controller.use_delta = False
        action_plan = collections.deque()
        for step in range(native.MAX_STEPS[case["task_suite_name"]]):
            if not action_plan:
                actions = native._predict_action_chunk(
                    raw_obs=observation, env=env, instruction=instruction, query_interval=30,
                    query_seed=seed + step, action_key=dependencies["action_key"], **policy_parts)
                if actions.ndim != 2 or len(actions) < 30:
                    raise ValueError(f"X-VLA returned invalid action chunk: {actions.shape}")
                action_plan.extend(actions[:30])
                queries.append({"policy_step": step, "query_seed": seed + step})
            observation, _, done, _ = env.step(np.asarray(action_plan.popleft()).tolist())
            steps = step + 1
            if done:
                success = True
                break
        if int(render_stats.get("fallbacks", 0)) != render_qa["fallbacks_before_policy"]:
            raise RuntimeError("PRO policy observation used an unstable render fallback")
        return {"case_id": case["case_id"], "success": success, "policy_steps": steps,
                "failure_category": None if success else "horizon_exhausted",
                "max_policy_steps": native.MAX_STEPS[case["task_suite_name"]],
                "query_interval": 30, "warmup_steps": 10, "policy_seed": seed,
                "init_state_index": case["init_state_index"], "instruction": instruction,
                "queries": queries, "substrate": getattr(env, "substrate_info", {}),
                "render_initialization_qa": render_qa, "wall_s": time.monotonic() - started}
    finally:
        env.close()


def run_pro(native, args, runtime: dict, capture) -> int:
    import numpy as np
    import torch

    sys.path.insert(0, str(Path(runtime["lerobot_root"]) / "src"))
    from lerobot.envs.libero import get_libero_dummy_action
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    from lerobot.policies.xvla.processor_xvla import make_xvla_libero_pre_post_processors
    from lerobot.utils.constants import ACTION
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero_max.env_factory import install_stable_camera_render, prime_offscreen_renderer

    cases = json.loads(args.manifest.read_text())["cases"]

    def reseed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    reseed(int(cases[0]["policy_seed"]))
    device = torch.device("cuda")
    policy = XVLAPolicy.from_pretrained(runtime["checkpoint"]).to(device).eval()
    if int(policy.config.n_action_steps) < 30:
        raise ValueError("fixed X-VLA checkpoint must support native 30-step chunks")
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, runtime["checkpoint"],
        preprocessor_overrides={"device_processor": {"device": str(device)}})
    env_preprocessor, env_postprocessor = make_xvla_libero_pre_post_processors()
    policy_parts = dict(policy=policy, preprocessor=preprocessor, postprocessor=postprocessor,
                        env_preprocessor=env_preprocessor, env_postprocessor=env_postprocessor)
    dependencies = dict(benchmark=benchmark, get_libero_path=get_libero_path,
                        env_type=OffScreenRenderEnv, suite_cache={}, reseed=reseed,
                        dummy_action=get_libero_dummy_action, action_key=ACTION,
                        install_stable_camera_render=install_stable_camera_render,
                        prime_offscreen_renderer=prime_offscreen_renderer)
    episodes = []
    for index, case in enumerate(cases, 1):
        episodes.append(run_pro_episode(native, case, policy_parts, dependencies, capture))
        (args.output_root / "pro_results.json").write_text(json.dumps({"episodes": episodes}, indent=2))
        print(f"[{index}/{len(cases)}] {case['case_id']} success={episodes[-1]['success']}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transform-worker", action="store_true")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("pro", "max"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--capture-frames", action="store_true")
    args = parser.parse_args()
    if args.transform_worker:
        return transform_worker(args.workspace)
    if any(value is None for value in (args.benchmark, args.manifest, args.runtime, args.output_root)):
        parser.error("rollout requires --benchmark, --manifest, --runtime and --output-root")
    args.output_root.mkdir(parents=True, exist_ok=True)
    runtime = json.loads(args.runtime.read_text())
    native = native_runner(runtime)
    transform = ImageTransform(runtime["python"], args.workspace, args.output_root / "transform-io")
    capture = FrameCapture(args.output_root / "frames") if args.capture_frames else None
    install_image_hook(native, transform, capture)
    try:
        return run_max(native, args, runtime, capture) if args.benchmark == "max" else run_pro(native, args, runtime, capture)
    finally:
        if capture is not None:
            capture.finish()
        transform.close()


if __name__ == "__main__":
    raise SystemExit(main())
