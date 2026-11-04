"""Trusted parent for one episode's persistent, CPU-only candidate controller."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_libero_adapter_protocol", Path(__file__).with_name("protocol.py"))
protocol = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(protocol)
IMAGE_KEYS = protocol.IMAGE_KEYS


def adapter_command(python: str, workspace: Path, io_root: Path) -> list[str]:
    """Launch only the neutral copy; omit the caller's label-bearing mount."""
    command = [python, "-u", str(io_root / "adapter_worker.py"),
               "--workspace", str(io_root / "workspace")]
    prefix = json.loads(os.environ.get("RSIB_SANDBOX") or "[]")
    if not prefix:
        return command
    neutral_prefix = []
    index = 0
    while index < len(prefix):
        if (prefix[index] == "--ro" and index + 1 < len(prefix)
                and os.path.realpath(prefix[index + 1]) == os.path.realpath(workspace)):
            index += 2
            continue
        neutral_prefix.append(prefix[index])
        index += 1
    roots = {str(Path(python).parent.parent), str(Path(os.path.realpath(python)).parent.parent)}
    extra_ro = [item for root in sorted(roots) for item in ("--ro", root)]
    return neutral_prefix + extra_ro + ["--rw", str(io_root), "--cwd", str(io_root), "--"] + command


class AdapterWorker:
    """Fresh process per arm; query_policy is the parent's fixed policy callable.

    `gateway_url` is this episode's relay address (runner.Relay). The controller reaches the run's
    model only through it; the evaluation's own gateway token never enters the worker."""

    def __init__(self, python: str, workspace: Path, io_root: Path, gateway_url: str | None = None):
        self.io_root = Path(io_root)
        self.io_root.mkdir(parents=True, exist_ok=True)
        self._runtime = tempfile.TemporaryDirectory(prefix="libero-adapter-", dir="/tmp")
        self.runtime_root = Path(self._runtime.name)
        shutil.copytree(workspace, self.runtime_root / "workspace")
        for name in ("adapter_worker.py", "protocol.py"):
            shutil.copyfile(Path(__file__).with_name(name), self.runtime_root / name)
        self.log = (self.runtime_root / "adapter.log").open("w")
        self.process = subprocess.Popen(
            adapter_command(python, Path(workspace), self.runtime_root), cwd=self.runtime_root,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            env={"PATH": str(Path(python).parent) + ":/usr/bin:/bin", "HOME": str(self.runtime_root),
                 "LANG": "C.UTF-8", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                 "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4",
                 "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                 **({"RSIB_GATEWAY_URL": gateway_url, "RSIB_GATEWAY_TOKEN": "episode"}
                    if gateway_url else {})})

    def act(self, observation, query_policy):
        header, arrays = protocol.observation_message(observation)
        shapes = header["shapes"]
        queries = 0
        try:
            protocol.write_message(self.process.stdin, header, arrays)
            while True:
                response = protocol.read_header(self.process.stdout)
                if set(response) == {"type", "error", "candidate_traceback"} and response["type"] == "error":
                    error = RuntimeError("adapter failed: " + str(response["error"]))
                    error.candidate_traceback = response["candidate_traceback"]
                    raise error
                if response.get("type") == "query":
                    queries += 1
                    if queries > 1:
                        raise ValueError("only one policy query is allowed per step")
                    if response != {"type": "query", "shapes": shapes}:
                        raise ValueError("query must preserve the two image keys and shapes")
                    images = {key: protocol.read_array(self.process.stdout, shapes[key]) for key in IMAGE_KEYS}
                    protocol.validate_images(images, shapes)
                    try:
                        chunk = protocol.validate_chunk(query_policy(images))
                    except Exception as exc:
                        protocol.write_message(self.process.stdin, {"type": "error", "error": f"{type(exc).__name__}: {exc}"})
                        raise
                    protocol.write_message(self.process.stdin, {"type": "chunk", "shape": [30, 7]}, [chunk])
                    continue
                if response != {"type": "action", "shape": [7]}:
                    raise ValueError("expected a seven-element action or policy query")
                return protocol.validate_action(protocol.read_array(self.process.stdout, (7,)))
        except (BrokenPipeError, EOFError) as exc:
            raise RuntimeError("adapter child stopped; see adapter.log") from exc
        except ValueError as exc:
            raise RuntimeError("invalid adapter response: " + str(exc)) from exc

    def close(self):
        if self.process.stdin.closed:
            return
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()
        self.log.close()
        shutil.copyfile(self.runtime_root / "adapter.log", self.io_root / "adapter.log")
        self._runtime.cleanup()
