#!/usr/bin/env python3
"""CPU candidate worker: one seeded Controller instance for one episode."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import random
import sys
import traceback
from pathlib import Path

import numpy as np

import protocol


def candidate_traceback(exc, workspace):
    """Return the public candidate frames without worker or library internals."""
    frames = []
    for frame in traceback.extract_tb(exc.__traceback__):
        path = Path(frame.filename)
        if path.is_relative_to(workspace):
            frames.append({"file": path.relative_to(workspace).as_posix(),
                           "line": frame.lineno, "function": frame.name,
                           "source": frame.line})
    return frames


def run(workspace):
    source, target = sys.stdin.buffer, sys.stdout.buffer
    try:
        random.seed(0)
        np.random.seed(0)
        sys.path.insert(0, str(workspace))
        with contextlib.redirect_stdout(sys.stderr):
            spec = importlib.util.spec_from_file_location("rsib_controller", workspace / "controller.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            controller = module.Controller()
            controller.reset()
        while True:
            try:
                header = protocol.read_header(source)
            except EOFError:
                return 0
            observation = protocol.read_observation(source, header)
            shapes = header["shapes"]
            attempts = 0

            def query_policy(images=None):
                nonlocal attempts
                attempts += 1
                if attempts > 1:
                    raise ValueError("only one policy query is allowed per step")
                images = observation["images"] if images is None else images
                protocol.validate_images(images, shapes)
                protocol.write_message(target, {"type": "query", "shapes": shapes},
                                       [images[key] for key in protocol.IMAGE_KEYS])
                response = protocol.read_header(source)
                if response.get("type") == "error":
                    raise RuntimeError(str(response["error"]))
                if response != {"type": "chunk", "shape": [30, 7]}:
                    raise ValueError("expected fixed float32 policy chunk shape (30, 7)")
                return protocol.validate_chunk(protocol.read_array(source, (30, 7)))

            with contextlib.redirect_stdout(sys.stderr):
                action = controller.step(observation, query_policy)
            if attempts > 1:
                raise ValueError("only one policy query is allowed per step")
            protocol.validate_action(action)
            protocol.write_message(target, {"type": "action", "shape": [7]}, [action])
    except Exception as exc:
        protocol.write_message(target, {"type": "error", "error": f"{type(exc).__name__}: {exc}",
                                        "candidate_traceback": candidate_traceback(exc, workspace)})
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.workspace))
