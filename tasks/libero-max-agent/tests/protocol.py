"""JSON headers and raw float32 payloads shared by the CPU adapter boundary."""

from __future__ import annotations

import json

import numpy as np

IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
PROPRIO_KEYS = ("eef_pos", "eef_rot6d", "gripper_qpos", "joint_qpos")
CAMERAS = ("agentview", "wrist")  # the same cameras as the two image keys, in order


def read_header(source):
    line = source.readline(65536)
    if not line:
        raise EOFError
    if not line.endswith(b"\n"):
        raise ValueError("expected a bounded JSON header")
    header = json.loads(line)
    if not isinstance(header, dict):
        raise ValueError("expected a JSON object")
    return header


def write_message(target, header, arrays=()):
    target.write(json.dumps(header, allow_nan=False).encode("utf-8") + b"\n")
    for array in arrays:
        target.write(np.ascontiguousarray(array, dtype=np.float32).tobytes())
    target.flush()


def read_array(source, shape):
    size = int(np.prod(shape)) * 4
    payload = source.read(size)
    if len(payload) != size:
        raise EOFError
    return np.frombuffer(bytearray(payload), dtype=np.float32).reshape(shape)


def validate_images(images, shapes=None):
    if not isinstance(images, dict) or set(images) != set(IMAGE_KEYS):
        raise ValueError("images must preserve the two image keys")
    for key in IMAGE_KEYS:
        value = images[key]
        if (not isinstance(value, np.ndarray) or value.dtype != np.float32
                or value.ndim != 4 or value.shape[:2] != (1, 3)
                or min(value.shape[-2:]) < 1
                or (shapes is not None and list(value.shape) != shapes[key])):
            raise ValueError(f"{key} must preserve float32 image shape [1, 3, H, W]")
        if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
            raise ValueError(f"{key} must contain finite pixels in [0, 1]")
    return {key: list(images[key].shape) for key in IMAGE_KEYS}


def validate_action(action):
    if not isinstance(action, np.ndarray) or action.dtype != np.float32 or action.shape != (7,):
        raise ValueError("step must return a float32 action with shape (7,)")
    if not np.isfinite(action).all():
        raise ValueError("action must contain finite values")
    return action


def validate_chunk(chunk):
    if not isinstance(chunk, np.ndarray) or chunk.dtype != np.float32 or chunk.shape != (30, 7):
        raise ValueError("query_policy must return float32 shape (30, 7)")
    if not np.isfinite(chunk).all():
        raise ValueError("policy chunk must contain finite values")
    return chunk


def observation_message(observation):
    """Whitelist public robot fields before any candidate process sees them."""
    images = observation["images"]
    shapes = validate_images(images)
    proprio = {}
    for key in PROPRIO_KEYS:
        if key not in observation["proprio"]:
            if key in ("eef_pos", "eef_rot6d"):
                raise ValueError(f"proprio requires {key}")
            continue
        value = np.asarray(observation["proprio"][key], dtype=np.float32)
        if value.ndim != 1 or not np.isfinite(value).all():
            raise ValueError(f"proprio {key} must be a finite vector")
        if key in ("eef_pos", "eef_rot6d") and len(value) != {"eef_pos": 3, "eef_rot6d": 6}[key]:
            raise ValueError(f"proprio {key} has the wrong shape")
        proprio[key] = value.tolist()
    last_action = observation["last_action"]
    if last_action is not None:
        last_action = validate_action(np.asarray(last_action, dtype=np.float32)).tolist()
    depth = {}
    for name in CAMERAS:
        value = np.asarray(observation["depth"][name], dtype=np.float32)
        if value.ndim != 2 or not np.isfinite(value).all():
            raise ValueError(f"depth {name} must be a finite [height, width] map")
        depth[name] = value
    cameras = {name: {"K": np.asarray(observation["cameras"][name]["K"], dtype=float).tolist(),
                      "camera_to_world": np.asarray(observation["cameras"][name]["camera_to_world"],
                                                    dtype=float).tolist(),
                      "width": int(observation["cameras"][name]["width"]),
                      "height": int(observation["cameras"][name]["height"])}
               for name in CAMERAS}
    header = {
        "type": "observation", "shapes": shapes,
        "depth_shapes": {name: list(depth[name].shape) for name in CAMERAS},
        "instruction": observation["instruction"], "step": int(observation["step"]),
        "proprio": proprio, "last_action": last_action, "cameras": cameras,
    }
    return header, [images[key] for key in IMAGE_KEYS] + [depth[name] for name in CAMERAS]


def read_observation(source, header):
    if header.get("type") != "observation":
        raise ValueError("expected observation")
    images = {key: read_array(source, header["shapes"][key]) for key in IMAGE_KEYS}
    depth = {name: read_array(source, header["depth_shapes"][name]) for name in CAMERAS}
    cameras = {name: {"K": np.asarray(cam["K"], dtype=np.float64),
                      "camera_to_world": np.asarray(cam["camera_to_world"], dtype=np.float64),
                      "width": cam["width"], "height": cam["height"]}
               for name, cam in header["cameras"].items()}
    return {
        "images": images, "depth": depth, "cameras": cameras,
        "instruction": header["instruction"], "step": header["step"],
        "proprio": {key: np.asarray(value, dtype=np.float32) for key, value in header["proprio"].items()},
        "last_action": (None if header["last_action"] is None
                        else np.asarray(header["last_action"], dtype=np.float32)),
    }
