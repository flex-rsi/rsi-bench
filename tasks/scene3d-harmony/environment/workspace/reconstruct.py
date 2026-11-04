"""Starter: an empty room box in flat colors, lit by one area light at the ceiling.

reconstruct(scene_dir, out_dir) writes out_dir/scene.blend with Blender ($BLENDER, see
instruction.md) running build_scene.py. The camera is a guess, the same for every photo: a
horizontal field of view of FOV_DEG with the principal point at the center, 1.5 m above the floor,
looking straight ahead (along +y; z is up). The room is a box 5 m wide and 2.8 m high that reaches
5 m in front of the camera and 1 m behind it. Each of its six faces (floor, ceiling, four walls:
the room's names, floor, ceiling and wall_*; no furniture at all) is one quad
facing into the room with a Principled BSDF whose base color is the mean color of the photo's
pixels that see that face, divided by what that face renders as when it is white (WHITE, measured
once with this light), so that the face renders about as bright as the photo there; a face the
photo does not see gets the mean color of the whole photo.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
FOV_DEG = 60.0
CAMERA_POSITION = (0.0, 0.0, 1.5)
BOX = ((-2.5, -1.0, 0.0), (2.5, 5.0, 2.8))       # min and max corner, z up
LIGHT = {"light_size": 3.0, "light_energy": 330.0, "light_y": 2.0}
# linear value of each face in the input view when its base color is white, with LIGHT
WHITE = {"floor": 0.86, "ceiling": 0.83, "wall_left": 0.83, "wall_right": 0.83, "wall_back": 0.855,
         "wall_front": 0.85}
FACES = {"floor": (2, 0), "ceiling": (2, 1), "wall_left": (0, 0), "wall_right": (0, 1),
         "wall_back": (1, 1), "wall_front": (1, 0)}     # name: (axis, 0 = min side / 1 = max side)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def blender() -> str:
    exe = os.environ.get("BLENDER") or shutil.which("blender")
    if not exe:
        raise RuntimeError("set BLENDER to the Blender executable (scorer/blender/blender)")
    return exe


def face_colors(photo: np.ndarray, cam: dict) -> dict:
    """Mean linear color of the photo's pixels whose ray from the camera hits each face first."""
    H, W = photo.shape[:2]
    j, i = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    # camera rays in the world: x right, y forward, z up
    d = np.stack([(j - cam["cx"]) / cam["fx"], np.ones_like(j), -(i - cam["cy"]) / cam["fy"]], -1)
    o = np.array(CAMERA_POSITION)
    lo, hi = np.array(BOX[0]), np.array(BOX[1])
    best = np.full((H, W), np.inf)
    face = np.full((H, W), -1)
    for k, (axis, side) in enumerate(FACES.values()):
        plane = (lo, hi)[side][axis]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (plane - o[axis]) / d[..., axis]
        better = (t > 0) & (t < best)
        best[better], face[better] = t[better], k
    lin = srgb_to_linear(photo.astype(np.float64) / 255.0)
    mean = lin.reshape(-1, 3).mean(0)
    out = {}
    for k, name in enumerate(FACES):
        sel = face == k
        c = lin[sel].mean(0) if sel.sum() > 100 else mean
        out[name] = [float(x) for x in np.clip(c / WHITE[name], 0.0, 1.0)]
    return out


def reconstruct(scene_dir: str, out_dir: str) -> None:
    scene_dir, out_dir = Path(scene_dir), Path(out_dir)
    photo = np.asarray(Image.open(scene_dir / "photo.png").convert("RGB"))
    H, W = photo.shape[:2]
    f = W / 2 / np.tan(np.radians(FOV_DEG / 2))
    cam = {"width": W, "height": H, "fx": f, "fy": f, "cx": W / 2, "cy": H / 2}
    spec = {"camera": cam, "colors": face_colors(photo, cam), "box": BOX,
            "camera_position": CAMERA_POSITION, "out": str(out_dir / "scene.blend"), **LIGHT}
    (out_dir / "spec.json").write_text(json.dumps(spec))
    subprocess.run([blender(), "-b", "--factory-startup", "-noaudio", "--python",
                    str(HERE / "build_scene.py"), "--", str(out_dir / "spec.json")],
                   check=True, timeout=300, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    (out_dir / "spec.json").unlink()
