"""Starter: an empty room box in the frames' mean color, with every camera at its center.

No camera is estimated: every frame's camera stands at the origin, level, looking along +y (z up).
The evaluator maps the cameras onto the true ones; with no spread of the centers the scale stays
1 (meters), so the box lands around the trajectory's centroid, upright. Its six faces are
single-sided quads that face inward, colored with the mean color of the frames. No objects:
everything is one background node.
"""

import json
import os

import numpy as np
from PIL import Image

from glb import srgb_to_linear, write_glb

HALF = 3.0        # meters: half the box's width and depth
FLOOR = -1.4      # meters: the floor, below the cameras
CEILING = 1.4     # meters: the ceiling, above the cameras
FRAMES = 32       # frames averaged for the color
LEVEL = [[1.0, 0.0, 0.0, 0.0],    # camera to world: the camera's x right, y up (world z), z backwards
         [0.0, 0.0, -1.0, 0.0],   # (world -y), so that it looks along world +y
         [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]]


def reconstruct(scene_dir: str, out_dir: str) -> None:
    names = sorted(os.listdir(os.path.join(scene_dir, "images")))
    with open(os.path.join(out_dir, "cameras.json"), "w") as f:
        json.dump({"frames": [{"file_path": f"images/{n}", "transform_matrix": LEVEL} for n in names]}, f)
    pick = np.linspace(0, len(names) - 1, min(FRAMES, len(names))).round().astype(int)
    color = np.mean([np.asarray(Image.open(os.path.join(scene_dir, "images", names[i]))
                                .convert("RGB").resize((64, 64)), np.float64).reshape(-1, 3).mean(0)
                     for i in pick], axis=0) / 255.0

    x0, y0, z0 = -HALF, -HALF, FLOOR
    x1, y1, z1 = HALF, HALF, CEILING
    corners = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    # counter-clockwise seen from inside the box
    quads = [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]]
    faces = np.array([[q[0], q[1], q[2]] for q in quads] + [[q[0], q[2], q[3]] for q in quads])
    write_glb(os.path.join(out_dir, "scene.glb"),
              [{"name": "room", "positions": corners, "faces": faces,
                "base_color": srgb_to_linear(color)}])
