"""Place a solution's scene in the ground truth's world frame from its cameras (v3).

A solution reconstructs in a frame and scale of its own and writes, with scene.glb, cameras.json:
the camera-to-world matrix it estimated for every input frame, in the same convention as the
ground truth's transforms.json (the camera's x axis right, y up, z backwards; OpenGL / nerfstudio):

    {"frames": [{"file_path": "images/frame_00000.jpg", "transform_matrix": [[...], ...]}, ...]}

The similarity x_gt = s R x + t that maps it onto the ground truth:
    R  the rotation that best maps the solution's camera orientations onto the true ones, the
       chordal L2 mean (the SVD of sum_i R_gt,i R_i^T): it needs no spread of the trajectory
    s, t  then least squares between the camera centers; when the centers do not spread (all at
       one point), s = 1 and the trajectories' centroids coincide
Reported: camera_rot_err (degrees, the mean angle between R R_i and R_gt,i), camera_ate (cm, the
root mean square distance of the mapped centers from the true ones) and camera_scale (s).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MIN_SPREAD = 1e-9        # mean squared distance of the centers from their centroid, solution units


def read_cameras(path: Path, frames: list[dict]) -> np.ndarray:
    """(n, 4, 4) camera-to-world matrices of cameras.json in the order of `frames` (the ground
    truth's transforms.json frames); ValueError when one is missing or not a rigid transform."""
    if not path.is_file():
        raise ValueError("no cameras.json")
    try:
        doc = json.loads(path.read_text())
        given = {str(f["file_path"]): np.asarray(f["transform_matrix"], np.float64)
                 for f in doc["frames"]}
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"cameras.json does not parse: {exc}") from None
    out = []
    for fr in frames:
        m = given.get(fr["file_path"])
        if m is None:
            raise ValueError(f"cameras.json has no camera for {fr['file_path']}")
        if m.shape != (4, 4) or not np.isfinite(m).all():
            raise ValueError(f"the camera of {fr['file_path']} is not a finite 4 x 4 matrix")
        rot = m[:3, :3]
        if (np.abs(rot.T @ rot - np.eye(3)).max() > 1e-3 or np.linalg.det(rot) < 0
                or np.abs(m[3] - (0, 0, 0, 1)).max() > 1e-6):
            raise ValueError(f"the camera of {fr['file_path']} is not a rotation and a translation")
        out.append(m)
    return np.stack(out)


def similarity(sol: np.ndarray, gt: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, dict]:
    """s, R, t with x_gt = s R x_sol + t, from (n, 4, 4) camera-to-world matrices of the solution
    and of the ground truth; and the residuals."""
    u, _, vt = np.linalg.svd(np.einsum("nij,nkj->ik", gt[:, :3, :3], sol[:, :3, :3]))
    rot = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    cs, cg = sol[:, :3, 3], gt[:, :3, 3]
    a = (cs - cs.mean(0)) @ rot.T
    b = cg - cg.mean(0)
    spread = float((a * a).sum(1).mean())
    scale = float((a * b).sum() / (a * a).sum()) if spread > MIN_SPREAD else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    trans = cg.mean(0) - scale * rot @ cs.mean(0)
    mapped = scale * cs @ rot.T + trans
    rel = np.einsum("nji,njk->nik", gt[:, :3, :3], np.einsum("ij,njk->nik", rot, sol[:, :3, :3]))
    cos = np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0)
    return scale, rot, trans, {
        "camera_rot_err": float(np.degrees(np.arccos(cos)).mean()),
        "camera_ate": 100.0 * float(np.sqrt(((mapped - cg) ** 2).sum(1).mean())),
        "camera_scale": scale,
    }
