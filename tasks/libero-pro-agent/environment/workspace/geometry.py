"""Camera geometry for the public observation (numpy only).

Public image and depth arrays are the upright camera view flipped vertically: row 0 of a public
array is the bottom row of the upright view, columns are unchanged. `observation["cameras"][name]`
gives K (pinhole intrinsics of the upright view) and camera_to_world (4x4, OpenCV axes: x right,
y down, z forward). Depth is the metric distance along the optical axis, in the public
orientation. The agentview calibration is measured once at the start of the episode.
"""
import numpy as np


def upright_rgb(image):
    """Public float32 [1, 3, H, W] image -> upright uint8 [H, W, 3], e.g. to show the model."""
    rgb = np.clip(np.asarray(image)[0].transpose(1, 2, 0) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb[::-1])


def project(points_world, camera):
    """World points [N, 3] -> (rows, cols) in the public array and camera-frame depth z [N]."""
    points = np.atleast_2d(np.asarray(points_world, dtype=np.float64))
    world_to_camera = np.linalg.inv(np.asarray(camera["camera_to_world"], dtype=np.float64))
    pc = (world_to_camera @ np.c_[points, np.ones(len(points))].T)[:3]
    uv = (np.asarray(camera["K"], dtype=np.float64) @ pc)[:2] / pc[2]
    rows = camera["height"] - 1 - uv[1]
    return rows, uv[0], pc[2]


def unproject(rows, cols, depth_map, camera):
    """Public-array pixels and the matching depth map -> world points [N, 3]."""
    rows = np.atleast_1d(np.asarray(rows)).astype(int)
    cols = np.atleast_1d(np.asarray(cols)).astype(int)
    z = np.asarray(depth_map, dtype=np.float64)[rows, cols]
    K = np.asarray(camera["K"], dtype=np.float64)
    v = camera["height"] - 1 - rows
    x = (cols - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pc = np.vstack([x, y, z, np.ones_like(z)])
    return (np.asarray(camera["camera_to_world"], dtype=np.float64) @ pc)[:3].T


def upright_pixel_to_public(row_upright, col, camera):
    """A pixel picked in the upright view (e.g. by the model) -> (row, col) of the public array."""
    return camera["height"] - 1 - int(round(row_upright)), int(round(col))
