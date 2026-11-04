"""Depth and mesh-index maps of triangle meshes seen from a pinhole camera, with a z-buffer on the
GPU (PyTorch only). The evaluator renders the images themselves with Blender; this module renders
only the geometry, for the depth metric and to tell which mesh each pixel shows.

Vertices are in the camera frame (OpenCV: x right, y down, z forward, meters or the scene's units);
fx, fy, cx, cy are in pixels with (0, 0) the top-left corner of the top-left pixel. A pixel shows
the nearest triangle that covers its center, from either side; triangles are clipped at z = NEAR.
"""

from __future__ import annotations

import numpy as np
import torch

NEAR = 1e-3
CHUNK = 1 << 23            # pixel-triangle candidates tested at once
_INF_KEY = torch.iinfo(torch.int64).max


def _clip(tri: torch.Tensor):
    """Clip camera-space triangles (T, 3, 3) at z = NEAR. Returns the sub-triangles and the index
    of the triangle each comes from."""
    dev = tri.device
    idx = torch.arange(len(tri), device=dev)
    behind = tri[..., 2] < NEAR
    nb = behind.sum(1)
    keep = nb == 0
    out_pos, out_idx = [tri[keep]], [idx[keep]]

    def rolled(sel, first):
        order = (first[:, None] + torch.arange(3, device=dev)) % 3
        return torch.gather(tri[sel], 1, order[..., None].expand(-1, -1, 3))

    def cut(p, q):
        t = ((NEAR - p[:, 2]) / (q[:, 2] - p[:, 2]))[:, None]
        return p + t * (q - p)

    one = nb == 1
    if one.any():  # a behind, b and c in front: the quad [ab, b, c, ca]
        v = rolled(one, behind[one].to(torch.int64).argmax(1))
        pab, pca = cut(v[:, 0], v[:, 1]), cut(v[:, 0], v[:, 2])
        out_pos += [torch.stack([pab, v[:, 1], v[:, 2]], 1), torch.stack([pab, v[:, 2], pca], 1)]
        out_idx += [idx[one], idx[one]]
    two = nb == 2
    if two.any():  # f in front, g and h behind: the triangle [f, fg, hf]
        v = rolled(two, (~behind[two]).to(torch.int64).argmax(1))
        out_pos.append(torch.stack([v[:, 0], cut(v[:, 0], v[:, 1]), cut(v[:, 0], v[:, 2])], 1))
        out_idx.append(idx[two])
    return torch.cat(out_pos), torch.cat(out_idx)


def _rasterize(xy: torch.Tensor, z: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Nearest triangle per pixel center as an int64 key: float32 depth bits << 32 | triangle."""
    dev = xy.device
    best = torch.full((H * W,), _INF_KEY, dtype=torch.int64, device=dev)
    if len(xy) == 0:
        return best
    jmin = torch.ceil(xy[..., 0].amin(1) - 0.5).clamp(0, W).to(torch.int64)
    jmax = torch.floor(xy[..., 0].amax(1) - 0.5).clamp(-1, W - 1).to(torch.int64)
    imin = torch.ceil(xy[..., 1].amin(1) - 0.5).clamp(0, H).to(torch.int64)
    imax = torch.floor(xy[..., 1].amax(1) - 0.5).clamp(-1, H - 1).to(torch.int64)
    nx = (jmax - jmin + 1).clamp(min=0)
    ny = (imax - imin + 1).clamp(min=0)
    area = nx * ny
    tris = torch.nonzero(area > 0).squeeze(1)
    if len(tris) == 0:
        return best
    area = area[tris]
    cum = torch.cumsum(area, 0)
    start = 0
    while start < len(tris):
        base = int(cum[start - 1]) if start else 0
        end = int(torch.searchsorted(cum, torch.tensor(base + CHUNK, device=dev), right=True))
        end = max(end, start + 1)
        t, a = tris[start:end], area[start:end]
        tri = torch.repeat_interleave(t, a)
        offs = (torch.arange(len(tri), device=dev)
                - torch.repeat_interleave(torch.cumsum(a, 0) - a, a))
        w = nx[tri]
        px = (jmin[tri] + offs % w).double() + 0.5
        py = (imin[tri] + torch.div(offs, w, rounding_mode="floor")).double() + 0.5
        q = xy[tri]
        x0, y0, x1, y1 = q[:, 0, 0], q[:, 0, 1], q[:, 1, 0], q[:, 1, 1]
        x2, y2 = q[:, 2, 0], q[:, 2, 1]
        den = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        l0 = ((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)) / den
        l1 = ((x0 - x2) * (py - y2) - (y0 - y2) * (px - x2)) / den
        l2 = 1.0 - l0 - l1
        inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        zt = z[tri]
        inv = l0 / zt[:, 0] + l1 / zt[:, 1] + l2 / zt[:, 2]
        inside &= inv > 0
        depth = (1.0 / inv[inside]).float()
        pix = (py[inside] - 0.5).long() * W + (px[inside] - 0.5).long()
        key = (depth.view(torch.int32).to(torch.int64) << 32) | tri[inside]
        best.scatter_reduce_(0, pix, key, reduce="amin")
        start = end
    return best


def render(meshes: list[tuple[np.ndarray, np.ndarray]], fx: float, fy: float, cx: float, cy: float,
           H: int, W: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Depth along the optical axis (H, W; inf where no surface) and the index into `meshes` of the
    mesh each pixel shows (H, W; -1 where none). meshes: (vertices (n, 3), faces (m, 3)) in the
    camera frame."""
    pos, tri_mesh = [], []
    for k, (v, f) in enumerate(meshes):
        if len(f) == 0:
            continue
        t = torch.as_tensor(np.asarray(v, np.float64), device=device)[
            torch.as_tensor(np.asarray(f, np.int64), device=device)]
        pos.append(t)
        tri_mesh.append(torch.full((len(t),), k, dtype=torch.int64, device=device))
    depth = torch.full((H, W), float("inf"), device=device)
    index = torch.full((H, W), -1, dtype=torch.int64, device=device)
    if not pos:
        return depth, index
    tri, owner = torch.cat(pos), torch.cat(tri_mesh)
    visible = (tri[..., 2] >= NEAR).any(1)
    sub, orig = _clip(tri[visible])
    owner = owner[visible][orig]
    z = sub[..., 2]
    xy = torch.stack([fx * sub[..., 0] / z + cx, fy * sub[..., 1] / z + cy], -1)
    area = ((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
            - (xy[:, 1, 1] - xy[:, 0, 1]) * (xy[:, 2, 0] - xy[:, 0, 0]))
    keep = area.abs() > 1e-12
    xy, z, owner = xy[keep], z[keep], owner[keep]
    best = _rasterize(xy, z, H, W)
    hit = best != _INF_KEY
    k = best[hit] & 0xFFFFFFFF
    depth.view(-1)[hit] = (best[hit] >> 32).to(torch.int32).view(torch.float32)
    index.view(-1)[hit] = owner[k]
    return depth, index
