"""Whether a reconstructed piece of furniture is a whole 3D object, judged by a vision-language
model (Qwen3-VL-8B-Instruct, in the dev data's scorer/) against the shortcuts built from the
object's own front surface.

For an object (its mesh objects together, in the camera frame of the scene's camera):

- its front surface: the depth of the object alone, seen from the scene's camera at the render
  size, back-projected; three shortcuts are built from it:
      sheet   that surface as a grid mesh, cut where neighboring depths differ by more than JUMP
      prism   the sheet extruded along the camera rays to the object's farthest depth (at least
              PRISM_MIN behind the surface)
      box     the object's box, aligned with up and with the principal axes of the object's
              vertices in the horizontal plane
- every shape is shown as a 2 x 2 grid of gray clay renders (VIEW pixels each, both sides of a
  face shaded alike) from four directions about the vertical axis through the center of its own
  box: the camera's direction, then 90, 180 and 270 degrees, at the camera's height, the shape
  filling the view;
- the model sees the photo's crop around the object, then two grids (the object and one shortcut,
  in both orders) and answers A or B to "which looks like a real <label> from all four
  directions"; P(object) = (P(A | object first) + P(B | object second)) / 2, from the logits of
  the two answers.
- score: the lowest P(object) over the three shortcuts (the box is skipped for box-shaped
  furniture: cabinets, drawers, wardrobes, shelves, ...); a shortcut against itself scores 0.5.
  realness = clip((score - 0.5) / 0.4, 0, 1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import raster

QWEN_DIR = "qwen3-vl-8b-instruct"
PYLIB_DIR = "pylib"
VIEW = 384
FOV = 40.0
JUMP = 0.05            # relative depth difference that cuts the sheet
PRISM_MIN = 0.03       # m (scene units): the prism is at least this thick
TIE, SURE = 0.5, 0.9   # realness 0 at a tie with a shortcut, 1 at P >= SURE
PROMPT = ("The first image is a crop of a photo of a room, centered on a {label}. The second image "
          "shows 3D model A and the third image 3D model B of that {label}, each in plain gray clay "
          "from four directions around it: from the photo's side (top left), from its right (top "
          "right), from behind (bottom left) and from its left (bottom right). Which model looks like "
          "a real {label} from all four directions, with the sides and back such a {label} has? "
          "Answer A or B.")


# --------------------------------------------------------------------------- shortcuts
def grid_mesh(points: np.ndarray, mask: np.ndarray, depth: np.ndarray):
    """Vertices and triangles between neighboring mask pixels whose depths differ by < JUMP."""
    idx = -np.ones(mask.shape, np.int64)
    idx[mask] = np.arange(mask.sum())
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    za, zb, zc, zd = depth[:-1, :-1], depth[:-1, 1:], depth[1:, :-1], depth[1:, 1:]

    def ok(*zs):
        z = np.stack(zs)
        return (z.max(0) - z.min(0)) < JUMP * z.min(0)
    t1 = (a >= 0) & (b >= 0) & (c >= 0) & ok(za, zb, zc)
    t2 = (b >= 0) & (d >= 0) & (c >= 0) & ok(zb, zd, zc)
    f = np.concatenate([np.stack([a[t1], b[t1], c[t1]], 1), np.stack([b[t2], d[t2], c[t2]], 1)])
    return points[mask], f


def extrude(v: np.ndarray, f: np.ndarray, z_back: np.ndarray):
    """The sheet (v, f) closed with its copy moved along the camera rays to depth z_back."""
    n = len(v)
    vb = v * (z_back / v[:, 2])[:, None]
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    _, inv, cnt = np.unique(np.sort(e, 1), axis=0, return_inverse=True, return_counts=True)
    border = e[cnt[inv.ravel()] == 1]
    sides = np.concatenate([np.stack([border[:, 0], border[:, 1], border[:, 1] + n], 1),
                            np.stack([border[:, 0], border[:, 1] + n, border[:, 0] + n], 1)])
    return np.concatenate([v, vb]), np.concatenate([f, f[:, ::-1] + n, sides])


def box_mesh(center: np.ndarray, axes: np.ndarray, half: np.ndarray):
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    v = center + (corners * half) @ axes
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return v, f


def shortcuts(v: np.ndarray, f: np.ndarray, k: tuple, size: tuple, up: np.ndarray,
              device) -> dict:
    """sheet, prism and box of an object (v, f: camera frame; k: intrinsics at size (W, H))."""
    W, H = size
    depth, _ = raster.render([(v, f)], *k, H, W, device)
    depth = depth.cpu().numpy()
    mask = np.isfinite(depth)
    out = {}
    used = v[np.unique(f)]
    if mask.sum() >= 3:
        d = np.where(mask, depth, 1.0)
        j, i = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        pts = np.stack([(j - k[2]) / k[0] * d, (i - k[3]) / k[1] * d, d], -1)
        sv, sf = grid_mesh(pts, mask, d)
        if len(sf):
            out["sheet"] = (sv, sf)
            out["prism"] = extrude(sv, sf, np.maximum(used[:, 2].max(), sv[:, 2] + PRISM_MIN))
    u = up / np.linalg.norm(up)
    flat = used - np.outer(used @ u, u)
    flat = flat - flat.mean(0)
    _, vec = np.linalg.eigh(flat.T @ flat)
    a1 = vec[:, -1] - (vec[:, -1] @ u) * u
    a1 /= np.linalg.norm(a1) + 1e-12
    axes = np.stack([a1, np.cross(u, a1), u])
    q = used @ axes.T
    out["box"] = box_mesh(((q.max(0) + q.min(0)) / 2) @ axes, axes, (q.max(0) - q.min(0)) / 2)
    return out


# --------------------------------------------------------------------------- clay views
def _triangle_ids(v, f, fpx, H, W, device):
    t = torch.as_tensor(v, device=device, dtype=torch.float64)[torch.as_tensor(f, device=device)]
    owner = torch.arange(len(t), device=device)
    vis = (t[..., 2] >= raster.NEAR).any(1)
    sub, orig = raster._clip(t[vis])
    owner = owner[vis][orig]
    z = sub[..., 2]
    xy = torch.stack([fpx * sub[..., 0] / z + W / 2, fpx * sub[..., 1] / z + H / 2], -1)
    area = ((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
            - (xy[:, 1, 1] - xy[:, 0, 1]) * (xy[:, 2, 0] - xy[:, 0, 0]))
    keep = area.abs() > 1e-12
    out = torch.full((H * W,), -1, dtype=torch.int64, device=device)
    if keep.any():
        best = raster._rasterize(xy[keep], z[keep], H, W)
        hit = best != raster._INF_KEY
        out[hit] = owner[keep][best[hit] & 0xFFFFFFFF]
    return out.view(H, W).cpu().numpy()


def clay(v: np.ndarray, f: np.ndarray, device) -> Image.Image:
    """Gray clay view (VIEW x VIEW) of a mesh in a camera frame looking down +z."""
    fpx = VIEW / 2 / np.tan(np.radians(FOV / 2))
    tid = _triangle_ids(v, f, fpx, VIEW, VIEW, device)
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    j, i = np.meshgrid(np.arange(VIEW) + 0.5, np.arange(VIEW) + 0.5)
    ray = np.stack([(j - VIEW / 2) / fpx, (i - VIEW / 2) / fpx, np.ones_like(j)], -1)
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
    img = np.ones((VIEW, VIEW))
    hit = tid >= 0
    nn, rr = n[tid[hit]], ray[hit]
    nn = np.where(((nn * rr).sum(-1) > 0)[:, None], -nn, nn)
    light = np.array([-0.4, -0.8, -0.5])
    light /= np.linalg.norm(light)
    img[hit] = 0.15 + 0.35 * np.clip(-(nn * rr).sum(-1), 0, 1) + 0.5 * np.clip(nn @ light, 0, 1)
    return Image.fromarray((img.clip(0, 1) * 255).astype(np.uint8)).convert("RGB")


def turntable(v: np.ndarray, f: np.ndarray, up: np.ndarray, device) -> Image.Image:
    """2 x 2 clay views about `up` through the center of the shape's box (camera frame)."""
    p = v[np.unique(f)]
    center = (p.max(0) + p.min(0)) / 2
    radius = max(float(np.linalg.norm(p.max(0) - p.min(0)) / 2), 1e-6)
    k = up / np.linalg.norm(up)
    d0 = center / max(np.linalg.norm(center), 1e-9)       # from the camera: its elevation stays
    dist = 1.05 * radius / np.sin(np.radians(FOV / 2))
    grid = Image.new("RGB", (2 * VIEW + 6, 2 * VIEW + 6), (160, 160, 160))
    for n, deg in enumerate((0, 90, 180, 270)):
        a = np.radians(deg)
        d = d0 * np.cos(a) + np.cross(k, d0) * np.sin(a) + k * (k @ d0) * (1 - np.cos(a))
        z = d / np.linalg.norm(d)
        y = -(k - (k @ z) * z)
        y /= max(np.linalg.norm(y), 1e-9)
        x = np.cross(y, z)
        view = clay((v - (center - dist * z)) @ np.stack([x, y, z]).T, f, device)
        grid.paste(view, ((n % 2) * (VIEW + 6), (n // 2) * (VIEW + 6)))
    return grid


def _lettered(grid: Image.Image, letter: str) -> Image.Image:
    g = grid.copy()
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 56)
    except OSError:
        font = ImageFont.load_default()
    ImageDraw.Draw(g).text((14, 8), letter, fill=(200, 30, 30), font=font)
    return g


# --------------------------------------------------------------------------- the model
class Judge:
    def __init__(self, scorer: Path, device):
        sys.path.insert(0, str(scorer / PYLIB_DIR))
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        path = scorer / QWEN_DIR
        self.proc = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, dtype=torch.bfloat16, local_files_only=True).to(device).eval()
        tok = self.proc.tokenizer
        self.a = tok.encode("A", add_special_tokens=False)[0]
        self.b = tok.encode("B", add_special_tokens=False)[0]
        self.device = device

    @torch.no_grad()
    def _p_first(self, crop, g1, g2, text) -> float:
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": crop}, {"type": "image", "image": _lettered(g1, "A")},
            {"type": "image", "image": _lettered(g2, "B")}, {"type": "text", "text": text}]}]
        inputs = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits[0, -1].float()
        return float(torch.softmax(torch.stack([logits[self.a], logits[self.b]]), 0)[0])

    def better(self, crop, g_obj, g_other, text) -> float:
        return (self._p_first(crop, g_obj, g_other, text)
                + 1.0 - self._p_first(crop, g_other, g_obj, text)) / 2

    def realness(self, v, f, k, size, up, crop: Image.Image, label: str, boxy: bool) -> dict:
        """{"vs": {shortcut: P(object)}, "score", "realness", "views": the object's grid}."""
        g_obj = turntable(v, f, up, self.device)
        text = PROMPT.format(label=label)
        vs = {}
        for name, (sv, sf) in shortcuts(v, f, k, size, up, self.device).items():
            if name == "box" and boxy:
                continue
            vs[name] = self.better(crop, g_obj, turntable(sv, sf, up, self.device), text)
        score = min(vs.values()) if vs else 0.0
        return {"vs": vs, "score": score,
                "realness": float(np.clip((score - TIE) / (SURE - TIE), 0.0, 1.0)), "views": g_obj}
