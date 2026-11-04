"""Render a scene read by gltf.py from a calibrated camera, with a z-buffer on the GPU (PyTorch only).

A pixel shows the base color of the nearest surface: the base-color texture (sRGB, sampled
bilinearly with the sampler's wrap modes) times baseColorFactor times COLOR_0, in linear RGB,
without lighting. Pixels that see no surface are black. Faces of single-sided materials are culled
when seen from behind (glTF front faces are counter-clockwise); triangles are clipped at a near
plane NEAR meters in front of the camera. A sample is covered when it is inside a triangle. Colors
are rendered at SSAA x SSAA samples per pixel, averaged in linear RGB and encoded to sRGB; the
node a pixel shows is the one most of its samples show.

Cameras are those of transforms.json: a 4 x 4 camera-to-world matrix with the camera's x axis
pointing right, y up and z backwards (OpenGL / nerfstudio), and a pinhole with focal lengths fx, fy
and principal point cx, cy in pixels, with (0, 0) the top-left corner of the top-left pixel.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from gltf import CLAMP, MIRROR, Scene

SSAA = 2
NEAR = 0.01                # meters
CHUNK = 1 << 23            # pixel-triangle candidates tested at once
_INF_KEY = torch.iinfo(torch.int64).max
GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def srgb_to_linear(c: torch.Tensor) -> torch.Tensor:
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: torch.Tensor) -> torch.Tensor:
    c = c.clamp(0.0, 1.0)
    return torch.where(c <= 0.0031308, c * 12.92, 1.055 * c.clamp(min=1e-12) ** (1 / 2.4) - 0.055)


def _wrap(t: torch.Tensor, mode: int) -> torch.Tensor:
    if mode == CLAMP:
        return t.clamp(0.0, 1.0)
    if mode == MIRROR:
        t = torch.remainder(t, 2.0)
        return 1.0 - (t - 1.0).abs()
    return t - torch.floor(t)


def world_to_camera(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation into camera coordinates with x right, y down and z forward."""
    c2w = np.asarray(c2w, np.float64)
    rot_cv = c2w[:3, :3] @ GL_TO_CV
    rot = rot_cv.T
    return rot, -rot @ c2w[:3, 3]


class Renderer:
    def __init__(self, scene: Scene, device: torch.device):
        self.device = device
        self.node_ids = sorted({p.node for p in scene.primitives if len(p.faces)})
        self.node_names = {}
        for p in scene.primitives:
            self.node_names.setdefault(p.node, p.name)
        self.empty = not self.node_ids
        if self.empty:
            return
        node_index = {n: i for i, n in enumerate(self.node_ids)}
        pos, faces, uv, col, mat, textured, node = [], [], [], [], [], [], []
        base = 0
        for p in scene.primitives:
            if not len(p.faces):
                continue
            pos.append(p.positions)
            faces.append(p.faces + base)
            uv.append(p.uv if p.uv is not None else np.zeros((len(p.positions), 2), np.float32))
            col.append(p.color if p.color is not None else np.ones((len(p.positions), 3), np.float32))
            mat.append(np.full(len(p.faces), p.material))
            node.append(np.full(len(p.faces), node_index[p.node]))
            # a textured material on a primitive without texture coordinates shows its factor only
            has_texture = p.uv is not None and scene.images[scene.materials[p.material].image] is not None
            textured.append(np.full(len(p.faces), has_texture))
            base += len(p.positions)
        dev = device
        mat = np.concatenate(mat)
        self.positions = torch.as_tensor(np.concatenate(pos), dtype=torch.float64, device=dev)
        self.faces = torch.as_tensor(np.concatenate(faces), dtype=torch.int64, device=dev)
        self.uv = torch.as_tensor(np.concatenate(uv), dtype=torch.float32, device=dev)
        self.color = torch.as_tensor(np.concatenate(col), dtype=torch.float32, device=dev)
        self.material = torch.as_tensor(mat, dtype=torch.int64, device=dev)
        self.node = torch.as_tensor(np.concatenate(node), dtype=torch.int64, device=dev)
        self.textured = torch.as_tensor(np.concatenate(textured), dtype=torch.bool, device=dev)
        self.single_sided = torch.as_tensor(
            np.array([not m.double_sided for m in scene.materials])[mat], device=dev)
        self.factor = torch.as_tensor(np.stack([m.factor for m in scene.materials]),
                                      dtype=torch.float32, device=dev)
        self.mat_image = [m.image for m in scene.materials]
        self.mat_wrap = [m.wrap for m in scene.materials]
        self.textures = {}
        for i, img in enumerate(scene.images):
            if img is not None:
                t = torch.as_tensor(img, device=dev).permute(2, 0, 1).float() / 255.0
                self.textures[i] = srgb_to_linear(t)[None].contiguous()

    # ----------------------------------------------------------------------- geometry
    @staticmethod
    def _clip(tri: torch.Tensor):
        """Clip camera-space triangles (T, 3, 3) at z = NEAR. Returns sub-triangles, the index of
        the triangle each comes from, and its vertices' barycentric coordinates in that triangle."""
        near = NEAR
        dev = tri.device
        idx = torch.arange(len(tri), device=dev)
        eye = torch.eye(3, dtype=tri.dtype, device=dev)
        behind = tri[..., 2] < near
        nb = behind.sum(1)
        out_pos, out_idx, out_bary = [], [], []
        keep = nb == 0
        out_pos.append(tri[keep])
        out_idx.append(idx[keep])
        out_bary.append(eye.expand(int(keep.sum()), 3, 3))

        def rolled(sel, first):
            order = (first[:, None] + torch.arange(3, device=dev)) % 3
            v = torch.gather(tri[sel], 1, order[..., None].expand(-1, -1, 3))
            return v, eye[order]

        def cut(p, q, bp, bq):
            t = ((near - p[:, 2]) / (q[:, 2] - p[:, 2]))[:, None]
            return p + t * (q - p), bp + t * (bq - bp)

        one = nb == 1
        if one.any():  # a behind, b and c in front: the quad [ab, b, c, ca]
            v, b = rolled(one, behind[one].to(torch.int64).argmax(1))
            pab, bab = cut(v[:, 0], v[:, 1], b[:, 0], b[:, 1])
            pca, bca = cut(v[:, 0], v[:, 2], b[:, 0], b[:, 2])
            out_pos += [torch.stack([pab, v[:, 1], v[:, 2]], 1), torch.stack([pab, v[:, 2], pca], 1)]
            out_bary += [torch.stack([bab, b[:, 1], b[:, 2]], 1), torch.stack([bab, b[:, 2], bca], 1)]
            out_idx += [idx[one], idx[one]]
        two = nb == 2
        if two.any():  # f in front, g and h behind: the triangle [f, fg, hf]
            v, b = rolled(two, (~behind[two]).to(torch.int64).argmax(1))
            pfg, bfg = cut(v[:, 0], v[:, 1], b[:, 0], b[:, 1])
            phf, bhf = cut(v[:, 0], v[:, 2], b[:, 0], b[:, 2])
            out_pos.append(torch.stack([v[:, 0], pfg, phf], 1))
            out_bary.append(torch.stack([b[:, 0], bfg, bhf], 1))
            out_idx.append(idx[two])
        return torch.cat(out_pos), torch.cat(out_idx), torch.cat(out_bary)

    @staticmethod
    def _edges(xy: torch.Tensor, px: torch.Tensor, py: torch.Tensor):
        """Screen-space barycentric coordinates of pixel centers (px, py) in triangles xy (n, 3, 2)."""
        x0, y0 = xy[:, 0, 0], xy[:, 0, 1]
        x1, y1 = xy[:, 1, 0], xy[:, 1, 1]
        x2, y2 = xy[:, 2, 0], xy[:, 2, 1]
        area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        cx, cy = px + 0.5, py + 0.5
        l0 = ((x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)) / area
        l1 = ((x0 - x2) * (cy - y2) - (y0 - y2) * (cx - x2)) / area
        return l0, l1, 1.0 - l0 - l1

    @staticmethod
    def rasterize(xy: torch.Tensor, z: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Nearest triangle per pixel as an int64 key: float32 depth bits << 32 | triangle."""
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
            t = tris[start:end]
            a = area[start:end]
            tri = torch.repeat_interleave(t, a)
            offs = torch.arange(len(tri), device=dev) - torch.repeat_interleave(torch.cumsum(a, 0) - a, a)
            w = nx[tri]
            px = jmin[tri] + offs % w
            py = imin[tri] + torch.div(offs, w, rounding_mode="floor")
            l0, l1, l2 = Renderer._edges(xy[tri], px.double(), py.double())
            inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
            zt = z[tri]
            inv = l0 / zt[:, 0] + l1 / zt[:, 1] + l2 / zt[:, 2]
            inside &= inv > 0
            depth = (1.0 / inv[inside]).float()
            key = (depth.view(torch.int32).to(torch.int64) << 32) | tri[inside]
            best.scatter_reduce_(0, (py * W + px)[inside], key, reduce="amin")
            start = end
        return best

    @staticmethod
    def project(positions: torch.Tensor, faces: torch.Tensor, c2w: np.ndarray, fx: float, fy: float,
                cx: float, cy: float, cull: torch.Tensor | None = None):
        """Clipped screen-space triangles of a mesh: xy (n, 3, 2), camera depth z (n, 3), the
        original triangle of each and the barycentric coordinates of its corners in it. Triangles
        whose `cull` flag is set are dropped when seen from behind."""
        dev = positions.device
        rot, trans = world_to_camera(c2w)
        v = positions @ torch.as_tensor(rot.T, device=dev) + torch.as_tensor(trans, device=dev)
        tri = v[faces]
        visible = (tri[..., 2] >= NEAR).any(1)
        sub, orig, bary = Renderer._clip(tri[visible])
        orig = torch.nonzero(visible).squeeze(1)[orig]
        z = sub[..., 2]
        xy = torch.stack([fx * sub[..., 0] / z + cx, fy * sub[..., 1] / z + cy], -1)
        area = ((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
                - (xy[:, 1, 1] - xy[:, 0, 1]) * (xy[:, 2, 0] - xy[:, 0, 0]))
        keep = area.abs() > 1e-12
        if cull is not None:
            # front faces are counter-clockwise seen from the camera: negative area with y down
            keep &= (area < 0) | ~cull[orig]
        return xy[keep], z[keep], orig[keep], bary[keep]

    # ----------------------------------------------------------------------- rendering
    def render(self, c2w: np.ndarray, fx: float, fy: float, cx: float, cy: float, H: int, W: int):
        """sRGB image (H, W, 3) in [0, 1], the index into self.node_ids of the node each pixel
        shows (H, W; -1 where none) and the covered share of each pixel (H, W)."""
        dev = self.device
        if self.empty:
            return (torch.zeros(H, W, 3, device=dev), torch.full((H, W), -1, dtype=torch.int64, device=dev),
                    torch.zeros(H, W, device=dev))
        s = SSAA
        Hs, Ws = H * s, W * s
        sub_xy, sub_z, orig, bary = self.project(self.positions, self.faces, c2w, fx * s, fy * s,
                                                 cx * s, cy * s, self.single_sided)
        best = self.rasterize(sub_xy, sub_z, Hs, Ws)
        hit = best != _INF_KEY
        pix = torch.nonzero(hit).squeeze(1)
        k = best[pix] & 0xFFFFFFFF
        px, py = (pix % Ws).double(), torch.div(pix, Ws, rounding_mode="floor").double()
        l0, l1, l2 = self._edges(sub_xy[k], px, py)
        zk = sub_z[k]
        w = torch.stack([l0 / zk[:, 0], l1 / zk[:, 1], l2 / zk[:, 2]], 1).clamp(min=0)
        w = w / w.sum(1, keepdim=True).clamp(min=1e-30)
        g = torch.einsum("ni,nij->nj", w, bary[k]).float()       # barycentrics in the original
        f = orig[k]
        vid = self.faces[f]                                       # (n, 3)
        col = (g[..., None] * self.color[vid]).sum(1)
        m = self.material[f]
        rgb = self.factor[m] * col
        tex = self.textured[f]
        if tex.any():
            uv = (g[..., None] * self.uv[vid]).sum(1)
            for mat_id in torch.unique(m[tex]).tolist():
                sel = tex & (m == mat_id)
                image = self.mat_image[mat_id]
                if image not in self.textures:
                    continue
                ws, wt = self.mat_wrap[mat_id]
                u, t = _wrap(uv[sel, 0], ws), _wrap(uv[sel, 1], wt)
                grid = torch.stack([u * 2 - 1, t * 2 - 1], -1)[None, None].to(torch.float32)
                smp = F.grid_sample(self.textures[image], grid, mode="bilinear",
                                    padding_mode="border", align_corners=False)[0, :, 0].T
                rgb[sel] = rgb[sel] * smp
        img = torch.zeros(Hs * Ws, 3, device=dev)
        img[pix] = rgb
        node = torch.full((Hs * Ws,), -1, dtype=torch.int64, device=dev)
        node[pix] = self.node[f]
        img = img.view(H, s, W, s, 3).mean((1, 3))
        cov = hit.float().view(H, s, W, s).mean((1, 3))
        node = node.view(H, s, W, s).permute(0, 2, 1, 3).reshape(H, W, s * s)
        node = torch.mode(node, dim=-1).values
        return linear_to_srgb(img), node, cov
