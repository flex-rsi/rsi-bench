"""Metrics of CoDimRecon (arXiv 2609.36024, Appendix C) other than rigid-body stability (physics.py).

Rendering, per input frame, on the full unmasked sRGB images in [0, 1]:
- PSNR: 10 log10(1 / MSE) over all pixels and channels.
- SSIM: Wang et al. (2004) with an 11 x 11 Gaussian window (sigma 1.5), K1 0.01, K2 0.03, per
  channel over the window positions inside the image, averaged.
- LPIPS: Zhang et al. (2018), AlexNet with the v0.1 linear layers, on images scaled to [-1, 1].

Geometry, between the predicted surface P and the ground-truth mesh Q:
- SAMPLES surface-area-weighted points from each (numpy's default generator, seed 0), with the
  normal of the face each point lies on.
- CD: the unsquared symmetric mean nearest-neighbor distance, (mean_p d(p, Q) + mean_q d(q, P)) / 2,
  in centimeters.
- F1: 2 Prec Rec / (Prec + Rec) in percent, Prec and Rec the shares of predicted and ground-truth
  points within F1_TAU (5 cm) of the other set.
- NC: the mean absolute cosine between the normal at each point and the normal at its nearest
  neighbor in the other set, over both directions, in percent.

Latent Similarity (BVB): 64 frames sampled uniformly from the input sequence and the same 64 views
rendered, each clip encoded by the frozen V-JEPA 2.1 ViT-G/16 at 384 x 384 (bf16 autocast; frames resized
to 384 on the short side, center-cropped, ImageNet normalization). Its last-layer tokens lie on a
T x H x W grid (32 x 24 x 24, one temporal step per two frames). Motion: the cosine of the two
clips' mean tokens. Layout: tokens averaged over time per patch, then the mean over patches of the
cosine of corresponding patches. Both in percent.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree

ALEXNET_FILE = "alexnet-owt-7be5be79.pth"
LPIPS_FILE = "lpips-v0.1-alex.pth"
VJEPA_DIR = "vjepa2_1"                       # the encoder's code (facebookresearch/vjepa2, MIT)
VJEPA_FILE = "vjepa2_1_vitG_384_encoder.pt"  # its weights, bf16

SAMPLES = 500_000
F1_TAU = 0.05
CD_EMPTY_CM = 100.0     # the CD of a scene without a surface
LS_FRAMES = 64
LS_SIZE = 384
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- rendering
def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """a, b: (H, W, 3) in [0, 1]."""
    mse = float(((a - b) ** 2).mean())
    return 10.0 * float(np.log10(1.0 / max(mse, 1e-10)))


def _gaussian(device: torch.device) -> torch.Tensor:
    x = torch.arange(11, dtype=torch.float32, device=device) - 5
    g = torch.exp(-x ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """a, b: (H, W, 3) in [0, 1]."""
    w = _gaussian(a.device)
    x, y = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    mx, my = F.conv2d(x, w, groups=3), F.conv2d(y, w, groups=3)
    sxx = F.conv2d(x * x, w, groups=3) - mx ** 2
    syy = F.conv2d(y * y, w, groups=3) - my ** 2
    sxy = F.conv2d(x * y, w, groups=3) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return float(m.mean())


class Lpips:
    """LPIPS v0.1 with AlexNet (lpips.LPIPS(net="alex"))."""

    _SHIFT = (-0.030, -0.088, -0.188)
    _SCALE = (0.458, 0.448, 0.450)

    def __init__(self, weights: Path, device: torch.device):
        a = torch.load(weights / ALEXNET_FILE, map_location=device, weights_only=True)
        self.conv = [(a[f"features.{i}.weight"], a[f"features.{i}.bias"]) for i in (0, 3, 6, 8, 10)]
        lw = torch.load(weights / LPIPS_FILE, map_location=device, weights_only=True)
        self.lin = [lw[f"lin{i}.model.1.weight"] for i in range(5)]
        self.device = device

    def _features(self, x: torch.Tensor) -> list[torch.Tensor]:
        c = self.conv
        out = []
        x = F.relu(F.conv2d(x, *c[0], stride=4, padding=2))
        out.append(x)
        x = F.relu(F.conv2d(F.max_pool2d(x, 3, 2), *c[1], padding=2))
        out.append(x)
        x = F.relu(F.conv2d(F.max_pool2d(x, 3, 2), *c[2], padding=1))
        out.append(x)
        x = F.relu(F.conv2d(x, *c[3], padding=1))
        out.append(x)
        x = F.relu(F.conv2d(x, *c[4], padding=1))
        out.append(x)
        return out

    @torch.no_grad()
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """a, b: (H, W, 3) in [0, 1]."""
        def prep(img):
            t = img.permute(2, 0, 1)[None].float() * 2.0 - 1.0
            shift = torch.tensor(self._SHIFT, device=self.device)[None, :, None, None]
            scale = torch.tensor(self._SCALE, device=self.device)[None, :, None, None]
            return (t - shift) / scale

        total = 0.0
        for fa, fb, lin in zip(self._features(prep(a)), self._features(prep(b)), self.lin):
            na = fa / (fa.pow(2).sum(1, keepdim=True).sqrt() + 1e-10)
            nb = fb / (fb.pow(2).sum(1, keepdim=True).sqrt() + 1e-10)
            total += float(F.conv2d((na - nb) ** 2, lin).mean())
        return total


# --------------------------------------------------------------------------- geometry
def sample_surface(vertices: np.ndarray, faces: np.ndarray, n: int = SAMPLES,
                   seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """n points on the triangles, with probability proportional to area, and their faces' unit
    normals. Degenerate triangles are never drawn; no triangle at all gives empty arrays."""
    v = np.asarray(vertices, np.float64)
    tri = v[np.asarray(faces, np.int64)]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1)
    ok = np.isfinite(area) & (area > 0)
    if not ok.any():
        return np.zeros((0, 3)), np.zeros((0, 3))
    tri, cross, area = tri[ok], cross[ok], area[ok]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(tri), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    pts = ((1 - s)[:, None] * tri[pick, 0] + (s * (1 - r2))[:, None] * tri[pick, 1]
           + (s * r2)[:, None] * tri[pick, 2])
    return pts, cross[pick] / area[pick][:, None]


def geometry(pred: tuple[np.ndarray, np.ndarray], gt: tuple[np.ndarray, np.ndarray]) -> dict:
    """CD (cm), precision, recall and F1 at F1_TAU (percent) and NC (percent) of two samples."""
    p, pn = pred
    q, qn = gt
    if len(p) == 0:
        return {"cd": CD_EMPTY_CM, "precision": 0.0, "recall": 0.0, "f1": 0.0, "nc": 0.0}
    d_pq, i_pq = cKDTree(q).query(p, workers=-1)
    d_qp, i_qp = cKDTree(p).query(q, workers=-1)
    prec = float((d_pq < F1_TAU).mean())
    rec = float((d_qp < F1_TAU).mean())
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    nc = (np.abs((pn * qn[i_pq]).sum(1)).mean() + np.abs((qn * pn[i_qp]).sum(1)).mean()) / 2
    return {"cd": float(100 * (d_pq.mean() + d_qp.mean()) / 2), "precision": 100 * prec,
            "recall": 100 * rec, "f1": 100 * f1, "nc": float(100 * nc)}


# --------------------------------------------------------------------------- latent similarity
def _timm_stub() -> None:
    """V-JEPA's blocks import timm's drop_path, which is the identity at inference."""
    try:
        import timm.models.layers  # noqa: F401
        return
    except ImportError:
        pass
    timm = types.ModuleType("timm")
    models = types.ModuleType("timm.models")
    layers = types.ModuleType("timm.models.layers")

    def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
        if drop_prob == 0.0 or not training:
            return x
        keep = 1 - drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep if scale_by_keep else x * mask

    layers.drop_path = drop_path
    timm.models, models.layers = models, layers
    sys.modules.update({"timm": timm, "timm.models": models, "timm.models.layers": layers})


class VJepa:
    """The frozen V-JEPA 2.1 ViT-G/16 encoder at 384 x 384, 64 frames (vjepa2_1_vit_gigantic_384)."""

    def __init__(self, weights: Path, device: torch.device):
        _timm_stub()
        sys.path.insert(0, str(weights / VJEPA_DIR))
        from app.vjepa_2_1.models import vision_transformer as vit  # noqa: E402

        # the arguments of src/hub/backbones.py: vjepa2_1_vit_gigantic_384
        enc = vit.vit_gigantic_xformers(
            patch_size=16, img_size=(LS_SIZE, LS_SIZE), num_frames=LS_FRAMES, tubelet_size=2,
            use_sdpa=True, use_silu=False, wide_silu=True, uniform_power=False, use_rope=True,
            img_temporal_dim_size=1, interpolate_rope=True)
        state = torch.load(weights / VJEPA_FILE, map_location="cpu", weights_only=True)
        enc.load_state_dict(state, strict=True)
        self.enc = enc.to(device=device, dtype=torch.float32).eval()
        self.device = device

    def _clip(self, frames: list[np.ndarray]) -> torch.Tensor:
        out = []
        for im in frames:
            img = Image.fromarray(im)
            w, h = img.size
            s = LS_SIZE / min(w, h)
            img = img.resize((max(LS_SIZE, round(w * s)), max(LS_SIZE, round(h * s))), Image.BICUBIC)
            w, h = img.size
            left, top = (w - LS_SIZE) // 2, (h - LS_SIZE) // 2
            out.append(np.asarray(img.crop((left, top, left + LS_SIZE, top + LS_SIZE)), np.float32))
        t = torch.as_tensor(np.stack(out), device=self.device) / 255.0          # (T, H, W, 3)
        t = (t - torch.tensor(_IMAGENET_MEAN, device=self.device)) / torch.tensor(_IMAGENET_STD, device=self.device)
        return t.permute(3, 0, 1, 2)[None]                                     # (1, 3, T, H, W)

    @torch.no_grad()
    def tokens(self, frames: list[np.ndarray]) -> torch.Tensor:
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            x = self.enc(self._clip(frames))
        x = x.float()[0]
        g = LS_SIZE // 16
        return x.view(LS_FRAMES // 2, g, g, -1)

    def similarity(self, source: list[np.ndarray], render: list[np.ndarray]) -> dict:
        a, b = self.tokens(source), self.tokens(render)
        motion = F.cosine_similarity(a.mean((0, 1, 2)), b.mean((0, 1, 2)), dim=0)
        layout = F.cosine_similarity(a.mean(0), b.mean(0), dim=-1).mean()
        return {"layout": 100 * float(layout), "motion": 100 * float(motion)}


def ls_indices(n: int) -> np.ndarray:
    """LS_FRAMES frame indices spread uniformly over a sequence of n frames (repeats when n < 64)."""
    return np.round(np.linspace(0, n - 1, LS_FRAMES)).astype(np.int64)
