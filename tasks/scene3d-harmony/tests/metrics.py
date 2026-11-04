"""Image and shape metrics.

Images, on full sRGB images in [0, 1] of the same size:
- N-CLIP (the HARMONY paper's): 1 - the cosine of the CLIP ViT-B/32 image embeddings of the two
  images (transformers' CLIPModel.get_image_features after CLIPProcessor: the short side to 224
  pixels, bicubic, the center 224 x 224).
- PL (the paper's): the mean squared difference of the 8-bit RGB values scaled to [0, 1].
- PSNR: 10 log10(1 / MSE) over all pixels and channels.
- SSIM: Wang et al. (2004) with an 11 x 11 Gaussian window (sigma 1.5), K1 0.01, K2 0.03, per
  channel over the window positions inside the image, averaged.
- LPIPS: Zhang et al. (2018), AlexNet with the v0.1 linear layers, on images scaled to [-1, 1].

Shapes, between a predicted surface P and a ground-truth mesh Q (one object):
- SAMPLES surface-area-weighted points on each (numpy's default generator, seed 0).
- precision and recall at tau: the shares of points of P within tau of Q's points, and of Q's
  within tau of P's; F = 2 precision recall / (precision + recall). All in percent.
- CD: the unsquared symmetric mean nearest-neighbor distance,
  (mean_p d(p, Q) + mean_q d(q, P)) / 2, in centimeters (CD_EMPTY_CM when P is empty).

The HARMONY paper's scene geometry (its release's evaluation/chamfer_eval.py, which uses Gen3DSR's
formula), between a scene's furniture and the true furniture in view:
- PAPER_SAMPLES points: sampled by area on the prediction, drawn among the true mesh's vertices;
  numpy's default generator with a fixed seed (the release draws without one).
- Each point set centered on its mean and divided by its largest bounding-box side.
- The prediction rigidly aligned to the truth: the PCA axes with four sign flips as starts, then up
  to 30 iterations of point-to-point ICP (pytorch3d's iterative_closest_point: the transform
  re-estimated from the original points each iteration; stop when the RMSE improves by less than
  1e-6 relatively); the start with the lowest sum of the two mean squared distances wins.
- Chamfer (the mean of the two mean distances) and precision, recall and F-score at PAPER_TAUS, in
  the normalized units. An empty prediction scores Chamfer 1 and F 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

ALEXNET_FILE = "alexnet-owt-7be5be79.pth"
CLIP_DIR = "clip-vit-base-patch32"
LPIPS_FILE = "lpips-v0.1-alex.pth"
SAMPLES = 50_000
CD_EMPTY_CM = 100.0
MIN_HIDDEN = 0.05      # share of hidden points below which the hidden recall is the recall
PAPER_SAMPLES = 30_000
PAPER_TAUS = (0.1, 0.01, 0.001)


# --------------------------------------------------------------------------- images
def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """a, b: (H, W, 3) in [0, 1]."""
    mse = float(((a - b) ** 2).mean())
    return 10.0 * float(np.log10(1.0 / max(mse, 1e-10)))


def pl(a: np.ndarray, b: np.ndarray) -> float:
    """a, b: (H, W, 3) uint8."""
    return float(np.mean((a.astype(np.float32) / 255.0 - b.astype(np.float32) / 255.0) ** 2))


class Clip:
    """N-CLIP: 1 - cos(CLIP(a), CLIP(b)), CLIP ViT-B/32 as the paper runs it."""

    def __init__(self, weights: Path, device: torch.device):
        from transformers import CLIPConfig, CLIPModel, CLIPProcessor
        path = weights / CLIP_DIR
        # the checkpoint is a pickle, which this transformers loads only with torch >= 2.6: load it
        # here as plain tensors (weights_only) into the model built from its config
        self.model = CLIPModel(CLIPConfig.from_pretrained(path, local_files_only=True))
        state = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
        missing, _ = self.model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"CLIP weights miss {missing[:5]}")
        self.model = self.model.to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(path, local_files_only=True)
        self.device = device

    @torch.no_grad()
    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        """a, b: (H, W, 3) uint8."""
        from PIL import Image
        inputs = self.proc(images=[Image.fromarray(a), Image.fromarray(b)], return_tensors="pt")
        f = self.model.get_image_features(pixel_values=inputs["pixel_values"].to(self.device))
        return float(1.0 - F.cosine_similarity(f[0], f[1], dim=-1))


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
        self.conv = [(a[f"features.{i}.weight"], a[f"features.{i}.bias"])
                     for i in (0, 3, 6, 8, 10)]
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


# --------------------------------------------------------------------------- shapes
def sample_surface(vertices: np.ndarray, faces: np.ndarray, n: int = SAMPLES,
                   seed: int = 0) -> np.ndarray:
    """n points on the triangles, with probability proportional to area. Degenerate triangles are
    never drawn; no triangle at all gives an empty array."""
    v = np.asarray(vertices, np.float64)
    if len(faces) == 0:
        return np.zeros((0, 3))
    tri = v[np.asarray(faces, np.int64)]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    ok = np.isfinite(area) & (area > 0) & np.isfinite(tri).all((1, 2))
    if not ok.any():
        return np.zeros((0, 3))
    tri, area = tri[ok], area[ok]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(tri), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    return ((1 - s)[:, None] * tri[pick, 0] + (s * (1 - r2))[:, None] * tri[pick, 1]
            + (s * r2)[:, None] * tri[pick, 2])


def _f(p: float, r: float) -> float:
    return 100 * (2 * p * r / (p + r) if p + r > 0 else 0.0)


def shape(pred: np.ndarray, gt: np.ndarray, taus: tuple[float, ...],
          hidden: np.ndarray | None = None) -> dict:
    """Precision, recall and F at each tau (percent) and CD (cm) of two point samples. With
    `hidden` (a mask over gt), also the recall of the hidden points alone and the F-score of the
    precision with it (the F-score of all points when fewer than MIN_HIDDEN of them are hidden)."""
    hidden = np.zeros(len(gt), bool) if hidden is None else hidden
    out = {"hidden_share": 100 * float(hidden.mean()) if len(gt) else 0.0}
    if len(pred) == 0 or len(gt) == 0:
        for t in taus:
            out.update({f"precision@{t:g}": 0.0, f"recall@{t:g}": 0.0, f"f@{t:g}": 0.0,
                        f"recall_hidden@{t:g}": 0.0, f"f_hidden@{t:g}": 0.0})
        out["cd"] = CD_EMPTY_CM
        return out
    d_pq = cKDTree(gt).query(pred, workers=-1)[0]
    d_qp = cKDTree(pred).query(gt, workers=-1)[0]
    few = hidden.mean() < MIN_HIDDEN
    for t in taus:
        p, r = float((d_pq < t).mean()), float((d_qp < t).mean())
        rh = r if few else float((d_qp[hidden] < t).mean())
        out[f"precision@{t:g}"], out[f"recall@{t:g}"] = 100 * p, 100 * r
        out[f"f@{t:g}"] = _f(p, r)
        out[f"recall_hidden@{t:g}"], out[f"f_hidden@{t:g}"] = 100 * rh, _f(p, rh)
    out["cd"] = float(min(CD_EMPTY_CM, 100 * (d_pq.mean() + d_qp.mean()) / 2))
    return out


# --------------------------------------------------------------------------- the paper's geometry
def _normalize(pc: np.ndarray) -> np.ndarray:
    return (pc - pc.mean(0)) / max(float((pc.max(0) - pc.min(0)).max()), 1e-9)


def _pca(pc: np.ndarray) -> np.ndarray:
    c = pc - pc.mean(0)
    _, vecs = np.linalg.eigh(c.T @ c / max(len(pc), 1))
    R = vecs[:, ::-1].copy()
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    return R


def _rigid(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R, T with Y ~ X @ R + T (no scale, no reflection)."""
    xm, ym = X.mean(0), Y.mean(0)
    U, _, Vh = np.linalg.svd((X - xm).T @ (Y - ym) / len(X))
    E = np.eye(3)
    E[2, 2] = np.sign(np.linalg.det(U @ Vh)) or 1.0
    R = U @ E @ Vh
    return R, ym - xm @ R


def _icp(X: np.ndarray, Y: np.ndarray, tree: cKDTree, iters: int = 30, thr: float = 1e-6):
    Xt, prev = X, None
    for _ in range(iters):
        nn = Y[tree.query(Xt, workers=-1)[1]]
        R, T = _rigid(X, nn)
        Xt = X @ R + T
        rmse = float(np.sqrt(((Xt - nn) ** 2).sum(1).mean()))
        if prev is not None and (prev - rmse) / max(prev, 1e-12) <= thr:
            break
        prev = rmse
    return Xt


def paper_geometry(pred_v: np.ndarray, pred_f: np.ndarray, gt_vertices: np.ndarray,
                   seed: int = 0) -> dict:
    """Chamfer and precision / recall / F at PAPER_TAUS of a prediction (vertices, faces) against
    the true mesh's vertices, as the paper computes them."""
    out = {"chamfer": 1.0}
    for t in PAPER_TAUS:
        out.update({f"precision@{t:g}": 0.0, f"recall@{t:g}": 0.0, f"f@{t:g}": 0.0})
    pred = sample_surface(pred_v, pred_f, PAPER_SAMPLES, seed)
    if len(pred) == 0:
        return out
    gt = np.asarray(gt_vertices, np.float64)
    if len(gt) > PAPER_SAMPLES:
        gt = gt[np.random.default_rng(seed).choice(len(gt), PAPER_SAMPLES, replace=False)]
    gt, pred = _normalize(gt), _normalize(pred)
    tree = cKDTree(gt)
    R_src, R_dst = _pca(pred).T, _pca(gt)
    best, best_cd = pred, np.inf
    for flip in (np.eye(3), np.diag([1.0, -1, -1]), np.diag([-1.0, 1, -1]), np.diag([-1.0, -1, 1])):
        aligned = _icp(pred @ (R_dst @ flip @ R_src).T, gt, tree)
        cd = float((tree.query(aligned, workers=-1)[0] ** 2).mean()
                   + (cKDTree(aligned).query(gt, workers=-1)[0] ** 2).mean())
        if cd < best_cd:
            best, best_cd = aligned, cd
    d_gp = cKDTree(best).query(gt, workers=-1)[0]
    d_pg = tree.query(best, workers=-1)[0]
    out["chamfer"] = float((d_gp.mean() + d_pg.mean()) / 2)
    for t in PAPER_TAUS:
        p, r = float((d_pg < t).mean()), float((d_gp < t).mean())
        out[f"precision@{t:g}"], out[f"recall@{t:g}"], out[f"f@{t:g}"] = 100 * p, 100 * r, _f(p, r)
    return out
