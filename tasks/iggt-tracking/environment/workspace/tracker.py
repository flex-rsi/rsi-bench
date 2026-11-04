"""Starter solution: colour clustering per frame, connected components, greedy linking across
frames by colour similarity. Works on any input, scores low."""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def _segments(img: np.ndarray, k: int = 6, iters: int = 8, seed: int = 0):
    h, w, _ = img.shape
    small = img[::4, ::4].reshape(-1, 3).astype(np.float32)
    rng = np.random.default_rng(seed)
    centers = small[rng.choice(len(small), k, replace=False)]
    for _ in range(iters):
        lab = ((small[:, None] - centers[None]) ** 2).sum(-1).argmin(1)
        for c in range(k):
            if (lab == c).any():
                centers[c] = small[lab == c].mean(0)
    full = ((img.reshape(-1, 3).astype(np.float32)[:, None] - centers[None]) ** 2).sum(-1).argmin(1)
    full = full.reshape(h, w)
    segs = []
    for c in range(k):
        comp, n = ndimage.label(full == c)
        for i in range(1, n + 1):
            m = comp == i
            area = m.sum()
            if 0.002 * h * w <= area <= 0.2 * h * w:
                segs.append((m, img[m].mean(0)))
    return segs


def predict(frames_dir: str) -> dict:
    """frames_dir holds 000.png ... ; returns {"idmap": (S, H, W) int32, "scores": (N,)}."""
    frames = [np.array(Image.open(p).convert("RGB")) for p in sorted(Path(frames_dir).glob("*.png"))]
    S, (h, w) = len(frames), frames[0].shape[:2]
    idmap = np.zeros((S, h, w), dtype=np.int32)
    tracks = []  # mean colour of each track
    for f, img in enumerate(frames):
        used = set()
        for mask, colour in sorted(_segments(img), key=lambda s: -s[0].sum()):
            dists = [np.linalg.norm(colour - c) if t not in used else 1e9 for t, c in enumerate(tracks)]
            if dists and min(dists) < 20:
                t = int(np.argmin(dists))
            else:
                tracks.append(colour)
                t = len(tracks) - 1
            used.add(t)
            idmap[f][mask & (idmap[f] == 0)] = t + 1
    return {"idmap": idmap, "scores": np.ones(len(tracks), dtype=np.float32)}
