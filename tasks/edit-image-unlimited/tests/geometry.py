"""Geometric fidelity and aligned preservation.

Generative editors often zoom, shift or stretch the whole picture by a few percent. Pixel
differences and the VLM judge both miss this, and an unaligned pixel comparison then mixes the
global warp with real content changes. So the output is first registered to the source:

1. source and output are compared at a 512-px short side (the output is resized to the source
   frame; for reframing tasks it keeps its own aspect ratio);
2. a source->output homography H is fitted on SIFT matches with RANSAC; source keypoints inside
   the edit mask are not used, and edited content elsewhere is rejected as outliers. A fit that
   explains the preserved structure (locally contrast-normalised grey difference) worse than no
   warp at all is rejected and the output counts as unwarped (fits locked onto repeated
   patterns or onto content next to the edit);
3. **geometry**: the displacement |H(x) - x| measures the global deformation (zoom, shift,
   stretch, shear, perspective). It is taken over the convex hull of the inlier keypoints, where
   the fit is observed (outside it the model only extrapolates); its 90th percentile drives the
   score g = exp(-max(0, p90 - 0.5) / 6) (px at a 512-px short side: 1 px -> 0.92, 3 px -> 0.66,
   5 px -> 0.47, 10 px -> 0.21, 20 px -> 0.04). For reframing tasks the best similarity (uniform scale + translation + rotation) is
   free and only the residual counts;
4. **preserve** (samples with an edit mask): the output is warped back onto the source with H and
   the mean absolute RGB difference (MAD) is taken outside the dilated edit mask:
   clip(1 - (MAD - 0.02) / 0.10, 0, 1). Local misalignment the homography cannot explain (moved or
   pasted patches) shows up here as a content change;
5. **detail**: MAD at 512 px misses smeared or redrawn fine texture (low-resolution output upscaled,
   generative smoothing). The aligned output is compared with the source at a 1024-px short side
   (source resolution if smaller) by the contrast-structure term of SSIM on grey levels
   (Gaussian window sigma 1.5; insensitive to brightness shifts): mean over the preserved region
   with a mask, median over the frame without one (robust to a small local edit);
   clip((cs - 0.55) / 0.40, 0, 1). Identity scores ~0.99, a 2x down-up resample ~0.94.

When too few matches support a homography, geometry is not scored and preserve is computed
without alignment.
"""

import cv2
import numpy as np
from PIL import Image

S = 512
TOL_PX, TAU_PX = 0.5, 6.0
DS = 1024  # detail analysis short side
CS_ZERO, CS_ONE = 0.55, 0.95
MIN_INLIERS = 30
RANSAC_PX = 3.0
MAD_TOL, MAD_SPAN = 0.02, 0.10
GROW_DIFF = 0.12  # mean abs RGB difference (0..1) counted as a side effect of a grounded edit
MIN_KEEP = 0.15  # fraction of the source frame that must remain outside the edit region
MIN_SPREAD = 0.15  # inlier convex hull / frame area below which an affine model is fitted instead
ACCEPT_RATIO = 1.0  # a warp that explains the preserved structure worse than no warp is rejected


def region_mask(mask, boxes, size):
    """Edit region at `size` (w, h): the mask file united with boxes [x1, y1, x2, y2] normalised to
    0-1000. Each box grows by 15% of its own size (cast shadows, reflections, loose grounding) plus
    4% of the short side, like the dataset masks. None when there is neither."""
    w, h = size
    m = None
    if mask:
        m = np.asarray(Image.open(mask).convert("L").resize(size, Image.NEAREST)) >= 128
    if boxes:
        m = np.zeros((h, w), bool) if m is None else m.copy()
        pad = 0.04 * min(w, h)
        for x1, y1, x2, y2 in boxes:
            x1, x2 = sorted((x1 / 1000 * w, x2 / 1000 * w))
            y1, y2 = sorted((y1 / 1000 * h, y2 / 1000 * h))
            px, py = 0.15 * (x2 - x1) + pad, 0.15 * (y2 - y1) + pad
            m[max(0, int(y1 - py)):int(np.ceil(y2 + py)), max(0, int(x1 - px)):int(np.ceil(x2 + px))] = True
    return m


def grow_region(m, boxes, diff, inside):
    """Add what a grounded object brings with it (cast shadow, reflection, steam, smoke): strong
    differences (> GROW_DIFF after smoothing) connected to the boxes, within one box size of them."""
    h, w = m.shape
    near = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((x1 / 1000 * w, x2 / 1000 * w))
        y1, y2 = sorted((y1 / 1000 * h, y2 / 1000 * h))
        bw, bh = x2 - x1, y2 - y1
        near[max(0, int(y1 - bh)):int(np.ceil(y2 + bh)), max(0, int(x1 - bw)):int(np.ceil(x2 + bw))] = 1
    # resampling at the frame edge and outside the warped output are not side effects
    e = round(0.02 * min(h, w)) * 2 + 1
    valid = cv2.erode(inside.astype(np.uint8), np.ones((e, e), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    strong = (cv2.GaussianBlur(diff, (0, 0), 2) > GROW_DIFF).astype(np.uint8) & near & valid
    strong = cv2.morphologyEx(strong, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab = cv2.connectedComponents(strong | m.astype(np.uint8))
    touch = np.unique(lab[m])
    grown = np.isin(lab, touch[touch > 0]) & (strong > 0)
    d = round(0.02 * min(h, w)) * 2 + 1
    return m | (cv2.dilate(grown.astype(np.uint8), np.ones((d, d), np.uint8)) > 0)


def load(src, out, mask, reframe, boxes=()):
    """-> source RGB, output RGB (float32 0..1, analysis size) and the edit region (bool) or None."""
    a = Image.open(src).convert("RGB")
    k = S / min(a.size)
    size = (round(a.width * k), round(a.height * k))
    b = Image.open(out).convert("RGB")
    if reframe:  # keep the output's own aspect; the free similarity absorbs scale/offset
        kb = S / min(b.size)
        b = b.resize((round(b.width * kb), round(b.height * kb)), Image.BICUBIC)
    else:
        b = b.resize(size, Image.BICUBIC)
    a = a.resize(size, Image.BICUBIC)
    m = region_mask(mask, boxes, size)
    f = lambda im: np.asarray(im, dtype=np.float32) / 255
    return f(a), f(b), m


def _gray(x):
    return cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)


def homography(a, b, mask=None):
    """Source->output homography (3x3), inlier count and inlier source points; H is None when
    unreliable."""
    sift = cv2.SIFT_create(5000)
    kmask = None if mask is None else (~mask).astype(np.uint8) * 255
    ka, da = sift.detectAndCompute(_gray(a), kmask)
    kb, db = sift.detectAndCompute(_gray(b), None)
    if da is None or db is None or len(ka) < MIN_INLIERS or len(kb) < MIN_INLIERS:
        return None, 0, None
    good = [x[0] for x in cv2.BFMatcher().knnMatch(da, db, k=2)
            if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
    if len(good) < MIN_INLIERS:
        return None, len(good), None
    pa = np.float32([ka[g.queryIdx].pt for g in good])
    pb = np.float32([kb[g.trainIdx].pt for g in good])
    H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, RANSAC_PX, maxIters=5000, confidence=0.999)
    n = int(inl.sum()) if inl is not None else 0
    if H is None or n < MIN_INLIERS:
        return None, n, None
    hull = cv2.contourArea(cv2.convexHull(pa[inl.ravel() > 0]))
    if hull < MIN_SPREAD * a.shape[0] * a.shape[1]:
        # inliers in a small patch: perspective terms would extrapolate wildly, use an affine model
        A, inl = cv2.estimateAffine2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX)
        n = int(inl.sum()) if inl is not None else 0
        if A is None or n < MIN_INLIERS:
            return None, n, None
        H = np.vstack([A, [0, 0, 1]])
    return H, n, pa[inl.ravel() > 0]


def _struct_err(a, b, region):
    """Mean |difference| of locally contrast-normalised grey images (ignores tone/colour changes)."""
    def norm(x):
        g = _gray(x).astype(np.float32)
        mu = cv2.GaussianBlur(g, (0, 0), 4)
        return (g - mu) / (np.sqrt(cv2.GaussianBlur((g - mu) ** 2, (0, 0), 4)) + 4)
    if region.sum() == 0:
        return np.inf
    return float(np.abs(norm(a) - norm(b))[region].mean())


def displacement_field(H, shape, reframe=False):
    """|H(x) - x| (or the residual after the best similarity, for reframing) on every source pixel."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    p = np.stack([xx.ravel(), yy.ravel()], 1)
    q = cv2.perspectiveTransform(p[None], H)[0]
    if reframe:
        sub = np.ascontiguousarray(p[:: max(1, len(p) // 4000)])
        qs = cv2.perspectiveTransform(sub[None], H)[0]
        A, _ = cv2.estimateAffinePartial2D(sub, qs, method=cv2.LMEDS)
        p = p @ A[:, :2].T + A[:, 2]
    return np.linalg.norm(q - p, axis=1).reshape(h, w)


def keep_region(mask, shape):
    """Pixels outside the edit mask dilated by 3% of the short side (all pixels without a mask)."""
    if mask is None:
        return np.ones(shape, bool)
    d = round(0.03 * min(shape)) // 2
    return ~(cv2.dilate(mask.astype(np.uint8), np.ones((2 * d + 1, 2 * d + 1), np.uint8)) > 0)


def aligned(a, b, H):
    """Output warped onto the source frame with H, and where it is defined."""
    h, w = a.shape[:2]
    if H is None:
        if b.shape != a.shape:
            b = cv2.resize(b, (w, h), interpolation=cv2.INTER_CUBIC)
        return b, np.ones((h, w), bool)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP  # bw(x) = b(H x)
    bw = cv2.warpPerspective(b, H, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT)
    inside = cv2.warpPerspective(np.ones(b.shape[:2], np.uint8), H, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
    return bw, cv2.erode(inside.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def analyse(src, out, mask=None, reframe=False, boxes=()):
    """Everything the checks and the data browser need, at analysis size."""
    a, b, m = load(src, out, mask, reframe, boxes)
    H, n, pts = homography(a, b, m)
    if H is None and m is not None:
        # too few matches outside a large edit region: fit on the whole frame (RANSAC drops the
        # matches on edited content) so that deformation is still measured
        H, n, pts = homography(a, b, None)
    rejected = False
    bw, inside = aligned(a, b, H)
    keep = keep_region(m, a.shape[:2]) & inside
    if H is not None and not reframe and keep.mean() >= MIN_KEEP:
        # a homography fitted on repeated patterns or on matches hugging the edit can be spurious;
        # keep it only if it explains the preserved structure clearly better than no warp at all
        k0 = keep_region(m, a.shape[:2])
        if _struct_err(a, bw, keep) > ACCEPT_RATIO * _struct_err(a, b, k0):
            H, rejected = np.eye(3), True
            bw, inside, keep = b, np.ones(a.shape[:2], bool), k0
    disp = None if H is None else displacement_field(H, a.shape[:2], reframe)
    support = None
    if pts is not None:  # where the fit is observed; outside it the model only extrapolates
        support = np.zeros(a.shape[:2], np.uint8)
        cv2.fillConvexPoly(support, cv2.convexHull(pts).astype(np.int32), 1)
        support = support > 0
    diff = np.abs(a - bw).mean(axis=2)
    if boxes and m is not None:
        m = grow_region(m, boxes, diff, inside)
        keep = keep_region(m, a.shape[:2]) & inside
    return dict(a=a, bw=bw, mask=m, H=H, inliers=n, disp=disp, keep=keep, inside=inside, diff=diff,
                rejected=rejected, support=support)


def full_aligned(src, out, r, reframe, short=None):
    """Source and the output warped onto it with the analysis homography, as uint8 RGB at the
    source resolution (or at `short` px short side, if smaller)."""
    A = Image.open(src).convert("RGB")
    B = Image.open(out).convert("RGB")
    if short and min(A.size) > short:
        k = short / min(A.size)
        A = A.resize((round(A.width * k), round(A.height * k)), Image.BICUBIC)
    ha, wa = r["a"].shape[:2]
    if reframe:
        kb = S / min(B.size)
        hb, wb = round(B.height * kb), round(B.width * kb)
    else:
        hb, wb = ha, wa
    H = np.eye(3) if r["H"] is None else r["H"]
    sa = np.diag([wa / A.width, ha / A.height, 1.0])  # source frame -> analysis
    sb = np.diag([B.width / wb, B.height / hb, 1.0])  # analysis output -> output pixels
    b = cv2.warpPerspective(np.asarray(B), sb @ H @ sa, A.size, flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))
    return np.asarray(A), b


def contrast_structure(x, y, sigma=1.5):
    """SSIM contrast-structure map of two grey float images (0..255)."""
    C = (0.03 * 255) ** 2
    g = lambda z: cv2.GaussianBlur(z, (0, 0), sigma)
    mx, my = g(x), g(y)
    vx, vy, cxy = g(x * x) - mx * mx, g(y * y) - my * my, g(x * y) - mx * my
    return (2 * cxy + C) / (vx + vy + C)


def detail(src, out, r, reframe, masked):
    """-> (score, cs) or None when too little of the frame is left to compare."""
    a, b = full_aligned(src, out, r, reframe, DS)
    h, w = a.shape[:2]
    keep = cv2.resize(r["keep"].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    keep = cv2.erode(keep, np.ones((9, 9), np.uint8)) > 0
    if keep.mean() < MIN_KEEP:
        return None
    gr = lambda z: cv2.cvtColor(z, cv2.COLOR_RGB2GRAY).astype(np.float32)
    m = contrast_structure(gr(a), gr(b))[keep]
    cs = float(m.mean() if masked else np.median(m))
    return float(np.clip((cs - CS_ZERO) / (CS_ONE - CS_ZERO), 0, 1)), cs


def fidelity(src, out, mask=None, reframe=False, boxes=(), geometry_on=True, preserve_on=None, detail_on=False):
    """-> {"geometry", "preserve", "detail": score or None, "info": {...}}.

    The edit region is the mask united with `boxes` (grounded objects, 0-1000 coordinates); without
    either the whole frame is compared. geometry is scored when `geometry_on` and a homography is
    found; preserve (default: when there is a mask) and detail when asked for and enough of the
    frame is left outside the edit region."""
    if preserve_on is None:
        preserve_on = mask is not None
    r = analyse(src, out, mask, reframe, boxes)
    info = {"homography_inliers": r["inliers"], "warp_rejected": r["rejected"]}
    geo = None
    if geometry_on and r["disp"] is not None:
        p90 = float(np.percentile(r["disp"][r["support"]], 90))
        geo = float(np.exp(-max(0.0, p90 - TOL_PX) / TAU_PX))
        info.update(disp_p90=round(p90, 2), disp_max=round(float(r["disp"][r["support"]].max()), 2),
                    support=round(float(r["support"].mean()), 3),
                    homography=[[round(float(v), 6) for v in row] for row in r["H"]])
    pres = None
    if r["mask"] is not None:
        info["region_frac"] = round(float(r["mask"].mean()), 3)
    if preserve_on and r["keep"].mean() >= MIN_KEEP:
        mad = float(r["diff"][r["keep"]].mean())
        pres = float(np.clip(1 - (mad - MAD_TOL) / MAD_SPAN, 0, 1))
        info.update(mad=round(mad, 4), aligned=r["H"] is not None)
    det = None
    if detail_on:
        d = detail(src, out, r, reframe, r["mask"] is not None)
        if d is not None:
            det = d[0]
            info["detail_cs"] = round(d[1], 4)
    return {"geometry": geo, "preserve": pres, "detail": det, "info": info}
