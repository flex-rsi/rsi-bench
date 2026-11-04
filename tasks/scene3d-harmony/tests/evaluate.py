"""Evaluator: reconstruct every photo of a split in a child process, then render each scene.blend
with Blender from the scene's own camera and score it: the render against the photo (the HARMONY
paper's N-CLIP, PL and LPIPS), each annotated piece of furniture's reconstruction as a whole 3D
object (judge.py), and, on the 3D-FRONT renders, the furniture's geometry against the truth (the
paper's Chamfer and F-scores, and per-object F-scores in the camera's frame).

The child (child_reconstruct.py, through the sandbox launcher when there is one) sees only the
inputs, inputs/<id>/photo.png, and Blender (read-only; BLENDER is its path). For each photo it calls
reconstruct(scene_dir, out_dir) of the workspace, which writes out_dir/scene.blend. The split must
finish within SECONDS_PER_SCENE x n + STARTUP_S seconds; then the child is stopped, and the photos
it has not done are empty. A child that exits with an error fails the evaluation.

Each scene.blend is opened by Blender (blender_eval.py: the rules, then a Cycles render of the
input view from the scene's camera, its pose and intrinsics, at half the photo's size with the long
side at most RENDER_MAX, and the geometry that renders). A scene that is missing, does not load or
breaks a hard limit renders black and has no geometry.

Labels (labels/<id>/): meta.json (difficulty: easy for the 3D-FRONT renders, medium or complicated
for the real photos; the true camera of the renders), objects.json (the annotated furniture: label,
boxy), and gt.npz (the true furniture in view, the camera's frame) with layout_depth.npy for the
renders, or instances.png (pixel value k: the k-th annotated object) for the real photos.

Per photo:
    n_clip, pl, lpips  the paper's appearance metrics of the render against the photo (averaged
                     over pixel blocks to the render's size); also psnr and ssim
    appearance_term  (max(0, 1 - n_clip / NCLIP_ZERO) + max(0, 1 - pl / PL_ZERO) + 1 - lpips) / 3
    objects          every rendered mesh object is assigned to the label (an annotated object,
                     other furniture or the room) that covers most of the pixels it shows in the
                     input view, the room's mesh objects (named floor*, ceiling* or wall*) to the
                     room; an annotated object's reconstruction is the mesh objects assigned to
                     it. object_iou: the intersection over union, in the input view, of the
                     pixels it shows and those the object covers; object_realness: judge.py's
                     realness of the reconstruction (0 when nothing is assigned)
    object_term      mean over the annotated objects of realness x min(1, iou / IOU_FULL) (none
                     for a photo without annotated furniture)
On the 3D-FRONT renders also:
    paper_chamfer, paper_f@0.1, paper_f@0.01, paper_f@0.001   the paper's geometry
                     (metrics.paper_geometry) of the scene's furniture, every rendered mesh object
                     whose name does not start with floor, ceiling or wall, against the true
                     furniture in view
    object_f         mean over the annotated objects of the F-score at TAU (5 cm) of the object's
                     reconstruction against its true mesh in the camera's frame, the
                     reconstruction scaled by s (the median of true / rendered depth over the
                     pixels that see a surface)
    object_f_hidden  the same with the recall over the true points the input view does not show
                     (outside the frame or more than SEEN_TOL behind the true depth of their pixel)
    geometry_term    (paper_f@0.1 + paper_f@0.01 + object_f + object_f_hidden) / 400
    depth_absrel, object_f10, object_cd, object_precision, object_recall, object_recall_hidden

The primary metric, higher is better:
    score = (appearance_term + object_term + geometry_term) / 3, the first two averaged over all
    photos, geometry_term over the 3D-FRONT renders.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import metrics  # noqa: E402
import raster  # noqa: E402

SECONDS_PER_SCENE = 120
STARTUP_S = 600
RENDER_MAX = 1024                       # the render's long side at most
SAMPLES = 128
ORBITS = (-30.0, 30.0)                  # extra views for the contact sheet only
ORBIT_SAMPLES = 32
BLENDER_TIMEOUT_S = 900
MAX_BLEND_MB = 512
TAU = 0.05
TAUS = (0.05, 0.10)
SEEN_TOL = (0.03, 0.02)                 # a true point is seen when it lies in the frame, at most
                                        # 3 cm + 2 % behind the true depth of its pixel
ROOM, OTHER = -1, -2                    # labels besides the annotated objects 0, 1, ...
ROOM_NAMES = ("floor", "ceiling", "wall")
NCLIP_ZERO, PL_ZERO = 0.25, 0.1         # appearance: n_clip and pl at which their parts are 0
IOU_FULL = 0.5
CROP_MARGIN = 0.15
GL_TO_CV = np.diag([1.0, -1.0, -1.0])
PALETTE = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
                    [145, 30, 180], [70, 240, 240], [240, 50, 230]], np.float64)


def render_size(width: int, height: int) -> tuple[int, int]:
    s = min(0.5, RENDER_MAX / max(width, height))
    return max(1, round(width * s)), max(1, round(height * s))


# --------------------------------------------------------------------------- the solution
def solution_cmd(python: str, io_dir: Path, blender_dir: Path) -> list[str]:
    """The solution's process: through the sandbox launcher when the orchestrator gives one, so it
    sees its own code, the run cache, io_dir and Blender, never the labels."""
    child = io_dir / "_child_reconstruct.py"
    shutil.copy(Path(__file__).with_name("child_reconstruct.py"), child)
    cmd = [python, str(child)]
    prefix = json.loads(os.environ.get("RSIB_SANDBOX") or "[]")
    if not prefix:
        return cmd
    return (prefix + ["--rw", str(io_dir), "--ro", str(blender_dir), "--cwd", str(io_dir), "--"]
            + cmd)


def run_solution(python: str, io_dir: Path, workspace: Path, blender_dir: Path, limit: float,
                 log_path: Path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("RSIB_")}
    env.update({"RSIB_IO_DIR": str(io_dir), "RSIB_SOLVER_DIR": str(workspace),
                "BLENDER": str(blender_dir / "blender"),
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"})
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(solution_cmd(python, io_dir, blender_dir), env=env, cwd=io_dir,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            proc.wait(timeout=limit)
            timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            timed_out = True
    return time.time() - t0, timed_out, proc.returncode


# --------------------------------------------------------------------------- Blender
def blender_scene(blend: Path, work: Path, size: tuple[int, int], blender_dir: Path,
                  cache: Path) -> dict:
    """Render and export one scene.blend with blender_eval.py. Returns the audit, or raises
    ValueError with the reason the scene does not load."""
    if not blend.is_file():
        raise ValueError("no scene.blend")
    if blend.stat().st_size > MAX_BLEND_MB * 2 ** 20:
        raise ValueError(f"scene.blend is larger than {MAX_BLEND_MB} MB")
    work.mkdir(parents=True, exist_ok=True)
    job = {"blend": str(blend), "out_dir": str(work), "width": size[0], "height": size[1],
           "samples": SAMPLES, "orbits": list(ORBITS), "orbit_samples": ORBIT_SAMPLES,
           "device": "CUDA" if torch.cuda.is_available() else "CPU",
           "threads": min(16, os.cpu_count() or 1)}
    (work / "job.json").write_text(json.dumps(job))
    env = {k: v for k, v in os.environ.items() if not k.startswith("RSIB_")}
    # Cycles has no CUDA kernels for these GPUs: the driver compiles them once (about 4 minutes) and
    # keeps them here for the other scenes (the container disables the cache by default)
    env.update(CUDA_CACHE_DISABLE="0", CUDA_CACHE_PATH=str(cache), CUDA_CACHE_MAXSIZE=str(4 << 30))
    cmd = [str(blender_dir / "blender"), "-b", "--factory-startup", "-noaudio",
           "--disable-autoexec", "--python", str(Path(__file__).with_name("blender_eval.py")), "--",
           str(work / "job.json")]
    with open(work / "blender.log", "w") as log:
        try:
            proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=BLENDER_TIMEOUT_S, start_new_session=True)
        except subprocess.TimeoutExpired:
            raise ValueError(f"Blender took more than {BLENDER_TIMEOUT_S} s") from None
    if (work / "error.txt").exists():
        raise ValueError((work / "error.txt").read_text()[:500])
    if proc.returncode != 0 or not (work / "render.png").exists():
        tail = (work / "blender.log").read_text(errors="replace")[-400:]
        raise ValueError(f"Blender failed (status {proc.returncode}): {tail}")
    return json.loads((work / "audit.json").read_text())


# --------------------------------------------------------------------------- labels
def ground_truth(labels: Path, k: tuple, size: tuple[int, int], device) -> dict:
    """A 3D-FRONT render: true depth and label map at `size`, each annotated object's mesh, and
    the true furniture's vertices."""
    W, H = size
    g = np.load(labels / "gt.npz")
    v, f, fo = g["vertices"].astype(np.float64), g["faces"].astype(np.int64), g["face_object"]
    n = len(json.loads((labels / "objects.json").read_text()))
    parts = [(v, f[fo == j]) for j in range(n)] + [(v, f[fo < 0])]
    depth, index = raster.render(parts, *k, H, W, device)
    layout = torch.as_tensor(np.load(labels / "layout_depth.npy"), dtype=torch.float32,
                             device=device)
    if tuple(layout.shape) != (H, W):
        raise SystemExit(f"{labels}: layout_depth.npy is {tuple(layout.shape)}, not {(H, W)}")
    label = torch.where(index == n, torch.full_like(index, OTHER), index)
    label = torch.where((index < 0) | (layout < depth), torch.full_like(index, ROOM), label)
    return {"depth": torch.minimum(depth, layout), "label": label, "n": n,
            "meshes": [(v, f[fo == j]) for j in range(n)], "vertices": v}


def real_labels(labels: Path, size: tuple[int, int], device) -> dict:
    """A real photo: the label map of its annotated objects at `size` (the room elsewhere)."""
    n = len(json.loads((labels / "objects.json").read_text()))
    inst = np.asarray(Image.open(labels / "instances.png").resize(size, Image.NEAREST), np.int64)
    label = np.where((inst >= 1) & (inst <= n), inst - 1, ROOM)
    return {"label": torch.as_tensor(label, device=device), "n": n}


def camera_frame(vertices: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """World vertices into the OpenCV frame of a Blender camera (x right, y down, z forward)."""
    return (vertices.astype(np.float64) - c2w[:3, 3]) @ (c2w[:3, :3] @ GL_TO_CV)


def depth_absrel(pred: torch.Tensor, gt: torch.Tensor) -> tuple[float, float]:
    """AbsRel of pred against gt after the median scale, and the scale."""
    valid = torch.isfinite(gt) & (gt > 0)
    hit = valid & torch.isfinite(pred) & (pred > 0)
    if not hit.any():
        return 1.0, 1.0
    s = torch.median(gt[hit] / pred[hit])
    err = torch.ones_like(gt)
    err[hit] = ((s * pred[hit] - gt[hit]).abs() / gt[hit]).clamp(max=1.0)
    return float(err[valid].mean()), float(s)


def seen(points: np.ndarray, depth: torch.Tensor, fx, fy, cx, cy) -> np.ndarray:
    """Which points (camera frame) the input view shows: in the frame and not behind the true
    surface of their pixel."""
    d = depth.cpu().numpy()
    H, W = d.shape
    z = points[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.floor(fx * points[:, 0] / z + cx)
        w = np.floor(fy * points[:, 1] / z + cy)
    ok = (z > 0) & (u >= 0) & (u < W) & (w >= 0) & (w < H)
    out = np.zeros(len(points), bool)
    zi = d[w[ok].astype(np.int64), u[ok].astype(np.int64)]
    out[ok] = z[ok] <= zi + SEEN_TOL[0] + SEEN_TOL[1] * zi
    return out


# --------------------------------------------------------------------------- scoring
def is_room(name: str) -> bool:
    return name.lower().startswith(ROOM_NAMES)


def assign(meshes: list, names: list[str], k: tuple, size: tuple[int, int], label: torch.Tensor,
           device):
    """Depth and mesh-index map of the scene seen from its camera, and {mesh: label}: the label
    covering most of the pixels each mesh object shows (the room's mesh objects: the room)."""
    depth, index = raster.render(meshes, *k, size[1], size[0], device)
    assigned = {}
    for m in torch.unique(index[index >= 0]).tolist():
        if is_room(names[m]):
            assigned[m] = ROOM
            continue
        labs, counts = torch.unique(label[index == m], return_counts=True)
        assigned[m] = int(labs[torch.argmax(counts)])
    return depth, index, assigned


def object_rows(n: int, meshes: list, index: torch.Tensor, assigned: dict, label: torch.Tensor,
                names: list[str]) -> list[dict]:
    out = []
    for j in range(n):
        mine = [m for m, lab in assigned.items() if lab == j]
        shows = (torch.isin(index, torch.tensor(mine, device=index.device)) if mine
                 else torch.zeros_like(index, dtype=torch.bool))
        covers = label == j
        union = int((shows | covers).sum())
        out.append({"object": j, "mine": mine, "meshes": [names[m] for m in mine][:20],
                    "iou": int((shows & covers).sum()) / union if union else 0.0})
    return out


def shape_scores(rows: list[dict], meshes: list, s: float, gt: dict, k_true: tuple) -> None:
    """The per-object F-scores (3D-FRONT renders) into rows."""
    for row in rows:
        j = row["object"]
        pred_f = (np.concatenate([meshes[m][1] for m in row["mine"]]) if row["mine"]
                  else np.zeros((0, 3), np.int64))
        v = meshes[0][0] if meshes else np.zeros((0, 3))
        q = metrics.sample_surface(*gt["meshes"][j])
        res = metrics.shape(metrics.sample_surface(v * s, pred_f), q, TAUS,
                            hidden=~seen(q, gt["depth"], *k_true))
        row.update(f=res["f@0.05"], f_hidden=res["f_hidden@0.05"], f10=res["f@0.1"], cd=res["cd"],
                   precision=res["precision@0.05"], recall=res["recall@0.05"],
                   recall_hidden=res["recall_hidden@0.05"])


def crop_of(photo_full: Image.Image, mask: np.ndarray) -> Image.Image:
    """The photo around a mask given at a smaller size, with a margin, at most judge.VIEW."""
    import judge
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:                     # an object hidden in the input view: the whole photo
        crop = photo_full.copy()
        crop.thumbnail((judge.VIEW, judge.VIEW))
        return crop
    sx, sy = photo_full.width / mask.shape[1], photo_full.height / mask.shape[0]
    m = CROP_MARGIN * max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
    box = (max(0, int((xs.min() - m) * sx)), max(0, int((ys.min() - m) * sy)),
           min(photo_full.width, int((xs.max() + 1 + m) * sx)),
           min(photo_full.height, int((ys.max() + 1 + m) * sy)))
    crop = photo_full.crop(box)
    crop.thumbnail((judge.VIEW, judge.VIEW))
    return crop


# --------------------------------------------------------------------------- images
def to_tensor(img: np.ndarray, device) -> torch.Tensor:
    return torch.as_tensor(img.copy(), device=device).float() / 255.0


def load_png(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def assignment_image(photo: np.ndarray, index: torch.Tensor | None, assigned: dict,
                     label: torch.Tensor) -> np.ndarray:
    """The photo overlaid with each mesh object colored as the annotated object it was assigned to
    (gray: other furniture, dark: the room); outlines of the annotated objects."""
    out = photo.astype(np.float64) * 0.35
    if index is not None:
        idx = index.cpu().numpy()
        col = np.zeros_like(out)
        for m, lab in assigned.items():
            col[idx == m] = PALETTE[lab % len(PALETTE)] if lab >= 0 else (128 if lab == OTHER else 40)
        out = np.where((idx >= 0)[..., None], 0.35 * photo + 0.65 * col, out)
    lab = label.cpu().numpy()
    edge = np.zeros(lab.shape, bool)
    edge[1:] |= lab[1:] != lab[:-1]
    edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    out[edge & (lab >= 0)] = 255
    return out.clip(0, 255).astype(np.uint8)


def error_image(pred: torch.Tensor | None, gt: torch.Tensor) -> np.ndarray:
    """|depth error| / depth: black 0, white >= 0.25 (red where the scene shows no surface)."""
    h, w = gt.shape
    if pred is None:
        img = np.zeros((h, w, 3), np.uint8)
        img[...] = (160, 0, 0)
        return img
    err = ((pred - gt).abs() / gt).clamp(max=0.25) / 0.25
    img = (err.cpu().numpy()[..., None] * 255).repeat(3, -1)
    img[~torch.isfinite(pred).cpu().numpy()] = (160, 0, 0)
    return img.astype(np.uint8)


def fit_image(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    canvas = Image.new("RGB", size, (0, 0, 0))
    im = img.copy()
    im.thumbnail(size)
    canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return np.asarray(canvas)


def contact_sheet(images: list[np.ndarray], path: Path) -> None:
    """photo, render, assignment / orbit -30, orbit +30, depth error (renders) or the clay views
    of the largest annotated object (real photos)."""
    h, w = images[0].shape[:2]
    sheet = np.zeros((2 * h + 4, 3 * w + 8, 3), np.uint8)
    for k, im in enumerate(images[:6]):
        r, c = divmod(k, 3)
        sheet[r * (h + 4):r * (h + 4) + h, c * (w + 4):c * (w + 4) + w] = im
    Image.fromarray(sheet).save(path, quality=88)


# --------------------------------------------------------------------------- one photo
def score_scene(sid: str, split_dir: Path, out: Path, work: Path, blender_dir: Path, cache: Path,
                models: dict, device) -> tuple[dict, list[np.ndarray], list[dict]]:
    lab_dir = split_dir / "labels" / sid
    meta = json.loads((lab_dir / "meta.json").read_text())
    objs = json.loads((lab_dir / "objects.json").read_text())
    photo_full = Image.open(split_dir / "inputs" / sid / "photo.png").convert("RGB")
    size = render_size(*photo_full.size)
    W, H = size
    photo = np.asarray(photo_full.resize(size, Image.BOX))
    easy = meta["difficulty"] == "easy"
    if easy:
        c = meta["camera"]
        s = W / c["width"]
        k_true = (c["fx"] * s, c["fy"] * s, c["cx"] * s, c["cy"] * s)
        gt = ground_truth(lab_dir, k_true, size, device)
    else:
        gt = real_labels(lab_dir, size, device)
    row: dict = {"error": None, "difficulty": meta["difficulty"]}
    black = np.zeros_like(photo)
    render, orbits, index, assigned, depth = black, [black, black], None, {}, None
    rows = [{"object": j, "mine": [], "meshes": [], "iou": 0.0} for j in range(gt["n"])]
    meshes = []
    try:
        t0 = time.time()
        audit = blender_scene(out / "scene.blend", work, size, blender_dir, cache)
        row["render_s"] = round(time.time() - t0, 1)
        render = load_png(work / "render.png")
        orbits = [load_png(work / f"orbit_{d:+.0f}.png") if (work / f"orbit_{d:+.0f}.png").exists()
                  else black for d in ORBITS]
        cam = json.loads((work / "camera.json").read_text())
        k = (cam["fx"], cam["fy"], cam["cx"], cam["cy"])
        geo = dict(np.load(work / "geometry.npz"))
        v = camera_frame(geo["vertices"], geo["c2w"])
        up = (geo["c2w"][:3, :3] @ GL_TO_CV)[2]              # the world's +z in the camera frame
        f, owner = geo["faces"].astype(np.int64), geo["face_object"]
        names = [str(x) for x in geo["names"]]
        meshes = [(v, f[owner == m]) for m in range(len(names))]
        depth, index, assigned = assign(meshes, names, k, size, gt["label"], device)
        rows = object_rows(gt["n"], meshes, index, assigned, gt["label"], names)
        row.update(fov=round(float(np.degrees(2 * np.arctan(W / 2 / k[0]))), 2),
                   objects=audit["meshes"], lights=audit["lights"], triangles=audit["triangles"],
                   materials_replaced=len(audit["materials_replaced"]),
                   rule_changes=audit["changes"][:30], replaced=audit["materials_replaced"][:20],
                   hidden=audit["objects_hidden"][:20], view_transform=audit["view_transform"])
        label_np = gt["label"].cpu().numpy()
        for r in rows:
            r["realness"] = 0.0
            if not r["mine"]:
                continue
            o = objs[r["object"]]
            fo = np.concatenate([meshes[m][1] for m in r["mine"]])
            res = models["judge"].realness(v, fo, k, size, up, crop_of(photo_full, label_np == r["object"]),
                                           o["label"], bool(o.get("boxy")))
            r.update(realness=res["realness"], judge_score=res["score"],
                     judge_vs={a: round(b, 4) for a, b in res["vs"].items()}, views=res["views"])
        if easy:
            absrel, s_depth = depth_absrel(depth, gt["depth"])
            shape_scores(rows, meshes, s_depth, gt, k_true)
            furniture = [m for m, nm in enumerate(names) if not is_room(nm)]
            ff = (np.concatenate([meshes[m][1] for m in furniture]) if furniture
                  else np.zeros((0, 3), np.int64))
            paper = metrics.paper_geometry(v, ff, gt["vertices"])
            row.update(depth_absrel=absrel, scale=s_depth, **{f"paper_{a}": b for a, b in paper.items()})
    except Exception as exc:  # noqa: BLE001 - the scene is empty, with the reason
        kind = "" if isinstance(exc, ValueError) else f"{type(exc).__name__}: "
        row["error"] = (kind + str(exc))[:500]
        render, orbits, index, assigned, depth = black, [black, black], None, {}, None
        rows = [{"object": j, "mine": [], "meshes": [], "iou": 0.0, "realness": 0.0}
                for j in range(gt["n"])]
        if easy:
            row.update(depth_absrel=1.0, **{f"paper_{a}": b for a, b in
                                            metrics.paper_geometry(np.zeros((0, 3)), np.zeros((0, 3), np.int64),
                                                                   gt["vertices"]).items()})
    if easy:
        for r in rows:
            if "f" not in r:
                r.update(f=0.0, f_hidden=0.0, f10=0.0, cd=metrics.CD_EMPTY_CM, precision=0.0,
                         recall=0.0, recall_hidden=0.0)
    a, b = to_tensor(render, device), to_tensor(photo, device)
    row.update(n_clip=models["clip"](render, photo), pl=metrics.pl(render, photo),
               lpips=models["lpips"](a, b), psnr=metrics.psnr(a, b), ssim=metrics.ssim(a, b))
    row["appearance_term"] = (max(0.0, 1 - row["n_clip"] / NCLIP_ZERO)
                              + max(0.0, 1 - row["pl"] / PL_ZERO) + 1 - row["lpips"]) / 3

    def avg(key):
        return float(np.mean([r[key] for r in rows])) if rows else 0.0
    if rows:          # a photo without annotated furniture has no object terms
        row.update(object_iou=avg("iou"), object_realness=avg("realness"),
                   object_found=float(np.mean([bool(r["mine"]) for r in rows])),
                   object_term=float(np.mean([r["realness"] * min(1.0, r["iou"] / IOU_FULL)
                                              for r in rows])))
    if easy:
        row.update(object_f=avg("f"), object_f_hidden=avg("f_hidden"), object_f10=avg("f10"),
                   object_cd=avg("cd"), object_precision=avg("precision"),
                   object_recall=avg("recall"), object_recall_hidden=avg("recall_hidden"))
        row["geometry_term"] = (row["paper_f@0.1"] + row["paper_f@0.01"] + row["object_f"]
                                + row["object_f_hidden"]) / 400
        sixth = error_image(depth * row.get("scale", 1.0) if depth is not None else None, gt["depth"])
    else:
        big = max((r for r in rows if r.get("views") is not None),
                  key=lambda r: int((gt["label"] == r["object"]).sum()), default=None)
        sixth = fit_image(big["views"], size) if big else np.zeros_like(photo)
    sheet = [photo, render, assignment_image(photo, index, assigned, gt["label"]), orbits[0],
             orbits[1], sixth]
    objects = [{key: val for key, val in r.items() if key not in ("mine", "views")} for r in rows]
    return row, sheet, objects


# --------------------------------------------------------------------------- the split
def load_models(scorer: Path, device) -> dict:
    sys.path.insert(0, str(scorer / "pylib"))
    import judge
    return {"lpips": metrics.Lpips(scorer, device), "clip": metrics.Clip(scorer, device),
            "judge": judge.Judge(scorer, device)}


def main() -> None:
    split_dir = Path(os.environ["RSIB_DATA_DIR"])
    workspace = Path(os.environ["RSIB_WORKSPACE"])
    out_dir = Path(os.environ["RSIB_OUTPUT_DIR"])
    scorer = Path(os.environ["RSIB_TRAIN_DIR"]) / "scorer"  # the dev data: in every evaluation
    blender_dir = scorer / "blender"
    python = os.environ.get("RSIB_PYTHON", sys.executable)
    runtime = workspace / "runtime.txt"  # optional: another interpreter (inside $RSIB_CACHE)
    if runtime.exists() and runtime.read_text().strip():
        python = runtime.read_text().strip().splitlines()[0].strip()

    scenes = sorted(p.name for p in (split_dir / "inputs").iterdir() if p.is_dir())
    limit = SECONDS_PER_SCENE * len(scenes) + STARTUP_S
    (out_dir / "renders").mkdir(exist_ok=True)
    cache = out_dir / ".blender-cache"       # the evaluator's own: the solution cannot write it
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="scene3d-", dir=out_dir) as tmp:
        io_dir = Path(tmp)
        for s in scenes:
            (io_dir / "inputs" / s).mkdir(parents=True)
            shutil.copy(split_dir / "inputs" / s / "photo.png", io_dir / "inputs" / s / "photo.png")
        seconds, timed_out, code = run_solution(python, io_dir, workspace, blender_dir, limit,
                                                out_dir / "solution.log")
        log_tail = (out_dir / "solution.log").read_text(errors="replace")[-6000:]
        print(log_tail[-3000:])
        if code != 0 and not timed_out:
            sys.exit(f"the solution's process exited with status {code}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models = load_models(scorer, device)
        rows, per_object = [], {}
        with tempfile.TemporaryDirectory(prefix="blender-", dir=out_dir) as wtmp:
            for s in scenes:
                row, sheet, objects = score_scene(s, split_dir, io_dir / "out" / s, Path(wtmp) / s,
                                                  blender_dir, cache, models, device)
                row = {"id": s, **row}
                per_object[s] = objects
                contact_sheet(sheet, out_dir / "renders" / f"{s}.jpg")
                rows.append(row)
                print(f"{s} ({row['difficulty']}): appearance {row['appearance_term']:.3f} "
                      f"objects {row.get('object_term', float('nan')):.3f}"
                      + (f" geometry {row['geometry_term']:.3f}" if "geometry_term" in row else "")
                      + (f" error: {row['error']}" if row["error"] else ""), flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    def mean(key, group=None):
        vals = [r[key] for r in (rows if group is None else group) if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    easy = [r for r in rows if r["difficulty"] == "easy"]
    loaded = [r for r in rows if r["error"] is None]
    reward = {k: mean(k) for k in ("appearance_term", "object_term", "n_clip", "pl", "lpips",
                                   "psnr", "ssim", "object_realness", "object_iou",
                                   "object_found")}
    reward.update({k: mean(k, easy) for k in (
        "geometry_term", "paper_chamfer", "paper_f@0.1", "paper_f@0.01", "paper_f@0.001",
        "object_f", "object_f_hidden", "object_f10", "object_cd", "object_precision",
        "object_recall", "object_recall_hidden", "depth_absrel")})
    reward["score"] = (reward["appearance_term"] + reward["object_term"]
                       + (reward["geometry_term"] or 0.0)) / 3
    for d in ("easy", "medium", "complicated"):
        group = [r for r in rows if r["difficulty"] == d]
        if group:
            reward[f"appearance_term_{d}"] = mean("appearance_term", group)
            reward[f"object_term_{d}"] = mean("object_term", group)
    reward.update(n=len(rows), n_easy=len(easy), failed=len(rows) - len(loaded),
                  objects=mean("objects", loaded), lights=mean("lights", loaded),
                  triangles=mean("triangles", loaded),
                  materials_replaced=mean("materials_replaced", loaded),
                  fov=mean("fov", loaded), render_s=mean("render_s", loaded),
                  seconds=round(seconds, 1), timed_out=int(timed_out))
    reward = {k: v for k, v in reward.items() if v is not None}
    for r in rows:
        r["per_object"] = per_object.get(r["id"], [])
    (out_dir / "per_scene.json").write_text(json.dumps(rows, indent=1))
    reward["artifacts"] = (
        [{"path": "per_scene.json", "role": "diagnosis", "note": "every photo's metrics, its "
          "objects' scores and the judge's answers, the rule changes Blender made and why a "
          "scene did not load"},
         {"path": "solution.log", "role": "log", "note": "output of the reconstruct() process"}]
        + [{"path": f"renders/{r['id']}.jpg", "role": "render",
            "note": "photo, render, mesh objects colored by the object they were assigned to / "
                    "orbit -30, orbit +30 (not scored), depth error (3D-FRONT renders) or the "
                    "judge's clay views of the largest annotated object (real photos)"}
           for r in rows])
    (out_dir / "reward.json").write_text(json.dumps(reward))
    print(json.dumps({k: v for k, v in reward.items() if k != "artifacts"}))


if __name__ == "__main__":
    main()
