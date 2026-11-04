"""Evaluator: reconstruct every scene of a split in a child process, then render each scene from all
of its input cameras and score it with the metrics of CoDimRecon (arXiv 2609.36024, Appendix C).

v3 follows CoDimRecon's setting: the child (child_reconstruct.py, through the sandbox launcher when
there is one) sees only the scenes' frames, inputs/<id>/images/. For each it calls
reconstruct(scene_dir, out_dir) of the workspace, which writes out_dir/scene.glb and
out_dir/cameras.json, its estimate of every frame's camera in the scene's own frame and scale. The
split must finish within SECONDS_PER_SCENE x n + STARTUP_S seconds; then the child is stopped and
the scenes it has not written count as empty. A child that exits with an error fails the evaluation.

Per scene (a scene that does not load, or whose cameras.json is missing or invalid, renders black
and has no surface and no objects):
    placement   align.py: the similarity that maps the solution's cameras onto the true ones (the
                rotation from their orientations, scale and translation from their centers) moves
                the scene into the ground truth's frame; everything below is scored there, with
                the true cameras (labels/<id>/transforms.json), masks (labels/<id>/masks/) and
                gravity. Reported: camera_rot_err (degrees), camera_ate (cm), camera_scale
    rendering   every input frame is rendered (raster.py) at the input resolution from its
                camera; psnr, ssim and lpips against the frame, averaged over the frames
    geometry    cd (cm), f1 (at 5 cm, percent) and nc (percent) between SAMPLES points of the
                whole scene and of the ground-truth mesh (metrics.py)
    objects     every glTF node is assigned to the label (0: background, k: annotated object k of
                the masks) that covers most of the pixels it shows over all frames; the nodes of
                one object together are that object's reconstruction, its iou the intersection
                over union of their pixels and the object's mask over all frames. object_iou is
                the mean over the annotated objects, object_recall the share with iou >= IOU_MATCH
    stability   physics.py: the nodes of the background, simplified to at most 100,000 faces,
                become fixed slabs, the nodes of each object one rigid body, without its faces
                outside the object's ground-truth bounding box grown by BOX_MARGIN (so that a base
                hidden under the floor, or floor labeled as the object, does not hold it up); each
                dynamic object (labels/<id>/objects.json) is dropped with everything else fixed.
                stable_all and stable_ground follow the paper: the share of the dynamic objects
                the scene reconstructs (and of those whose lowest point is within 5 cm of the
                floor) that stay in place. stable_all_gt counts every dynamic object of the ground
                truth: it is stable when its iou is at least IOU_MATCH and it stays in place
    latent      layout and motion: V-JEPA 2.1 ViT-G similarity of 64 input frames and the same
                views rendered (metrics.py)
    score       primary, higher is better, in [0, 1]:
                (geometry_term + appearance_term + physics_term) / 3 with
                geometry_term = (f1 + nc) / 200, appearance_term = (ssim + 1 - lpips) / 2 and
                physics_term = stable_all_gt / 100
Metrics are averaged over the scenes; also n, failed (scenes that do not load), nodes and
triangles (mean over the scenes that load), coverage (share of pixels that see a surface),
seconds (the child's run time) and timed_out.
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
import align  # noqa: E402
import gltf  # noqa: E402
import metrics  # noqa: E402
import physics  # noqa: E402
import raster  # noqa: E402

SECONDS_PER_SCENE = 1200          # v3: the cameras are estimated too
STARTUP_S = 600
MAX_GLB_MB = 512
IOU_MATCH = 0.5
GROUND_TOL = 0.05
BOX_MARGIN = 0.10                 # m: an object's faces outside its box grown by this do not collide
CONTACT_FRAMES = 4
SOURCES = ("replica", "scannetpp", "gibson", "custom")
_PALETTE = np.random.default_rng(7).integers(40, 256, (4096, 3)).astype(np.uint8)
_PALETTE[0] = (90, 90, 90)


def solution_cmd(python: str, io_dir: Path) -> list[str]:
    """The solution's process: through the sandbox launcher when the orchestrator gives one, so it
    sees its own code, the run cache and io_dir, never the labels."""
    child = io_dir / "_child_reconstruct.py"
    shutil.copy(Path(__file__).with_name("child_reconstruct.py"), child)
    cmd = [python, str(child)]
    prefix = json.loads(os.environ.get("RSIB_SANDBOX") or "[]")
    return prefix + ["--rw", str(io_dir), "--cwd", str(io_dir), "--"] + cmd if prefix else cmd


def run_solution(python: str, io_dir: Path, workspace: Path, limit: float, log_path: Path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("RSIB_")}
    env.update({"RSIB_IO_DIR": str(io_dir), "RSIB_SOLVER_DIR": str(workspace),
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"})
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(solution_cmd(python, io_dir), env=env, cwd=io_dir, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            proc.wait(timeout=limit)
            timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            timed_out = True
    return time.time() - t0, timed_out, proc.returncode


def load_scene(out: Path) -> gltf.Scene:
    glb = out / "scene.glb"
    if not glb.is_file():
        raise ValueError("no scene.glb")
    if glb.stat().st_size > MAX_GLB_MB * 2 ** 20:
        raise ValueError(f"scene.glb is larger than {MAX_GLB_MB} MB")
    scene = gltf.load(str(glb))
    for p in scene.primitives:
        if not np.isfinite(p.positions).all():
            raise ValueError(f"node {p.name!r} has a vertex that is not finite")
    return scene


def node_meshes(scene: gltf.Scene) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """{node: (world vertices, faces)} with every primitive of the node together."""
    parts: dict[int, list] = {}
    for p in scene.primitives:
        parts.setdefault(p.node, []).append(p)
    out = {}
    for node, prims in parts.items():
        offsets = np.cumsum([0] + [len(p.positions) for p in prims[:-1]])
        out[node] = (np.concatenate([p.positions for p in prims]),
                     np.concatenate([p.faces + o for p, o in zip(prims, offsets)]))
    return out


def merge(meshes: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    if not meshes:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    offsets = np.cumsum([0] + [len(v) for v, _ in meshes[:-1]])
    return (np.concatenate([v for v, _ in meshes]),
            np.concatenate([f + o for (_, f), o in zip(meshes, offsets)]))


def frame_list(labels: Path) -> tuple[dict, list[dict]]:
    t = json.loads((labels / "transforms.json").read_text())
    return t, t["frames"]


def to_uint8(img: torch.Tensor) -> np.ndarray:
    return (img.clamp(0, 1) * 255 + 0.5).to(torch.uint8).cpu().numpy()


def contact_sheet(rows: list[list[np.ndarray]], path: Path) -> None:
    """One row per frame: the input frame, its render, the annotated masks, the rendered labels."""
    h, w = rows[0][0].shape[:2]
    scale = min(1.0, 384 / w)
    h2, w2 = max(1, round(h * scale)), max(1, round(w * scale))
    sheet = np.zeros((len(rows) * (h2 + 4), 4 * (w2 + 4), 3), np.uint8)
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            im = np.asarray(Image.fromarray(im).resize((w2, h2), Image.BILINEAR))
            sheet[r * (h2 + 4):r * (h2 + 4) + h2, c * (w2 + 4):c * (w2 + 4) + w2] = im
    Image.fromarray(sheet).save(path, quality=88)


def score_scene(scene_dir: Path, labels: Path, out: Path, lpips: metrics.Lpips,
                vjepa: metrics.VJepa, device: torch.device, pylib: Path, sheet: Path) -> dict:
    meta = json.loads((labels / "objects.json").read_text())
    floor_z = float(meta["floor_z"])
    objects = {int(o["id"]): o for o in meta["objects"]}
    n_labels = 1 + max(objects, default=0)
    t, frames = frame_list(labels)
    H, W = int(t["h"]), int(t["w"])
    fx, fy, cx, cy = (float(t[k]) for k in ("fl_x", "fl_y", "cx", "cy"))
    row: dict = {"error": None}
    clock = {"start": time.time()}
    try:
        scene = load_scene(out)
        true_c2w = np.stack([np.asarray(fr["transform_matrix"], np.float64) for fr in frames])
        s, rot, trans, cams = align.similarity(align.read_cameras(out / "cameras.json", frames), true_c2w)
        for p in scene.primitives:
            p.positions = s * p.positions @ rot.T + trans
        row.update(cams)
        renderer = raster.Renderer(scene, device)
        row.update(nodes=scene.meshes, triangles=scene.triangles)
    except Exception as exc:  # noqa: BLE001 - the scene is empty, with the reason
        kind = "" if isinstance(exc, ValueError) else f"{type(exc).__name__}: "
        row.update(error=(kind + str(exc))[:500])
        scene = gltf.Scene([], [], [], 0, 0)
        renderer = raster.Renderer(scene, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    n_nodes = len(renderer.node_ids) if not renderer.empty else 0

    # rendering and label overlap, frame by frame
    conf = torch.zeros((n_nodes + 1) * n_labels, dtype=torch.int64, device=device)
    ls_idx = set(metrics.ls_indices(len(frames)).tolist())
    contact_idx = set(np.linspace(0, len(frames) - 1, CONTACT_FRAMES).round().astype(int).tolist())
    ls_src, ls_ren, contact, scores = {}, {}, [], {"psnr": [], "ssim": [], "lpips": [], "coverage": []}
    for i, fr in enumerate(frames):
        src_u8 = np.array(Image.open(scene_dir / fr["file_path"]).convert("RGB"))
        mask = np.array(Image.open(labels / fr["mask_path"]), np.int64)
        src = torch.as_tensor(src_u8, device=device).float() / 255.0
        img, node, cov = renderer.render(np.asarray(fr["transform_matrix"]), fx, fy, cx, cy, H, W)
        scores["psnr"].append(metrics.psnr(img, src))
        scores["ssim"].append(metrics.ssim(img, src))
        scores["lpips"].append(lpips(img, src))
        scores["coverage"].append(float(cov.mean()))
        m = torch.as_tensor(np.minimum(mask, n_labels - 1), device=device)
        conf += torch.bincount(((node + 1) * n_labels + m).reshape(-1), minlength=conf.numel())
        if i in ls_idx or i in contact_idx:
            ren_u8 = to_uint8(img)
            if i in ls_idx:
                ls_src[i], ls_ren[i] = src_u8, ren_u8
            if i in contact_idx:
                contact.append((src_u8, ren_u8, mask, node.cpu().numpy()))
    row.update({k: float(np.mean(v)) for k, v in scores.items()})
    conf = conf.view(n_nodes + 1, n_labels).cpu().numpy()
    clock["render"] = time.time()

    # nodes -> labels, objects' iou
    shown = conf[1:].sum(1)
    assign = {}                                   # node index -> label
    for n in range(n_nodes):
        if shown[n] > 0:
            assign[n] = int(np.argmax(conf[n + 1]))
    gt_pixels = conf.sum(0)
    per_object = {}
    for k, o in objects.items():
        members = [n for n, lab in assign.items() if lab == k]
        inter = int(conf[[n + 1 for n in members], k].sum()) if members else 0
        union = int(shown[members].sum()) + int(gt_pixels[k]) - inter if members else int(gt_pixels[k])
        per_object[k] = {"id": k, "category": o["category"], "ground": o["ground"],
                         "nodes": [renderer.node_names[renderer.node_ids[n]] for n in members][:20],
                         "iou": inter / union if union else 0.0}
    seen = [k for k in objects if gt_pixels[k] > 0]
    row["object_iou"] = float(np.mean([per_object[k]["iou"] for k in seen])) if seen else 0.0
    row["object_recall"] = float(np.mean([per_object[k]["iou"] >= IOU_MATCH for k in seen])) if seen else 0.0
    row["unseen_nodes"] = n_nodes - len(assign)
    if contact:
        lut = np.full(n_nodes + 1, -1, np.int64)
        for n, lab in assign.items():
            lut[n + 1] = lab
        sheet_rows = []
        for src_u8, ren_u8, mask, node in contact:
            lab = lut[node + 1]
            colored = np.where((lab >= 0)[..., None], _PALETTE[np.maximum(lab, 0) % 4096], 0).astype(np.uint8)
            sheet_rows.append([src_u8, ren_u8, _PALETTE[mask % 4096], colored])
        contact_sheet(sheet_rows, sheet)

    # geometry
    meshes = node_meshes(scene)
    pred_v, pred_f = merge(list(meshes.values()))
    gt = np.load(labels / "gt.npz")
    row.update(metrics.geometry(metrics.sample_surface(pred_v, pred_f),
                                metrics.sample_surface(gt["vertices"], gt["faces"])))
    clock["geometry"] = time.time()

    # stability
    groups: dict[int, list] = {}
    for n, lab in assign.items():
        groups.setdefault(lab, []).append(meshes[renderer.node_ids[n]])
    env = physics.worker_env(pylib)
    background = physics.simplify_background(*merge(groups.pop(0, [])), sys.executable, env)
    bodies = {}
    for k, ms in groups.items():
        if k in objects:
            v, f = merge(ms)
            lo, hi = (np.asarray(b, np.float64) for b in objects[k]["bbox"])
            inside = ((v >= lo - BOX_MARGIN) & (v <= hi + BOX_MARGIN)).all(1)
            used, f = np.unique(f[inside[f].all(1)], return_inverse=True)
            bodies[k] = (v[used], f.reshape(-1, 3))
    workers = max(1, min(16, os.cpu_count() or 1))
    hulls = physics.decompose_all(bodies, sys.executable, env, workers)
    slabs = physics.background_slabs(*background)
    tested = {k: float(np.linalg.norm(v.max(0) - v.min(0))) if len(v) else 0.0
              for k, (v, _) in bodies.items() if objects[k]["category"] == "dynamic"}
    results = physics.drop_tests(tested, {k: h for k, (h, _) in hulls.items()}, slabs, floor_z,
                                 sys.executable, env, workers)
    for k, res in results.items():
        v = bodies[k][0]
        per_object[k].update(tested=True, collision=hulls[k][1], hulls=len(hulls[k][0]),
                             on_floor=bool(len(v) and v[:, 2].min() - floor_z <= GROUND_TOL), **res)
    dyn = [k for k, o in objects.items() if o["category"] == "dynamic" and gt_pixels[k] > 0]
    fallen = [results[k]["fallen"] for k in results]
    ground = [results[k]["fallen"] for k in results if per_object[k]["on_floor"]]
    row["stable_all"] = 100.0 * (1 - np.mean(fallen)) if fallen else None
    row["stable_ground"] = 100.0 * (1 - np.mean(ground)) if ground else None

    def stable(k):
        return per_object[k]["iou"] >= IOU_MATCH and k in results and not results[k]["fallen"]

    row["stable_all_gt"] = 100.0 * float(np.mean([stable(k) for k in dyn])) if dyn else 0.0
    dyn_ground = [k for k in dyn if objects[k]["ground"]]
    row["stable_ground_gt"] = 100.0 * float(np.mean([stable(k) for k in dyn_ground])) if dyn_ground else None
    row["dynamic_objects"] = len(dyn)
    row["background_faces"] = len(background[1])
    row["background_slabs"] = len(slabs)
    clock["stability"] = time.time()

    # latent similarity
    idx = metrics.ls_indices(len(frames))
    row.update(vjepa.similarity([ls_src[i] for i in idx], [ls_ren[i] for i in idx]))
    clock["latent"] = time.time()
    stages = list(clock)
    row["stage_seconds"] = {b: round(clock[b] - clock[a], 1) for a, b in zip(stages, stages[1:])}

    row["geometry_term"] = (row["f1"] + row["nc"]) / 200.0
    row["appearance_term"] = (row["ssim"] + 1.0 - row["lpips"]) / 2.0
    row["physics_term"] = row["stable_all_gt"] / 100.0
    row["score"] = (row["geometry_term"] + row["appearance_term"] + row["physics_term"]) / 3.0
    row["objects"] = [per_object[k] for k in sorted(per_object)]
    return row


def main() -> None:
    split_dir = Path(os.environ["RSIB_DATA_DIR"])
    workspace = Path(os.environ["RSIB_WORKSPACE"])
    out_dir = Path(os.environ["RSIB_OUTPUT_DIR"])
    weights = Path(os.environ["RSIB_TRAIN_DIR"]) / "scorer"  # the dev data: mounted in every evaluation
    python = os.environ.get("RSIB_PYTHON", sys.executable)
    runtime = workspace / "runtime.txt"  # optional: another interpreter (inside $RSIB_CACHE)
    if runtime.exists() and runtime.read_text().strip():
        python = runtime.read_text().strip().splitlines()[0].strip()

    scenes = sorted(p.name for p in (split_dir / "inputs").iterdir() if p.is_dir())
    limit = SECONDS_PER_SCENE * len(scenes) + STARTUP_S
    (out_dir / "renders").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="holoscene-", dir=out_dir) as tmp:
        io_dir = Path(tmp)
        for s in scenes:
            shutil.copytree(split_dir / "inputs" / s, io_dir / "inputs" / s)
        seconds, timed_out, code = run_solution(python, io_dir, workspace, limit,
                                                out_dir / "solution.log")
        log_tail = (out_dir / "solution.log").read_text(errors="replace")[-6000:]
        print(log_tail[-3000:])
        if code != 0 and not timed_out:
            sys.exit(f"the solution's process exited with status {code}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lpips = metrics.Lpips(weights, device)
        vjepa = metrics.VJepa(weights, device)
        rows = []
        for s in scenes:
            source = json.loads((split_dir / "labels" / s / "meta.json").read_text())["source"]
            t0 = time.time()
            row = score_scene(split_dir / "inputs" / s, split_dir / "labels" / s, io_dir / "out" / s,
                              lpips, vjepa, device, weights / "pylib", out_dir / "renders" / f"{s}.jpg")
            row = {"id": s, "source": source, **row, "eval_seconds": round(time.time() - t0, 1)}
            rows.append(row)
            print(f"{s} ({source}): score {row['score']:.4f} | cd {row['cd']:.2f} f1 {row['f1']:.2f} "
                  f"nc {row['nc']:.2f} | psnr {row['psnr']:.2f} ssim {row['ssim']:.4f} lpips "
                  f"{row['lpips']:.4f} | stable_all_gt {row['stable_all_gt']:.1f} object_recall "
                  f"{row['object_recall']:.2f} | layout {row['layout']:.2f} motion {row['motion']:.2f}"
                  + (f" | error: {row['error']}" if row["error"] else ""), flush=True)

    def mean(key, group=rows):
        vals = [r[key] for r in group if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    loaded = [r for r in rows if r["error"] is None]
    keys = ("score", "geometry_term", "appearance_term", "physics_term", "cd", "f1", "nc",
            "precision", "recall", "psnr", "ssim", "lpips", "layout", "motion", "stable_all",
            "stable_ground", "stable_all_gt", "stable_ground_gt", "object_iou", "object_recall",
            "coverage", "camera_rot_err", "camera_ate")
    reward = {k: mean(k) for k in keys}
    reward.update(n=len(rows), failed=len(rows) - len(loaded), nodes=mean("nodes", loaded),
                  triangles=mean("triangles", loaded), seconds=round(seconds, 1),
                  timed_out=int(timed_out))
    for src in SOURCES:
        reward[f"score_{src}"] = mean("score", [r for r in rows if r["source"] == src])
    reward = {k: v for k, v in reward.items() if v is not None}
    (out_dir / "per_scene.json").write_text(json.dumps(rows, indent=1))
    reward["artifacts"] = (
        [{"path": "per_scene.json", "role": "diagnosis", "note": "every scene's metrics, every "
          "annotated object's iou, collision geometry and drop test, and why a scene did not load"},
         {"path": "solution.log", "role": "log", "note": "output of the reconstruct() process"}]
        + [{"path": f"renders/{r['id']}.jpg", "role": "render",
            "note": "four input frames: the frame, its render, the annotated masks, the labels the "
                    "rendered nodes are assigned to"} for r in rows])
    (out_dir / "reward.json").write_text(json.dumps(reward))
    print(json.dumps({k: v for k, v in reward.items() if k != "artifacts"}))


if __name__ == "__main__":
    main()
