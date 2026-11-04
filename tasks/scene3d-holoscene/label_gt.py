"""Second step of the data preparation (after prepare.py), on a GPU in the task's container:

    python label_gt.py --out <data root>/scene3d-holoscene

1. The scorer's V-JEPA 2.1 ViT-G/16 encoder: downloads vjepa2_1_vitG_384.pt (the full training
   checkpoint, about 30 GB, checked against its SHA-256) into <out>/downloads, keeps its
   target_encoder in bf16 as dev/scorer/vjepa2_1_vitG_384_encoder.pt and deletes the checkpoint.
2. The ground truth of every scene, from labels/<id>/: mesh.ply, masks/ and transforms.json:
   - every face of the mesh gets the label (0 background, k object k) that most of the pixels it
     is the nearest surface of show in masks/ over all frames; -1 when it is never the nearest.
     labels/<id>/gt.npz: vertices (float32), faces (int32), face_label (int16).
   - the floor height: the level (1 cm bins) with the most area of near-horizontal background
     faces below the median camera height.
   - per annotated object (labels/<id>/objects.json): its `parent` in the support tree of
     graph.json (HoloScene's annotation, read breadth first from the background, node 0: an
     object rests on its parent), its lowest point above the floor (`z_min`: the lowest vertex
     of its labeled faces; `z_low`: their 0.2nd percentile) and its category. `dynamic` (dropped
     in the stability test): it rests on another object, or on the background within
     HANG_TOL of the floor; `static` (fixed): a child of the background hanging higher (from a
     wall or the ceiling). `ground`: its lowest point is within GROUND_TOL of the floor
     (z_low; for a child of the background, z_min within HANG_TOL), the paper's ground objects.
"""

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))
from plyio import read_ply  # noqa: E402
from raster import Renderer, _INF_KEY  # noqa: E402

VJEPA_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt"
VJEPA_SHA256 = "7aae1a3c7a31d258af9c985388b5d2f20587469380f2e11b54d7876ac8cfe58a"
VJEPA_KEY = "target_encoder"
ENCODER_FILE = "vjepa2_1_vitG_384_encoder.pt"
GROUND_TOL = 0.05
HANG_TOL = 0.10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def vjepa_encoder(out: Path) -> None:
    dst = out / "dev" / "scorer" / ENCODER_FILE
    if dst.exists():
        return
    ckpt = out / "downloads" / "vjepa2_1_vitG_384.pt"
    if not ckpt.exists():
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {VJEPA_URL} (about 30 GB)", flush=True)
        tmp = ckpt.with_name(ckpt.name + ".part")
        with urllib.request.urlopen(urllib.request.Request(VJEPA_URL, headers={"User-Agent": "rsib-prepare"}),
                                    timeout=600) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 24)
        tmp.replace(ckpt)
    digest = sha256(ckpt)
    if VJEPA_SHA256 and digest != VJEPA_SHA256:
        raise SystemExit(f"{ckpt.name}: SHA-256 {digest} is not the pinned {VJEPA_SHA256}")
    state = torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)[VJEPA_KEY]
    state = {k.replace("module.", "").replace("backbone.", ""): v.to(torch.bfloat16).clone()
             for k, v in state.items()}
    tmp = dst.with_name(dst.name + ".part")
    torch.save(state, tmp)
    tmp.replace(dst)
    print(f"{dst.name}: {len(state)} tensors, {sum(v.numel() for v in state.values()) / 1e9:.2f} B "
          f"parameters, sha256 {sha256(dst)} (checkpoint {digest})", flush=True)
    ckpt.unlink()


def face_labels(verts: np.ndarray, faces: np.ndarray, lab: Path, device: torch.device):
    t = json.loads((lab / "transforms.json").read_text())
    H, W = int(t["h"]), int(t["w"])
    fx, fy, cx, cy = (float(t[k]) for k in ("fl_x", "fl_y", "cx", "cy"))
    pos = torch.as_tensor(verts, dtype=torch.float64, device=device)
    fac = torch.as_tensor(faces, dtype=torch.int64, device=device)
    masks = [np.asarray(Image.open(lab / fr["mask_path"])) for fr in t["frames"]]
    n_labels = int(max(m.max() for m in masks)) + 1
    votes = torch.zeros(len(faces) * n_labels, dtype=torch.float32, device=device)
    pixels = np.zeros(n_labels, np.int64)
    for fr, mask in zip(t["frames"], masks):
        xy, z, orig, _ = Renderer.project(pos, fac, np.asarray(fr["transform_matrix"]), fx, fy, cx, cy)
        best = Renderer.rasterize(xy, z, H, W)
        hit = best != _INF_KEY
        f = orig[best[hit] & 0xFFFFFFFF]
        lab = torch.as_tensor(mask.astype(np.int64), device=device).reshape(-1)[hit]
        votes.index_add_(0, f * n_labels + lab, torch.ones(len(f), device=device))
        pixels += np.bincount(mask.reshape(-1), minlength=n_labels)[:n_labels]
    votes = votes.view(len(faces), n_labels)
    label = votes.argmax(1).cpu().numpy().astype(np.int16)
    label[(votes.sum(1) == 0).cpu().numpy()] = -1
    return label, pixels, t


def support_tree(graph: list[dict]) -> dict[int, int]:
    """Parent of every node of graph.json, breadth first from the background (node 0)."""
    adj = defaultdict(set)
    for node in graph:
        for a in node["adj_nodes"]:
            adj[node["node_id"]].add(a)
            adj[a].add(node["node_id"])
    parent, queue = {0: -1}, deque([0])
    while queue:
        u = queue.popleft()
        for v in sorted(adj[u]):
            if v not in parent:
                parent[v] = u
                queue.append(v)
    return parent


def floor_height(verts, faces, label, cams_z) -> float:
    tri = verts[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1)
    nz = np.abs(cross[:, 2]) / np.maximum(area, 1e-30)
    z = tri[:, :, 2].mean(1)
    sel = (label == 0) & (nz > 0.95) & (z < np.median(cams_z))
    if not sel.any():
        sel = (nz > 0.95) & (z < np.median(cams_z))
    zs, a = z[sel], area[sel]
    bins = np.floor(zs / 0.01).astype(np.int64)
    uniq, inv = np.unique(bins, return_inverse=True)
    acc = np.bincount(inv, weights=a)
    k = uniq[int(np.argmax(acc))]
    return float(np.average(zs[bins == k], weights=a[bins == k]))


def objects_of(verts, faces, label, pixels, floor_z, n_objects, parent) -> list[dict]:
    out = []
    for k in range(1, n_objects + 1):
        p = parent.get(k, 0)
        o = {"id": k, "parent": p, "pixels": int(pixels[k]) if k < len(pixels) else 0,
             "faces": int((label == k).sum())}
        vid = np.unique(faces[label == k])
        if len(vid) == 0:
            o.update(category="dynamic", ground=p == 0, z_min=None, z_low=None,
                     note="no labeled face: placed by its parent only")
            out.append(o)
            continue
        z = verts[vid, 2] - floor_z
        z_min, z_low = float(z.min()), float(np.percentile(z, 0.2))
        hangs = p == 0 and z_min > HANG_TOL
        ground = z_low <= GROUND_TOL or (p == 0 and z_min <= HANG_TOL)
        o.update(category="static" if hangs else "dynamic", ground=bool(ground),
                 z_min=round(z_min, 4), z_low=round(z_low, 4),
                 bbox=[verts[vid].min(0).round(3).tolist(), verts[vid].max(0).round(3).tolist()])
        out.append(o)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-vjepa", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if not args.skip_vjepa:
        vjepa_encoder(out)
    device = torch.device("cuda")
    for split in ("dev", "val", "test"):
        for scene_dir in sorted((out / split / "inputs").iterdir()):
            lab = out / split / "labels" / scene_dir.name
            verts, faces = read_ply(str(lab / "mesh.ply"))
            label, pixels, t = face_labels(verts, faces, lab, device)
            graph = json.loads((lab / "graph.json").read_text())
            n_objects = max(len(graph) - 1, len(pixels) - 1)
            cams_z = np.array([fr["transform_matrix"][2][3] for fr in t["frames"]])
            floor_z = floor_height(verts, faces, label, cams_z)
            objs = objects_of(verts, faces, label, pixels, floor_z, n_objects, support_tree(graph))
            np.savez_compressed(lab / "gt.npz", vertices=verts.astype(np.float32),
                                faces=faces.astype(np.int32), face_label=label)
            (lab / "objects.json").write_text(json.dumps({"floor_z": floor_z, "objects": objs}, indent=1))
            cats = {c: sum(o["category"] == c for o in objs) for c in ("dynamic", "static")}
            cats["ground"] = sum(o["ground"] for o in objs)
            print(f"{split}/{scene_dir.name}: {len(faces)} faces, {(label >= 0).mean():.1%} labeled, "
                  f"floor {floor_z:.3f}, objects {len(objs)} {cats}, graph nodes {len(graph)}", flush=True)


if __name__ == "__main__":
    main()
