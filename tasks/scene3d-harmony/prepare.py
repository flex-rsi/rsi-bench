"""Build the scene3d-harmony splits under --out from HARMONY300's 300 photos (v3).
Sources, downloaded at pinned revisions (not part of the task package):
    huggingface.co/datasets/ShufanSun/harmony   HARMONY300: only its input photos (input.*) and
                                                attribution.csv (difficulty, name, source); the
                                                download also holds a method's reconstructions,
                                                which are never fetched
    huggingface.co/datasets/andreead-a/FRONT3D  the ground truth of the 100 3D-FRONT renders (the
                                                "easy" third, the same pixels as their HARMONY300
                                                photos): camera, depth, the depth of the room
                                                without its furniture (bgdepth, 648 x 484), the 3D
                                                boxes and labels of the annotated furniture and the
                                                meshes of the furniture in view (sceneobjgt/, in
                                                the camera's frame)
    the scorer's weights: torchvision's AlexNet and the LPIPS v0.1 linear layers, CLIP ViT-B/32
    (openai/clip-vit-base-patch32), Qwen3-VL-8B-Instruct (Qwen/Qwen3-VL-8B-Instruct, Apache-2.0),
    and the Python packages they need that the task container lacks (scorer/pylib: transformers,
    tokenizers, huggingface_hub, safetensors, regex, from PyPI wheels)
    Blender 4.5.14 LTS (download.blender.org) and, because the task container lacks them, the X11
    and GL libraries it links against, from Ubuntu 22.04's archive; all checked against SHA-256
Then label_real.py (a GPU step, in the PyTorch container) annotates the furniture of the 200 real
photos: Mask2Former (Swin-L, ADE20K panoptic) instances of furniture classes.

Splits, each difficulty (easy: the 3D-FRONT renders; medium, complicated: real photos) the same
way, with a fixed seed (the renders stratified by the number of annotated objects):
    dev     10 + 10 + 10 photos, scored by evaluate_dev; labels visible
    val     10 + 10 + 10                                                        hidden
    test    20 + 20 + 20                                                        hidden
    reserve 59 + 60 + 60 photos with their labels (reserve/), outside every split a run sees or
            scores: kept for checks outside the runs (the scores of all 299 photos)
living_room_14 (a render) is left out: its HARMONY300 photo is another render than its FRONT3D
ground truth's (EXCLUDE). Each split has inputs/<id>/photo.png (the HARMONY300 photo as RGB, no metadata) and labels/<id>/:
    meta.json          difficulty, name (HARMONY300's), source, width, height; for the renders also
                       camera (the true pinhole: width, height, fx, fy, cx, cy in pixels)
    objects.json       the annotated furniture: label, boxy (box-shaped: cabinets, drawers,
                       wardrobes, shelves, ...); for the renders also its box (center, axes,
                       half_size) in the camera's frame
    the renders:  gt.npz (the furniture in view: vertices, float32, the camera's frame: x right, y
                  down, z forward, meters; faces, int32; face_object, int16: the annotated object a
                  face belongs to, -1 for furniture that is not annotated), layout_depth.npy (depth
                  of the room without its furniture, float32, half the photo's size) and depth.npy
                  (the photo's depth, float32)
    the real photos (label_real.py): instances.png (uint8, the photo's size: k on the pixels of the
                  k-th annotated object, 0 elsewhere)
<out>/manifest.json maps the ids to the HARMONY300 photos. dev/scorer/ holds the scorer's weights,
pylib/ and blender/ (blender/blender runs Blender with its libraries). Licenses: 3D-FRONT and
3D-FUTURE are for non-commercial research only; HARMONY300 is CC-BY-NC-4.0 (its real photos come
from Pexels and Unsplash under their licenses); Qwen3-VL is Apache-2.0.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HARMONY = "ShufanSun/harmony"
HARMONY_REV = "636ad7970291c23c1a08f16b45c71e76ef9d37a9"
FRONT3D = "andreead-a/FRONT3D"
FRONT3D_REV = "e31d96e6d63d53273c689ab9ac5bce28093dab84"
HF_MODELS = {  # scorer dir: (repo, revision, files)
    "clip-vit-base-patch32": ("openai/clip-vit-base-patch32", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
                              ("config.json", "merges.txt", "preprocessor_config.json",
                               "pytorch_model.bin", "special_tokens_map.json", "tokenizer.json",
                               "tokenizer_config.json", "vocab.json")),
    "qwen3-vl-8b-instruct": ("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
                             ("chat_template.json", "config.json", "generation_config.json",
                              "merges.txt", "model-00001-of-00004.safetensors",
                              "model-00002-of-00004.safetensors", "model-00003-of-00004.safetensors",
                              "model-00004-of-00004.safetensors", "model.safetensors.index.json",
                              "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
                              "video_preprocessor_config.json", "vocab.json")),
}
PYPI = "https://files.pythonhosted.org/packages/"
WHEELS = {  # scorer/pylib: the packages the scorer imports that the task container lacks
    "71/d3/c16c3b3cf7655a67db1144da94b021c200ac1303f82428f2beef6c2e72bb/transformers-4.57.1-py3-none-any.whl":
        "b10d05da8fa67dc41644dbbf9bc45a44cb86ae33da6f9295f5fbf5b7890bd267",
    "d0/c6/dc3a0db5a6766416c32c034286d7c2d406da1f498e4de04ab1b8959edd00/tokenizers-0.22.1-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl":
        "e2ef6063d7a84994129732b47e7915e8710f27f99f3a3260b8a38fc7ccd083f4",
    "31/a0/651f93d154cb72323358bf2bbae3e642bdb5d2f1bfc874d096f7cb159fa0/huggingface_hub-0.35.3-py3-none-any.whl":
        "0e3a01829c19d86d03793e4577816fe3bdfc1602ac62c7fb220d593d351224ba",
    "fe/5d/5a514d7b88e310c8b146e2404e0dc161282e78634d9358975fd56dfd14be/safetensors-0.6.2-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl":
        "8045db2c872db8f4cbe3faa0495932d89c38c899c603f21e9b6486951a5ecb8f",
    "2b/d5/1c712c7362f2563d389be66bae131c8bab121a3fabfa04b0b5bfc9e73c51/regex-2025.9.18-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl":
        "3acc471d1dd7e5ff82e6cacb3b286750decd949ecd4ae258696d04f019817ef8",
}
SCORER = {  # file: (url, sha256)
    "alexnet-owt-7be5be79.pth": (
        "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
        "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"),
    "lpips-v0.1-alex.pth": (
        "https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/"
        "082bb24f84c091ea94de2867d34c4544f68e0963/lpips/weights/v0.1/alex.pth",
        "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"),
}
BLENDER = ("blender-4.5.14-linux-x64",
           "https://download.blender.org/release/Blender4.5/blender-4.5.14-linux-x64.tar.xz",
           "9ba871ff2ecd36526b77432745980b7e6664ecd0c7ca11c48849073dcfe06da3")
UBUNTU = "http://archive.ubuntu.com/ubuntu/"
DEBS = {  # Ubuntu 22.04 packages with the libraries Blender links against: path: sha256
    "pool/main/libx/libx11/libx11-6_1.7.5-1ubuntu0.3_amd64.deb":
        "d382a3064ecd7576f24661a6d234adc737b854a9a0ec1d9f42cb4d007e77d547",
    "pool/main/libx/libxrender/libxrender1_0.9.10-1build4_amd64.deb":
        "a68f6db4320b5963dd499b83b6cc8c7b5cd39b05e84862d5945da9244e4c96d6",
    "pool/main/libx/libxfixes/libxfixes3_6.0.0-1_amd64.deb":
        "f627555440060edf19801d77d369c99547cf47123fe4e30c5bdb8b664f0b4467",
    "pool/main/libx/libxi/libxi6_1.8-1build1_amd64.deb":
        "47662736cca4be456fd42a5f0a9d3bed8a5cfad4ee08b809a95dde6004daf275",
    "pool/main/libx/libxkbcommon/libxkbcommon0_1.4.0-1_amd64.deb":
        "a0470edb237d52b71ad41f755c043164abed01423585ed5ea19525b3baaf84f7",
    "pool/main/libs/libsm/libsm6_1.2.3-1build2_amd64.deb":
        "e652e286a79d9e8e15189d79b290bdce20aca83651f3dfed1b9b7dc0bbf0702f",
    "pool/main/libi/libice/libice6_1.0.10-1build2_amd64.deb":
        "eea6d52d12ad610d70b0f70c825ed3c560dafb09f4c272e3c6b0b493d984d13f",
    "pool/main/libx/libxext/libxext6_1.3.4-1build1_amd64.deb":
        "875ec849749511dff61c8cc5a775f6f93427b9c3f392fdcec6925f14428e89cd",
    "pool/main/libg/libglvnd/libgl1_1.4.0-1_amd64.deb":
        "4fb2bbff94bb1a59bbdb40041e7a8ab58db2a003a151f4055357af03f0b78666",
    "pool/main/libg/libglvnd/libglx0_1.4.0-1_amd64.deb":
        "950068e8a3ad3905058d21a8c870c37989e48c0a537a7c57231c277af19f5395",
    "pool/main/libg/libglvnd/libglvnd0_1.4.0-1_amd64.deb":
        "6956d8be93b160b439a043141a061adfd54561d877423a956ace84d2fb3bfb4e",
    "pool/main/libx/libxcb/libxcb1_1.14-3ubuntu3_amd64.deb":
        "9258f04136434e7a0ccd76041505421934284209693373cfa9c24acb3a641ec5",
    "pool/main/libx/libxau/libxau6_1.0.9-1build5_amd64.deb":
        "551fa4101394af894f5f401285eeca9756da94bb31ecb08ed36a549ec580cd10",
    "pool/main/libx/libxdmcp/libxdmcp6_1.1.3-0ubuntu5_amd64.deb":
        "6f6c869dcfc072fa77d24b72ba127bdf15d606bdffbeba4314fb39f4f324363d",
}
WRAPPER = """#!/bin/sh
# Blender 4.5.14 LTS with the X11 and GL libraries it links against (the task container lacks
# them).
here="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="$here/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$here/blender-4.5.14-linux-x64/blender" "$@"
"""
SEED = 20261013
SIZES = {"dev": 10, "reserve": 50, "val": 10, "test": 30}     # per difficulty (reserve: what is left)
TEST_KEEP = 20        # then test keeps this many per difficulty (one evaluation job fits in 4 hours)
EXCLUDE = {  # name: why
    "living_room_14": "its HARMONY300 photo is another render than FRONT3D's rgb_003530 (shifted by "
                      "2 pixels, 11 % of its pixels differ by more than 20 levels after a blur), so "
                      "the true geometry does not fit it",
}
STRATA = ((1, 3), (4, 5), (6, 99))          # the renders: number of annotated objects
PREFIX = {"dev": "d", "reserve": "x", "val": "v", "test": "t"}
BOXY = ("cabinet", "drawer", "chest", "nightstand", "wardrobe", "armoire", "bookcase", "shelf",
        "sideboard", "tv stand", "refrigerator", "counter", "kitchen island", "buffet", "washer",
        "stove", "dishwasher", "case")
GL_TO_CV = np.array([1.0, -1.0, -1.0])
BOX_MARGIN = 0.02     # m: a part of the furniture mesh belongs to an annotated object when at
INSIDE = 0.95         # least INSIDE of its vertices lie in the object's box grown by BOX_MARGIN


def fetch(url: str, tries: int = 5) -> bytes:
    headers = {"User-Agent": "rsib-prepare"}
    if os.environ.get("HF_TOKEN") and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.read()
        except Exception:  # noqa: BLE001 - retried, then raised
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def front3d(path: str) -> bytes:
    return fetch(f"https://huggingface.co/datasets/{FRONT3D}/resolve/{FRONT3D_REV}/{path}")


def harmony(path: str) -> bytes:
    return fetch(f"https://huggingface.co/datasets/{HARMONY}/resolve/{HARMONY_REV}/{path}")


def harmony_input(difficulty: str, name: str) -> bytes:
    """A HARMONY300 photo, whatever its extension (input.jpg, input.jpeg, input.png, ...)."""
    url = f"https://huggingface.co/api/datasets/{HARMONY}/tree/{HARMONY_REV}/{difficulty}/{name}"
    files = [x["path"] for x in json.loads(fetch(url)) if x["path"].split("/")[-1].startswith("input.")]
    if len(files) != 1:
        raise SystemExit(f"{difficulty}/{name}: expected one input photo, found {files}")
    return harmony(files[0])


def is_boxy(label: str) -> bool:
    return any(b in label.lower() for b in BOXY)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, path: Path, digest: str) -> None:
    if path.exists() and sha256(path) == digest:
        return
    print(f"downloading {url}", flush=True)
    tmp = path.with_suffix(path.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "rsib-prepare"})
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 22)
    if sha256(tmp) != digest:
        tmp.unlink()
        raise SystemExit(f"{path.name}: SHA-256 mismatch for {url}")
    tmp.replace(path)


def get_scorer(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in SCORER.items():
        download(url, d / name, digest)
    for sub, (repo, rev, files) in HF_MODELS.items():
        (d / sub).mkdir(exist_ok=True)
        for f in files:
            if not (d / sub / f).exists():
                print(f"downloading {repo}/{f}", flush=True)
                url = f"https://huggingface.co/{repo}/resolve/{rev}/{f}"
                tmp = d / sub / (f + ".part")
                req = urllib.request.Request(url, headers={"User-Agent": "rsib-prepare"})
                with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as out:
                    shutil.copyfileobj(r, out, 1 << 22)
                tmp.replace(d / sub / f)
        (d / sub / "REVISION").write_text(f"{repo}@{rev}\n")
    lib = d / "pylib"
    if not (lib / "transformers").exists():
        lib.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            for path, digest in WHEELS.items():
                whl = Path(tmp) / Path(path).name
                download(PYPI + path, whl, digest)
                with zipfile.ZipFile(whl) as z:
                    z.extractall(lib)


def deb_files(deb: Path, dest: Path) -> None:
    """Extract a .deb's data into dest (dpkg-deb, or ar + zstd/xz)."""
    if shutil.which("dpkg-deb"):
        subprocess.run(["dpkg-deb", "-x", str(deb), str(dest)], check=True)
        return
    data = deb.read_bytes()
    assert data[:8] == b"!<arch>\n", f"{deb} is not an ar archive"
    pos = 8
    while pos < len(data):
        name = data[pos:pos + 16].decode().strip().rstrip("/")
        size = int(data[pos + 48:pos + 58].decode().strip())
        body = data[pos + 60:pos + 60 + size]
        pos += 60 + size + (size & 1)
        if not name.startswith("data.tar"):
            continue
        if name.endswith(".zst"):
            body = subprocess.run(["zstd", "-dc"], input=body, capture_output=True,
                                  check=True).stdout
            mode = "r:"
        else:
            mode = "r:*"
        with tarfile.open(fileobj=io.BytesIO(body), mode=mode) as tf:
            tf.extractall(dest, filter="tar")
        return
    raise SystemExit(f"{deb}: no data member")


def get_blender(d: Path) -> None:
    """dev/scorer/blender: Blender, its libraries in lib/ and the wrapper `blender`."""
    d.mkdir(parents=True, exist_ok=True)
    name, url, digest = BLENDER
    if not (d / name / "blender").exists():
        tarball = d / f"{name}.tar.xz"
        download(url, tarball, digest)
        print("unpacking Blender", flush=True)
        with tarfile.open(tarball) as tf:
            tf.extractall(d, filter="tar")
        tarball.unlink()
    lib = d / "lib"
    if not (lib / "libX11.so.6").exists():
        lib.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            for path, digest in DEBS.items():
                deb = Path(tmp) / Path(path).name
                download(UBUNTU + path, deb, digest)
                deb_files(deb, Path(tmp) / "x")
            for so in (Path(tmp) / "x" / "usr" / "lib" / "x86_64-linux-gnu").glob("*.so.*"):
                # libX.so.N, the name the loader asks for
                if so.name.count(".") <= 2 or so.is_symlink():
                    shutil.copyfile(so.resolve(), lib / so.name)
    (d / "blender").write_text(WRAPPER)
    (d / "blender").chmod(0o755)


# --------------------------------------------------------------------------- ground truth
def read_ply(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode()
    if "binary_little_endian" not in header:
        raise SystemExit("unexpected PLY format")
    nv = int(header.split("element vertex ")[1].split()[0])
    nf = int(header.split("element face ")[1].split()[0])
    props = [ln.split()[-1] for ln in header.splitlines()
             if ln.startswith("property") and "list" not in ln]
    if props[:3] != ["x", "y", "z"] or any(ln.split()[1] != "float" for ln in header.splitlines()
                                           if ln.startswith("property") and "list" not in ln):
        raise SystemExit(f"unexpected PLY vertex properties {props}")
    v = np.frombuffer(data, np.float32, count=nv * len(props), offset=end).reshape(nv, len(props))
    rec = np.dtype([("n", "u1"), ("i", "<i4", 3)])
    f = np.frombuffer(data, rec, count=nf, offset=end + nv * 4 * len(props))
    if not (f["n"] == 3).all():
        raise SystemExit("PLY with non-triangular faces")
    return v[:, :3].astype(np.float64), f["i"].astype(np.int64)


def box_frame(obj: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center, axes (rows) and half sizes of an annotated object's box in the OpenCV camera
    frame."""
    corners = np.array(obj["bbox3d_camera"], np.float64).T * GL_TO_CV
    center = corners.mean(0)
    d = corners - corners[0]
    n = np.linalg.norm(d, axis=1)
    axes = []
    for k in np.argsort(n)[1:]:
        v = d[k] / n[k]
        if all(abs(v @ a) < 0.1 for a in axes):
            axes.append(v)
        if len(axes) == 3:
            break
    axes = np.array(axes)
    return center, axes, np.abs((corners - center) @ axes.T).max(0)


def split_objects(vertices: np.ndarray, faces: np.ndarray, boxes: list) -> np.ndarray:
    """The annotated object each face belongs to (-1 for the others): every connected part of the
    mesh goes to the box that holds at least INSIDE of its vertices (the smaller box on a tie)."""
    _, weld = np.unique(np.round(vertices, 5), axis=0, return_inverse=True)
    fw = weld.reshape(-1)[faces]
    rows = np.repeat(np.arange(len(fw)), 3)
    A = coo_matrix((np.ones(len(rows)), (rows, fw.ravel())), shape=(len(fw), fw.max() + 1)).tocsr()
    ncomp, comp = connected_components(A @ A.T, directed=False)
    order = np.argsort(comp, kind="stable")
    bounds = np.searchsorted(comp[order], np.arange(ncomp + 1))
    face_object = np.full(len(faces), -1, np.int16)
    for c in range(ncomp):
        fs = order[bounds[c]:bounds[c + 1]]
        vs = vertices[np.unique(faces[fs])]
        best, best_share = -1, 0.0
        for k, (center, axes, half) in enumerate(boxes):
            share = float((np.abs((vs - center) @ axes.T) <= half + BOX_MARGIN).all(1).mean())
            if share > best_share + 1e-9 or (abs(share - best_share) <= 1e-9 and best >= 0
                                             and np.prod(half) < np.prod(boxes[best][2])):
                best, best_share = k, share
        if best >= 0 and best_share >= INSIDE:
            face_object[fs] = best
    return face_object


def build_easy(rgb: int, name: str, inp: Path, lab: Path) -> dict:
    """A 3D-FRONT render: its HARMONY300 photo and its FRONT3D ground truth."""
    num = f"{rgb:06d}"
    ann = json.loads(front3d(f"annotation/annotation_{num}.json"))
    photo = Image.open(io.BytesIO(harmony_input("easy", name))).convert("RGB")
    ref = Image.open(io.BytesIO(front3d(f"rgb/rgb_{num}.jpeg"))).convert("RGB")
    if photo.size != ref.size or (np.asarray(photo) != np.asarray(ref)).any():
        raise SystemExit(f"easy/{name}: the HARMONY300 photo is not FRONT3D's rgb_{num}")
    depth = np.load(io.BytesIO(front3d(f"depth/depth_{num}.npy"))).astype(np.float32)
    layout = np.load(io.BytesIO(front3d(f"bgdepth/bgdepth_{num}.npy"))).astype(np.float32)
    vertices, faces = read_ply(front3d(f"sceneobjgt/sceneobjgt_{num}.ply"))
    W, H = photo.size
    K = np.array(ann["camera_intrinsics"], np.float64)
    if (W, H) != (1296, 968) or depth.shape != (H, W) or layout.shape != (H // 2, W // 2):
        raise SystemExit(f"easy/{name}: unexpected sizes {photo.size} {depth.shape} {layout.shape}")
    objs = [ann["obj_dict"][k] for k in sorted(ann["obj_dict"], key=int)]
    boxes = [box_frame(o) for o in objs]
    face_object = split_objects(vertices, faces, boxes)
    inp.mkdir(parents=True, exist_ok=True)
    lab.mkdir(parents=True, exist_ok=True)
    photo.save(inp / "photo.png", format="PNG")
    (lab / "meta.json").write_text(json.dumps(
        {"difficulty": "easy", "name": name, "source": "3D-FRONT", "width": W, "height": H,
         "camera": {"width": W, "height": H, "fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2],
                    "cy": K[1, 2]}}, indent=1))
    np.savez_compressed(lab / "gt.npz", vertices=vertices.astype(np.float32),
                        faces=faces.astype(np.int32), face_object=face_object)
    (lab / "objects.json").write_text(json.dumps([
        {"label": o["label"][0], "boxy": is_boxy(o["label"][0]), "center": c.round(5).tolist(),
         "axes": a.round(6).tolist(), "half_size": h.round(5).tolist()}
        for o, (c, a, h) in zip(objs, boxes)], indent=1))
    np.save(lab / "layout_depth.npy", layout)
    np.save(lab / "depth.npy", depth)
    return {"objects": len(objs)}


def build_real(difficulty: str, name: str, source: str, inp: Path, lab: Path) -> dict:
    """A real photo: its HARMONY300 photo (label_real.py adds its annotations)."""
    photo = Image.open(io.BytesIO(harmony_input(difficulty, name))).convert("RGB")
    inp.mkdir(parents=True, exist_ok=True)
    lab.mkdir(parents=True, exist_ok=True)
    photo.save(inp / "photo.png", format="PNG")
    (lab / "meta.json").write_text(json.dumps(
        {"difficulty": difficulty, "name": name, "source": source, "width": photo.width,
         "height": photo.height}, indent=1))
    return {}


def assign(rows: list[dict], counts: dict[str, int]) -> dict[str, list[dict]]:
    """Split each difficulty into SIZES, the renders stratified by their number of objects."""
    rng = random.Random(SEED)
    order = ("val", "dev", "reserve")
    out = {split: [] for split in SIZES}
    for difficulty in ("easy", "medium", "complicated"):
        pool_all = sorted((r for r in rows if r["difficulty"] == difficulty and r["name"] not in EXCLUDE),
                          key=lambda r: r["name"])
        strata = ([[r for r in pool_all if lo <= counts[r["name"]] <= hi] for lo, hi in STRATA]
                  if difficulty == "easy" else [pool_all])
        part = {split: [] for split in SIZES}
        for pool in strata:
            rng.shuffle(pool)
            k = 0
            for split in order:
                n = round(SIZES[split] * len(pool) / len(pool_all))
                part[split] += pool[k:k + n]
                k += n
            part["test"] += pool[k:]
        for split in ("val", "dev"):  # rounding: move photos from test until the sizes match
            while len(part[split]) < SIZES[split]:
                part[split].append(part["test"].pop(rng.randrange(len(part["test"]))))
            while len(part[split]) > SIZES[split]:
                part["test"].append(part[split].pop())
        while len(part["test"]) < SIZES["test"]:
            part["test"].append(part["reserve"].pop(rng.randrange(len(part["reserve"]))))
        while len(part["test"]) > SIZES["test"]:      # what is left goes to the reserve
            part["reserve"].append(part["test"].pop(rng.randrange(len(part["test"]))))
        if len(part["test"]) != SIZES["test"]:
            raise SystemExit(f"{difficulty}: {len(pool_all)} photos are too few")
        for split in SIZES:
            out[split] += part[split]
    for split in out:
        rng.shuffle(out[split])
    return trim_test(out, counts)


def trim_test(splits: dict[str, list[dict]], counts: dict[str, int]) -> dict[str, list[dict]]:
    """Keep TEST_KEEP test photos of each difficulty (the renders in proportion to their strata),
    in their order; the others go to the end of the reserve. A generator of its own, so dev and val are
    those of assign() before the trim."""
    rng = random.Random(SEED + 1)
    drop = []
    for difficulty in ("easy", "medium", "complicated"):
        members = [r for r in splits["test"] if r["difficulty"] == difficulty]
        strata = ([[r for r in members if lo <= counts[r["name"]] <= hi] for lo, hi in STRATA]
                  if difficulty == "easy" else [members])
        keep = []
        for pool in strata:
            pool = sorted(pool, key=lambda r: r["name"])
            rng.shuffle(pool)
            keep += pool[:round(TEST_KEEP * len(pool) / len(members))]
        rest = sorted((r for r in members if r not in keep), key=lambda r: r["name"])
        rng.shuffle(rest)
        while len(keep) < TEST_KEEP:
            keep.append(rest.pop())
        while len(keep) > TEST_KEEP:
            rest.append(keep.pop())
        drop += rest
    splits["test"] = [r for r in splits["test"] if r not in drop]
    splits["reserve"] = splits["reserve"] + drop
    return splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    get_scorer(out / "dev" / "scorer")
    get_blender(out / "dev" / "scorer" / "blender")
    rows = list(csv.DictReader(io.StringIO(harmony("attribution.csv").decode())))
    by = {d: [r for r in rows if r["difficulty"] == d] for d in ("easy", "medium", "complicated")}
    if [len(v) for v in by.values()] != [100, 100, 100]:
        raise SystemExit(f"expected 100 photos of each difficulty, found {[len(v) for v in by.values()]}")
    rgb = {r["name"]: int(r["pexels_id"].split("_")[1]) for r in by["easy"]}  # easy: rgb_<FRONT3D id>
    counts = {name: len(json.loads(front3d(f"annotation/annotation_{n:06d}.json"))["obj_dict"])
              for name, n in rgb.items()}
    splits = assign(rows, counts)
    roots = {"dev": out / "dev", "reserve": out / "reserve", "val": out / "val", "test": out / "test"}
    for root in roots.values():
        for part in ("inputs", "labels"):
            if (root / part).exists():
                shutil.rmtree(root / part)
    if (out / "dev" / "extra").exists():  # where an earlier build put these photos, in the solver's view
        shutil.rmtree(out / "dev" / "extra")
    manifest = {"harmony": f"{HARMONY}@{HARMONY_REV}", "front3d": f"{FRONT3D}@{FRONT3D_REV}",
                "seed": SEED, "excluded": EXCLUDE, "photos": {}}
    jobs = []
    for split, members in splits.items():
        for i, r in enumerate(members):
            sid = f"{PREFIX[split]}{i:03d}"
            inp, lab = roots[split] / "inputs" / sid, roots[split] / "labels" / sid
            if r["difficulty"] == "easy":
                jobs.append(lambda r=r, inp=inp, lab=lab: build_easy(rgb[r["name"]], r["name"], inp, lab))
            else:
                jobs.append(lambda r=r, inp=inp, lab=lab: build_real(r["difficulty"], r["name"], r["source"],
                                                                      inp, lab))
            manifest["photos"][f"{split}/{sid}"] = {"difficulty": r["difficulty"], "name": r["name"]}
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda job: job(), jobs))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    for split, members in splits.items():
        print(f"{split}: {len(members)} photos ("
              + ", ".join(f"{d} {sum(r['difficulty'] == d for r in members)}"
                          for d in ("easy", "medium", "complicated")) + ")")
    print("next: label_real.py --out", out, "(GPU, the PyTorch container)")


if __name__ == "__main__":
    main()
