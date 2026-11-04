"""Build the scene3d-holoscene splits under --out from the HoloScene release.

Two steps: this script (no GPU), then label_gt.py on a GPU in the task's container (see there).

Sources, downloaded at pinned revisions and checked against their SHA-256 (not part of the task
package):
    huggingface.co/datasets/hongchi/HoloScene   the nine scenes HoloScene (Xia et al., NeurIPS 2025)
                                                released, one zip each: RGB frames, instance masks,
                                                transforms.json (cameras), graph.json (support graph)
                                                and mesh.ply, the ground-truth surface. MIT. Only
                                                the zips are downloaded: the dataset's folders of
                                                textured surface_*.obj meshes are HoloScene's own
                                                reconstructions and stay out of the task.
    the scorer (tests/metrics.py, tests/physics.py): torchvision's AlexNet and the LPIPS v0.1 linear
    layers; the five source files of the V-JEPA 2.1 encoder (facebookresearch/vjepa2, MIT; its
    weights are fetched by label_gt.py); wheels of MuJoCo, CoACD and fast-simplification for
    Python 3.10 on x86-64 Linux, unpacked into scorer/pylib.

Splits (fixed):
    test   the six scenes CoDimRecon (arXiv 2609.36024) evaluates: Replica room_0, room_1, room_2
           and ScanNet++ 67d702f2e8, 7831862f02, acd69a1746
    val    Gibson Beechwood_1_int (an iGibson house)                                    hidden
    dev    Gibson Beechwood_0_int and siebelgame (a game room captured with a phone), labels visible
Ids are opaque (d1, d2, v1, t1..t6) in an order shuffled with SEED; <out>/manifest.json maps them
to the scenes.

Per scene, inputs/<id>/ (v3: CoDimRecon's setting, the frames alone):
    images/frame_NNNNN.{jpg,png}   the RGB frames in capture order (sorted file names), every
                                   FRAME_STEP-th for the long Gibson and phone sequences
labels/<id>/:
    masks/frame_NNNNN.png          instance masks: 0 the background, k the annotated object k
                                   (the release's value + 1; its 255 becomes 0)
    transforms.json                fl_x, fl_y, cx, cy, w, h and per frame file_path, mask_path and
                                   transform_matrix (camera to world; the camera's x right, y up,
                                   z backwards; world z up; meters)
    graph.json                     the release's support graph: node k is object k, node 0 the
                                   background; the tree rooted at node 0 says what rests on what
    mesh.ply (the released ground truth), meta.json ({"source", "scene"}); label_gt.py adds gt.npz
    and objects.json.
The phone sequence has lens distortion (OpenCV k1, k2, p1, p2): its frames and masks are resampled
to a distortion-free pinhole with the same fx, fy, cx, cy. dev/scorer/ holds the scorer: evaluations of every split can read the dev data (in a sandboxed container job,
the evaluator sees only its task copy, the split it scores, the dev data, the workspace and its
output directory).
"""

import argparse
import hashlib
import io
import json
import os
import random
import re
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

HOLOSCENE = "hongchi/HoloScene"
HOLOSCENE_REV = "26e56c1ab5cd3ef03ee485c50ecb411e5095ba11"
SCENES = {  # zip in the release: sha256
    "replica/room_0": "a1154e2a1a9f2344b2bcfc9c75f6d93904d9b95b2a6c22b97589fb140b2f1262",
    "replica/room_1": "e32fe12ef850fee03c579d999d569507ac0526416e607fddea250fef0b272910",
    "replica/room_2": "d5bb1ee5c2a62c759fb2001598fb2f51a71a4a71b505173f6660f4c1d8058820",
    "scannetpp/67d702f2e8": "7b057c264f436467cb3e0a34a157c6d98b9c797dd00e20dd23753512c4199cf4",
    "scannetpp/7831862f02": "0d33314c96d03549f572ad25ddc8dd2be39e3e14b8103a21213b4bfdef510883",
    "scannetpp/acd69a1746": "8f413c0d0e22d21ad843e2485f9a2e4a60e95f6361c8d0c7a4a26686413b2449",
    "gibson/Beechwood_0_int": "fa06c2b754ef58803dabd81a5031b1bbee23dd73afe3b8151b4879a579c14674",
    "gibson/Beechwood_1_int": "2b72d96f9a97e2b51c36db583f43c5637ba275c19df95c93ba3b663956e70a58",
    "custom/siebelgame": "95fe7e2c9b3125be91b0a12f92667280cb1050bc162bb878bd2d2779a373801e",
}
SPLITS = {
    "test": ["replica/room_0", "replica/room_1", "replica/room_2", "scannetpp/67d702f2e8",
             "scannetpp/7831862f02", "scannetpp/acd69a1746"],
    "val": ["gibson/Beechwood_1_int"],
    "dev": ["gibson/Beechwood_0_int", "custom/siebelgame"],
}
FRAME_STEP = {"gibson": 3, "custom": 3}
SEED = 20261009

VJEPA_REV = "204698b45b3712590f06245fbfba32d3be539812"
VJEPA_FILES = {
    "app/vjepa_2_1/models/vision_transformer.py": "d2932eabeba684d8f558302a13cfd4be70a0170ee5112f5a794652d0a29089b9",
    "app/vjepa_2_1/models/utils/modules.py": "64be6a87bd9f18d385f4e44186db3347d1665e18a1f0511d51d3b305531562e2",
    "app/vjepa_2_1/models/utils/patch_embed.py": "29e11ab97ab3ccdef107d6a7d0d7b374b58e712076cc3561f07b7e603c9b5165",
    "src/masks/utils.py": "833f111a0fa5ffdbd3a6412e2dace2517c3c178f49c14f8bb631d9f6a070dfd0",
    "src/utils/tensors.py": "782b58bd2af456e184750e5318ab773105108383f61b280fe4c7a90f46add2c8",
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
_PYPI = "https://files.pythonhosted.org/packages/"
WHEELS = {  # unpacked into scorer/pylib for the evaluator (Python 3.10, x86-64 Linux)
    "mujoco-3.15.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl": (
        _PYPI + "38/49/71acbef3ead0b9bb5ec6802600c8a298bd28ba91d72b285e996c3f055cc6/",
        "277774ddfabaf2b08df6f7c6946f22bddd0cbe7e92b1cf968864335f40673081"),
    "coacd-1.0.14-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl": (
        _PYPI + "88/7c/ec7edfc77c48fe97434ae813eca7614103289105bb546952ed9a0a266ce8/",
        "6c135b717af096523bbebf1fd8d326e8950c024237fe9806ad46dedd8b4c31f0"),
    "fast_simplification-0.2.0-cp310-cp310-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl": (
        _PYPI + "ae/cd/997dec130d62022c38005dc3908dc06d154661e9416ab16067769392c2d4/",
        "327421879656f3a4cd34b8869a3b2c86cf4dcc6fd100b9d5dee8f57aaeaf80d7"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, path: Path, digest: str, tries: int = 5) -> Path:
    """Download url to path unless a file with that SHA-256 is already there."""
    if path.exists() and sha256(path) == digest:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "rsib-prepare"}
    if os.environ.get("HF_TOKEN") and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    tmp = path.with_name(path.name + ".part")
    for attempt in range(tries):
        try:
            print(f"downloading {url}", flush=True)
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=600) as r, \
                    open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 22)
            break
        except Exception:  # noqa: BLE001 - retried, then raised
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)
    if sha256(tmp) != digest:
        tmp.unlink()
        raise SystemExit(f"{path.name}: SHA-256 mismatch for {url}")
    tmp.replace(path)
    return path


def get_scorer(d: Path, cache: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in SCORER.items():
        download(url, d / name, digest)
    for rel, digest in VJEPA_FILES.items():
        download(f"https://raw.githubusercontent.com/facebookresearch/vjepa2/{VJEPA_REV}/{rel}",
                 d / "vjepa2_1" / rel, digest)
    pylib = d / "pylib"
    stamp = pylib / ".wheels"
    if not (stamp.exists() and stamp.read_text().split() == sorted(WHEELS)):
        if pylib.exists():
            shutil.rmtree(pylib)
        pylib.mkdir(parents=True)
        for name, (url, digest) in WHEELS.items():
            zipfile.ZipFile(download(url + name, cache / name, digest)).extractall(pylib)
        stamp.write_text("\n".join(sorted(WHEELS)))


def _natural(name: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def _mask_name(stem: str, masks: set[str]) -> str:
    for cand in (stem, stem.replace("frame", "")):
        if cand + ".png" in masks:
            return cand + ".png"
    raise SystemExit(f"no instance mask for frame {stem}")


def _undistort_maps(t: dict) -> tuple[np.ndarray, np.ndarray]:
    """For every pixel of the pinhole image, where it lies in the distorted frame (x, y in pixels,
    0 at the top-left pixel's center)."""
    w, h = int(t["w"]), int(t["h"])
    fx, fy, cx, cy = t["fl_x"], t["fl_y"], t["cx"], t["cy"]
    k1, k2, p1, p2 = (float(t.get(k, 0.0)) for k in ("k1", "k2", "p1", "p2"))
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    x, y = (u - cx) / fx, (v - cy) / fy
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return fx * xd + cx - 0.5, fy * yd + cy - 0.5


def _bilinear(img: np.ndarray, mx: np.ndarray, my: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x = np.clip(mx, 0, w - 1)
    y = np.clip(my, 0, h - 1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    a = img.astype(np.float64)
    out = ((a[y0, x0] * (1 - fx) + a[y0, x1] * fx) * (1 - fy)
           + (a[y1, x0] * (1 - fx) + a[y1, x1] * fx) * fy)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def build(name: str, archive: Path, inp: Path, lab: Path) -> dict:
    source, scene = name.split("/")
    z = zipfile.ZipFile(archive)
    files = {i.filename.split("/", 1)[1]: i for i in z.infolist()
             if not i.is_dir() and "/" in i.filename}
    t = json.loads(z.read(files["transforms.json"]))
    masks = {p.split("/", 1)[1] for p in files if p.startswith("instance_mask/")}
    frames = sorted(t["frames"], key=lambda f: _natural(Path(f["file_path"]).name))
    frames = frames[::FRAME_STEP.get(source, 1)]
    distorted = any(abs(float(t.get(k, 0.0))) > 0 for k in ("k1", "k2", "p1", "p2"))
    maps = _undistort_maps(t) if distorted else None
    (inp / "images").mkdir(parents=True, exist_ok=True)
    (lab / "masks").mkdir(parents=True, exist_ok=True)
    out_frames = []
    for i, fr in enumerate(frames):
        src = Path(fr["file_path"])
        ext = ".jpg" if src.suffix.lower() in (".jpg", ".jpeg") else ".png"
        img_name, mask_name = f"images/frame_{i:05d}{ext}", f"masks/frame_{i:05d}.png"
        raw = z.read(files[str(src)])
        m = np.asarray(Image.open(io.BytesIO(z.read(files["instance_mask/" + _mask_name(src.stem, masks)]))))
        if m.ndim != 2:
            raise SystemExit(f"{name}: mask of {src} is not single-channel")
        m = np.where(m == 255, 0, m.astype(np.int64) + 1).astype(np.uint8)
        if maps is None:
            (inp / img_name).write_bytes(raw)
        else:
            img = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
            Image.fromarray(_bilinear(img, *maps)).save(inp / img_name)
            mx, my = maps
            m = m[np.clip(np.round(my), 0, m.shape[0] - 1).astype(np.int64),
                  np.clip(np.round(mx), 0, m.shape[1] - 1).astype(np.int64)]
        w, h = Image.open(inp / img_name).size
        if (w, h) != (int(t["w"]), int(t["h"])) or m.shape != (h, w):
            raise SystemExit(f"{name}: frame {src} is {w}x{h}, mask {m.shape}, transforms say "
                             f"{t['w']}x{t['h']}")
        Image.fromarray(m).save(lab / mask_name)
        out_frames.append({"file_path": img_name, "mask_path": mask_name,
                           "transform_matrix": fr["transform_matrix"]})
    cams = {k: t[k] for k in ("fl_x", "fl_y", "cx", "cy", "w", "h")}
    (lab / "transforms.json").write_text(json.dumps({**cams, "frames": out_frames}, indent=1))
    (lab / "graph.json").write_bytes(z.read(files["graph.json"]))
    with z.open(files["mesh.ply"]) as src, open(lab / "mesh.ply", "wb") as dst:
        shutil.copyfileobj(src, dst, 1 << 22)
    (lab / "meta.json").write_text(json.dumps({"source": source, "scene": scene}))
    return {"frames": len(out_frames), "undistorted": distorted}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "downloads"
    get_scorer(out / "dev" / "scorer", cache)
    archives = {name: download(f"https://huggingface.co/datasets/{HOLOSCENE}/resolve/{HOLOSCENE_REV}/{name}.zip",
                               cache / f"{name}.zip", digest) for name, digest in SCENES.items()}
    rng = random.Random(SEED)
    manifest = {"holoscene": f"{HOLOSCENE}@{HOLOSCENE_REV}", "vjepa2": VJEPA_REV, "seed": SEED,
                "scenes": {}}
    for split, names in SPLITS.items():
        for part in ("inputs", "labels"):
            if (out / split / part).exists():
                shutil.rmtree(out / split / part)
        names = list(names)
        rng.shuffle(names)
        for i, name in enumerate(names, 1):
            sid = f"{split[0]}{i}"
            info = build(name, archives[name], out / split / "inputs" / sid, out / split / "labels" / sid)
            manifest["scenes"][f"{split}/{sid}"] = {"scene": name, **info}
            print(f"{split}/{sid}: {name} {info}", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("next: label_gt.py on a GPU in the task's container (see its docstring)")


if __name__ == "__main__":
    main()
