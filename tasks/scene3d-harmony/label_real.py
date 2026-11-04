"""Annotate the furniture of the real photos after prepare.py (a GPU step, in the PyTorch
container): Mask2Former (Swin-L, ADE20K panoptic, facebook/mask2former-swin-large-ade-panoptic at a
pinned revision, downloaded into --cache) on each photo at 640 pixels on its short side (at most
1280 on its long side), its panoptic segments of furniture classes (FURNITURE) with a score of at
least 0.5 and at least MIN_SHARE of the photo, the MAX_OBJECTS largest. Writes labels/<id>/
instances.png (uint8, the photo's size: k on the k-th object's pixels, 0 elsewhere) and
objects.json ([{label, boxy, score, pixels}], largest first). On the 3D-FRONT renders the same
annotation finds 85 % of the true annotated objects at an IoU of at least 0.5 (median IoU 0.80).

Usage: python label_real.py --out <data root> [--cache <dir>]   (PYTHONPATH: dev/scorer/pylib)
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from prepare import is_boxy  # noqa: E402

M2F = ("facebook/mask2former-swin-large-ade-panoptic", "ae53ed7ab37f8ea74f23d0e4f8c3d83c184a0caa",
       ("config.json", "model.safetensors", "preprocessor_config.json"))
FURNITURE = {"bed ", "cabinet", "table", "chair", "sofa", "shelf", "armchair", "seat", "desk",
             "wardrobe", "bathtub", "chest of drawers", "counter", "refrigerator", "bookcase",
             "coffee table", "toilet", "bench", "kitchen island", "swivel chair", "ottoman",
             "buffet", "stool", "stove", "washer", "pool table", "cradle", "case", "sink"}
MIN_SHARE = 0.005
MAX_OBJECTS = 8


def model_dir(cache: Path) -> Path:
    repo, rev, files = M2F
    d = cache / repo.replace("/", "--") / rev
    d.mkdir(parents=True, exist_ok=True)
    for f in files:
        if not (d / f).exists():
            print(f"downloading {repo}/{f}", flush=True)
            req = urllib.request.Request(f"https://huggingface.co/{repo}/resolve/{rev}/{f}",
                                         headers={"User-Agent": "rsib-prepare"})
            with urllib.request.urlopen(req, timeout=600) as r:
                (d / (f + ".part")).write_bytes(r.read())
            (d / (f + ".part")).replace(d / f)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    sys.path.insert(0, str(out / "dev" / "scorer" / "pylib"))
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    path = model_dir(Path(args.cache) if args.cache else out / ".label-cache")
    device = torch.device("cuda")
    proc = AutoImageProcessor.from_pretrained(path, local_files_only=True)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(path, local_files_only=True).to(device).eval()
    id2label = model.config.id2label
    roots = [out / "dev", out / "reserve", out / "val", out / "test"]
    counts = []
    for root in roots:
        for lab in sorted((root / "labels").iterdir()):
            meta = json.loads((lab / "meta.json").read_text())
            if meta["difficulty"] == "easy":
                continue
            img = Image.open(root / "inputs" / lab.name / "photo.png").convert("RGB")
            inputs = proc(images=img, size={"shortest_edge": 640, "longest_edge": 1280},
                          return_tensors="pt").to(device)
            with torch.no_grad():
                pred = model(**inputs)
            pan = proc.post_process_panoptic_segmentation(
                pred, target_sizes=[(img.height, img.width)], threshold=0.5,
                overlap_mask_area_threshold=0.8, label_ids_to_fuse=set())[0]
            seg = pan["segmentation"].cpu().numpy()
            cands = []
            for s in pan["segments_info"]:
                label = id2label[s["label_id"]]
                pix = int((seg == s["id"]).sum())
                if label in FURNITURE and pix >= MIN_SHARE * img.width * img.height:
                    cands.append((pix, s["id"], label.strip(), float(s["score"])))
            cands.sort(reverse=True)
            inst = np.zeros(seg.shape, np.uint8)
            objs = []
            for k, (pix, sid, label, score) in enumerate(cands[:MAX_OBJECTS], 1):
                inst[seg == sid] = k
                objs.append({"label": label, "boxy": is_boxy(label), "score": round(score, 4),
                             "pixels": pix})
            Image.fromarray(inst).save(lab / "instances.png")
            (lab / "objects.json").write_text(json.dumps(objs, indent=1))
            counts.append(len(objs))
            print(f"{root.name}/{lab.name} ({meta['difficulty']}): {[o['label'] for o in objs]}", flush=True)
    print(f"{len(counts)} real photos, {np.mean(counts):.1f} objects each, "
          f"{sum(c == 0 for c in counts)} with none")


if __name__ == "__main__":
    main()
