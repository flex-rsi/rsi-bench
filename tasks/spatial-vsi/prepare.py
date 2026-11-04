"""Build the splits of spatial-vsi from VSI-Bench (nyu-visionx/VSI-Bench, pinned revision).

    VSI_RAW=/path/to/raw rsib task prepare spatial-vsi

$VSI_RAW holds test.jsonl, pruned_ids.txt and the three video folders of the dataset
(<source>/<source>/<scene>.mp4), as the dataset's zips unpack. Without it, prepare downloads
them (5.4 GB) into <out>/_raw.

Scenes never cross splits. Scenes are shuffled with a fixed seed and dealt to test, val and dev;
then each split keeps a subset of its questions, stratified by question type (the 8 groups of the
official score), so every split measures every group:

    dev   QUESTIONS["dev"]   questions.json WITH answers, and the videos of its scenes
    val   VAL_TOTAL          questions.json (no answers) + labels.json, videos
    test  QUESTIONS["test"]  same
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path

REV = "3a6a1cf9ac54c403509b15260897cc6ddc41b408"
URL = f"https://huggingface.co/datasets/nyu-visionx/VSI-Bench/resolve/{REV}"
SOURCES = ("arkitscenes", "scannet", "scannetpp")
QUESTIONS = {"dev": 80, "val": 160, "test": 320}
# val grows to VAL_TOTAL by adding questions from the same val scenes with a separate rng,
# after the three splits above are drawn exactly as before: dev and test stay byte for byte the same,
# and the first 160 val questions stay in val. Three times the questions cut the val noise sigma
# (0.033 at 160, measured) by about sqrt(3).
VAL_TOTAL = 480
SCENE_SHARE = {"test": 0.4, "val": 0.3, "dev": 0.3}
SEED = 20261020


def group(qtype: str) -> str:
    """The official score's 8 groups: the three relative-direction levels are one."""
    return "object_rel_direction" if qtype.startswith("object_rel_direction") else qtype


def fetch(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for name in ("test.jsonl", "pruned_ids.txt") + tuple(f"{s}.zip" for s in SOURCES):
        stem = name.removesuffix(".zip")
        if (raw / name).exists() or (name.endswith(".zip") and (raw / stem).is_dir()):
            continue
        subprocess.run(["curl", "-sfL", "--retry", "5", "-o", str(raw / name), f"{URL}/{name}"],
                       check=True)
        if name.endswith(".zip"):
            with zipfile.ZipFile(raw / name) as z:
                z.extractall(raw / stem)
            (raw / name).unlink()


def video(raw: Path, q: dict) -> Path:
    return raw / q["dataset"] / q["dataset"] / f"{q['scene_name']}.mp4"


def stratified(qs: list[dict], n: int, rng: random.Random) -> list[dict]:
    """n questions weighted as the official score weighs them: 1/8 per group, and the group of
    the three relative-direction levels split evenly among them. Dealt round-robin over those
    weights, so a short type gives its share to the others."""
    by = defaultdict(list)
    for q in qs:
        by[q["question_type"]].append(q)
    for v in by.values():
        rng.shuffle(v)
    # each direction level appears once per three rounds of the others
    order = sorted(t for t in by if group(t) == t)
    levels = sorted(t for t in by if group(t) != t)
    picked, r = [], 0
    while len(picked) < n and any(by.values()):
        for t in order + ([levels[r % len(levels)]] if levels else []):
            if by[t] and len(picked) < n:
                picked.append(by[t].pop())
        r += 1
    return sorted(picked, key=lambda q: q["id"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    raw = Path(os.environ["VSI_RAW"]).resolve() if os.environ.get("VSI_RAW") else out / "_raw"
    if not os.environ.get("VSI_RAW"):
        fetch(raw)
    rows = [json.loads(line) for line in (raw / "test.jsonl").read_text().splitlines()]
    pruned = set((raw / "pruned_ids.txt").read_text().split())
    for q in rows:
        q["debiased"] = str(q["id"]) not in pruned
        if not video(raw, q).is_file():
            raise SystemExit(f"missing video {video(raw, q)}")

    rng = random.Random(SEED)
    scenes = sorted({(q["dataset"], q["scene_name"]) for q in rows})
    rng.shuffle(scenes)
    cut1 = round(len(scenes) * SCENE_SHARE["test"])
    cut2 = cut1 + round(len(scenes) * SCENE_SHARE["val"])
    owner = {s: ("test" if i < cut1 else "val" if i < cut2 else "dev") for i, s in enumerate(scenes)}

    for split, n in QUESTIONS.items():
        pool = [q for q in rows if owner[(q["dataset"], q["scene_name"])] == split]
        qs = stratified(pool, n, rng)
        if split == "val" and VAL_TOTAL > n:
            taken = {q["id"] for q in qs}
            more = stratified([q for q in pool if q["id"] not in taken], VAL_TOTAL - n,
                              random.Random(SEED + 1))
            qs = sorted(qs + more, key=lambda q: q["id"])
        d = out / split
        if d.exists():
            shutil.rmtree(d)
        (d / "videos").mkdir(parents=True)
        for s in sorted({(q["dataset"], q["scene_name"]) for q in qs}):
            src, dst = video(raw, {"dataset": s[0], "scene_name": s[1]}), d / "videos" / f"{s[0]}__{s[1]}.mp4"
            # dev is the agent's data and the sandbox mounts only the split: real files, not links
            # into the raw tree it hides. val and test are read by the evaluator alone.
            shutil.copy2(src, dst) if split == "dev" else dst.symlink_to(src)
        public = [{"id": q["id"], "video": f"videos/{q['dataset']}__{q['scene_name']}.mp4",
                   "question_type": q["question_type"], "question": q["question"],
                   "options": q["options"]} for q in qs]
        labels = {str(q["id"]): {"answer": q["ground_truth"], "debiased": q["debiased"]} for q in qs}
        if split == "dev":  # the agent's data: answers included
            for p, q in zip(public, qs):
                p["answer"], p["debiased"] = q["ground_truth"], q["debiased"]
        (d / "questions.json").write_text(json.dumps(public, indent=1))
        (d / "labels.json").write_text(json.dumps(labels, indent=1))
        groups = sorted({group(q["question_type"]) for q in qs})
        print(f"{split}: {len(qs)} questions, {len({q['scene_name'] for q in qs})} scenes, "
              f"{len(groups)} groups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
