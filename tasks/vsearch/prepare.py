"""Build the vsearch splits under --out from two public visual-search benchmarks:

  V*Bench (Wu & Xie, CVPR 2024): 191 questions on ~2K photos, 115 direct-attribute questions with
    four options and 76 relative-position questions with two;
  HR-Bench 8K (Wang et al., arXiv 2408.15556): 200 questions on 151 photos of about 8K (median
    39 MP), 100 single-instance and 100 cross-instance perception questions with four options. The
    benchmark lists every question four times with its options rotated; one rotation is taken per
    question (fixed seed), so the right letter varies.

    rsib task prepare vsearch          # downloads both at pinned revisions (needs pyarrow)
    VSTAR_SOURCE=... HRBENCH_PARQUET=... rsib task prepare vsearch   # or local copies

Splits (V* by category, HR-Bench by photo so that no photo is in two splits; fixed seed):
    dev   111 questions (V* 51, HR 60)    val 140 (V* 70, HR 70)    test 140 (V* 70, HR 70)
and, of dev and val, a fixed-seed subset stratified by benchmark and category (every question costs
model calls of the model under test); test is kept whole:
    dev    70 questions (V* 33, HR 37)    val  80 (V* 40, HR 40)    test 140 (V* 70, HR 70)
Each split gets images/ (originals, renamed), questions.jsonl ({"id", "image", "source",
"category", "question", "options", "prompt"}) and labels.jsonl ({"id", "answer"}); dev also gets
targets.jsonl with the boxes of the objects each V* question is about.
"""

import argparse
import base64
import hashlib
import json
import os
import random
import re
import shutil
import urllib.request
from pathlib import Path

VSTAR = ("craigwu/vstar_bench", "d9ae62c903da0c98336e85c5ee89cd863b04b4da")
HRBENCH = ("DreamMr/HR-Bench", "83b9013d6293", "hr_bench_8k.parquet")
SEED = 0
VSTAR_SPLIT = {"dev": {"direct_attributes": 31, "relative_position": 20},
               "val": {"direct_attributes": 42, "relative_position": 28},
               "test": {"direct_attributes": 42, "relative_position": 28}}
HR_SPLIT = {"dev": 60, "val": 70, "test": 70}
KEEP = {"dev": {("vstar", "direct_attributes"): 20, ("vstar", "relative_position"): 13,
                ("hrbench8k", "single"): 21, ("hrbench8k", "cross"): 16},
        "val": {("vstar", "direct_attributes"): 24, ("vstar", "relative_position"): 16,
                ("hrbench8k", "single"): 19, ("hrbench8k", "cross"): 21}}  # test: all
FORMAT = "Answer with the option's letter from the given choices directly."
OPTION = re.compile(r"^\(([A-Z])\) (.*)$")


def download(repo: str, rev: str, rel: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        urllib.request.urlretrieve(f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{rel}", tmp)
        tmp.rename(dest)
    return dest


def vstar_items(cache: Path) -> list[dict]:
    src = Path(os.environ["VSTAR_SOURCE"]) if os.environ.get("VSTAR_SOURCE") else None
    get = (lambda rel: src / rel) if src else (lambda rel: download(*VSTAR, rel, cache / "vstar" / rel))
    items = []
    for line in get("test_questions.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        lines = row["text"].split("\n")
        meta = json.loads(get(str(Path(row["image"]).with_suffix(".json"))).read_text())
        items.append({"key": f"vstar-{int(row['question_id']):03d}", "source": "vstar",
                      "category": row["category"], "question": lines[0],
                      "options": [m.group(2) for ln in lines if (m := OPTION.match(ln))],
                      "prompt": row["text"], "answer": row["label"],
                      "image_bytes": get(row["image"]).read_bytes(), "group": row["image"],
                      "targets": [{"name": n, "box": b} for n, b in zip(meta["target_object"],
                                                                       meta["bbox"])]})
    return items


def hrbench_items(cache: Path) -> list[dict]:
    import pyarrow.parquet as pq  # only data preparation needs it
    path = Path(os.environ["HRBENCH_PARQUET"]) if os.environ.get("HRBENCH_PARQUET") else \
        download(HRBENCH[0], HRBENCH[1], HRBENCH[2], cache / "hrbench" / HRBENCH[2])
    groups: dict[int, list[dict]] = {}
    for row in pq.read_table(path).to_pylist():  # each question four times, options rotated
        groups.setdefault(int(row["index"]) // 4, []).append(row)
    rng = random.Random(SEED)
    items = []
    for g in sorted(groups):
        rows = sorted(groups[g], key=lambda r: int(r["cycle_category"]))
        assert [int(r["cycle_category"]) for r in rows] == [0, 1, 2, 3]
        row = rows[rng.randrange(4)]
        data = base64.b64decode(row["image"])
        options = [row[k] for k in "ABCD"]
        items.append({"key": f"hr8k-{g:03d}", "source": "hrbench8k", "category": row["category"],
                      "question": row["question"], "options": options,
                      "prompt": "\n".join([row["question"]] + [f"({k}) {o}" for k, o in
                                                                zip("ABCD", options)] + [FORMAT]),
                      "answer": row["answer"], "image_bytes": data,
                      "group": hashlib.sha1(data).hexdigest(), "targets": []})
    return items


def split_vstar(items: list[dict], rng: random.Random) -> dict[str, list[dict]]:
    by_cat: dict[str, list[dict]] = {}
    for it in sorted(items, key=lambda i: i["key"]):
        by_cat.setdefault(it["category"], []).append(it)
    out = {s: [] for s in VSTAR_SPLIT}
    for cat, group in by_cat.items():
        rng.shuffle(group)
        start = 0
        for split, counts in VSTAR_SPLIT.items():
            out[split] += group[start:start + counts[cat]]
            start += counts[cat]
        assert start == len(group)
    return out


def split_hrbench(items: list[dict], rng: random.Random) -> dict[str, list[dict]]:
    """Whole photos to splits (a photo's questions stay together), filling dev, val, test in turn."""
    photos: dict[str, list[dict]] = {}
    for it in sorted(items, key=lambda i: i["key"]):
        photos.setdefault(it["group"], []).append(it)
    order = sorted(photos)
    rng.shuffle(order)
    out = {s: [] for s in HR_SPLIT}
    splits = list(HR_SPLIT)
    for g in order:  # the first split still short of its target takes the photo
        target = next((s for s in splits if len(out[s]) < HR_SPLIT[s]), splits[-1])
        out[target] += photos[g]
    return out


def subsample(items: list[dict], keep: dict[tuple[str, str], int], rng: random.Random) -> list[dict]:
    """keep[(source, category)] of the split's questions of each benchmark and category."""
    out = []
    for (source, category), n in keep.items():
        group = sorted((it for it in items if (it["source"], it["category"]) == (source, category)),
                       key=lambda i: i["key"])
        out += rng.sample(group, n)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    cache = out / "_download"
    rng, keep_rng = random.Random(SEED), random.Random(SEED + 1)
    parts = [split_vstar(vstar_items(cache), rng), split_hrbench(hrbench_items(cache), rng)]
    for split in ("dev", "val", "test"):
        d = out / split
        if d.exists():
            shutil.rmtree(d)
        (d / "images").mkdir(parents=True)
        chosen = parts[0][split] + parts[1][split]
        if split in KEEP:
            chosen = subsample(chosen, KEEP[split], keep_rng)
        rng.shuffle(chosen)
        questions, labels, targets = [], [], []
        for i, it in enumerate(chosen):
            qid = f"{split}-{i:03d}"
            ext = "png" if it["image_bytes"][:4] == b"\x89PNG" else "jpg"
            image = f"images/{qid}.{ext}"
            (d / image).write_bytes(it["image_bytes"])
            questions.append({"id": qid, "image": image, "source": it["source"],
                              "category": it["category"], "question": it["question"],
                              "options": it["options"], "prompt": it["prompt"]})
            labels.append({"id": qid, "answer": it["answer"]})
            targets.append({"id": qid, "targets": it["targets"]})
        write = lambda name, rows: (d / name).write_text("".join(json.dumps(r) + "\n" for r in rows))  # noqa: E731
        write("questions.jsonl", questions)
        write("labels.jsonl", labels)
        if split == "dev":
            write("targets.jsonl", targets)
        n = {s: sum(q["source"] == s for q in questions) for s in ("vstar", "hrbench8k")}
        print(f"{split}: {len(questions)} questions {n}")
    print(f"wrote vsearch splits to {out}")


if __name__ == "__main__":
    main()
