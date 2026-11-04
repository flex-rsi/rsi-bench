"""Build the edit-multiturn splits under --out from MSE-Bench, the multi-turn editing benchmark of
VINCIE (Qu et al., "VINCIE: Unlocking In-context Image Editing from Video", arXiv:2506.10941, 2025;
Apache-2.0): 100 editing sessions on real photos, each with 5 instructions applied one after another.

    rsib task prepare edit-multiturn                                 # downloads at a pinned revision
    MSE_BENCH_PARQUET=/path/to/train-00000-of-00001.parquet rsib task prepare edit-multiturn

Splits are by session, fixed lists below (dev 30 / val 30 / test 40 sessions), chosen by a seeded
search (seed 0) that balances the turn-level counts of the 20 operation types, the sessions with a
camera operation (dolly, tilt, pan, orbiting) and the images under 512 px on the short side.
Each split gets
    images/<id>.png   the source image, byte-identical to MSE-Bench (not resized)
    sessions.jsonl    {"id", "image", "width", "height", "instructions": [5 strings]}
    labels.jsonl      {"id", "index", "operation": [5], "description_target": [5],
                      "description_source", "summary"}: the MSE-Bench annotations, for analysis
                      and the means per operation type (the judge sees only the instructions)
A solution never gets these files: the evaluator hands it the image and one instruction at a time.
dev keeps everything the agent may read; val and test have the same layout but only the evaluator
reads them.
"""

import argparse
import hashlib
import io
import json
import os
import shutil
import urllib.request
from pathlib import Path

REPO, REV = "leigangqu/MSE-Bench", "1f77f7832b65d0c9e0539019840a521f676d0f2c"
FILE = "data/train-00000-of-00001.parquet"
SHA256 = "4f252b4f74af790a853602d4b4720b6781b820afab39fda441bac709d3d3b5bd"
SPLITS = {  # MSE-Bench "index" of the sessions in each split
    "dev": [3, 4, 8, 9, 11, 16, 17, 18, 20, 28, 41, 42, 43, 45, 46, 47, 48, 59, 60, 61, 62, 73, 77,
            79, 82, 83, 85, 87, 90, 99],
    "val": [0, 10, 13, 19, 21, 23, 24, 25, 26, 33, 34, 35, 36, 37, 49, 56, 63, 64, 66, 67, 72, 76,
            80, 81, 86, 88, 92, 93, 95, 97],
    "test": [1, 2, 5, 6, 7, 12, 14, 15, 22, 27, 29, 30, 31, 32, 38, 39, 40, 44, 50, 51, 52, 53, 54,
             55, 57, 58, 65, 68, 69, 70, 71, 74, 75, 78, 84, 89, 91, 94, 96, 98],
}
TURNS = 5


def fetch(dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/{FILE}"
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    return dest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonl(path: Path, records: list) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit("prepare.py needs pyarrow (pip install pyarrow) to read MSE-Bench")
    from PIL import Image

    cache = out / "_download"
    src = os.environ.get("MSE_BENCH_PARQUET")
    table = Path(src) if src else fetch(cache / Path(FILE).name)
    if sha256(table) != SHA256:
        raise SystemExit(f"{table}: not the MSE-Bench file of revision {REV[:7]} (sha256 differs)")
    rows = {r["index"]: r for r in pq.read_table(table).to_pylist()}
    assigned = {i: split for split, ids in SPLITS.items() for i in ids}
    assert len(assigned) == 100 and set(rows) == set(assigned), "split lists must cover MSE-Bench"

    for split, ids in SPLITS.items():
        d = out / split
        if d.exists():
            shutil.rmtree(d)
        (d / "images").mkdir(parents=True)
        sessions, labels = [], []
        for i in ids:
            r = rows[i]
            assert len(r["context"]) == len(r["operation"]) == len(r["description_target"]) == TURNS
            sid = f"s{i:03d}"
            data = r["image"]["bytes"]
            im = Image.open(io.BytesIO(data))
            assert im.format == "PNG", f"{sid}: {im.format}"
            (d / "images" / f"{sid}.png").write_bytes(data)
            sessions.append({"id": sid, "image": f"images/{sid}.png", "width": im.width,
                             "height": im.height, "instructions": list(r["context"])})
            labels.append({"id": sid, "index": i, "operation": list(r["operation"]),
                           "description_target": list(r["description_target"]),
                           "description_source": r["description_source"], "summary": r["summary"]})
        jsonl(d / "sessions.jsonl", sessions)
        jsonl(d / "labels.jsonl", labels)
        print(f"{split}: {len(ids)} sessions, {len(ids) * TURNS} turns")
    shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    main()
