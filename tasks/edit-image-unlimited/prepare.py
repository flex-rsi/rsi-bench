"""Build the edit-image splits under --out (dev/val/test).

The images and annotations are NOT in this repository: they live in a private store given by
$EDIT_IMAGE_SRC (default below), laid out as

    annotations.jsonl    one sample per line, with a "split" field (dev / val / test)
    images/<id>.png      source images (CC0 / public domain, see SOURCES.md there)
    masks/<id>.png       optional edit masks (white = may change)

Each split directory gets samples.jsonl, images/ and masks/. dev keeps everything the agent may
read; val and test have the same layout but are only read by the evaluator.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

DEFAULT_SRC = "/data/edit-image/source"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=os.environ.get("EDIT_IMAGE_SRC", DEFAULT_SRC))
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    rows = [json.loads(l) for l in (src / "annotations.jsonl").read_text().splitlines() if l.strip()]
    counts = {}
    for split in ("dev", "val", "test"):
        d = out / split
        if d.exists():
            shutil.rmtree(d)
        (d / "images").mkdir(parents=True)
        (d / "masks").mkdir()
        with open(d / "samples.jsonl", "w") as f:
            for r in rows:
                if r["split"] != split:
                    continue
                shutil.copy(src / r["image"], d / r["image"])
                if r.get("edit_mask"):
                    shutil.copy(src / r["edit_mask"], d / r["edit_mask"])
                rec = {k: v for k, v in r.items() if k not in ("split", "source")}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[split] = counts.get(split, 0) + 1
    print(f"wrote edit-image splits to {out}: {counts}")


if __name__ == "__main__":
    main()
