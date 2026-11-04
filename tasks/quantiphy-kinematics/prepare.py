"""Build the quantiphy-kinematics splits under --out from the labeled QuantiPhy validation set
(Li et al., "QuantiPhy", Stanford 2025; CC BY 4.0): 158 questions on 24 videos, each asking for a
size, distance, velocity or acceleration in physical units, given one prior in physical units
(a known length, speed or acceleration, sometimes with depths).

    rsib task prepare quantiphy-kinematics                    # downloads at a pinned revision
    QUANTIPHY_SOURCE=/path/to/QuantiPhy-validation rsib task prepare quantiphy-kinematics

Splits are by video (no video is in two splits), fixed lists below, balanced by source
(simulation, lab, internet, segmentation) and by inference type (2D/3D x static/dynamic prior):
    dev 39 questions on 6 videos    val 52 on 8    test 67 on 10
Each split gets videos/ (the mp4 files), questions.jsonl ({"id", "video", "video_id", "source",
"video_type", "fps", "inference_type", "question", "prior", "depth_info"}) and labels.jsonl
({"id", "answer"}).
"""

import argparse
import json
import math
import os
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

REPO, REV = "PaulineLi/QuantiPhy-validation", "2d56ff31c22c12bf377b2c2092ff229d291fd830"
SPLITS = {
    "dev": ["captured_0013", "internet_0001", "internet_0004", "internet_0019", "simulation_0060",
            "simulation_0098_segmented"],
    "val": ["captured_0027s", "captured_0034as", "internet_0027", "simulation_0034_segmented",
            "simulation_0170_segmented", "simulation_0193", "simulation_0196", "simulation_0300"],
    "test": ["captured_0021x", "captured_0041x", "captured_0061", "internet_0005", "internet_0013",
             "simulation_0009", "simulation_0015", "simulation_0089", "simulation_0163",
             "simulation_0171"],
}


def fetch(rel: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/{urllib.parse.quote(rel)}"
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    return dest


def text(value) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value).strip() or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    import pandas as pd

    src = Path(os.environ["QUANTIPHY_SOURCE"]) if os.environ.get("QUANTIPHY_SOURCE") else None
    cache = out / "_download"
    table = src / "validation_dataset.parquet" if src else \
        fetch("validation_dataset.parquet", cache / "validation_dataset.parquet")
    rows = pd.read_parquet(table).to_dict("records")
    names = None
    if src:
        names = {p.name.strip(): p for p in (src / "validation_videos").iterdir()}
    else:
        from huggingface_hub import list_repo_files
        names = {Path(f).name.strip(): f for f in list_repo_files(REPO, repo_type="dataset",
                                                                    revision=REV)
                 if f.startswith("validation_videos/")}

    assigned = {v: split for split, vids in SPLITS.items() for v in vids}
    assert len(assigned) == 24 and {r["video_id"] for r in rows} == set(assigned)
    per_video: dict[str, int] = {}
    for split in SPLITS:
        d = out / split
        if d.exists():
            shutil.rmtree(d)
        (d / "videos").mkdir(parents=True)
    questions = {s: [] for s in SPLITS}
    labels = {s: [] for s in SPLITS}
    for r in rows:
        vid, split = r["video_id"], assigned[r["video_id"]]
        k = per_video[vid] = per_video.get(vid, 0) + 1
        qid = f"{vid}_q{k:02d}"
        video = out / split / "videos" / f"{vid}.mp4"
        if not video.exists():
            origin = names[f"{vid}.mp4"]
            got = origin if src else fetch(origin, cache / "videos" / f"{vid}.mp4")
            shutil.copy(got, video)
        questions[split].append({
            "id": qid, "video": f"videos/{vid}.mp4", "video_id": vid, "source": r["video_source"],
            "video_type": r["video_type"], "fps": float(r["fps"]),
            "inference_type": r["inference_type"], "question": text(r["question"]),
            "prior": text(r["prior"]), "depth_info": text(r["depth_info"])})
        labels[split].append({"id": qid, "answer": float(r["answer"])})
    for split in SPLITS:
        d = out / split
        (d / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in questions[split]))
        (d / "labels.jsonl").write_text("".join(json.dumps(x) + "\n" for x in labels[split]))
        print(split, len(questions[split]), "questions,", len(SPLITS[split]), "videos")
    shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    main()
