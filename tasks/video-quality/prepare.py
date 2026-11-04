"""Build the video-quality splits under --out from a local copy of VQA-TestSet-200 (VQA_ROOT).

  VQA-TestSet-200 is a 200-video test set for video quality assessment in two parts:
  A. 100 real user-generated videos from YouTube-UGC (Wang, Inguva, Adsumilli, MMSP 2019;
     CC BY 4.0): 16.6-20 s, 360p-1080p, 14 content categories, each with a mean opinion score
     (MOS, 1-5) from 105-154 raters.
  B. 50 pairs of text-to-video clips from Rapidata's Veo 3.1 human-preference set (2025,
     Apache-2.0): Veo 3.1 against one of seven other models on the same prompt, 5-6 s. Each pair has
     15 weighted votes on each of three questions: which clip matches the prompt better
     (alignment), which shows more artifacts (glitch_rating; the clip with fewer wins), which looks
     better (preference). A side with a share of 0.45-0.55 is a tie.

    VQA_ROOT=/path/to/VQA_TestSet_200 rsib task prepare video-quality

Splits, fixed seed: the rated videos are dealt in order of content category and MOS, so every
split covers every category and the whole range of scores; the pairs in order of the opposing
model. dev / val / test = 30 / 30 / 40 videos and 15 / 15 / 20 pairs. Videos get opaque ids and
their metadata is stripped (stream copy); the two clips of a pair are put in a seeded random order
(Veo 3.1 is always first in the source). Each split gets
    videos/<video>.mp4           H.264, no audio
    frames/<video>/NNNNNN.jpg    two frames per second at full resolution: frame k is at k/2 s
    videos.jsonl      {"video", "duration_s", "width", "height", "fps", "frames"}
    items.jsonl       {"id", "kind": "rate", "video", "category"} and
                      {"id", "kind": "compare", "prompt", "videos": [first, second]}
    labels.jsonl      {"id", "mos"} and {"id", "alignment", "artifacts", "preference"}, each
                      "A" (the first clip is better), "B" or "tie"
and dev also details.jsonl: the MOS's spread, raters and 10-second-window scores of each rated
video, and the vote shares of each pair (in the split's order; artifacts as the share that found
the clip the more glitchy one).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SEED = 0
SHARES = {"dev": 0.30, "val": 0.30, "test": 0.40}
FRAMES_PER_SECOND = 2
CRITERIA = {"alignment": "alignment", "artifacts": "glitch_rating", "preference": "preference"}
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE", "ffprobe")


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def deal(items: list, rng: random.Random) -> dict[str, list]:
    """Items in the given order go, one at a time, to the split furthest below its share so far;
    ties are broken in a seeded order of the splits."""
    order = list(SHARES)
    rng.shuffle(order)
    out = {s: [] for s in SHARES}
    for n, item in enumerate(items, 1):
        split = max(order, key=lambda s: SHARES[s] * n - len(out[s]))
        out[split].append(item)
    return out


def winner(row: dict, column: str, swapped: bool) -> str:
    """The split's label of one criterion: the source's winner (the better clip), in our order."""
    w = row[f"{column}_winner"]
    if w not in ("A", "B"):
        return "tie"
    return {"A": "B", "B": "A"}[w] if swapped else w


def probe(path: Path) -> dict:
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,avg_frame_rate:format=duration", "-of", "json", str(path)],
                         check=True, capture_output=True, text=True)
    info = json.loads(out.stdout)
    s = info["streams"][0]
    num, den = (int(x) for x in s["avg_frame_rate"].split("/"))
    return {"duration_s": round(float(info["format"]["duration"]), 3), "width": s["width"],
            "height": s["height"], "fps": round(num / den, 3)}


def build_video(src: Path, dest: Path, frames: Path) -> dict:
    """Copy the stream without metadata and cut two frames per second; returns its facts."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-map", "0:v:0",
                    "-c", "copy", "-map_metadata", "-1", "-movflags", "+faststart", str(dest)], check=True)
    frames.mkdir(parents=True, exist_ok=True)
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(dest),
                    "-vf", f"fps={FRAMES_PER_SECOND}", "-q:v", "2", "-start_number", "0",
                    str(frames / "%06d.jpg")], check=True)
    return {**probe(dest), "frames": len(list(frames.glob("*.jpg")))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    root = Path(os.environ["VQA_ROOT"])
    rng = random.Random(SEED)

    rated = read_csv(root / "annotations/A_real_ugc_mos.csv")
    pairs = read_csv(root / "annotations/B_aigc_pairwise_summary.csv")
    assert len(rated) == 100 and len(pairs) == 50, (len(rated), len(pairs))
    rated_splits = deal(sorted(rated, key=lambda r: (r["category"], float(r["mos_full"]), r["item_id"])), rng)
    opponent = {r["pair_id"]: r["model_B"] for r in pairs}
    assert all(r["model_A"] == "veo31" for r in pairs)
    pair_splits = deal(sorted(pairs, key=lambda r: (opponent[r["pair_id"]], r["pair_id"])), rng)

    # opaque ids for every clip, in a seeded order; a seeded side order for every pair
    sources = [r["local_path"] for r in rated] + [r[f"video_{s}"] for r in pairs for s in "AB"]
    ids = [f"v{n:03d}" for n in range(len(sources))]
    rng.shuffle(ids)
    video_of = dict(zip(sources, ids))
    swapped = {r["pair_id"]: rng.random() < 0.5 for r in sorted(pairs, key=lambda r: r["pair_id"])}
    split_of = {r["local_path"]: s for s, rows in rated_splits.items() for r in rows}
    split_of.update({r[f"video_{x}"]: s for s, rows in pair_splits.items() for r in rows for x in "AB"})

    for split in SHARES:
        if (out / split).exists():
            shutil.rmtree(out / split)

    def one(src: str):
        d = out / split_of[src]
        vid = video_of[src]
        return src, {"video": vid, **build_video(root / src, d / "videos" / f"{vid}.mp4", d / "frames" / vid)}

    with ThreadPoolExecutor(args.workers) as pool:
        facts = dict(pool.map(one, sorted(sources)))

    for split in SHARES:
        d = out / split
        items, labels, details = [], [], []
        for r in rated_splits[split]:
            items.append({"kind": "rate", "video": video_of[r["local_path"]], "category": r["category"]})
            labels.append({"mos": float(r["mos_full"])})
            details.append({"mos": float(r["mos_full"]), "mos_std": float(r["std_full"]),
                            "raters": int(r["num_subjects"]),
                            "mos_windows": {w: float(r[f"mos_chunk_{w}"]) for w in ("0_10s", "5_15s", "10_20s")}})
        for r in pair_splits[split]:
            sw = swapped[r["pair_id"]]
            first, second = (r["video_B"], r["video_A"]) if sw else (r["video_A"], r["video_B"])
            items.append({"kind": "compare", "prompt": r["prompt"], "videos": [video_of[first], video_of[second]]})
            labels.append({k: winner(r, col, sw) for k, col in CRITERIA.items()})
            shares = {}
            for k, col in CRITERIA.items():
                a, b = float(r[f"{col}_A"]), float(r[f"{col}_B"])
                shares[k] = [b, a] if sw else [a, b]
            details.append({"votes_per_criterion": int(r["preference_num_votes"]), "shares": shares})
        order = list(range(len(items)))
        rng.shuffle(order)
        rows = []
        for n, i in enumerate(order):
            qid = f"{split}-{n:03d}"
            rows.append(({"id": qid, **items[i]}, {"id": qid, **labels[i]}, {"id": qid, **details[i]}))
        videos = sorted((f for s, f in facts.items() if split_of[s] == split), key=lambda f: f["video"])
        (d / "videos.jsonl").write_text("".join(json.dumps(f) + "\n" for f in videos))
        (d / "items.jsonl").write_text("".join(json.dumps(r[0]) + "\n" for r in rows))
        (d / "labels.jsonl").write_text("".join(json.dumps(r[1]) + "\n" for r in rows))
        if split == "dev":
            (d / "details.jsonl").write_text("".join(json.dumps(r[2]) + "\n" for r in rows))
        rates = [r for r in rows if r[0]["kind"] == "rate"]
        ties = sum(v == "tie" for r in rows if r[0]["kind"] == "compare" for k, v in r[1].items() if k != "id")
        print(f"{split}: {len(rates)} rated videos ({len({r[0]['category'] for r in rates})} categories, "
              f"MOS {min(r[1]['mos'] for r in rates):.2f}-{max(r[1]['mos'] for r in rates):.2f}), "
              f"{len(rows) - len(rates)} pairs ({ties} tied judgments), {len(videos)} videos, "
              f"{sum(f['frames'] for f in videos)} frames")


if __name__ == "__main__":
    main()
