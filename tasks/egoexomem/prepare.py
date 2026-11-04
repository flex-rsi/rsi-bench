"""Build the egoexomem splits under --out from EgoExoMem's EgoExo4D questions and EgoExo4D videos.

  EgoExoMem (Liu et al., arXiv 2605.18734): multiple-choice questions about memory over
    synchronized egocentric and exocentric video. This task uses its EgoExo4D part:
    764 human-verified questions with four options on 125 cooking and bike-repair takes. The
    questions are downloaded from the benchmark's repository at a pinned commit (CC BY 4.0).
  EgoExo4D (Grauman et al., CVPR 2024): the takes' frame-aligned videos, downscaled to 448 pixels:
    the wearer's Aria camera (ego, 448 x 448) and exocentric camera cam01 (exo, 796 x 448), both
    30 fps and synchronized. EgoExo4D's licence does not allow redistribution, so the videos come
    from a local copy: set EGOEXO4D_ROOT to the directory that holds takes/.

    EGOEXO4D_ROOT=/path/to/egoexo4d rsib task prepare egoexomem
    EGOEXOMEM_QA=/path/to/qa_merged_v6.json ...    # a local copy of the questions instead

Splits keep every capture (the takes of one kitchen or workshop session) in one split, stratified
by scenario family (cooking, bike repair), fixed seed: about 35% / 25% / 40% of the questions for
dev / val / test (277 / 189 / 298). Every question costs model calls of the model under test, so
each split keeps a fixed-seed sample: dev 80, val 80, test 120 questions, stratified by scenario
family and category and spread over the split's videos; only the videos of kept questions are
built. The options of every question are shuffled once (fixed seed); videos and questions get
opaque ids. Each split gets
    videos/<video>/{ego,exo}.mp4          the 448-pixel videos, metadata stripped
    frames/<video>/{ego,exo}/NNNNNN.jpg   one frame per second: frame k is at k seconds
    videos.jsonl      {"video", "duration_s", "ego": {...}, "exo": {...}}
    questions.jsonl   {"id", "video", "family", "scenario", "category", "question", "options", "prompt",
                       "duration_s"}
    labels.jsonl      {"id", "answer"}
and dev also evidence.jsonl ({"id", "timestamps", "view", "note"}: the moments the answer is
visible, the view that shows it, the annotators' check note).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import shutil
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

QA = ("RuipingL/EgoExoMem", "c5fb50a4d3c6a7a3ebd0adca2a679073c4f96a2c", "data/EgoExo4D/qa_merged_v6.json")
SEED = 0
SHARES = {"dev": 0.35, "val": 0.25, "test": 0.40}
KEEP = {"dev": 80, "val": 80, "test": 120}   # questions kept per split (every one costs model calls)
FPS = 30            # both streams of every take (checked)
CATEGORIES = {"Q1": "habitual_location", "Q2": "instantaneous_position", "Q3": "resulting_location",
              "Q4": "egocentric_direction", "Q5": "object_state", "Q6": "allocentric_relation",
              "Q7": "third_person_activity", "Q8": "temporal_ordering"}
FORMAT = "Answer with the option's letter from the given choices directly."
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE", "ffprobe")


def questions(cache: Path) -> list[dict]:
    path = Path(os.environ["EGOEXOMEM_QA"]) if os.environ.get("EGOEXOMEM_QA") else cache / "qa_merged_v6.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        repo, rev, rel = QA
        urllib.request.urlretrieve(f"https://raw.githubusercontent.com/{repo}/{rev}/{rel}", path)
    return json.loads(path.read_text())


def sources(root: Path, take: str) -> dict:
    base = root / "takes" / take / "frame_aligned_videos" / "downscaled" / "448"
    ego = sorted(base.glob("aria*_214-1.mp4"))
    if not ego or not (base / "cam01.mp4").exists():
        raise FileNotFoundError(f"{base}: needs aria*_214-1.mp4 and cam01.mp4")
    return {"ego": ego[0], "exo": base / "cam01.mp4"}


def probe(path: Path) -> dict:
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets",
                          "-of", "json", str(path)], check=True, capture_output=True, text=True)
    s = json.loads(out.stdout)["streams"][0]
    num, den = (int(x) for x in s["r_frame_rate"].split("/"))
    assert num == FPS * den, f"{path}: {s['r_frame_rate']} fps"
    return {"width": s["width"], "height": s["height"], "fps": FPS, "frames": int(s["nb_read_packets"])}


def build_video(src: dict, dest: Path, frames: Path) -> dict:
    """Copy both streams without metadata and cut one frame per second; returns their facts."""
    info = {}
    for view, path in src.items():
        dest.mkdir(parents=True, exist_ok=True)
        out = dest / f"{view}.mp4"
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(path), "-map", "0:v:0",
                        "-c", "copy", "-map_metadata", "-1", "-movflags", "+faststart", str(out)], check=True)
        fdir = frames / view
        fdir.mkdir(parents=True, exist_ok=True)
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(out),
                        "-vf", f"select=not(mod(n\\,{FPS}))", "-fps_mode", "vfr", "-q:v", "3",
                        "-start_number", "0", str(fdir / "%06d.jpg")], check=True)
        info[view] = {**probe(out), "frames_1fps": len(list(fdir.glob("*.jpg")))}
    info["duration_s"] = round(min(info[v]["frames"] for v in src) / FPS, 3)
    return info


def family(take: str) -> str:
    return "bike" if "bike" in take else "cooking"


def capture(take: str) -> str:
    """The session a take belongs to: its name without the take number (same room, same people)."""
    return re.sub(r"_\d+$", "", take)


def split_captures(qa: list[dict]) -> dict[str, str]:
    """capture -> split: within each scenario family, captures in seeded order go to the split
    furthest below its share of that family's questions."""
    count = collections.Counter(capture(q["take_name"]) for q in qa)
    fam_of = {capture(q["take_name"]): family(q["take_name"]) for q in qa}
    rng = random.Random(SEED)
    out = {}
    for fam in sorted(set(fam_of.values())):
        caps = sorted(c for c in count if fam_of[c] == fam)
        rng.shuffle(caps)
        total, have = sum(count[c] for c in caps), dict.fromkeys(SHARES, 0)
        for c in caps:
            split = max(SHARES, key=lambda s: SHARES[s] * total - have[s])
            out[c] = split
            have[split] += count[c]
    return out


def subsample(items: list, n: int, rng: random.Random) -> list:
    """n of a split's questions: per (scenario family, category) in proportion to the split
    (largest remainders), and within each such group one question per video in turn, videos in
    seeded order, so that the sample spreads over the split's videos."""
    groups = collections.defaultdict(list)
    for it in items:
        groups[(family(it[0]["take_name"]), it[0]["qa_type"])].append(it)
    share = {g: n * len(v) / len(items) for g, v in groups.items()}
    quota = {g: int(x) for g, x in share.items()}
    for g in sorted(share, key=lambda g: (quota[g] - share[g], g))[: n - sum(quota.values())]:
        quota[g] += 1
    out = []
    for g in sorted(groups):
        by_video = collections.defaultdict(list)
        for it in groups[g]:
            by_video[it[0]["take_name"]].append(it)
        videos = sorted(by_video)
        rng.shuffle(videos)
        for v in videos:
            rng.shuffle(by_video[v])
        picked = []
        while len(picked) < quota[g]:
            for v in videos:
                if by_video[v] and len(picked) < quota[g]:
                    picked.append(by_video[v].pop())
        out += picked
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    root = Path(os.environ["EGOEXO4D_ROOT"])
    qa = questions(out / "_download")
    where = split_captures(qa)
    rng = random.Random(SEED)

    rows = collections.defaultdict(list)
    for q in sorted(qa, key=lambda q: q["qa_id"]):
        split = where[capture(q["take_name"])]
        order = list(range(len(q["options"])))
        rng.shuffle(order)
        options = [q["options"][i] for i in order]
        answer = "ABCD"[order.index(q["correct_answer"])]
        rows[split].append((q, options, answer))
    rows = {split: subsample(items, KEEP[split], rng) for split, items in rows.items()}

    takes = sorted({q["take_name"] for items in rows.values() for q, _, _ in items})
    ids = [f"v{n:03d}" for n in range(len(takes))]
    rng.shuffle(ids)
    video_of = dict(zip(takes, ids))

    for split in SHARES:
        if (out / split).exists():
            shutil.rmtree(out / split)

    def one(take: str):
        split, vid = where[capture(take)], video_of[take]
        info = build_video(sources(root, take), out / split / "videos" / vid, out / split / "frames" / vid)
        return take, {"video": vid, **info}

    with ThreadPoolExecutor(args.workers) as pool:
        facts = dict(pool.map(one, takes))

    for split, items in rows.items():
        rng.shuffle(items)
        d = out / split
        qs, labels, evidence = [], [], []
        for n, (q, options, answer) in enumerate(items):
            qid = f"{split}-{n:03d}"
            fact = facts[q["take_name"]]
            prompt = "\n".join([q["question"]] + [f"({'ABCD'[i]}) {o}" for i, o in enumerate(options)] + [FORMAT])
            qs.append({"id": qid, "video": fact["video"], "family": family(q["take_name"]), "scenario": q["scenario"],
                       "category": CATEGORIES[q["qa_type"]], "question": q["question"],
                       "options": options, "prompt": prompt, "duration_s": fact["duration_s"]})
            labels.append({"id": qid, "answer": answer})
            v = q.get("verification") or {}
            evidence.append({"id": qid, "timestamps": [round(t, 2) for t in v.get("timestamps") or []],
                             "view": q.get("required_view"), "note": v.get("check_description", "")})
        videos = sorted((f for t, f in facts.items() if where[capture(t)] == split), key=lambda f: f["video"])
        (d / "videos.jsonl").write_text("".join(json.dumps(f) + "\n" for f in videos))
        (d / "questions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in qs))
        (d / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in labels))
        if split == "dev":
            (d / "evidence.jsonl").write_text("".join(json.dumps(r) + "\n" for r in evidence))
        cats = collections.Counter(r["category"] for r in qs)
        print(f"{split}: {len(qs)} questions on {len(videos)} videos, "
              f"{sum(f['duration_s'] for f in videos) / 60:.0f} min; {dict(sorted(cats.items()))}")


if __name__ == "__main__":
    main()
