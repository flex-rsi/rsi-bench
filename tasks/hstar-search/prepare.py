"""Build the hstar-search splits under --out from H*Bench and an in-the-wild test set.

Sources (not part of the task package):
    --bench   H*Bench scenes: <bench>/hos_bench/<scene>/ and <bench>/hps_bench/<scene>/, each with
              one panorama (.jpg/.png) and annotation.json (the benchmark of "Thinking in 360°:
              Humanoid Visual Search in the Wild", huggingface.co/datasets/humanoid-vstar/hstar_bench)
    --wild    the in-the-wild test scenes: <wild>/<scene>/annotation.json with the panorama in the
              same directory or next to it as <wild>/<scene>.(jpg|png)

Every annotated task becomes one episode with one starting direction, drawn with a fixed seed from
the task's annotated starting yaws (HOS: 'initial yaw' with a difficulty 'level' per start; tasks
without them start from 0, 90, 180 or 270 degrees). A task with a 'pitch' range is an object
search (HOS), one without is a path search (HPS).

    dev    80 H*Bench episodes (40 HOS, 40 HPS)    labels and panoramas visible to the agent
    val   100 H*Bench episodes (50 HOS, 50 HPS)    hidden
    test  every task of the in-the-wild set          hidden

dev and val are stratified by kind and difficulty level in proportion to H*Bench, and every
panorama gives at most one episode, so the two splits never share a scene (identical panoramas,
found by content hash, count as one scene).

Each split: episodes.jsonl ({"id", "kind", "task", "start_yaw", "scene"}), labels.jsonl ({"id",
"kind", "level", "yaw": [lo, hi], "pitch": [lo, hi] | null}) and panoramas/<scene>.<ext>.
"""

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20261007
SIZES = {"dev": {"hos": 40, "hps": 40}, "val": {"hos": 50, "hps": 50}}
IMAGE_EXT = (".jpg", ".jpeg", ".png")


def panorama_of(scene_dir: Path) -> Path | None:
    inside = sorted(p for p in scene_dir.iterdir() if p.suffix.lower() in IMAGE_EXT)
    if inside:
        return inside[0]
    beside = [scene_dir.with_suffix(ext) for ext in IMAGE_EXT if scene_dir.with_suffix(ext).exists()]
    return beside[0] if beside else None


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def episodes_of(scene_dir: Path, prefix: str, rng: random.Random) -> list[dict]:
    """One episode per annotated task: kind, instruction, a starting yaw and its level, targets."""
    tasks = json.loads((scene_dir / "annotation.json").read_text())
    tasks = tasks if isinstance(tasks, list) else [tasks]
    out = []
    for i, t in enumerate(tasks):
        kind = "hos" if "pitch" in t else "hps"
        if t.get("initial yaw"):
            j = rng.randrange(len(t["initial yaw"]))
            start, level = t["initial yaw"][j], t["level"][j] if isinstance(t["level"], list) else t["level"]
        else:
            start, level = rng.choice([0, 90, 180, 270]), t.get("level")
        out.append({"id": f"{prefix}{scene_dir.name}-{i}", "kind": kind, "task": t["task"],
                    "start_yaw": int(start), "level": level, "yaw": [float(v) for v in t["yaw"]],
                    "pitch": [float(v) for v in t["pitch"]] if kind == "hos" else None})
    return out


def quotas(pool: list[dict], n: int) -> dict:
    """n spread over difficulty levels in proportion to the pool (largest remainder)."""
    count = Counter(e["level"] for e in pool)
    raw = {lv: n * c / len(pool) for lv, c in count.items()}
    q = {lv: int(v) for lv, v in raw.items()}
    for lv in sorted(raw, key=lambda k: raw[k] - q[k], reverse=True)[: n - sum(q.values())]:
        q[lv] += 1
    return q


def split_bench(bench: Path, rng: random.Random) -> dict[str, list[dict]]:
    chosen = {"dev": [], "val": []}
    for kind, sub in (("hos", "hos_bench"), ("hps", "hps_bench")):
        groups = defaultdict(list)  # panorama content hash -> its scenes' episodes
        for scene in sorted((bench / sub).iterdir()):
            pano = panorama_of(scene) if scene.is_dir() and (scene / "annotation.json").exists() else None
            if pano is None:
                continue
            for ep in episodes_of(scene, f"{kind}-", rng):
                if ep["kind"] == kind:
                    groups[md5(pano)].append({**ep, "scene": f"{kind}-{scene.name}", "_pano": pano})
        pool = [ep for eps in groups.values() for ep in eps]
        need = {split: quotas(pool, SIZES[split][kind]) for split in ("val", "dev")}
        keys = sorted(groups)
        rng.shuffle(keys)
        for key in keys:  # one episode per panorama, to whichever split still needs its level
            eps = groups[key][:]
            rng.shuffle(eps)
            placed = False
            for ep in eps:
                for split in ("val", "dev"):
                    if need[split].get(ep["level"], 0) > 0:
                        need[split][ep["level"]] -= 1
                        chosen[split].append(ep)
                        placed = True
                        break
                if placed:
                    break
            if all(v == 0 for s in need.values() for v in s.values()):
                break
        short = {s: {lv: v for lv, v in q.items() if v} for s, q in need.items() if any(q.values())}
        if short:
            raise SystemExit(f"{kind}: not enough scenes for the quotas, missing {short}")
    return chosen


def split_wild(wild: Path, rng: random.Random) -> list[dict]:
    out, missing = [], []
    for scene in sorted(p for p in wild.iterdir() if p.is_dir()):
        if not (scene / "annotation.json").exists():
            continue
        pano = panorama_of(scene)
        if pano is None:
            missing.append(scene.name)
            continue
        out += [{**ep, "scene": f"wild-{scene.name}", "_pano": pano}
                for ep in episodes_of(scene, "wild-", rng)]
    if missing:
        raise SystemExit(f"in-the-wild scenes without a panorama: {missing}")
    return out


def write_split(out: Path, name: str, episodes: list[dict]) -> None:
    d = out / name
    if d.exists():
        shutil.rmtree(d)
    (d / "panoramas").mkdir(parents=True)
    episodes = sorted(episodes, key=lambda e: e["id"])
    for ep in episodes:
        dst = d / "panoramas" / (ep["scene"] + ep["_pano"].suffix.lower())
        if not dst.exists():
            shutil.copy2(ep["_pano"], dst)
        ep["panorama"] = dst.name
    with open(d / "episodes.jsonl", "w") as f:
        for ep in episodes:
            f.write(json.dumps({k: ep[k] for k in ("id", "kind", "task", "start_yaw", "scene",
                                                   "panorama")}) + "\n")
    with open(d / "labels.jsonl", "w") as f:
        for ep in episodes:
            f.write(json.dumps({k: ep[k] for k in ("id", "kind", "level", "yaw", "pitch")}) + "\n")
    kinds = Counter(e["kind"] for e in episodes)
    levels = Counter((e["kind"], e["level"]) for e in episodes)
    print(f"{name}: {len(episodes)} episodes {dict(kinds)}, levels {dict(sorted(levels.items()))}, "
          f"{len({e['scene'] for e in episodes})} scenes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--wild", required=True)
    ap.add_argument("--only", choices=["dev", "val", "test"], action="append",
                    help="rewrite only these splits (the others keep their files)")
    args = ap.parse_args()
    rng = random.Random(SEED)
    bench = split_bench(Path(args.bench), rng)
    test = split_wild(Path(args.wild), rng)
    out = Path(args.out)
    for name, eps in (("dev", bench["dev"]), ("val", bench["val"]), ("test", test)):
        if not args.only or name in args.only:
            write_split(out, name, eps)
    (out / "manifest.json").write_text(json.dumps({
        "seed": SEED, "sizes": SIZES, "sources": {"bench": str(Path(args.bench).name),
                                                  "wild": str(Path(args.wild).name)}}, indent=1))


if __name__ == "__main__":
    main()
