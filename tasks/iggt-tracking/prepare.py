"""Build the iggt-tracking splits under --out.

    val   the 8 ScanNet++ clips of the IGGT 3D tracking benchmark
    test  the 10 ScanNet clips of the IGGT 3D tracking benchmark
    dev   100 clips built from ScanNet++ v2 training scenes (scenes of val excluded)

Every split has inputs/<clip>/frames/000.png ... and labels/<clip>/gt.npz
(inst_ids (S, 448, 592) int32, 0 = background; num_instances; exhaustive).

Sources (not part of the task package):
    --bench   directory with the exported benchmark clips clip_00000 ... clip_00017
    --train   ScanNet++ v2 processed scenes (<scene>/images/*.jpg, <scene>/refined_ins_ids/*.jpg.npy)
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

H, W, S_DEV, N_DEV = 448, 592, 8, 100
SCANNET_CLIPS = [f"clip_{i:05d}" for i in range(0, 10)]
SCANNETPP_CLIPS = [f"clip_{i:05d}" for i in range(10, 18)]
VAL_SCENES = {"0a76e06478", "0a7cc12c0e", "0b031f3119", "0cf2e9402d", "e050c15a8d", "e898c76c1f",
              "e9e16b6043", "f6659a3107", "ebc200e928"}


def copy_bench(bench: Path, clips: list[str], out: Path) -> None:
    for clip in clips:
        src = bench / clip
        dst_in, dst_lab = out / "inputs" / clip / "frames", out / "labels" / clip
        dst_in.mkdir(parents=True, exist_ok=True)
        dst_lab.mkdir(parents=True, exist_ok=True)
        for f in sorted((src / "frames").glob("*.png")):
            shutil.copy(f, dst_in / f.name)
        shutil.copy(src / "gt.npz", dst_lab / "gt.npz")


def pick_instances(ids: np.ndarray, rng: random.Random) -> list[int]:
    """Object-like instances: not huge (walls, floors), not tiny, visible in >= 2 frames."""
    S = ids.shape[0]
    out = []
    for i in np.unique(ids):
        if i <= 0:
            continue
        m = ids == i
        per_frame = m.reshape(S, -1).sum(1) / (H * W)
        frames = int((per_frame * H * W >= 50).sum())
        if frames >= 2 and per_frame.max() <= 0.25 and m.sum() / ids.size >= 0.001:
            out.append(int(i))
    rng.shuffle(out)
    return out[:8]


def build_dev_clip(scene: Path, rng: random.Random) -> tuple[list[Image.Image], np.ndarray, list[str]] | None:
    names = sorted(p.name[:-len(".npy")] for p in (scene / "refined_ins_ids").glob("*.jpg.npy"))
    names = [n for n in names if (scene / "images" / n).exists()]
    if len(names) < 3 * S_DEV:
        return None
    for _ in range(6):
        span = max(S_DEV * 3, int(len(names) * rng.uniform(0.3, 0.8)))
        start = rng.randrange(0, len(names) - span + 1)
        chosen = sorted(rng.sample(names[start:start + span], S_DEV))
        ids = np.stack([np.array(Image.fromarray(np.load(scene / "refined_ins_ids" / f"{n}.npy").astype(np.int32))
                                 .resize((W, H), Image.NEAREST)) for n in chosen])
        keep = pick_instances(ids, rng)
        if len(keep) >= 4:
            gt = np.zeros_like(ids, dtype=np.int32)
            for new, old in enumerate(keep, start=1):
                gt[ids == old] = new
            frames = [Image.open(scene / "images" / n).convert("RGB").resize((W, H), Image.BILINEAR)
                      for n in chosen]
            return frames, gt, chosen
    return None


def build_dev(train: Path, out: Path, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    scenes = sorted(p.name for p in train.iterdir() if p.is_dir() and p.name not in VAL_SCENES)
    rng.shuffle(scenes)
    manifest = []
    for scene in scenes:
        if len(manifest) == N_DEV:
            break
        try:
            built = build_dev_clip(train / scene, rng)
        except Exception as exc:  # unreadable scene: skip it, never abort the build
            print(f"skip {scene}: {exc}")
            continue
        if built is None:
            continue
        frames, gt, chosen = built
        clip = f"dev_{len(manifest):03d}"
        fdir = out / "inputs" / clip / "frames"
        fdir.mkdir(parents=True, exist_ok=True)
        for k, img in enumerate(frames):
            img.save(fdir / f"{k:03d}.png")
        (out / "labels" / clip).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "labels" / clip / "gt.npz", inst_ids=gt,
                            num_instances=np.int64(gt.max()), exhaustive=np.bool_(True))
        manifest.append({"clip": clip, "scene": scene, "frames": chosen,
                         "num_instances": int(gt.max())})
    if len(manifest) < N_DEV:
        raise SystemExit(f"only {len(manifest)} usable training scenes")
    assert not ({m["scene"] for m in manifest} & VAL_SCENES)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--train", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    copy_bench(Path(args.bench), SCANNETPP_CLIPS, out / "val")
    copy_bench(Path(args.bench), SCANNET_CLIPS, out / "test")
    manifest = build_dev(Path(args.train), out / "dev")
    for p in out.rglob("*"):
        if p.is_file():
            p.chmod(0o444)
    print(f"val {len(SCANNETPP_CLIPS)} clips, test {len(SCANNET_CLIPS)} clips, dev {len(manifest)} clips -> {out}")


if __name__ == "__main__":
    main()
