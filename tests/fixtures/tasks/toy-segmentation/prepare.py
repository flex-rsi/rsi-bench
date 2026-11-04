"""Build the toy-segmentation splits under --out (dev/val/test).

Each image is 24x24 grey values: one or two bright disks on a dark background, an illumination
gradient across the image, and Gaussian noise. The mask marks the disks (1) and background (0).
"""

import argparse
import json
import math
import random
from pathlib import Path

SIZE = 24


def make(rng: random.Random):
    disks = [(rng.uniform(5, SIZE - 5), rng.uniform(5, SIZE - 5), rng.uniform(2.5, 5.5))
             for _ in range(rng.choice((1, 2)))]
    angle = rng.uniform(0, 2 * math.pi)
    strength = rng.uniform(0.2, 0.7)
    image, mask = [], []
    for r in range(SIZE):
        row, mrow = [], []
        for c in range(SIZE):
            fg = any((r - y) ** 2 + (c - x) ** 2 <= rad ** 2 for y, x, rad in disks)
            light = strength * (0.5 + 0.5 * ((c - SIZE / 2) * math.cos(angle)
                                             + (r - SIZE / 2) * math.sin(angle)) / (SIZE / 2))
            value = (1.0 if fg else 0.0) + light + rng.gauss(0.0, 0.3)
            row.append(round(value, 4))
            mrow.append(1 if fg else 0)
        image.append(row)
        mask.append(mrow)
    return image, mask


def split(n: int, seed: int):
    rng = random.Random(seed)
    pairs = [make(rng) for _ in range(n)]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def write(d: Path, prefix: str, images, masks) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{prefix}inputs.json").write_text(json.dumps({"images": images}))
    (d / f"{prefix}labels.json").write_text(json.dumps({"masks": masks}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    write(out / "dev", "train_", *split(30, 101))
    write(out / "dev", "", *split(20, 102))
    write(out / "val", "", *split(40, 201))
    write(out / "test", "", *split(60, 301))
    print(f"wrote toy-segmentation splits to {out}")


if __name__ == "__main__":
    main()
