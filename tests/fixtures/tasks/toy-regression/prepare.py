"""Build the toy-regression splits under --out (dev/val/test).

dev/train_inputs.json + dev/train_labels.json   training set (visible to the agent)
<split>/inputs.json + <split>/labels.json       query points and their targets
"""

import argparse
import json
import math
import random
from pathlib import Path


def target(x: float) -> float:
    return 0.5 * x ** 3 - x + 0.3 * math.sin(3 * x)


def sample(n: int, seed: int) -> tuple[list[float], list[float]]:
    rng = random.Random(seed)
    xs = [rng.uniform(-2.0, 2.0) for _ in range(n)]
    ys = [target(x) + rng.gauss(0.0, 0.05) for x in xs]
    return xs, ys


def write(d: Path, prefix: str, xs, ys) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{prefix}inputs.json").write_text(json.dumps({"x": xs}))
    (d / f"{prefix}labels.json").write_text(json.dumps({"y": ys}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    write(out / "dev", "train_", *sample(200, 11))
    write(out / "dev", "", *sample(100, 12))
    write(out / "val", "", *sample(300, 21))
    write(out / "test", "", *sample(500, 31))
    print(f"wrote toy-regression splits to {out}")


if __name__ == "__main__":
    main()
