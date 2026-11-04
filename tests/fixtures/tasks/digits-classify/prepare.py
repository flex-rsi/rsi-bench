"""Build the digits-classify splits (scikit-learn's 8x8 digits, bundled, no download).

dev/train_inputs.json + dev/train_labels.json   900 training images (visible to the agent)
dev/inputs.json + dev/labels.json               200 dev query images
val/, test/                                     300 / 397 hidden query images
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits


def write(d: Path, prefix: str, x, y) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{prefix}inputs.json").write_text(json.dumps({"x": x.tolist()}))
    (d / f"{prefix}labels.json").write_text(json.dumps({"y": y.tolist()}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    data = load_digits()
    order = np.random.RandomState(0).permutation(len(data.target))
    x, y = data.data[order].astype(float), data.target[order].astype(int)
    cuts = [900, 1100, 1400]
    write(out / "dev", "train_", x[:cuts[0]], y[:cuts[0]])
    write(out / "dev", "", x[cuts[0]:cuts[1]], y[cuts[0]:cuts[1]])
    write(out / "val", "", x[cuts[1]:cuts[2]], y[cuts[1]:cuts[2]])
    write(out / "test", "", x[cuts[2]:], y[cuts[2]:])
    print(f"wrote digits-classify splits to {out}")


if __name__ == "__main__":
    main()
