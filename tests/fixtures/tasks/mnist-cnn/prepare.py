"""Build the mnist-cnn splits under --out (downloads MNIST once; needs internet and numpy).

dev/train_images.npy, dev/train_labels.npy   50,000 training images (uint8 [N, 28, 28])
dev/images.npy, dev/labels.npy               5,000 dev query images (from the official train set)
val/images.npy, val/labels.npy               5,000 hidden query images (official train set)
test/images.npy, test/labels.npy             the official 10,000 test images
"""

import argparse
import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {"train_x": "train-images-idx3-ubyte.gz", "train_y": "train-labels-idx1-ubyte.gz",
         "test_x": "t10k-images-idx3-ubyte.gz", "test_y": "t10k-labels-idx1-ubyte.gz"}


def fetch(name: str, cache: Path) -> np.ndarray:
    path = cache / name
    if not path.exists():
        urllib.request.urlretrieve(MIRROR + name, path)
    raw = gzip.decompress(path.read_bytes())
    magic = struct.unpack(">I", raw[:4])[0]
    if magic == 2051:
        n, rows, cols = struct.unpack(">III", raw[4:16])
        return np.frombuffer(raw[16:], dtype=np.uint8).reshape(n, rows, cols)
    n = struct.unpack(">I", raw[4:8])[0]
    return np.frombuffer(raw[8:], dtype=np.uint8).reshape(n)


def save(d: Path, prefix: str, x, y) -> None:
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{prefix}images.npy", x)
    np.save(d / f"{prefix}labels.npy", y.astype(np.int64))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    cache = out / "_download"
    cache.mkdir(parents=True, exist_ok=True)
    arrays = {k: fetch(v, cache) for k, v in FILES.items()}
    order = np.random.RandomState(0).permutation(len(arrays["train_y"]))
    x, y = arrays["train_x"][order], arrays["train_y"][order]
    save(out / "dev", "train_", x[:50000], y[:50000])
    save(out / "dev", "", x[50000:55000], y[50000:55000])
    save(out / "val", "", x[55000:], y[55000:])
    save(out / "test", "", arrays["test_x"], arrays["test_y"])
    print(f"wrote mnist-cnn splits to {out}")


if __name__ == "__main__":
    main()
