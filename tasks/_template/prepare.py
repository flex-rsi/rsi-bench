"""Build $RSIB_DATA_ROOT/<task-id>/{dev,val,test}. Called as: python prepare.py --out <dir>.

Keep labels for val and test out of anything the agent can read. The dev split may include
labels (the agent trains and debugs on it).
"""

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    for split in ("dev", "val", "test"):
        (out / split).mkdir(parents=True, exist_ok=True)
    raise SystemExit("fill in prepare.py")


if __name__ == "__main__":
    main()
