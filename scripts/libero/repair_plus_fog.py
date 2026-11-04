#!/usr/bin/env python3
"""Repair pinned LIBERO-plus fog for the official 360px X-VLA renderer."""

import argparse
from pathlib import Path


ORIGINAL = "    x += c[0] * plasma_fractal(wibbledecay=c[1])[:height_x, :weight_x][..., np.newaxis]"
REPAIRED = (
    "    fog_map_size = max(256, 1 << (max(height_x, weight_x) - 1).bit_length())\n"
    "    x += c[0] * plasma_fractal(mapsize=fog_map_size, wibbledecay=c[1])"
    "[:height_x, :weight_x][..., np.newaxis]"
)


def repair(plus_root: Path) -> bool:
    source = plus_root / "libero/libero/envs/env_wrapper.py"
    text = source.read_text()
    if REPAIRED in text:
        return False
    if text.count(ORIGINAL) != 1:
        raise ValueError(f"Pinned LIBERO-plus fog expression not found in {source}")
    source.write_text(text.replace(ORIGINAL, REPAIRED, 1))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plus-root", required=True, type=Path)
    args = parser.parse_args()
    changed = repair(args.plus_root)
    print("LIBERO-plus fog: " + ("repaired" if changed else "already repaired"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
