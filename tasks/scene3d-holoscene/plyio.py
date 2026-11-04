"""Read a binary little-endian PLY triangle mesh with numpy: vertices (x, y, z) and faces."""

from __future__ import annotations

import numpy as np

_TYPES = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2", "int": "i4", "uint": "u4",
          "float": "f4", "double": "f8", "int8": "i1", "uint8": "u1", "int16": "i2",
          "uint16": "u2", "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8"}


def read_ply(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (n, 3) float64 and triangles (m, 3) int64; polygons are split into fans."""
    with open(path, "rb") as f:
        data = f.read()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii", "replace").splitlines()
    if not any(line.startswith("format binary_little_endian") for line in header):
        raise ValueError(f"{path}: only binary little-endian PLY files are read")
    elements = []
    for line in header:
        p = line.split()
        if not p:
            continue
        if p[0] == "element":
            elements.append((p[1], int(p[2]), []))
        elif p[0] == "property":
            if p[1] == "list":
                elements[-1][2].append((p[4], _TYPES[p[2]], _TYPES[p[3]]))
            else:
                elements[-1][2].append((p[2], _TYPES[p[1]]))
    off, verts, faces = end, None, None
    for name, n, props in elements:
        if all(len(pr) == 2 for pr in props):
            dt = np.dtype([(pr[0], "<" + pr[1]) for pr in props])
            arr = np.frombuffer(data, dt, count=n, offset=off)
            off += n * dt.itemsize
            if name == "vertex":
                verts = np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float64)
            continue
        if len(props) != 1:
            raise ValueError(f"{path}: element {name} mixes lists and scalars")
        _, count_t, index_t = props[0]
        cs, isz = np.dtype(count_t).itemsize, np.dtype(index_t).itemsize
        first = int(np.frombuffer(data, "<" + count_t, count=1, offset=off)[0]) if n else 3
        dt = np.dtype([("c", "<" + count_t), ("i", "<" + index_t, (first,))])
        if n and (np.frombuffer(data, dt, count=n, offset=off)["c"] == first).all():
            arr = np.frombuffer(data, dt, count=n, offset=off)
            off += n * dt.itemsize
            polys = [arr["i"].astype(np.int64)]
        else:  # mixed polygon sizes: walk them
            polys, cur = [], off
            for _ in range(n):
                c = int(np.frombuffer(data, "<" + count_t, count=1, offset=cur)[0])
                polys.append(np.frombuffer(data, "<" + index_t, count=c, offset=cur + cs).astype(np.int64)[None])
                cur += cs + c * isz
            off = cur
        if name == "face":
            tris = []
            for p in polys:
                k = p.shape[1]
                for j in range(1, k - 1):
                    tris.append(np.stack([p[:, 0], p[:, j], p[:, j + 1]], 1))
            faces = np.concatenate(tris) if tris else np.zeros((0, 3), np.int64)
    if verts is None or faces is None:
        raise ValueError(f"{path}: no vertex or face element")
    return verts, faces
