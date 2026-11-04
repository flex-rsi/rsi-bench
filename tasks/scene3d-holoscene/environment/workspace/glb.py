"""Write meshes to a glTF 2.0 binary (.glb) with numpy and Pillow; use it or any other exporter.

Each mesh becomes one node. A mesh has `positions` (n, 3) and `faces` (m, 3), counter-clockwise
when seen from the side that should be visible, and optionally:

    name          node name
    uv            (n, 2) texture coordinates, (0, 0) at the top-left of the texture
    texture       (h, w, 3) uint8 sRGB image (or a PIL image), stored as PNG
    colors        (n, 3) vertex colors, linear RGB in [0, 1]
    base_color    (3,) or (4,) baseColorFactor, linear RGB(A); default white
    double_sided  render both sides (default False: faces seen from behind are culled)
"""

from __future__ import annotations

import io
import json
import struct

import numpy as np
from PIL import Image


def srgb_to_linear(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def write_glb(path: str, meshes: list[dict]) -> None:
    blob = bytearray()
    views, accessors, images, textures, materials, gl_meshes, nodes = [], [], [], [], [], [], []

    def add_view(data: bytes, target: int | None = None) -> int:
        while len(blob) % 4:
            blob.append(0)
        view = {"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        blob.extend(data)
        views.append(view)
        return len(views) - 1

    def add_accessor(arr: np.ndarray, kind: str, component: int, target: int,
                     bounds: bool = False) -> int:
        acc = {"bufferView": add_view(arr.tobytes(), target), "componentType": component,
               "count": int(arr.shape[0]), "type": kind}
        if bounds:
            acc["min"] = arr.min(0).tolist()
            acc["max"] = arr.max(0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    for i, m in enumerate(meshes):
        pos = np.ascontiguousarray(m["positions"], np.float32)
        faces = np.ascontiguousarray(m["faces"], np.uint32).reshape(-1)
        attrs = {"POSITION": add_accessor(pos, "VEC3", 5126, 34962, bounds=True)}
        material = {"pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 1.0},
                    "doubleSided": bool(m.get("double_sided", False))}
        color = m.get("base_color")
        if color is not None:
            color = [float(c) for c in color] + [1.0] * (4 - len(color))
            material["pbrMetallicRoughness"]["baseColorFactor"] = color
        if m.get("colors") is not None:
            attrs["COLOR_0"] = add_accessor(np.ascontiguousarray(m["colors"], np.float32),
                                            "VEC3", 5126, 34962)
        if m.get("texture") is not None and m.get("uv") is not None:
            attrs["TEXCOORD_0"] = add_accessor(np.ascontiguousarray(m["uv"], np.float32),
                                               "VEC2", 5126, 34962)
            img = m["texture"]
            img = img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img, np.uint8))
            png = io.BytesIO()
            img.convert("RGB").save(png, format="PNG")
            images.append({"bufferView": add_view(png.getvalue()), "mimeType": "image/png"})
            textures.append({"source": len(images) - 1, "sampler": 0})
            material["pbrMetallicRoughness"]["baseColorTexture"] = {"index": len(textures) - 1}
        materials.append(material)
        index = add_accessor(faces, "SCALAR", 5125, 34963)
        gl_meshes.append({"primitives": [{"attributes": attrs, "indices": index,
                                          "material": len(materials) - 1}]})
        nodes.append({"mesh": i, "name": str(m.get("name", f"mesh_{i}"))})

    doc = {"asset": {"version": "2.0", "generator": "scene3d-holoscene glb.py"},
           "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes,
           "meshes": gl_meshes, "materials": materials, "accessors": accessors,
           "bufferViews": views, "buffers": [{"byteLength": len(blob)}]}
    if images:
        doc.update(images=images, textures=textures,
                   samplers=[{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}])
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    while len(blob) % 4:
        blob.append(0)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(js) + 8 + len(blob)))
        f.write(struct.pack("<I4s", len(js), b"JSON") + js)
        f.write(struct.pack("<I4s", len(blob), b"BIN\x00") + bytes(blob))
