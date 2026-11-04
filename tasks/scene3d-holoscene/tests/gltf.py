"""Read a glTF 2.0 binary (.glb) into world-space triangles with their base-color appearance.

numpy and Pillow only. What the evaluator reads of a scene:

- every node of the default scene (every root node when there is none), with its matrix or its
  translation, rotation and scale, through the whole hierarchy;
- mesh primitives in TRIANGLES, TRIANGLE_STRIP and TRIANGLE_FAN mode, indexed or not (points and
  lines are skipped); any accessor component type, normalized integers, interleaved buffer views;
- POSITION, TEXCOORD_n and COLOR_0 (linear RGB, as glTF defines it);
- materials: pbrMetallicRoughness.baseColorFactor and baseColorTexture (its texCoord, the
  sampler's wrap modes, KHR_texture_transform) and doubleSided; KHR_materials_unlit reads the same;
- images embedded in the .glb (bufferView) or given as data: URIs, in a format Pillow decodes
  (PNG, JPEG, WebP; EXT_texture_webp too).

Ignored: lighting and every other material property (emission, metal and roughness, normal and
occlusion maps, alpha: every surface is opaque), cameras, lights, skins, morph targets and
animations. A scene that needs external files, Draco or meshopt compression, sparse accessors or
KTX2 textures, or that exceeds MAX_TRIANGLES or MAX_TEXTURE_PIXELS, does not load (GLBError).
"""

from __future__ import annotations

import base64
import io
import json
import math
import struct
from dataclasses import dataclass

import numpy as np
from PIL import Image

MAX_TRIANGLES = 4_000_000
MAX_TEXTURE_PIXELS = 256 * 1024 * 1024      # decoded, all textures of a scene together
REPEAT, CLAMP, MIRROR = 10497, 33071, 33648  # sampler wrap modes

_COMPONENTS = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
_UNSUPPORTED = {"KHR_draco_mesh_compression": "Draco compression",
                "EXT_meshopt_compression": "meshopt compression",
                "KHR_meshopt_compression": "meshopt compression",
                "KHR_texture_basisu": "KTX2 textures"}


class GLBError(ValueError):
    """The file is not a scene the evaluator can render."""


@dataclass
class Material:
    factor: np.ndarray           # (3,) linear RGB (baseColorFactor without alpha)
    image: int | None            # index into Scene.images
    wrap: tuple[int, int]        # wrapS, wrapT
    double_sided: bool


@dataclass
class Primitive:
    positions: np.ndarray        # (n, 3) float64, world space
    faces: np.ndarray            # (m, 3) int64, counter-clockwise when seen from the front
    uv: np.ndarray | None        # (n, 2) float32, the coordinates the material's texture uses
    color: np.ndarray | None     # (n, 3) float32, linear RGB
    material: int                # index into Scene.materials
    node: int
    name: str                    # the node's name


@dataclass
class Scene:
    primitives: list[Primitive]
    materials: list[Material]
    images: list[np.ndarray | None]   # (h, w, 3) uint8 sRGB; None for images no material uses
    meshes: int                       # nodes with at least one triangle
    triangles: int


def _chunks(data: bytes) -> tuple[dict, bytes | None]:
    if len(data) < 20:
        raise GLBError("not a .glb file (too short)")
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise GLBError("not a .glb file (no glTF header)")
    if version != 2:
        raise GLBError(f"glTF version {version}, expected 2")
    pos, doc, binary = 12, None, None
    end = min(length, len(data))
    while pos + 8 <= end:
        size, kind = struct.unpack_from("<I4s", data, pos)
        chunk = data[pos + 8:pos + 8 + size]
        pos += 8 + size
        if kind == b"JSON" and doc is None:
            try:
                doc = json.loads(chunk.decode("utf-8").rstrip(" \t\r\n\x00"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise GLBError(f"the JSON chunk does not parse: {exc}") from None
        elif kind == b"BIN\x00" and binary is None:
            binary = chunk
    if not isinstance(doc, dict):
        raise GLBError("no JSON chunk")
    return doc, binary


def _data_uri(uri: str, what: str) -> bytes:
    if not uri.startswith("data:") or "," not in uri:
        raise GLBError(f"{what} is an external file ({uri[:60]}): embed everything in the .glb")
    head, body = uri.split(",", 1)
    if not head.endswith(";base64"):
        raise GLBError(f"{what}: only base64 data URIs are supported")
    return base64.b64decode(body)


class _Reader:
    def __init__(self, doc: dict, binary: bytes | None):
        self.doc = doc
        self.buffers = []
        for i, buf in enumerate(doc.get("buffers", [])):
            if "uri" in buf:
                self.buffers.append(_data_uri(buf["uri"], f"buffer {i}"))
            elif i == 0 and binary is not None:
                self.buffers.append(binary)
            else:
                raise GLBError(f"buffer {i} has no data")

    def view(self, index: int) -> tuple[bytes, int, int, int | None]:
        bv = self.doc["bufferViews"][index]
        buf = self.buffers[bv["buffer"]]
        offset, size = bv.get("byteOffset", 0), bv["byteLength"]
        if offset + size > len(buf):
            raise GLBError(f"bufferView {index} runs past the end of its buffer")
        return buf, offset, size, bv.get("byteStride")

    def accessor(self, index: int) -> np.ndarray:
        """(count, width) array; normalized integers become floats in [0, 1] or [-1, 1]."""
        acc = self.doc["accessors"][index]
        if "sparse" in acc:
            raise GLBError("sparse accessors are not supported")
        dtype = np.dtype(_COMPONENTS[acc["componentType"]])
        count, width = acc["count"], _WIDTH[acc["type"]]
        if "bufferView" not in acc:
            out = np.zeros((count, width), dtype)
        else:
            buf, offset, size, stride = self.view(acc["bufferView"])
            offset += acc.get("byteOffset", 0)
            stride = stride or dtype.itemsize * width
            need = stride * (count - 1) + dtype.itemsize * width if count else 0
            if count and offset + need > len(buf):
                raise GLBError(f"accessor {index} runs past the end of its buffer")
            out = np.ndarray((count, width), dtype, buffer=buf, offset=offset,
                             strides=(stride, dtype.itemsize)).copy()
        if acc.get("normalized") and dtype.kind in "iu":
            top = float(np.iinfo(dtype).max)
            out = np.maximum(out.astype(np.float32) / top, -1.0)
        return out

    def image(self, index: int) -> Image.Image:
        img = self.doc["images"][index]
        if "bufferView" in img:
            buf, offset, size, _ = self.view(img["bufferView"])
            raw = buf[offset:offset + size]
        elif "uri" in img:
            raw = _data_uri(img["uri"], f"image {index}")
        else:
            raise GLBError(f"image {index} has no data")
        try:
            return Image.open(io.BytesIO(raw))
        except Exception as exc:  # noqa: BLE001 - any decoder error
            raise GLBError(f"image {index} does not decode: {exc}") from None


def _local_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], np.float64).reshape(4, 4).T  # column-major
    x, y, z, w = np.asarray(node.get("rotation", [0, 0, 0, 1]), np.float64)
    norm = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    m = np.eye(4)
    m[:3, :3] = rot * np.asarray(node.get("scale", [1, 1, 1]), np.float64)[None, :]
    m[:3, 3] = node.get("translation", [0, 0, 0])
    return m


def _texture_transform(ext: dict) -> np.ndarray:
    ox, oy = ext.get("offset", [0.0, 0.0])
    sx, sy = ext.get("scale", [1.0, 1.0])
    r = ext.get("rotation", 0.0)
    t = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], np.float64)
    rot = np.array([[math.cos(r), math.sin(r), 0], [-math.sin(r), math.cos(r), 0], [0, 0, 1]])
    s = np.diag([sx, sy, 1.0])
    return t @ rot @ s


def _faces(index: np.ndarray, mode: int) -> np.ndarray | None:
    n = len(index)
    if mode == 4:
        return index[: n // 3 * 3].reshape(-1, 3)
    if mode == 5 and n >= 3:  # strip: every other triangle reversed to keep the winding
        i = np.arange(n - 2)
        odd = i % 2 == 1
        a, b, c = index[i], index[i + 1], index[i + 2]
        return np.stack([np.where(odd, b, a), np.where(odd, a, b), c], 1)
    if mode == 6 and n >= 3:  # fan
        i = np.arange(1, n - 1)
        return np.stack([index[i], index[i + 1], np.full(len(i), index[0])], 1)
    return None


def load(path: str) -> Scene:
    with open(path, "rb") as f:
        doc, binary = _chunks(f.read())
    try:
        return _load(doc, binary)
    except GLBError:
        raise
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise GLBError(f"malformed glTF ({type(exc).__name__}: {exc})") from None


def _load(doc: dict, binary: bytes | None) -> Scene:
    for name in doc.get("extensionsRequired", []):
        if name in _UNSUPPORTED:
            raise GLBError(f"{_UNSUPPORTED[name]} ({name}) is not supported")
    reader = _Reader(doc, binary)

    # materials; the last one is glTF's default material
    materials, texcoord, uv_transform = [], [], []
    for m in doc.get("materials", []):
        pbr = m.get("pbrMetallicRoughness", {})
        factor = np.asarray(pbr.get("baseColorFactor", [1, 1, 1, 1]), np.float32)[:3]
        image, wrap, tc, transform = None, (REPEAT, REPEAT), 0, None
        tex = pbr.get("baseColorTexture")
        if tex is not None:
            texture = doc["textures"][tex["index"]]
            ext = texture.get("extensions", {})
            image = texture.get("source")
            if "EXT_texture_webp" in ext:
                image = ext["EXT_texture_webp"].get("source", image)
            if image is None and "KHR_texture_basisu" in ext:
                raise GLBError("KTX2 textures (KHR_texture_basisu) are not supported")
            sampler = doc["samplers"][texture["sampler"]] if "sampler" in texture else {}
            wrap = (sampler.get("wrapS", REPEAT), sampler.get("wrapT", REPEAT))
            tc = tex.get("texCoord", 0)
            tt = tex.get("extensions", {}).get("KHR_texture_transform")
            if tt is not None:
                transform = _texture_transform(tt)
                tc = tt.get("texCoord", tc)
        materials.append(Material(factor, image, wrap, bool(m.get("doubleSided", False))))
        texcoord.append(tc)
        uv_transform.append(transform)
    default = len(materials)
    materials.append(Material(np.ones(3, np.float32), None, (REPEAT, REPEAT), False))
    texcoord.append(0)
    uv_transform.append(None)

    # nodes of the default scene, depth first
    nodes = doc.get("nodes", [])
    scenes = doc.get("scenes", [])
    if scenes:
        roots = scenes[doc.get("scene", 0)].get("nodes", [])
    else:
        children = {c for n in nodes for c in n.get("children", [])}
        roots = [i for i in range(len(nodes)) if i not in children]
    stack = [(i, np.eye(4)) for i in reversed(roots)]
    seen: set[int] = set()
    prims: list[Primitive] = []
    mesh_nodes, triangles = set(), 0
    while stack:
        index, parent = stack.pop()
        if index in seen:
            raise GLBError(f"node {index} is reached twice (glTF nodes form a tree)")
        seen.add(index)
        node = nodes[index]
        world = parent @ _local_matrix(node)
        stack.extend((c, world) for c in reversed(node.get("children", [])))
        if "mesh" not in node:
            continue
        flip = np.linalg.det(world[:3, :3]) < 0
        for prim in doc["meshes"][node["mesh"]]["primitives"]:
            if "extensions" in prim:
                for name in prim["extensions"]:
                    if name in _UNSUPPORTED:
                        raise GLBError(f"{_UNSUPPORTED[name]} ({name}) is not supported")
            attrs = prim.get("attributes", {})
            if "POSITION" not in attrs:
                continue
            pos = reader.accessor(attrs["POSITION"])[:, :3].astype(np.float64)
            if "indices" in prim:
                index_ = reader.accessor(prim["indices"]).reshape(-1).astype(np.int64)
            else:
                index_ = np.arange(len(pos), dtype=np.int64)
            faces = _faces(index_, prim.get("mode", 4))
            if faces is None or len(faces) == 0:
                continue
            if faces.min() < 0 or faces.max() >= len(pos):
                raise GLBError(f"mesh {node['mesh']}: an index points past its vertices")
            if flip:
                faces = faces[:, [0, 2, 1]]
            triangles += len(faces)
            if triangles > MAX_TRIANGLES:
                raise GLBError(f"more than {MAX_TRIANGLES} triangles")
            mat = prim.get("material", default)
            uv = None
            if materials[mat].image is not None and f"TEXCOORD_{texcoord[mat]}" in attrs:
                uv = reader.accessor(attrs[f"TEXCOORD_{texcoord[mat]}"])[:, :2].astype(np.float64)
                if uv_transform[mat] is not None:
                    uv = uv @ uv_transform[mat][:2, :2].T + uv_transform[mat][:2, 2]
                uv = uv.astype(np.float32)
            color = None
            if "COLOR_0" in attrs:
                color = reader.accessor(attrs["COLOR_0"])[:, :3].astype(np.float32)
                if len(color) != len(pos):
                    raise GLBError(f"mesh {node['mesh']}: COLOR_0 and POSITION differ in length")
            if uv is not None and len(uv) != len(pos):
                raise GLBError(f"mesh {node['mesh']}: TEXCOORD and POSITION differ in length")
            world_pos = pos @ world[:3, :3].T + world[:3, 3]
            prims.append(Primitive(world_pos, faces, uv, color, mat, index,
                                   str(node.get("name") or f"node {index}")))
            mesh_nodes.add(index)

    # decode the images that a primitive's material uses
    used = {materials[p.material].image for p in prims
            if materials[p.material].image is not None and p.uv is not None}
    images: list[np.ndarray | None] = [None] * len(doc.get("images", []))
    pixels = 0
    Image.MAX_IMAGE_PIXELS = None  # the cap below applies instead
    for i in sorted(used):
        img = reader.image(i)
        pixels += img.size[0] * img.size[1]
        if pixels > MAX_TEXTURE_PIXELS:
            raise GLBError(f"textures larger than {MAX_TEXTURE_PIXELS} pixels in total")
        try:
            if img.mode in ("RGBA", "LA", "PA", "P"):
                img = img.convert("RGBA")
            images[i] = np.asarray(img.convert("RGB"), np.uint8)
        except Exception as exc:  # noqa: BLE001 - any decoder error
            raise GLBError(f"image {i} does not decode: {exc}") from None
    return Scene(prims, materials, images, len(mesh_nodes), triangles)
