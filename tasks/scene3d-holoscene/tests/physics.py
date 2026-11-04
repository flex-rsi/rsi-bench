"""Rigid-body stability of CoDimRecon (arXiv 2609.36024, Appendix C.2): a MuJoCo drop test of each
dynamic object, with everything else fixed.

Collision geometry:
- Background: first simplified to at most MAX_BACKGROUND_FACES faces (vertices welded on a grid
  of WELD_TOL, then fast-simplification; vertex clustering on a coarsening grid when that does not
  work), so that a dense or noisy surface does not become hundreds of thousands of slabs. Its
  triangles are then clustered by face-normal direction (greedily, around the largest remaining
  triangle, within SLAB_ANGLE_DEG), each cluster is bisected along its longest axis until every
  vertex lies within SLAB_PLANAR_TOL of the cluster's fitted plane, and each patch becomes a convex
  slab SLAB_THICKNESS thick behind the side its faces face, which keeps the room's hollow interior.
- Objects: decimated to at most MAX_FACES faces (fast-simplification) and decomposed into at most
  MAX_HULLS convex hulls by CoACD (threshold COACD_THRESHOLD, a search of COACD_MCTS_ITERATIONS
  iterations and depth COACD_MCTS_DEPTH, seed 0). When CoACD fails or takes longer than
  COACD_TIMEOUT_S, the object's convex hull is used instead.
The time limits (COACD_TIMEOUT_S, TEST_TIMEOUT_S) are a safety net several times longer than these
steps take on the simplified geometry, so that the results do not depend on the machine.
Each test runs in its own process: the tested object is a free body of uniform density DENSITY
(kg/m^3), the other objects and the background slabs within NEARBY meters of it are fixed, and an
infinite ground plane lies at the floor height. Simulation: SIM_SECONDS at a time step of DT,
gravity -9.81 m/s^2 along z, friction FRICTION. A background slab that initially penetrates the
object by more than PENETRATION_TOL is removed and the test restarted, at most MAX_RETRIES times.
The object has fallen when its local z axis tilts by more than FALL_TILT_DEG or it moves by more
than max(FALL_MIN_SHIFT, FALL_SHIFT_FRAC x its bounding-box diagonal).

MuJoCo, CoACD and fast-simplification come from the scorer's pylib directory (PHYSICS_PYLIB).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError

SIM_SECONDS = 3.0
DT = 0.002
DENSITY = 300.0
FRICTION = 1.0
FALL_TILT_DEG = 30.0
FALL_MIN_SHIFT = 0.05
FALL_SHIFT_FRAC = 0.2
SLAB_ANGLE_DEG = 20.0
SLAB_PLANAR_TOL = 0.03
SLAB_THICKNESS = 0.05
PENETRATION_TOL = 0.02
MAX_RETRIES = 3
MAX_FACES = 20_000
MAX_HULLS = 32
COACD_THRESHOLD = 0.05
COACD_MCTS_ITERATIONS = 60      # CoACD's defaults (150, 3) take 2.5 times as long
COACD_MCTS_DEPTH = 2
COACD_TIMEOUT_S = 900
TEST_TIMEOUT_S = 600
NEARBY = 0.5
MIN_HULL_VOLUME = 1e-8          # m^3; flatter hulls have no inertia and are left out
MAX_BACKGROUND_FACES = 100_000
WELD_TOL = 0.001                # m: the grid on which background vertices are welded
CLUSTER_CELL = 0.02             # m: first grid of the vertex-clustering fallback
MIN_AREA_KEPT = 0.5
SIMPLIFY_TIMEOUT_S = 300


# --------------------------------------------------------------------------- collision geometry
def _hull_points(points: np.ndarray) -> np.ndarray | None:
    """The vertices of the convex hull of points, or None when it has no volume."""
    points = np.asarray(points, np.float64)
    if len(points) < 4 or not np.isfinite(points).all():
        return None
    try:
        hull = ConvexHull(points)
    except (QhullError, ValueError):
        return None
    if hull.volume < MIN_HULL_VOLUME:
        return None
    return points[hull.vertices]


def _cluster(v: np.ndarray, f: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Vertex clustering: the vertices in one cell of a grid of size `cell` become their mean. Faces
    that collapse are dropped, and of faces with the same vertices in the same order one is kept."""
    _, inv, counts = np.unique(np.floor(v / cell).astype(np.int64), axis=0, return_inverse=True,
                               return_counts=True)
    inv = inv.reshape(-1)
    v2 = np.stack([np.bincount(inv, weights=v[:, i], minlength=len(counts)) for i in range(3)], 1)
    v2 /= counts[:, None]
    f2 = inv[f]
    f2 = f2[(f2[:, 0] != f2[:, 1]) & (f2[:, 1] != f2[:, 2]) & (f2[:, 2] != f2[:, 0])]
    if len(f2) == 0:
        return v2, f2
    roll = (np.argmin(f2, axis=1)[:, None] + np.arange(3)) % 3      # same orientation, lowest first
    _, first = np.unique(np.take_along_axis(f2, roll, 1), axis=0, return_index=True)
    return v2, f2[np.sort(first)]


def _simplify_worker(src: Path, dst: Path) -> None:
    import fast_simplification

    data = np.load(src)
    v, f = fast_simplification.simplify(data["vertices"].astype(np.float32),
                                        data["faces"].astype(np.int32),
                                        target_count=MAX_BACKGROUND_FACES)
    np.savez(dst, vertices=v, faces=f)


def _area(v: np.ndarray, f: np.ndarray) -> float:
    cross = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    return float(np.linalg.norm(cross, axis=1).sum() / 2)


def simplify_background(vertices: np.ndarray, faces: np.ndarray, python: str,
                        env: dict) -> tuple[np.ndarray, np.ndarray]:
    """The background with at most MAX_BACKGROUND_FACES faces. Fast-simplification's result is
    kept only if it keeps at least MIN_AREA_KEPT of the area (on unconnected triangles it reaches
    the face count by deleting them); otherwise the welded background is clustered on a grid of
    CLUSTER_CELL, grown by half until it fits."""
    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64)
    if len(f) <= MAX_BACKGROUND_FACES:
        return v, f
    v, f = _cluster(v, f, WELD_TOL)
    if len(f) <= MAX_BACKGROUND_FACES:
        return v, f
    with tempfile.TemporaryDirectory(prefix="simplify-") as tmp:
        src, dst = Path(tmp) / "background.npz", Path(tmp) / "simplified.npz"
        np.savez(src, vertices=v, faces=f)
        try:
            subprocess.run([python, __file__, "simplify", str(src), str(dst)], env=env,
                           timeout=SIMPLIFY_TIMEOUT_S, check=True, capture_output=True)
            with np.load(dst) as z:
                v2, f2 = z["vertices"].astype(np.float64), z["faces"].astype(np.int64)
            if 0 < len(f2) <= MAX_BACKGROUND_FACES and _area(v2, f2) >= MIN_AREA_KEPT * _area(v, f):
                return v2, f2
        except (subprocess.SubprocessError, OSError, ValueError):
            pass
    cell = CLUSTER_CELL
    while True:
        v2, f2 = _cluster(v, f, cell)
        if len(f2) <= MAX_BACKGROUND_FACES:
            return v2, f2
        cell *= 1.5


def background_slabs(vertices: np.ndarray, faces: np.ndarray) -> list[np.ndarray]:
    """Convex slabs (each an array of points) of the background's planar patches."""
    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64)
    if len(f) == 0:
        return []
    tri = v[f]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1)
    ok = np.isfinite(area) & (area > 1e-12)
    f, tri, normal, area = f[ok], tri[ok], cross[ok] / area[ok][:, None], area[ok]
    cos_tol = math.cos(math.radians(SLAB_ANGLE_DEG))
    remaining = np.argsort(-area)
    slabs = []
    while len(remaining):
        seed = normal[remaining[0]]
        member = normal[remaining] @ seed > cos_tol
        cluster, remaining = remaining[member], remaining[~member]
        n_mean = (normal[cluster] * area[cluster, None]).sum(0)
        n_mean /= max(np.linalg.norm(n_mean), 1e-12)
        stack = [cluster]
        while stack:
            idx = stack.pop()
            pts = tri[idx].reshape(-1, 3)
            c = pts.mean(0)
            d = pts - c
            evals, evecs = np.linalg.eigh(d.T @ d)
            m = evecs[:, 0]
            if np.abs(d @ m).max() <= SLAB_PLANAR_TOL or len(idx) == 1:
                if m @ n_mean < 0:
                    m = -m
                slab = _hull_points(np.concatenate([pts, pts - SLAB_THICKNESS * m]))
                if slab is not None:
                    slabs.append(slab)
                continue
            cen = tri[idx].mean(1)
            axis = int(np.argmax(cen.max(0) - cen.min(0)))
            order = np.argsort(cen[:, axis], kind="stable")
            half = len(order) // 2
            stack += [idx[order[:half]], idx[order[half:]]]
    return slabs


def _decompose_worker(src: Path, dst: Path) -> None:
    import coacd
    import fast_simplification

    data = np.load(src)
    v, f = data["vertices"].astype(np.float64), data["faces"].astype(np.int64)
    if len(f) > MAX_FACES:
        v2, f2 = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32),
                                              target_count=MAX_FACES)
        if len(f2):
            v, f = v2.astype(np.float64), f2.astype(np.int64)
    coacd.set_log_level("error")
    parts = coacd.run_coacd(coacd.Mesh(v, f), threshold=COACD_THRESHOLD,
                            max_convex_hull=MAX_HULLS, mcts_iterations=COACD_MCTS_ITERATIONS,
                            mcts_max_depth=COACD_MCTS_DEPTH, seed=0)
    np.savez(dst, *[np.asarray(p[0], np.float64) for p in parts])


def decompose_all(meshes: dict, python: str, env: dict, workers: int) -> dict:
    """CoACD hulls of every mesh {key: (vertices, faces)}: {key: (list of point arrays, method)}."""
    out = {}
    with tempfile.TemporaryDirectory(prefix="coacd-") as tmp:
        def run(key):
            v, f = meshes[key]
            src, dst = Path(tmp) / f"{key}.npz", Path(tmp) / f"{key}-hulls.npz"
            np.savez(src, vertices=v, faces=f)
            method = "coacd"
            try:
                subprocess.run([python, __file__, "decompose", str(src), str(dst)], env=env,
                               timeout=COACD_TIMEOUT_S, check=True, capture_output=True)
                with np.load(dst) as z:
                    hulls = [z[k] for k in z.files]
            except (subprocess.SubprocessError, OSError, ValueError):
                hulls, method = [v], "convex hull"
            hulls = [h for h in (_hull_points(h) for h in hulls) if h is not None]
            if not hulls and method == "coacd":
                hulls, method = [h for h in [_hull_points(v)] if h is not None], "convex hull"
            out[key] = (hulls, method)

        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(run, sorted(meshes)))
    return out


# --------------------------------------------------------------------------- one drop test
def _mjcf(body: list[np.ndarray], center: np.ndarray, fixed: list[np.ndarray],
          slabs: list[np.ndarray], floor_z: float) -> str:
    def verts(p):
        return " ".join(f"{x:.6f}" for x in np.asarray(p).ravel())

    assets, world, geoms = [], [], []
    for i, h in enumerate(fixed):
        assets.append(f'<mesh name="f{i}" vertex="{verts(h)}"/>')
        world.append(f'<geom name="f{i}" type="mesh" mesh="f{i}"/>')
    for i, h in enumerate(slabs):
        assets.append(f'<mesh name="s{i}" vertex="{verts(h)}"/>')
        world.append(f'<geom name="s{i}" type="mesh" mesh="s{i}"/>')
    for i, h in enumerate(body):
        assets.append(f'<mesh name="b{i}" vertex="{verts(h - center)}"/>')
        geoms.append(f'<geom name="b{i}" type="mesh" mesh="b{i}" density="{DENSITY}"/>')
    return f"""<mujoco>
<compiler boundmass="1e-6" boundinertia="1e-10"/>
<option timestep="{DT}" gravity="0 0 -9.81"/>
<size memory="512M"/>
<default><geom friction="{FRICTION} 0.005 0.0001"/></default>
<asset>{''.join(assets)}</asset>
<worldbody>
<geom name="ground" type="plane" size="0 0 1" pos="0 0 {floor_z:.6f}"/>
{''.join(world)}
<body name="object" pos="{verts(center)}"><freejoint/>{''.join(geoms)}</body>
</worldbody>
</mujoco>"""


def _drop_worker(src: Path, dst: Path) -> None:
    import mujoco

    data = np.load(src, allow_pickle=False)
    body = [data[k] for k in sorted(data.files) if k.startswith("body_")]
    fixed = [data[k] for k in sorted(data.files) if k.startswith("fixed_")]
    slabs = {int(k.split("_")[1]): data[k] for k in data.files if k.startswith("slab_")}
    floor_z, diag = float(data["floor_z"]), float(data["diag"])
    center = np.concatenate(body).mean(0)
    removed = []
    for attempt in range(MAX_RETRIES + 1):
        keys = sorted(slabs)
        model = mujoco.MjModel.from_xml_string(_mjcf(body, center, fixed, [slabs[k] for k in keys], floor_z))
        sim = mujoco.MjData(model)
        mujoco.mj_forward(model, sim)
        obj = model.body("object").id
        bad = set()
        for c in sim.contact[:sim.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = model.geom_bodyid[g1], model.geom_bodyid[g2]
            if c.dist < -PENETRATION_TOL and obj in (b1, b2):
                other = model.geom(g2 if b1 == obj else g1).name
                if other.startswith("s"):
                    bad.add(keys[int(other[1:])])
        if not bad or attempt == MAX_RETRIES:
            break
        removed += sorted(bad)
        for k in bad:
            del slabs[k]
    p0 = sim.xpos[obj].copy()
    for _ in range(int(round(SIM_SECONDS / DT))):
        mujoco.mj_step(model, sim)
    shift = float(np.linalg.norm(sim.xpos[obj] - p0))
    z_axis = sim.xmat[obj].reshape(3, 3)[:, 2]
    tilt = math.degrees(math.acos(float(np.clip(z_axis[2], -1.0, 1.0))))
    fallen = (not np.isfinite(shift) or tilt > FALL_TILT_DEG
              or shift > max(FALL_MIN_SHIFT, FALL_SHIFT_FRAC * diag))
    dst.write_text(json.dumps({"fallen": bool(fallen), "shift": shift if np.isfinite(shift) else None,
                               "tilt": tilt, "removed_slabs": len(removed)}))


def _boxes(parts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not parts:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.stack([p.min(0) for p in parts]), np.stack([p.max(0) for p in parts])


def _near(lo_all: np.ndarray, hi_all: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Which boxes come within NEARBY of the box [lo, hi]."""
    return (hi_all >= lo - NEARBY).all(1) & (lo_all <= hi + NEARBY).all(1)


def drop_tests(tested: dict, hulls: dict, slabs: list[np.ndarray], floor_z: float,
               python: str, env: dict, workers: int) -> dict:
    """Drop test of every key in `tested` ({key: bounding-box diagonal}) with the hulls of all
    other keys and the slabs fixed. Returns {key: result}."""
    out = {}
    owners = [k for k, hs in hulls.items() for _ in hs]
    parts = [h for hs in hulls.values() for h in hs]
    part_lo, part_hi = _boxes(parts)
    slab_lo, slab_hi = _boxes(slabs)
    with tempfile.TemporaryDirectory(prefix="drop-") as tmp:
        def run(key):
            body = hulls.get(key, [])
            if not body:
                out[key] = {"fallen": True, "error": "no collision geometry (the object has no volume)"}
                return
            pts = np.concatenate(body)
            lo, hi = pts.min(0), pts.max(0)
            arrays = {f"body_{i:03d}": h for i, h in enumerate(body)}
            near = _near(part_lo, part_hi, lo, hi)
            for j in np.nonzero(near)[0]:
                if owners[j] != key:
                    arrays[f"fixed_{j:05d}"] = parts[j]
            for i in np.nonzero(_near(slab_lo, slab_hi, lo, hi))[0]:
                arrays[f"slab_{i}"] = slabs[i]
            src, dst = Path(tmp) / f"{key}.npz", Path(tmp) / f"{key}.json"
            np.savez(src, floor_z=np.float64(floor_z), diag=np.float64(tested[key]), **arrays)
            try:
                proc = subprocess.run([python, __file__, "drop", str(src), str(dst)], env=env,
                                      timeout=TEST_TIMEOUT_S, capture_output=True, text=True)
                if proc.returncode != 0:
                    lines = (proc.stderr or proc.stdout).strip().splitlines()
                    raise RuntimeError(lines[-1][:300] if lines else f"exit status {proc.returncode}")
                out[key] = json.loads(dst.read_text())
            except subprocess.TimeoutExpired:
                out[key] = {"fallen": True, "error": f"the simulation took longer than {TEST_TIMEOUT_S} s"}
            except (RuntimeError, OSError, ValueError) as exc:
                out[key] = {"fallen": True, "error": f"the simulation failed: {exc}"}

        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(run, sorted(tested)))
    return out


def worker_env(pylib: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(pylib) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    return env


if __name__ == "__main__":
    if sys.argv[1] == "decompose":
        _decompose_worker(Path(sys.argv[2]), Path(sys.argv[3]))
    elif sys.argv[1] == "simplify":
        _simplify_worker(Path(sys.argv[2]), Path(sys.argv[3]))
    elif sys.argv[1] == "drop":
        _drop_worker(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        raise SystemExit(f"unknown command {sys.argv[1]}")
