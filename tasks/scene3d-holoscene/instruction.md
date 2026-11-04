# A simulation-ready 3D scene from an indoor video

Each input is one indoor scene: a few hundred RGB frames of a video walking through a room or a
house, and nothing else: no cameras, no masks, no depth. Estimate the cameras and build the scene
in 3D as separate objects on a fixed background (floor, walls, ceiling), textured so that it looks
like the frames from their cameras, with the true shape, and physically plausible: every object
that stands on the floor or on another object must stay in place when it is dropped under gravity.

This is the setting of CoDimRecon (arXiv 2609.36024), whose metrics (Appendix C) are used:
geometry against the ground-truth surface, rendering of the input views, latent video similarity
and rigid-body stability.

## The interface: `reconstruct.py`

`reconstruct.py` in the workspace must define

```python
def reconstruct(scene_dir: str, out_dir: str) -> None:
```

`scene_dir` holds only `images/frame_NNNNN.jpg` (or `.png`): the frames, in capture order. They
are distortion-free pinhole images; the intrinsics are not given.

It writes two files into `out_dir` (an existing, empty directory):

- `scene.glb`: glTF 2.0 binary, everything embedded, at most 512 MB and 4,000,000 triangles, in a
  world frame and scale of your choice.
- `cameras.json`: your camera of every frame in that frame,
  `{"frames": [{"file_path": "images/frame_00000.jpg", "transform_matrix": [[...], ...]}, ...]}`,
  each `transform_matrix` the 4 x 4 camera-to-world matrix (a rotation and a translation) with the
  camera's x axis pointing right, y up and z backwards (OpenGL / nerfstudio). Every frame needs
  one; a missing or invalid `cameras.json` leaves the scene empty.

The evaluator calls `reconstruct` once per scene, one scene after another, in one process, so
models loaded at module level stay loaded. A call that raises leaves that scene empty and the
others go on. Each call must use only its own scene.

## How a scene is placed

The evaluator moves your scene into the ground truth's frame with the similarity (rotation, scale,
translation) that maps your cameras onto the true ones (`tests/align.py`): the rotation that best
maps your camera orientations onto the true orientations, then scale and translation by least
squares between the camera centers (if your centers do not spread, the scale is 1). Everything is
then scored there: rendered from the **true** cameras with their true intrinsics, matched against
the true masks, and dropped under the **true** gravity. So camera errors show up as renders that
miss the frames and as a scene that is tilted or at the wrong scale; and the floor must be level
with respect to the gravity of the scene your cameras imply. Reported: `camera_rot_err` (the mean
angle in degrees between your placed camera orientations and the true ones), `camera_ate` (the
root mean square distance in cm of your placed camera centers from the true ones) and
`camera_scale`.

## Objects and the background

Each glTF node with a mesh is one part. The evaluator renders the placed scene from every true
camera and assigns each node to the label of the ground-truth masks, background (0: walls, floor,
ceiling and everything not annotated, such as pictures, windows and doors) or annotated object k,
that covers most of the pixels the node shows, summed over all frames. The masks are not given:
finding the objects is part of the task. All nodes assigned to object k together are the
reconstruction of object k, one rigid body; the nodes assigned to the background are fixed. A node
that no frame sees is assigned to nothing: it counts for geometry but takes no part in the
physics. So: one node per annotated object (or several, if they all land on that object), and the
background in one or more nodes of its own. An object merged into the background is a missing
object.

## How a scene is rendered

The evaluator renders `scene.glb` itself (`tests/raster.py` and `tests/gltf.py` in the task
package; read them to see exactly what it does):

- A pixel shows the **base color** of the nearest surface: the material's `baseColorTexture`
  (sRGB) times `baseColorFactor` times the vertex color `COLOR_0`, in linear RGB, as glTF defines
  them, **without lighting**. Bake the appearance of the frames into textures or colors.
  Exporters may set `baseColorFactor` below 1 (trimesh's default material sets 0.4), which
  darkens the texture. Pixels that see no surface are black.
- Faces of materials that are not `doubleSided` are culled when seen from behind (glTF front faces
  are counter-clockwise). Surfaces closer than 1 cm to the camera are clipped.
- Read: every node of the default scene with its transforms; triangle meshes (also strips and
  fans); `TEXCOORD_n` with the sampler's wrap modes and `KHR_texture_transform`; embedded PNG, JPEG
  and WebP images (at most 256 million texture pixels in total). Ignored: lights, cameras,
  emission, metal and roughness, normal maps, alpha (everything is opaque), skins, morph targets,
  animations. A scene that uses Draco or meshopt compression, KTX2 textures, sparse accessors or
  external files does not load and counts as empty.
- Every input frame is rendered at its resolution with 2 x 2 samples per pixel, from its true
  camera (after the placement above).

`glb.py` in the starter writes meshes with flat colors, vertex colors or textures to a `.glb`; you
may use it, change it, or use any other exporter.

## Metric

The primary metric is **`score`, higher is better, in [0, 1]**, per scene and averaged over the
scenes (`tests/evaluate.py`, `tests/metrics.py`, `tests/physics.py`):

```
score = (geometry_term + appearance_term + physics_term) / 3
geometry_term   = (f1 + nc) / 200
appearance_term = (ssim + 1 - lpips) / 2
physics_term    = stable_all_gt / 100
```

- **Geometry**, the whole scene (every node) against the ground-truth mesh, walls, floor and
  ceiling included, without visibility culling: 500,000 points sampled on each by area. `cd`:
  the symmetric mean nearest-neighbor distance in cm. `f1`: the F-score at 5 cm, in percent, of
  precision (share of your points within 5 cm of the ground truth) and recall (share of
  ground-truth points within 5 cm of yours). `nc`: the mean absolute cosine of the normals of
  nearest neighbors, both ways, in percent. The ground truth covers what the capture saw: surfaces
  you add where nothing was ever seen lower the precision.
- **Rendering**, every input frame against its render, full images: `psnr`, `ssim` (11 x 11
  Gaussian window) and `lpips` (AlexNet, v0.1), averaged over the frames.
- **Stability** (`tests/physics.py`), as in the paper: the background is simplified to at most
  100,000 faces and its faces become fixed convex slabs (planar patches within 3 cm, extruded 5 cm
  behind the side they face); every object loses its faces outside its ground-truth bounding box
  grown by 10 cm (geometry beyond the object, such as a base hidden under the floor or floor
  assigned to the object, does not hold it up), then it is decimated to at most 20,000 faces and
  decomposed by CoACD (at most 32 convex hulls, threshold 0.05, a search of 60 iterations and depth
  2; its convex hull if CoACD fails or takes more than 15 minutes); an infinite ground plane lies
  at the floor height. Each dynamic object is then released alone in MuJoCo, with everything else
  fixed: 3 s at 2 ms steps, gravity 9.81 m/s^2, density 300 kg/m^3, friction 1.0. A background slab
  that initially penetrates it by more than 2 cm is removed (at most three times). It has fallen
  when it tilts by more than 30 degrees or moves more than max(5 cm, 0.2 x its bounding-box
  diagonal), or when its simulation does not finish within 10 minutes. Dynamic objects are the
  annotated objects that rest on the floor or on something else; objects that hang from a wall or
  the ceiling are fixed (`labels/<id>/objects.json` in the dev data lists them).
  **`stable_all_gt`**: the share of the scene's dynamic objects that are reconstructed (their
  nodes' pixels overlap the object's masks with an intersection over union of at least 0.5, summed
  over all frames) and stay in place. Also reported as in the paper: `stable_all` and
  `stable_ground`, the share of the dynamic objects you reconstruct (and of those that touch the
  floor) that stay in place.
- **Latent similarity**: 64 frames spread over the sequence and their renders, encoded by the
  frozen V-JEPA 2.1 ViT-G/16 (384 x 384): `motion`, the cosine of the two clips' mean tokens, and
  `layout`, the mean cosine of corresponding patches averaged over time, in percent.

A scene that is missing or does not load, or whose `cameras.json` is missing or invalid, renders
black and has no surface and no objects. Also reported: the three terms, `camera_rot_err`,
`camera_ate`, `precision`, `recall`, `object_iou` (mean over the annotated objects)
and `object_recall` (share with an intersection over union of at least 0.5), `stable_ground_gt`,
`coverage` (share of pixels that see a surface), `n`, `failed` (scenes that did not load),
`nodes`, `triangles`, `seconds` and `timed_out`.

`evaluate_dev` returns these metrics, `per_scene.json` (every scene's metrics and, for every
annotated object, its intersection over union, its nodes, its collision geometry and its drop
test), `solution.log` (the output of your process) and `renders/<id>.jpg` for every scene: four
frames, each with its render, the annotated masks and the labels your rendered nodes were
assigned to.

## Data and rules

The dev data directory holds `inputs/<id>/` (the frames) and `labels/<id>/` for the two scenes
`evaluate_dev` scores: a synthetic house and a game room filmed with a phone (its frames are
undistorted). The labels are what the evaluator scores with: `transforms.json` (the true
intrinsics `fl_x`, `fl_y`, `cx`, `cy`, `w`, `h` and every frame's true camera-to-world matrix,
z up, meters), `masks/frame_NNNNN.png` (8-bit, 0 the background, k annotated object k, the same
number in every frame), `graph.json` (HoloScene's support graph: node k is object k, node 0 the
background; read as a tree rooted at node 0, a node's children rest on it), `mesh.ply` (the
ground-truth surface), `gt.npz` (its `vertices`, `faces` and
`face_label`: the label each face shows in the masks, -1 where no frame shows it), `objects.json`
(the floor height and, per object, its category: `dynamic` or `static`, and whether it touches
the floor) and `meta.json`. `scorer/` holds what the evaluator scores with: the LPIPS weights, the
V-JEPA 2.1 encoder and its code, and `pylib/` with MuJoCo, CoACD and fast-simplification for
Python 3.10, so you can compute the metrics yourself (put `scorer/pylib` on `sys.path`). The
hidden val split is one more synthetic house; the hidden test split has six scenes, three
photorealistic renders of scanned rooms and three real camera captures, so test differs from dev
and val: real frames have exposure changes and blur, and their true cameras (from the capture's own
reconstruction) are less exact.

The scenes come from HoloScene's public release, whose download and repository also contain
HoloScene's own reconstructions and the ground truth of the hidden scenes. Do not download, search
for or use the HoloScene release or code, the source datasets of these scenes (Replica, ScanNet++,
Gibson / iGibson) or their meshes, or any method's reconstructions of them, and do not look for
the validation or test data. Use only the dev data, your own work and publicly released models
and libraries (depth, multi-view geometry, segmentation, image-to-3D and inpainting models, for
example).

## Resources

- The evaluation runs on one GPU (80 GB), with Python, numpy, scipy, Pillow and PyTorch, and no
  network. Download weights, code and Python packages into `$RSIB_CACHE` while you work
  (`HF_HOME`, `TORCH_HOME` and `PIP_CACHE_DIR` point there); the evaluator reads it but cannot
  download anything. If your solution needs another Python interpreter, it must live in
  `$RSIB_CACHE`; write its absolute path in `runtime.txt` in the workspace.
- Time: a split must finish within 1200 seconds per scene plus 600 seconds (dev, 2 scenes: 50
  minutes; test, 6 scenes: 130 minutes). At that point the process is stopped and the scenes not
  yet written count as empty. Scoring takes several more minutes per scene.
- Files larger than 5 MB are never kept with the solution; keep large weights in `$RSIB_CACHE`.
