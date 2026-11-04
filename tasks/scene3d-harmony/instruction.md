# A Blender scene from one photo of a room, with every piece of furniture a real 3D object

Each input is one photo of a furnished room, one of the photos of the HARMONY300 benchmark: a
third are renders of 3D-FRONT rooms, the others real photographs, half of them cluttered. Nothing
else is given: not the camera, not the depth. Rebuild the room in Blender: its shell (floor, walls,
ceiling) and every piece of furniture as its own 3D object with its whole shape, the sides and back
the photo does not show included, the shape a real piece of furniture of its kind has (not a
surface lifted from the photo); with physically based materials and lights; and the camera the
photo was taken with, so that Blender's path tracer renders the scene from that camera like the
photo.

## The interface: `reconstruct.py`

`reconstruct.py` in the workspace must define

```python
def reconstruct(scene_dir: str, out_dir: str) -> None:
```

`scene_dir` holds `photo.png`, the photo as HARMONY300 publishes it, in RGB: 1296 x 968 for the
renders, between 1200 and 3000 pixels on the long side for the real photos. The function writes
`scene.blend` into `out_dir` (an existing, empty directory): a Blender file of at most 512 MB, with
every image it uses packed into it (no linked libraries or other external files), that Blender 4.5
opens (see Resources), whose active scene holds:

- **the camera** (`scene.camera`): your estimate of the photo's camera, its pose and its
  intrinsics. Estimate it from each photo; do not take a field of view or any other camera
  parameter from the labels (the 3D-FRONT renders happen to share one camera).
- **the world's +z axis pointing up** (gravity along -z).
- **the room** as mesh objects whose names start with `floor`, `ceiling` or `wall` (they are
  never taken for furniture); every other mesh object counts as furniture or another object of the
  room.

The evaluator calls `reconstruct` once per photo, one photo after another, in one process, so
models loaded at module level stay loaded. A call that raises leaves that photo empty and the
others go on. Each call must use only its own photo.

## How the evaluator renders `scene.blend`

The evaluator opens the file in Blender 4.5.14 LTS without running its scripts and uses the file's
active scene (`tests/blender_eval.py` in the task package; read it to see exactly what it does):

- **Camera**: the scene's camera, with its position, orientation and intrinsics (its scale and any
  mirroring are dropped, depth of field is turned off). It must be a perspective camera; its focal
  length, sensor size and fit and its shift set the field of view and the principal point, with
  square pixels. A horizontal field of view outside 10 to 150 degrees is an error. Blender cameras
  look along their local -z axis with +y up.
- **Render**: Cycles at half the photo's size, the long side at most 1024 pixels (648 x 484 for the
  renders), 128 samples per pixel without adaptive sampling, seed 0, OpenImageDenoise; at most 12
  bounces (4 diffuse, 4 glossy, 12 transmission, 8 transparent), indirect light clamped at 10, a
  1.5-pixel Blackman-Harris filter. Color management: sRGB display, your view transform if it is
  Standard, Filmic or AgX (otherwise Standard), your exposure (within +-10), no look, gamma 1, no
  curves; an 8-bit PNG.
- **Geometry**: only mesh objects render, with their modifiers except Geometry Nodes and particle
  systems, and without instancing. Other objects (curves, text, point clouds, volumes, ...) are
  hidden: convert them to meshes. Every mesh object is visible to camera, diffuse, glossy,
  transmission, volume and shadow rays; holdouts, shadow catchers, holdout and indirect-only
  collections, view-layer overrides and light linking are turned off.
- **Materials** are physically based. Their node trees, groups included, may use only the nodes in
  `blender_eval.SHADER_NODES`: the Principled, Diffuse, Glossy, Glass, Refraction, Translucent,
  Transparent, Sheen, Metallic and Subsurface Scattering BSDFs, Mix Shader, packed image textures,
  procedural textures and the input, color, vector and converter nodes listed there. A material
  with any other node (Emission, Add Shader, Light Path, Ambient Occlusion, Camera Data, Holdout,
  volume shaders, scripts, ...) or an image that is not packed is replaced by a gray (0.5)
  Principled BSDF. In the others, the Principled BSDF's emission is set to zero, every color that
  feeds a BSDF is clamped to [0, 1], and so are the Principled BSDF's weights (metallic, roughness,
  specular, transmission, coat, sheen, subsurface, alpha, anisotropy) and the Mix Shader factors;
  the Volume output is disconnected, and displacement only bumps: it does not move the surface.
- **Lights**: point, sun, spot and area lights, without nodes, casting shadows, with a power of at
  least zero; at most 64 (the 64 strongest are kept).
- **World**: a Background shader with a constant color or a Sky Texture; otherwise it is black.

## Metric

The primary metric is **`score`, higher is better, in [0, 1]** (`tests/evaluate.py`,
`tests/metrics.py`, `tests/judge.py`):

```
score = (appearance_term + object_term + geometry_term) / 3
```

`appearance_term` and `object_term` are averaged over all the photos of a split, `geometry_term`
over its 3D-FRONT renders (the only photos with true 3D geometry).

- **Appearance**, the HARMONY paper's metrics of the render against the photo (averaged over pixel
  blocks to the render's size): `n_clip`, 1 - the cosine similarity of their CLIP ViT-B/32 image
  embeddings (CLIP's preprocessing: the short side to 224 pixels, the center 224 x 224); `pl`, the
  mean squared difference of their RGB values in [0, 1]; `lpips` (AlexNet, v0.1). Also `psnr` and
  `ssim`.

  ```
  appearance_term = (max(0, 1 - n_clip / 0.25) + max(0, 1 - pl / 0.1) + 1 - lpips) / 3
  ```

- **Objects**, on every photo: the annotated pieces of furniture (1 to 8 per photo; on the real
  photos found by a segmentation model, see Data). Every rendered mesh object is assigned to the
  label, an annotated object, other furniture or the room, that covers most of the pixels it shows
  in the input view (the room's mesh objects to the room); the mesh objects assigned to an
  annotated object are its reconstruction.
  `object_iou` is the intersection over union, in the input view, of the pixels the reconstruction
  shows and those the object covers. `object_realness` says whether the reconstruction is a real 3D
  piece of furniture (`tests/judge.py`):
  - from the reconstruction's own front surface (its depth seen alone from your camera), the
    evaluator builds three shortcuts: that surface as a sheet; the sheet extruded along the
    camera's rays to the reconstruction's farthest depth (a prism); the reconstruction's box,
    upright and aligned with its horizontal principal axes (left out for box-shaped furniture:
    cabinets, drawers, wardrobes, shelves, ...);
  - each shape is rendered in plain gray clay, without its materials, from four directions about
    the vertical axis (+z) through the center of its box: from your camera's side, then 90, 180
    and 270 degrees around, at your camera's height;
  - the vision-language model Qwen3-VL-8B-Instruct (in `scorer/`) sees the photo's crop around the
    object, then the reconstruction's four views and a shortcut's, in both orders, and answers
    which looks like a real <label> from all four directions, with the sides and back such a
    <label> has. P is the probability, from its answer's logits, that it picks the reconstruction,
    averaged over the two orders; a shortcut against itself gives 0.5;
  - `realness = clip((min P over the shortcuts - 0.5) / 0.4, 0, 1)`; nothing assigned gives 0.

  ```
  object_term = mean over the annotated objects of realness x min(1, object_iou / 0.5)
  ```

- **Geometry**, on the 3D-FRONT renders only, against the true meshes of the furniture in view:
  - the HARMONY paper's scene geometry (its release's `evaluation/chamfer_eval.py`): your
    furniture, every rendered mesh object whose name does not start with `floor`, `ceiling` or
    `wall`, against the true furniture: 30,000 points on each, each point set centered and divided
    by the largest side of its bounding box, yours rigidly aligned to the truth (ICP from the
    principal axes with four sign flips), then `paper_chamfer` and the F-scores `paper_f@0.1`,
    `paper_f@0.01` and `paper_f@0.001` in those normalized units (`metrics.paper_geometry`; the
    points are drawn with a fixed seed);
  - per annotated object in the camera's frame: `object_f`, the F-score at 5 cm, in percent,
    between the reconstruction (scaled by s, the median of true / rendered depth over the pixels
    that see a surface) and the object's true mesh, 50,000 points sampled by area on each;
    `object_f_hidden`, the same F-score with the recall counted only over the object's points the
    photo does not show (outside the frame or more than 3 cm + 2 % behind the true depth of their
    pixel), or `object_f` for an object with fewer than 5 % such points. An object with nothing
    assigned scores 0.

  ```
  geometry_term = (paper_f@0.1 + paper_f@0.01 + object_f + object_f_hidden) / 400
  ```

A photo whose scene is missing or does not load, has more than 4,000,000 triangles, or that Blender
takes more than 15 minutes to open and render, renders black and has no geometry. Also reported:
`object_found` (the share of annotated objects with a reconstruction), `object_f10`, `object_cd`
(cm), `object_precision`, `object_recall`, `object_recall_hidden`, `depth_absrel` (your geometry's
depth against the true depth of the renders, after the median scale), each term per difficulty,
`n`, `n_easy`, `failed`, `objects`, `lights`, `triangles`, `materials_replaced`, `fov` (your
camera's horizontal field of view), `render_s`, `seconds` and `timed_out`.

`evaluate_dev` returns these metrics, `per_scene.json` (every photo's metrics, its objects' scores
with the judge's answers and the mesh objects assigned to them, the changes Blender made to follow
the rules, and why a scene did not load), `solution.log` (the output of your process) and
`renders/<id>.jpg` for every photo: the photo, the render, your mesh objects colored by the object
they were assigned to (with the annotated objects' outlines) / two views with the camera turned by
-30 and +30 degrees about its up axis through the point at the median depth (not scored), and the
depth error (renders) or the judge's clay views of the largest annotated object (real photos).

## Data and rules

The dev data directory holds `inputs/<id>/photo.png` and `labels/<id>/` for the 30 photos
`evaluate_dev` scores (10 of each difficulty). The hidden val split has 30 photos and the hidden
test split 60, drawn the same way (10 and 20 of each difficulty); no photo is in two splits. The
labels:

- `meta.json`: `difficulty` (`easy` for the 3D-FRONT renders, `medium` for real photos with one
  dominant layout, `complicated` for cluttered ones), `name`, `source`, `width`, `height`; for the
  renders also `camera`, their true pinhole (`fx`, `fy`, `cx`, `cy` in pixels; (0, 0) is the
  top-left corner of the top-left pixel).
- `objects.json`: the annotated furniture, `label` and `boxy` (whether the judge leaves out the
  box), and for the renders its box (`center`, `axes`, `half_size`) in the camera's frame.
- The renders: `gt.npz` (the furniture in view: `vertices` in the camera's frame, x right, y down,
  z forward, in meters; `faces`; and `face_object`, the annotated object each face belongs to, -1
  for furniture that is not annotated), `layout_depth.npy` (the depth of the room without its
  furniture, at the render's size) and `depth.npy` (the photo's depth, 1296 x 968).
- The real photos: `instances.png` (the photo's size; k on the pixels of the k-th object of
  `objects.json`, 0 elsewhere), from Mask2Former (Swin-L, ADE20K panoptic) segments of furniture
  classes; on the renders this annotation finds 85 % of the true annotated objects.

`scorer/` holds the weights the evaluator scores with (AlexNet and the LPIPS layers, CLIP ViT-B/32,
Qwen3-VL-8B-Instruct), `pylib/` (the Python packages they need) and `blender/`.

The photos come from HARMONY300, a public benchmark whose download also holds another method's
reconstructions, renders, cameras and segmentations of every photo. Do not download, search for or
use HARMONY300, its source photos or datasets (Pexels and Unsplash pages of these photos, 3D-FRONT,
3D-FUTURE (its furniture models) or copies of its models, the FRONT3D subset on Hugging Face, the
SSR evaluation data), any method's reconstructions of these photos, or the HARMONY paper's code, and
do not look for the validation or test data. Use only the dev data, your own work and publicly
released models and libraries (depth, camera, segmentation, layout, image-to-3D, inpainting,
material and lighting estimation models, for example).

## Resources

- Blender 4.5.14 LTS is in the dev data at `scorer/blender/`; `scorer/blender/blender` runs it with
  the X11 and GL libraries it needs in the task container. The evaluator renders with it and, when
  it runs your solution, sets `BLENDER` to that path (read-only). Save `scene.blend` with it or
  another Blender 4.5: a file saved by a newer Blender may not open. Cycles has no precompiled CUDA
  kernels for these GPUs, so the first GPU render compiles them (about 4 minutes); the container
  sets `CUDA_CACHE_DISABLE=1`: set `CUDA_CACHE_DISABLE=0`, `CUDA_CACHE_PATH` to a directory and
  `CUDA_CACHE_MAXSIZE=4294967296` to keep them (`$RSIB_CACHE` is read-only during an evaluation).
- The evaluation runs on one GPU (80 GB), with Python, numpy, scipy, Pillow and PyTorch, and no
  network. Download weights, code and Python packages into `$RSIB_CACHE` while you work (`HF_HOME`,
  `TORCH_HOME` and `PIP_CACHE_DIR` point there); the evaluator reads it but cannot download
  anything. If your solution needs another Python interpreter, it must live in `$RSIB_CACHE`; write
  its absolute path in `runtime.txt` in the workspace.
- Time: a split must finish within 120 seconds per photo plus 600 seconds (dev, 30 photos: 70
  minutes; test, 60 photos: 130 minutes). At that point the process is stopped and the photos not
  yet done count as empty. Scoring then compiles Blender's GPU kernels (about 4 minutes) and takes
  about 30 seconds per photo (the render and the judge).
- Files larger than 5 MB are never kept with the solution; keep large weights in `$RSIB_CACHE`.
