"""Runs inside Blender (blender -b --factory-startup --python blender_eval.py -- <job.json>).

Opens a submitted scene.blend without running its scripts, enforces the task's rules on it, renders
the input view (and two orbit views for the contact sheet) with Cycles at the evaluator's settings,
and exports exactly the geometry that renders. job.json holds:

    blend, out_dir     the scene and where to write
    width, height      the render size in pixels (the photo's aspect ratio)
    samples            Cycles samples per pixel; device "CUDA" or "CPU"
    orbits             degrees to turn the camera for the extra views ([] for none)
    orbit_samples      samples for the extra views

It writes render.png (and orbit_<deg>.png), geometry.npz (every rendered mesh object's triangles in
world coordinates, the camera-to-world matrix), camera.json (the camera's intrinsics in pixels at
the render size) and audit.json (what was changed and why). Rules, as in instruction.md:

- Only mesh objects render, with their modifiers except Geometry Nodes and particle systems, and
  no instancing. Other geometry (curves, text, point clouds, volumes, ...) is hidden.
- Every mesh is visible to every kind of ray; no holdouts, shadow catchers, indirect-only or
  holdout collections, material or world overrides, light linking.
- Materials are physically based: the node types in SHADER_NODES only, no Emission (Principled
  BSDF's emission is set to zero), no volume; displacement only bumps. A material that breaks this
  is replaced by DEFAULT_COLOR Principled BSDF. Colors that feed a BSDF are clamped to [0, 1],
  weights too.
- Lights are point, sun, spot or area lights without nodes, casting shadows, with a power of at
  least zero; at most MAX_LIGHTS (the strongest).
- The world is a Background shader of a constant color (or a procedural Sky Texture); otherwise it
  is replaced by a black one.
- The camera is the scene's: a perspective camera without depth of field, its pose (scale and
  mirroring dropped), focal length, sensor fit and shift, with square pixels; its horizontal field
  of view at the render size must be within FOV_RANGE. Render settings and color management are
  the evaluator's, except the view transform (Standard, Filmic or AgX) and the exposure.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

DEFAULT_COLOR = (0.5, 0.5, 0.5, 1.0)
VIEW_TRANSFORMS = ("Standard", "Filmic", "AgX")
MAX_TRIANGLES = 4_000_000
MAX_LIGHTS = 64
FOV_RANGE = (10.0, 150.0)       # degrees, horizontal

BSDFS = {"ShaderNodeBsdfPrincipled", "ShaderNodeBsdfDiffuse", "ShaderNodeBsdfGlossy",
         "ShaderNodeBsdfAnisotropic", "ShaderNodeBsdfGlass", "ShaderNodeBsdfRefraction",
         "ShaderNodeBsdfTranslucent", "ShaderNodeBsdfTransparent", "ShaderNodeBsdfSheen",
         "ShaderNodeBsdfVelvet", "ShaderNodeSubsurfaceScattering", "ShaderNodeBsdfMetallic"}
SHADER_NODES = BSDFS | {
    "ShaderNodeMixShader", "ShaderNodeOutputMaterial", "ShaderNodeOutputAOV",
    "ShaderNodeTexImage", "ShaderNodeTexCoord", "ShaderNodeUVMap", "ShaderNodeMapping",
    "ShaderNodeNormalMap", "ShaderNodeBump", "ShaderNodeDisplacement",
    "ShaderNodeVectorDisplacement", "ShaderNodeMix", "ShaderNodeMixRGB", "ShaderNodeRGB",
    "ShaderNodeValue", "ShaderNodeVertexColor", "ShaderNodeAttribute", "ShaderNodeSeparateColor",
    "ShaderNodeCombineColor", "ShaderNodeSeparateXYZ", "ShaderNodeCombineXYZ",
    "ShaderNodeSeparateRGB", "ShaderNodeCombineRGB", "ShaderNodeSeparateHSV",
    "ShaderNodeCombineHSV", "ShaderNodeHueSaturation", "ShaderNodeBrightContrast",
    "ShaderNodeGamma", "ShaderNodeInvert", "ShaderNodeRGBToBW", "ShaderNodeValToRGB",
    "ShaderNodeRGBCurve", "ShaderNodeVectorCurve", "ShaderNodeFloatCurve", "ShaderNodeMath",
    "ShaderNodeVectorMath", "ShaderNodeMapRange", "ShaderNodeClamp", "ShaderNodeFresnel",
    "ShaderNodeLayerWeight", "ShaderNodeNewGeometry", "ShaderNodeObjectInfo",
    "ShaderNodeTangent", "ShaderNodeNormal", "ShaderNodeVectorRotate",
    "ShaderNodeVectorTransform", "ShaderNodeBlackbody", "ShaderNodeWavelength",
    "ShaderNodeTexNoise", "ShaderNodeTexVoronoi", "ShaderNodeTexWave", "ShaderNodeTexMagic",
    "ShaderNodeTexChecker", "ShaderNodeTexBrick", "ShaderNodeTexGradient",
    "ShaderNodeTexWhiteNoise", "ShaderNodeTexGabor", "ShaderNodeTexMusgrave",
    "ShaderNodeWireframe", "ShaderNodeBevel", "ShaderNodeGroup", "NodeGroupInput",
    "NodeGroupOutput", "NodeReroute", "NodeFrame"}
WORLD_NODES = {"ShaderNodeBackground", "ShaderNodeOutputWorld", "ShaderNodeRGB", "ShaderNodeValue",
               "ShaderNodeTexSky", "NodeReroute", "NodeFrame"}
# Principled BSDF inputs that weigh a layer: kept in [0, 1]
WEIGHTS = {"Metallic", "Roughness", "Specular IOR Level", "Transmission Weight", "Coat Weight",
           "Sheen Weight", "Subsurface Weight", "Alpha", "Anisotropic", "Coat Roughness",
           "Sheen Roughness", "Thin Film Thickness"}
WEIGHTS -= {"Thin Film Thickness"}

audit: dict = {"changes": [], "materials_replaced": [], "objects_hidden": []}


def note(msg: str) -> None:
    if len(audit["changes"]) < 200:
        audit["changes"].append(msg)


# --------------------------------------------------------------------------- materials
def reachable(tree, start_sockets) -> list:
    """Nodes that feed the given input sockets, through reroutes and links (not into groups)."""
    seen, stack, out = set(), list(start_sockets), []
    while stack:
        sock = stack.pop()
        for link in sock.links:
            if not link.is_valid or link.is_muted:
                continue
            node = link.from_node
            if node.name in seen:
                continue
            seen.add(node.name)
            out.append(node)
            stack.extend(i for i in node.inputs if i.is_linked)
    return out


def tree_problems(tree, depth=0) -> list[str]:
    """Rule violations among all nodes of a (group) tree."""
    bad = []
    for node in tree.nodes:
        if node.bl_idname not in SHADER_NODES:
            bad.append(f"node {node.bl_idname}")
        elif node.bl_idname == "ShaderNodeGroup":
            if node.node_tree is None:
                continue
            if depth > 8:
                bad.append("node groups nested too deep")
            else:
                bad += tree_problems(node.node_tree, depth + 1)
        elif node.bl_idname == "ShaderNodeTexImage" and node.image is not None:
            img = node.image
            unpacked = img.source == "FILE" and img.packed_file is None
            if img.source not in ("FILE", "GENERATED") or unpacked:
                bad.append(f"image {img.name!r} is not packed")
    return bad


def clamp_input(tree, node, sock, lo=0.0, hi=1.0) -> None:
    """Keep what flows into a BSDF's color or weight input within [lo, hi]."""
    if sock.is_linked:
        link = sock.links[0]
        src = link.from_socket
        if sock.type == "RGBA":
            mix = tree.nodes.new("ShaderNodeMix")
            mix.data_type = "RGBA"
            mix.clamp_result = True
            mix.inputs[0].default_value = 0.0
            a = [i for i in mix.inputs if i.type == "RGBA" and i.enabled][0]
            res = [o for o in mix.outputs if o.type == "RGBA" and o.enabled][0]
            tree.links.new(src, a)
            tree.links.new(res, sock)
        else:
            cl = tree.nodes.new("ShaderNodeClamp")
            cl.inputs["Min"].default_value, cl.inputs["Max"].default_value = lo, hi
            tree.links.new(src, cl.inputs["Value"])
            tree.links.new(cl.outputs["Result"], sock)
    elif sock.type == "RGBA":
        v = list(sock.default_value)
        sock.default_value = [min(max(c, lo), hi) for c in v[:3]] + v[3:]
    elif sock.type == "VALUE":
        sock.default_value = min(max(sock.default_value, lo), hi)


def sanitize_bsdfs(tree, done: set) -> None:
    """Clamp the color and weight inputs of every BSDF in a tree and its groups; no emission."""
    if tree.as_pointer() in done:      # material trees are all called "Shader Nodetree"
        return
    done.add(tree.as_pointer())
    for node in list(tree.nodes):
        if node.bl_idname == "ShaderNodeGroup" and node.node_tree is not None:
            sanitize_bsdfs(node.node_tree, done)
        if node.bl_idname not in BSDFS:
            continue
        for sock in list(node.inputs):
            if sock.name == "Weight":
                for link in list(sock.links):
                    tree.links.remove(link)
                if sock.type == "VALUE":
                    sock.default_value = 1.0
            elif sock.name == "Emission Strength":
                if sock.is_linked or sock.default_value != 0.0:
                    note(f"{tree.name!r}: emission of {node.name!r} set to zero")
                for link in list(sock.links):
                    tree.links.remove(link)
                sock.default_value = 0.0
            elif sock.name == "Emission Color":
                for link in list(sock.links):
                    tree.links.remove(link)
                sock.default_value = (0.0, 0.0, 0.0, 1.0)
            elif sock.type == "RGBA":
                clamp_input(tree, node, sock)
            elif node.bl_idname == "ShaderNodeBsdfPrincipled" and sock.name in WEIGHTS:
                clamp_input(tree, node, sock)
    for node in tree.nodes:
        if node.bl_idname == "ShaderNodeMixShader":
            clamp_input(tree, node, node.inputs[0])


def default_material(name: str):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes.get("Principled BSDF")
    p.inputs["Base Color"].default_value = DEFAULT_COLOR
    p.inputs["Roughness"].default_value = 0.5
    return m


def check_materials(meshes: list) -> None:
    replace = {}
    done: set = set()
    for obj in meshes:
        for slot in obj.material_slots:
            m = slot.material
            if m is None or m.name in replace:
                continue
            m.displacement_method = "BUMP"
            if not m.use_nodes or m.node_tree is None:
                continue
            tree = m.node_tree
            out = tree.get_output_node("CYCLES")
            bad = tree_problems(tree)
            if out is None:
                bad.append("no material output")
            else:
                vol = out.inputs.get("Volume")
                if vol is not None and vol.is_linked:
                    for link in list(vol.links):
                        tree.links.remove(link)
                    note(f"material {m.name!r}: volume shader removed")
            if bad:
                replace[m.name] = sorted(set(bad))[:5]
                continue
            sanitize_bsdfs(tree, done)
    for name, why in replace.items():
        new = default_material(f"rsib_default_{name}")
        old = bpy.data.materials[name]
        for obj in meshes:
            for slot in obj.material_slots:
                if slot.material is old:
                    slot.material = new
        audit["materials_replaced"].append({"material": name, "reasons": why})


# --------------------------------------------------------------------------- scene
def check_world(scene) -> None:
    w = scene.world
    ok = w is not None and w.use_nodes and w.node_tree is not None
    if ok:
        tree = w.node_tree
        bad = [n.bl_idname for n in tree.nodes if n.bl_idname not in WORLD_NODES]
        out = tree.get_output_node("CYCLES")
        if bad or out is None:
            ok = False
            note(f"world: not allowed ({', '.join(sorted(set(bad))[:5]) or 'no output'})")
        else:
            vol = out.inputs.get("Volume")
            if vol is not None and vol.is_linked:
                for link in list(vol.links):
                    tree.links.remove(link)
                note("world: volume removed")
    elif w is not None:
        ok = not w.use_nodes  # a plain world color
    if not ok:
        new = bpy.data.worlds.new("rsib_world")
        new.use_nodes = True
        new.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
        scene.world = new
        note("world replaced by a black one")


def check_objects(scene) -> tuple[list, list]:
    meshes, lights = [], []
    vl = bpy.context.view_layer
    if getattr(vl, "material_override", None) is not None:
        vl.material_override = None
        note("view layer material override removed")
    if getattr(vl, "world_override", None) is not None:
        vl.world_override = None
        note("view layer world override removed")
    if hasattr(vl, "samples"):
        vl.samples = 0
    vl.use_freestyle = False

    def layer_collections(lc):
        yield lc
        for ch in lc.children:
            yield from layer_collections(ch)

    for lc in layer_collections(vl.layer_collection):
        if lc.holdout or lc.indirect_only:
            lc.holdout = lc.indirect_only = False
            note(f"collection {lc.name!r}: holdout / indirect only turned off")
        lc.hide_viewport = False
        lc.collection.hide_viewport = lc.collection.hide_render
    for obj in scene.objects:
        obj.hide_viewport = obj.hide_render
        try:
            obj.hide_set(False)
        except RuntimeError:  # not in the view layer: it does not render
            continue
        if obj.type == "EMPTY":
            if obj.instance_type != "NONE":
                obj.instance_type = "NONE"
                note(f"{obj.name!r}: instancing turned off")
            continue
        if obj.type in ("CAMERA", "ARMATURE", "LATTICE"):
            continue
        if obj.type == "LIGHT":
            light = obj.data
            if light.type not in ("POINT", "SUN", "SPOT", "AREA"):
                obj.hide_render = obj.hide_viewport = True
                audit["objects_hidden"].append(obj.name)
                continue
            if light.use_nodes:
                light.use_nodes = False
                note(f"light {obj.name!r}: nodes turned off")
            light.use_shadow = True
            if hasattr(light, "cycles") and hasattr(light.cycles, "cast_shadow"):
                light.cycles.cast_shadow = True
            if hasattr(obj, "light_linking"):
                obj.light_linking.receiver_collection = None
                obj.light_linking.blocker_collection = None
            if light.energy < 0:
                light.energy = 0.0
                note(f"light {obj.name!r}: negative power set to zero")
            if not obj.hide_render:
                lights.append(obj)
            continue
        if obj.type != "MESH":
            if not obj.hide_render:
                audit["objects_hidden"].append(obj.name)
            obj.hide_render = obj.hide_viewport = True
            continue
        for mod in list(obj.modifiers):
            kind = mod.type
            if kind in ("NODES", "PARTICLE_SYSTEM", "PARTICLE_INSTANCE"):
                obj.modifiers.remove(mod)
                note(f"{obj.name!r}: modifier {kind} removed")
                continue
            mod.show_viewport = mod.show_render
            if hasattr(mod, "render_levels") and hasattr(mod, "levels"):
                mod.levels = mod.render_levels
        if obj.instance_type != "NONE":
            obj.instance_type = "NONE"
            note(f"{obj.name!r}: instancing turned off")
        for attr in ("visible_camera", "visible_diffuse", "visible_glossy", "visible_transmission",
                     "visible_volume_scatter", "visible_shadow"):
            if hasattr(obj, attr) and not getattr(obj, attr):
                setattr(obj, attr, True)
                note(f"{obj.name!r}: {attr} turned on")
        if getattr(obj, "is_holdout", False):
            obj.is_holdout = False
            note(f"{obj.name!r}: holdout turned off")
        if getattr(obj, "is_shadow_catcher", False):
            obj.is_shadow_catcher = False
            note(f"{obj.name!r}: shadow catcher turned off")
        if hasattr(obj, "light_linking"):
            obj.light_linking.receiver_collection = None
            obj.light_linking.blocker_collection = None
        if not obj.hide_render:
            meshes.append(obj)
    if len(lights) > MAX_LIGHTS:
        lights.sort(key=lambda o: -float(o.data.energy))
        for obj in lights[MAX_LIGHTS:]:
            obj.hide_render = obj.hide_viewport = True
            audit["objects_hidden"].append(obj.name)
        note(f"{len(lights) - MAX_LIGHTS} lights beyond the {MAX_LIGHTS} strongest hidden")
        lights = lights[:MAX_LIGHTS]
    return meshes, lights


def set_camera(scene, job: dict):
    """The scene's camera, its scale and mirroring dropped; returns it and its intrinsics in pixels
    at the render size (fx, fy, cx, cy; Blender's sensor fit and shift, square pixels)."""
    cam = scene.camera
    if cam is None or cam.type != "CAMERA":
        raise RuntimeError("the scene has no camera")
    loc, rot, scale = cam.matrix_world.decompose()
    if (scale - Vector((1.0, 1.0, 1.0))).length > 1e-6 or cam.matrix_world.determinant() < 0:
        note(f"camera scale {tuple(round(x, 4) for x in scale)} dropped")
    cam.matrix_world = Matrix.Translation(loc) @ rot.to_matrix().to_4x4()   # no scale, no mirror
    c = cam.data
    if c.type != "PERSP":
        raise RuntimeError(f"the camera is {c.type}, not a perspective camera")
    W, H = job["width"], job["height"]
    fit = c.sensor_fit
    if fit == "AUTO":
        fit, sensor = ("HORIZONTAL" if W >= H else "VERTICAL"), c.sensor_width
    else:
        sensor = c.sensor_width if fit == "HORIZONTAL" else c.sensor_height
    view = W if fit == "HORIZONTAL" else H        # pixels the sensor size spans
    if c.lens <= 0 or sensor <= 0:
        raise RuntimeError("the camera has no focal length")
    f = c.lens / sensor * view
    intr = {"width": W, "height": H, "fx": f, "fy": f,
            "cx": W / 2 - c.shift_x * view, "cy": H / 2 + c.shift_y * view}
    fov = math.degrees(2 * math.atan(W / 2 / f))
    if not FOV_RANGE[0] <= fov <= FOV_RANGE[1]:
        raise RuntimeError(f"the camera's horizontal field of view is {fov:.1f} degrees "
                           f"(allowed {FOV_RANGE[0]:g} to {FOV_RANGE[1]:g})")
    c.clip_start, c.clip_end = 1e-3, 1e5
    if c.dof.use_dof:
        note("depth of field turned off")
    c.dof.use_dof = False
    return cam, intr


def set_render(scene, job: dict, device: str) -> None:
    r = scene.render
    r.engine = "CYCLES"
    r.resolution_x, r.resolution_y, r.resolution_percentage = job["width"], job["height"], 100
    r.pixel_aspect_x = r.pixel_aspect_y = 1.0
    r.use_border = r.use_crop_to_border = False
    r.film_transparent = False
    r.use_compositing = r.use_sequencer = False
    r.use_motion_blur = False
    r.use_freestyle = False
    r.use_simplify = False
    r.use_multiview = False
    r.threads_mode, r.threads = "FIXED", job.get("threads", 16)
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    r.image_settings.color_depth = "8"
    cy = scene.cycles
    cy.feature_set = "SUPPORTED"
    cy.samples = job["samples"]
    cy.use_adaptive_sampling = False
    cy.seed, cy.use_animated_seed = 0, False
    cy.use_denoising, cy.denoiser = True, "OPENIMAGEDENOISE"
    cy.denoising_use_gpu = False
    cy.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    cy.denoising_prefilter = "ACCURATE"
    cy.max_bounces, cy.diffuse_bounces, cy.glossy_bounces = 12, 4, 4
    cy.transmission_bounces, cy.volume_bounces, cy.transparent_max_bounces = 12, 0, 8
    cy.sample_clamp_direct, cy.sample_clamp_indirect = 0.0, 10.0
    cy.blur_glossy = 1.0
    cy.caustics_reflective = cy.caustics_refractive = True
    cy.film_exposure = 1.0
    cy.pixel_filter_type, cy.filter_width = "BLACKMAN_HARRIS", 1.5
    cy.use_light_tree = True
    cy.light_sampling_threshold = 0.01
    if hasattr(cy, "use_guiding"):
        cy.use_guiding = False
    cy.device = "GPU" if device != "CPU" else "CPU"
    if device != "CPU":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = device
        prefs.refresh_devices()
        found = False
        for d in prefs.devices:
            d.use = d.type == device
            found |= d.use
        if not found:
            cy.device = "CPU"
            note(f"no {device} device: rendered on the CPU")
    vs = scene.view_settings
    scene.display_settings.display_device = "sRGB"
    if vs.view_transform not in VIEW_TRANSFORMS:
        note(f"view transform {vs.view_transform!r} replaced by Standard")
        vs.view_transform = "Standard"
    vs.look = "None"
    vs.exposure = min(max(vs.exposure, -10.0), 10.0)
    vs.gamma = 1.0
    vs.use_curve_mapping = False
    if hasattr(vs, "use_white_balance"):
        vs.use_white_balance = False
    scene.sequencer_colorspace_settings.name = "sRGB"


# --------------------------------------------------------------------------- export
def export(scene, cam, path: Path) -> int:
    dg = bpy.context.evaluated_depsgraph_get()
    verts, faces, owner, names = [], [], [], []
    base = 0
    for inst in dg.object_instances:
        ob = inst.object
        if ob.type != "MESH" or ob.original.hide_render or not inst.show_self:
            continue
        mesh = ob.to_mesh()
        try:
            mesh.calc_loop_triangles()
            n, t = len(mesh.vertices), len(mesh.loop_triangles)
            if n == 0 or t == 0:
                continue
            co = np.empty(n * 3, np.float32)
            mesh.vertices.foreach_get("co", co)
            tri = np.empty(t * 3, np.int32)
            mesh.loop_triangles.foreach_get("vertices", tri)
        finally:
            ob.to_mesh_clear()
        M = np.array(inst.matrix_world, np.float64)
        co = co.reshape(-1, 3).astype(np.float64) @ M[:3, :3].T + M[:3, 3]
        verts.append(co.astype(np.float32))
        faces.append(tri.reshape(-1, 3).astype(np.int64) + base)
        owner.append(np.full(t, len(names), np.int32))
        names.append(ob.original.name)
        base += n
        if base > 4 * MAX_TRIANGLES:
            break
    tris = sum(len(f) for f in faces)
    c2w = np.array(cam.matrix_world, np.float64)
    np.savez(path, vertices=np.concatenate(verts) if verts else np.zeros((0, 3), np.float32),
             faces=np.concatenate(faces).astype(np.int32) if faces else np.zeros((0, 3), np.int32),
             face_object=np.concatenate(owner) if owner else np.zeros(0, np.int32),
             names=np.array(names), c2w=c2w)
    return tris


def orbit_matrix(c2w: Matrix, pivot_dist: float, degrees: float) -> Matrix:
    """The camera turned by `degrees` about its own up axis through the point pivot_dist ahead."""
    rot = c2w.to_3x3()
    pivot = c2w.to_translation() - (rot @ Vector((0.0, 0.0, 1.0))) * pivot_dist
    R = Matrix.Rotation(math.radians(degrees), 4, rot @ Vector((0.0, 1.0, 0.0)))
    return Matrix.Translation(pivot) @ R @ Matrix.Translation(-pivot) @ c2w


def median_depth(path: Path, c2w: np.ndarray, job: dict) -> float:
    """Median camera depth of the exported vertices in view (the orbit views' pivot)."""
    g = np.load(path)
    v = g["vertices"].astype(np.float64)
    if not len(v):
        return 3.0
    rel = (v - c2w[:3, 3]) @ c2w[:3, :3]          # camera frame, OpenGL: z backwards
    z = -rel[:, 2]
    ok = z > 1e-3
    u = job["intr"]["fx"] * rel[ok, 0] / z[ok] + job["intr"]["cx"]
    w = -job["intr"]["fy"] * rel[ok, 1] / z[ok] + job["intr"]["cy"]
    inside = (u >= 0) & (u < job["width"]) & (w >= 0) & (w < job["height"])
    return float(np.median(z[ok][inside])) if inside.any() else 3.0


def main() -> None:
    job = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())
    out = Path(job["out_dir"])
    t0 = time.time()
    bpy.ops.wm.open_mainfile(filepath=job["blend"], load_ui=False, use_scripts=False)
    if bpy.data.libraries:
        raise RuntimeError("the scene links other .blend files")
    scene = bpy.context.scene
    meshes, lights = check_objects(scene)
    check_materials(meshes)
    check_world(scene)
    cam, intr = set_camera(scene, job)
    job["intr"] = intr
    (out / "camera.json").write_text(json.dumps(intr))
    set_render(scene, job, job.get("device", "CUDA"))
    bpy.context.view_layer.update()
    tris = export(scene, cam, out / "geometry.npz")
    audit.update(meshes=len(meshes), lights=len(lights), triangles=tris,
                 view_transform=scene.view_settings.view_transform,
                 exposure=scene.view_settings.exposure, load_s=round(time.time() - t0, 2))
    if tris > MAX_TRIANGLES:
        raise RuntimeError(f"{tris} triangles (at most {MAX_TRIANGLES})")
    t1 = time.time()
    scene.render.filepath = str(out / "render.png")
    bpy.ops.render.render(write_still=True)
    audit["render_s"] = round(time.time() - t1, 2)
    if job.get("orbits"):
        c2w = cam.matrix_world.copy()
        scene.cycles.samples = job.get("orbit_samples", job["samples"])
        pivot = median_depth(out / "geometry.npz", np.array(c2w, np.float64), job)
        for deg in job["orbits"]:
            cam.matrix_world = orbit_matrix(c2w, pivot, deg)
            scene.render.filepath = str(out / f"orbit_{deg:+.0f}.png")
            bpy.ops.render.render(write_still=True)
        cam.matrix_world = c2w
    audit["total_s"] = round(time.time() - t0, 2)
    (out / "audit.json").write_text(json.dumps(audit, indent=1))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - reported to the evaluator
        import traceback
        traceback.print_exc()
        job = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())
        (Path(job["out_dir"]) / "error.txt").write_text(f"{type(exc).__name__}: {exc}"[:1000])
        sys.exit(3)
