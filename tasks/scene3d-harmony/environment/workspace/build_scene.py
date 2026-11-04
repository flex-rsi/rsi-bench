"""Runs inside Blender (blender -b --factory-startup --python build_scene.py -- spec.json): builds
the starter's room box from spec.json and saves it as a .blend.

spec.json: camera (width, height, fx, fy, cx, cy: the starter's guess), colors ({face: linear RGB
base color}), out (the .blend path). The box and the camera are those of reconstruct.py (z up, the
camera looks along +y); the room's faces are named floor, ceiling and wall_*.
"""

import json
import math
import sys

import bpy

spec = json.loads(open(sys.argv[sys.argv.index("--") + 1]).read())
cam, colors = spec["camera"], spec["colors"]
lo, hi = spec["box"]
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


def material(name, rgb):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*rgb, 1.0)
    p.inputs["Roughness"].default_value = 0.8
    return m


# one quad per face of the box, facing into the room
X, Y, Z = (lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])
QUADS = {  # corners counter-clockwise seen from inside the box
    "floor": [(X[0], Y[0], Z[0]), (X[1], Y[0], Z[0]), (X[1], Y[1], Z[0]), (X[0], Y[1], Z[0])],
    "ceiling": [(X[0], Y[0], Z[1]), (X[0], Y[1], Z[1]), (X[1], Y[1], Z[1]), (X[1], Y[0], Z[1])],
    "wall_left": [(X[0], Y[0], Z[0]), (X[0], Y[1], Z[0]), (X[0], Y[1], Z[1]), (X[0], Y[0], Z[1])],
    "wall_right": [(X[1], Y[0], Z[0]), (X[1], Y[0], Z[1]), (X[1], Y[1], Z[1]), (X[1], Y[1], Z[0])],
    "wall_back": [(X[0], Y[1], Z[0]), (X[1], Y[1], Z[0]), (X[1], Y[1], Z[1]), (X[0], Y[1], Z[1])],
    "wall_front": [(X[0], Y[0], Z[0]), (X[0], Y[0], Z[1]), (X[1], Y[0], Z[1]), (X[1], Y[0], Z[0])],
}
for name, quad in QUADS.items():
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(quad, [], [(0, 1, 2, 3)])
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    obj.data.materials.append(material(name, colors[name]))

light = bpy.data.lights.new("ceiling_light", "AREA")
light.size, light.energy = spec["light_size"], spec["light_energy"]
lamp = bpy.data.objects.new("ceiling_light", light)
lamp.location = ((X[0] + X[1]) / 2, spec["light_y"], Z[1] - 0.01)
scene.collection.objects.link(lamp)

c = bpy.data.cameras.new("camera")
c.sensor_fit, c.sensor_width = "HORIZONTAL", 36.0
c.lens = cam["fx"] * 36.0 / cam["width"]
camera = bpy.data.objects.new("camera", c)
camera.location = spec["camera_position"]
camera.rotation_euler = (math.radians(90.0), 0.0, 0.0)
scene.collection.objects.link(camera)
scene.camera = camera
scene.render.resolution_x, scene.render.resolution_y = cam["width"], cam["height"]
world = bpy.data.worlds.new("world")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
scene.world = world
bpy.ops.wm.save_as_mainfile(filepath=spec["out"], compress=True)
