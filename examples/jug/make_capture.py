"""Author the demo object and render it as a hand-held phone orbit video, headless in Blender.

    blender -b --factory-startup --python-exit-code 1 -P make_capture.py -- OUT_DIR [--frames N] [--size WxH] [--samples S]

Writes into OUT_DIR:
  frames/0001.png ...    the rendered frames (encode them with ffmpeg into capture.mp4)
  ground_truth.ply       the jug only, in metres, world coordinates (Z up), for measuring the reconstruction
  cameras_gt.json        the true camera centre of every frame, for aligning the reconstruction before measuring

Everything here is made in code for this repository: geometry, procedural materials and lights. No
third-party assets are used. The jug is about 0.19 m tall and stands on a wooden table top."""
import json
import math
import os
import sys

import bpy
import bmesh
from mathutils import Euler, Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if not argv:
    raise SystemExit("usage: blender -b -P make_capture.py -- OUT_DIR [--frames N] [--size WxH] [--samples S]")
OUT = os.path.abspath(argv[0])
opts = {argv[i]: argv[i + 1] for i in range(1, len(argv) - 1, 2)}
FRAMES = int(opts.get("--frames", 264))
W, H = (int(v) for v in opts.get("--size", "1280x720").split("x"))
SAMPLES = int(opts.get("--samples", 12))
FPS = 24
os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


# ---------- materials (procedural, so there is real texture for feature matching) ----------
def node_mat(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat, nt, bsdf


def glaze_material():
    """Painted ceramic: a cream glaze with blue bands, a speckle, and a scatter of terracotta dots."""
    mat, nt, bsdf = node_mat("glaze")
    coord = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(coord.outputs["Object"], sep.inputs["Vector"])
    # bands by height, wobbled by noise so every band edge is unique
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 9.0
    noise.inputs["Detail"].default_value = 6.0
    nt.links.new(coord.outputs["Object"], noise.inputs["Vector"])
    wob = nt.nodes.new("ShaderNodeMath"); wob.operation = "MULTIPLY_ADD"
    wob.inputs[1].default_value = 0.025
    nt.links.new(noise.outputs["Fac"], wob.inputs[0])
    nt.links.new(sep.outputs["Z"], wob.inputs[2])
    wave = nt.nodes.new("ShaderNodeMath"); wave.operation = "SINE"
    mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; mul.inputs[1].default_value = 95.0
    nt.links.new(wob.outputs[0], mul.inputs[0]); nt.links.new(mul.outputs[0], wave.inputs[0])
    band = nt.nodes.new("ShaderNodeMapRange")
    band.inputs["From Min"].default_value = 0.55; band.inputs["From Max"].default_value = 0.75
    nt.links.new(wave.outputs[0], band.inputs["Value"])
    cream = (0.86, 0.80, 0.66, 1); blue = (0.06, 0.16, 0.42, 1)
    mix1 = nt.nodes.new("ShaderNodeMix"); mix1.data_type = "RGBA"
    mix1.inputs[6].default_value = cream; mix1.inputs[7].default_value = blue
    nt.links.new(band.outputs["Result"], mix1.inputs[0])
    # voronoi dots in terracotta
    vor = nt.nodes.new("ShaderNodeTexVoronoi"); vor.feature = "DISTANCE_TO_EDGE"
    vor.inputs["Scale"].default_value = 26.0
    nt.links.new(coord.outputs["Object"], vor.inputs["Vector"])
    dots = nt.nodes.new("ShaderNodeMapRange")
    dots.inputs["From Min"].default_value = 0.02; dots.inputs["From Max"].default_value = 0.035
    dots.inputs["To Min"].default_value = 1.0; dots.inputs["To Max"].default_value = 0.0
    nt.links.new(vor.outputs["Distance"], dots.inputs["Value"])
    mix2 = nt.nodes.new("ShaderNodeMix"); mix2.data_type = "RGBA"
    mix2.inputs[7].default_value = (0.55, 0.18, 0.08, 1)
    nt.links.new(mix1.outputs[2], mix2.inputs[6]); nt.links.new(dots.outputs["Result"], mix2.inputs[0])
    # fine speckle
    sp = nt.nodes.new("ShaderNodeTexNoise"); sp.inputs["Scale"].default_value = 160.0
    nt.links.new(coord.outputs["Object"], sp.inputs["Vector"])
    spr = nt.nodes.new("ShaderNodeMapRange")
    spr.inputs["From Min"].default_value = 0.62; spr.inputs["From Max"].default_value = 0.70
    spr.inputs["To Min"].default_value = 0.0; spr.inputs["To Max"].default_value = 0.8
    nt.links.new(sp.outputs["Fac"], spr.inputs["Value"])
    mix3 = nt.nodes.new("ShaderNodeMix"); mix3.data_type = "RGBA"
    mix3.inputs[7].default_value = (0.12, 0.09, 0.07, 1)
    nt.links.new(mix2.outputs[2], mix3.inputs[6]); nt.links.new(spr.outputs["Result"], mix3.inputs[0])
    nt.links.new(mix3.outputs[2], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.45
    return mat


def wood_material():
    mat, nt, bsdf = node_mat("table")
    coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1.0, 14.0, 1.0)
    nt.links.new(coord.outputs["Object"], mapping.inputs["Vector"])
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 6.0; noise.inputs["Detail"].default_value = 12.0
    noise.inputs["Distortion"].default_value = 3.0
    nt.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.10, 0.045, 0.02, 1)
    ramp.color_ramp.elements[1].color = (0.52, 0.32, 0.16, 1)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.6
    return mat


def wall_material():
    mat, nt, bsdf = node_mat("wall")
    noise = nt.nodes.new("ShaderNodeTexNoise"); noise.inputs["Scale"].default_value = 3.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.55, 0.56, 0.58, 1)
    ramp.color_ramp.elements[1].color = (0.75, 0.74, 0.70, 1)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.9
    return mat


# ---------- the jug: a lathed body, a pinched spout and a handle ----------
PROFILE = [  # (radius, height) in metres, from the foot up; a pitcher-like curve
    (0.000, 0.000), (0.046, 0.000), (0.050, 0.006), (0.058, 0.030), (0.066, 0.065), (0.067, 0.090),
    (0.062, 0.115), (0.050, 0.140), (0.042, 0.158), (0.043, 0.172), (0.049, 0.186), (0.052, 0.190),
    (0.046, 0.190), (0.040, 0.176),
]
SEG = 128


def build_jug():
    bm = bmesh.new()
    rings = []
    for r, z in PROFILE:
        ring = []
        for k in range(SEG):
            a = 2 * math.pi * k / SEG
            rr = r
            # pinch a spout into the rim, pointing along +X
            if z > 0.15 and r > 0:
                pull = max(0.0, math.cos(a)) ** 8
                rr = r + pull * (z - 0.15) * 0.75
                zz = z + pull * (z - 0.15) * 0.25
            else:
                zz = z
            ring.append(bm.verts.new((rr * math.cos(a), rr * math.sin(a), zz)) if r > 0 else None)
        rings.append(ring)
    centre = bm.verts.new((0, 0, 0))
    for i in range(1, len(rings) - 1):
        lo, hi = rings[i], rings[i + 1]
        for k in range(SEG):
            k2 = (k + 1) % SEG
            bm.faces.new((lo[k], lo[k2], hi[k2], hi[k]))
    for k in range(SEG):
        bm.faces.new((centre, rings[1][(k + 1) % SEG], rings[1][k]))
    # close the inside with a floor so the mesh is watertight-ish for the reference
    top = rings[-1]
    inner_floor = bm.verts.new((0, 0, 0.176))
    for k in range(SEG):
        bm.faces.new((inner_floor, top[k], top[(k + 1) % SEG]))
    me = bpy.data.meshes.new("jug")
    bm.normal_update()
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    body = bpy.data.objects.new("jug", me)
    scene.collection.objects.link(body)
    mod = body.modifiers.new("smooth", "SUBSURF"); mod.levels = 1; mod.render_levels = 1
    for p in me.polygons:
        p.use_smooth = True

    # handle: a bevelled curve on the side opposite the spout
    cu = bpy.data.curves.new("handle", "CURVE"); cu.dimensions = "3D"
    sp = cu.splines.new("BEZIER"); sp.bezier_points.add(2)
    pts = [(-0.060, 0, 0.150), (-0.112, 0, 0.115), (-0.064, 0, 0.055)]
    for bp, co in zip(sp.bezier_points, pts):
        bp.co = co; bp.handle_left_type = bp.handle_right_type = "AUTO"
    cu.bevel_depth = 0.009; cu.bevel_resolution = 6; cu.use_fill_caps = True
    handle = bpy.data.objects.new("handle", cu)
    handle.scale = (1.0, 1.6, 1.0)  # a flat strap, wider than deep
    scene.collection.objects.link(handle)

    glaze = glaze_material()
    for ob in (body, handle):
        ob.data.materials.append(glaze)
    return [body, handle]


jug_parts = build_jug()

# table, back walls
bpy.ops.mesh.primitive_plane_add(size=1.6, location=(0, 0, 0))
table = bpy.context.active_object; table.data.materials.append(wood_material())
for loc, rot in (((0, 1.4, 0.6), (math.radians(90), 0, 0)), ((1.4, 0, 0.6), (math.radians(90), 0, math.radians(90)))):
    bpy.ops.mesh.primitive_plane_add(size=3.0, location=loc, rotation=rot)
    bpy.context.active_object.data.materials.append(wall_material())

# lights: a big soft window light, a fill, a dim warm room
bpy.ops.object.light_add(type="AREA", location=(0.9, -0.6, 1.2))
win = bpy.context.active_object; win.data.energy = 90; win.data.size = 1.2
win.rotation_euler = (Vector((0, 0, 0.1)) - win.location).to_track_quat("-Z", "Y").to_euler()
bpy.ops.object.light_add(type="AREA", location=(-1.0, 0.8, 0.9))
fill = bpy.context.active_object; fill.data.energy = 45; fill.data.size = 1.5
fill.rotation_euler = (Vector((0, 0, 0.1)) - fill.location).to_track_quat("-Z", "Y").to_euler()
world = bpy.data.worlds.new("room"); scene.world = world
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.42, 0.40, 0.37, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.35

# ---------- the camera: a hand-held orbit, a 26 mm equivalent phone lens ----------
cam_data = bpy.data.cameras.new("phone")
cam_data.sensor_fit = "HORIZONTAL"; cam_data.sensor_width = 36.0; cam_data.lens = 26.0
cam = bpy.data.objects.new("phone", cam_data); scene.collection.objects.link(cam); scene.camera = cam


def shake(t, seed):
    """Smooth hand tremor: a few incommensurate sines, deterministic."""
    return sum(math.sin(t * f + seed * p) * a for f, p, a in ((1.3, 1.7, 0.5), (2.9, 2.3, 0.3), (6.1, 0.7, 0.2)))


def camera_pose(i):
    u = i / max(FRAMES - 1, 1)
    t = i / FPS
    turn = 1.15 * 2 * math.pi * u + 0.12 * math.sin(2.1 * t)      # walking pace is not constant
    elev = math.radians(18 + 26 * u + 3 * shake(t, 1))            # spirals up as the person walks round
    dist = 0.50 + 0.04 * math.sin(1.1 * t) + 0.01 * shake(t, 2)
    target = Vector((0.004 * shake(t, 3), 0.004 * shake(t, 4), 0.085 + 0.006 * shake(t, 5)))
    pos = target + Vector((dist * math.cos(elev) * math.cos(turn), dist * math.cos(elev) * math.sin(turn), dist * math.sin(elev)))
    look = (target - pos).to_track_quat("-Z", "Y").to_euler()
    roll = math.radians(2.5 * shake(t, 6))
    rot = look.to_matrix() @ Euler((0, 0, roll)).to_matrix()
    # a few quick jerks of the hand: these frames come out motion-blurred, and frame selection should skip them
    jerk = sum(math.exp(-((i - c) / 2.0) ** 2) for c in (40, 110, 175, 230)) * math.radians(4) * math.sin(9 * t)
    rot = rot @ Euler((jerk, jerk * 0.5, 0)).to_matrix()
    return pos, rot


cams = []
for i in range(FRAMES):
    pos, rot = camera_pose(i)
    cam.matrix_world = Matrix.Translation(pos) @ rot.to_4x4()
    cam.keyframe_insert("location", frame=i + 1)
    cam.keyframe_insert("rotation_euler", frame=i + 1)
    cams.append({"frame": i + 1, "centre": list(pos)})

scene.frame_start, scene.frame_end = 1, FRAMES
scene.render.fps = FPS
scene.render.resolution_x, scene.render.resolution_y = W, H
scene.render.resolution_percentage = 100
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True
scene.render.threads_mode = "FIXED"; scene.render.threads = 8
scene.render.use_motion_blur = True
scene.render.motion_blur_shutter = 0.5
scene.view_settings.view_transform = "Standard"
scene.view_settings.exposure = -0.6
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = os.path.join(OUT, "frames", "")

# ground truth: the jug only, modifiers applied, world coordinates
deps = bpy.context.evaluated_depsgraph_get()
verts, faces = [], []
for ob in jug_parts:
    ev = ob.evaluated_get(deps)
    me = ev.to_mesh()
    base = len(verts)
    verts += [tuple(ob.matrix_world @ v.co) for v in me.vertices]
    for p in me.polygons:
        idx = [base + v for v in p.vertices]
        for k in range(1, len(idx) - 1):
            faces.append((idx[0], idx[k], idx[k + 1]))
    ev.to_mesh_clear()
with open(os.path.join(OUT, "ground_truth.ply"), "w") as fh:
    fh.write(f"ply\nformat ascii 1.0\ncomment scan-studio demo jug, metres, Z up\nelement vertex {len(verts)}\n"
             "property float x\nproperty float y\nproperty float z\n"
             f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
    fh.writelines(f"{x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in verts)
    fh.writelines(f"3 {a} {b} {c}\n" for a, b, c in faces)
with open(os.path.join(OUT, "cameras_gt.json"), "w") as fh:
    json.dump({"fps": FPS, "width": W, "height": H, "lens_mm": 26.0, "sensor_mm": 36.0, "up": [0, 0, 1],
               "cameras": cams}, fh)

if "--no-render" not in argv:
    # --start/--end render a part of the orbit (to split the work or to try the look on one frame)
    scene.frame_start = int(opts.get("--start", 1))
    scene.frame_end = int(opts.get("--end", FRAMES))
    bpy.ops.render.render(animation=True)
