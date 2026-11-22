"""Write the final model as GLB (binary glTF 2.0) and check any GLB without trusting it."""
from __future__ import annotations

import io
import json
import struct

import numpy as np

from .errors import ScanError

MAX_GLB_BYTES = 512 * 1024 ** 2
_COMPONENT_SIZE = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
_TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


def build(vertices, faces, *, uv=None, texture=None, vertex_colors=None, name: str = "scan") -> bytes:
    """GLB bytes for one mesh: textured (uv + texture) or with vertex colours. Vertices are in the output frame."""
    import trimesh
    from PIL import Image
    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64)
    if len(v) == 0 or len(f) == 0:
        raise ScanError("the model is empty")
    if uv is not None and texture is not None:
        # JPEG: a photographic texture is several times smaller than PNG, and every glTF viewer reads it
        buf = io.BytesIO()
        Image.fromarray(np.asarray(texture, np.uint8)).save(buf, format="JPEG", quality=90)
        buf.seek(0)
        material = trimesh.visual.material.PBRMaterial(
            name=f"{name}-material", baseColorTexture=Image.open(buf),
            metallicFactor=0.0, roughnessFactor=0.85, doubleSided=False)
        visual = trimesh.visual.TextureVisuals(uv=np.asarray(uv, np.float64), material=material)
        mesh = trimesh.Trimesh(v, f, visual=visual, process=False)
    else:
        cols = None
        if vertex_colors is not None:
            c = np.asarray(vertex_colors, float)
            c = (np.clip(c, 0, 1) * 255).round() if c.max() <= 1.0 else np.clip(c, 0, 255)
            cols = np.concatenate([c, np.full((len(c), 1), 255)], 1).astype(np.uint8)
        mesh = trimesh.Trimesh(v, f, vertex_colors=cols, process=False)
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name=name, geom_name=name)
    data = scene.export(file_type="glb")
    validate(data)
    return data


def validate(data: bytes) -> dict:
    """Check a GLB; any malformed structure (wrong types included) is a ScanError."""
    try:
        return _validate(data)
    except (KeyError, TypeError, IndexError, AttributeError, ValueError, struct.error) as e:
        raise ScanError(f"GLB: malformed ({e.__class__.__name__})") from None


def _validate(data: bytes) -> dict:
    """Parse a GLB's container and JSON, and check every accessor fits in its buffer. Returns stats.
    Raises ScanError on anything malformed. Does not decode images."""
    if not isinstance(data, (bytes, bytearray)):
        raise ScanError("GLB: not bytes")
    if len(data) > MAX_GLB_BYTES:
        raise ScanError("GLB: file too large to check")
    if len(data) < 20:
        raise ScanError("GLB: file too short")
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ScanError("GLB: wrong magic (not a binary glTF file)")
    if version != 2:
        raise ScanError(f"GLB: version {version}, expected 2")
    if length != len(data):
        raise ScanError(f"GLB: header says {length} bytes, file has {len(data)}")
    off = 12
    chunks = []
    while off < length:
        if off + 8 > length:
            raise ScanError("GLB: truncated chunk header")
        clen, ctype = struct.unpack_from("<II", data, off)
        off += 8
        if clen % 4 or off + clen > length:
            raise ScanError("GLB: chunk length not aligned or past the end")
        chunks.append((ctype, bytes(data[off:off + clen])))
        off += clen
    if not chunks or chunks[0][0] != 0x4E4F534A:
        raise ScanError("GLB: first chunk is not JSON")
    try:
        gltf = json.loads(chunks[0][1].decode("utf-8").rstrip(" \x00"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ScanError(f"GLB: JSON chunk does not parse ({e.__class__.__name__})") from None
    if not isinstance(gltf, dict) or str(gltf.get("asset", {}).get("version")) != "2.0":
        raise ScanError("GLB: asset.version is not 2.0")
    binary = chunks[1][1] if len(chunks) > 1 and chunks[1][0] == 0x004E4942 else b""
    buffers = gltf.get("buffers", [])
    views = gltf.get("bufferViews", [])
    for bv in views:
        if bv.get("buffer", 0) != 0 or bv.get("byteOffset", 0) + bv.get("byteLength", 0) > len(binary):
            raise ScanError("GLB: a bufferView points outside the binary chunk")
    if buffers and buffers[0].get("byteLength", 0) > len(binary):
        raise ScanError("GLB: buffer longer than the binary chunk")
    accessors = gltf.get("accessors", [])
    for a in accessors:
        if "bufferView" not in a:
            continue
        bv = views[a["bufferView"]] if 0 <= a["bufferView"] < len(views) else None
        if bv is None or a.get("componentType") not in _COMPONENT_SIZE or a.get("type") not in _TYPE_COUNT:
            raise ScanError("GLB: an accessor has a bad bufferView, componentType or type")
        elem = _COMPONENT_SIZE[a["componentType"]] * _TYPE_COUNT[a["type"]]
        stride = bv.get("byteStride", elem)
        need = a.get("byteOffset", 0) + stride * (a.get("count", 0) - 1) + elem if a.get("count", 0) else 0
        if need > bv.get("byteLength", 0):
            raise ScanError("GLB: an accessor reads past its bufferView")
    triangles = 0
    vertices = 0
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            pos = prim.get("attributes", {}).get("POSITION")
            if pos is None or not 0 <= pos < len(accessors):
                raise ScanError("GLB: a primitive has no POSITION")
            acc = accessors[pos]
            vertices += acc.get("count", 0)
            if "min" in acc and "max" in acc:
                lo = np.minimum(lo, acc["min"])
                hi = np.maximum(hi, acc["max"])
            if "indices" in prim:
                triangles += accessors[prim["indices"]].get("count", 0) // 3
            else:
                triangles += acc.get("count", 0) // 3
    if vertices == 0:
        raise ScanError("GLB: no mesh data")
    return {"bytes": len(data), "vertices": int(vertices), "triangles": int(triangles),
            "textures": len(gltf.get("textures", [])), "materials": len(gltf.get("materials", [])),
            "bounds_min": lo.tolist(), "bounds_max": hi.tolist()}


def load_mesh(data: bytes):
    """Read a GLB written by `build` back as one trimesh (for rescaling an earlier result)."""
    import trimesh
    validate(data)
    scene = trimesh.load(io.BytesIO(data), file_type="glb", force="scene")
    meshes = [g for g in scene.dump()] if hasattr(scene, "dump") else list(scene.geometry.values())
    if len(meshes) != 1:
        raise ScanError("expected a GLB with one mesh (as scan-studio writes it)")
    return meshes[0]
