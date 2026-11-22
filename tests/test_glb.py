import io
import json
import struct

import numpy as np
import pytest
import trimesh

from scan_studio import glb
from scan_studio.errors import ScanError


def cube():
    m = trimesh.creation.box(extents=[0.1, 0.2, 0.3])
    return np.asarray(m.vertices), np.asarray(m.faces)


def test_vertex_colour_glb_is_valid_and_reads_back():
    v, f = cube()
    data = glb.build(v, f, vertex_colors=np.tile([[0.2, 0.4, 0.6]], (len(v), 1)), name="box")
    st = glb.validate(data)
    assert st["triangles"] == len(f) and st["vertices"] == len(v) and st["textures"] == 0
    assert np.allclose(np.array(st["bounds_max"]) - st["bounds_min"], [0.1, 0.2, 0.3], atol=1e-6)
    back = trimesh.load(io.BytesIO(data), file_type="glb", force="mesh")
    assert len(back.faces) == len(f)


def test_textured_glb_has_a_texture_and_uvs():
    v, f = cube()
    uv = np.random.default_rng(0).random((len(v), 2))
    tex = np.zeros((64, 64, 3), np.uint8); tex[..., 0] = 200
    data = glb.build(v, f, uv=uv, texture=tex)
    st = glb.validate(data)
    assert st["textures"] == 1 and st["materials"] == 1
    js = json.loads(data[20:20 + struct.unpack_from("<I", data, 12)[0]])
    assert "TEXCOORD_0" in js["meshes"][0]["primitives"][0]["attributes"]
    assert js["images"][0]["mimeType"] == "image/jpeg"


def test_load_mesh_round_trip():
    v, f = cube()
    m = glb.load_mesh(glb.build(v, f))
    assert len(m.faces) == len(f)


def test_empty_model_is_refused():
    with pytest.raises(ScanError):
        glb.build(np.zeros((0, 3)), np.zeros((0, 3), int))


def valid():
    v, f = cube()
    return bytearray(glb.build(v, f))


def test_bad_magic_version_and_length():
    d = valid(); d[0:4] = b"gltf"
    with pytest.raises(ScanError, match="magic"):
        glb.validate(bytes(d))
    d = valid(); struct.pack_into("<I", d, 4, 1)
    with pytest.raises(ScanError, match="version"):
        glb.validate(bytes(d))
    d = valid() + b"\0\0\0\0"
    with pytest.raises(ScanError, match="header says"):
        glb.validate(bytes(d))
    with pytest.raises(ScanError):
        glb.validate(b"glTF")
    with pytest.raises(ScanError):
        glb.validate("not bytes")


def test_chunk_running_past_the_end():
    d = valid()
    struct.pack_into("<I", d, 12, len(d) * 2)
    with pytest.raises(ScanError, match="chunk"):
        glb.validate(bytes(d))


def rewrite_json(data: bytearray, change) -> bytes:
    jlen = struct.unpack_from("<I", data, 12)[0]
    js = json.loads(bytes(data[20:20 + jlen]))
    change(js)
    raw = json.dumps(js).encode()
    raw += b" " * (-len(raw) % 4)
    rest = bytes(data[20 + jlen:])
    body = struct.pack("<II", len(raw), 0x4E4F534A) + raw + rest
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def test_accessor_reading_past_its_view_is_refused():
    def grow(js):
        js["accessors"][0]["count"] = 10 ** 6
    with pytest.raises(ScanError, match="accessor"):
        glb.validate(rewrite_json(valid(), grow))


def test_buffer_view_outside_binary_is_refused():
    def move(js):
        js["bufferViews"][0]["byteOffset"] = 10 ** 7
    with pytest.raises(ScanError, match="bufferView"):
        glb.validate(rewrite_json(valid(), move))


def test_wrong_asset_version_is_refused():
    def old(js):
        js["asset"]["version"] = "1.0"
    with pytest.raises(ScanError, match="asset"):
        glb.validate(rewrite_json(valid(), old))
