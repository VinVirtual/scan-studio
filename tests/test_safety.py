"""Path safety, budgets, escaping in the report, and the ar-models hand-off."""
import json
import os
import re

import numpy as np
import pytest
from PIL import Image

from scan_studio import ar, budgets, media, names, report, safeio
from scan_studio.errors import BudgetError, ScanError, UsageError


# ------------------------------------------------------------------ writes
def test_write_is_atomic_and_replaces(tmp):
    p = os.path.join(tmp, "a.txt")
    safeio.write_text(p, "one")
    safeio.write_text(p, "two")
    assert open(p).read() == "two"
    assert [n for n in os.listdir(tmp) if n.endswith(".tmp")] == []


def test_symlinked_output_file_is_refused(tmp):
    target = os.path.join(tmp, "victim.txt")
    open(target, "w").write("keep")
    link = os.path.join(tmp, "model.glb")
    os.symlink(target, link)
    with pytest.raises(safeio.UnsafeOutput):
        safeio.write_bytes(link, b"x")
    assert open(target).read() == "keep"


def test_symlinked_output_folder_is_refused(tmp):
    real = os.path.join(tmp, "real"); os.mkdir(real)
    link = os.path.join(tmp, "out"); os.symlink(real, link)
    with pytest.raises(safeio.UnsafeOutput):
        safeio.ensure_dir(link)
    with pytest.raises(safeio.UnsafeOutput):
        safeio.write_bytes(os.path.join(link, "x.bin"), b"x")


def test_output_folder_that_is_a_file_is_refused(tmp):
    p = os.path.join(tmp, "f"); open(p, "w").close()
    with pytest.raises(safeio.UnsafeOutput):
        safeio.ensure_dir(p)


def test_npz_has_no_pickles(tmp):
    p = safeio.write_npz(os.path.join(tmp, "a.npz"), points=np.zeros((3, 3), np.float32))
    with np.load(p, allow_pickle=False) as z:
        assert z["points"].shape == (3, 3)


# ------------------------------------------------------------------ names and inputs
@pytest.mark.parametrize("bad", ["../x", ".hidden", "a/b", "", "x" * 65, "a b", "a..b", "<script>"])
def test_bad_names_are_refused(bad):
    with pytest.raises(UsageError):
        names.check(bad)


def test_name_from_input_is_safe():
    assert names.from_input("/x/My Jug (final)!.mov") == "My-Jug-final"
    assert names.from_input("/x/<script>.mp4") == "script"
    assert names.from_input("/x/!!!.mp4") == "scan"


def test_urls_and_missing_files_are_refused(tmp):
    for bad in ("https://example.com/a.mp4", "rtsp://h/x.mp4", "concat:a.mp4|b.mp4"):
        with pytest.raises(UsageError):
            media.probe_video(bad)
    with pytest.raises(UsageError):
        media.probe_video(os.path.join(tmp, "missing.mp4"))
    with pytest.raises(UsageError):
        media.probe_video("a\x00b.mp4")


def test_wrong_extension_is_refused(tmp):
    p = os.path.join(tmp, "notes.txt"); open(p, "w").write("x")
    with pytest.raises(UsageError):
        media.probe_video(p)


def test_video_byte_budget(tmp):
    p = os.path.join(tmp, "big.mp4"); open(p, "wb").write(b"\0" * 2048)
    with pytest.raises(BudgetError):
        media.probe_video(p, budgets.with_overrides(input_bytes=1024))


def test_photo_pixel_budget_checked_before_decoding(tmp):
    p = os.path.join(tmp, "a.png")
    Image.new("RGB", (400, 300)).save(p)
    with pytest.raises(BudgetError):
        media.load_photo(p, budgets.with_overrides(photo_pixels=100_000))
    assert media.load_photo(p).shape == (300, 400, 3)


def test_photo_count_and_byte_budgets(tmp):
    for i in range(5):
        Image.new("RGB", (16, 16)).save(os.path.join(tmp, f"{i}.jpg"))
    open(os.path.join(tmp, "notes.txt"), "w").write("ignored")
    assert len(media.list_photos(tmp)) == 5
    with pytest.raises(BudgetError):
        media.list_photos(tmp, budgets.with_overrides(photos=4))
    with pytest.raises(BudgetError):
        media.list_photos(tmp, budgets.with_overrides(input_bytes=100))


def test_photo_symlinks_out_of_the_folder_are_skipped(tmp):
    outside = os.path.join(tmp, "outside.jpg")
    Image.new("RGB", (16, 16)).save(outside)
    folder = os.path.join(tmp, "set"); os.mkdir(folder)
    Image.new("RGB", (16, 16)).save(os.path.join(folder, "a.jpg"))
    os.symlink(outside, os.path.join(folder, "b.jpg"))
    assert [os.path.basename(p) for p in media.list_photos(folder)] == ["a.jpg"]


def test_empty_photo_folder(tmp):
    with pytest.raises(UsageError):
        media.list_photos(tmp)


def test_photos_are_scaled_to_work_size(tmp):
    p = os.path.join(tmp, "wide.jpg")
    Image.new("RGB", (3200, 1800)).save(p)
    assert media.load_photo(p).shape[:2] == (900, 1600)


# ------------------------------------------------------------------ budgets
def test_candidate_plan_caps_total_pixels():
    b = budgets.DEFAULT
    n, w, h = budgets.candidate_plan(100_000, 3840, 2160)
    assert n == b.candidate_frames
    assert n * w * h <= b.candidate_pixels
    assert w % 2 == 0 and h % 2 == 0
    n2, w2, h2 = budgets.candidate_plan(50, 1920, 1080)
    assert n2 == 50 and max(w2, h2) <= b.work_long_side


def test_candidate_plan_refuses_empty_video():
    with pytest.raises(UsageError):
        budgets.candidate_plan(0, 100, 100)


def test_frame_target_bounds():
    assert budgets.frame_target(None) == 90
    assert budgets.frame_target(150) == 150
    for bad in (5, 500, "many"):
        with pytest.raises(UsageError):
            budgets.frame_target(bad)


def test_check_total():
    budgets.check_total("x", 10, 10)
    with pytest.raises(BudgetError):
        budgets.check_total("x", 11, 10)


# ------------------------------------------------------------------ report escaping
def fake_report(name="jug", file="jug.mp4"):
    return {"tool": "scan-studio", "version": "t", "name": name,
            "input": {"kind": "video", "file": file, "candidates": 3},
            "frames": {"scores": [1.0, 2.0, 3.0], "chosen": [0, 2], "blurry": [], "duplicates": [], "thresholds": [0.5] * 3,
                       "times": [0, 1], "names": ["0000.jpg", "0001.jpg"], "sources": []},
            "faces": {"verdict": "clear", "message": "no faces", "frames_checked": 2, "frames_with_faces": [], "boxes": []},
            "sfm": {"registered": 2, "total": 2, "points": 10, "reprojection_error_px": 0.5, "cameras": []},
            "dense": {"method": "sgbm-tsdf", "triangles": 10, "depth_maps": 2},
            "scale": {"method": "none", "units": "unknown"},
            "model": {"file": f"{name}.glb", "size": [0.1, 0.2, 0.1], "triangles": 12, "bytes": 100, "texture": 0},
            "timings": {"stages": [{"stage": "x", "seconds": 1.0}], "total_seconds": 1.0}}


def render(rep):
    return report.render(rep, b"glTF-not-really", {"points": np.zeros((2, 3), np.float32), "colors": np.zeros((2, 3), np.uint8)})


def test_hostile_file_name_is_escaped_everywhere():
    evil = '</script><script>alert(1)</script><img src=x onerror=alert(2)>.mp4'
    page = render(fake_report(file=evil))
    assert "<script>alert(1)" not in page and "<img src=x" not in page
    assert "&lt;/script&gt;" in page
    data = page.split('<script id="scan-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert "<" not in data
    assert json.loads(data)["report"]["input"]["file"] == evil


def test_placeholders_in_input_are_not_expanded():
    page = render(fake_report(file="__JS____CSS____DATA__.mp4"))
    assert page.count("<script>") == 1
    assert "__JS____CSS____DATA__.mp4" in page


def test_csp_hashes_match_inline_script_and_style():
    page = render(fake_report())
    csp = re.search(r'Content-Security-Policy" content="([^"]+)"', page).group(1)
    assert "default-src 'none'" in csp and "unsafe-inline" not in csp and "connect-src data: blob:" in csp
    js = page.split("<script>", 1)[1].rsplit("</script>", 1)[0]
    css = page.split("<style>", 1)[1].split("</style>", 1)[0]
    assert report.csp_hash(js) in csp and report.csp_hash(css) in csp


def test_no_external_urls_in_the_page():
    page = render(fake_report())
    assert not re.search(r'(src|href)\s*=\s*["\']?https?:', page)
    assert "@import" not in page


def test_script_json_escapes_line_separators():
    s = report.script_json({"a": " </script>&"})
    assert " " not in s and "</" not in s and "&" not in s
    assert json.loads(s)["a"] == " </script>&"


def test_nan_becomes_null():
    assert report._clean_json({"a": float("nan"), "b": [float("inf"), 1.0], "c": np.float32(2)}) == \
        {"a": None, "b": [None, 1.0], "c": 2.0}


def test_report_size_budget(tmp):
    with pytest.raises(BudgetError):
        report.write(tmp, fake_report(), b"x" * 1000, {"points": np.zeros((1, 3)), "colors": np.zeros((1, 3))},
                     budgets.with_overrides(report_bytes=1000))


# ------------------------------------------------------------------ ar-models hand-off
def test_ar_command_is_a_plain_argument_list(tmp):
    cmd = ar.command("/opt/ar/ar", os.path.join(tmp, "x; rm -rf ~.glb"), tmp, "jug", 0.2)
    assert cmd[0] == "/opt/ar/ar" and cmd[1] == "convert" and "--height" in cmd and cmd[cmd.index("--height") + 1] == "0.20000"
    assert any(c.endswith("x; rm -rf ~.glb") for c in cmd)


def test_ar_command_refuses_unreal_sizes(tmp):
    for h in (0.0, 0.001, 80):
        with pytest.raises(ScanError):
            ar.command("/opt/ar/ar", "a.glb", tmp, "jug", h)


def test_ar_find_uses_env_and_requires_executable(tmp, monkeypatch):
    monkeypatch.setenv("HOME", tmp)
    monkeypatch.setenv("SCAN_STUDIO_AR_MODELS", tmp)
    assert ar.find() is None
    exe = os.path.join(tmp, "ar"); open(exe, "w").write("#!/bin/sh\n"); os.chmod(exe, 0o755)
    assert ar.find() == os.path.realpath(exe)


def test_ar_convert_reports_missing_skill(tmp, monkeypatch):
    monkeypatch.setenv("HOME", tmp)
    monkeypatch.setenv("SCAN_STUDIO_AR_MODELS", os.path.join(tmp, "none"))
    with pytest.raises(ScanError, match="ar-models"):
        ar.convert(os.path.join(tmp, "a.glb"), tmp, "a", height=0.2)


def test_ar_convert_runs_the_skill_and_checks_its_output(tmp, monkeypatch):
    skill = os.path.join(tmp, "skill"); os.mkdir(skill)
    exe = os.path.join(skill, "ar")
    open(exe, "w").write('#!/bin/sh\n# fake ar-models: write <out>/<name>.usdz\nout=$4; name=$6\n'
                         'printf usdz > "$out/$name.usdz"; echo \'{"name":"x"}\'\n')
    os.chmod(exe, 0o755)
    monkeypatch.setenv("SCAN_STUDIO_AR_MODELS", skill)
    res = ar.convert(os.path.join(tmp, "jug.glb"), tmp, "jug", height=0.2, log=lambda *_: None)
    assert res["usdz"] == "ar/jug.usdz" and res["bytes"] == 4


def test_rescale_refuses_foreign_or_broken_reports(tmp):
    from scan_studio.rescale import rescale
    with pytest.raises(UsageError):
        rescale(tmp, height=0.2)
    open(os.path.join(tmp, "report.json"), "w").write("{not json")
    with pytest.raises(UsageError):
        rescale(tmp, height=0.2)
    json.dump({"tool": "scan-studio", "model": {"file": "../../etc/x.glb", "size": [1, 1, 1], "triangles": 1}},
              open(os.path.join(tmp, "report.json"), "w"))
    with pytest.raises(ScanError):
        rescale(tmp, height=0.2)


def test_glb_validate_turns_wrong_types_into_scan_errors():
    import struct
    from scan_studio import glb
    js = json.dumps({"asset": {"version": "2.0"}, "meshes": "nope"}).encode()
    js += b" " * (-len(js) % 4)
    body = struct.pack("<II", len(js), 0x4E4F534A) + js
    with pytest.raises(ScanError):
        glb.validate(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)


def test_evaluate_refuses_a_bad_camera_file(tmp):
    from scan_studio.cli import main
    assert main(["evaluate", tmp, "--truth", "x.ply", "--cameras", "y.json"]) == 2
