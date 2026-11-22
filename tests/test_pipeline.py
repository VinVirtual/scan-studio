"""End to end on 19 small frames of the demo capture (tests/data/jug-frames, rendered for this repository),
plus the refusal path and the command line. These run in CI too: about a minute on a CPU."""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from scan_studio import glb
from scan_studio.cli import main
from conftest import HERE, ROOT, textured
from drawn_face import drawn_face

FRAMES = os.path.join(HERE, "data", "jug-frames")


@pytest.fixture(scope="module")
def run_out(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("jug") / "jug-out")
    code = main(["run", FRAMES, "--out", out, "--name", "jug", "--height", "0.2", "--triangles", "20000",
                 "--texture", "512", "--threads", "4"])
    assert code == 0
    return out


@pytest.mark.pipeline
def test_pipeline_writes_a_valid_model_and_report(run_out):
    rep = json.load(open(os.path.join(run_out, "report.json")))
    assert rep["input"]["kind"] == "photos" and rep["faces"]["verdict"] == "clear"
    assert rep["sfm"]["registered"] >= 15 and rep["sfm"]["reprojection_error_px"] < 2.0
    assert rep["dense"]["method"] in ("sgbm-tsdf", "sparse-poisson")
    data = open(os.path.join(run_out, "jug.glb"), "rb").read()
    st = glb.validate(data)
    assert 1000 <= st["triangles"] <= 20000 and st["textures"] == 1
    assert rep["model"]["size"][1] == pytest.approx(0.2, abs=1e-3)       # the height we gave
    assert st["bounds_min"][1] == pytest.approx(0.0, abs=1e-6)            # standing on the floor
    assert 0.05 < rep["model"]["size"][0] < 0.25                          # a jug, not a table
    assert [s["stage"] for s in rep["timings"]["stages"]][0].startswith("frames")
    page = open(os.path.join(run_out, "report.html"), encoding="utf-8").read()
    assert "default-src 'none'" in page and len(page) < 20 * 1024 ** 2


@pytest.mark.pipeline
def test_scale_command_rescales_model_and_report(run_out, tmp_path):
    out = str(tmp_path / "copy")
    shutil.copytree(run_out, out)
    assert main(["scale", out, "--height", "0.1"]) == 0
    rep = json.load(open(os.path.join(out, "report.json")))
    assert rep["model"]["size"][1] == pytest.approx(0.1, abs=1e-4)
    assert rep["scale"]["method"] == "height"
    st = glb.validate(open(os.path.join(out, "jug.glb"), "rb").read())
    assert st["bounds_max"][1] == pytest.approx(0.1, abs=1e-4)
    # two points 0.05 apart on the model, said to be 0.1 m apart: doubles it
    assert main(["scale", out, "--points", "0,0,0", "0,0.05,0", "--length", "0.1"]) == 0
    rep = json.load(open(os.path.join(out, "report.json")))
    assert rep["model"]["size"][1] == pytest.approx(0.2, abs=1e-4)


@pytest.mark.pipeline
def test_report_command_rewrites_the_page(run_out):
    os.unlink(os.path.join(run_out, "report.html"))
    assert main(["report", run_out]) == 0
    assert os.path.isfile(os.path.join(run_out, "report.html"))


def test_people_in_the_frames_stop_the_run_and_frames_are_deleted(tmp):
    import cv2
    src = os.path.join(tmp, "people"); os.mkdir(src)
    for i in range(14):
        img = textured(720, 1280, seed=i)
        img[150:500, 400 + 20 * (i % 5):750 + 20 * (i % 5)] = drawn_face(350)
        cv2.imwrite(os.path.join(src, f"{i:02d}.jpg"), img)
    out = os.path.join(tmp, "out"); os.makedirs(os.path.join(out, "ar"))
    # an earlier run's outputs in a reused folder must not survive a refusal
    for stale in ("report.html", "people.glb", os.path.join("ar", "people.usdz")):
        open(os.path.join(out, stale), "w").write("old")
    assert main(["run", src, "--out", out]) == 4
    for stale in ("report.html", "people.glb", os.path.join("ar", "people.usdz")):
        assert not os.path.exists(os.path.join(out, stale))
    rep = json.load(open(os.path.join(out, "report.json")))
    assert rep["refused"] == "faces" and rep["faces"]["verdict"] == "refused"
    assert not os.path.exists(os.path.join(out, "work", "images"))
    assert not os.path.exists(os.path.join(out, "report.html"))


def test_faces_command_exit_codes(tmp):
    import cv2
    p = os.path.join(tmp, "a.png")
    img = textured(720, 1280); img[150:500, 400:750] = drawn_face(350)
    cv2.imwrite(p, img)
    assert main(["faces", p]) == 0          # one frame with a face is a warning, not a refusal
    folder = os.path.join(tmp, "set"); os.mkdir(folder)
    for i in range(3):
        cv2.imwrite(os.path.join(folder, f"{i}.png"), img)
    assert main(["faces", folder]) == 4


@pytest.mark.parametrize("args,code", [
    (["run", FRAMES, "--points", "0,0,0", "1,1,1"], 2),
    (["run", FRAMES, "--length", "0.2"], 2),
    (["run", FRAMES, "--height", "0.2", "--longest", "0.3"], 2),
    (["run", FRAMES, "--usdz"], 2),
    (["run", FRAMES, "--texture", "1000"], 2),
    (["run", FRAMES, "--triangles", "10"], 2),
    (["run", FRAMES, "--frames", "3"], 2),
    (["run", FRAMES, "--name", "../evil"], 2),
    (["run", "/definitely/not/here.mp4"], 2),
    (["run", FRAMES, "--height", "-1"], 2),
    (["run", FRAMES, "--height", "nan"], 2),
    (["run", FRAMES, "--points", "1,2", "0,0,0", "--length", "0.1"], 2),
    (["run", FRAMES, "--box", "1,1,1,0,0,0"], 2),
    (["scale", "/definitely/not/here", "--height", "0.2"], 2),
])
def test_usage_mistakes_exit_2_before_any_work(args, code, tmp):
    if args[0] == "run" and "--out" not in args:
        args = args + ["--out", os.path.join(tmp, "o")]
    assert main(args) == code
    assert not os.path.exists(os.path.join(tmp, "o", "work", "images"))   # refused before reconstruction


def test_launcher_runs_isolated_from_the_current_folder(tmp):
    # a planted numpy.py in the working folder must not be imported
    open(os.path.join(tmp, "numpy.py"), "w").write("raise SystemExit('planted module imported')\n")
    proc = subprocess.run([os.path.join(ROOT, "scan-studio"), "env"], cwd=tmp, capture_output=True, text=True,
                          env={**os.environ, "SCAN_STUDIO_PYTHON": sys.executable})
    assert "planted" not in proc.stdout + proc.stderr
    assert "numpy" in proc.stdout


def test_version_flag():
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0


@pytest.mark.pipeline
def test_scale_without_usdz_removes_a_stale_ar_pair(run_out, tmp_path):
    out = str(tmp_path / "copy")
    shutil.copytree(run_out, out)
    os.makedirs(os.path.join(out, "ar"), exist_ok=True)
    for ext in ("glb", "usdz"):
        open(os.path.join(out, "ar", f"jug.{ext}"), "w").write("old size")
    rep = json.load(open(os.path.join(out, "report.json")))
    rep["usdz"] = {"usdz": "ar/jug.usdz", "glb": "ar/jug.glb"}
    json.dump(rep, open(os.path.join(out, "report.json"), "w"))
    assert main(["scale", out, "--height", "0.15"]) == 0
    assert not os.path.exists(os.path.join(out, "ar", "jug.usdz"))
    assert "usdz" not in json.load(open(os.path.join(out, "report.json")))


@pytest.mark.pipeline
def test_video_input_and_a_failing_usdz_keep_the_model_and_report(tmp, monkeypatch, capsys):
    import cv2
    from test_safety import fake_ar
    from conftest import write_video
    frames = [cv2.cvtColor(cv2.imread(os.path.join(FRAMES, n)), cv2.COLOR_BGR2RGB) for n in sorted(os.listdir(FRAMES))]
    video = write_video(os.path.join(tmp, "jug.mp4"), frames, fps=4)
    fake_ar(tmp, monkeypatch, fail=True)
    out = os.path.join(tmp, "jug-out")
    code = main(["run", video, "--out", out, "--height", "0.2", "--triangles", "10000", "--texture", "256",
                 "--threads", "4", "--usdz"])
    assert code == 1
    assert "USDZ failed" in capsys.readouterr().err
    rep = json.load(open(os.path.join(out, "report.json")))
    assert rep["input"]["kind"] == "video" and "error" in rep["usdz"]
    assert os.path.isfile(os.path.join(out, "report.html")) and os.path.isfile(os.path.join(out, "jug.glb"))
    assert rep["sfm"]["registered"] >= 0.3 * rep["sfm"]["total"]


def test_too_few_cameras_placed_is_a_failure(tmp, monkeypatch):
    from scan_studio import pipeline, sfm
    def few(image_dir, names, work, **kw):
        V = [sfm.View(n, np.eye(3), np.zeros(4), np.eye(3), np.zeros(3), 640, 360) for n in names[:3]]
        return sfm.SparseModel(V, np.zeros((5, 3)), np.zeros((5, 3), np.uint8), [[0]] * 5, 0.5, 3, len(names))
    monkeypatch.setattr(pipeline.sfm, "run", few)
    assert main(["run", FRAMES, "--out", os.path.join(tmp, "o")]) == 1


def test_launcher_reports_a_killed_process(tmp):
    fake = os.path.join(tmp, "python")
    open(fake, "w").write("#!/bin/sh\nkill -9 $$\n")
    os.chmod(fake, 0o755)
    proc = subprocess.run([os.path.join(ROOT, "scan-studio"), "env"], capture_output=True, text=True,
                          env={**os.environ, "SCAN_STUDIO_PYTHON": fake})
    assert proc.returncode == 137
    assert "killed by signal 9" in proc.stderr and "out of memory" in proc.stderr
