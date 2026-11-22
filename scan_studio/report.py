"""report.json and the single-file report.html (three.js, fonts, the GLB and the before cloud all embedded).

The page's content security policy allows only its own inline script and style, by hash: no network requests,
no other scripts. Every string that came from the input (file names, the model name) is HTML-escaped where it
is placed in markup, and the data block is JSON that cannot close its <script> element."""
from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import html
import json
import os
import re

import numpy as np

from . import __version__
from .budgets import DEFAULT, check_total
from .safeio import write_text

PLACEHOLDER = re.compile(r"__(?:CSP|TITLE|NAME|LEDE|META|VERSION|CSS|DATA|JS)__")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer")


def script_json(obj) -> str:
    """JSON that is safe inside <script type="application/json">: no '<', '>' or '&' and no line separators."""
    s = json.dumps(obj, separators=(",", ":"), allow_nan=False)
    return (s.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def esc(text) -> str:
    return html.escape(str(text), quote=True)


def csp_hash(text: str) -> str:
    return "'sha256-" + base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode("ascii") + "'"


def _read(name: str, mode: str = "r"):
    with open(os.path.join(HERE, name), mode, **({} if "b" in mode else {"encoding": "utf-8"})) as fh:
        return fh.read()


def _clean_json(obj):
    """Replace NaN and infinities (not valid JSON) with null, everywhere."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, np.generic):
        return _clean_json(obj.item())
    return obj


def lede(rep: dict) -> str:
    src = rep.get("input", {})
    kind = "video" if src.get("kind") == "video" else "photo set"
    return (f"A 3D model made from a {kind}, {esc(src.get('file', ''))}, on a CPU: "
            f"{len(rep['frames']['chosen'])} frames, {rep['sfm']['registered']} cameras placed.")


def meta(rep: dict) -> str:
    when = _dt.date.today().isoformat()
    items = [("file", rep["input"].get("file", "")), ("model", rep["model"]["file"]),
             ("scale", rep["scale"]["method"] if rep["scale"]["method"] != "none" else "not set"),
             ("people", rep["faces"]["verdict"]), ("made", when)]
    return "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in items)


def render(rep: dict, glb_bytes: bytes, before: dict) -> str:
    rep = _clean_json(rep)
    data = {"report": rep, "glb": base64.b64encode(glb_bytes).decode("ascii"),
            "before": {"points": base64.b64encode(np.asarray(before["points"], np.float32).tobytes()).decode("ascii"),
                       "colors": base64.b64encode(np.asarray(before["colors"], np.uint8).tobytes()).decode("ascii")}}
    css = (_read("report.css")
           .replace("__SERIF__", base64.b64encode(_read("InstrumentSerif-Regular.ttf", "rb")).decode("ascii"))
           .replace("__MONO__", base64.b64encode(_read("JetBrainsMono.ttf", "rb")).decode("ascii")))
    js = _read("viewer.bundle.js").replace("</script", "<\\/script")
    csp = ("default-src 'none'; " f"script-src {csp_hash(js)}; style-src {csp_hash(css)}; "
           "font-src data:; img-src data: blob:; connect-src data: blob:; base-uri 'none'; form-action 'none'")
    fields = {"__CSP__": csp, "__TITLE__": esc(f"{rep['name']} · scan-studio"), "__NAME__": esc(rep["name"]),
              "__LEDE__": lede(rep), "__META__": meta(rep), "__VERSION__": esc(__version__),
              "__CSS__": css, "__DATA__": script_json(data), "__JS__": js}
    # one pass over the template: text put in is never searched again, so input cannot inject a placeholder
    return PLACEHOLDER.sub(lambda m: fields[m.group(0)], _read("report.html"))


def write(out: str, rep: dict, glb_bytes: bytes, before: dict, budgets=DEFAULT) -> tuple[str, str]:
    rep = _clean_json(rep)
    jpath = write_text(os.path.join(out, "report.json"), json.dumps(rep, indent=1, allow_nan=False))
    page = render(rep, glb_bytes, before)
    check_total("report size in bytes", len(page.encode("utf-8")), budgets.report_bytes)
    hpath = write_text(os.path.join(out, "report.html"), page)
    return jpath, hpath
