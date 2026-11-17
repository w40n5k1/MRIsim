# By Bruno Mota - METAbio
"""Tests of the phantom files and of the phantom editor (run headless with the Agg backend).

1. test_grid_phantom_resampling: a painted grid renders identically at its own N and by nearest pixel at other N.
2. test_editor_tools: the editor's tools change only the selected tissue (gains from air, losses to air), undo
   restores, and a save/load round trip through a phantom file preserves the grid.
"""
import json
import pathlib
import types

import matplotlib
matplotlib.use("Agg")
import numpy as np

from mrisim import load_tissues, render_phantom
from mrisim.phantom_editor import PhantomEditor, max_inscribed_convex_mask

ROOT = pathlib.Path(__file__).resolve().parent.parent
TIS = load_tissues(ROOT / "tissues.json")


def test_grid_phantom_resampling():
    rows = [["air"] * 5, ["air", "gm", "gm", "gm", "air"], ["air", "gm", "wm", "gm", "air"], ["air", "gm", "gm", "gm", "air"], ["air"] * 5]
    defn = {"format": "didactic-mri-phantom/1", "id": "t", "background": "air", "grid": {"N": 5, "labels": rows}}
    assert render_phantom(defn, 5, TIS).label == [k for r in rows for k in r]
    ph15 = render_phantom(defn, 15, TIS)
    g = np.array(ph15.label).reshape(15, 15)
    assert (g[6:9, 6:9] == "wm").all() and (g[3:12, 3:12] != "air").all() and (g[0:3, :] == "air").all()
    assert render_phantom(defn, 3, TIS).label[4] == "wm"


def _event(ed, i, j, button):
    return types.SimpleNamespace(inaxes=ed.ax, xdata=float(j), ydata=float(i), button=button)


def test_editor_tools(tmp_path):
    ed = PhantomEditor(TIS, N=9)
    gm, wm, air = ed.keys.index("gm"), ed.keys.index("wm"), ed.AIR
    # Paint a 5 x 5 ring of gray matter by dragging along its four sides.
    for (i0, j0, i1, j1) in [(2, 2, 2, 6), (6, 2, 6, 6), (2, 2, 6, 2), (2, 6, 6, 6)]:
        ed.on_press(_event(ed, i0, j0, 1))
        ed.on_move(_event(ed, i1, j1, 1))
        ed.on_release(None)
    assert (ed.grid == gm).sum() == 16
    # Fill mode: the enclosed air becomes white matter; other tissues untouched.
    ed.select(wm)
    ed.toggle_fill(None)
    ed.on_press(_event(ed, 4, 4, 1))
    ed.toggle_fill(None)
    assert (ed.grid == wm).sum() == 9 and (ed.grid == gm).sum() == 16
    # Tools on the selected tissue (gm): fill holes takes air only, so with wm inside nothing changes.
    ed.select(gm)
    ed.on_fill_holes(None)
    assert (ed.grid == gm).sum() == 16 and (ed.grid == wm).sum() == 9
    ed.on_negative(None)                      # gm <-> air, wm untouched
    assert (ed.grid == gm).sum() == 81 - 16 - 9 and (ed.grid == wm).sum() == 9
    ed.on_undo(None)
    assert (ed.grid == gm).sum() == 16
    ed.on_erode(None)                         # a ring one pixel wide has no pixel with three same neighbors
    assert (ed.grid == gm).sum() == 0 and (ed.grid == wm).sum() == 9
    ed.on_undo(None)
    # Save / load round trip and rendering of the saved file.
    path = tmp_path / "ring.json"
    ed.save(str(path))
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["grid"]["N"] == 9 and d["id"] == "ring"
    ed2 = PhantomEditor(TIS, N=5)
    ed2.load(str(path))
    assert ed2.N == 9 and np.array_equal(ed2.grid, ed.grid)
    ph = render_phantom(d, 9, TIS)
    assert (np.array(ph.label) == "wm").sum() == 9
    # Resize gives a blank grid of the new size; an even N is refused.
    ed.on_resize("15")
    assert ed.N == 15 and (ed.grid == air).all()
    ed.on_resize("16")
    assert ed.N == 15


def test_convex_core_of_an_L():
    m = np.zeros((9, 9), bool)
    m[1:8, 1:4] = True
    m[5:8, 1:8] = True
    core = max_inscribed_convex_mask(m)
    assert core.sum() < m.sum() and (core & ~m).sum() == 0 and core.sum() == 21
