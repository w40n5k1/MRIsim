# By Bruno Mota - METAbio
"""Phantom editor: paint an N x N tissue map with the mouse and save it as a phantom file for the simulator.

    python -m mrisim.phantom_editor [phantom.json] [--tissues tissues.json] [-N 9] [--lang pt|en]

The grid is N x N with N odd, from 3 to 35 (the sizes the Python viewer accepts; the browser page stops at
17).  Every pixel holds one tissue key of the tissue dictionary (tissues.json); the background is air.  The
window shows:
  * top row: the tissues of the dictionary as colored buttons; click one to select it (the selected button is
    framed).  Save and Load are at the left of the same row.
  * the grid, drawn in the tissue colors.  Left-drag paints the selected tissue, right-drag paints air.
  * bottom rows: the editing tools, which act on the selected tissue only (and on air, when they add pixels):
      Fill on/off     with fill on, a left click fills the connected air region under the cursor with the
                      selected tissue, and a right click turns the connected region of the selected tissue
                      under the cursor into air
      Erode           selected-tissue pixels with fewer than three orthogonal neighbors of the same tissue
                      become air
      Grow            air pixels with at least two orthogonal neighbors of the selected tissue become that tissue
      Fill holes      air pockets enclosed by the selected tissue become that tissue
      Convex hull     air pixels inside the convex hull of the selected tissue become that tissue
      Convex core     selected-tissue pixels outside its largest inscribed convex subset (Borgefors and
                      Strand's concavity-removal algorithm) become air
      Open / Close    3 x 3 morphological opening (removes thin parts, to air) / closing (fills thin gaps,
                      from air) of the selected tissue
      Smooth          3 x 3 majority filter of the selected tissue (gains from air, losses to air)
      Negative        swaps the selected tissue and air; other tissues are untouched
      Undo            reverts the last change
    and a text box with N: submitting a new odd N from 3 to 35 starts a blank grid of that size (Undo brings
    the previous drawing back).

Files.  A phantom is saved as a phantom file (format didactic-mri-phantom/1) carrying a "grid" block,
    {"format": "didactic-mri-phantom/1", "id": ..., "name": {...}, "background": "air",
     "grid": {"N": 9, "labels": [["air", "gm", ...], ...]}}
which the simulator renders at any odd N (nearest pixel) and which the Python viewer lists when the file is put
in its phantoms/ folder.  Load accepts any phantom file: a shape-based one (rect / disk) is rendered at the
editor's current N; a grid-based one sets N to its own.  The file dialogs use tkinter.

The structure follows the mask editor the course has used for cortical masks (mask_maker.py): the same figure
layout, mouse handling and tool set, with the binary mask replaced by a map of tissue keys.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import Button, TextBox
from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening, label as cc_label, median_filter
from scipy.spatial import ConvexHull

from .engine import loc
from .phantoms import load_tissues, render_phantom

ROOT = pathlib.Path(__file__).resolve().parent.parent
N_MIN, N_MAX = 3, 35
TXT = {
    "pt": dict(save="Salvar", load="Carregar", fill_on="Preencher: sim", fill_off="Preencher: não", erode="Erodir",
               grow="Crescer", undo="Desfazer", holes="Tapar buracos", core="Núcleo convexo", hull="Casco convexo",
               neg="Negativo", open="Abrir", close="Fechar", smooth="Suavizar", title="Editor de fantomas",
               newfile="<novo>", status="{tissue}: {n} pixels · ar: {air} · N = {N}", saved="fantoma salvo em",
               loaded="fantoma carregado de", badN=f"N deve ser ímpar, entre {N_MIN} e {N_MAX}"),
    "en": dict(save="Save", load="Load", fill_on="Fill on", fill_off="Fill off", erode="Erode", grow="Grow",
               undo="Undo", holes="Fill holes", core="Convex core", hull="Convex hull", neg="Negative", open="Open",
               close="Close", smooth="Smooth", title="Phantom editor", newfile="<new>",
               status="{tissue}: {n} pixels · air: {air} · N = {N}", saved="phantom saved to",
               loaded="phantom loaded from", badN=f"N must be odd, between {N_MIN} and {N_MAX}"),
}


# =============================================================================================== geometry

def max_inscribed_convex_mask(mask: np.ndarray, max_iters: int = 20) -> np.ndarray:
    """Approximate largest convex subset of a binary mask (Borgefors and Strand's concavity removal).

    Each iteration finds the convex deficiency (background inside the convex hull), locates its deepest point
    (largest geodesic distance from the outside, the object acting as a wall), and cuts the object along the
    straight line through that point, among eight orientations (horizontal, vertical, the two diagonals, each
    on either side), that removes the fewest object pixels.  Stops when the object is convex or after
    max_iters cuts.  Returns a boolean mask.
    """
    mask = mask.astype(bool).copy()
    rows, cols = mask.shape
    jj, ii = np.meshgrid(np.arange(cols), np.arange(rows))
    for _ in range(max_iters):
        pts = np.argwhere(mask)
        if len(pts) < 3:
            break
        # Convex hull of the object (border pixels suffice) and its deficiency.
        inner = np.zeros_like(mask)
        inner[1:-1, 1:-1] = mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
        border = np.argwhere(mask & ~inner)
        hull_pts = border if len(border) >= 3 else pts
        xy = np.column_stack((hull_pts[:, 1], hull_pts[:, 0]))
        try:
            hull = ConvexHull(xy)
        except Exception:            # degenerate (collinear) object: already convex
            break
        inside = Path(xy[hull.vertices], closed=True).contains_points(np.column_stack((jj.ravel(), ii.ravel()))).reshape(rows, cols)
        deficiency = inside & ~mask
        if not deficiency.any():
            break
        # Geodesic distance of the background from the image border, the object being a wall (BFS).
        dist = np.full((rows, cols), -1, int)
        queue = deque()
        for i in range(rows):
            for j in (0, cols - 1):
                if not mask[i, j] and dist[i, j] < 0:
                    dist[i, j] = 0
                    queue.append((i, j))
        for j in range(cols):
            for i in (0, rows - 1):
                if not mask[i, j] and dist[i, j] < 0:
                    dist[i, j] = 0
                    queue.append((i, j))
        while queue:
            i, j = queue.popleft()
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                a, b = i + di, j + dj
                if 0 <= a < rows and 0 <= b < cols and not mask[a, b] and dist[a, b] < 0:
                    dist[a, b] = dist[i, j] + 1
                    queue.append((a, b))
        depth = np.where(deficiency, dist, 0)
        if depth.max() <= 0:
            break
        ci, cj = np.argwhere(depth == depth.max())[0]
        # Candidate cuts: half-planes through the deepest point; keep the one removing the fewest pixels.
        candidates = [ii > ci - 0.5, ii < ci + 0.5, jj > cj - 0.5, jj < cj + 0.5,
                      (ii - jj) > (ci - cj) - 0.5, (ii - jj) < (ci - cj) + 0.5,
                      (ii + jj) > (ci + cj) - 0.5, (ii + jj) < (ci + cj) + 0.5]
        best, best_n = None, None
        for half in candidates:
            n_removed = np.count_nonzero(mask & half)
            if n_removed and (best_n is None or n_removed < best_n):
                best, best_n = half, n_removed
        if best is None:
            break
        mask &= ~best
    return mask


# =============================================================================================== editor

class PhantomEditor:
    """The editor window.  `grid` holds one tissue index per pixel (row 0 at the top); `keys` maps indices to
    the dictionary's tissue keys; `selected` is the index of the tissue being painted."""

    def __init__(self, tissues: dict, N: int = 9, labels=None, filename: str | None = None, lang: str = "pt"):
        self.tissues = tissues
        self.keys = list(tissues)
        if "air" not in self.keys:
            raise ValueError("the tissue dictionary needs an 'air' entry (the background)")
        self.AIR = self.keys.index("air")
        self.lang = lang
        self.T = TXT.get(lang, TXT["en"])
        self.filename = filename
        self.history = []
        self.fill_mode = False
        self.drawing = False
        self.paint_value = None
        self.last = None
        self.selected = next((i for i, k in enumerate(self.keys) if k != "air"), 0)
        if labels is not None:
            self.N = int(np.sqrt(len(labels))) if np.ndim(labels) == 1 else len(labels)
            self.grid = np.array([[self.keys.index(k) for k in row] for row in np.reshape(labels, (self.N, self.N))], int)
        else:
            self.N = N
            self.grid = np.full((N, N), self.AIR, int)
        self._build_window()
        self.refresh()

    # ------------------------------------------------------------------ window

    def _build_window(self):
        T = self.T
        self.fig, self.ax = plt.subplots(figsize=(8, 9))
        self.fig.canvas.manager.set_window_title(f"{T['title']} — {self.filename or T['newfile']}")
        plt.subplots_adjust(left=0.05, right=0.95, top=0.86, bottom=0.21)
        self.im = self.ax.imshow(self._rgb(), interpolation="nearest", origin="upper")
        self._set_axes()
        self.status = self.fig.text(0.05, 0.875, "", fontsize=9)
        self.widgets = {}

        # Top row: Save, Load, then one button per tissue.
        n_cells = 2 + len(self.keys)
        w = 0.96 / n_cells
        self.widgets["save"] = Button(self.fig.add_axes([0.02, 0.93, w - 0.005, 0.05]), T["save"])
        self.widgets["save"].on_clicked(self.on_save)
        self.widgets["load"] = Button(self.fig.add_axes([0.02 + w, 0.93, w - 0.005, 0.05]), T["load"])
        self.widgets["load"].on_clicked(self.on_load)
        self.tissue_buttons = []
        for i, k in enumerate(self.keys):
            color = self.tissues[k]["color"]
            ax = self.fig.add_axes([0.02 + (2 + i) * w, 0.93, w - 0.005, 0.05])
            b = Button(ax, loc(self.tissues[k]["name"], self.lang), color=color, hovercolor=color)
            b.label.set_fontsize(8)
            b.label.set_color("white" if _dark(color) else "black")
            b.on_clicked(lambda _e, i=i: self.select(i))
            self.tissue_buttons.append(b)

        # Bottom rows: the tools, as in the mask editor.
        layout = [("toggle", T["fill_off"], 0.02, 0.10), ("undo", T["undo"], 0.17, 0.10), ("holes", T["holes"], 0.32, 0.10),
                  ("core", T["core"], 0.47, 0.10), ("hull", T["hull"], 0.62, 0.10), ("neg", T["neg"], 0.77, 0.10),
                  ("grow", T["grow"], 0.02, 0.03), ("erode", T["erode"], 0.17, 0.03), ("open", T["open"], 0.32, 0.03),
                  ("close", T["close"], 0.47, 0.03), ("smooth", T["smooth"], 0.62, 0.03)]
        actions = {"toggle": self.toggle_fill, "undo": self.on_undo, "holes": self.on_fill_holes, "core": self.on_convex_core,
                   "hull": self.on_convex_hull, "neg": self.on_negative, "grow": self.on_grow, "erode": self.on_erode,
                   "open": self.on_open, "close": self.on_close, "smooth": self.on_smooth}
        for key, text, x, y in layout:
            b = Button(self.fig.add_axes([x, y, 0.13, 0.05]), text)
            b.label.set_fontsize(8)
            b.on_clicked(actions[key])
            self.widgets[key] = b
        self.tb_N = TextBox(self.fig.add_axes([0.83, 0.03, 0.06, 0.05]), "N ", initial=str(self.N))
        self.tb_N.on_submit(self.on_resize)

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)

    def _set_axes(self):
        N = self.N
        self.ax.set_xlim(-0.5, N - 0.5)
        self.ax.set_ylim(N - 0.5, -0.5)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
        self.ax.grid(which="minor", color="white", linewidth=1)
        self.ax.tick_params(which="minor", length=0)

    def _rgb(self) -> np.ndarray:
        """(N, N, 3) image of the grid in the tissue colors."""
        palette = np.array([_hex_to_rgb(self.tissues[k]["color"]) for k in self.keys])
        return palette[self.grid]

    def refresh(self):
        self.im.set_data(self._rgb())
        self.im.set_extent((-0.5, self.N - 0.5, self.N - 0.5, -0.5))
        for i, b in enumerate(self.tissue_buttons):
            for sp in b.ax.spines.values():
                sp.set_linewidth(3 if i == self.selected else 0.5)
                sp.set_edgecolor("#ff5c9a" if i == self.selected else "black")
        name = loc(self.tissues[self.keys[self.selected]]["name"], self.lang)
        self.status.set_text(self.T["status"].format(tissue=name, n=int((self.grid == self.selected).sum()),
                                                     air=int((self.grid == self.AIR).sum()), N=self.N))
        self.fig.canvas.draw_idle()

    def select(self, i: int):
        self.selected = i
        self.refresh()

    # ------------------------------------------------------------------ painting

    def _coords(self, event):
        return int(round(event.ydata)), int(round(event.xdata))

    def _push(self):
        self.history.append(self.grid.copy())

    def on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        i, j = self._coords(event)
        if not (0 <= i < self.N and 0 <= j < self.N):
            return
        if self.fill_mode:
            if event.button == 1 and self.grid[i, j] == self.AIR:
                self._push()
                self._fill_component(i, j, self.AIR, self.selected)
            elif event.button == 3 and self.grid[i, j] == self.selected:
                self._push()
                self._fill_component(i, j, self.selected, self.AIR)
            return
        if event.button == 1:
            self.paint_value = self.selected
        elif event.button == 3:
            self.paint_value = self.AIR
        else:
            return
        self._push()
        self.drawing = True
        self.last = (i, j)
        self._paint_point(i, j)

    def on_release(self, _event):
        self.drawing = False
        self.last = None

    def on_move(self, event):
        if not self.drawing or event.inaxes != self.ax or event.xdata is None:
            return
        i, j = self._coords(event)
        self._paint_line(*self.last, i, j)
        self.last = (i, j)

    def _paint_point(self, i, j):
        if 0 <= i < self.N and 0 <= j < self.N:
            self.grid[i, j] = self.paint_value
            self.refresh()

    def _paint_line(self, i0, j0, i1, j1):
        steps = max(abs(i1 - i0), abs(j1 - j0))
        if steps == 0:
            self._paint_point(i1, j1)
            return
        for ii, jj in zip(np.linspace(i0, i1, steps + 1).round().astype(int), np.linspace(j0, j1, steps + 1).round().astype(int)):
            self._paint_point(ii, jj)

    def _fill_component(self, i, j, source, target):
        """Turn the 4-connected region of tissue `source` containing (i, j) into tissue `target`."""
        comps, _ = cc_label(self.grid == source)
        self.grid[comps == comps[i, j]] = target
        self.refresh()

    # ------------------------------------------------------------------ tools (selected tissue only)

    def _mask(self) -> np.ndarray:
        return self.grid == self.selected

    def _air(self) -> np.ndarray:
        return self.grid == self.AIR

    def _apply(self, new_mask: np.ndarray):
        """Replace the selected tissue's mask: pixels gained are taken from air only, pixels lost become air."""
        old = self._mask()
        self.grid[old & ~new_mask] = self.AIR
        self.grid[new_mask & ~old & self._air()] = self.selected
        self.refresh()

    def toggle_fill(self, _event):
        self.fill_mode = not self.fill_mode
        b = self.widgets["toggle"]
        b.label.set_text(self.T["fill_on"] if self.fill_mode else self.T["fill_off"])
        face = "0.6" if self.fill_mode else "0.9"
        b.color = b.hovercolor = face
        b.ax.set_facecolor(face)
        self.fig.canvas.draw_idle()

    def on_undo(self, _event):
        if not self.history:
            return
        self.grid = self.history.pop()
        self.N = self.grid.shape[0]
        self.tb_N.set_val(str(self.N))
        self._set_axes()
        self.refresh()

    def on_negative(self, _event):
        self._push()
        sel, air = self._mask(), self._air()
        self.grid[sel] = self.AIR
        self.grid[air] = self.selected
        self.refresh()

    def on_fill_holes(self, _event):
        self._push()
        self._apply(binary_fill_holes(self._mask()))

    def on_erode(self, _event):
        self._push()
        m = self._mask()
        neighbors = _count_neighbors(m)
        self._apply(m & (neighbors >= 3))

    def on_grow(self, _event):
        self._push()
        m = self._mask()
        self._apply(m | (_count_neighbors(m) >= 2))

    def on_convex_hull(self, _event):
        self._push()
        pts = np.argwhere(self._mask())
        if len(pts) >= 3:
            xy = np.column_stack((pts[:, 1], pts[:, 0]))
            try:
                hull = ConvexHull(xy)
            except Exception:        # collinear pixels: nothing to fill
                return
            jj, ii = np.meshgrid(np.arange(self.N), np.arange(self.N))
            inside = Path(xy[hull.vertices]).contains_points(np.column_stack((jj.ravel(), ii.ravel()))).reshape(self.N, self.N)
            self._apply(self._mask() | inside)

    def on_convex_core(self, _event):
        self._push()
        self._apply(max_inscribed_convex_mask(self._mask()))

    def on_open(self, _event):
        self._push()
        self._apply(binary_opening(self._mask(), structure=np.ones((3, 3), bool)))

    def on_close(self, _event):
        self._push()
        self._apply(binary_closing(self._mask(), structure=np.ones((3, 3), bool)))

    def on_smooth(self, _event):
        self._push()
        self._apply(median_filter(self._mask().astype(int), size=3).astype(bool))

    def on_resize(self, text):
        try:
            n = int(text)
            if n % 2 == 0 or not N_MIN <= n <= N_MAX:
                raise ValueError
        except ValueError:
            print(self.T["badN"])
            self.tb_N.set_val(str(self.N))
            return
        if n == self.N:
            return
        self._push()
        self.N = n
        self.grid = np.full((n, n), self.AIR, int)
        self._set_axes()
        self.refresh()

    # ------------------------------------------------------------------ files

    def to_definition(self, name: str | None = None) -> dict:
        """The phantom file content for the current grid."""
        stem = name or (pathlib.Path(self.filename).stem if self.filename else "fantoma")
        return {"format": "didactic-mri-phantom/1", "id": stem, "name": {"pt": stem, "en": stem}, "background": "air",
                "grid": {"N": self.N, "labels": [[self.keys[v] for v in row] for row in self.grid]}}

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_definition(pathlib.Path(path).stem), f, ensure_ascii=False, indent=1)
        self.filename = os.path.basename(path)
        self.fig.canvas.manager.set_window_title(f"{self.T['title']} — {self.filename}")
        print(self.T["saved"], path)

    def load(self, path):
        """Load a phantom file: a grid-based one sets N; a shape-based one is rendered at the current N."""
        defn = render_definition(path, self.N, self.tissues)
        grid_block = defn.get("grid")
        if grid_block:
            self.N = int(grid_block.get("N", len(grid_block["labels"])))
        self._push()
        ph = render_phantom(defn, self.N, self.tissues)
        self.grid = np.array([self.keys.index(k) for k in ph.label], int).reshape(self.N, self.N)
        self.filename = os.path.basename(path)
        self.tb_N.set_val(str(self.N))
        self.fig.canvas.manager.set_window_title(f"{self.T['title']} — {self.filename}")
        self._set_axes()
        self.refresh()
        print(self.T["loaded"], path)

    def on_save(self, _event):
        path = _file_dialog(save=True, initial=self.filename)
        if path:
            self.save(path)

    def on_load(self, _event):
        path = _file_dialog(save=False)
        if path:
            self.load(path)

    def show(self):
        plt.show()


# =============================================================================================== helpers

def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def _dark(h: str) -> bool:
    r, g, b = _hex_to_rgb(h)
    return 0.299 * r + 0.587 * g + 0.114 * b < 0.5


def _count_neighbors(m: np.ndarray) -> np.ndarray:
    """Number of orthogonal (4-connected) True neighbors of every pixel of a boolean mask."""
    n = np.zeros(m.shape, int)
    n[1:, :] += m[:-1, :]
    n[:-1, :] += m[1:, :]
    n[:, 1:] += m[:, :-1]
    n[:, :-1] += m[:, 1:]
    return n


def render_definition(path, N: int, tissues: dict) -> dict:
    """Read a phantom file and check that its tissue keys exist in the dictionary."""
    with open(path, encoding="utf-8") as f:
        defn = json.load(f)
    used = {defn.get("background", "air")} | {sh["tissue"] for sh in defn.get("shapes", [])}
    if defn.get("grid"):
        used |= {k for row in defn["grid"]["labels"] for k in row}
    unknown = used - set(tissues)
    if unknown:
        raise ValueError("unknown tissues in phantom file: " + ", ".join(sorted(unknown)))
    return defn


def _file_dialog(save: bool, initial: str | None = None):
    """tkinter file dialog for a phantom .json; returns the path or None."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    kw = dict(filetypes=[("phantom JSON", "*.json")], initialdir=os.getcwd())
    if save:
        path = filedialog.asksaveasfilename(title="Save phantom", defaultextension=".json", initialfile=initial or "fantoma.json", **kw)
    else:
        path = filedialog.askopenfilename(title="Load phantom", **kw)
    root.destroy()
    return path or None


def main(argv=None):
    ap = argparse.ArgumentParser(description="phantom editor for the didactic MRI simulator")
    ap.add_argument("phantom", nargs="?", help="phantom .json to load (shape-based or grid-based)")
    ap.add_argument("--tissues", default=str(ROOT / "tissues.json"))
    ap.add_argument("-N", type=int, default=9, help=f"grid size for a new phantom or for rendering a shape-based one (odd, {N_MIN}-{N_MAX})")
    ap.add_argument("--lang", default="pt")
    a = ap.parse_args(argv)
    if a.N % 2 == 0 or not N_MIN <= a.N <= N_MAX:
        ap.error(TXT["en"]["badN"])
    tissues = load_tissues(a.tissues)
    editor = PhantomEditor(tissues, N=a.N, lang=a.lang)
    if a.phantom:
        editor.load(a.phantom)
        editor.history.clear()
    editor.show()


if __name__ == "__main__":
    main()
