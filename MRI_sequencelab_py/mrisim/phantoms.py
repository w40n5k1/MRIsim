# By Bruno Mota - METAbio
"""Phantoms: the object being imaged, as a map of tissues on the N x N pixel grid.

A phantom file (format didactic-mri-phantom/1) names a background tissue and lists shapes painted on it in
order, in coordinates normalized to [-1, 1] over the field of view (pixel centers at x = (col - HALF)/(N/2),
y = (HALF - row)/(N/2), so the same file renders at any odd N):
  {"type": "rect", "tissue": "gm",  "x": [x0, x1], "y": [y0, y1]}
  {"type": "disk", "tissue": "csf", "center": [cx, cy], "r": r}      (a disk covers at least its nearest pixel)
An optional "pd" expression in x and y gives a proton-density factor multiplying the tissue's pd (used by the
ramp phantom).  A phantom painted pixel by pixel (the phantom editor's output) carries a "grid" instead of, or
before, the shapes:
  "grid": {"N": 9, "labels": [["air", "air", ...], ...]}      rows from the top, N x N tissue keys
The grid is rendered at any odd N by nearest-neighbor resampling of the pixel centers; shapes, if present, are
painted on top of it.

The tissue dictionary (tissues.json) gives, for each tissue key, its name, pd (proton density, 0..1), T1 and T2
in ms (values at 3 T) and a display color.  render_phantom() turns a phantom file plus the dictionary into the
per-pixel arrays M0, T1, T2 that the simulation needs, keeping the tissue label of every pixel for display.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine import eval_expr, load_json, pixel_xy


@dataclass
class Phantom:
    """A phantom rendered at one matrix size.

    defn      the phantom file as a dict
    N         matrix size
    tissues   the tissue dictionary used
    label     tissue key of every pixel (row-major, row 0 at the top)
    pdf       proton-density factor per pixel (1 unless the file has a "pd" expression)
    M0        equilibrium magnetization per pixel = tissue pd x pdf
    T1, T2    relaxation times per pixel (ms)
    """
    defn: dict
    N: int
    tissues: dict
    label: list
    pdf: np.ndarray
    M0: np.ndarray
    T1: np.ndarray
    T2: np.ndarray

    def grid(self, arr):
        """Reshape a per-pixel array to the (N, N) image layout (row 0 at the top)."""
        return np.asarray(arr).reshape(self.N, self.N)

    def colors(self) -> np.ndarray:
        """(N, N, 3) RGB image in [0, 1] with every pixel in its tissue's display color."""
        out = np.zeros((self.N * self.N, 3))
        for p, k in enumerate(self.label):
            h = self.tissues[k]["color"].lstrip("#")
            out[p] = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        return out.reshape(self.N, self.N, 3)


def load_tissues(path) -> dict:
    """Read the tissue dictionary: {key: {name, pd, T1, T2, color}}.

    The file may hold the dictionary directly or under a top-level "tissues" key.
    """
    d = load_json(path)
    return d["tissues"] if "tissues" in d else d


def render_phantom(defn: dict, N: int, tissues: dict) -> Phantom:
    """Paint the shapes of a phantom definition on an N x N grid and look up the tissue properties.

    A pixel belongs to a rectangle when its center lies inside it (closed interval) and to a disk when its
    center is within the radius; a disk smaller than a pixel still claims the pixel nearest to its center, so
    small features (the lesion of the brain phantom) survive at coarse N.  Later shapes overwrite earlier ones.
    """
    if N % 2 == 0 or N < 1:
        raise ValueError("N must be odd")
    X, Y = pixel_xy(N)
    xn, yn = X / (N / 2), Y / (N / 2)             # pixel centers in normalized coordinates
    label = [defn.get("background", "air")] * (N * N)
    if defn.get("grid"):
        label = _resample_grid(defn["grid"], N)
    for sh in defn.get("shapes", []):
        if sh["type"] == "rect":
            (x0, x1), (y0, y1) = sh["x"], sh["y"]
            inside = (xn >= x0) & (xn <= x1) & (yn >= y0) & (yn <= y1)
        elif sh["type"] == "disk":
            cx, cy = sh.get("center", [0, 0])
            r_px = max(sh["r"] * N / 2, 0.5)
            dist = np.hypot(X - cx * N / 2, Y - cy * N / 2)
            inside = dist <= r_px
            if not inside.any():
                inside[np.argmin(dist)] = True
        else:
            raise ValueError("unknown shape " + sh["type"])
        for p in np.flatnonzero(inside):
            label[p] = sh["tissue"]
    pdf = np.ones(N * N)
    if defn.get("pd"):
        pdf = np.array([float(eval_expr(defn["pd"], {"x": float(xn[p]), "y": float(yn[p]), "N": N}))
                        for p in range(N * N)])
    M0 = np.array([tissues[k]["pd"] for k in label]) * pdf
    T1 = np.array([float(tissues[k]["T1"]) for k in label])
    T2 = np.array([float(tissues[k]["T2"]) for k in label])
    return Phantom(defn, N, tissues, label, pdf, M0, T1, T2)


def _resample_grid(grid: dict, N: int) -> list:
    """Tissue keys of an N x N grid taken from a painted grid of size n (nearest pixel center, in normalized
    coordinates); the identity when n == N."""
    rows = grid["labels"]
    n = int(grid.get("N", len(rows)))
    if len(rows) != n or any(len(r) != n for r in rows):
        raise ValueError("phantom grid is not N x N")
    if n == N:
        return [k for row in rows for k in row]
    src = np.rint((np.arange(N) - (N - 1) / 2) * n / N + (n - 1) / 2).astype(int)   # nearest source index per target index
    src = np.clip(src, 0, n - 1)
    return [rows[src[i]][src[j]] for i in range(N) for j in range(N)]


def load_phantom(path, N: int, tissues: dict) -> Phantom:
    """Read a phantom file and render it at matrix size N with the given tissue dictionary."""
    return render_phantom(load_json(path), N, tissues)
