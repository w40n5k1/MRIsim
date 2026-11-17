"""Phantoms as tissue maps, rendered at any odd N from a phantom file (format didactic-mri-phantom/1).

A phantom file has a background tissue and a list of shapes painted in order, in normalized coordinates
x, y in [-1, 1] over the field of view (pixel centers at x = (col - H)/(N/2), y = (H - row)/(N/2)):
  {"type": "rect", "tissue": "gm", "x": [x0, x1], "y": [y0, y1]}
  {"type": "disk", "tissue": "csf", "center": [cx, cy], "r": r}     (a disk covers at least its nearest pixel)
An optional "pd" expression in x, y gives a proton-density factor multiplying the tissue pd (the ramp).
The tissue dictionary (tissues.json) gives, per tissue key: name, pd, T1, T2 (ms) and a display color.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine import eval_expr, load_json, pixel_xy


@dataclass
class Phantom:
    defn: dict
    N: int
    tissues: dict
    label: list          # tissue key per pixel (row-major, row 0 at the top)
    pdf: np.ndarray      # proton-density factor per pixel
    M0: np.ndarray
    T1: np.ndarray
    T2: np.ndarray

    def grid(self, arr):
        return np.asarray(arr).reshape(self.N, self.N)

    def colors(self) -> np.ndarray:
        """(N, N, 3) RGB in [0, 1] from the tissue colors."""
        out = np.zeros((self.N * self.N, 3))
        for p, k in enumerate(self.label):
            h = self.tissues[k]["color"].lstrip("#")
            out[p] = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        return out.reshape(self.N, self.N, 3)


def load_tissues(path) -> dict:
    d = load_json(path)
    return d["tissues"] if "tissues" in d else d


def render_phantom(defn: dict, N: int, tissues: dict) -> Phantom:
    if N % 2 == 0 or N < 1:
        raise ValueError("N must be odd")
    X, Y = pixel_xy(N)
    xn, yn = X / (N / 2), Y / (N / 2)
    label = [defn.get("background", "air")] * (N * N)
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
        pdf = np.array([float(eval_expr(defn["pd"], {"x": float(xn[p]), "y": float(yn[p]), "N": N})) for p in range(N * N)])
    M0 = np.array([tissues[k]["pd"] for k in label]) * pdf
    T1 = np.array([float(tissues[k]["T1"]) for k in label])
    T2 = np.array([float(tissues[k]["T2"]) for k in label])
    return Phantom(defn, N, tissues, label, pdf, M0, T1, T2)


def load_phantom(path, N: int, tissues: dict) -> Phantom:
    return render_phantom(load_json(path), N, tissues)
