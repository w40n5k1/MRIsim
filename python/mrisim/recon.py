"""Reconstruction: inverse DFT of the acquired samples, mode by mode (conjugate pairs), and the residual
against the sequence's ideal contrast image (each pixel's |M_perp| at the k = 0 sample)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine import Simulation, pixel_xy


@dataclass
class Mode:
    kx: int
    ky: int
    cells: list          # [(kx, ky)] or [(kx, ky), (-kx, -ky)]
    Z: complex           # S(k) + conj(S(-k))  (S(0) for DC)
    amp: float           # |Z| / N^2
    phase: float         # arg Z
    pattern: np.ndarray  # cos(2 pi k.r / N + phase), per pixel
    contrib: np.ndarray  # amp * pattern


@dataclass
class Recon:
    N: int
    modes: list
    image: np.ndarray    # real reconstruction (sum of all modes), per pixel
    ideal: np.ndarray    # |M_perp| per pixel at the k = 0 sample
    c: float             # least-squares scale between |image| and ideal
    residual: np.ndarray # (|image|/c - ideal) / max(ideal)
    n_acquired: int

    def grid(self, arr):
        return np.asarray(arr).reshape(self.N, self.N)

    def partial(self, n_modes: int) -> np.ndarray:
        """Sum of the first n_modes modes (in increasing |k|), per pixel."""
        out = np.zeros(self.N * self.N)
        for m in self.modes[:n_modes]:
            out += m.contrib
        return out


def reconstruct(sim: Simulation, samples=None) -> Recon:
    """samples: subset of sim.samples (default all) - e.g. those acquired up to a given time."""
    S = sim.S
    N, NPIX, H = S.N, S.npix, (S.N - 1) // 2
    samples = sim.samples if samples is None else samples
    table = {(s["kx"], s["ky"]): s["S"] for s in samples}
    get = lambda kx, ky: table.get((kx, ky), 0j)
    X, Y = pixel_xy(N)
    reps = [(kx, ky) for kx in range(-H, H + 1) for ky in range(-H, H + 1) if kx > 0 or (kx == 0 and ky > 0)]
    reps.sort(key=lambda k: (k[0] ** 2 + k[1] ** 2, np.arctan2(k[1], k[0])))
    modes = [Mode(0, 0, [(0, 0)], get(0, 0), 0.0, 0.0, None, None)]
    for kx, ky in reps:
        modes.append(Mode(kx, ky, [(kx, ky), (-kx, -ky)], get(kx, ky) + np.conj(get(-kx, -ky)), 0.0, 0.0, None, None))
    for m in modes:
        m.amp = abs(m.Z) / NPIX
        m.phase = float(np.angle(m.Z)) if m.Z != 0 else 0.0
        m.pattern = np.cos(2 * np.pi * (m.kx * X + m.ky * Y) / N + m.phase)
        m.contrib = m.amp * m.pattern
    image = sum(m.contrib for m in modes)
    s00 = next((s for s in samples if s["kx"] == 0 and s["ky"] == 0), None)
    ideal = np.hypot(sim.Mx[s00["idx"]], sim.My[s00["idx"]]) if s00 else np.zeros(NPIX)
    mag = np.abs(image)
    den = float((ideal ** 2).sum())
    c = float((mag * ideal).sum() / den) if den > 0 and (mag * ideal).sum() != 0 else 1.0
    refmax = max(1e-9, float(ideal.max()))
    residual = (mag / c - ideal) / refmax
    return Recon(N, modes, image, ideal, c, residual, len(samples))
