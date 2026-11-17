# By Bruno Mota - METAbio
"""Image reconstruction from the acquired samples, one Fourier mode at a time.

The samples are the forward DFT of the pixel magnetizations, so the image is the inverse DFT
    image(r) = (1/N^2) sum_k S(k) exp(+2 pi i k.r / N).
Because the object is real, S(-k) = conj(S(k)) in the ideal case, and the terms k and -k combine into one real
cosine pattern.  Writing Z = S(k) + conj(S(-k)) = |Z| exp(i phi), the pair contributes
    (|Z| / N^2) cos(2 pi k.r / N + phi)
to the image; the k = 0 term contributes S(0)/N^2 (a constant; its phase is the global signal phase).  The modes
are listed in increasing |k| so that adding them one by one shows the image building up from coarse to fine.
Any deviation of the pair sum from a real cosine (for instance from relaxation between the two acquisitions)
is still represented exactly, since |Z| and phi are taken from the actual samples.

The sequence's *ideal* image is the magnitude of the transverse magnetization of each pixel at the instant the
k = 0 sample is taken: that is the contrast the sequence encodes, free of the Fourier truncation and of any
inconsistency between k lines.  The residual compares the reconstruction with it after a least-squares scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine import Simulation, pixel_xy


@dataclass
class Mode:
    """One term of the reconstruction: the k = 0 constant, or a conjugate pair {k, -k}.

    kx, ky    the representative k of the pair (kx > 0, or kx = 0 and ky > 0); (0, 0) for the constant
    cells     the k cells it uses: [(kx, ky)] or [(kx, ky), (-kx, -ky)]
    Z         S(k) + conj(S(-k)), or S(0) for the constant
    amp       |Z| / N^2, the amplitude of the pattern
    phase     arg Z
    pattern   cos(2 pi k.r / N + phase) per pixel (normalized pattern, between -1 and 1)
    contrib   amp * pattern: what this mode adds to the image
    """
    kx: int
    ky: int
    cells: list
    Z: complex
    amp: float
    phase: float
    pattern: np.ndarray
    contrib: np.ndarray


@dataclass
class Recon:
    """A reconstruction from a set of samples.

    N            matrix size
    modes        the Mode list in increasing |k| (the constant first), (N^2 + 1)/2 modes in all
    image        the real image, sum of all mode contributions, per pixel
    ideal        |M_perp| of every pixel at the k = 0 sample (the sequence's ideal contrast image)
    c            least-squares scale factor between |image| and ideal
    residual     (|image| / c - ideal) / max(ideal), per pixel
    n_acquired   number of samples used
    """
    N: int
    modes: list
    image: np.ndarray
    ideal: np.ndarray
    c: float
    residual: np.ndarray
    n_acquired: int

    def grid(self, arr):
        """Reshape a per-pixel array to the (N, N) image layout (row 0 at the top)."""
        return np.asarray(arr).reshape(self.N, self.N)

    def partial(self, n_modes: int) -> np.ndarray:
        """Sum of the first n_modes modes (in increasing |k|), per pixel: the image at that stage of the
        mode-by-mode reconstruction."""
        out = np.zeros(self.N * self.N)
        for m in self.modes[:n_modes]:
            out += m.contrib
        return out


def reconstruct(sim: Simulation, samples=None) -> Recon:
    """Inverse DFT of the acquired samples, grouped in conjugate pairs.

    samples   a subset of sim.samples (default: all of them), e.g. those acquired up to a given time; a k cell
              without a sample counts as zero.
    """
    S = sim.S
    N, NPIX, H = S.N, S.npix, (S.N - 1) // 2
    samples = sim.samples if samples is None else samples
    table = {(s["kx"], s["ky"]): s["S"] for s in samples}
    get = lambda kx, ky: table.get((kx, ky), 0j)
    X, Y = pixel_xy(N)
    # One representative per conjugate pair, ordered by |k| then angle.
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
    # Ideal contrast image and residual.
    s00 = next((s for s in samples if s["kx"] == 0 and s["ky"] == 0), None)
    ideal = np.hypot(sim.Mx[s00["idx"]], sim.My[s00["idx"]]) if s00 else np.zeros(NPIX)
    mag = np.abs(image)
    den = float((ideal ** 2).sum())
    c = float((mag * ideal).sum() / den) if den > 0 and (mag * ideal).sum() != 0 else 1.0
    refmax = max(1e-9, float(ideal.max()))
    residual = (mag / c - ideal) / refmax
    return Recon(N, modes, image, ideal, c, residual, len(samples))
