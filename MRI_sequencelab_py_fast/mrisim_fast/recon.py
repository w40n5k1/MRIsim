# By Bruno Mota - METAbio
"""Image from the samples of an Acquisition, by fast Fourier transform.

The definition is the one of mrisim.recon, without the list of modes.  The samples are the forward DFT of the
pixel magnetizations, S(k) = sum_r m(r) exp(-2 pi i k.r / N).  The object is real up to the phase of the echo,
phi0 = arg S(0), which is removed first; the image is then the real part of the inverse DFT,
    image(r) = Re[ (1/N^2) sum_k S(k) exp(-i phi0) exp(+2 pi i k.r / N) ],
with zero in the cells that were not sampled and the last sample in the cells sampled more than once.  The
ideal image is |M_perp| of each pixel when the k = 0 sample was taken, and
    residual = (|image| / c - ideal) / max(ideal),      c = sum(|image| ideal) / sum(ideal^2).
mrisim.reconstruct builds the same image as a sum of (N^2 + 1)/2 cosine modes, which is what one wants to watch
and costs N^4 numbers; the FFT costs N^2 log N.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine import Acquisition


@dataclass
class Image:
    """A reconstruction.

    N            matrix size
    image        (N, N) real array, row 0 at the top (y = +HALF), column 0 at the left (x = -HALF); the picture
                 shown is its magnitude, np.abs(image)
    phase0       the phase of the echo that was removed, arg S(0), in radians
    ideal        (N, N): |M_perp| of each pixel at the k = 0 sample of the samples used; None if not available
    c            least-squares scale between |image| and ideal
    residual     (N, N): (|image| / c - ideal) / max(ideal); None without an ideal image
    n_acquired   number of samples used
    """
    N: int
    image: np.ndarray
    phase0: float
    ideal: np.ndarray | None
    c: float
    residual: np.ndarray | None
    n_acquired: int


def reconstruct_fast(acq: Acquisition, mask=None) -> Image:
    """Inverse FFT of the samples of an Acquisition.

    mask   boolean array over the samples (see Acquisition.select): the samples to use.  Default: all.

    The phase of the echo comes from the first k = 0 sample among those used; if there is none, from the first
    k = 0 sample of the whole acquisition.
    """
    N, H = acq.N, (acq.N - 1) // 2
    use = np.ones(len(acq), bool) if mask is None else np.asarray(mask, bool)
    center = (acq.kx == 0) & (acq.ky == 0)
    first = np.flatnonzero(center & use)
    ref = first[0] if len(first) else (np.flatnonzero(center)[0] if center.any() else None)
    phase0 = float(np.angle(acq.signal[ref])) if ref is not None and abs(acq.signal[ref]) > 1e-12 * max(acq.Smax, 1e-300) else 0.0
    # k-space table in FFT order: index (k_y mod N, k_x mod N).  Samples are time-ordered, so in a cell sampled
    # more than once the last sample stays.
    K = np.zeros((N, N), complex)
    q = np.flatnonzero(use)
    K[acq.ky[q] % N, acq.kx[q] % N] = acq.signal[q] * np.exp(-1j * phase0)
    m = np.fft.ifft2(K)                               # index (y mod N, x mod N)
    rows = (H - np.arange(N)) % N                     # row r of the picture is y = HALF - r
    cols = (np.arange(N) - H) % N                     # column c is x = c - HALF
    image = m[np.ix_(rows, cols)].real
    ideal = acq.ideal.get(int(first[0])) if len(first) else None
    if ideal is None:
        return Image(N, image, phase0, None, 1.0, None, int(use.sum()))
    mag = np.abs(image)
    num, den = float((mag * ideal).sum()), float((ideal ** 2).sum())
    c = num / den if den > 0 and num != 0 else 1.0
    residual = (mag / c - ideal) / max(1e-9, float(ideal.max()))
    return Image(N, image, phase0, ideal, c, residual, int(use.sum()))
