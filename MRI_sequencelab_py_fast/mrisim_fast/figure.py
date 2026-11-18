# By Bruno Mota - METAbio
"""One static figure for an Acquisition: the object, k-space, the image, the ideal image and the residual."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from .engine import Acquisition
from .recon import Image


def _tissue_map(phantom) -> np.ndarray:
    """The phantom as an RGB picture in the colors of the tissue dictionary."""
    N = int(round(len(phantom.label) ** 0.5))
    rgb = np.array([to_rgb(phantom.tissues[k].get("color", "#888888")) for k in phantom.label])
    return rgb.reshape(N, N, 3)


def show(acq: Acquisition, img: Image, phantom=None, title: str | None = None, out: str | None = None, block: bool = True):
    """Draw the figure; save it to `out` if given; open a window unless block is False.  Returns the figure.

    Panels: the tissue map (if a phantom is given); log10 |S(k)| on the k grid; |image|; the ideal image; the
    residual on a symmetric gray scale (mid-gray = 0).
    """
    N, H = acq.N, (acq.N - 1) // 2
    panels = (["object"] if phantom is not None else []) + ["k", "image"] + (["ideal", "residual"] if img.ideal is not None else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 4.1), squeeze=False)
    ext = (-N / 2, N / 2, -N / 2, N / 2)
    for ax, what in zip(axes[0], panels):
        if what == "object":
            ax.imshow(_tissue_map(phantom), extent=ext, interpolation="nearest")
            ax.set_title(f"object, {N} × {N}", fontsize=9)
        elif what == "k":
            K = np.abs(acq.kspace())
            floor = 1e-6 * acq.Smax
            ax.imshow(np.log10(np.where(np.isnan(K), floor, np.maximum(K, floor)) / acq.Smax), extent=ext, cmap="magma", interpolation="nearest")
            ax.set_title("k-space: log$_{10}$ |S| / ΣM$_0$" + ("" if not np.isnan(K).any() else " (dark: not sampled)"), fontsize=9)
            ax.set_xlabel("k$_x$")
            ax.set_ylabel("k$_y$")
        elif what == "image":
            ax.imshow(np.abs(img.image), extent=ext, cmap="gray", vmin=0, interpolation="nearest")
            ax.set_title(f"|image|, {img.n_acquired} samples", fontsize=9)
        elif what == "ideal":
            ax.imshow(img.ideal, extent=ext, cmap="gray", vmin=0, interpolation="nearest")
            ax.set_title("ideal image: |M$_\\perp$| at the k = 0 sample", fontsize=9)
        else:
            r = max(1e-9, float(np.abs(img.residual).max()))
            ax.imshow(img.residual, extent=ext, cmap="gray", vmin=-r, vmax=r, interpolation="nearest")
            ax.set_title(f"residual, max |Δ| = {r:.4f}", fontsize=9)
        if what != "k":
            ax.set_xlabel("x")
            ax.set_ylabel("y")
    fig.suptitle(title or "", fontsize=10)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
    if block:
        plt.show()
    return fig
