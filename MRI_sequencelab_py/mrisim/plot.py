# By Bruno Mota - METAbio
"""Static matplotlib views of a simulation.

plot_diagram   the sequence diagram: rows RF, Gpe (y), Gfe (x), Signal, with the bands, arrows and marks
               declared in the sequence file, the |S(t)| envelope, the readouts and the acquired samples,
               and an optional time cursor
plot_pixels    the pixel array at one instant: cells colored by proton density, tissue or transverse phase,
               with the transverse magnetization (red arrow) and Mz (blue arrow, drawn along y)
plot_kspace    the acquired samples on the k grid (hue = phase, intensity = |S|) with the k marker
plot_image     the reconstruction, the ideal image or the residual
overview       one figure with all of the above at a chosen time
animate        the same as a frame-by-frame animation written to a .gif or .mp4 file

Idle gaps longer than GAPMIN ms (TR tails, inversion delays) are drawn compressed to GAP_W ms of width, with
dashed baselines; the Layout class holds the piecewise-linear map between time and drawing coordinate.
The interactive viewer (viewer.py) reuses kspace_rgb and plot_diagram.
"""
from __future__ import annotations

import colorsys
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .engine import Schedule, Simulation, loc
from .recon import Recon, reconstruct

GAPMIN = 25.0      # ms; idle gaps longer than this are compressed
GAP_W = 10.0       # ms of drawing width given to a compressed gap
COL = {"rf": "#2e7d2a", "gy": "#d4701c", "gx": "#2b5aa8", "sig": "#1f4e9a", "env": "#9aa8bd",
       "te": "#1a9e2a", "tr": "#e0231d", "esp": "#7b1fa2", "cursor": "#c2185b"}


# =============================================================================================== layout

class Layout:
    """Piecewise-linear drawing coordinate u(t): to scale inside and around events, compressed across idle gaps.

    segs   list of (t0, t1, u0, u1, compressed) segments covering [0, t_end]
    u_end  total drawing width
    """

    def __init__(self, S: Schedule, compress: bool = True, margin: float = 2.0):
        busy = sorted((max(0.0, e.t - margin), min(S.t_end, e.t1 + margin)) for e in S.events if e.type != "spoil")
        merged = []                                       # union of the busy intervals
        for a, b in busy:
            if merged and a <= merged[-1][1] + 1e-9:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        self.segs = []
        u = cur = 0.0

        def push(t0, t1, c):
            nonlocal u
            if t1 <= t0 + 1e-9:
                return
            du = GAP_W if c else t1 - t0
            self.segs.append((t0, t1, u, u + du, c))
            u += du

        for a, b in merged:
            if a > cur:
                push(cur, a, compress and a - cur > GAPMIN)
            push(a, b, False)
            cur = b
        if cur < S.t_end:
            push(cur, S.t_end, compress and S.t_end - cur > GAPMIN)
        self.u_end = u

    def u(self, t):
        """Drawing coordinate of time t (clamped to the drawn range)."""
        t = min(max(t, self.segs[0][0]), self.segs[-1][1])
        for t0, t1, u0, u1, _ in self.segs:
            if t0 <= t <= t1:
                return u0 + (t - t0) * (u1 - u0) / (t1 - t0)
        return self.u_end

    def t(self, u):
        """Time of drawing coordinate u (inverse of u(t))."""
        for t0, t1, u0, u1, _ in self.segs:
            if u0 <= u <= u1:
                return t0 + (u - u0) * (t1 - t0) / (u1 - u0)
        return self.segs[-1][1]

    def pieces(self, t0, t1):
        """The time interval [t0, t1] cut at the segment boundaries: list of (a, b, compressed)."""
        return [(max(t0, a), min(t1, b), c) for a, b, _, _, c in self.segs if min(t1, b) > max(t0, a) + 1e-9]


def axis_ticks(N: int) -> list:
    """Tick positions of a pixel or k axis that runs from -(N-1)/2 to (N-1)/2: every integer while the labels
    fit, then every 2, every 5, every 10."""
    H = (N - 1) // 2
    step = 1 if N <= 9 else 2 if N <= 15 else 5 if N <= 41 else 10
    return [v for v in range(-H, H + 1) if v % step == 0]


def _sinc(ax, xc, y0, halfw, A, color):
    """Draw a schematic RF pulse (apodized sinc) centered at xc with half-width halfw and height A."""
    u = np.linspace(-1, 1, 91)
    x = u * 2.6
    s = np.where(np.abs(x) < 1e-9, 1.0, np.sin(np.pi * x) / (np.pi * x + 1e-12))
    ap = np.cos(np.pi * u / 2) ** 0.6
    ax.plot(xc + u * halfw, y0 + A * s * ap, color=color, lw=1.3)


def _tex(label: str) -> str:
    """Convert a label with a_b, a_{bc} or a^b markup into matplotlib mathtext."""
    if "_" in label or "^" in label:
        return "$" + label.replace("ef", r"\mathrm{ef}").replace("E,", "E,") + "$"
    return label


# =============================================================================================== diagram

def plot_diagram(S: Schedule, sim: Simulation | None = None, ax=None, compress: bool = True,
                 t_cursor: float | None = None, lang: str = "pt"):
    """Draw the sequence diagram of a Schedule on `ax` (a new figure if None).

    With `sim`, the signal row also shows |S(t)| (gray), the readouts (real part in blue, filled envelope) and
    the samples as dots colored by phase.  `t_cursor` draws a vertical cursor.  Returns (ax, Layout).
    """
    L = Layout(S, compress)
    if ax is None:
        _, ax = plt.subplots(figsize=(min(16, 3 + L.u_end / 60), 4.2))
    rows = {"rf": 3.0, "gy": 2.0, "gx": 1.0, "sig": 0.0}
    xu = L.u
    # Alternating shot bands with their labels, and the baselines (dashed where compressed).
    for i, b in enumerate(S.bands):
        x0, x1 = xu(b.t), xu(min(b.t1, S.t_end))
        if i % 2 == 1:
            ax.add_patch(Rectangle((x0, -0.9), x1 - x0, 4.6, color="#f2f2f2", zorder=0))
        if len(S.bands) <= 16:                            # band, flip-angle and readout labels only while they fit
            ax.text((x0 + x1) / 2, 3.72, b.label, ha="center", va="bottom", fontsize=8)
    for key, y in rows.items():
        for a, b, dashed in L.pieces(0, S.t_end):
            ax.plot([xu(a), xu(b)], [y, y], color="#8f8f8f", lw=0.8, ls=(0, (3, 3)) if dashed else "-")
    ax.set_yticks(list(rows.values()))
    ax.set_yticklabels(["RF", "G$_{pe}$ = ∂B$_z$/∂y", "G$_{fe}$ = ∂B$_z$/∂x", "Signal"])
    # RF pulses and gradient lobes (gradient heights relative to the largest readout amplitude).
    grads = [e for e in S.events if e.type == "grad"]
    amp_ref = max([abs(g.amp) for g in grads if g.axis == "x"] + [1e-9])
    n_rf = sum(e.type == "rf" for e in S.events)
    for e in S.events:
        if e.type == "rf":
            A = max(0.25, min(0.7, 0.4 + 0.3 * (abs(e.flip) - 90) / 90))
            _sinc(ax, xu(e.tc), rows["rf"], (xu(e.t1) - xu(e.t)) * 0.45, A, COL["rf"])
            if n_rf <= 40:
                ax.text(xu(e.tc), rows["rf"] + A + 0.08, f"{e.flip:g}°", ha="center", va="bottom", fontsize=8)
        elif e.type == "grad":
            y0 = rows["gy"] if e.axis == "y" else rows["gx"]
            h = 0.45 * e.amp / amp_ref
            col = COL["gy"] if e.axis == "y" else COL["gx"]
            if abs(e.area) < 1e-9:
                ax.plot([xu(e.t), xu(e.t1)], [y0, y0], color=col, lw=3)
            else:
                ax.add_patch(Rectangle((xu(e.t), min(y0, y0 + h)), xu(e.t1) - xu(e.t), abs(h), color=col))
            if e.label and S.N <= 7:                      # lobe labels only while they still fit
                ax.text(xu(e.t1) + 0.5, y0 + h + (0.04 if h >= 0 else -0.14), e.label, color=col, fontsize=7, fontweight="bold")
    # Signal row.
    if sim is not None:
        ys = lambda v: rows["sig"] + 0.65 * v / sim.Smax
        mag = np.abs(sim.signal)
        ax.plot([xu(t) for t in sim.times], ys(mag), color=COL["env"], lw=0.7)
        for a in S.adcs:
            i0, i1 = sim.index_at(a.t - 0.8), sim.index_at(a.t1 + 0.8)
            tt = [xu(t) for t in sim.times[i0:i1 + 1]]
            ax.fill_between(tt, rows["sig"], ys(mag[i0:i1 + 1]), color="#dbe5f6")
            ax.plot(tt, ys(sim.signal.real[i0:i1 + 1]), color=COL["sig"], lw=1.2)
            for s in sim.samples:
                if s["adc"] is a:
                    ax.plot(xu(s["t"]), ys(s["S"].real), "o", ms=3.5, color=_phase_color(np.angle(s["S"])), mec="#333", mew=0.5)
            if a.d.get("label") and len(S.adcs) <= 16:
                ax.text(xu(a.tc) + 1, rows["sig"] + 0.72, loc(a.d["label"], lang), color=COL["sig"], fontsize=8)
    # Marks (dashed verticals) and arrows (horizontal spans below the signal row).
    for m in S.marks:
        col = COL.get(m.style or "te", COL["te"])
        ax.plot([xu(m.t)] * 2, [-0.6 - 0.3 * m.level, rows["rf"] + 0.3], color=col, lw=0.7, ls=(0, (4, 3)))
        if m.label:
            ax.text(xu(m.t) + 0.6, rows["rf"] - 0.3, _tex(m.label), color=col, fontsize=8, fontweight="bold")
    for a in S.arrows:
        col = COL.get(a.style or "te", COL["te"])
        y = -0.6 - 0.3 * a.level
        for p0, p1, dashed in L.pieces(a.t, a.t1):
            ax.plot([xu(p0), xu(p1)], [y, y], color=col, lw=2.5, ls=(0, (3, 3)) if dashed else "-")
        ax.plot([xu(a.t), xu(a.t1)], [y, y], "|", color=col, ms=9, mew=2)
        ax.text((xu(a.t) + xu(a.t1)) / 2, y - 0.1, _tex(a.label), color=col, fontsize=10, fontweight="bold", ha="center", va="top")
    if t_cursor is not None:
        ax.axvline(xu(t_cursor), color=COL["cursor"], lw=1.2)
    ax.set_xlim(-2, L.u_end + 2)
    ax.set_ylim(-1.4, 4.1)
    ax.set_xticks([])
    ax.set_xlabel("tempo (trechos tracejados comprimidos)" if lang == "pt" else "time (dashed stretches compressed)")
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    return ax, L


# =============================================================================================== colors

def _phase_color(phi, light=0.46):
    """RGB color encoding a phase angle as hue (0 = red, 2 pi/3 = green, 4 pi/3 = blue)."""
    h = (phi / (2 * np.pi)) % 1.0
    return colorsys.hls_to_rgb(h, light, 0.85)


def kspace_rgb(K: np.ndarray, Smax: float, bg=(1, 1, 1), sqrt_intensity: bool = True) -> np.ndarray:
    """Color-code a complex grid (k-space samples, or pixel magnetizations): hue = phase, intensity = |value|/Smax
    (square-rooted by default, to make weak samples visible), blended with the background color; NaN entries
    (not acquired) stay at the background."""
    out = np.tile(np.asarray(bg, float), K.shape + (1,))
    for idx in np.ndindex(K.shape):
        v = K[idx]
        if np.isnan(v):
            continue
        a = min(1.0, abs(v) / Smax)
        a = math.sqrt(a) if sqrt_intensity else a
        c = np.array(_phase_color(np.angle(v)))
        out[idx] = np.asarray(bg) + a * (c - np.asarray(bg))
    return out


# =============================================================================================== panels

def plot_pixels(sim: Simulation, i: int, ax=None, phantom=None, mode: str = "density"):
    """Pixel array at time index i.

    mode   "density": gray = proton density;  "tissue": the phantom's tissue colors (needs `phantom`);
           "phase": hue = phase of M_perp, intensity = |M_perp|
    Arrows: red = transverse magnetization (Mx, My); blue = Mz, drawn along y for lack of a third axis.
    """
    N = sim.S.N
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    m0max = max(1e-9, sim.M0.max())
    if mode == "tissue" and phantom is not None:
        img = phantom.colors()
    elif mode == "phase":
        K = (sim.Mx[i] + 1j * sim.My[i]).reshape(N, N)
        img = kspace_rgb(K, m0max)
    else:
        g = (0.1 + 0.9 * np.minimum(1, sim.M0 / m0max)).reshape(N, N)
        img = np.stack([g, g, g], -1)
    ax.imshow(img, extent=(-N / 2, N / 2, -N / 2, N / 2), interpolation="nearest")
    X = np.arange(N) - (N - 1) / 2
    xx, yy = np.meshgrid(X, X[::-1])
    mx, my, mz = sim.Mx[i].reshape(N, N), sim.My[i].reshape(N, N), sim.Mz[i].reshape(N, N)
    ax.quiver(xx, yy, mx, my, color="#d62828", angles="xy", scale_units="xy", scale=m0max / 0.42, width=0.012)
    ax.quiver(xx, yy, np.zeros_like(mz), mz, color="#1d5fd1", angles="xy", scale_units="xy", scale=m0max / 0.42, width=0.008, alpha=0.8)
    ax.set_xticks(axis_ticks(N))
    ax.set_yticks(axis_ticks(N))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{N}×{N} pixels: M$_\\perp$ (red), M$_z$ (blue, drawn along y)", fontsize=9)
    return ax


def plot_kspace(sim: Simulation, i: int | None = None, ax=None, sqrt_intensity: bool = True):
    """The samples acquired up to time index i (all of them if None) on the k grid, with the k marker and its
    recent trail when i is given."""
    N, H = sim.S.N, (sim.S.N - 1) // 2
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    samples = sim.samples if i is None else [s for s in sim.samples if s["idx"] <= i]
    K = np.full((N, N), np.nan + 0j)
    for s in samples:
        K[H - s["ky"], s["kx"] + H] = s["S"]
    ax.imshow(kspace_rgb(K, sim.Smax, bg=(0.91, 0.93, 0.95), sqrt_intensity=sqrt_intensity),
              extent=(-N / 2, N / 2, -N / 2, N / 2), interpolation="nearest")
    if i is not None:
        ax.plot(sim.kx[i], sim.ky[i], "o", color=COL["cursor"], mec="w", ms=7)
        j0 = max(0, i - 2000)
        ax.plot(sim.kx[j0:i + 1], sim.ky[j0:i + 1], color=COL["cursor"], lw=0.8, alpha=0.6)
    ax.set_xticks(axis_ticks(N))
    ax.set_yticks(axis_ticks(N))
    ax.set_xlabel("k$_x$")
    ax.set_ylabel("k$_y$")
    ax.set_title("k-space: hue = phase, intensity = |S|", fontsize=9)
    return ax


def plot_image(rec: Recon, ax=None, n_modes: int | None = None, what: str = "image"):
    """One of the reconstruction panels.

    what   "image": |sum of the first n_modes modes| (all modes if None), gray scale up to the full image's maximum;
           "ideal": the sequence's ideal contrast image;  "residual": (|image|/c - ideal)/max, mid-gray = 0
    """
    N = rec.N
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    ext = (-N / 2, N / 2, -N / 2, N / 2)
    if what == "residual":
        ax.imshow(rec.grid(rec.residual), cmap="gray", vmin=-1, vmax=1, extent=ext, interpolation="nearest")
        ax.set_title(f"residual (|image|/c − ideal)/max, max |Δ| = {np.abs(rec.residual).max():.3f}", fontsize=9)
    elif what == "ideal":
        ax.imshow(rec.grid(rec.ideal), cmap="gray", vmin=0, extent=ext, interpolation="nearest")
        ax.set_title("ideal image: |M$_\\perp$| at the k = 0 sample", fontsize=9)
    else:
        img = np.abs(rec.partial(n_modes) if n_modes is not None else rec.image)
        vmax = max(1e-9, np.abs(rec.image).max())
        ax.imshow(rec.grid(img), cmap="gray", vmin=0, vmax=vmax, extent=ext, interpolation="nearest")
        ax.set_title(f"|Σ modes| ({n_modes if n_modes is not None else len(rec.modes)}/{len(rec.modes)})", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def overview(sim: Simulation, phantom=None, t: float | None = None, compress: bool = True, lang: str = "pt"):
    """One static figure: the diagram with a cursor at t (end of the sequence if None) on top; below it the pixel
    array and k-space at that instant, and the full reconstruction, the ideal image and the residual.
    Returns (figure, Recon)."""
    rec = reconstruct(sim)
    i = sim.index_at(t) if t is not None else len(sim.times) - 1
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.1, 1.3])
    axd = fig.add_subplot(gs[0, :])
    plot_diagram(sim.S, sim, ax=axd, compress=compress, t_cursor=t, lang=lang)
    plot_pixels(sim, i, fig.add_subplot(gs[1, 0]), phantom, "tissue" if phantom is not None else "density")
    plot_kspace(sim, i, fig.add_subplot(gs[1, 1]))
    plot_image(rec, fig.add_subplot(gs[1, 2]), what="image")
    plot_image(rec, fig.add_subplot(gs[1, 3]), what="ideal")
    plot_image(rec, fig.add_subplot(gs[1, 4]), what="residual")
    fig.suptitle(loc(sim.S.defn.get("name"), lang) + " · " + loc(sim.S.defn.get("subtitle"), lang), fontsize=11)
    fig.tight_layout()
    return fig, rec


def animate(sim: Simulation, phantom=None, path: str = "sequence.mp4", frames: int = 200, fps: int = 20,
            compress: bool = True, lang: str = "pt"):
    """Write a timeline animation (cursor on the diagram, pixel arrows, k-space filling) to `path`.

    The frames are equally spaced in the drawing coordinate, so compressed gaps pass quickly.  A .gif needs
    pillow, anything else ffmpeg.  Returns the path.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    L = Layout(sim.S, compress)
    us = np.linspace(0, L.u_end, frames)
    fig = plt.figure(figsize=(14, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3])
    axd, axp, axk = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    def draw(f):
        t = L.t(us[f])
        i = sim.index_at(t)
        for ax in (axd, axp, axk):
            ax.cla()
        plot_diagram(sim.S, sim, ax=axd, compress=compress, t_cursor=t, lang=lang)
        plot_pixels(sim, i, axp, phantom, "tissue" if phantom is not None else "density")
        plot_kspace(sim, i, axk)
        axd.set_title(f"t = {t:.1f} ms", fontsize=9, loc="left")

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps)
    if path.endswith(".gif"):
        anim.save(path, writer=PillowWriter(fps=fps))
    else:
        anim.save(path, fps=fps)
    plt.close(fig)
    return path
