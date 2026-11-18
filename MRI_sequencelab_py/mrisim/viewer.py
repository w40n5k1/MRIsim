# By Bruno Mota - METAbio
"""Interactive matplotlib viewer.

    from mrisim import lab, view
    lab()                                   # selectors for sequence, phantom, N and the sequence's parameters
    lab(sequences_dir, phantoms_dir, tissues_path, N=9, lang="en")
    view(sim, phantom)                      # one prebuilt simulation, no selectors

Layout of the window
--------------------
Left column (when selectors are on): the sequence (every file in the sequences folder), the phantom (every
file in the phantoms folder), the matrix size N (odd, 3 to 35), and the parameters declared by the chosen
sequence file (sliders, option lists and checkboxes).  Any change recompiles and resimulates; sliders apply when
the mouse button is released, since they fire continuously while dragged.  The whole history of every pixel is
kept for the time slider, about 40 MB at N = 15 and 0.6 GB at N = 35 for a spin echo, so large N is slow to
rebuild; for images without the history use the mrisim_fast package.
Top: the sequence diagram with a time cursor.  Bottom row: the pixel array (cells colored by tissue, proton
density or transverse phase; arrows for M_perp and Mz), k-space filling in as the samples are acquired, the
partial reconstruction |sum of the first n modes|, the mode being added (green frame; its two conjugate cells are
boxed in k-space), and the ideal image, replaced by the residual once every mode is in.
Controls: a time slider with a play button (the slider runs in the diagram's drawing coordinate, so compressed
gaps pass quickly); a mode slider with a Fourier play button; cell-color mode; residual gain.

Implementation notes
--------------------
The static parts (the diagram, the panel frames) are drawn once per model; the time slider only moves the
cursor, updates the quiver arrows and the two color images.  The selectors (two radios and the N slider) are
permanent; everything else is rebuilt by _rebuild() after a model change, which removes the dynamic axes and recreates them.  Widgets
are kept in self.widgets so that matplotlib does not garbage-collect their callbacks.
"""
from __future__ import annotations

import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

from .engine import Simulation, compile_sequence, load_json, loc, param_defaults, simulate
from .phantoms import load_tissues, render_phantom
from .plot import COL, Layout, axis_ticks, kspace_rgb, plot_diagram
from .recon import reconstruct

ROOT = pathlib.Path(__file__).resolve().parent.parent      # folder holding sequences/, phantoms/, tissues.json
ALIEN = "#39ff14"                                          # frame color of the current-mode panel
N_MIN, N_MAX = 3, 35                                       # matrix sizes offered (odd): the stored history grows roughly as N^4
# Didactic order of the built-in sequences and phantoms in the selectors; other files follow alphabetically.
ORDER = ["se", "fse", "flair", "squareAir", "letterFAir", "uniform", "square", "letterF", "ramp", "brain", "face", "monalisa"]

# Interface texts.
TXT = {
    "pt": dict(t="t", play="▶ aquisição", pause="❚❚ pausa", modes="modos", fourier="▶ Fourier", fpause="❚❚ pausa",
               cell=("tecido", "densidade", "fase"), image="|Σ modos| ({}/{})", mode="modo k = ({}, {}), A = {:.3f}",
               resid="resíduo (|imagem|/c − ideal)/máx", ideal="imagem ideal: |M⊥| em k = 0", gain="ganho do resíduo",
               seq="sequência", phantom="fantoma", n="N × N pixels", params="parâmetros"),
    "en": dict(t="t", play="▶ acquisition", pause="❚❚ pause", modes="modes", fourier="▶ Fourier", fpause="❚❚ pause",
               cell=("tissue", "density", "phase"), image="|Σ modes| ({}/{})", mode="mode k = ({}, {}), A = {:.3f}",
               resid="residual (|image|/c − ideal)/max", ideal="ideal image: |M⊥| at k = 0", gain="residual gain",
               seq="sequence", phantom="phantom", n="N × N pixels", params="parameters"),
}


def _small(widget, size=8):
    """Set the font size of a RadioButtons / CheckButtons widget's labels; returns the widget."""
    for l in widget.labels:
        l.set_fontsize(size)
    return widget


class _TextHolder:
    """Wraps figure-level texts so that they can be removed together with the dynamic axes."""

    def __init__(self, texts):
        self.texts = texts

    def remove(self):
        for t in self.texts:
            t.remove()


class Lab:
    """The interactive window.

    sequences   {id: sequence definition (dict)}
    phantoms    {id: phantom definition (dict)}
    tissues     the tissue dictionary
    N           initial matrix size
    seq_id, phantom_id   initial choices (defaults: first sequence, and the phantom it suggests)
    compress    draw idle gaps compressed
    lang        'pt' or 'en'
    fps, duration   playback rate and the duration of one full acquisition playback, in seconds
    selectors   show the left column
    fixed       (Simulation, Phantom) to display one prebuilt simulation instead of simulating (used by view())

    After construction, self.sim and self.rec hold the current simulation and its reconstruction, self.P the
    current parameter values, self.widgets the live widgets.
    """

    def __init__(self, sequences: dict, phantoms: dict, tissues: dict, N: int = 5, seq_id: str | None = None,
                 phantom_id: str | None = None, compress: bool = True, lang: str = "pt", fps: int = 25,
                 duration: float = 12.0, selectors: bool = True, fixed: tuple | None = None):
        self.seqs, self.phs, self.tissues = sequences, phantoms, tissues
        self.lang, self.compress, self.fps, self.duration = lang, compress, fps, duration
        self.T = TXT.get(lang, TXT["en"])
        self.selectors = selectors
        self.fixed = fixed
        self.seq_id = seq_id or next(iter(sequences))
        self.N = N
        self.P = param_defaults(sequences[self.seq_id])
        self.ph_id = phantom_id or sequences[self.seq_id].get("phantom") or next(iter(phantoms))
        if self.ph_id not in phantoms:
            self.ph_id = next(iter(phantoms))
        grid = phantoms[self.ph_id].get("grid") if fixed is None else None
        if grid and N_MIN <= int(grid.get("N", len(grid["labels"]))) <= N_MAX:
            self.N = int(grid.get("N", len(grid["labels"])))          # a painted phantom brings its own N
        self.cell_mode = "tissue"
        self.gain = 1.0
        self.playing = self.fplaying = False      # acquisition playback / Fourier playback running
        self._pending = False                     # a slider change waiting for the mouse release
        self._quiet = False                       # suppress selector callbacks during programmatic changes
        self._fcount = 0                          # tick counter for the Fourier playback
        self.dyn_axes: list = []                  # axes (and text holders) removed at every rebuild
        self.widgets: dict = {}

        self.fig = plt.figure(figsize=(17.5, 9.6))
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.timer = self.fig.canvas.new_timer(interval=int(1000 / fps))
        self.timer.add_callback(self._tick)
        if selectors:
            self._build_selectors()
        self._rebuild()
        self.timer.start()

    # ------------------------------------------------------------------ model

    def _simulate(self):
        """Render the chosen phantom at N, compile the chosen sequence with the current parameters, simulate."""
        if self.fixed is not None:
            self.sim, self.ph = self.fixed
            return
        defn = self.seqs[self.seq_id]
        self.ph = render_phantom(self.phs[self.ph_id], self.N, self.tissues)
        S = compile_sequence(defn, self.P, self.lang, self.N)
        self.sim = simulate(S, self.ph.M0, self.ph.T1, self.ph.T2)

    def _rebuild(self):
        """Resimulate and rebuild everything except the permanent selectors."""
        self.playing = self.fplaying = False
        self._simulate()
        sim, N = self.sim, self.sim.S.N
        self.H = (N - 1) // 2
        self.rec = reconstruct(sim)
        self.L = Layout(sim.S, self.compress)
        self.ext = (-N / 2, N / 2, -N / 2, N / 2)
        self.m0max = max(1e-9, sim.M0.max())
        self.n_modes = len(self.rec.modes)
        self.du = self.L.u_end / (self.fps * self.duration)     # slider advance per tick during playback
        for ax in self.dyn_axes:
            ax.remove()
        self.dyn_axes = []
        for k in [k for k in self.widgets if k not in ("seq", "ph", "n")]:
            self.widgets.pop(k)
        self._build_params()
        self._build_panels()
        self._build_controls()
        self._on_t(self.s_t.val)
        self._on_modes(self.n_modes)
        self.fig.canvas.draw_idle()

    def _ax(self, rect, **kw):
        """Add an axes that will be removed at the next rebuild."""
        ax = self.fig.add_axes(rect, **kw)
        self.dyn_axes.append(ax)
        return ax

    # ------------------------------------------------------------------ left column

    def _build_selectors(self):
        """The permanent selectors: sequence and phantom radios, and the N slider."""
        fig, T, lang = self.fig, self.T, self.lang
        x0, w = 0.012, 0.15
        y = 0.95
        fig.text(x0, y, T["seq"], fontsize=9, fontweight="bold")
        y -= 0.012
        self.seq_ids = list(self.seqs)
        h = 0.028 * len(self.seq_ids)
        r = RadioButtons(fig.add_axes([x0, y - h, w, h], frameon=False),
                         [loc(self.seqs[k].get("name"), lang) for k in self.seq_ids],
                         active=self.seq_ids.index(self.seq_id))
        _small(r)
        r.on_clicked(self._on_seq)
        self.widgets["seq"] = r
        y -= h + 0.03

        fig.text(x0, y, T["phantom"], fontsize=9, fontweight="bold")
        y -= 0.012
        self.ph_ids = list(self.phs)
        h = 0.026 * len(self.ph_ids)
        r = RadioButtons(fig.add_axes([x0, y - h, w, h], frameon=False),
                         [loc(self.phs[k].get("name"), lang) for k in self.ph_ids],
                         active=self.ph_ids.index(self.ph_id))
        _small(r)
        r.on_clicked(self._on_ph)
        self.widgets["ph"] = r
        y -= h + 0.03

        fig.text(x0, y, T["n"], fontsize=9, fontweight="bold")
        y -= 0.012
        h = 0.02
        s = Slider(fig.add_axes([x0 + 0.035, y - h, w - 0.045, h]), "N", N_MIN, N_MAX, valinit=self.N, valstep=2,
                   valfmt="%d")
        s.label.set_fontsize(8)
        s.valtext.set_fontsize(8)
        s.on_changed(self._on_n)
        self.widgets["n"] = s
        y -= h + 0.03
        self.y_params = y                         # where the parameter widgets start

    def _build_params(self):
        """One widget per parameter declared in the sequence file: slider, option list or checkbox."""
        if not self.selectors or self.fixed is not None:
            return
        fig, T, lang = self.fig, self.T, self.lang
        x0, w = 0.012, 0.15
        y = self.y_params
        defn = self.seqs[self.seq_id]
        texts = [fig.text(x0, y, T["params"], fontsize=9, fontweight="bold")]
        y -= 0.02
        for p in defn.get("params", []):
            pid, typ = p["id"], p.get("type", "slider")
            if typ == "checkbox":
                h = 0.03
                c = CheckButtons(self._ax([x0, y - h, w, h], frameon=False), [loc(p["label"], lang)], [bool(self.P.get(pid))])
                _small(c)
                c.on_clicked(lambda _label, pid=pid, c=c: self._set_param(pid, 1 if c.get_status()[0] else 0))
                self.widgets[pid] = c
                y -= h + 0.01
            elif typ == "select":
                values = [o["value"] for o in p["options"]]
                h = 0.026 * len(values)
                texts.append(fig.text(x0, y, loc(p["label"], lang), fontsize=8))
                y -= 0.012
                r = RadioButtons(self._ax([x0, y - h, w, h], frameon=False), [loc(o["label"], lang) for o in p["options"]],
                                 active=values.index(self.P.get(pid, p["default"])))
                _small(r)
                r.on_clicked(lambda label, pid=pid, r=r, values=values:
                             self._set_param(pid, values[[l.get_text() for l in r.labels].index(label)]))
                self.widgets[pid] = r
                y -= h + 0.015
            else:
                h = 0.02
                s = Slider(self._ax([x0 + 0.035, y - h, w - 0.045, h]), loc(p["label"], lang), p["min"], p["max"],
                           valinit=self.P.get(pid, p["default"]), valstep=p.get("step", 1), valfmt="%g " + p.get("unit", ""))
                s.label.set_fontsize(8)
                s.valtext.set_fontsize(8)
                s.on_changed(lambda v, pid=pid: self._set_param(pid, v, deferred=True))
                self.widgets[pid] = s
                y -= h + 0.018
        self.dyn_axes.append(_TextHolder(texts))

    # ------------------------------------------------------------------ main panels

    def _build_panels(self):
        """The diagram and the five bottom panels for the current simulation."""
        sim, N, fig, lang = self.sim, self.sim.S.N, self.fig, self.lang
        left = 0.21 if self.selectors else 0.05
        gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 1.25], left=left, right=0.985, top=0.93, bottom=0.2,
                              hspace=0.35, wspace=0.3)
        self.axd = fig.add_subplot(gs[0, :])
        self.axp, self.axk, self.axi, self.axm, self.axr = (fig.add_subplot(gs[1, j]) for j in range(5))
        self.dyn_axes += [self.axd, self.axp, self.axk, self.axi, self.axm, self.axr]
        self.axd.set_title(loc(sim.S.defn.get("name"), lang) + " · " + loc(sim.S.defn.get("subtitle"), lang), fontsize=10)

        # Diagram, drawn once; only the cursor and the time label change afterwards.
        plot_diagram(sim.S, sim, ax=self.axd, compress=self.compress, t_cursor=None, lang=lang)
        self.cursor = self.axd.axvline(0, color=COL["cursor"], lw=1.2)
        self.tlabel = self.axd.text(0.0, 1.02, "t = 0.0 ms", transform=self.axd.transAxes, fontsize=9)

        # Pixel array: color image plus two quivers (M_perp in red, Mz along y in blue).
        X = np.arange(N) - self.H
        xx, yy = np.meshgrid(X, X[::-1])
        self.img_p = self.axp.imshow(self._cell_img(0), extent=self.ext, interpolation="nearest")
        z = np.zeros((N, N))
        self.q_perp = self.axp.quiver(xx, yy, z, z, color="#d62828", angles="xy", scale_units="xy",
                                      scale=self.m0max / 0.42, width=0.012)
        self.q_z = self.axp.quiver(xx, yy, z, z, color="#1d5fd1", angles="xy", scale_units="xy",
                                   scale=self.m0max / 0.42, width=0.008, alpha=0.8)
        self.axp.set_xticks(axis_ticks(N))
        self.axp.set_yticks(axis_ticks(N))
        self.axp.set_xlabel("x")
        self.axp.set_ylabel("y")
        self.axp.set_title(f"{N}×{N} pixels: M$_\\perp$ (red), M$_z$ (blue, along y)", fontsize=9)

        # k-space: color image, marker with trail, and two boxes for the cells of the current mode.
        self.img_k = self.axk.imshow(self._k_img(0), extent=self.ext, interpolation="nearest")
        self.k_trail, = self.axk.plot([], [], color=COL["cursor"], lw=0.8, alpha=0.6)
        self.k_dot, = self.axk.plot([], [], "o", color=COL["cursor"], mec="w", ms=7)
        self.k_boxes = [self.axk.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec=ALIEN, lw=2.2, visible=False))
                        for _ in range(2)]
        self.axk.set_xticks(axis_ticks(N))
        self.axk.set_yticks(axis_ticks(N))
        self.axk.set_xlabel("k$_x$")
        self.axk.set_ylabel("k$_y$")
        self.axk.set_title("k-space: hue = phase, intensity = |S|", fontsize=9)

        # Partial image, current mode, ideal/residual.
        vmax = max(1e-9, np.abs(self.rec.image).max())
        self.img_i = self.axi.imshow(np.zeros((N, N)), cmap="gray", vmin=0, vmax=vmax, extent=self.ext, interpolation="nearest")
        self.img_m = self.axm.imshow(np.zeros((N, N)), cmap="gray", vmin=-vmax / 2, vmax=vmax / 2, extent=self.ext, interpolation="nearest")
        for sp in self.axm.spines.values():
            sp.set_edgecolor(ALIEN)
            sp.set_linewidth(2.2)
        self.img_r = self.axr.imshow(np.zeros((N, N)), cmap="gray", vmin=-1, vmax=1, extent=self.ext, interpolation="nearest")
        for ax in (self.axi, self.axm, self.axr):
            ax.set_xticks([])
            ax.set_yticks([])

    def _build_controls(self):
        """Time and mode sliders, play buttons, cell-mode and gain radios."""
        T = self.T
        left = 0.24 if self.selectors else 0.08
        wslider = 0.46 if self.selectors else 0.55
        xb = left + wslider + 0.03
        self.s_t = Slider(self._ax([left, 0.11, wslider, 0.025]), T["t"], 0.0, self.L.u_end, valinit=self.L.u_end)
        self.s_t.valtext.set_visible(False)
        self.s_t.on_changed(self._on_t)
        self.b_play = Button(self._ax([xb, 0.105, 0.09, 0.035]), T["play"])
        self.b_play.on_clicked(self._toggle_play)
        self.s_m = Slider(self._ax([left, 0.05, wslider, 0.025]), T["modes"], 0, self.n_modes, valinit=self.n_modes, valstep=1)
        self.s_m.on_changed(self._on_modes)
        self.b_four = Button(self._ax([xb, 0.045, 0.09, 0.035]), T["fourier"])
        self.b_four.on_clicked(self._toggle_fourier)
        self.r_cell = RadioButtons(self._ax([xb + 0.11, 0.035, 0.09, 0.11], frameon=False), T["cell"],
                                   active={"tissue": 0, "density": 1, "phase": 2}[self.cell_mode])
        _small(self.r_cell, 9)
        self.r_cell.on_clicked(self._on_cell)
        self.r_gain = RadioButtons(self._ax([xb + 0.2, 0.035, 0.07, 0.11], frameon=False), ("×1", "×10", "×100"),
                                   active={1.0: 0, 10.0: 1, 100.0: 2}[self.gain])
        _small(self.r_gain, 9)
        self.r_gain.on_clicked(self._on_gain)
        self.gain_txt = self.fig.text(xb + 0.205, 0.15, T["gain"], fontsize=9)
        self.dyn_axes.append(_TextHolder([self.gain_txt]))
        self.widgets.update(s_t=self.s_t, b_play=self.b_play, s_m=self.s_m, b_four=self.b_four,
                            r_cell=self.r_cell, r_gain=self.r_gain)

    # ------------------------------------------------------------------ images

    def _cell_img(self, i):
        """RGB image of the pixel cells at time index i in the current cell mode."""
        N = self.sim.S.N
        if self.cell_mode == "tissue":
            return self.ph.colors()
        if self.cell_mode == "phase":
            return kspace_rgb((self.sim.Mx[i] + 1j * self.sim.My[i]).reshape(N, N), self.m0max)
        g = (0.1 + 0.9 * np.minimum(1, self.sim.M0 / self.m0max)).reshape(N, N)
        return np.stack([g, g, g], -1)

    def _k_img(self, i):
        """RGB image of k-space with the samples acquired up to time index i."""
        N, H = self.sim.S.N, self.H
        K = np.full((N, N), np.nan + 0j)
        for s in self.sim.samples:
            if s["idx"] <= i:
                K[H - s["ky"], s["kx"] + H] = s["S"]
        return kspace_rgb(K, self.sim.Smax, bg=(0.91, 0.93, 0.95))

    # ------------------------------------------------------------------ callbacks: selectors

    def _on_seq(self, label):
        """Sequence radio: reset the parameters to the new file's defaults, switch to its suggested phantom."""
        if self._quiet:
            return
        self.seq_id = self.seq_ids[[l.get_text() for l in self.widgets["seq"].labels].index(label)]
        self.P = param_defaults(self.seqs[self.seq_id])
        suggested = self.seqs[self.seq_id].get("phantom")
        if suggested in self.phs and suggested != self.ph_id:
            self.ph_id = suggested
            self._quiet = True
            self.widgets["ph"].set_active(self.ph_ids.index(suggested))
            self._quiet = False
            self._adopt_grid_n()
        self._rebuild()

    def _on_ph(self, label):
        """Phantom radio; a painted (grid) phantom brings its own N with it."""
        if self._quiet:
            return
        self.ph_id = self.ph_ids[[l.get_text() for l in self.widgets["ph"].labels].index(label)]
        self._adopt_grid_n()
        self._rebuild()

    def _adopt_grid_n(self):
        """If the chosen phantom is a painted grid, select its N (quietly, so the slider does not rebuild twice)."""
        grid = self.phs[self.ph_id].get("grid")
        if not grid:
            return
        n = int(grid.get("N", len(grid["labels"])))
        if N_MIN <= n <= N_MAX and n != self.N:
            self.N = n
            if "n" in self.widgets:
                self._quiet = True
                self.widgets["n"].set_val(n)
                self._quiet = False

    def _on_n(self, value):
        """Matrix-size slider: like the parameter sliders, it applies when the mouse button is released."""
        n = int(round(value))
        if self._quiet or n == self.N:
            return
        self.N = n
        self._pending = True

    def _set_param(self, pid, value, deferred=False):
        """Store a parameter value; rebuild now, or (sliders) at the next mouse release."""
        self.P[pid] = value
        if deferred:
            self._pending = True
        else:
            self._rebuild()

    def _on_release(self, _event):
        if self._pending:
            self._pending = False
            self._rebuild()

    # ------------------------------------------------------------------ callbacks: time and Fourier

    def _on_t(self, u):
        """Time slider (in drawing coordinate): move the cursor and update the arrows and the two color images."""
        sim, N = self.sim, self.sim.S.N
        t = self.L.t(u)
        i = sim.index_at(t)
        self.cursor.set_xdata([u, u])
        self.tlabel.set_text(f"t = {t:.1f} ms")
        self.img_p.set_data(self._cell_img(i))
        self.q_perp.set_UVC(sim.Mx[i].reshape(N, N), sim.My[i].reshape(N, N))
        self.q_z.set_UVC(np.zeros((N, N)), sim.Mz[i].reshape(N, N))
        self.img_k.set_data(self._k_img(i))
        j0 = max(0, i - 2000)
        self.k_trail.set_data(sim.kx[j0:i + 1], sim.ky[j0:i + 1])
        self.k_dot.set_data([sim.kx[i]], [sim.ky[i]])
        self.fig.canvas.draw_idle()

    def _on_modes(self, n):
        """Mode slider: show the sum of the first n modes, the n-th mode itself, and box its k cells."""
        n = int(n)
        rec = self.rec
        self.n_modes = n
        self.img_i.set_data(rec.grid(np.abs(rec.partial(n))))
        self.axi.set_title(self.T["image"].format(n, len(rec.modes)), fontsize=9)
        for b in self.k_boxes:
            b.set_visible(False)
        if 0 < n <= len(rec.modes):
            m = rec.modes[n - 1]
            self.img_m.set_data(rec.grid(m.contrib))
            self.axm.set_title(self.T["mode"].format(m.kx, m.ky, m.amp), fontsize=9, color="#1a7a0a")
            for b, (kx, ky) in zip(self.k_boxes, m.cells):
                b.set_xy((kx - 0.5, ky - 0.5))
                b.set_visible(True)
        else:
            self.img_m.set_data(np.zeros((self.sim.S.N,) * 2))
            self.axm.set_title("", fontsize=9)
        self._draw_resid()
        self.fig.canvas.draw_idle()

    def _draw_resid(self):
        """Fifth panel: the residual (with gain) when all modes are in, the ideal image otherwise."""
        rec = self.rec
        if self.n_modes >= len(rec.modes):
            self.img_r.set_data(np.clip(rec.grid(rec.residual) * self.gain, -1, 1))
            self.img_r.set_clim(-1, 1)
            self.axr.set_title(self.T["resid"] + f" ×{self.gain:g}, max |Δ| = {np.abs(rec.residual).max():.3f}", fontsize=8)
        else:
            self.img_r.set_data(rec.grid(rec.ideal))
            self.img_r.set_clim(0, max(1e-9, rec.ideal.max()))
            self.axr.set_title(self.T["ideal"], fontsize=9)

    def _on_cell(self, label):
        """Cell-color radio."""
        self.cell_mode = {self.T["cell"][0]: "tissue", self.T["cell"][1]: "density", self.T["cell"][2]: "phase"}[label]
        self._on_t(self.s_t.val)

    def _on_gain(self, label):
        """Residual-gain radio ('×1', '×10', '×100')."""
        self.gain = float(label[1:])
        self._draw_resid()
        self.fig.canvas.draw_idle()

    def _toggle_play(self, _=None):
        """Acquisition play/pause; restarts from t = 0 when at the end.  Stops the Fourier playback."""
        self.playing = not self.playing
        if self.playing:
            self.fplaying = False
            self.b_four.label.set_text(self.T["fourier"])
            if self.s_t.val >= self.L.u_end - 1e-9:
                self.s_t.set_val(0.0)
        self.b_play.label.set_text(self.T["pause"] if self.playing else self.T["play"])

    def _toggle_fourier(self, _=None):
        """Fourier play/pause; restarts from zero modes when complete.  Stops the acquisition playback and moves
        the cursor to the end, since the reconstruction uses the complete acquisition."""
        self.fplaying = not self.fplaying
        if self.fplaying:
            self.playing = False
            self.b_play.label.set_text(self.T["play"])
            if self.s_m.val >= len(self.rec.modes):
                self.s_m.set_val(0)
            self.s_t.set_val(self.L.u_end)
        self.b_four.label.set_text(self.T["fpause"] if self.fplaying else self.T["fourier"])

    def _tick(self):
        """Timer callback: advance whichever playback is running."""
        if self.playing:
            u = self.s_t.val + self.du
            if u >= self.L.u_end:
                u = self.L.u_end
                self._toggle_play()
            self.s_t.set_val(u)
        elif self.fplaying:
            self._fcount += 1
            if self._fcount % 8 == 0:             # one mode every 8 ticks
                n = int(self.s_m.val) + 1
                if n > len(self.rec.modes):
                    self._toggle_fourier()
                else:
                    self.s_m.set_val(n)


def _load_dir(path, pattern="*.json"):
    """Read every JSON file of a folder into {id: definition}, in didactic order (ORDER first, then alphabetical)."""
    items = []
    for f in sorted(pathlib.Path(path).glob(pattern)):
        d = load_json(f)
        items.append((d.get("id", f.stem), d))
    items.sort(key=lambda kv: (ORDER.index(kv[0]) if kv[0] in ORDER else len(ORDER), kv[0]))
    return dict(items)


def lab(sequences_dir=ROOT / "sequences", phantoms_dir=ROOT / "phantoms", tissues_path=ROOT / "tissues.json",
        N: int = 5, sequence: str | None = None, phantom: str | None = None, params: dict | None = None,
        compress: bool = True, lang: str = "pt", block: bool = True) -> Lab:
    """Open the viewer with selectors.

    sequence, phantom   initial choices, as ids (the file's "id") or file stems; a path to a file outside the
                        folders is loaded and added to the list
    params              initial parameter values {id: value} overriding the sequence's defaults
    block               call plt.show() and return when the window is closed
    """
    seqs, phs = _load_dir(sequences_dir), _load_dir(phantoms_dir)
    seq_id = next((k for k in seqs if k == sequence or sequence and pathlib.Path(sequence).stem == k), None) if sequence else None
    if sequence and seq_id is None:
        d = load_json(sequence)
        seqs = {d["id"]: d, **seqs}
        seq_id = d["id"]
    ph_id = next((k for k in phs if k == phantom), None) if phantom else None
    if phantom and ph_id is None:
        d = load_json(phantom)
        phs = {d["id"]: d, **phs}
        ph_id = d["id"]
    L = Lab(seqs, phs, load_tissues(tissues_path), N=N, seq_id=seq_id, phantom_id=ph_id, compress=compress, lang=lang)
    if params:
        L.P.update(params)
        L._rebuild()
    if block:
        plt.show()
    return L


def view(sim: Simulation, phantom, compress: bool = True, lang: str = "pt", block: bool = True) -> Lab:
    """Viewer for one prebuilt simulation (no selectors)."""
    L = Lab({sim.S.defn.get("id", "seq"): sim.S.defn}, {"ph": phantom.defn}, phantom.tissues, N=sim.S.N,
            compress=compress, lang=lang, selectors=False, fixed=(sim, phantom))
    if block:
        plt.show()
    return L


Viewer = Lab
