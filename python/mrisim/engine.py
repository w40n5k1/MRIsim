# By Bruno Mota - METAbio
"""Sequence engine: reads a pulse sequence described as a JSON file, turns it into a timed list of events
(the *schedule*) and runs the Bloch simulation of that schedule on a pixel phantom.

Physical model (identical to the browser version, so both read the same sequence files)
----------------------------------------------------------------------------------------
* The object is an N x N array of isochromats (N odd), one per pixel, at integer positions
  x, y in {-(N-1)/2, ..., (N-1)/2}.  Each pixel has its own equilibrium magnetization M0 (proton density),
  T1 and T2.
* Everything is written in the rotating frame at the Larmor frequency, so the transverse magnetization
  precesses only under the gradients.  A gradient lobe of area dk (in units of 1/FOV) multiplies the
  transverse magnetization m = Mx + i My of the pixel at (x, y) by exp(-i dphi), dphi = 2 pi (dkx x + dky y)/N.
  Consequently the acquired samples are exactly the forward discrete Fourier transform of the pixel
  magnetizations at the sampling instants.
* RF pulses are instantaneous: the whole rotation is applied at the pulse center, and relaxation runs
  uninterrupted before and after it, so the textbook formulas (e.g. the spin-echo signal
  M0 [1 - 2 exp(-(TR - TE/2)/T1) + exp(-TR/T1)] exp(-TE/T2)) hold exactly.  The pulse width drawn on the
  diagram is only an animation of that rotation: inside the width, the displayed magnetization is the physical
  state turned by the angle accrued so far, so the arrows can be watched turning.  A pulse flagged "refocus"
  maps the k label to -k at its center; any other pulse (an excitation) resets the k label to 0 at its center,
  because the transverse magnetization it creates is not yet encoded.  Transverse magnetization left over from
  earlier shots keeps its own accumulated phase and enters the samples as a ghost, which is what a spoiler
  removes.
* Relaxation is exponential, per pixel: Mxy -> Mxy exp(-dt/T2), Mz -> M0 + (Mz - M0) exp(-dt/T1).
  T1 or T2 = inf switches that relaxation off.
* The signal is S(t) = sum over pixels of (Mx + i My), unnormalized (its maximum is the sum of M0).
Units are ms for time and dk = 1/FOV for k.

Sequence file (format didactic-mri-sequence/1)
----------------------------------------------
The file declares parameters (which become the controls of the viewers), constants, tables, derived variables
and a list of events per shot (one repetition of duration TR).  Event types:
  rf     {t, dur, flip (deg), axis 'x'|'y', refocus?}
  grad   {axis 'x'|'y', t, dur, area}         constant lobe; area in dk; amplitude = area/dur
  adc    {t, dur, n}                            n samples at the centers of n equal sub-intervals
  spoil  {t}                                    ideal spoiler: transverse magnetization zeroed at t
  repeat {count, var, vars?, events}            loop; the loop variable is visible to the inner expressions
  band / arrow / mark                           drawing annotations (bands also name the shots in the status line)
Any number in the file may be an expression string.  Expressions see the parameters, constants, tables, vars,
N, HALF = (N-1)/2, the shot index n, the loop variables, and the helpers range(a, b, step), centric(H),
indexOf(list, v), rev(list), plus the usual math functions.  This subset is common to the Python and
JavaScript engines; keep to it so that files stay portable.  Expressions are evaluated as Python code, so load
only sequence files from a source you trust.

Main entry points
-----------------
compile_sequence(defn, P, lang, N) -> Schedule      JSON definition + parameter values -> timed events
simulate(S, M0, T1, T2)            -> Simulation    Bloch simulation of a Schedule on a phantom
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# =============================================================================================== expressions

def _range(a, b, step=None):
    """Inclusive arithmetic progression from a to b (step defaults to +1 or -1 according to the direction).

    range(-2, 2, 1) -> [-2, -1, 0, 1, 2];  range(2, -2, -1) -> [2, 1, 0, -1, -2].
    Used in the sequence files for the linear phase-encoding order.
    """
    step = step if step is not None else (1 if b >= a else -1)
    out, v = [], a
    while (v <= b + 1e-9) if step > 0 else (v >= b - 1e-9):
        out.append(v)
        v += step
    return out


def _centric(H):
    """Centric phase-encoding order: 0, +1, -1, +2, -2, ..., +H, -H (center of k-space first)."""
    out = [0]
    for k in range(1, int(H) + 1):
        out += [k, -k]
    return out


# Names available inside the expressions of a sequence file, besides its own parameters and variables.
HELPERS = {
    "range": _range,
    "centric": _centric,
    "indexOf": lambda arr, v: list(arr).index(v),       # position of a value in a list (e.g. of ky = 0 in the order)
    "rev": lambda arr: list(arr)[::-1],                   # reversed copy of a list
    "min": min, "max": max, "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil,
    "exp": math.exp, "log": math.log, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
    "PI": math.pi, "pi": math.pi,
}


def eval_expr(expr: Any, ctx: dict) -> Any:
    """Evaluate one entry of a sequence file: a number, a boolean, a list or an expression string.

    Booleans become 0/1 so that they can be used in arithmetic ("if" conditions and checkbox parameters).
    Strings are evaluated as Python with the HELPERS and the context `ctx` (parameters, variables, N, ...)
    as the only visible names; builtins are disabled, which is a convenience rather than a security barrier.
    """
    if isinstance(expr, bool):
        return 1 if expr else 0
    if isinstance(expr, (int, float)):
        return expr
    if expr is None:
        return 0
    if isinstance(expr, list):
        return expr
    if not isinstance(expr, str):
        raise ValueError(f"bad expression {expr!r}")
    v = eval(expr, {"__builtins__": {}}, {**HELPERS, **ctx})  # noqa: S307 (sequence files are trusted input)
    if isinstance(v, bool):
        return 1 if v else 0
    return v


def eval_vars(vars_: dict | None, ctx: dict) -> None:
    """Evaluate a {name: expression} block in order, adding each result to `ctx` (later ones may use earlier ones)."""
    for k, e in (vars_ or {}).items():
        ctx[k] = eval_expr(e, ctx)


def fmt_text(tpl: str, ctx: dict) -> str:
    """Fill a label template: every ${expr} is replaced by the value of expr, rounded to two decimals.

    A leading + inside the braces ("${+pe}") prints the sign of positive values, so that gradient labels read
    +2, +1, 0, -1, -2.  Negative numbers use a true minus sign.
    """
    def sub(m):
        e = m.group(1)
        v = eval_expr(e, ctx)
        r = round(float(v) * 100) / 100
        r = int(r) if r == int(r) else r
        if r < 0:
            return "−" + str(-r)
        return ("+" if e.startswith("+") and r > 0 else "") + str(r)
    return re.sub(r"\$\{([^}]+)\}", sub, str(tpl))


def loc(v: Any, lang: str = "pt") -> str:
    """Localize a text entry: a plain string is returned as is; a {"pt": ..., "en": ...} dict picks `lang`,
    falling back to Portuguese, then English."""
    if isinstance(v, dict):
        return v.get(lang) or v.get("pt") or v.get("en") or ""
    return "" if v is None else str(v)


# =============================================================================================== compilation

@dataclass
class Event:
    """One timed event of the schedule, with absolute times in ms.

    Common fields: `type` (rf, grad, adc, spoil, band, mark, arrow), `d` (the raw JSON entry), `ctx` (the
    expression context in which it was evaluated, kept for labels), `t` (start), `dur`, `t1` (end), `shot`.
    Type-specific fields: rf -> flip (deg), axis, refocus, tc (center);  grad -> axis, area (dk), amp (dk/ms);
    adc -> n (number of samples), samples (their instants), tc;  annotations -> label, style, level.
    `desc` is the localized description shown in the status line; `label` the text drawn on the diagram.
    """
    type: str
    d: dict
    ctx: dict
    t: float
    dur: float
    t1: float
    shot: int = 0
    flip: float = 0.0
    axis: str = "x"
    refocus: bool = False
    tc: float = 0.0
    area: float = 0.0
    amp: float = 0.0
    n: int = 0
    samples: list = field(default_factory=list)
    label: str | None = None
    desc: str | None = None
    style: str | None = None
    level: int = 0


@dataclass
class Schedule:
    """A compiled sequence: every event of every shot with absolute times, ready to simulate or draw.

    defn, P, N, lang   the definition, the parameter values used, the matrix size and the language
    ctx0               the base expression context (parameters, constants, tables, vars)
    n_shots, TR        number of repetitions and their period (ms); t_end = n_shots * TR
    t_active           longest busy stretch of a shot (the rest of each TR is an idle tail)
    shots              per shot: {n, T0 (start time), ctx, t_active, label}
    events             rf / grad / adc / spoil events, all shots, in file order
    bands, marks, arrows   drawing annotations (bands: one per shot, also used for the status line)
    adcs               the adc events (subset of `events`)
    samples            every sampling instant, time-ordered: {t, shot, adc, j (index inside its adc)}
    """
    defn: dict
    P: dict
    N: int
    lang: str
    ctx0: dict
    n_shots: int
    TR: float
    t_active: float
    t_end: float
    shots: list
    events: list
    bands: list
    marks: list
    arrows: list
    adcs: list
    samples: list

    @property
    def npix(self):
        """Number of pixels, N * N."""
        return self.N * self.N


def param_defaults(defn: dict) -> dict:
    """Default value of every parameter declared in a sequence file ({id: value}; checkboxes give 0/1)."""
    P = {}
    for p in defn.get("params", []):
        P[p["id"]] = p["default"] if p.get("type") != "checkbox" else (1 if p.get("default") else 0)
    return P


def base_context(defn: dict, P: dict, N: int) -> dict:
    """Build the expression context shared by all shots: N and HALF, then the file's constants, tables,
    parameters and derived vars, in that order.

    A "select" parameter contributes the `values` block of its chosen option (for instance the phase-encoding
    table `pe` for the chosen k_y order) and its own index among the options.
    """
    ctx = {"N": N, "HALF": (N - 1) / 2}
    ctx.update(defn.get("constants", {}))
    for k, v in defn.get("tables", {}).items():
        ctx[k] = eval_expr(v, ctx)
    for p in defn.get("params", []):
        if p.get("type") == "select":
            opts = p.get("options", [])
            opt = next((o for o in opts if o["value"] == P[p["id"]]), opts[0])
            for k, v in (opt.get("values") or {}).items():
                ctx[k] = eval_expr(v, ctx)
            ctx[p["id"]] = opts.index(opt)
        else:
            ctx[p["id"]] = P[p["id"]]
    eval_vars(defn.get("vars"), ctx)
    return ctx


def _expand(events: list, ctx: dict, out: list, T0: float) -> None:
    """Turn the event list of one shot into Event objects with absolute times (T0 = start of the shot).

    "repeat" blocks are unrolled here, each iteration with its own copy of the context carrying the loop
    variable and the block's `vars`.  An event with an "if" expression that evaluates to 0 is skipped.
    """
    for ev in events or []:
        if ev.get("if") is not None and not eval_expr(ev["if"], ctx):
            continue
        if ev["type"] == "repeat":
            count = int(round(eval_expr(ev["count"], ctx)))
            for i in range(count):
                c = dict(ctx)
                c[ev.get("var", "i")] = i
                eval_vars(ev.get("vars"), c)
                _expand(ev["events"], c, out, T0)
            continue
        t = T0 + eval_expr(ev["t"], ctx)
        dur = eval_expr(ev["dur"], ctx) if ev.get("dur") is not None else 0.0
        e = Event(type=ev["type"], d=ev, ctx=ctx, t=t, dur=dur, t1=t + dur)
        if ev["type"] == "rf":
            e.flip = float(eval_expr(ev["flip"], ctx))
            e.axis = ev.get("axis", "x")
            e.refocus = bool(ev.get("refocus"))
            e.tc = t + dur / 2
        elif ev["type"] == "grad":
            e.axis = ev["axis"]
            e.area = float(eval_expr(ev["area"], ctx))
            e.amp = e.area / dur if dur > 0 else 0.0
        elif ev["type"] == "adc":
            e.n = int(round(eval_expr(ev["n"], ctx)))
            e.samples = [t + (j + 0.5) * dur / e.n for j in range(e.n)]   # sample at the center of each dwell interval
            e.tc = t + dur / 2
        elif ev["type"] == "spoil":
            pass
        elif ev["type"] in ("band", "mark", "arrow"):
            if ev.get("to") is not None:
                e.t1 = T0 + eval_expr(ev["to"], ctx)
            e.style = ev.get("style")
            e.level = int(ev.get("level", 0))
        else:
            raise ValueError("unknown event type " + ev["type"])
        out.append(e)


def compile_sequence(defn: dict, P: dict | None = None, lang: str = "pt", N: int = 5) -> Schedule:
    """Compile a sequence definition (the parsed JSON) into a Schedule.

    defn   the sequence file as a dict (see load_json)
    P      parameter values {id: value}; missing ones take the file's defaults
    lang   language of the labels and descriptions ('pt' or 'en')
    N      matrix size (odd)

    Steps: build the base context; evaluate `shots` and `TR`; for each shot n, add the shot variables, expand
    the events with absolute times n*TR + t, localize and fill their labels; collect the annotations; check that
    TR covers the active part and that no RF pulse overlaps a gradient or a readout (the hard-pulse model needs
    that); finally list every sampling instant in time order.
    """
    P = dict(param_defaults(defn), **(P or {}))
    ctx0 = base_context(defn, P, N)
    n_shots = max(1, int(round(eval_expr(defn.get("shots", 1), ctx0))))
    TR = float(eval_expr(defn["TR"], ctx0))
    shots, events, bands, marks, arrows, adcs = [], [], [], [], [], []
    for n in range(n_shots):
        ctx = dict(ctx0, n=n)
        eval_vars(defn.get("shotVars"), ctx)
        T0, out = n * TR, []
        _expand(defn["events"], ctx, out, T0)
        t_active = 0.0
        for e in out:
            if e.d.get("desc") is not None:
                e.desc = fmt_text(loc(e.d["desc"], lang), e.ctx)
            if e.type == "grad" and e.d.get("label") is not None:
                e.label = fmt_text(loc(e.d["label"], lang), e.ctx)
            e.shot = n
            if e.type == "band":
                e.label = fmt_text(loc(e.d.get("label"), lang), e.ctx) if e.d.get("label") is not None else None
                bands.append(e)
            elif e.type == "mark":
                if n == 0:                      # marks and arrows are drawn on the first shot only
                    e.label = loc(e.d.get("label"), lang) or None
                    marks.append(e)
            elif e.type == "arrow":
                if n == 0:
                    e.label = loc(e.d.get("label"), lang)
                    arrows.append(e)
            else:
                events.append(e)
                if e.type != "spoil":
                    t_active = max(t_active, e.t1 - T0)
                if e.type == "adc":
                    adcs.append(e)
        label = fmt_text(loc(defn["shotLabel"], lang), ctx) if defn.get("shotLabel") else ""
        shots.append({"n": n, "T0": T0, "ctx": ctx, "t_active": t_active, "label": label})
    t_active = max(s["t_active"] for s in shots)
    if TR < t_active:
        raise ValueError(f"TR shorter than the active part ({t_active:.1f} ms)")
    # Without explicit bands, each shot gets one band starting at its first RF pulse, named by shotLabel.
    if not bands:
        for s in shots:
            rf = next((e for e in events if e.shot == s["n"] and e.type == "rf"), None)
            bands.append(Event(type="band", d={}, ctx=s["ctx"], t=rf.tc if rf else s["T0"], dur=0, t1=0,
                               shot=s["n"], label=s["label"]))
    bands.sort(key=lambda b: b.t)
    for i, b in enumerate(bands):                # each band runs until the next one
        b.t1 = bands[i + 1].t if i + 1 < len(bands) else n_shots * TR
        if b.label is None:
            b.label = shots[b.shot]["label"]
    # Hard-pulse model: RF must not overlap gradients or readouts.
    for r in events:
        if r.type != "rf":
            continue
        for e in events:
            if e is not r and e.type != "spoil" and e.shot == r.shot and e.t < r.t1 - 1e-9 and e.t1 > r.t + 1e-9:
                raise ValueError(f"RF pulse at {r.t:.1f} ms overlaps a {e.type} event: increase the timing parameters")
    samples = [{"t": t, "shot": a.shot, "adc": a, "j": j} for a in adcs for j, t in enumerate(a.samples)]
    samples.sort(key=lambda s: s["t"])
    return Schedule(defn, P, N, lang, ctx0, n_shots, TR, t_active, n_shots * TR,
                    shots, events, bands, marks, arrows, adcs, samples)


# =============================================================================================== simulation

def pixel_xy(N: int):
    """Coordinates (x, y) of every pixel, in pixel units, for the row-major index p = row * N + col.

    Row 0 is the top of the image (y = +HALF), column 0 the left (x = -HALF); the center pixel is (0, 0).
    """
    H = (N - 1) / 2
    p = np.arange(N * N)
    return (p % N) - H, H - (p // N)


@dataclass
class Simulation:
    """Result of simulate(): the magnetization of every pixel at every instant of the time grid.

    S            the Schedule that was simulated
    M0, T1, T2   the phantom, one value per pixel (T1/T2 may be inf)
    times        the time grid (ms), Nt instants; shot_of[i] is the shot index at times[i]
    Mx, My, Mz   arrays (Nt, N*N): magnetization components per pixel; row i is the state at times[i]
    signal       complex array (Nt,): S(t) = sum of Mx + i My over the pixels
    kx, ky       arrays (Nt,): the k label of the freshly excited magnetization at each instant
    samples      the acquired samples, time-ordered: {idx (time index), t, shot, adc, j, kx, ky, S (complex)}
    Smax         sum of M0, the largest possible |S|; used to normalize displays
    """
    S: Schedule
    M0: np.ndarray
    T1: np.ndarray
    T2: np.ndarray
    times: np.ndarray
    shot_of: np.ndarray
    Mx: np.ndarray
    My: np.ndarray
    Mz: np.ndarray
    signal: np.ndarray
    kx: np.ndarray
    ky: np.ndarray
    samples: list
    Smax: float

    def index_at(self, t: float) -> int:
        """Index of the grid instant nearest to t (clamped to the grid)."""
        i = int(np.searchsorted(self.times, t))
        if i <= 0:
            return 0
        if i >= len(self.times):
            return len(self.times) - 1
        return i if abs(self.times[i] - t) < abs(self.times[i - 1] - t) else i - 1

    def kspace(self) -> np.ndarray:
        """The acquired samples on the N x N k grid as a complex array (row 0 = ky = +HALF, column 0 = kx = -HALF);
        NaN where no sample was acquired."""
        N, H = self.S.N, (self.S.N - 1) // 2
        K = np.full((N, N), np.nan + 0j)
        for s in self.samples:
            K[H - s["ky"], s["kx"] + H] = s["S"]
        return K


def _overlap(a, b, t0, t1):
    """Length of the intersection of the intervals [a, b] and [t0, t1] (0 if disjoint)."""
    return max(0.0, min(b, t1) - max(a, t0))


def _rotate(mx, my, mz, th, axis):
    """Right-handed rotation of the magnetization by th radians about the x or y axis; returns new arrays.

    About y: (0, 0, 1) -> (sin th, 0, cos th), so a 90 degree pulse about y takes +z to +x.
    About x: (0, 0, 1) -> (0, -sin th, cos th).
    """
    c, s_ = math.cos(th), math.sin(th)
    if axis == "y":
        return mx * c + mz * s_, my.copy(), -mx * s_ + mz * c
    return mx.copy(), my * c - mz * s_, my * s_ + mz * c


def _time_grid(S: Schedule, dt: float, n_gap: int) -> np.ndarray:
    """Adaptive time grid: step dt inside and 1 ms around every event, at most n_gap points across each idle gap
    (TR tails, inversion delays), plus the exact sampling instants and the start of every shot."""
    margin = 1.0
    busy = sorted((max(0.0, e.t - margin), min(S.t_end, e.t1 + margin)) for e in S.events if e.type != "spoil")
    merged = []                                   # union of the busy intervals
    for a, b in busy:
        if merged and a <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    grid, cursor = [], 0.0

    def coarse(a, b):
        n = max(1, min(n_gap, int(round((b - a) / dt))))
        grid.extend(a + i * (b - a) / n for i in range(1, n))

    for a, b in merged:
        if a > cursor:
            grid.append(cursor)
            coarse(cursor, a)
        n = int(round((b - a) / dt))
        grid.extend(a + i * dt for i in range(n + 1))
        cursor = b
    if cursor < S.t_end:
        grid.append(cursor)
        coarse(cursor, S.t_end)
    grid.append(S.t_end)
    grid.extend(s["t"] for s in S.samples)
    grid.extend(sh["T0"] for sh in S.shots)
    grid.sort()
    times = []
    for t in grid:                                # drop duplicates closer than 0.1 ns
        if times and t - times[-1] < 1e-7:
            continue
        times.append(t)
    return np.array(times)


def simulate(S: Schedule, M0, T1, T2, dt: float = 0.1, n_gap: int = 80) -> Simulation:
    """Bloch simulation of a Schedule on a phantom.

    M0, T1, T2   per-pixel arrays of length N*N (row-major, see pixel_xy); scalars are broadcast; inf disables
                 the corresponding relaxation
    dt           time step (ms) inside and around events
    n_gap        maximum number of steps across an idle gap

    The state starts at equilibrium (Mz = M0, Mxy = 0).  From one grid instant to the next, in this order:
    relaxation up to the center of any RF pulse inside the interval, the full rotation of that pulse, relaxation
    over the rest of the interval; the precession from the gradient areas accumulated in the interval, applied
    as a phase per pixel; the spoiler, if its instant falls in the interval.  The k label is updated alongside:
    refocusing pulse -> -k, excitation -> 0, gradients -> k + dk.  Each sample stores the signal and the k label
    at its instant.  The stored magnetization (Mx, My, Mz and the signal) is the display state: inside the
    drawn width of a pulse it is the physical state turned by the angle accrued so far, so that the rotation
    can be watched; everywhere else it is the physical state itself.
    """
    N, NPIX = S.N, S.npix
    M0 = np.asarray(M0, float).reshape(NPIX)
    T1 = np.broadcast_to(np.asarray(T1, float), (NPIX,)).astype(float)
    T2 = np.broadcast_to(np.asarray(T2, float), (NPIX,)).astype(float)
    times = _time_grid(S, dt, n_gap)
    Nt = len(times)
    shot_of = np.minimum(S.n_shots - 1, np.floor(times / S.TR)).astype(int)
    X, Y = pixel_xy(N)
    rfs = [e for e in S.events if e.type == "rf"]
    grads = [e for e in S.events if e.type == "grad"]
    spoils = [e for e in S.events if e.type == "spoil"]
    # Map each sampling instant to its grid index (the grid contains these instants exactly).
    sample_idx: dict[int, list] = {}
    si = 0
    for smp in S.samples:
        while si < Nt - 1 and times[si] < smp["t"] - 1e-7:
            si += 1
        sample_idx.setdefault(si, []).append(smp)
    # Storage for the whole history.
    Mx = np.zeros((Nt, NPIX))
    My = np.zeros((Nt, NPIX))
    Mz = np.zeros((Nt, NPIX))
    sig = np.zeros(Nt, complex)
    kxM = np.zeros(Nt)
    kyM = np.zeros(Nt)
    # Current physical state: equilibrium.
    mx = np.zeros(NPIX)
    my = np.zeros(NPIX)
    mz = M0.copy()
    e1_fin, e2_fin = np.isfinite(T1), np.isfinite(T2)
    kx = ky = 0.0
    samples = []

    def relax(dt):
        """Exponential relaxation of the physical state over dt ms."""
        nonlocal mx, my, mz
        if dt <= 0:
            return
        e2 = np.where(e2_fin, np.exp(-dt / np.where(e2_fin, T2, 1.0)), 1.0)
        e1 = np.where(e1_fin, np.exp(-dt / np.where(e1_fin, T1, 1.0)), 1.0)
        mx = mx * e2
        my = my * e2
        mz = M0 + (mz - M0) * e1

    for i in range(Nt):
        t = times[i]
        if i > 0:
            a, b = times[i - 1], t
            # 1. Relaxation, interrupted only by the instantaneous rotation of a pulse whose center falls in (a, b].
            t_prev = a
            for r in sorted((r for r in rfs if a < r.tc <= b), key=lambda r: r.tc):
                relax(r.tc - t_prev)
                mx, my, mz = _rotate(mx, my, mz, math.radians(r.flip), r.axis)
                kx, ky = (-kx, -ky) if r.refocus else (0.0, 0.0)   # refocusing: k -> -k; excitation: fresh M_perp at k = 0
                t_prev = r.tc
            relax(b - t_prev)
            # 2. Gradients: the k increment of the interval is a phase per pixel (precession in the rotating frame).
            dkx = dky = 0.0
            for g in grads:
                o = _overlap(a, b, g.t, g.t1)
                if o > 0:
                    if g.axis == "x":
                        dkx += g.amp * o
                    else:
                        dky += g.amp * o
            if dkx != 0.0 or dky != 0.0:
                kx += dkx
                ky += dky
                ph = 2 * np.pi * (dkx * X + dky * Y) / N
                c, s_ = np.cos(ph), np.sin(ph)
                mx, my = mx * c + my * s_, -mx * s_ + my * c    # m -> m exp(-i ph)
            # 3. Ideal spoiler: transverse magnetization and k label zeroed.
            for sp in spoils:
                if a < sp.t <= b:
                    mx[:] = 0
                    my[:] = 0
                    kx = ky = 0.0
        # Display state: inside a pulse's drawn width, show the rotation in progress.
        dx, dy, dz = mx, my, mz
        for r in rfs:
            if r.t < t < r.t1:
                th_now = math.radians(r.flip) * (t - r.t) / r.dur
                th_applied = math.radians(r.flip) if t >= r.tc else 0.0
                dx, dy, dz = _rotate(mx, my, mz, th_now - th_applied, r.axis)
        Mx[i] = dx
        My[i] = dy
        Mz[i] = dz
        sig[i] = dx.sum() + 1j * dy.sum()
        kxM[i] = kx
        kyM[i] = ky
        for smp in sample_idx.get(i, []):
            samples.append(dict(smp, idx=i, kx=int(round(kx)), ky=int(round(ky)), S=sig[i]))
    return Simulation(S, M0, T1, T2, times, shot_of, Mx, My, Mz, sig, kxM, kyM, samples, float(M0.sum()))


# =============================================================================================== helpers

def segment_at(S: Schedule, t: float) -> dict:
    """What is happening at time t, for a status line: {"kind": ...} with kind in
    rf (with the pulse), adc (with the readout and the gradients on), grad (gradients only), tail (idle end of the
    TR), rest (before the first event of the shot) or free (between events)."""
    at = [e for e in S.events if e.type != "spoil" and e.t <= t <= e.t1]
    rf = next((e for e in at if e.type == "rf"), None)
    if rf:
        return {"kind": "rf", "e": rf}
    adc = next((e for e in at if e.type == "adc"), None)
    if adc:
        return {"kind": "adc", "e": adc, "grads": [e for e in at if e.type == "grad"]}
    gr = [e for e in at if e.type == "grad"]
    if gr:
        return {"kind": "grad", "grads": gr}
    shot = S.shots[min(S.n_shots - 1, max(0, int(t // S.TR)))]
    if t - shot["T0"] > shot["t_active"] + 1e-9:
        return {"kind": "tail"}
    first = min((e.t for e in S.events if e.shot == shot["n"]), default=math.inf)
    return {"kind": "rest" if t < first else "free"}


def event_times(S: Schedule) -> list:
    """Sorted list of the instants where something starts, ends or peaks (for stepping through the sequence)."""
    ts = {0.0, S.t_end}
    for e in S.events:
        ts.add(e.t)
        if e.dur > 0:
            ts.add(e.t1)
        if e.type in ("rf", "adc"):
            ts.add(e.tc)
    for s in S.shots:
        ts.add(s["T0"] + s["t_active"])
    return sorted(ts)


def load_json(path) -> dict:
    """Read a UTF-8 JSON file (sequence, phantom or tissue dictionary)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
