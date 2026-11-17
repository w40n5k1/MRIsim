"""Generic engine for didactic 2D imaging sequences described as JSON files (format didactic-mri-sequence/1).

Physics (same model as the browser version):
  * N x N isochromats (N odd), one per pixel, at x, y in {-(N-1)/2, ..., (N-1)/2}.
  * Rotating frame at the Larmor frequency: transverse magnetization precesses only under gradients,
        m -> m exp(-i dphi),  dphi = 2 pi (dkx x + dky y) / N,
    with k in units of dk = 1/FOV.  The acquired samples are the forward DFT of the pixel magnetizations.
  * Hard RF pulses: the flip angle accrues linearly over the pulse width (only so the rotation can be watched);
    relaxation is suspended during a pulse.  A pulse flagged "refocus" maps k -> -k at its center; any other pulse
    (excitation) resets the k label to 0 at its center, since the magnetization it creates is unencoded.  Transverse
    magnetization left from an earlier shot keeps its own accumulated phase and contributes to the samples as a ghost.
  * Exponential relaxation, per pixel: Mxy ~ exp(-dt/T2), Mz -> M0 + (Mz - M0) exp(-dt/T1).  inf disables.
  * Signal S(t) = sum over pixels of (Mx + i My), unnormalized.
Units: ms, dk.

Sequence file: parameters, constants, tables, derived variables and a list of events per shot (repetition):
  rf    {t, dur, flip (deg), axis 'x'|'y', refocus?}
  grad  {axis 'x'|'y', t, dur, area}          constant lobe, area in dk
  adc   {t, dur, n}                             n samples at the centers of n equal sub-intervals
  spoil {t}                                     ideal spoiler: transverse magnetization zeroed at t
  repeat {count, var, vars?, events}            loop
  band / arrow / mark                           drawing annotations (bands are also used for the status)
Numbers may be expression strings evaluated with the parameters, constants, tables, vars, N, HALF, the shot
index n, loop variables and the helpers range(a, b, step), centric(H), indexOf(list, v), rev(list), plus math.
The same files are read by the JavaScript version; the expression subset is common to both (arithmetic,
indexing, these helpers).  Expressions are evaluated as Python: load only files from a known source.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ----------------------------------------------------------------------------------------------- expressions

def _range(a, b, step=None):
    step = step if step is not None else (1 if b >= a else -1)
    out, v = [], a
    while (v <= b + 1e-9) if step > 0 else (v >= b - 1e-9):
        out.append(v)
        v += step
    return out


def _centric(H):
    out = [0]
    for k in range(1, int(H) + 1):
        out += [k, -k]
    return out


HELPERS = {
    "range": _range, "centric": _centric, "indexOf": lambda arr, v: list(arr).index(v), "rev": lambda arr: list(arr)[::-1],
    "min": min, "max": max, "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil,
    "exp": math.exp, "log": math.log, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "PI": math.pi, "pi": math.pi,
}


def eval_expr(expr: Any, ctx: dict) -> Any:
    """Evaluate a number, boolean or expression string in the context."""
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
    for k, e in (vars_ or {}).items():
        ctx[k] = eval_expr(e, ctx)


def fmt_text(tpl: str, ctx: dict) -> str:
    """Substitute ${expr} in a label; a leading + in the expression prints the sign."""
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
    if isinstance(v, dict):
        return v.get(lang) or v.get("pt") or v.get("en") or ""
    return "" if v is None else str(v)


# ----------------------------------------------------------------------------------------------- compile

@dataclass
class Event:
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
    samples: list   # dicts {t, shot, adc, j}

    @property
    def npix(self):
        return self.N * self.N


def param_defaults(defn: dict) -> dict:
    P = {}
    for p in defn.get("params", []):
        P[p["id"]] = p["default"] if p.get("type") != "checkbox" else (1 if p.get("default") else 0)
    return P


def base_context(defn: dict, P: dict, N: int) -> dict:
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
            e.flip = float(eval_expr(ev["flip"], ctx)); e.axis = ev.get("axis", "x"); e.refocus = bool(ev.get("refocus")); e.tc = t + dur / 2
        elif ev["type"] == "grad":
            e.axis = ev["axis"]; e.area = float(eval_expr(ev["area"], ctx)); e.amp = e.area / dur if dur > 0 else 0.0
        elif ev["type"] == "adc":
            e.n = int(round(eval_expr(ev["n"], ctx))); e.samples = [t + (j + 0.5) * dur / e.n for j in range(e.n)]; e.tc = t + dur / 2
        elif ev["type"] == "spoil":
            pass
        elif ev["type"] in ("band", "mark", "arrow"):
            if ev.get("to") is not None:
                e.t1 = T0 + eval_expr(ev["to"], ctx)
            e.style = ev.get("style"); e.level = int(ev.get("level", 0))
        else:
            raise ValueError("unknown event type " + ev["type"])
        out.append(e)


def compile_sequence(defn: dict, P: dict | None = None, lang: str = "pt", N: int = 5) -> Schedule:
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
                if n == 0:
                    e.label = loc(e.d.get("label"), lang) or None; marks.append(e)
            elif e.type == "arrow":
                if n == 0:
                    e.label = loc(e.d.get("label"), lang); arrows.append(e)
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
    if not bands:
        for s in shots:
            rf = next((e for e in events if e.shot == s["n"] and e.type == "rf"), None)
            bands.append(Event(type="band", d={}, ctx=s["ctx"], t=rf.tc if rf else s["T0"], dur=0, t1=0, shot=s["n"], label=s["label"]))
    bands.sort(key=lambda b: b.t)
    for i, b in enumerate(bands):
        b.t1 = bands[i + 1].t if i + 1 < len(bands) else n_shots * TR
        if b.label is None:
            b.label = shots[b.shot]["label"]
    # hard-pulse model: RF must not overlap gradients or readouts
    for r in events:
        if r.type != "rf":
            continue
        for e in events:
            if e is not r and e.type != "spoil" and e.shot == r.shot and e.t < r.t1 - 1e-9 and e.t1 > r.t + 1e-9:
                raise ValueError(f"RF pulse at {r.t:.1f} ms overlaps a {e.type} event: increase the timing parameters")
    samples = [{"t": t, "shot": a.shot, "adc": a, "j": j} for a in adcs for j, t in enumerate(a.samples)]
    samples.sort(key=lambda s: s["t"])
    return Schedule(defn, P, N, lang, ctx0, n_shots, TR, t_active, n_shots * TR, shots, events, bands, marks, arrows, adcs, samples)


# ----------------------------------------------------------------------------------------------- simulate

def pixel_xy(N: int):
    """x, y of pixel p = row*N + col (row 0 at the top, y = +HALF)."""
    H = (N - 1) / 2
    p = np.arange(N * N)
    return (p % N) - H, H - (p // N)


@dataclass
class Simulation:
    S: Schedule
    M0: np.ndarray
    T1: np.ndarray
    T2: np.ndarray
    times: np.ndarray
    shot_of: np.ndarray
    Mx: np.ndarray      # (Nt, NPIX)
    My: np.ndarray
    Mz: np.ndarray
    signal: np.ndarray  # complex (Nt,)
    kx: np.ndarray
    ky: np.ndarray
    samples: list       # dicts {idx, t, shot, adc, j, kx, ky, S (complex)}
    Smax: float

    def index_at(self, t: float) -> int:
        i = int(np.searchsorted(self.times, t))
        if i <= 0:
            return 0
        if i >= len(self.times):
            return len(self.times) - 1
        return i if abs(self.times[i] - t) < abs(self.times[i - 1] - t) else i - 1

    def kspace(self) -> np.ndarray:
        """Acquired samples on the N x N k grid (row 0 = ky = +HALF), complex; NaN where not acquired."""
        N, H = self.S.N, (self.S.N - 1) // 2
        K = np.full((N, N), np.nan + 0j)
        for s in self.samples:
            K[H - s["ky"], s["kx"] + H] = s["S"]
        return K


def _overlap(a, b, t0, t1):
    return max(0.0, min(b, t1) - max(a, t0))


def simulate(S: Schedule, M0, T1, T2, dt: float = 0.1, n_gap: int = 80) -> Simulation:
    """Bloch simulation of the schedule on a phantom (M0, T1, T2 per pixel; scalars allowed; inf disables)."""
    N, NPIX = S.N, S.npix
    M0 = np.asarray(M0, float).reshape(NPIX)
    T1 = np.broadcast_to(np.asarray(T1, float), (NPIX,)).astype(float)
    T2 = np.broadcast_to(np.asarray(T2, float), (NPIX,)).astype(float)
    # time grid: fine inside and around events, coarse across idle gaps, plus the exact sample instants
    margin = 1.0
    busy = sorted((max(0.0, e.t - margin), min(S.t_end, e.t1 + margin)) for e in S.events if e.type != "spoil")
    merged = []
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
            grid.append(cursor); coarse(cursor, a)
        n = int(round((b - a) / dt))
        grid.extend(a + i * dt for i in range(n + 1))
        cursor = b
    if cursor < S.t_end:
        grid.append(cursor); coarse(cursor, S.t_end)
    grid.append(S.t_end)
    grid.extend(s["t"] for s in S.samples)
    grid.extend(sh["T0"] for sh in S.shots)
    grid.sort()
    times = []
    for t in grid:
        if times and t - times[-1] < 1e-7:
            continue
        times.append(t)
    times = np.array(times)
    Nt = len(times)
    shot_of = np.minimum(S.n_shots - 1, np.floor(times / S.TR)).astype(int)
    X, Y = pixel_xy(N)
    rfs = [e for e in S.events if e.type == "rf"]
    grads = [e for e in S.events if e.type == "grad"]
    spoils = [e for e in S.events if e.type == "spoil"]
    sample_idx: dict[int, list] = {}
    si = 0
    for smp in S.samples:
        while si < Nt - 1 and times[si] < smp["t"] - 1e-7:
            si += 1
        sample_idx.setdefault(si, []).append(smp)
    Mx = np.zeros((Nt, NPIX)); My = np.zeros((Nt, NPIX)); Mz = np.zeros((Nt, NPIX))
    sig = np.zeros(Nt, complex); kxM = np.zeros(Nt); kyM = np.zeros(Nt)
    mx = np.zeros(NPIX); my = np.zeros(NPIX); mz = M0.copy()
    e1_fin, e2_fin = np.isfinite(T1), np.isfinite(T2)
    kx = ky = 0.0
    samples = []
    for i in range(Nt):
        if i > 0:
            a, b = times[i - 1], times[i]
            d = b - a
            o_pulse = sum(_overlap(a, b, r.t, r.t1) for r in rfs)
            dR = d - o_pulse
            if dR > 0:
                e2 = np.where(e2_fin, np.exp(-dR / np.where(e2_fin, T2, 1.0)), 1.0)
                e1 = np.where(e1_fin, np.exp(-dR / np.where(e1_fin, T1, 1.0)), 1.0)
                mx *= e2; my *= e2; mz = M0 + (mz - M0) * e1
            for r in rfs:
                o = _overlap(a, b, r.t, r.t1)
                if o > 0:
                    th = math.radians(r.flip) * o / r.dur
                    c, s_ = math.cos(th), math.sin(th)
                    if r.axis == "y":      # right-handed rotation about y: (0,0,1) -> (sin th, 0, cos th)
                        mx, mz = mx * c + mz * s_, -mx * s_ + mz * c
                    else:                  # about x
                        my, mz = my * c - mz * s_, my * s_ + mz * c
                if a < r.tc <= b:          # at the pulse center: refocusing maps k -> -k; an excitation creates fresh M_perp at k = 0
                    kx, ky = (-kx, -ky) if r.refocus else (0.0, 0.0)
            dkx = dky = 0.0
            for g in grads:
                o = _overlap(a, b, g.t, g.t1)
                if o > 0:
                    if g.axis == "x":
                        dkx += g.amp * o
                    else:
                        dky += g.amp * o
            if dkx != 0.0 or dky != 0.0:
                kx += dkx; ky += dky
                ph = 2 * np.pi * (dkx * X + dky * Y) / N
                c, s_ = np.cos(ph), np.sin(ph)
                mx, my = mx * c + my * s_, -mx * s_ + my * c
            for sp in spoils:
                if a < sp.t <= b:
                    mx[:] = 0; my[:] = 0; kx = ky = 0.0
        Mx[i] = mx; My[i] = my; Mz[i] = mz
        sig[i] = mx.sum() + 1j * my.sum(); kxM[i] = kx; kyM[i] = ky
        for smp in sample_idx.get(i, []):
            samples.append(dict(smp, idx=i, kx=int(round(kx)), ky=int(round(ky)), S=sig[i]))
    return Simulation(S, M0, T1, T2, times, shot_of, Mx, My, Mz, sig, kxM, kyM, samples, float(M0.sum()))


# ----------------------------------------------------------------------------------------------- helpers

def segment_at(S: Schedule, t: float) -> dict:
    """Which event is active at t (for status text): rf > adc > grad > tail > rest/free."""
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
    with open(path, encoding="utf-8") as f:
        return json.load(f)
