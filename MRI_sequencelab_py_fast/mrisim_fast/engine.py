# By Bruno Mota - METAbio
"""Fast simulation of a compiled sequence: the acquired samples, and nothing else.

Same model as mrisim.engine (rotating frame; one isochromat per pixel of an N x N array, N odd; instantaneous RF
pulses; a gradient lobe of area dk multiplies m = Mx + i My by exp(-2 pi i (dk_x x + dk_y y)/N); exponential
relaxation per pixel; an ideal spoiler), and the same Schedule, produced by mrisim.compile_sequence.  The
difference is in what is computed and kept.

mrisim.simulate advances every pixel on a fine time grid and stores the whole history, because its purpose is to
show the magnetization at any instant.  The memory grows roughly as N^4, which stops it near N = 35.  Here:

* Nothing is stored but the samples (memoryless).  The state is one complex array m and one real array Mz.
* The time axis is cut only where something happens: the edges of the gradient lobes, the centers of the RF
  pulses, the spoilers and the sampling instants.  Between two such instants the evolution is exact in closed
  form: m is multiplied by one complex factor per pixel,
      F = exp(-dt/T2) exp(-2 pi i (dk_x x + dk_y y)/N),
  with dt the duration of the step and (dk_x, dk_y) the gradient area inside it.
* Everything that does not depend on the object is worked out before the pixels are touched (the Plan): the
  steps, their durations and areas, what happens at the end of each one, and the k label of every sample.
* The factors F are computed once per distinct (dt, dk_x, dk_y) and reused.  All the dwell intervals of all the
  readouts of a sequence share a single F.
* Mz is not advanced step by step.  It relaxes undisturbed between RF pulses, so it is brought up to date, in
  closed form, only when a pulse needs it: Mz -> M0 + (Mz - M0) exp(-dt/T1), dt since the previous pulse.

The cost is one multiplication and one sum over the N^2 pixels per sample, about N^4 operations for a sequence
that fills k-space once, and the memory is a few arrays of N^2 numbers.  N = 255 runs in seconds.

The samples agree with mrisim.simulate to rounding (tests/test_fast.py compares them).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ._didactic import mrisim            # noqa: F401  (makes the didactic package importable)
from mrisim.engine import Schedule, pixel_xy

MERGE = 1e-9          # instants closer than this (ms) are the same instant


# =============================================================================================== the plan

@dataclass
class Plan:
    """What a Schedule asks for, step by step, independently of the object.

    N           matrix size
    times       the instants where something happens (ms), increasing; times[0] = 0
    dt          dt[i] = times[i] - times[i-1], the duration of step i (dt[0] = 0)
    dkx, dky    gradient area inside step i, in units of 1/FOV
    prop        prop[i]: index, in `props`, of the propagator of step i
    props       the distinct (dt, dk_x, dk_y) triples
    pulses      {i: [rf events whose center is times[i]]}
    spoils      the set of i at which a spoiler acts
    samples     {i: [indices of the samples taken at times[i]]}
    t, shot, adc, j, kx, ky   one entry per sample, time-ordered: instant, shot, index of its acquisition
                window in `adcs`, index inside the window, and the k label (integers)
    adcs        the acquisition events of the Schedule (mrisim Event objects)
    """
    N: int
    times: np.ndarray
    dt: np.ndarray
    dkx: np.ndarray
    dky: np.ndarray
    prop: np.ndarray
    props: list
    pulses: dict
    spoils: set
    samples: dict
    t: np.ndarray
    shot: np.ndarray
    adc: np.ndarray
    j: np.ndarray
    kx: np.ndarray
    ky: np.ndarray
    adcs: list = field(repr=False, default_factory=list)


def _merge(instants) -> np.ndarray:
    """Sorted instants with those closer than MERGE collapsed into the first of them."""
    a = np.sort(np.asarray(list(instants), float))
    keep = np.ones(len(a), bool)
    keep[1:] = np.diff(a) > MERGE
    # a run of near-equal instants keeps only its first element
    return a[keep]


def _index(times: np.ndarray, t: float) -> int:
    """Index of the instant of `times` that equals t (to within MERGE)."""
    i = int(np.searchsorted(times, t - MERGE))
    if i >= len(times) or abs(times[i] - t) > 10 * MERGE:
        raise ValueError(f"instant {t} ms is not on the time axis")
    return i


def plan(S: Schedule) -> Plan:
    """Turn a compiled sequence into a Plan.

    The time axis holds 0, the end of the sequence, both edges of every gradient lobe, the center of every RF
    pulse, every spoiler and every sampling instant.  A lobe is therefore either on or off during a whole step,
    and a pulse, a spoiler or a sample always sits at the end of a step.  A lobe cannot be on in the step that
    ends at a pulse center, because the compiler refuses a lobe that overlaps the drawn width of a pulse.

    The k label follows the same rules as in mrisim.simulate: an excitation (an rf without "refocus") sets it to
    0, a refocusing pulse maps k to -k, a lobe adds its area, a spoiler sets it to 0.
    """
    rfs = [e for e in S.events if e.type == "rf"]
    grads = [e for e in S.events if e.type == "grad"]
    spoils = [e for e in S.events if e.type == "spoil" and e.t <= S.t_end + MERGE]
    instants = [0.0, S.t_end] + [r.tc for r in rfs] + [s.t for s in spoils] + [s["t"] for s in S.samples]
    for g in grads:
        instants += [g.t, g.t1]
    times = _merge(t for t in instants if -MERGE <= t <= S.t_end + MERGE)
    n = len(times)
    dt = np.diff(times, prepend=times[0])
    # Gradient area per step: each lobe adds amp * dt to the steps it covers.
    dkx, dky = np.zeros(n), np.zeros(n)
    for g in grads:
        i0, i1 = _index(times, g.t), _index(times, g.t1)
        (dkx if g.axis == "x" else dky)[i0 + 1:i1 + 1] += g.amp * dt[i0 + 1:i1 + 1]
    # Distinct propagators.  Durations and areas are rounded to 1e-9 only to recognize equal steps.
    index, props, prop = {}, [], np.zeros(n, int)
    for i in range(1, n):
        key = (round(dt[i], 9), round(dkx[i], 9), round(dky[i], 9))
        if key not in index:
            index[key] = len(props)
            props.append((dt[i], dkx[i], dky[i]))
        prop[i] = index[key]
    pulses: dict = {}
    for r in rfs:
        pulses.setdefault(_index(times, r.tc), []).append(r)
    spoil_at = {_index(times, s.t) for s in spoils}
    samples: dict = {}
    for q, s in enumerate(S.samples):
        samples.setdefault(_index(times, s["t"]), []).append(q)
    # k label of every sample: bookkeeping only, in the order the engine applies things inside a step.
    adc_index = {id(a): q for q, a in enumerate(S.adcs)}
    ns = len(S.samples)
    kx_s, ky_s = np.zeros(ns, int), np.zeros(ns, int)
    kx = ky = 0.0
    for i in range(1, n):
        kx += dkx[i]
        ky += dky[i]
        for r in pulses.get(i, ()):
            if dkx[i] != 0.0 or dky[i] != 0.0:
                raise ValueError(f"a gradient lobe is on at the center of the RF pulse at {r.tc:.3f} ms")
            kx, ky = (-kx, -ky) if r.refocus else (0.0, 0.0)
        if i in spoil_at:
            kx = ky = 0.0
        for q in samples.get(i, ()):
            kx_s[q], ky_s[q] = int(round(kx)), int(round(ky))
    return Plan(S.N, times, dt, dkx, dky, prop, props, pulses, spoil_at, samples,
                np.array([s["t"] for s in S.samples], float), np.array([s["shot"] for s in S.samples], int),
                np.array([adc_index[id(s["adc"])] for s in S.samples], int), np.array([s["j"] for s in S.samples], int),
                kx_s, ky_s, list(S.adcs))


# =============================================================================================== the run

@dataclass
class Acquisition:
    """Result of simulate_fast(): the samples.

    S                the Schedule that was simulated
    N                matrix size
    t, shot, adc, j  per sample, time-ordered: instant (ms), shot, index of its acquisition window in `adcs`,
                     index inside the window
    kx, ky           per sample: its k cell (integers)
    signal           per sample: S = sum over the pixels of Mx + i My (complex)
    adcs             the acquisition events; adcs[q].d is the entry of the sequence file and adcs[q].ctx its
                     variables, loop variables included
    ideal            {q: |M_perp| of every pixel when sample q was taken}, for the samples taken at k = 0 (the
                     first 32 of them), each an (N, N) array with row 0 at the top: the contrast the sequence
                     encodes at that echo
    Smax             sum of M0, the largest possible |S|
    seconds          time the simulation took; n_steps, n_props: steps taken and distinct propagators used
    """
    S: Schedule
    N: int
    t: np.ndarray
    shot: np.ndarray
    adc: np.ndarray
    j: np.ndarray
    kx: np.ndarray
    ky: np.ndarray
    signal: np.ndarray
    adcs: list
    ideal: dict
    Smax: float
    seconds: float
    n_steps: int
    n_props: int

    def __len__(self):
        return len(self.signal)

    def select(self, test) -> np.ndarray:
        """Boolean mask of the samples whose acquisition event passes `test`, for instance
            acq.select(lambda a: a.ctx.get("i") == 0)          the first pass of a repeat block (one echo)
            acq.select(lambda a: a.d.get("label") == "UTE")    one of two labeled acquisition windows
        """
        ok = np.array([bool(test(a)) for a in self.adcs], bool)
        return ok[self.adc]

    def kspace(self, mask=None) -> np.ndarray:
        """The samples on the N x N k grid, complex, row 0 = k_y = +HALF and column 0 = k_x = -HALF, NaN where no
        sample was taken.  When a cell was sampled more than once the last sample stays.  `mask` restricts the
        samples used."""
        N, H = self.N, (self.N - 1) // 2
        K = np.full((N, N), np.nan + 0j)
        sel = slice(None) if mask is None else np.asarray(mask, bool)
        K[H - self.ky[sel], self.kx[sel] + H] = self.signal[sel]
        return K


def simulate_fast(S: Schedule, M0, T1, T2, cache: int = 64) -> Acquisition:
    """Simulate a compiled sequence on a phantom and return the samples.

    S            a Schedule from mrisim.compile_sequence
    M0, T1, T2   per-pixel arrays of length N*N (row-major, row 0 at the top), as in a mrisim Phantom; scalars are
                 broadcast; inf disables the corresponding relaxation
    cache        how many propagators (complex arrays of N*N numbers) are kept at a time

    The state starts at equilibrium.  Each step multiplies m by the propagator of the step; then, at the instant
    that ends the step and in this order: RF pulses rotate (m, Mz), a spoiler zeroes m, samples are read.
    """
    t_start = time.perf_counter()
    P = plan(S)
    N, NPIX = S.N, S.npix
    M0 = np.asarray(M0, float).reshape(NPIX).copy()
    T1 = np.broadcast_to(np.asarray(T1, float), (NPIX,))
    T2 = np.broadcast_to(np.asarray(T2, float), (NPIX,))
    R1 = np.where(np.isfinite(T1), 1.0 / np.where(np.isfinite(T1), T1, 1.0), 0.0)      # relaxation rates, 1/ms
    R2 = np.where(np.isfinite(T2), 1.0 / np.where(np.isfinite(T2), T2, 1.0), 0.0)
    X, Y = pixel_xy(N)
    w = 2 * np.pi / N

    @lru_cache(maxsize=cache)
    def propagator(q: int) -> np.ndarray:
        """F of the distinct step q: decay over dt and precession from the area (dk_x, dk_y), per pixel."""
        dt, dkx, dky = P.props[q]
        if dkx == 0.0 and dky == 0.0:
            return np.exp(-dt * R2).astype(complex)
        return np.exp(-dt * R2 - 1j * w * (dkx * X + dky * Y))

    m = np.zeros(NPIX, complex)             # transverse magnetization, Mx + i My
    mz = M0.copy()                          # longitudinal magnetization, as it was at time t_z
    t_z = 0.0
    signal = np.zeros(len(P.t), complex)
    ideal: dict = {}
    events_at = set(P.pulses) | P.spoils    # steps that end with something other than samples
    prop, samples, times = P.prop.tolist(), P.samples, P.times
    center = set(np.flatnonzero((P.kx == 0) & (P.ky == 0)).tolist())
    for i in range(1, len(times)):
        m *= propagator(prop[i])
        if i in events_at:
            for r in P.pulses.get(i, ()):
                mz = M0 + (mz - M0) * np.exp(-(times[i] - t_z) * R1)       # bring Mz up to date, then rotate
                t_z = times[i]
                c, s = math.cos(math.radians(r.flip)), math.sin(math.radians(r.flip))
                if r.axis == "y":           # (Mx, Mz) -> (Mx c + Mz s, -Mx s + Mz c); My unchanged
                    mx = m.real * c + mz * s
                    mz = -m.real * s + mz * c
                    m = mx + 1j * m.imag
                else:                       # about x: (My, Mz) -> (My c - Mz s, My s + Mz c); Mx unchanged
                    my = m.imag * c - mz * s
                    mz = m.imag * s + mz * c
                    m = m.real + 1j * my
            if i in P.spoils:
                m[:] = 0
        for q in samples.get(i, ()):
            signal[q] = m.sum()
            if q in center and len(ideal) < 32:
                ideal[q] = np.abs(m).reshape(N, N)
    return Acquisition(S, N, P.t, P.shot, P.adc, P.j, P.kx, P.ky, signal, P.adcs, ideal, float(M0.sum()),
                       time.perf_counter() - t_start, len(times) - 1, len(P.props))
