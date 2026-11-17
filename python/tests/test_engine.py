# By Bruno Mota - METAbio
"""Tests of the simulation engine.  Run from the python/ folder:  python -m pytest -q

1. test_samples_are_dft: for spin echo and fast spin echo, every acquired sample equals the DFT of M0 at its
   (kx, ky), times the T2 decay accumulated outside the pulses since the excitation, times the longitudinal
   magnetization available at that excitation (relative to the sum of M0).  This is the model's defining
   property: the readout samples the Fourier transform of the object.
2. test_matches_javascript: the Python engine and the JavaScript engine of the browser page give the same
   samples for the same sequence file and phantom (needs node and the JavaScript file next to the package).
3. test_no_spoiler_labels_and_ghost: without the spoiler every k cell is still acquired once (the k label resets
   at each excitation), and the only difference is the ghost of the previous shot's transverse magnetization,
   negligible when TR >> T2 and visible when TR ~ T2.
"""
import json
import pathlib
import shutil
import subprocess

import numpy as np
import pytest
from mrisim import load_json, load_tissues, load_phantom, compile_sequence, simulate, pixel_xy

ROOT = pathlib.Path(__file__).resolve().parent.parent
TIS = load_tissues(ROOT / "tissues.json")


@pytest.mark.parametrize("seq", ["spin_echo", "fast_spin_echo"])
@pytest.mark.parametrize("N", [3, 5, 9])
def test_samples_are_dft(seq, N):
    defn = load_json(ROOT / "sequences" / f"{seq}.json")
    ph = load_phantom(ROOT / "phantoms" / "square_air.json", N, TIS)
    S = compile_sequence(defn, None, "pt", N)
    T1, T2 = 80.0, 60.0
    sim = simulate(S, ph.M0, T1, T2)
    X, Y = pixel_xy(N)
    rfs = [e for e in S.events if e.type == "rf"]
    worst = 0.0
    for s in sim.samples:
        ref = (ph.M0 * np.exp(-2j * np.pi * (s["kx"] * X + s["ky"] * Y) / N)).sum()
        first = next(r for r in rfs if r.shot == s["shot"])
        tp = sum(r.dur for r in rfs if r is not first and r.shot == s["shot"] and r.t1 <= s["t"])
        mz_avail = sim.Mz[sim.index_at(first.t)].sum()
        dec = np.exp(-(s["t"] - first.t1 - tp) / T2) * mz_avail / sim.Smax
        worst = max(worst, abs(s["S"] - ref * dec))
    assert worst < 1e-6, worst
    assert sorted({s["ky"] for s in sim.samples}) == list(range(-(N - 1) // 2, (N - 1) // 2 + 1))


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
@pytest.mark.parametrize("seq,phantom", [("spin_echo", "square_air"), ("fast_spin_echo", "square"), ("flair", "brain")])
def test_matches_javascript(seq, phantom):
    js = next((c for c in (ROOT.parent / "lab" / "seq_engine_v1.js", ROOT.parent / "Sequence Lab" / "seq_engine_v1.js") if c.exists()), None)
    if js is None:
        pytest.skip("JavaScript engine not found next to the package")
    N = 9
    defn = load_json(ROOT / "sequences" / f"{seq}.json")
    ph = load_phantom(ROOT / "phantoms" / f"{phantom}.json", N, TIS)
    S = compile_sequence(defn, None, "pt", N)
    sim = simulate(S, ph.M0, ph.T1, ph.T2)
    script = f"""
const E=require({json.dumps(str(js))});const def={json.dumps(defn)};
const S=E.compile(def,E.paramDefaults(def),'pt',{N});
const sim=E.simulate(S,{json.dumps(list(map(float, ph.M0)))},{{T1:{json.dumps(list(map(float, ph.T1)))},T2:{json.dumps(list(map(float, ph.T2)))}}});
console.log(JSON.stringify(sim.samples.map(s=>[s.kx,s.ky,s.re,s.im])));"""
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout
    js_samples = {(kx, ky): complex(re, im) for kx, ky, re, im in json.loads(out)}
    worst = max(abs(js_samples[(s["kx"], s["ky"])] - s["S"]) for s in sim.samples)
    assert worst < 1e-4 * sim.Smax, worst


def test_no_spoiler_labels_and_ghost():
    """Without the spoiler every k cell is still acquired once (labels reset at each excitation); the residual
    transverse magnetization from earlier shots is the only difference, negligible when TR >> T2."""
    N = 9
    defn = load_json(ROOT / "sequences" / "spin_echo.json")
    ph = load_phantom(ROOT / "phantoms" / "ramp.json", N, TIS)
    S1 = compile_sequence(defn, {"spoiler": 1}, "pt", N)
    S0 = compile_sequence(defn, {"spoiler": 0}, "pt", N)
    k1 = simulate(S1, ph.M0, ph.T1, ph.T2).kspace()
    k0 = simulate(S0, ph.M0, ph.T1, ph.T2).kspace()
    assert not np.isnan(k0).any()
    assert np.abs(k0 - k1).max() < 1e-6
    S0 = compile_sequence(defn, {"spoiler": 0, "TR": 100, "TE": 22}, "pt", N)
    k0 = simulate(S0, ph.M0, ph.T1, ph.T2).kspace()
    S1 = compile_sequence(defn, {"spoiler": 1, "TR": 100, "TE": 22}, "pt", N)
    k1 = simulate(S1, ph.M0, ph.T1, ph.T2).kspace()
    assert not np.isnan(k0).any()
    assert np.abs(k0 - k1).max() > 1e-3        # TR ~ T2: the unspoiled ghost is visible
