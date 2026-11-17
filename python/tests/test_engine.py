# By Bruno Mota - METAbio
"""Tests of the simulation engine.  Run from the python/ folder:  python -m pytest -q

1. test_samples_are_dft: for spin echo and fast spin echo, every acquired sample equals the DFT of M0 at its
   (kx, ky), times the T2 decay since the center of the excitation pulse, times the longitudinal magnetization
   available at that instant (relative to the sum of M0).  This is the model's defining property: the readout
   samples the Fourier transform of the object, and the pulses are instantaneous.
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
        mz_avail = sim.Mz[sim.index_at(first.t) - 1].sum()        # just before the excitation (grid point before its start)
        mz_avail = ph.M0.sum() + (mz_avail - ph.M0.sum()) * np.exp(-(first.tc - sim.times[sim.index_at(first.t) - 1]) / T1)
        dec = np.exp(-(s["t"] - first.tc) / T2) * mz_avail / sim.Smax
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


@pytest.mark.parametrize("tissue", ["gm", "wm", "csf"])
@pytest.mark.parametrize("TE,TR", [(32, 2000), (100, 2000), (22, 100), (200, 300)])
def test_spin_echo_textbook_formula(tissue, TE, TR):
    """From the second repetition on (the 90 degree pulse erases the memory of Mz, so the steady state is reached
    at once), the k = 0 sample of a uniform phantom equals N^2 pd [1 - 2 exp(-(TR - TE/2)/T1) + exp(-TR/T1)]
    exp(-TE/T2): the instantaneous-pulse spin-echo formula, to machine precision."""
    N = 5
    t = TIS[tissue]
    defn = load_json(ROOT / "sequences" / "spin_echo.json")
    S = compile_sequence(defn, {"TE": TE, "TR": TR}, "pt", N)
    sim = simulate(S, np.full(N * N, t["pd"]), t["T1"], t["T2"])
    s00 = next(s for s in sim.samples if s["kx"] == 0 and s["ky"] == 0)
    assert s00["shot"] >= 1
    exact = t["pd"] * N * N * (1 - 2 * np.exp(-(TR - TE / 2) / t["T1"]) + np.exp(-TR / t["T1"])) * np.exp(-TE / t["T2"])
    assert abs(abs(s00["S"]) - exact) < 1e-9 * exact
