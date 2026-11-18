# By Bruno Mota - METAbio
"""Tests of the fast package.  Run from the folder of this package:  python -m pytest -q

They need the didactic package (the folder MRI_sequencelab_py next to this one), which is also what the fast
package itself needs.

1. test_same_samples_as_the_didactic_engine: sample by sample, the k cell and the complex signal equal those of
   mrisim.simulate, which advances every pixel on a fine time grid.
2. test_same_image_as_the_didactic_reconstruction: the FFT image, the ideal image and the residual equal those of
   mrisim.reconstruct, which sums cosine modes.
3. test_spin_echo_textbook_formula_at_large_n: at N = 63, beyond what the didactic engine holds comfortably, the
   k = 0 sample of a uniform phantom equals the instantaneous-pulse spin-echo formula.
4. test_every_cell_once_and_a_sharp_image_at_127: N = 127 fills k-space exactly once, and the image of a
   two-tissue phantom has the right values in the interior of each region.
5. test_echo_phase_and_selection: the image does not depend on the axis of the excitation pulse, and the samples
   of one echo of a train can be selected and reconstructed.
"""
import copy

import numpy as np
import pytest

from mrisim_fast import DATA, compile_sequence, load_json, load_phantom, load_tissues, reconstruct_fast, simulate_fast
from mrisim import reconstruct, simulate

TIS = load_tissues(DATA / "tissues.json")
SEQ = lambda name: load_json(DATA / "sequences" / f"{name}.json")
PH = lambda name, N: load_phantom(DATA / "phantoms" / f"{name}.json", N, TIS)
CASES = [("spin_echo", "ramp", 9), ("fast_spin_echo", "square", 9), ("flair", "brain", 9), ("spin_echo", "brain", 15),
         ("fast_spin_echo", "monalisa", 15)]


@pytest.mark.parametrize("seq,phantom,N", CASES)
def test_same_samples_as_the_didactic_engine(seq, phantom, N):
    ph = PH(phantom, N)
    S = compile_sequence(SEQ(seq), None, "pt", N)
    ref = simulate(S, ph.M0, ph.T1, ph.T2)
    acq = simulate_fast(S, ph.M0, ph.T1, ph.T2)
    assert len(acq) == len(ref.samples)
    assert [(s["kx"], s["ky"]) for s in ref.samples] == list(zip(acq.kx.tolist(), acq.ky.tolist()))
    assert np.abs(acq.signal - np.array([s["S"] for s in ref.samples])).max() < 1e-10 * ref.Smax


@pytest.mark.parametrize("seq,phantom,N", CASES)
def test_same_image_as_the_didactic_reconstruction(seq, phantom, N):
    ph = PH(phantom, N)
    S = compile_sequence(SEQ(seq), None, "pt", N)
    rec = reconstruct(simulate(S, ph.M0, ph.T1, ph.T2))
    img = reconstruct_fast(simulate_fast(S, ph.M0, ph.T1, ph.T2))
    assert np.abs(img.image.ravel() - rec.image).max() < 1e-10
    assert np.abs(img.ideal.ravel() - rec.ideal).max() < 1e-10
    assert np.abs(img.residual.ravel() - rec.residual).max() < 1e-9
    assert abs(img.phase0 - rec.phase0) < 1e-9


@pytest.mark.parametrize("tissue,TE,TR", [("gm", 32, 500), ("wm", 80, 2000), ("csf", 120, 4000)])
def test_spin_echo_textbook_formula_at_large_n(tissue, TE, TR):
    N, t = 63, TIS[tissue]
    S = compile_sequence(SEQ("spin_echo"), {"TE": TE, "TR": TR}, "pt", N)
    acq = simulate_fast(S, np.full(N * N, t["pd"]), t["T1"], t["T2"])
    q = np.flatnonzero((acq.kx == 0) & (acq.ky == 0))
    assert len(q) == 1 and acq.shot[q[0]] >= 1                    # the center line is not the first repetition
    exact = t["pd"] * N * N * (1 - 2 * np.exp(-(TR - TE / 2) / t["T1"]) + np.exp(-TR / t["T1"])) * np.exp(-TE / t["T2"])
    assert abs(abs(acq.signal[q[0]]) - exact) < 1e-10 * exact


def test_every_cell_once_and_a_sharp_image_at_127():
    N = 127
    ph = PH("square", N)                                          # a square of gray matter on white matter
    acq = simulate_fast(compile_sequence(SEQ("spin_echo"), {"TE": 32, "TR": 4000}, "pt", N), ph.M0, ph.T1, ph.T2)
    assert len(acq) == N * N and len(set(zip(acq.kx.tolist(), acq.ky.tolist()))) == N * N
    assert not np.isnan(acq.kspace()).any()
    img = reconstruct_fast(acq)
    assert np.abs(img.residual).max() < 0.05                      # ringing at the edges of the square only
    lab = np.array(ph.label).reshape(N, N)
    for key in ("gm", "wm"):                                      # deep inside each region the image is the ideal one
        inside = (lab == key)
        inside[1:-1, 1:-1] &= (lab[:-2, 1:-1] == key) & (lab[2:, 1:-1] == key) & (lab[1:-1, :-2] == key) & (lab[1:-1, 2:] == key)
        inside[[0, -1], :] = inside[:, [0, -1]] = False
        assert np.abs(img.residual[inside]).mean() < 0.01


def test_echo_phase_and_selection():
    N = 31
    ph = PH("letter_f_air", N)
    base = SEQ("spin_echo")
    ref = reconstruct_fast(simulate_fast(compile_sequence(base, {"TR": 4000}, "pt", N), ph.M0, ph.T1, ph.T2))
    d = copy.deepcopy(base)
    for e in d["events"]:
        if e["type"] == "rf" and not e.get("refocus"):
            e["axis"] = "x"                                       # the echo now forms along +y
    img = reconstruct_fast(simulate_fast(compile_sequence(d, {"TR": 4000}, "pt", N), ph.M0, ph.T1, ph.T2))
    assert abs(abs(img.phase0) - np.pi / 2) < 1e-9 and np.abs(img.image - ref.image).max() < 1e-9
    # One echo of the fast spin echo train: N samples, one k line.
    acq = simulate_fast(compile_sequence(SEQ("fast_spin_echo"), None, "pt", N), ph.M0, ph.T1, ph.T2)
    first = acq.select(lambda a: a.ctx.get("i") == 0)
    assert first.sum() == N and len(set(acq.ky[first].tolist())) == 1
    assert reconstruct_fast(acq, first).n_acquired == N
