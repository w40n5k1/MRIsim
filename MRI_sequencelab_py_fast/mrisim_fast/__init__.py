# By Bruno Mota - METAbio
"""mrisim_fast: the samples and the image of a pulse sequence, without the history.  For large matrices.

It runs the same sequence files, on the same phantoms, with the same model as the didactic package mrisim, and
gives the same samples; it keeps nothing but the samples, and works out in advance everything that does not
depend on the object.  mrisim stops near N = 35 because it stores the magnetization of every pixel at every
instant; this package goes to N = 255.  It has no viewer: use mrisim to watch a sequence, this to measure one.

    from mrisim_fast import *
    tissues = load_tissues(DATA / "tissues.json")
    ph  = load_phantom(DATA / "phantoms" / "brain.json", N=127, tissues=tissues)
    S   = compile_sequence(load_json(DATA / "sequences" / "spin_echo.json"), {"TE": 80, "TR": 3000}, N=127)
    acq = simulate_fast(S, ph.M0, ph.T1, ph.T2)       # the samples: acq.kx, acq.ky, acq.signal, ...
    img = reconstruct_fast(acq)                       # img.image, img.ideal, img.residual: (N, N) arrays
    show(acq, img, ph)                                # k-space, image, ideal image, residual

Modules: engine (plan and simulation), recon (FFT image), figure (one static figure).  The compiler, the phantom
renderer and the data files are those of mrisim (see _didactic), re-exported here for convenience; DATA is the
folder that holds sequences/, phantoms/ and tissues.json.
"""
from ._didactic import mrisim, DATA
from mrisim import compile_sequence, param_defaults, load_json, load_tissues, load_phantom, render_phantom, pixel_xy
from .engine import plan, simulate_fast, Plan, Acquisition
from .recon import reconstruct_fast, Image
from .figure import show

N_MAX = 255          # largest matrix the command line accepts; nothing in the engine depends on it

__all__ = ["DATA", "N_MAX", "compile_sequence", "param_defaults", "load_json", "load_tissues", "load_phantom", "render_phantom",
           "pixel_xy", "plan", "simulate_fast", "Plan", "Acquisition", "reconstruct_fast", "Image", "show"]
