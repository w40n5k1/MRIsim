# By Bruno Mota - METAbio
"""mrisim: a didactic 2D MRI pulse-sequence simulator (Python version of the browser Sequence Lab).

Typical use:

    from mrisim import *
    tissues = load_tissues("tissues.json")
    ph  = load_phantom("phantoms/brain.json", N=9, tissues=tissues)      # per-pixel M0, T1, T2 and tissue labels
    S   = compile_sequence(load_json("sequences/flair.json"), {"TI": 2750}, lang="pt", N=9)
    sim = simulate(S, ph.M0, ph.T1, ph.T2)                              # Bloch simulation on the time grid
    rec = reconstruct(sim)                                              # inverse DFT, mode by mode
    overview(sim, ph, t=2790)                                           # static figure at a cursor time
    view(sim, ph)                                                       # interactive window for this simulation
    lab()                                                               # interactive window with selectors

Modules: engine (sequence files -> schedule -> simulation), phantoms (tissues and phantom files), recon
(reconstruction), plot (static figures and animation), viewer (interactive window), phantom_editor (paint phantoms).
"""
from .engine import compile_sequence, simulate, param_defaults, load_json, segment_at, event_times, pixel_xy, Schedule, Simulation
from .phantoms import load_tissues, load_phantom, render_phantom, Phantom
from .recon import reconstruct, Recon
from .plot import plot_diagram, plot_pixels, plot_kspace, plot_image, overview, animate
from .viewer import lab, view, Lab, Viewer
from .phantom_editor import PhantomEditor

__all__ = ["compile_sequence", "simulate", "param_defaults", "load_json", "segment_at", "event_times", "pixel_xy",
           "Schedule", "Simulation", "load_tissues", "load_phantom", "render_phantom", "Phantom", "reconstruct", "Recon",
           "plot_diagram", "plot_pixels", "plot_kspace", "plot_image", "overview", "animate", "lab", "view", "Lab", "Viewer", "PhantomEditor"]
