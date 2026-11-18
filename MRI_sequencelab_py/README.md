<!-- By Bruno Mota - METAbio -->
# mrisim — didactic MRI sequence simulator (Python)

Self-contained Python port of the Sequence Lab page: the same physics, the same sequence files, a tissue
dictionary and phantom files, with matplotlib views and an animation. Made to be forked and extended.

```
tissues.json          tissue dictionary: key -> {name, pd, T1, T2 (ms, 3 T), color}
phantoms/*.json       tissue maps: shapes on a background (rendered at any odd N) or pixel maps painted in the editor
sequences/*.json      pulse sequences (spin_echo, fast_spin_echo, flair); same files as the browser version
mrisim/engine.py      expressions, sequence compiler (JSON -> schedule), Bloch simulation
mrisim/phantoms.py    tissue dictionary and phantom rendering
mrisim/recon.py       inverse DFT by conjugate-pair modes, ideal contrast image, residual
mrisim/plot.py        diagram, pixel array, k-space, image panels, static overview figure, animation to file
mrisim/viewer.py      interactive window: sequence / phantom / N / parameter selectors, time slider and play, cell mode,
                      mode-by-mode Fourier, residual gain
mrisim/phantom_editor.py  paint a phantom pixel by pixel (python -m mrisim.phantom_editor), save/load phantom files
mrisim/__main__.py    command line
tests/                samples = DFT(M0) x decay; the textbook spin-echo formula; samples independent of the time grid;
                      sample-by-sample cross-check with the JavaScript engine; phantom files and editor; data files
                      identical to the browser version's
```

Requirements: Python 3.10+, numpy, matplotlib (pillow for .gif, ffmpeg for .mp4); scipy and tkinter for the phantom
editor only; pytest for the tests (`python -m pytest -q`; the two checks against the browser version need its folder,
`MRI_sequencelab_web`, next to this one, and the engine cross-check needs node: they are skipped otherwise).

## Use

```
python -m mrisim                                                   # interactive viewer (all sequences and phantoms in the two folders)
python -m mrisim --static                                          # static overview instead
python -m mrisim --sequence sequences/flair.json --phantom phantoms/brain.json -N 9 --param TI=600 --out flair.png
python -m mrisim --sequence sequences/fast_spin_echo.json --param order=centric --animate fse.gif
python -m mrisim.phantom_editor [phantoms/brain.json] -N 9         # phantom editor; save into phantoms/ to see it in the viewer
```

```python
from mrisim import *
tissues = load_tissues("tissues.json")
ph  = load_phantom("phantoms/brain.json", N=9, tissues=tissues)         # ph.M0, ph.T1, ph.T2, ph.label per pixel
S   = compile_sequence(load_json("sequences/flair.json"), {"TI": 2750}, lang="pt", N=9)
sim = simulate(S, ph.M0, ph.T1, ph.T2)      # sim.times, sim.Mx/My/Mz (Nt x N^2), sim.signal, sim.kx/ky, sim.samples
rec = reconstruct(sim)                      # rec.modes (DC + conjugate pairs), rec.image, rec.ideal, rec.residual, rec.phase0
fig, rec = overview(sim, ph, t=2790)        # static: diagram + panels at a cursor time
view(sim, ph)                               # interactive viewer for this one simulation
lab(N=9, lang="en")                         # viewer with selectors; lab(sequences_dir, phantoms_dir, tissues_path, ...)
```

`sim.samples` is the list of acquired k-space points: `{t, shot, adc, j, idx, kx, ky, S}`; `sim.kspace()` puts
them on the N x N grid. `rec.partial(n)` is the sum of the first n modes (increasing |k|), for the mode-by-mode
reconstruction.

`adc` is the acquisition event the sample belongs to: `s["adc"].d` is its entry in the sequence file and
`s["adc"].ctx` holds the variables at that event, loop variables included. A multi-echo acquisition is therefore
rebuilt one echo at a time: `reconstruct(sim, samples=[s for s in sim.samples if s["adc"].ctx["i"] == 0])` for the
first pass of a `repeat` block with loop variable `i`, or select by `s["adc"].d.get("label")`. When a k cell has
several samples, the reconstruction uses the last one and the ideal image uses the first sample at k = 0 (in the
browser page too); a cell without a sample counts as zero.

## Model (common to both versions)

Rotating frame at the Larmor frequency; N x N isochromats, one per pixel, at x, y in {-(N-1)/2, ..., (N-1)/2};
transverse magnetization precesses only under gradients, m -> m exp(-i 2 pi (dkx x + dky y)/N), k in units of
1/FOV, so the samples are the forward DFT of the pixel magnetizations; instantaneous hard RF pulses (the whole
rotation at the pulse center, relaxation never interrupted, so the textbook formulas hold exactly; the drawn width
only animates the rotation); exponential relaxation per pixel; signal S = sum of
(Mx + i My). The k label follows the magnetization created by the latest excitation (reset to 0 at its center,
k -> -k at a refocusing pulse); whatever transverse magnetization survives from earlier shots keeps its own phase and
enters the samples as a ghost, which is what the spoiler removes. Idle gaps (TR tails, the TI wait) use a coarse
time grid and are drawn compressed. Reconstruction: the phase of the echo, arg S(0), is removed from the samples
(the echo forms along +x after a 90 degree pulse about y, along +y after one about x; the object is real only up
to that phase); then image(r) = (1/N^2) sum_k S(k) exp(+2 pi i k.r/N), grouped in conjugate pairs, each pair one
real cosine. The residual compares |image| with the sequence's ideal contrast image (each pixel's |M_perp| at the
k = 0 sample).

Matrix sizes. N is odd. The viewer and the editor offer 3 to 35 (the browser page, 3 to 17). The engine has no
limit of its own, but it stores the magnetization of every pixel at every instant, which is what the viewer
shows: about 40 MB at N = 15, 0.6 GB at N = 35 and 3.4 GB at N = 63 for a spin echo. For images at larger N,
without the history, use the package `mrisim_fast` in the folder `MRI_sequencelab_py_fast`, next to this one: it
runs the same files, gives the same samples and goes to N = 255.

## Sequence file format (didactic-mri-sequence/1)

Top level: id, name, subtitle, lead, notes (string or {pt, en}); relax {T1, T2} (suggested global values);
phantom (suggested); params (controls: slider {id, label, min, max, step, default, unit}, checkbox, select with
options whose `values` become variables); constants; tables; vars (derived, in order); shots; TR; shotVars;
shotLabel; envelope; events.

Events, per shot, times in ms relative to the shot start, any number may be an expression string:
```
{"type": "rf",    "t", "dur", "flip", "axis": "x"|"y", "refocus": true?, "desc"}     refocus: k -> -k at the center
{"type": "grad",  "axis": "x"|"y", "t", "dur", "area", "label"?, "ghost"?, "desc"}    area in dk
{"type": "adc",   "t", "dur", "n", "label"?, "desc"}                                  n samples at sub-interval centers
{"type": "spoil", "t", "if"?}                                                         transverse magnetization zeroed
{"type": "repeat", "count", "var", "vars"?, "events"}
{"type": "band",  "t", "label"}      {"type": "arrow", "t", "to", "label", "sub"?, "style", "level"}      {"type": "mark", "t", "label"?, "style", "level"?}
```
Expressions see params, constants, tables, vars, N, HALF = (N-1)/2, the shot index n, loop variables, and the
helpers range(a, b, step) (inclusive), centric(H), indexOf(list, v), rev(list), plus min, max, abs, floor, ceil, sqrt,
exp, log, sin, cos and PI. This subset is common to the Python and JavaScript engines; keep to it so files stay
portable. Not portable: `pi` in lower case (Python only), `%` with a negative operand and `round` at half-integers
(the two languages disagree), `//`, `and`/`or`/`not`, `&&`/`||`, conditional expressions and method calls; a list
index must be an integer (`pe[n]`, not `pe[HALF]`): wrap a computed index or count in `floor(...)`, because in
Python a division gives a float and so does a slider of the viewer once it is moved. For a conditional event use
its "if" field (any event type). Labels support a_b, a_{bc}, a^b; `${expr}` is replaced by its value in `desc`,
in `shotLabel` and in the labels of `grad` and `band` events, and printed as written in the labels of `adc`,
`mark` and `arrow` events.
Expressions are evaluated as code (Python eval / JavaScript Function): load only files from a known source.

## Phantom file format (didactic-mri-phantom/1)

```
{"id", "name", "background": "air", "pd": "0.2 + 0.8*(x + 1)/2"?,
 "shapes": [{"type": "rect", "tissue": "gm", "x": [x0, x1], "y": [y0, y1]},
            {"type": "disk", "tissue": "csf", "center": [cx, cy], "r": r}]}
```
Coordinates are normalized to [-1, 1] over the field of view (pixel centers at x = (col - H)/(N/2)); shapes are
painted in order; a disk covers at least its nearest pixel; `pd` is an optional proton-density factor in x, y.
A phantom painted in the editor carries a pixel map instead (shapes, if any, go on top of it):
```
{"id", "name", "background": "air", "grid": {"N": 9, "labels": [["air", "gm", ...], ...]}}
```
rendered at any odd N by nearest-neighbor resampling of the pixel centers.

The editor (`python -m mrisim.phantom_editor`) shows the dictionary's tissues as colored buttons; left-drag paints the
selected tissue, right-drag paints air; the tools (fill, erode, grow, fill holes, convex hull, convex core, open, close,
smooth, negative, undo) act on the selected tissue only. N is odd, 3 to 35 (the browser editor stops at 17).

## Extending

A new sequence is a new JSON file (copy one and edit); a new tissue is a line in tissues.json; a new phantom is a
list of shapes, or a grid painted in the editor. For physics beyond the model (several isochromats per pixel for T2* and imperfect refocusing,
slice profiles, off-resonance, noise) edit `simulate` in engine.py: it is a plain loop over the time grid with
numpy arrays over pixels.
