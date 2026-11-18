<!-- By Bruno Mota - METAbio -->
# mrisim_fast: the same simulator without the history, for large matrices

The didactic package (`MRI_sequencelab_py`, package `mrisim`) advances every pixel on a fine time grid and keeps
the whole history, because its purpose is to show the magnetization at any instant and the image as a sum of
Fourier modes. That costs memory growing roughly as N^4 and stops it near N = 35. This package runs the same
sequence files, on the same phantoms, with the same model, and returns the same samples, to rounding. It keeps
nothing but the samples and reconstructs with an FFT. It goes to N = 255. It has no viewer: watch a sequence
with `mrisim`, measure one with `mrisim_fast`.

It is not self-contained. The sequence compiler, the phantom renderer and the data files are those of the
didactic package, so that a sequence file means the same thing in both: keep the folder `MRI_sequencelab_py`
next to this one (or put it on `PYTHONPATH`).

```
mrisim_fast/_didactic.py   finds the didactic package; DATA is its folder (sequences/, phantoms/, tissues.json)
mrisim_fast/engine.py      plan(): what the sequence asks for, step by step; simulate_fast(): the samples
mrisim_fast/recon.py       reconstruct_fast(): image, ideal image and residual by FFT
mrisim_fast/figure.py      show(): object, k-space, image, ideal image, residual in one figure
mrisim_fast/__main__.py    command line
tests/                     same samples and same image as mrisim; textbook formula at N = 63; N = 127
```

Requirements: Python 3.10+, numpy, matplotlib; pytest for the tests (`python -m pytest -q`).

## Use

```
python -m mrisim_fast                                                 spin echo on the brain, N = 127
python -m mrisim_fast --sequence flair --phantom brain -N 255 --param TI=2750
python -m mrisim_fast --sequence my_sequence.json --phantom my_phantom.json -N 63 --out image.png --no-show
```

`--sequence` and `--phantom` take a file, or the name of one of the files of the didactic package. N is odd,
3 to 255.

```python
from mrisim_fast import *
tissues = load_tissues(DATA / "tissues.json")
ph  = load_phantom(DATA / "phantoms" / "brain.json", N=127, tissues=tissues)
S   = compile_sequence(load_json(DATA / "sequences" / "spin_echo.json"), {"TE": 80, "TR": 3000}, N=127)
acq = simulate_fast(S, ph.M0, ph.T1, ph.T2)      # acq.kx, acq.ky, acq.signal, acq.t, acq.shot, acq.adc: one entry per sample
img = reconstruct_fast(acq)                      # img.image, img.ideal, img.residual: (N, N) arrays, row 0 at the top
show(acq, img, ph)
```

`acq.kspace()` puts the samples on the N x N grid. `acq.select(test)` gives the mask of the samples whose
acquisition event passes a test, and `reconstruct_fast(acq, mask)` rebuilds from those alone:
`acq.select(lambda a: a.ctx.get("i") == 0)` is the first echo of a `repeat` block, and
`acq.select(lambda a: a.d.get("label") == "UTE")` one of two labeled acquisition windows.

## How it is fast

The model has no approximation in time: between two events the evolution of a pixel is known in closed form.
The didactic engine does not use that, on purpose. This one does.

1. Memoryless. The state is one complex array `m = Mx + i My` and one real array `Mz`, N^2 numbers each.
   Nothing else is kept but the samples.
2. The time axis is cut only where something happens: the edges of the gradient lobes, the centers of the RF
   pulses, the spoilers and the sampling instants. A spin echo at N = 15 takes 330 steps instead of 7600.
3. Everything that does not depend on the object is worked out first (`plan`): the steps, their durations and
   gradient areas, what happens at the end of each one, and the k label of every sample.
4. A step multiplies `m` by one complex factor per pixel, `F = exp(-dt/T2) exp(-2 pi i (dk_x x + dk_y y)/N)`.
   `F` is computed once per distinct `(dt, dk_x, dk_y)` and reused: every dwell interval of every readout of a
   sequence shares one `F`. A spin echo at N = 255 has 66810 steps and 263 distinct factors.
5. `Mz` is not advanced step by step. It relaxes undisturbed between RF pulses, so it is brought up to date,
   `Mz -> M0 + (Mz - M0) exp(-dt/T1)`, only when a pulse needs it.
6. The image is one FFT of the k-space table, after the phase of the echo, `arg S(0)`, is removed; the didactic
   package builds the same image as a sum of `(N^2 + 1)/2` cosine modes.

What is left is one multiplication and one sum over the N^2 pixels per sample: about N^4 operations for a
sequence that fills k-space once, and a few arrays of N^2 numbers.

| N | samples | simulate | didactic package |
|---|---|---|---|
| 35 | 1225 | 0.02 s | 3 s, 0.6 GB of history |
| 63 | 3969 | 0.07 s | 8 s, 3.4 GB |
| 127 | 16129 | 0.5 s | not possible |
| 255 | 65025 | 7 s, 0.2 GB | not possible |

(Spin echo on the brain phantom, two cores of a cloud machine.)

## What to keep in mind at large N

- The sequence files scale with N as written. The single-shot sequences (fast spin echo, FLAIR) get an echo
  train of N echoes: at N = 127 and ESP = 26 ms it lasts 3.3 s, far beyond T2, and the image shows it. That is
  the physics of the file, not an error.
- The readout window keeps its duration, so the dwell time shrinks as 1/N.
- Shape phantoms are rendered at any N. A painted phantom is resampled from its own grid by nearest pixel.
- The model is still one isochromat per pixel: no T2*, no slice, no noise.
