# mrisim — a didactic MRI pulse-sequence simulator

Teaching material for an undergraduate course on the physics of magnetic resonance imaging (Física Médica, UFRJ).
An N × N array of isochromats (one per pixel, with per-tissue proton density, T1 and T2) is driven through a pulse
sequence in the rotating frame; the page shows the sequence diagram with a time cursor, the magnetization of every
pixel, k-space filling in as the samples are acquired, and the image rebuilt one Fourier mode at a time.

Two implementations share the same physics, the same tissue dictionary and the same sequence files:

| | where | run |
|---|---|---|
| **Browser page** (no install) | [`web/`](web) — built page: `web/sequence_lab_v1.html` | open the HTML file, or the hosted copy at **https://w40n5k1.github.io/mrisim/** |
| **Python package** | [`python/`](python) | `cd python && pip install -r requirements.txt && python -m mrisim` |

Sequences included: spin echo, fast spin echo (CPMG train, selectable k_y order), FLAIR (inversion + fast spin echo).
Phantoms: filled/empty squares and letters, uniform, proton-density ramp, and a schematic brain (gray and white matter,
CSF, fat, bone, a lesion). Languages: Portuguese and English.

## Model

Rotating frame; instantaneous hard RF pulses (the whole rotation at the pulse center, relaxation never interrupted, so
the textbook signal formulas hold exactly; the width drawn on the diagram only animates the rotation);
gradients act only through the phase 2π(Δk_x x + Δk_y y)/N, so the acquired samples are exactly the discrete Fourier
transform of the pixel magnetizations at the sampling instants; exponential T1/T2 relaxation per pixel; ideal
spoiler as an option. The reconstruction is the inverse DFT grouped in conjugate pairs, so each mode is a real cosine
pattern that can be added to the image one at a time. Idealizations are deliberate (exact 180°, no slice profile,
one isochromat per pixel): the point is to see the mechanism, not to match a scanner.

## Writing your own sequence

A sequence is a JSON file (`python/sequences/*.json`, identical to the ones embedded in the page): parameters that
become controls, timing expressions, and events (`rf`, `grad`, `adc`, `spoil`, `repeat`, plus annotation events for
bands, arrows and marks). The browser page loads such a file with the *carregar sequência* button; the Python
viewer lists every file in its `sequences/` folder. Phantoms are JSON shape lists in `python/phantoms/`, tissues a
table in `tissues.json`. Both formats are documented in [`python/README.md`](python/README.md).

## Getting the code

```
git clone https://github.com/w40n5k1/mrisim
```
or **Code → Download ZIP** on this page. Python 3.10+, numpy and matplotlib for the package; any browser for the page.

## License

MIT — see [LICENSE](LICENSE).
