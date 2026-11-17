<!-- By Bruno Mota - METAbio -->
# MRI Sequence Lab — browser version (v1)

Two self-contained pages; open them in any browser, nothing to install.

- `sequence_lab_v1.html`: the simulator. The heading selects the sequence (Spin Echo, Fast Spin Echo, FLAIR, or
  any sequence file loaded from disk). Below it: the sequence diagram with a time cursor; the N x N pixel array
  (Bloch spheres up to N = 9, transverse arrows only above that); k-space filling in as the samples are acquired;
  the inverse Fourier reconstruction, one mode at a time, with the ideal image and the residual; a tab with the
  equations behind each pane.
- `phantom_editor_v1.html`: the phantom editor. The simulator links to it and it links back, so keep the two
  files in the same folder.

Hosted copies: https://w40n5k1.github.io/MRIsim/ and https://w40n5k1.github.io/MRIsim/phantom_editor_v1.html
(the path is case-sensitive).

Controls of the simulator, row by row: acquisition (play, restart, step by event, speed); the parameters declared
by the sequence file, then the global T1/T2 override; object (the phantom), N (odd, 3 to 15), linear intensity,
compress TR, language (pt/en); copy diagram SVG, load sequence (.json), copy sequence JSON, load phantom (.json),
phantom editor. A painted phantom (one that carries a `grid`) sets N to its own size when chosen or loaded.

## Files

```
sequence_lab_v1.html             built page: the simulator          (generated, do not edit)
phantom_editor_v1.html           built page: the phantom editor     (generated, do not edit)
lab_page_template_v1.html        source of the simulator: markup, styles, interface code and texts (pt/en)
phantom_editor_template_v1.html  source of the editor
seq_engine_v1.js                 engine: expressions, sequence compiler, phantom renderer, Bloch simulation (global SeqEngine)
seq_draw_v1.js                   sequence diagram as SVG (global SeqDraw)
seq_*.json                       the three sequence files embedded in the simulator
phantom_*.json                   the nine phantom files embedded in the simulator
tissues.json                     tissue dictionary, embedded in both pages
lab_build_v1.py                  build script
test_lab_v1.js                   tests (node)
```

`seq_*.json`, `phantom_*.json` and `tissues.json` are copies of the data files of the Python version
(`sequences/*.json`, `phantoms/*.json`, `tissues.json` there). Change both copies together; the Python tests
compare them when the two folders sit side by side.

## Build and test

```
python lab_build_v1.py     inlines the engine, the drawer, the tissues, the sequences and the phantoms into the two
                           templates; writes the two pages, and the same pages without the html/head/body skeleton
                           (*_body.html, the form a claude.ai artifact takes)
node test_lab_v1.js        every acquired sample equals the DFT of the object times the decay (spin echo and fast spin
                           echo, N = 3, 5, 9); the spin-echo signal equals the textbook steady-state formula; the
                           phantom files render. Also rewrites seq_spin_echo.svg and seq_fast_spin_echo.svg.
```

A new sequence or phantom file is embedded only if it is listed in `SEQUENCE_FILES` or `PHANTOM_FILES` in the build
script (the build warns about files it finds and does not embed). Without rebuilding, any file can be loaded into
the page with the two load buttons.

## Model and file formats

The physical model and the formats of the sequence, phantom and tissue files are common to both versions and are
specified once, in the README of the Python version (`MRI_sequencelab_py/README.md`, next to this folder). In the
browser, sequence expressions are evaluated as JavaScript: load only files from a source you trust.
