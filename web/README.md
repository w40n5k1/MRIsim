# Sequence Lab (N x N) — sequence simulator with sequences described as text files

Current version: v1. Open `sequence_lab_v1.html` in any browser (self-contained). The heading is a selector between the
loaded sequences (Spin Echo is the default; Fast Spin Echo and FLAIR are the others). The pixel matrix is N x N with N odd,
3..15 (selector next to the phantom); Bloch spheres are drawn up to N = 9, transverse arrows only up to N = 15.
All phantoms are tissue maps (tissues.json: gray matter, white matter, CSF, fat, cortical bone, air, lesion; pd, T1, T2 at
3 T and a display color): square and letter F 'filled/empty' = GM on air (default, the simplest case); uniform = GM;
square and letter F = GM on WM; ramp = GM with a proton-density ramp; brain = air,
scalp fat, skull, CSF, GM, WM, ventricle and one lesion. The pixel pane has three cell modes: intensity (gray = M0),
intensity + phase (transverse magnetization), tissue (color). The 'global T1/T2' box replaces the tissue values by the
sliders. The reconstructed image is shown as a magnitude, like a scanner. "load sequence (.json)" adds a sequence
from disk; "copy sequence JSON" exports the current one.

## Files
- sequence_lab_v1.html        the page (engine, drawer and the two sequences embedded)
- seq_engine_v1.js            generic engine: compiles a sequence file into a schedule and runs the Bloch simulation (global SeqEngine)
- seq_draw_v1.js              generic diagram builder (global SeqDraw)
- seq_spin_echo.json          Spin Echo sequence file
- seq_fast_spin_echo.json     Fast Spin Echo sequence file
- seq_flair.json              FLAIR (inversion recovery + fast spin-echo readout) sequence file
- tissues.json                tissue dictionary (embedded in the page by the build)
- seq_spin_echo.svg, seq_fast_spin_echo.svg, seq_flair.svg   static diagrams (default parameters)
- lab_page_template_v1.html   page source with /*__ENGINE__*/, /*__DRAW__*/, /*__SEQS__*/ placeholders
- lab_build_v1.py             rebuilds the page:  python lab_build_v1.py
- test_lab_v1.js              check: samples = DFT(M0) x T2 decay for both sequences; regenerates the SVGs:  node test_lab_v1.js

## Sequence file format (didactic-mri-sequence/1)
Top level: id, name, subtitle, lead, notes (each a string or {pt, en}); relax {T1, T2} (suggested defaults); phantom (suggested);
params (controls), constants, tables, vars (derived, evaluated in order), shots (repetitions), TR, shotVars,
shotLabel, envelope (draw e^{-t/T2}), events.

params: {id, label, min, max, step, default, unit, title} slider; {id, type: "checkbox", default, label};
{id, type: "select", default, label, options: [{value, label, values: {name: value...}}]} — the chosen option's
`values` become variables (e.g. the ky table "pe").

events (per shot; times in ms relative to the shot start; any number may be an expression string):
- {type: "rf", t, dur, flip, axis: "x"|"y", refocus: true?, desc}   hard pulse; refocus marks k -> -k at its center
- {type: "grad", axis: "x"|"y", t, dur, area, label?, ghost?, desc}  constant lobe; area in dk = 1/FOV; ghost names a table drawn dashed
- {type: "adc", t, dur, n, label?, desc}                             n samples at the centers of n equal sub-intervals
- {type: "spoil", t, if?}                                            ideal spoiler: transverse magnetization zeroed at t
- {type: "repeat", count, var, vars?, events}                        loop; var and vars visible inside
- {type: "band", t, label}                                           a background band (and header) starting at t; default: one per shot
- {type: "arrow", t, to, label, sub?, style: "te"|"tr"|"esp", level: 0|1}   dimension arrow drawn in the first shot
- {type: "mark", t, label?, style, level?}                           dashed vertical in the first shot
Any event may carry "if": expression (0 = omitted). Expressions see the params, constants, tables, vars, N and HALF = (N-1)/2, the helpers range(a, b, step) and centric(H),
the shot index n, loop variables, and Math.*. Labels support a_b, a_{bc}, a^b markup. desc is shown in the status line.
Expressions are evaluated as JavaScript: load only files from a known source.

Idle gaps longer than 25 ms (TR tails, the TI wait) are drawn compressed (dashed) when 'compress TR' is on, and the
simulation uses a coarse time grid across them, so long TI/TR cost nothing.

Physics: rotating frame, N x N isochromats (one per pixel), per-pixel T1/T2, hard pulses with relaxation suspended, exponential relaxation,
S = sum(Mx + i My). Samples are the forward DFT of the pixel magnetizations.
