# By Bruno Mota - METAbio
"""Command line: simulate one sequence on one phantom and show k-space, the image, the ideal image and the residual.

    python -m mrisim_fast                                                 spin echo on the brain, N = 127
    python -m mrisim_fast --sequence flair --phantom brain -N 255 --param TI=2750
    python -m mrisim_fast --sequence my_sequence.json --phantom my_phantom.json -N 63 --out image.png --no-show

--sequence and --phantom take a file, or the name of one of the files of the didactic package (sequences/NAME.json,
phantoms/NAME.json).  N is odd, 3 to 255.  --param id=value sets a parameter of the sequence and can be repeated.
"""
from __future__ import annotations

import argparse
import pathlib

from . import DATA, N_MAX, compile_sequence, load_json, load_phantom, load_tissues, param_defaults, reconstruct_fast, show, simulate_fast


def _resolve(arg: str, folder: str) -> pathlib.Path:
    """A path as given, or the file NAME.json of the didactic package's folder."""
    p = pathlib.Path(arg)
    return p if p.suffix == ".json" or p.exists() else DATA / folder / f"{arg}.json"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mrisim_fast", description="Fast simulation: samples and image, no history.")
    ap.add_argument("--sequence", default="spin_echo", help="sequence file, or the name of one in the didactic package")
    ap.add_argument("--phantom", default="brain", help="phantom file, or the name of one in the didactic package")
    ap.add_argument("--tissues", default=str(DATA / "tissues.json"))
    ap.add_argument("-N", type=int, default=127, help=f"matrix size (odd, 3 to {N_MAX})")
    ap.add_argument("--param", action="append", default=[], help="id=value (repeatable)")
    ap.add_argument("--lang", default="pt")
    ap.add_argument("--out", help="save the figure to this file")
    ap.add_argument("--no-show", action="store_true", help="do not open a window")
    a = ap.parse_args(argv)
    if a.N % 2 == 0 or not 3 <= a.N <= N_MAX:
        ap.error(f"N must be odd, between 3 and {N_MAX}")
    defn = load_json(_resolve(a.sequence, "sequences"))
    P = param_defaults(defn)
    for item in a.param:
        k, v = item.split("=", 1)
        try:
            P[k] = float(v) if "." in v or "e" in v.lower() else int(v)
        except ValueError:
            P[k] = v
    tissues = load_tissues(a.tissues)
    ph = load_phantom(_resolve(a.phantom, "phantoms"), a.N, tissues)
    acq = simulate_fast(compile_sequence(defn, P, a.lang, a.N), ph.M0, ph.T1, ph.T2)
    img = reconstruct_fast(acq)
    name = defn.get("name")
    name = name.get(a.lang, name.get("en", "")) if isinstance(name, dict) else (name or defn.get("id", ""))
    values = ", ".join(f"{k} = {v:g}" if isinstance(v, (int, float)) else f"{k} = {v}" for k, v in P.items())
    print(f"{name}: N = {a.N}, {len(acq)} samples, {acq.n_steps} steps, {acq.n_props} propagators, {acq.seconds:.2f} s"
          + (f"; residual max {abs(img.residual).max():.4f}" if img.residual is not None else ""))
    show(acq, img, ph, title=f"{name} · {values}", out=a.out, block=not a.no_show)


if __name__ == "__main__":
    main()
