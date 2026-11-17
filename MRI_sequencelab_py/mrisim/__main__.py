# By Bruno Mota - METAbio
"""Command-line entry point:  python -m mrisim [options]

    python -m mrisim                                                    interactive viewer, all sequences and phantoms
    python -m mrisim --sequence sequences/flair.json --phantom phantoms/brain.json -N 9 --param TI=600
    python -m mrisim --static --out figure.png                          static overview figure instead
    python -m mrisim --animate sequence.gif                             animation written to a file

--param id=value sets any parameter declared in the sequence file (repeatable); --t puts the cursor of the static
figure at a time in ms; --lang pt|en; --no-compress draws idle gaps at true scale.
"""
import argparse
import pathlib
import sys

from . import load_json, load_tissues, load_phantom, compile_sequence, simulate, reconstruct, overview, animate, lab

ROOT = pathlib.Path(__file__).resolve().parent.parent      # folder holding sequences/, phantoms/, tissues.json


def _parse_value(v: str):
    """Parameter value from the command line: int, float, or the string itself (for select options)."""
    try:
        return float(v) if "." in v or "e" in v.lower() else int(v)
    except ValueError:
        return v


def main(argv=None):
    ap = argparse.ArgumentParser(description="didactic MRI sequence simulator")
    ap.add_argument("--sequence", default=str(ROOT / "sequences" / "spin_echo.json"))
    ap.add_argument("--phantom", default=str(ROOT / "phantoms" / "square_air.json"))
    ap.add_argument("--tissues", default=str(ROOT / "tissues.json"))
    ap.add_argument("-N", type=int, default=5, help="matrix size (odd)")
    ap.add_argument("--param", action="append", default=[], help="id=value (repeatable)")
    ap.add_argument("--lang", default="pt")
    ap.add_argument("--no-compress", action="store_true")
    ap.add_argument("--out", default=None, help="static overview figure path (.png/.pdf)")
    ap.add_argument("--static", action="store_true", help="show the static overview instead of the interactive viewer")
    ap.add_argument("--animate", default=None, help="animation path (.gif or .mp4)")
    ap.add_argument("--t", type=float, default=None, help="cursor time (ms) for the static overview")
    a = ap.parse_args(argv)
    P = {}
    for kv in a.param:
        k, v = kv.split("=", 1)
        P[k] = _parse_value(v)

    # Simulate once; the viewer resimulates on its own when its selectors change.
    tissues = load_tissues(a.tissues)
    ph = load_phantom(a.phantom, a.N, tissues)
    S = compile_sequence(load_json(a.sequence), P, a.lang, a.N)
    sim = simulate(S, ph.M0, ph.T1, ph.T2)
    rec = reconstruct(sim)
    print(f"{S.defn['id']}: N={a.N}, {len(sim.samples)} samples, {len(sim.times)} time points, "
          f"residual max {abs(rec.residual).max():.4f}")

    if a.animate:
        print("animation:", animate(sim, ph, a.animate, compress=not a.no_compress, lang=a.lang))
    if a.out or a.static:
        fig, _ = overview(sim, ph, t=a.t, compress=not a.no_compress, lang=a.lang)
        if a.out:
            fig.savefig(a.out, dpi=130)
            print("figure:", a.out)
        else:
            import matplotlib.pyplot as plt
            plt.show()
    elif not a.animate:
        lab(ROOT / "sequences", ROOT / "phantoms", a.tissues, N=a.N, sequence=a.sequence, phantom=a.phantom,
            params=P, compress=not a.no_compress, lang=a.lang)


if __name__ == "__main__":
    sys.exit(main())
