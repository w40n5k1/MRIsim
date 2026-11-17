"""Command line:  python -m mrisim --sequence sequences/flair.json --phantom phantoms/brain.json -N 9
Opens the interactive viewer (time slider, play, Fourier) unless --out (static figure) or --animate is given.
Parameters of the sequence can be set with --param TI=600 (repeatable). --animate out.gif writes an animation."""
import argparse, pathlib, sys
from . import load_json, load_tissues, load_phantom, compile_sequence, simulate, reconstruct, overview, animate, view, lab

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main(argv=None):
    ap = argparse.ArgumentParser(description="didactic MRI sequence simulator")
    ap.add_argument("--sequence", default=str(ROOT / "sequences" / "spin_echo.json"))
    ap.add_argument("--phantom", default=str(ROOT / "phantoms" / "square_air.json"))
    ap.add_argument("--tissues", default=str(ROOT / "tissues.json"))
    ap.add_argument("-N", type=int, default=5)
    ap.add_argument("--param", action="append", default=[], help="id=value (repeatable)")
    ap.add_argument("--lang", default="pt")
    ap.add_argument("--no-compress", action="store_true")
    ap.add_argument("--out", default=None, help="static overview figure path (.png/.pdf)")
    ap.add_argument("--static", action="store_true", help="show the static overview instead of the interactive viewer")
    ap.add_argument("--animate", default=None, help="animation path (.gif or .mp4)")
    ap.add_argument("--t", type=float, default=None, help="cursor time (ms) for the overview")
    a = ap.parse_args(argv)
    defn = load_json(a.sequence)
    P = {}
    for kv in a.param:
        k, v = kv.split("=", 1)
        try:
            P[k] = float(v) if "." in v or "e" in v.lower() else int(v)
        except ValueError:
            P[k] = v
    tissues = load_tissues(a.tissues)
    ph = load_phantom(a.phantom, a.N, tissues)
    S = compile_sequence(defn, P, a.lang, a.N)
    sim = simulate(S, ph.M0, ph.T1, ph.T2)
    rec = reconstruct(sim)
    print(f"{S.defn['id']}: N={a.N}, {len(sim.samples)} samples, {len(sim.times)} time points, residual max {abs(rec.residual).max():.4f}")
    if a.animate:
        print("animation:", animate(sim, ph, a.animate, compress=not a.no_compress, lang=a.lang))
    if a.out or a.static:
        fig, _ = overview(sim, ph, t=a.t, compress=not a.no_compress, lang=a.lang)
        if a.out:
            fig.savefig(a.out, dpi=130); print("figure:", a.out)
        else:
            import matplotlib.pyplot as plt
            plt.show()
    elif not a.animate:
        lab(ROOT / "sequences", ROOT / "phantoms", a.tissues, N=a.N, sequence=a.sequence, phantom=a.phantom, params=P,
            compress=not a.no_compress, lang=a.lang)


if __name__ == "__main__":
    sys.exit(main())
