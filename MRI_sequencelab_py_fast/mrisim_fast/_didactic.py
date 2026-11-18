# By Bruno Mota - METAbio
"""Locate the didactic package, mrisim, on which this one is built.

mrisim_fast replaces only two steps of mrisim, the simulation and the reconstruction.  The sequence compiler,
the phantom renderer and the data files (sequences, phantoms, tissues) are mrisim's, so that a sequence file
means the same thing in both.  mrisim is imported if Python already finds it; otherwise it is looked for in the
folder MRI_sequencelab_py, next to the folder of this package.
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent          # the folder of this package (MRI_sequencelab_py_fast)
SIBLING = HERE.parent / "MRI_sequencelab_py"                   # where the didactic package is expected

try:
    import mrisim
except ImportError:
    if not (SIBLING / "mrisim").is_dir():
        raise ImportError("mrisim_fast needs the didactic package mrisim: keep the folder MRI_sequencelab_py next to "
                          f"{HERE.name}, or add it to PYTHONPATH") from None
    sys.path.insert(0, str(SIBLING))
    import mrisim

DATA = pathlib.Path(mrisim.__file__).resolve().parent.parent   # folder holding sequences/, phantoms/, tissues.json
