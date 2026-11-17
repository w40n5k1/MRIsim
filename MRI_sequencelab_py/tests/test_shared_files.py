# By Bruno Mota - METAbio
"""The browser version embeds copies of this package's data files under other names:
    tissues.json            = tissues.json
    sequences/NAME.json     = seq_NAME.json
    phantoms/NAME.json      = phantom_NAME.json
This test compares the two sets when the browser version's folder, MRI_sequencelab_web/, sits next to this
package, and is skipped otherwise.  A skip is not a pass.
"""
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
WEB = ROOT.parent / "MRI_sequencelab_web"          # the browser version, next to this package
if not (WEB / "seq_engine_v1.js").exists():
    WEB = None


def _load(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


@pytest.mark.skipif(WEB is None, reason="browser version not found next to the package")
def test_shared_data_files_are_identical():
    pairs = [(ROOT / "tissues.json", WEB / "tissues.json")]
    pairs += [(f, WEB / f"seq_{f.name}") for f in sorted((ROOT / "sequences").glob("*.json"))]
    pairs += [(f, WEB / f"phantom_{f.name}") for f in sorted((ROOT / "phantoms").glob("*.json"))]
    missing = [w.name for _, w in pairs if not w.exists()]
    assert not missing, f"missing in the browser version: {missing}"
    different = [f.name for f, w in pairs if _load(f) != _load(w)]
    assert not different, f"content differs between the two versions: {different}"
    extra = {w.name for w in WEB.glob("seq_*.json")} | {w.name for w in WEB.glob("phantom_*.json")}
    extra -= {w.name for _, w in pairs}
    assert not extra, f"only in the browser version: {sorted(extra)}"
