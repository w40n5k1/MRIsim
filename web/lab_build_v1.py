# By Bruno Mota - METAbio
"""Build script of the Sequence Lab page:  python lab_build_v1.py

Reads the page template (lab_page_template_v1.html) and inlines into it the sequence engine (seq_engine_v1.js),
the drawing module (seq_draw_v1.js), the tissue dictionary (tissues.json), the sequence files (seq_*.json) and the
phantom files (phantom_*.json), at the placeholders /*__ENGINE__*/, /*__DRAW__*/, /*__TISSUES__*/, /*__SEQS__*/ and
/*__PHANTOMS__*/.  Does the same for the phantom editor page (phantom_editor_template_v1.html: engine and tissues).

Writes four files:
  sequence_lab_body.html     the page without the <html>/<head>/<body> skeleton, as the artifact viewer expects it
  sequence_lab_v1.html       the self-contained page to open in any browser (and to host, e.g. on GitHub Pages)
  phantom_editor_body.html   the phantom editor, same two forms
  phantom_editor_v1.html
The simulator links to phantom_editor_v1.html (same folder); the editor links back to the page that opened it.
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
SEQUENCE_FILES = ["seq_spin_echo.json", "seq_fast_spin_echo.json", "seq_flair.json"]
PHANTOM_FILES = ["phantom_square_air.json", "phantom_letter_f_air.json", "phantom_uniform.json", "phantom_square.json",
                 "phantom_letter_f.json", "phantom_ramp.json", "phantom_brain.json", "phantom_face.json"]      # menu order


def read(name):
    return (HERE / name).read_text(encoding="utf-8")


def compact(obj):
    """JSON without whitespace, non-ASCII kept as is (the labels carry accents and symbols)."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def standalone(body: str, title: str) -> str:
    """Wrap a page body in the html/head/body skeleton; the template's <title> and viewport meta move to the head."""
    head_lines, body_lines = [], []
    for line in body.split("\n"):
        if line.startswith("<title>") or line.startswith('<meta name="claude:viewport-fit-sides"'):
            head_lines.append(line)
        else:
            body_lines.append(line)
    return ("<!doctype html>\n<!-- By Bruno Mota - METAbio -->\n<html><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">" + "".join(head_lines)
            + "</head><body style=\"margin:0\">\n" + "\n".join(body_lines) + "\n</body></html>\n")


def main():
    body = read("lab_page_template_v1.html")
    body = body.replace("/*__ENGINE__*/", read("seq_engine_v1.js"))
    body = body.replace("/*__DRAW__*/", read("seq_draw_v1.js"))
    body = body.replace("/*__TISSUES__*/", "const TISSUES = " + compact(json.loads(read("tissues.json"))["tissues"]) + ";")
    body = body.replace("/*__SEQS__*/", "const SEQUENCES = " + compact([json.loads(read(f)) for f in SEQUENCE_FILES]) + ";")
    body = body.replace("/*__PHANTOMS__*/", "const PHANTOMS = " + compact([json.loads(read(f)) for f in PHANTOM_FILES]) + ";")
    (HERE / "sequence_lab_body.html").write_text(body, encoding="utf-8")
    page = standalone(body, "Sequence Lab")
    (HERE / "sequence_lab_v1.html").write_text(page, encoding="utf-8")
    print("ok", len(page))

    editor = read("phantom_editor_template_v1.html")
    editor = editor.replace("/*__ENGINE__*/", read("seq_engine_v1.js"))
    editor = editor.replace("/*__TISSUES__*/", "const TISSUES = " + compact(json.loads(read("tissues.json"))["tissues"]) + ";")
    (HERE / "phantom_editor_body.html").write_text(editor, encoding="utf-8")
    page = standalone(editor, "Editor de fantomas")
    (HERE / "phantom_editor_v1.html").write_text(page, encoding="utf-8")
    print("ok editor", len(page))


if __name__ == "__main__":
    main()
