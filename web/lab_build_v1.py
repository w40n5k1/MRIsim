import pathlib, json
d = pathlib.Path(__file__).parent
seqs = [json.loads((d / f).read_text(encoding='utf-8')) for f in ['seq_spin_echo.json', 'seq_fast_spin_echo.json', 'seq_flair.json']]
t = (d / 'lab_page_template_v1.html').read_text(encoding='utf-8')
t = t.replace('/*__ENGINE__*/', (d / 'seq_engine_v1.js').read_text(encoding='utf-8'))
t = t.replace('/*__DRAW__*/', (d / 'seq_draw_v1.js').read_text(encoding='utf-8'))
t = t.replace('/*__TISSUES__*/', 'const TISSUES = ' + json.dumps(json.loads((d / 'tissues.json').read_text(encoding='utf-8'))['tissues'], ensure_ascii=False, separators=(',', ':')) + ';')
t = t.replace('/*__SEQS__*/', 'const SEQUENCES = ' + json.dumps(seqs, ensure_ascii=False, separators=(',', ':')) + ';')
(d / 'sequence_lab_body.html').write_text(t, encoding='utf-8')          # artifact body (the viewer adds the skeleton)
page = '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sequence Lab 5×5</title></head><body style="margin:0">' + t + '</body></html>'
(d / 'sequence_lab_v1.html').write_text(page, encoding='utf-8')
print('ok', len(page))
