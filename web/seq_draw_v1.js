/* seq_draw_v1.js — generic SVG drawing of a compiled sequence schedule (SeqEngine.compile) with its simulation:
 * rows RF, Gpe (y), Gfe (x), Signal; bands, arrows and marks declared in the sequence file; timeline cursor and
 * reveal clip for animation; optional compression of each shot's tail. Pure string builder (browser and node). */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.SeqDraw = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  const S_PX = 5, GUT = 172, RIGHT = 22, WTAIL = 50;
  const ROWS = { rf: 74, gy: 138, gx: 202, sig: 266 };
  const Y_LEVEL = [314, 344], HEIGHT = 376, AMP_SIG = 34;
  const C = { text: '#232323', base: '#8f8f8f', ghost: '#e6b58a', band: '#f4f4f4', hl: '#fff4d6', rf: '#2e7d2a', gy: '#d4701c', gx: '#2b5aa8',
    sig: '#1f4e9a', sigGhost: '#b8c7e0', sigFill: '#dbe5f6', env: '#9aa8bd', cursor: '#c2185b',
    te: '#1a9e2a', tr: '#e0231d', esp: '#7b1fa2' };
  const f = v => (Math.round(v * 100) / 100).toString();
  function phaseColor(phi) { let h = (phi / (2 * Math.PI)) % 1; if (h < 0) h += 1; return `hsl(${Math.round(h * 360)} 85% 46%)`; }
  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');
  // label markup: a_b or a_{bc} -> subscript, a^b / a^{bc} -> superscript
  function markup(s, base) {
    const sub = Math.round(base * 0.7), out = []; let i = 0; s = String(s);
    while (i < s.length) {
      const ch = s[i];
      if ((ch === '_' || ch === '^') && i + 1 < s.length) {
        let txt; if (s[i + 1] === '{') { const j = s.indexOf('}', i + 2); txt = s.slice(i + 2, j < 0 ? s.length : j); i = j < 0 ? s.length : j + 1; } else { txt = s[i + 1]; i += 2; }
        const dy = ch === '_' ? Math.round(base * 0.3) : -Math.round(base * 0.4);
        out.push(`<tspan font-size="${sub}" dy="${dy}">${esc(txt)}</tspan><tspan dy="${-dy}" font-size="${base}"> </tspan>`.replace('> </tspan>', '></tspan>'));
      } else { let j = i; while (j < s.length && s[j] !== '_' && s[j] !== '^') j++; out.push(esc(s.slice(i, j))); i = j; }
    }
    return out.join('');
  }
  const GAPMIN = 25;   // idle gaps longer than this (ms) are compressed to WTAIL px when compress is on
  // piecewise-linear display coordinate u(t): to-scale segments around events, compressed idle gaps
  function layout(S, compress) {
    const comp = compress !== false, MARGIN = 2;
    const busy = []; for (const e of S.events) if (e.type !== 'spoil') busy.push([Math.max(0, e.t - MARGIN), Math.min(S.tEnd, e.t1 + MARGIN)]);
    busy.sort((p, q) => p[0] - q[0]);
    const merged = []; for (const iv of busy) { const l = merged[merged.length - 1]; if (l && iv[0] <= l[1] + 1e-9) l[1] = Math.max(l[1], iv[1]); else merged.push([iv[0], iv[1]]); }
    const segs = []; let cur = 0, u = 0;                      // {t0, t1, u0, u1, compressed}
    const push = (t0, t1, c) => { if (t1 <= t0 + 1e-9) return; const du = c ? WTAIL / S_PX : t1 - t0; segs.push({ t0, t1, u0: u, u1: u + du, compressed: c }); u += du; };
    for (const [a0, b0] of merged) { if (a0 > cur) push(cur, a0, comp && a0 - cur > GAPMIN); push(a0, b0, false); cur = b0; }
    if (cur < S.tEnd) push(cur, S.tEnd, comp && S.tEnd - cur > GAPMIN);
    const uOf = t => { t = Math.max(0, Math.min(S.tEnd, t)); let lo = 0, hi = segs.length - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (segs[m].t0 <= t) lo = m; else hi = m - 1; } const g = segs[lo]; return g.u0 + (t - g.t0) * (g.u1 - g.u0) / (g.t1 - g.t0); };
    const tOf = uu => { uu = Math.max(0, Math.min(u, uu)); let lo = 0, hi = segs.length - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (segs[m].u0 <= uu) lo = m; else hi = m - 1; } const g = segs[lo]; return g.t0 + (uu - g.u0) * (g.t1 - g.t0) / (g.u1 - g.u0); };
    const xOfU = v => GUT + S_PX * v, xOf = t => xOfU(uOf(t));
    return { segs, uOf, tOf, xOf, xOfU, uEnd: u, s: S_PX, gutter: GUT, rows: ROWS, width: GUT + S_PX * u + RIGHT, height: HEIGHT, compressed: comp };
  }
  function rect(x1, x2, y0, h) { return `M${f(x1)},${f(y0)} L${f(x1)},${f(y0 - h)} L${f(x2)},${f(y0 - h)} L${f(x2)},${f(y0)} Z`; }
  function sincPath(xc, y0, halfW, A) {
    const K = 2.6, pts = [];
    for (let i = 0; i <= 90; i++) { const u = -1 + 2 * i / 90, x = u * K, s = Math.abs(x) < 1e-9 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x), ap = Math.pow(Math.cos(Math.PI * u / 2), 0.6); pts.push(`${f(xc + u * halfW)},${f(y0 - A * s * ap)}`); }
    return 'M' + pts.join(' L');
  }
  // split [t0,t1] into pieces following the layout segments; compressed gaps are drawn dashed
  function pieces(L, S, t0, t1) {
    const out = [];
    for (const g of L.segs) { const a0 = Math.max(t0, g.t0), b0 = Math.min(t1, g.t1); if (b0 > a0 + 1e-9) out.push({ t0: a0, t1: b0, dashed: g.compressed }); }
    return out;
  }
  /* opts: {lang, compress, revealAll, cursorT, showCursor, labels:{peDesc, feDesc, scale, envDesc}} */
  function build(S, sim, opts) {
    opts = opts || {};
    const L = layout(S, opts.compress), x = L.xOf, R = ROWS, W = L.width, H = HEIGHT, lang = opts.lang || 'pt';
    const loc = v => (v && typeof v === 'object') ? (v[lang] || v.pt || v.en || '') : (v == null ? '' : v);
    const lab = Object.assign({ peDesc: 'codificação por fase', feDesc: 'codificação por frequência', scale: '10 ms', envDesc: 'e^{−t/T2}' }, opts.labels || {});
    const out = [];
    out.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${f(W)} ${H}" width="${f(W)}" height="${H}" font-family="'IBM Plex Sans', 'Segoe UI', Helvetica, Arial, sans-serif" style="user-select:none">`);
    out.push('<defs>');
    for (const st of ['te', 'tr', 'esp']) out.push(`<marker id="ah_${st}" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 Z" fill="${C[st]}"/></marker>`);
    out.push(`<clipPath id="reveal"><rect id="revealRect" x="0" y="0" width="${opts.revealAll ? f(W) : 0}" height="${H}"/></clipPath></defs>`);
    out.push(`<rect width="${f(W)}" height="${H}" fill="#ffffff"/>`);
    // bands
    const bandX = i => x(S.bands[i].t), bandW = i => Math.min(x(S.bands[i].t1), L.xOfU(L.uEnd)) - bandX(i);
    S.bands.forEach((bd, i) => { if (i % 2 === 1) out.push(`<rect x="${f(bandX(i))}" y="12" width="${f(bandW(i))}" height="${H - 24}" fill="${C.band}"/>`); });
    out.push(`<rect id="trHighlight" x="${f(bandX(0))}" y="12" width="${f(bandW(0))}" height="${H - 24}" fill="${C.hl}" opacity="0"/>`);
    S.bands.forEach((bd, i) => { const xm = bandX(i) + bandW(i) / 2; out.push(`<text class="trh" id="trh${i}" x="${f(xm)}" y="17" text-anchor="middle" font-size="13" fill="${C.text}">${markup(loc(bd.label), 13)}</text>`); });
    // baselines and row labels
    for (const [key, y] of Object.entries(R)) {
      for (const pc of pieces(L, S, 0, S.tEnd)) out.push(`<line x1="${f(x(pc.t0))}" y1="${y}" x2="${f(x(pc.t1))}" y2="${y}" stroke="${C.base}" stroke-width="1.2"${pc.dashed ? ' stroke-dasharray="5 5"' : ''}/>`);
      if (key === 'gy' || key === 'gx') {
        const ax = key === 'gy' ? 'y' : 'x', sub = key === 'gy' ? 'pe' : 'fe';
        out.push(`<text x="${GUT - 12}" y="${y + 1}" text-anchor="end" font-size="18" font-weight="600" fill="${C.text}">G<tspan font-size="12" dy="3">${sub}</tspan><tspan dy="-3" font-size="15" font-weight="400"> = ∂B</tspan><tspan font-size="10" dy="3" font-weight="400">z</tspan><tspan dy="-3" font-size="15" font-weight="400">/∂${ax}</tspan></text>`);
        out.push(`<text x="${GUT - 12}" y="${y + 17}" text-anchor="end" font-size="11.5" fill="${C.text}">${key === 'gy' ? lab.peDesc : lab.feDesc}</text>`);
      } else out.push(`<text x="${GUT - 12}" y="${y + 6}" text-anchor="end" font-size="18" font-weight="600" fill="${C.text}">${key === 'rf' ? 'RF' : 'Signal'}</text>`);
    }
    // amplitude scale: the strongest x gradient (readout) is 24 px high
    const grads = S.events.filter(e => e.type === 'grad'), rfs = S.events.filter(e => e.type === 'rf'), adcs = S.events.filter(e => e.type === 'adc');
    const ampRef = Math.max(1e-9, ...grads.filter(g => g.axis === 'x').map(g => Math.abs(g.amp)), ...grads.map(g => Math.abs(g.amp)) .map(v => v * 0.6));
    const hOf = g => 24 * g.amp / ampRef;
    // RF
    for (const r of rfs) {
      const A = Math.max(12, Math.min(38, 20 + 18 * (Math.abs(r.flip) - 90) / 90)), halfW = 0.9 * r.dur * S_PX;
      out.push(`<path d="${sincPath(x(r.tc), R.rf, halfW, A)}" fill="none" stroke="${C.rf}" stroke-width="1.6"/>`);
      out.push(`<text x="${f(x(r.tc))}" y="${f(R.rf - A - (A > 30 ? 4 : 10))}" text-anchor="middle" font-size="14" fill="${C.text}">${f(r.flip)}°</text>`);
    }
    // gradients (ghost tables first, then lobes)
    for (const g of grads) {
      const y0 = g.axis === 'y' ? R.gy : R.gx, col = g.axis === 'y' ? C.gy : C.gx;
      if (g.def.ghost && g.ctx[g.def.ghost]) for (const v of g.ctx[g.def.ghost]) { if (v === 0 || Math.abs(v * (g.dur ? 1 / g.dur : 0) - g.amp) < 1e-9) continue; const hh = 24 * (v / g.dur) / ampRef; out.push(`<path d="${rect(x(g.t), x(g.t1), y0, hh)}" fill="none" stroke="${C.ghost}" stroke-width="1" stroke-dasharray="3 2"/>`); }
    }
    for (const g of grads) {
      const y0 = g.axis === 'y' ? R.gy : R.gx, col = g.axis === 'y' ? C.gy : C.gx, h = hOf(g);
      if (Math.abs(g.area) < 1e-9) out.push(`<line x1="${f(x(g.t))}" y1="${y0}" x2="${f(x(g.t1))}" y2="${y0}" stroke="${col}" stroke-width="3.2" stroke-linecap="round"/>`);
      else out.push(`<path d="${rect(x(g.t), x(g.t1), y0, h)}" fill="${col}" stroke="${col}" stroke-width="1"/>`);
      if (g.label != null && S.N <= 7) out.push(`<text x="${f(x(g.t1) + 4)}" y="${f(y0 - h + (h >= 0 ? -3 : 12))}" font-size="12" font-weight="600" fill="${col}">${esc(g.label)}</text>`);
    }
    // signal: optional T2 envelope of the largest possible signal, then the simulated signal in ADC windows
    const yS = v => R.sig - AMP_SIG * v / sim.Smax, T2 = sim.relax.T2;
    if (S.def.envelope && Number.isFinite(T2) && rfs.length) {
      const r0 = rfs[0], tA0 = S.shots[0].tActive; let env = '', last = null;
      for (let t = r0.t1; t <= tA0; t += 0.5) { let tp = 0; for (const r of rfs) if (r !== r0 && r.shot === 0) tp += Math.max(0, Math.min(t, r.t1) - r.t); last = sim.Smax * Math.exp(-(t - r0.t1 - tp) / T2); env += (env ? 'L' : 'M') + f(x(t)) + ',' + f(yS(last)) + ' '; }
      out.push(`<path d="${env}" fill="none" stroke="${C.env}" stroke-width="1" stroke-dasharray="3 3"/>`);
      out.push(`<text x="${f(x(tA0) + 4)}" y="${f(yS(last) + 4)}" font-size="11" fill="${C.env}">${lab.envDesc}</text>`);
    }
    for (const sh of S.shots) {                 // net transverse magnetization |S(t)| through the whole active part
      const i0 = SeqIndex(sim, sh.T0), i1 = SeqIndex(sim, Math.min(S.tEnd, sh.T0 + S.TR)); let mag = '';
      for (let i = i0; i <= i1; i++) mag += (i === i0 ? 'M' : 'L') + f(x(sim.times[i])) + ',' + f(yS(Math.hypot(sim.sRe[i], sim.sIm[i]))) + ' ';
      out.push(`<path d="${mag}" fill="none" stroke="${C.sigGhost}" stroke-width="0.9" stroke-dasharray="2 2"/>`);
      out.push(`<g clip-path="url(#reveal)"><path d="${mag}" fill="none" stroke="${C.env}" stroke-width="1.1"/></g>`);
    }
    for (const a of adcs) {
      const i0 = SeqIndex(sim, a.t - 0.8), i1 = SeqIndex(sim, a.t1 + 0.8);
      let line = '', env = `M${f(x(sim.times[i0]))},${R.sig} `;
      for (let i = i0; i <= i1; i++) { const xx = f(x(sim.times[i])); line += (i === i0 ? 'M' : 'L') + xx + ',' + f(yS(sim.sRe[i])) + ' '; env += 'L' + xx + ',' + f(yS(Math.hypot(sim.sRe[i], sim.sIm[i]))) + ' '; }
      env += `L${f(x(sim.times[i1]))},${R.sig} Z`;
      out.push(`<path d="${line}" fill="none" stroke="${C.sigGhost}" stroke-width="1.2"/>`);
      out.push(`<g clip-path="url(#reveal)"><path d="${env}" fill="${C.sigFill}" stroke="none"/><path d="${line}" fill="none" stroke="${C.sig}" stroke-width="1.7"/>`);
      for (const s of sim.samples) if (s.adc === a) out.push(`<circle cx="${f(x(s.t))}" cy="${f(yS(s.re))}" r="3.4" fill="${phaseColor(Math.atan2(s.im, s.re))}" stroke="#333" stroke-width="0.8"/>`);
      out.push('</g>');
      if (a.def.label) out.push(`<text x="${f(x(a.tc) + 8)}" y="${R.sig - AMP_SIG - 4}" font-size="13" fill="${C.sig}">${esc(loc(a.def.label))}</text>`);
    }
    // marks (dashed verticals) and arrows (shot 0)
    for (const m of S.marks) {
      const col = C[m.style] || C.te, yb = Y_LEVEL[Math.min(1, m.level || 0)];
      out.push(`<line x1="${f(x(m.t))}" y1="${R.rf - 24}" x2="${f(x(m.t))}" y2="${yb}" stroke="${col}" stroke-width="1" stroke-dasharray="4 3"/>`);
      if (m.label) out.push(`<text x="${f(x(m.t) + 4)}" y="${R.rf + 18}" font-size="13" font-weight="600" fill="${col}">${markup(loc(m.label), 13)}</text>`);
    }
    for (const a of S.arrows) {
      const col = C[a.style] || C.te, y = Y_LEVEL[Math.min(1, a.level || 0)], ps = pieces(L, S, a.t, a.t1);
      ps.forEach((pc, i) => out.push(`<line x1="${f(x(pc.t0))}" y1="${y}" x2="${f(x(pc.t1))}" y2="${y}" stroke="${col}" stroke-width="4"${pc.dashed ? ' stroke-dasharray="6 6"' : ''}${i === 0 ? ` marker-start="url(#ah_${a.style || 'te'})"` : ''}${i === ps.length - 1 ? ` marker-end="url(#ah_${a.style || 'te'})"` : ''}/>`));
      out.push(`<text x="${f((x(a.t) + x(a.t1)) / 2)}" y="${y + 22}" text-anchor="middle" font-size="20" font-weight="700" fill="${col}">${markup(loc(a.label), 20)}${a.def.sub ? `<tspan font-size="11" font-weight="400" dx="6">${esc(loc(a.def.sub))}</tspan>` : ''}</text>`);
    }
    // scale bar, cursor
    const sx1 = W - RIGHT - 10 * S_PX, sx2 = W - RIGHT, ysb = Y_LEVEL[1];
    out.push(`<line x1="${f(sx1)}" y1="${ysb}" x2="${f(sx2)}" y2="${ysb}" stroke="${C.text}" stroke-width="1.5"/><line x1="${f(sx1)}" y1="${ysb - 4}" x2="${f(sx1)}" y2="${ysb + 4}" stroke="${C.text}" stroke-width="1.5"/><line x1="${f(sx2)}" y1="${ysb - 4}" x2="${f(sx2)}" y2="${ysb + 4}" stroke="${C.text}" stroke-width="1.5"/>`);
    out.push(`<text x="${f((sx1 + sx2) / 2)}" y="${ysb + 16}" text-anchor="middle" font-size="11" fill="${C.text}">${lab.scale}</text>`);
    const cx = opts.cursorT == null ? GUT : x(opts.cursorT);
    out.push(`<g id="cursor" transform="translate(${f(cx)},0)" style="${opts.cursorT == null && !opts.showCursor ? 'display:none' : ''}"><polygon points="-6,36 6,36 0,44" fill="${C.cursor}"/><line x1="0" y1="42" x2="0" y2="${H - 8}" stroke="${C.cursor}" stroke-width="1.6"/></g>`);
    out.push('</svg>');
    return { svg: out.join('\n'), L, bandX, bandW, colors: C };
  }
  function SeqIndex(sim, t) { const a = sim.times; let lo = 0, hi = a.length - 1; if (t <= a[0]) return 0; if (t >= a[hi]) return hi; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (a[m] <= t) lo = m; else hi = m; } return (t - a[lo] < a[hi] - t) ? lo : hi; }
  return { build, layout, markup, phaseColor, ROWS, COLORS: C };
});
