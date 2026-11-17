// By Bruno Mota - METAbio
/* seq_engine_v1.js — generic engine for didactic 2D imaging sequences described as text (JSON).
 *
 * A sequence file declares parameters, derived variables and a list of events per shot (repetition):
 *   rf    {t, dur, flip (deg), axis 'x'|'y', refocus?}   hard pulse; the angle accrues linearly over dur (visible rotation);
 *                                                        relaxation is suspended during the pulse; refocus: k -> -k at the center,
 *                                                        otherwise (excitation) the k label resets to 0 at the center
 *   grad  {axis 'x'|'y', t, dur, area}                   constant lobe; area in dk = 1/FOV units (k moved by the lobe)
 *   adc   {t, dur, n}                                    readout window; n samples at the centers of n equal sub-intervals
 *   spoil {t}                                            ideal spoiler: transverse magnetization zeroed at t
 *   repeat {count, var, vars?, events}                   loop; the loop variable and local vars are visible inside
 * Numbers may be expressions (strings) in the parameters, the table values, the shot index n, loop variables and
 * the derived vars (evaluated in order). Expressions are plain JavaScript arithmetic evaluated with Function();
 * a sequence file is code you choose to run, like a notebook cell.
 *
 * Physics (same didactic model as the hand-written programs): N x N isochromats (N odd) at x,y in {-(N-1)/2..(N-1)/2}, rotating frame
 * (precession only under gradients: m -> m exp(-i 2 pi (dkx x + dky y)/5)), hard pulses, exponential relaxation,
 * signal S = sum of (Mx + i My) over the pixels. Samples are the forward DFT of the pixel magnetizations.
 * Units: ms, dk.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.SeqEngine = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  // N (odd) pixels per side; pixel p = row*N + col, row 0 at the top (y = +HALF), x = col - HALF
  function pixelXY(p, N) { const H = (N - 1) / 2; return { x: (p % N) - H, y: H - Math.floor(p / N) }; }
  const PHANTOM_NAMES = ['squareAir', 'letterFAir', 'uniform', 'square', 'letterF', 'ramp', 'brain'];
  // Phantoms are tissue maps: every pixel has a tissue label (from a tissue dictionary {key: {pd, T1, T2, color, name}})
  // and an optional proton-density factor. Returns {label[], pdf[], M0[], T1[], T2[], tissues}, any odd N.
  function phantom(name, N, tissues) {
    const H = (N - 1) / 2, NP = N * N, at = (x, y) => (H - y) * N + (x + H);
    const label = new Array(NP).fill(name.endsWith('Air') ? 'air' : 'wm'), pdf = new Array(NP).fill(1);
    if (name.endsWith('Air')) name = name.slice(0, -3);                 // filled/empty versions: figure on air
    if (name === 'uniform') label.fill('gm');
    else if (name === 'square') { const h = Math.max(1, Math.floor(0.3 * N)); for (let y = -h; y <= h; y++) for (let x = -h; x <= h; x++) label[at(x, y)] = 'gm'; }
    else if (name === 'letterF') {
      const t = Math.max(1, Math.round(N / 5)), x0 = -H, wTop = N - 1, wMid = Math.round(0.6 * (N - 1)), yMid = H - Math.round(0.4 * (N - 1));
      for (let y = -H; y <= H; y++) for (let k = 0; k < t; k++) label[at(x0 + k, y)] = 'gm';
      for (let k = 0; k < t; k++) for (let x = x0; x <= x0 + wTop; x++) label[at(x, H - k)] = 'gm';
      for (let k = 0; k < t; k++) for (let x = x0; x <= x0 + wMid; x++) label[at(x, yMid - k)] = 'gm';
    } else if (name === 'ramp') { label.fill('gm'); for (let p = 0; p < NP; p++) pdf[p] = 0.2 + 0.8 * (p % N) / Math.max(1, N - 1); }
    else if (name === 'brain') {
      // concentric head: air, scalp fat, skull, CSF, gray matter, white matter, CSF ventricle at the center, one lesion
      const R = 0.46 * N + 0.1;
      for (let y = -H; y <= H; y++) for (let x = -H; x <= H; x++) {
        const r = Math.hypot(x, y) / R; let k;
        if (r > 1.0) k = 'air'; else if (r > 0.93) k = 'fat'; else if (r > 0.86) k = 'bone'; else if (r > 0.76) k = 'csf'; else if (r > 0.6) k = 'gm'; else if (r < 0.2) k = 'csf'; else k = 'wm';
        if (k === 'wm' && x > 0 && y > 0 && Math.hypot(x - 0.36 * R, y - 0.28 * R) <= Math.max(0.5, 0.11 * R)) k = 'lesion';
        label[at(x, y)] = k;
      }
    } else throw new Error('unknown phantom ' + name);
    const M0 = new Array(NP), T1 = new Array(NP), T2 = new Array(NP);
    for (let p = 0; p < NP; p++) { const t = tissues[label[p]]; if (!t) throw new Error('unknown tissue ' + label[p]); M0[p] = t.pd * pdf[p]; T1[p] = t.T1; T2[p] = t.T2; }
    return { label, pdf, M0, T1, T2, tissues };
  }
  // helpers available to sequence expressions
  function range(a, b, step) { step = step || (b >= a ? 1 : -1); const out = []; for (let v = a; step > 0 ? v <= b + 1e-9 : v >= b - 1e-9; v += step) out.push(v); return out; }
  function centric(H) { const out = [0]; for (let k = 1; k <= H; k++) out.push(k, -k); return out; }
  function indexOf(arr, v) { return arr.indexOf(v); }
  function rev(arr) { return arr.slice().reverse(); }

  /* ---------- expressions ---------- */
  const fnCache = new Map();
  function evalExpr(expr, ctx) {
    if (typeof expr === 'number') return expr;
    if (typeof expr === 'boolean') return expr ? 1 : 0;
    if (expr == null) return 0;
    if (typeof expr !== 'string') throw new Error('bad expression ' + JSON.stringify(expr));
    const keys = Object.keys(ctx), sig = keys.join(',') + '|' + expr;
    let fn = fnCache.get(sig);
    if (!fn) { fn = new Function(...keys, 'with (Math) { return (' + expr + '); }'); fnCache.set(sig, fn); }
    const v = fn(...keys.map(k => ctx[k]));
    if (typeof v === 'boolean') return v ? 1 : 0;
    if (typeof v !== 'number' || Number.isNaN(v)) throw new Error('expression "' + expr + '" gave ' + v);
    return v;
  }
  function evalVars(vars, ctx) { if (!vars) return; for (const [k, e] of Object.entries(vars)) ctx[k] = evalExpr(e, ctx); }
  // text templates: ${expr} substituted; |v| formatting helpers in ctx
  function fmtText(tpl, ctx) {
    return String(tpl).replace(/\$\{([^}]+)\}/g, (_, e) => {
      const v = evalExpr(e, ctx); const r = Math.round(v * 100) / 100;
      return (r > 0 && /^\+/.test(e) ? '+' : '') + (r < 0 ? '−' + (-r) : String(r));
    });
  }

  /* ---------- compile: sequence definition + parameter values -> absolute schedule ---------- */
  function paramDefaults(def) {
    const P = {};
    for (const p of def.params || []) P[p.id] = p.type === 'select' ? p.default : (p.type === 'checkbox' ? (p.default ? 1 : 0) : p.default);
    return P;
  }
  function evalTable(v, ctx) { return typeof v === 'string' ? (new Function(...Object.keys(ctx), 'with (Math) { return (' + v + '); }'))(...Object.values(ctx)) : v; }
  function baseContext(def, P, N) {
    const ctx = { N, HALF: (N - 1) / 2, PI: Math.PI, range, centric, indexOf, rev };
    for (const [k, v] of Object.entries(def.constants || {})) ctx[k] = v;
    for (const [k, v] of Object.entries(def.tables || {})) ctx[k] = evalTable(v, ctx);
    for (const p of def.params || []) {
      if (p.type === 'select') {                         // a select contributes its chosen option's values (arrays or expressions)
        const opt = (p.options || []).find(o => o.value === P[p.id]) || p.options[0];
        if (opt && opt.values) for (const [k, v] of Object.entries(opt.values)) ctx[k] = evalTable(v, ctx);
        ctx[p.id] = (p.options || []).indexOf(opt);
      } else ctx[p.id] = P[p.id];
    }
    evalVars(def.vars, ctx);
    return ctx;
  }
  function expandEvents(list, ctx, out, T0) {
    for (const ev of list || []) {
      if (ev.if != null && !evalExpr(ev.if, ctx)) continue;
      if (ev.type === 'repeat') {
        const count = Math.round(evalExpr(ev.count, ctx));
        for (let i = 0; i < count; i++) { const c = Object.assign({}, ctx); c[ev.var || 'i'] = i; evalVars(ev.vars, c); expandEvents(ev.events, c, out, T0); }
        continue;
      }
      const e = { type: ev.type, def: ev, ctx };
      e.t = T0 + evalExpr(ev.t, ctx); e.dur = ev.dur != null ? evalExpr(ev.dur, ctx) : 0; e.t1 = e.t + e.dur;
      if (ev.type === 'rf') { e.flip = evalExpr(ev.flip, ctx); e.axis = ev.axis || 'x'; e.refocus = !!ev.refocus; e.tc = e.t + e.dur / 2; }
      else if (ev.type === 'grad') { e.axis = ev.axis; e.area = evalExpr(ev.area, ctx); e.amp = e.dur > 0 ? e.area / e.dur : 0; }
      else if (ev.type === 'adc') { e.n = Math.round(evalExpr(ev.n, ctx)); e.samples = []; for (let j = 0; j < e.n; j++) e.samples.push(e.t + (j + 0.5) * e.dur / e.n); e.tc = e.t + e.dur / 2; }
      else if (ev.type === 'spoil') { /* t only */ }
      else if (ev.type === 'band' || ev.type === 'mark' || ev.type === 'arrow') { if (ev.to != null) e.t1 = T0 + evalExpr(ev.to, ctx); e.label = ev.label; e.style = ev.style; e.level = ev.level || 0; }
      else throw new Error('unknown event type ' + ev.type);
      out.push(e);
    }
  }
  function compile(def, P, lang, N) {
    lang = lang || 'pt'; N = N || 5;
    const loc = v => (v && typeof v === 'object') ? (v[lang] || v.pt || v.en || '') : (v == null ? '' : String(v));
    const ctx0 = baseContext(def, P, N);
    const nShots = Math.max(1, Math.round(evalExpr(def.shots == null ? 1 : def.shots, ctx0)));
    const TR = evalExpr(def.TR, ctx0);
    const shots = [], events = [], bands = [], marks = [], arrows = [], adcs = [];
    for (let n = 0; n < nShots; n++) {
      const ctx = Object.assign({}, ctx0, { n }); evalVars(def.shotVars, ctx);
      const T0 = n * TR, out = [];
      expandEvents(def.events, ctx, out, T0);
      let tActive = 0;
      for (const e of out) {
        if (e.def.desc != null) e.desc = fmtText(loc(e.def.desc), e.ctx);
        if (e.type === 'grad' && e.def.label != null) e.label = fmtText(loc(e.def.label), e.ctx);
        if (e.type === 'band') bands.push(Object.assign({ shot: n }, e));
        else if (e.type === 'mark') { if (n === 0) marks.push(e); }
        else if (e.type === 'arrow') { if (n === 0) arrows.push(e); }
        else { e.shot = n; events.push(e); if (e.type !== 'spoil') tActive = Math.max(tActive, e.t1 - T0); if (e.type === 'adc') adcs.push(e); }
      }
      shots.push({ n, T0, ctx, tActive, label: def.shotLabel ? fmtText(loc(def.shotLabel), ctx) : '' });
    }
    const tActive = Math.max(...shots.map(s => s.tActive));
    if (TR < tActive) throw new Error('TR shorter than the active part (' + tActive.toFixed(1) + ' ms)');
    // bands: default one band per shot from its first rf center
    if (!bands.length) for (const s of shots) { const rf = events.find(e => e.shot === s.n && e.type === 'rf'); bands.push({ shot: s.n, t: rf ? rf.tc : s.T0, label: s.label }); }
    bands.sort((a, b) => a.t - b.t);
    for (let i = 0; i < bands.length; i++) { bands[i].t1 = i + 1 < bands.length ? bands[i + 1].t : nShots * TR; if (bands[i].label == null) bands[i].label = shots[bands[i].shot].label; else bands[i].label = fmtText(loc(bands[i].label), bands[i].ctx); }
    const samples = []; for (const a of adcs) a.samples.forEach((t, j) => samples.push({ t, shot: a.shot, adc: a, j }));
    samples.sort((a, b) => a.t - b.t);
    // hard-pulse model: RF must not overlap gradients or readouts
    for (const r of events) if (r.type === 'rf') for (const e of events) if (e !== r && e.type !== 'spoil' && e.shot === r.shot && e.t < r.t1 - 1e-9 && e.t1 > r.t + 1e-9) throw new Error('RF pulse at ' + r.t.toFixed(1) + ' ms overlaps a ' + e.type + ' event: increase the timing parameters');
    for (const e of events) if (e.t < 0 || e.t1 > S_tEnd(nShots, TR) + 1e-9) throw new Error('event outside the repetition');
    return { def, P, N, NPIX: N * N, ctx0, nShots, TR, tActive, tEnd: nShots * TR, shots, events, bands, marks, arrows, adcs, samples, lang };
  }

  function S_tEnd(n, TR) { return n * TR; }

  /* ---------- simulate ---------- */
  function overlap(a, b, iv) { return Math.max(0, Math.min(b, iv[1]) - Math.max(a, iv[0])); }
  function rotY(mx, my, mz, th) { const c = Math.cos(th), s = Math.sin(th); for (let p = 0; p < mx.length; p++) { const x = mx[p], z = mz[p]; mx[p] = x * c + z * s; mz[p] = -x * s + z * c; } }
  function rotX(mx, my, mz, th) { const c = Math.cos(th), s = Math.sin(th); for (let p = 0; p < mx.length; p++) { const y = my[p], z = mz[p]; my[p] = y * c - z * s; mz[p] = y * s + z * c; } }

  /* relax: {T1, T2} numbers or per-pixel arrays (Infinity disables), dt (fine step, ms), nGap (points per idle gap) */
  function simulate(S, M0, relax) {
    const N = S.N, NPIX = S.NPIX, dt = relax.dt || 0.1, nGap = relax.nGap || relax.nTail || 80;
    if (M0.length !== NPIX) throw new Error('phantom size ' + M0.length + ' does not match N = ' + N);
    const arr = v => Array.isArray(v) ? v : new Array(NPIX).fill(v);
    const T1 = arr(relax.T1), T2 = arr(relax.T2);
    // time grid: fine (dt) inside and around events, coarse (nGap points) across idle gaps, plus the exact sample instants
    const MARGIN = 1.0, grid = [], busy = [];
    for (const e of S.events) if (e.type !== 'spoil') busy.push([Math.max(0, e.t - MARGIN), Math.min(S.tEnd, e.t1 + MARGIN)]);
    busy.sort((a, b) => a[0] - b[0]);
    const merged = []; for (const iv of busy) { const l = merged[merged.length - 1]; if (l && iv[0] <= l[1] + 1e-9) l[1] = Math.max(l[1], iv[1]); else merged.push([iv[0], iv[1]]); }
    let cursor = 0;
    const coarse = (a, b) => { const n = Math.max(1, Math.min(nGap, Math.round((b - a) / dt))); for (let i = 1; i < n; i++) grid.push(a + i * (b - a) / n); };
    for (const [a, b] of merged) { if (a > cursor) { grid.push(cursor); coarse(cursor, a); } const n = Math.round((b - a) / dt); for (let i = 0; i <= n; i++) grid.push(a + i * dt); cursor = b; }
    if (cursor < S.tEnd) { grid.push(cursor); coarse(cursor, S.tEnd); }
    grid.push(S.tEnd); for (const smp of S.samples) grid.push(smp.t); for (const sh of S.shots) grid.push(sh.T0);
    grid.sort((a, b) => a - b);
    const times = [], shotOf = [];
    for (const t of grid) { if (times.length && t - times[times.length - 1] < 1e-7) continue; times.push(t); shotOf.push(Math.min(S.nShots - 1, Math.floor(t / S.TR))); }
    const Nt = times.length;
    const Mx = new Float32Array(NPIX * Nt), My = new Float32Array(NPIX * Nt), Mz = new Float32Array(NPIX * Nt);
    const sRe = new Float32Array(Nt), sIm = new Float32Array(Nt), kxM = new Float32Array(Nt), kyM = new Float32Array(Nt);
    const mx = new Float64Array(NPIX), my = new Float64Array(NPIX), mz = Float64Array.from(M0);
    const X = new Float64Array(NPIX), Y = new Float64Array(NPIX);
    for (let p = 0; p < NPIX; p++) { const q = pixelXY(p, N); X[p] = q.x; Y[p] = q.y; }
    const rfs = S.events.filter(e => e.type === 'rf'), grads = S.events.filter(e => e.type === 'grad'), spoils = S.events.filter(e => e.type === 'spoil');
    // sample lookup by grid index (the sample instants are grid points)
    const sampleIdx = new Map(); let si = 0;
    for (const smp of S.samples) { while (si < Nt - 1 && times[si] < smp.t - 1e-7) si++; sampleIdx.set(si, (sampleIdx.get(si) || []).concat([smp])); }
    let kx = 0, ky = 0; const samples = [];
    for (let i = 0; i < Nt; i++) {
      if (i > 0) {
        const a = times[i - 1], bb = times[i], d = bb - a;
        let oP = 0; for (const r of rfs) oP += overlap(a, bb, [r.t, r.t1]);
        const dR = d - oP;
        if (dR > 0) for (let p = 0; p < NPIX; p++) {
          const e1 = Number.isFinite(T1[p]) ? Math.exp(-dR / T1[p]) : 1, e2 = Number.isFinite(T2[p]) ? Math.exp(-dR / T2[p]) : 1;
          mx[p] *= e2; my[p] *= e2; mz[p] = M0[p] + (mz[p] - M0[p]) * e1;
        }
        for (const r of rfs) {
          const o = overlap(a, bb, [r.t, r.t1]);
          if (o > 0) { const th = r.flip * Math.PI / 180 * o / r.dur; if (r.axis === 'y') rotY(mx, my, mz, th); else rotX(mx, my, mz, th); }
          if (a < r.tc && bb >= r.tc) { if (r.refocus) { kx = -kx; ky = -ky; } else { kx = 0; ky = 0; } }   // refocusing: k -> -k; excitation: fresh M_perp at k = 0
        }
        let dkx = 0, dky = 0;
        for (const g of grads) { const o = overlap(a, bb, [g.t, g.t1]); if (o > 0) { if (g.axis === 'x') dkx += g.amp * o; else dky += g.amp * o; } }
        if (dkx !== 0 || dky !== 0) {
          kx += dkx; ky += dky;
          for (let p = 0; p < NPIX; p++) { const ph = 2 * Math.PI * (dkx * X[p] + dky * Y[p]) / N, c = Math.cos(ph), s = Math.sin(ph); const x = mx[p], y = my[p]; mx[p] = x * c + y * s; my[p] = -x * s + y * c; }
        }
        for (const sp of spoils) if (a < sp.t && bb >= sp.t) { mx.fill(0); my.fill(0); kx = 0; ky = 0; }
      }
      let re = 0, im = 0;
      for (let p = 0; p < NPIX; p++) { Mx[i * NPIX + p] = mx[p]; My[i * NPIX + p] = my[p]; Mz[i * NPIX + p] = mz[p]; re += mx[p]; im += my[p]; }
      sRe[i] = re; sIm[i] = im; kxM[i] = kx; kyM[i] = ky;
      const sl = sampleIdx.get(i);
      if (sl) for (const smp of sl) samples.push({ idx: i, t: times[i], shot: smp.shot, adc: smp.adc, j: smp.j, kx: Math.round(kx), ky: Math.round(ky), re, im });
    }
    const Smax = M0.reduce((s, v) => s + v, 0);
    return { S, N, NPIX, M0, relax, times, shotOf, Mx, My, Mz, sRe, sIm, kxM, kyM, samples, Smax };
  }
  function indexAt(sim, t) {
    const a = sim.times; let lo = 0, hi = a.length - 1;
    if (t <= a[0]) return 0; if (t >= a[hi]) return hi;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (a[m] <= t) lo = m; else hi = m; }
    return (t - a[lo] < a[hi] - t) ? lo : hi;
  }
  // which event is active at t (for the status line): rf > adc > grad > spoil-tail > free
  function segmentAt(S, t) {
    const at = S.events.filter(e => e.type !== 'spoil' && t >= e.t && t <= e.t1);
    const rf = at.find(e => e.type === 'rf'); if (rf) return { kind: 'rf', e: rf };
    const adc = at.find(e => e.type === 'adc'); if (adc) return { kind: 'adc', e: adc, grads: at.filter(e => e.type === 'grad') };
    const gr = at.filter(e => e.type === 'grad'); if (gr.length) return { kind: 'grad', grads: gr };
    const shot = S.shots[Math.min(S.nShots - 1, Math.max(0, Math.floor(t / S.TR)))];
    if (t - shot.T0 > shot.tActive + 1e-9) return { kind: 'tail' };
    const first = S.events.filter(e => e.shot === shot.n).reduce((m, e) => Math.min(m, e.t), Infinity);
    return { kind: t < first ? 'rest' : 'free' };
  }
  function eventTimes(S) {       // for stepping
    const ts = new Set([0, S.tEnd]);
    for (const e of S.events) { ts.add(e.t); if (e.dur > 0) ts.add(e.t1); if (e.tc != null) ts.add(e.tc); }
    for (const s of S.shots) ts.add(s.T0 + s.tActive);
    return [...ts].sort((a, b) => a - b);
  }
  return { PHANTOM_NAMES, phantom, pixelXY, range, centric, indexOf, rev, evalExpr, fmtText, paramDefaults, compile, simulate, indexAt, segmentAt, eventTimes };
});
