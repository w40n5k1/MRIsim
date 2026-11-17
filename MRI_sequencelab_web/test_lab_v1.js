// By Bruno Mota - METAbio
// Checks: (1) for both sequence files, every acquired sample equals DFT(M0) at its (kx,ky) times the T2 decay since the
// center of the excitation pulse, scaled by the longitudinal magnetization available at that instant (pulses are instantaneous);
// (2) for a uniform phantom, the k = 0 sample of the spin echo equals the textbook steady-state formula.   node test_lab_v1.js
const E = require('./seq_engine_v1.js'), D = require('./seq_draw_v1.js'), fs = require('fs');
const TIS = JSON.parse(fs.readFileSync('tissues.json', 'utf8')).tissues;
for (const file of ['seq_spin_echo.json', 'seq_fast_spin_echo.json']) for (const N of [3, 5, 9]) {
  const def = JSON.parse(fs.readFileSync(file, 'utf8')), P = E.paramDefaults(def), S = E.compile(def, P, 'pt', N);
  const M0 = E.renderPhantom(JSON.parse(fs.readFileSync('phantom_square.json', 'utf8')), N, TIS).M0, relax = { T1: def.relax.T1, T2: def.relax.T2 }, sim = E.simulate(S, M0, relax);
  const rfs = S.events.filter(e => e.type === 'rf'); let worst = 0;
  for (const s of sim.samples) {
    let re = 0, im = 0; for (let p = 0; p < N * N; p++) { const q = E.pixelXY(p, N), ph = -2 * Math.PI * (s.kx * q.x + s.ky * q.y) / N; re += M0[p] * Math.cos(ph); im += M0[p] * Math.sin(ph); }
    const first = rfs.find(r => r.shot === s.shot);
    const i0 = E.indexAt(sim, first.t) - 1;   // grid point before the pulse starts: physical state, relaxed up to the pulse center below
    let mzAvail = 0; for (let p = 0; p < N * N; p++) mzAvail += sim.Mz[i0 * N * N + p];
    mzAvail = sim.Smax + (mzAvail - sim.Smax) * Math.exp(-(first.tc - sim.times[i0]) / relax.T1);   // longitudinal magnetization available at the excitation
    const dec = Math.exp(-(s.t - first.tc) / relax.T2) * mzAvail / sim.Smax;
    worst = Math.max(worst, Math.hypot(s.re - re * dec, s.im - im * dec));
  }
  console.log(file, 'N', N, 'samples', sim.samples.length, 'k lines', [...new Set(sim.samples.map(s => s.ky))].join(','), 'worst error vs DFT x decay', worst.toExponential(2));
  if (N === 5) fs.writeFileSync(file.replace('.json', '.svg'), D.build(S, sim, { revealAll: true, lang: 'pt' }).svg);
}

// (2) textbook spin-echo formula, uniform phantom, from the second repetition on
{
  const def = JSON.parse(fs.readFileSync('seq_spin_echo.json', 'utf8')), N = 5; let worst = 0;
  for (const key of ['gm', 'wm', 'csf']) for (const [TE, TR] of [[32, 2000], [100, 2000], [22, 100], [200, 300]]) {
    const tis = TIS[key], S = E.compile(def, Object.assign(E.paramDefaults(def), { TE, TR }), 'pt', N);
    const sim = E.simulate(S, new Array(N * N).fill(tis.pd), { T1: tis.T1, T2: tis.T2 });
    const s00 = sim.samples.find(s => s.kx === 0 && s.ky === 0);
    const exact = tis.pd * N * N * (1 - 2 * Math.exp(-(TR - TE / 2) / tis.T1) + Math.exp(-TR / tis.T1)) * Math.exp(-TE / tis.T2);
    worst = Math.max(worst, Math.abs(Math.hypot(s00.re, s00.im) - exact) / exact);
  }
  console.log('spin echo vs textbook steady-state formula: worst relative error', worst.toExponential(2));
  if (worst > 1e-5) throw new Error('spin-echo formula check failed');
}

// (3) phantom files: the painted face renders at its own N and resamples to others; the brain's lesion survives at N = 5
{
  const face = JSON.parse(fs.readFileSync('phantom_face.json', 'utf8')), brain = JSON.parse(fs.readFileSync('phantom_brain.json', 'utf8'));
  const f15 = E.renderPhantom(face, 15, TIS), f9 = E.renderPhantom(face, 9, TIS), b5 = E.renderPhantom(brain, 5, TIS);
  if (f15.label.length !== 225 || f9.label.length !== 81 || !b5.label.includes('lesion') || !f9.label.includes('lesion')) throw new Error('phantom file check failed');
  console.log('phantom files: face 15 ->', new Set(f15.label).size, 'tissues; face resampled to 9 ok; brain at 5 has its lesion');
}
