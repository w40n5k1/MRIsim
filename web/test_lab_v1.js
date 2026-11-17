// Checks: for both sequence files, every acquired sample equals DFT(M0) at its (kx,ky) times the T2 decay over the
// non-pulse time since the excitation, scaled by the longitudinal magnetization available at that excitation.   node test_lab_v1.js
const E = require('./seq_engine_v1.js'), D = require('./seq_draw_v1.js'), fs = require('fs');
const TIS = JSON.parse(fs.readFileSync('tissues.json', 'utf8')).tissues;
for (const file of ['seq_spin_echo.json', 'seq_fast_spin_echo.json']) for (const N of [3, 5, 9]) {
  const def = JSON.parse(fs.readFileSync(file, 'utf8')), P = E.paramDefaults(def), S = E.compile(def, P, 'pt', N);
  const M0 = E.phantom('square', N, TIS).M0, relax = { T1: def.relax.T1, T2: def.relax.T2 }, sim = E.simulate(S, M0, relax);
  const rfs = S.events.filter(e => e.type === 'rf'); let worst = 0;
  for (const s of sim.samples) {
    let re = 0, im = 0; for (let p = 0; p < N * N; p++) { const q = E.pixelXY(p, N), ph = -2 * Math.PI * (s.kx * q.x + s.ky * q.y) / N; re += M0[p] * Math.cos(ph); im += M0[p] * Math.sin(ph); }
    const first = rfs.find(r => r.shot === s.shot); let tp = 0; for (const r of rfs) if (r !== first && r.shot === s.shot && r.t1 <= s.t) tp += r.dur;
    const i0 = E.indexAt(sim, first.t); let mzAvail = 0; for (let p = 0; p < N * N; p++) mzAvail += sim.Mz[i0 * N * N + p];   // longitudinal magnetization available at the excitation
    const dec = Math.exp(-(s.t - first.t1 - tp) / relax.T2) * mzAvail / sim.Smax;
    worst = Math.max(worst, Math.hypot(s.re - re * dec, s.im - im * dec));
  }
  console.log(file, 'N', N, 'samples', sim.samples.length, 'k lines', [...new Set(sim.samples.map(s => s.ky))].join(','), 'worst error vs DFT x decay', worst.toExponential(2));
  if (N === 5) fs.writeFileSync(file.replace('.json', '.svg'), D.build(S, sim, { revealAll: true, lang: 'pt' }).svg);
}
