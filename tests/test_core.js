// Unit tests for src/core.js. Run: node tests/test_core.js
'use strict';
const assert = require('assert');
const C = require('../src/core.js');

let passed = 0;
function test(name, fn) {
  try { fn(); passed++; console.log('ok   ' + name); }
  catch (e) { console.log('FAIL ' + name + '\n     ' + (e && e.stack ? e.stack.split('\n').slice(0, 3).join('\n     ') : e)); process.exitCode = 1; }
}
const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, (msg || '') + ` ${a} vs ${b} (tol ${tol})`);

test('channel names normalize to classic 10-20', () => {
  const cases = { 'EEG FP1-REF': 'Fp1', 'T7': 'T3', 'p8': 'T6', 'EEG EKG1-REF': 'EKG', 'ECG EKG-REF': 'EKG',
    'Fz': 'Fz', 'C3-LE': 'C3', 'EEG T8-Ref': 'T4', 'O2-A2': 'O2', 'SQ D-C': 'SQ D-C', 'CZ': 'Cz' };
  for (const [raw, want] of Object.entries(cases)) assert.strictEqual(C.normalizeName(raw), want, raw);
});

test('int16 encode/decode round trip stays within half a quantization step', () => {
  const n = 2400;
  const rows = [0, 1, 2].map(k => Float64Array.from({ length: n }, (_, i) => 80 * Math.sin(i / (7 + k)) + 3000));
  const enc = C.encodeClip(rows);
  const dec = C.decodeClip(enc.data, 3, n, enc.scale);
  rows.forEach((r, c) => {
    let mean = 0; for (const v of r) mean += v; mean /= n;
    for (let i = 0; i < n; i += 37) close(dec[c][i], r[i] - mean, enc.scale[c] * 0.51 + 1e-6, 'row ' + c);
  });
  assert.throws(() => C.decodeClip(enc.data, 4, n, enc.scale.concat([1])), /expected/);
});

test('double banana and transverse rows are first minus second electrode', () => {
  const n = 10;
  const scalp = new Map(C.SCALP.map((name, k) => [name, Float32Array.from({ length: n }, () => k)]));
  const rows = C.deriveRows('banana', scalp, []);
  assert.strictEqual(rows.length, 18);
  const r = rows.find(x => x.label === 'Fp1-F7');
  assert.strictEqual(r.values[3], C.SCALP.indexOf('Fp1') - C.SCALP.indexOf('F7'));
  assert.deepStrictEqual([...new Set(rows.map(x => x.group))], [0, 1, 2, 3, 4]);
  const tr = C.deriveRows('transverse', scalp, []);
  assert.strictEqual(tr.length, 18);
  assert.strictEqual(tr[0].label, 'F7-Fp1');
});

test('average reference rows sum to zero and skip bad electrodes', () => {
  const n = 50;
  const scalp = new Map(C.SCALP.map((name, k) => [name, Float32Array.from({ length: n }, (_, i) => Math.sin(i * (k + 1) / 9) * 40 + k)]));
  scalp.set('T5', null);
  const rows = C.deriveRows('average', scalp, []);
  assert.strictEqual(rows.length, 19);
  assert.strictEqual(rows.find(r => r.a === 'T5').values, null);
  for (let i = 0; i < n; i += 7) {
    let s = 0; for (const r of rows) if (r.values) s += r.values[i];
    close(s, 0, 1e-3, 'sample ' + i);
  }
});

test('extra channel groups follow the montage with their own group index', () => {
  const n = 5;
  const scalp = new Map(C.SCALP.map(name => [name, new Float32Array(n)]));
  const extras = [{ name: 'EKG', group: 'EKG', values: new Float32Array(n) },
    { name: 'SQ D-C', group: 'SubQ', values: new Float32Array(n) }, { name: 'SQ P-C', group: 'SubQ', values: new Float32Array(n) }];
  const rows = C.deriveRows('banana', scalp, extras);
  assert.strictEqual(rows.length, 21);
  assert.deepStrictEqual(rows.slice(18).map(r => r.group), [5, 6, 6]);
  const hidden = C.deriveRows('banana', scalp, extras, new Set(['SubQ']));
  assert.strictEqual(hidden.length, 19);
});

function sine(f, fs, n, amp) { return Float32Array.from({ length: n }, (_, i) => amp * Math.sin(2 * Math.PI * f * i / fs)); }
function rmsMid(x) { const a = Math.floor(x.length * 0.25), b = Math.floor(x.length * 0.75); let s = 0; for (let i = a; i < b; i++) s += x[i] * x[i]; return Math.sqrt(s / (b - a)); }

test('display filters: passband kept, stopbands attenuated, no phase shift', () => {
  const fs = 200, n = 2400;
  const secs = C.designFilters(fs, { lff: 1, hff: 70, notch: false });
  const pass = C.applyFilters(sine(10, fs, n, 50), secs, fs);
  close(rmsMid(pass) / (50 / Math.SQRT2), 1, 0.03, '10 Hz passband');
  const slow = C.applyFilters(sine(0.15, fs, n, 50), secs, fs);
  assert.ok(rmsMid(slow) < 0.05 * 50 / Math.SQRT2 * 4, '0.15 Hz drift should be strongly attenuated');
  const hff15 = C.designFilters(fs, { lff: 0, hff: 15, notch: false });
  assert.ok(rmsMid(C.applyFilters(sine(45, fs, n, 50), hff15, fs)) < 0.05 * 50 / Math.SQRT2, '45 Hz under a 15 Hz HFF');
  const notch = C.designFilters(fs, { lff: 0, hff: 0, notch: true });
  assert.ok(rmsMid(C.applyFilters(sine(60, fs, n, 50), notch, fs)) < 0.1 * 50 / Math.SQRT2, '60 Hz under the notch');
  close(rmsMid(C.applyFilters(sine(30, fs, n, 50), notch, fs)) / (50 / Math.SQRT2), 1, 0.05, '30 Hz kept by the notch');
  // zero phase: a symmetric pulse keeps its peak position
  const pulse = Float32Array.from({ length: n }, (_, i) => -100 * Math.exp(-((i - 1200) ** 2) / (2 * 6 * 6)));
  const out = C.applyFilters(pulse, secs, fs);
  let imin = 0; for (let i = 1; i < n; i++) if (out[i] < out[imin]) imin = i;
  assert.strictEqual(imin, 1200);
  // options at or above the usable band are dropped at low sampling rates
  assert.strictEqual(C.designFilters(100, { lff: 1, hff: 70, notch: true }).length, 1);
  assert.strictEqual(C.designFilters(128, { lff: 1, hff: 70, notch: true }).length, 2);
});

test('seeded order is a stable permutation that differs by rater', () => {
  const batch = { id: 'b01', orderSeed: 2026 };
  const a = C.shuffledOrder(100, C.raterSeed(batch, 'EC'));
  const a2 = C.shuffledOrder(100, C.raterSeed(batch, 'ec'));
  const b = C.shuffledOrder(100, C.raterSeed(batch, 'SJ'));
  assert.deepStrictEqual(a, a2);
  assert.notDeepStrictEqual(a, b);
  assert.deepStrictEqual(a.slice().sort((x, y) => x - y), Array.from({ length: 100 }, (_, i) => i));
});

test('CSV writer and parser round trip quotes, commas and newlines', () => {
  const rows = [{ a: 'plain', b: 'has, comma', c: 'say "hi"' }, { a: 'line\nbreak', b: '', c: '=1+1' }];
  const text = C.toCSV(['a', 'b', 'c'], rows);
  const back = C.parseCSV(text);
  assert.deepStrictEqual(back, rows);
  assert.deepStrictEqual(C.parseCSV('﻿x,y\r\n1,2\r\n'), [{ x: '1', y: '2' }]);
});

test('location summary derives side and region from maximum electrodes', () => {
  assert.deepStrictEqual(C.locationSummary(['F7', 'T3'], false), { side: 'left', region: 'temporal' });
  assert.deepStrictEqual(C.locationSummary(['F4'], false), { side: 'right', region: 'frontal' });
  assert.deepStrictEqual(C.locationSummary(['T3', 'T4'], false), { side: 'bilateral', region: 'temporal' });
  assert.deepStrictEqual(C.locationSummary(['Cz'], false), { side: 'midline', region: 'central' });
  assert.deepStrictEqual(C.locationSummary(['T3'], true), { side: 'bilateral', region: 'generalized' });
  assert.deepStrictEqual(C.locationSummary([], false), { side: '', region: '' });
});

test('demo batch parses, decodes, and shows a phase reversal at T3', () => {
  const raw = C.makeDemoBatch('rater');
  const batch = C.parseBatch(JSON.parse(JSON.stringify(raw)));
  assert.strictEqual(batch.clips.length, C.DEMO_SPECS.length);
  assert.strictEqual(batch.mode, 'rater');
  assert.ok(!batch.clips.some(c => c.meta || c.markers), 'rater batch must not carry meta or markers');
  assert.deepStrictEqual(batch.groups, { EKG: ['EKG'] });
  const clip = batch.clips[0]; // spike, maximum T3, at 6.06 s
  const sig = C.clipSignals(batch, clip);
  const rows = C.deriveRows('banana', sig.scalp, sig.extras);
  const secs = C.designFilters(batch.fs, { lff: 1, hff: 70 });
  const get = label => C.applyFilters(rows.find(r => r.label === label).values, secs, batch.fs);
  const i = Math.round(6.06 * batch.fs);
  const peakAround = (x, sign) => { let best = 0; for (let k = i - 6; k <= i + 6; k++) if (sign * x[k] > sign * best) best = x[k]; return best; };
  assert.ok(peakAround(get('F7-T3'), 1) > 30, 'F7-T3 deflects downward (positive) at the spike');
  assert.ok(peakAround(get('T3-T5'), -1) < -30, 'T3-T5 deflects upward (negative) at the spike');
  const research = C.parseBatch(C.makeDemoBatch('research'));
  assert.strictEqual(research.mode, 'research');
  assert.deepStrictEqual(research.groups, { EKG: ['EKG'], SubQ: ['SQ D-C', 'SQ P-C'] });
  assert.ok(research.clips[0].meta && research.clips[0].markers.length === 2);
});

test('exported labels import back into the same label objects', () => {
  const batch = C.parseBatch(C.makeDemoBatch('rater'));
  const order = C.shuffledOrder(batch.clips.length, C.raterSeed(batch, 'SJ'));
  const labels = {};
  labels[batch.clips[0].id] = { decision: 'yes', max: ['T3', 'F7'], generalized: false, artifact: false, marks: [2.7, -1.25], note: 'clear, "classic"', secs: 12.4, t: '2026-10-09T03:00:00.000Z' };
  labels[batch.clips[1].id] = { decision: 'no', max: ['T4'], generalized: false, artifact: true, marks: [], note: '', secs: 3, t: '2026-10-09T03:01:00.000Z' };
  const rows = C.exportRows(batch, order, labels, 'SJ', 123);
  const csv = C.toCSV(C.EXPORT_COLUMNS, rows);
  const back = C.importRows(batch, C.parseCSV(csv), 'sj');
  assert.deepStrictEqual(back.labels[batch.clips[0].id].max, ['T3', 'F7']);
  assert.deepStrictEqual(back.labels[batch.clips[0].id].marks, [-1.25, 2.7]);
  assert.strictEqual(back.labels[batch.clips[0].id].note, 'clear, "classic"');
  assert.strictEqual(back.labels[batch.clips[1].id].decision, 'no');
  const noRow = rows.find(r => r.clip_id === batch.clips[1].id);
  assert.strictEqual(noRow.max_electrodes, '', 'a No decision exports no location');
  assert.strictEqual(rows.find(r => r.clip_id === batch.clips[0].id).side, 'left');
  const other = C.importRows(batch, C.parseCSV(csv), 'EC');
  assert.strictEqual(other.skippedRater, rows.length);
});

test('parseBatch rejects malformed files with readable errors', () => {
  assert.throws(() => C.parseBatch({}), /not an IED review batch/);
  const raw = C.makeDemoBatch('rater');
  assert.throws(() => C.parseBatch(Object.assign({}, raw, { version: 9 })), /format version 9/);
  const dup = Object.assign({}, raw, { clips: [raw.clips[0], raw.clips[0]] });
  assert.throws(() => C.parseBatch(dup), /appears twice/);
});

test('batches say which answers they offer; older batches keep Unsure', () => {
  const raw = C.makeDemoBatch('rater');
  delete raw.answers;
  assert.deepStrictEqual(C.parseBatch(JSON.parse(JSON.stringify(raw))).answers, ['yes', 'no', 'unsure']);
  assert.deepStrictEqual(C.parseBatch(Object.assign({}, raw, { answers: ['no', 'yes'] })).answers, ['yes', 'no']);
  assert.deepStrictEqual(C.parseBatch(Object.assign({}, raw, { answers: ['yes', 'no', 'maybe'] })).answers, ['yes', 'no']);
  assert.throws(() => C.parseBatch(Object.assign({}, raw, { answers: ['yes'] })), /both Yes and No/);
});

test('admin view files: owner-only extras attach to their own batch, never another', () => {
  const batch = C.parseBatch(C.makeDemoBatch('rater'));
  const id = batch.clips[0].id;
  const obj = { format: 'ied-review-admin', version: 1, batch_id: batch.id, fs: batch.fs, clip_s: batch.clipS,
    clips: [{ id, implant_side: 'R', drawn_by: ['spikenet2', 'random'],
      strips: [{ name: 'SpikeNet2', kind: 'scalp', len: 1, secs: [[0, 0.1], [5, 0.92], [6, null]] },
        { name: 'Cyclops', kind: 'subq', secs: [[0.5, 0.3]] }],
      subq: { channels: ['D-C', 'P-C'], scale: [0.1, 0.1], data: 'AAAA' }, virtual: [['F8', 'T4'], ['F8', 'XX'], ['T4', 'T4']] },
    { id: 'not-in-batch', implant_side: 'L' }] };
  const m = C.parseAdmin(obj, batch);
  assert.strictEqual(m.size, 1);
  const it = m.get(id);
  assert.strictEqual(it.implantSide, 'R');
  assert.deepStrictEqual(it.drawnBy, ['spikenet2', 'random']);
  assert.deepStrictEqual(it.strips[0].secs, [[0, 0.1], [5, 0.92], [6, null]]);
  assert.strictEqual(it.strips[1].kind, 'subq');
  assert.deepStrictEqual(it.subq.channels, ['D-C', 'P-C']);
  assert.deepStrictEqual(it.virtual, [['F8', 'T4']], 'only real 10-20 pairs become virtual lanes');
  assert.throws(() => C.parseAdmin(Object.assign({}, obj, { batch_id: 'other' }), batch), /another batch/);
  assert.throws(() => C.parseAdmin(Object.assign({}, obj, { fs: 999 }), batch), /sampling rate/);
  assert.throws(() => C.parseAdmin({}, batch), /not an admin view/);
});

test('Yes/No keys (1.7.5): arrows step sensitivity and move one clip; other keys do nothing', () => {
  const k = (key, admin) => C.simpleKey(key, !!admin);
  assert.deepStrictEqual(k('y'), { act: 'decide', value: 'yes' });
  assert.deepStrictEqual(k('N'), { act: 'decide', value: 'no' });
  assert.deepStrictEqual(k('t'), { act: 'montage', value: 'transverse' });
  assert.deepStrictEqual(k('ArrowUp'), { act: 'sens', value: -1 }, 'Up: one rung more sensitive (fewer uV/mm)');
  assert.deepStrictEqual(k('ArrowDown'), { act: 'sens', value: 1 }, 'Down: one rung less sensitive');
  assert.deepStrictEqual(k('ArrowLeft'), { act: 'go', value: -1 });
  assert.deepStrictEqual(k('ArrowRight'), { act: 'go', value: 1 });
  assert.deepStrictEqual(k('s', true), { act: 'admin', value: 'subq' });
  assert.deepStrictEqual(k('d', true), { act: 'admin', value: 'strips' });
  // broken-input twins: keys outside the Yes/No set, and S/D without an admin table, do nothing
  for (const key of ['Enter', 'm', 'w', 'g', 'u', '?', ' ', 'Shift', 'ArrowUpX', 's', 'd']) {
    assert.strictEqual(k(key), null, `${key} must do nothing`);
  }
});

test('training examples (1.8.1, view both): the montage, then the UNEEG channels labelled as such', () => {
  const raw = C.makeDemoBatch('rater');
  assert.strictEqual(C.parseBatch(Object.assign({}, raw, { view: 'both' })).view, 'both');
  assert.strictEqual(C.parseBatch(Object.assign({}, raw, { view: 'subq' })).view, 'subq');
  assert.strictEqual(C.parseBatch(Object.assign({}, raw, { view: 'other' })).view, 'scalp');
  const n = 5;
  const scalp = new Map(C.SCALP.map(name => [name, new Float32Array(n)]));
  const extras = [{ name: 'EKG', group: 'EKG', values: new Float32Array(n) },
    { name: 'D-C', group: 'SubQ', values: new Float32Array(n) }, { name: 'D-P', group: 'SubQ', values: null }];
  const both = C.deriveRows('banana', scalp, extras, new Set(), 'both');
  assert.strictEqual(both.length, 21);
  assert.strictEqual(both[0].label, 'Fp1-F7');
  assert.deepStrictEqual(both.slice(18).map(r => r.label), ['EKG', 'UNEEG D-C', 'UNEEG D-P']);
  assert.strictEqual(both[20].values, null);
  // the scalp view keeps the plain names, and the UNEEG-only view draws no montage
  assert.deepStrictEqual(C.deriveRows('banana', scalp, extras, new Set(), 'scalp').slice(19).map(r => r.label), ['D-C', 'D-P']);
  assert.deepStrictEqual(C.deriveRows('banana', scalp, extras.slice(1), new Set(), 'subq').map(r => r.label), ['D-C', 'D-P']);
});

test('training examples (1.8.5) are shown, not asked: answers "seen", kept on import, type passed through', () => {
  const raw = C.makeDemoBatch('rater');
  raw.training = true;
  raw.answers = ['seen'];
  raw.clips.forEach(c => { c.ref = 'yes'; c.ref_type = 'focal left'; });
  const b = C.parseBatch(raw);
  assert.strictEqual(b.teaching, true);
  assert.deepStrictEqual(b.answers, ['seen']);
  assert.strictEqual(b.clips[0].ref_type, 'focal left');
  const id = b.clips[0].id;
  const got = C.importRows(b, [{ batch_id: b.id, clip_id: id, rater: 'EC', decision: 'seen' }], 'EC');
  assert.strictEqual(got.labels[id].decision, 'seen');
  // an older training batch still asks Yes or No, and a plain batch never takes "seen"
  const old = C.parseBatch(Object.assign({}, raw, { answers: ['yes', 'no'] }));
  assert.strictEqual(old.teaching, false);
  const plain = C.parseBatch(C.makeDemoBatch('rater'));
  assert.strictEqual(C.importRows(plain, [{ batch_id: plain.id, clip_id: plain.clips[0].id, decision: 'seen' }]).labels[plain.clips[0].id].decision, null);
  assert.throws(() => C.parseBatch(Object.assign({}, C.makeDemoBatch('rater'), { answers: ['seen'] })), /Yes and No/);
});

console.log(`\n${passed} passed${process.exitCode ? ', some failed' : ''}`);
