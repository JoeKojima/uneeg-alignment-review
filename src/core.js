/* IED Clip Review core: pure functions with no DOM access.
   Inlined into ied_review.html by build.py and unit-tested in Node (tests/test_core.js). */
(function (root) {
  'use strict';

  const VERSION = '1.8.6';
  const FORMAT = 'ied-review-batch';
  const FORMAT_VERSION = 2;  // 2: staged studies (kit 1.8: parts, training examples, UNEEG-only clips)

  // Classic 10-20 names, in the order MORGOTH and Cyclops use for referential input.
  const SCALP = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz',
    'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2'];
  const SCALP_UPPER = new Map(SCALP.map(n => [n.toUpperCase(), n]));
  const ALIASES = { T7: 'T3', T8: 'T4', P7: 'T5', P8: 'T6' };

  // 2D head-map positions: x to the right, y toward the nose, Cz at the origin,
  // the outer ring (Fp, F7/8, T, O) on radius 0.8 at 36 degree spacing.
  const POS = {
    Fp1: [-0.247, 0.761], Fp2: [0.247, 0.761],
    F7: [-0.647, 0.470], F3: [-0.320, 0.420], Fz: [0, 0.400], F4: [0.320, 0.420], F8: [0.647, 0.470],
    T3: [-0.800, 0], C3: [-0.400, 0], Cz: [0, 0], C4: [0.400, 0], T4: [0.800, 0],
    T5: [-0.647, -0.470], P3: [-0.320, -0.420], Pz: [0, -0.400], P4: [0.320, -0.420], T6: [0.647, -0.470],
    O1: [-0.247, -0.761], O2: [0.247, -0.761]
  };

  const REGION = {
    Fp1: 'frontal', Fp2: 'frontal', F3: 'frontal', F4: 'frontal', Fz: 'frontal',
    F7: 'temporal', F8: 'temporal', T1: 'temporal', T2: 'temporal',
    T3: 'temporal', T4: 'temporal', T5: 'temporal', T6: 'temporal',
    C3: 'central', C4: 'central', Cz: 'central',
    P3: 'parietal', P4: 'parietal', Pz: 'parietal',
    O1: 'occipital', O2: 'occipital'
  };

  // Montages. Bipolar entries are [first, second]; the trace is first minus second.
  const MONTAGES = {
    banana: {
      key: 'banana', name: 'Double banana', short: 'Banana', groups: [
        { label: 'L temporal', side: 'L', pairs: [['Fp1', 'F7'], ['F7', 'T3'], ['T3', 'T5'], ['T5', 'O1']] },
        { label: 'R temporal', side: 'R', pairs: [['Fp2', 'F8'], ['F8', 'T4'], ['T4', 'T6'], ['T6', 'O2']] },
        { label: 'L parasagittal', side: 'L', pairs: [['Fp1', 'F3'], ['F3', 'C3'], ['C3', 'P3'], ['P3', 'O1']] },
        { label: 'R parasagittal', side: 'R', pairs: [['Fp2', 'F4'], ['F4', 'C4'], ['C4', 'P4'], ['P4', 'O2']] },
        { label: 'Midline', side: 'Z', pairs: [['Fz', 'Cz'], ['Cz', 'Pz']] }
      ]
    },
    transverse: {
      key: 'transverse', name: 'Transverse', short: 'Transverse', groups: [
        { label: 'Frontopolar', side: 'X', pairs: [['F7', 'Fp1'], ['Fp1', 'Fp2'], ['Fp2', 'F8']] },
        { label: 'Frontal', side: 'X', pairs: [['F7', 'F3'], ['F3', 'Fz'], ['Fz', 'F4'], ['F4', 'F8']] },
        { label: 'Central', side: 'X', pairs: [['T3', 'C3'], ['C3', 'Cz'], ['Cz', 'C4'], ['C4', 'T4']] },
        { label: 'Parietal', side: 'X', pairs: [['T5', 'P3'], ['P3', 'Pz'], ['Pz', 'P4'], ['P4', 'T6']] },
        { label: 'Occipital', side: 'X', pairs: [['T5', 'O1'], ['O1', 'O2'], ['O2', 'T6']] }
      ]
    },
    average: {
      key: 'average', name: 'Average reference', short: 'Average', ref: 'avg', groups: [
        { label: 'L temporal', side: 'L', chans: ['Fp1', 'F7', 'T3', 'T5', 'O1'] },
        { label: 'R temporal', side: 'R', chans: ['Fp2', 'F8', 'T4', 'T6', 'O2'] },
        { label: 'L parasagittal', side: 'L', chans: ['F3', 'C3', 'P3'] },
        { label: 'R parasagittal', side: 'R', chans: ['F4', 'C4', 'P4'] },
        { label: 'Midline', side: 'Z', chans: ['Fz', 'Cz', 'Pz'] }
      ]
    }
  };
  const MONTAGE_ORDER = ['banana', 'transverse', 'average'];

  function normalizeName(raw) {
    let s = String(raw == null ? '' : raw).trim();
    s = s.replace(/^EEG[\s_:-]+/i, '');
    s = s.replace(/[\s_-]+(REF|LE|AVG|AV|CAR|A1|A2|M1|M2|A1A2|LINKED)$/i, '');
    const up = s.toUpperCase().replace(/\s+/g, '');
    if (SCALP_UPPER.has(up)) return SCALP_UPPER.get(up);
    if (ALIASES[up]) return ALIASES[up];
    if (/(EKG|ECG)/.test(up)) return 'EKG';
    return s;
  }

  function sideOf(e) {
    if (/z$/i.test(e)) return 'midline';
    const d = parseInt(String(e).replace(/\D/g, ''), 10);
    if (!isFinite(d)) return '';
    return d % 2 ? 'left' : 'right';
  }

  function locationSummary(maxEls, generalized) {
    if (generalized) return { side: 'bilateral', region: 'generalized' };
    const els = (maxEls || []).filter(Boolean);
    if (!els.length) return { side: '', region: '' };
    const sides = new Set(els.map(sideOf));
    let side = 'midline';
    if (sides.has('left') && sides.has('right')) side = 'bilateral';
    else if (sides.has('left')) side = 'left';
    else if (sides.has('right')) side = 'right';
    const regions = [];
    for (const e of els) { const r = REGION[e]; if (r && !regions.includes(r)) regions.push(r); }
    return { side, region: regions.join(';') };
  }

  /* ---------- signal decoding ---------- */

  function b64ToBytes(b64) {
    const bin = atob(b64);
    const n = bin.length;
    const out = new Uint8Array(n);
    for (let i = 0; i < n; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  function bytesToB64(bytes) {
    let s = '';
    const CH = 0x8000;
    for (let i = 0; i < bytes.length; i += CH) {
      s += String.fromCharCode.apply(null, bytes.subarray(i, i + CH));
    }
    return btoa(s);
  }

  // data: base64 of little-endian int16, channel-major [nCh][nS]; value_uV = int16 * scale[ch]
  function decodeClip(b64, nCh, nS, scales) {
    const bytes = b64ToBytes(b64);
    const want = nCh * nS * 2;
    if (bytes.length !== want) {
      throw new Error('Clip data holds ' + bytes.length + ' bytes; expected ' + want +
        ' (' + nCh + ' channels x ' + nS + ' samples x 2 bytes).');
    }
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const out = new Array(nCh);
    for (let c = 0; c < nCh; c++) {
      const a = new Float32Array(nS);
      const s = Number(scales[c]);
      const base = c * nS * 2;
      for (let i = 0; i < nS; i++) a[i] = dv.getInt16(base + 2 * i, true) * s;
      out[c] = a;
    }
    return out;
  }

  // rows: array of numeric arrays (µV). Removes each row's mean, quantizes to int16 per row.
  function encodeClip(rows) {
    const nCh = rows.length;
    const nS = rows[0].length;
    const buf = new ArrayBuffer(nCh * nS * 2);
    const dv = new DataView(buf);
    const scale = [];
    for (let c = 0; c < nCh; c++) {
      const r = rows[c];
      let mean = 0;
      for (let i = 0; i < nS; i++) mean += r[i];
      mean /= nS;
      let peak = 0;
      for (let i = 0; i < nS; i++) { const v = Math.abs(r[i] - mean); if (v > peak) peak = v; }
      const s = peak > 0 ? Number((peak / 32000).toPrecision(6)) : 1;
      scale.push(s);
      for (let i = 0; i < nS; i++) dv.setInt16((c * nS + i) * 2, Math.round((r[i] - mean) / s), true);
    }
    return { scale, data: bytesToB64(new Uint8Array(buf)) };
  }

  /* ---------- batches ---------- */

  // A batch encrypted by encrypt_batch.py: "ALN1", 16-byte salt, 12-byte IV, then AES-256-GCM ciphertext of the
  // .json.gz bytes; the key is PBKDF2-SHA256 of the password (iterations in bytes 4-8, big-endian).
  async function decryptBatchBytes(bytes, password) {
    const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
    if (magic !== 'ALN1') throw new Error('This is not an encrypted review batch.');
    const iter = ((bytes[4] << 24) | (bytes[5] << 16) | (bytes[6] << 8) | bytes[7]) >>> 0;
    const salt = bytes.slice(8, 24), iv = bytes.slice(24, 36), ct = bytes.slice(36);
    const subtle = (typeof crypto !== 'undefined' && crypto.subtle) ? crypto.subtle : require('crypto').webcrypto.subtle;
    const base = await subtle.importKey('raw', new TextEncoder().encode(password), 'PBKDF2', false, ['deriveKey']);
    const key = await subtle.deriveKey({ name: 'PBKDF2', salt, iterations: iter, hash: 'SHA-256' }, base,
      { name: 'AES-GCM', length: 256 }, false, ['decrypt']);
    return new Uint8Array(await subtle.decrypt({ name: 'AES-GCM', iv }, key, ct));
  }

  function parseBatch(obj) {
    if (!obj || typeof obj !== 'object' || obj.format !== FORMAT) {
      throw new Error('This file is not an IED review batch. Batch files come from make_review_batches.py.');
    }
    if (Number(obj.version) > FORMAT_VERSION) {
      throw new Error('This batch uses format version ' + obj.version + ', and this viewer reads version ' +
        FORMAT_VERSION + '. Use the viewer that shipped with the batch.');
    }
    const fs = Number(obj.fs);
    if (!(fs > 0)) throw new Error('The batch has no valid sampling rate ("fs").');
    const clipS = Number(obj.clip_s);
    if (!(clipS > 0)) throw new Error('The batch has no valid clip length ("clip_s").');
    if (!Array.isArray(obj.channels) || !obj.channels.length) throw new Error('The batch lists no channels.');
    if (!Array.isArray(obj.clips) || !obj.clips.length) throw new Error('The batch contains no clips.');
    const channels = obj.channels.map(normalizeName);
    const nS = Math.round(clipS * fs);
    const seen = new Set();
    obj.clips.forEach((c, i) => {
      if (!c || !c.id) throw new Error('Clip ' + (i + 1) + ' has no id.');
      if (seen.has(c.id)) throw new Error('Clip id ' + c.id + ' appears twice in this batch.');
      seen.add(c.id);
      if (typeof c.data !== 'string') throw new Error('Clip ' + c.id + ' has no signal data.');
      if (obj.training && c.ref !== 'yes' && c.ref !== 'no') throw new Error('Training example ' + c.id + ' has no expert answer.');
      if (!Array.isArray(c.scale) || c.scale.length !== channels.length) {
        throw new Error('Clip ' + c.id + ' has ' + (c.scale ? c.scale.length : 0) + ' scale factors for ' +
          channels.length + ' channels.');
      }
    });
    const groups = {};
    const assigned = new Set();
    if (obj.groups && typeof obj.groups === 'object') {
      for (const [g, list] of Object.entries(obj.groups)) {
        const names = (list || []).map(normalizeName).filter(n => channels.includes(n) && !SCALP.includes(n));
        if (names.length) { groups[g] = names; names.forEach(n => assigned.add(n)); }
      }
    }
    const other = channels.filter(n => !SCALP.includes(n) && !assigned.has(n));
    if (other.length) groups.Other = other;
    // the answers this study offers; batches made before kit 1.6 say nothing and keep Unsure
    let answers = ['yes', 'no', 'unsure'];
    // training examples since kit 1.8.5 are looked at, not answered: the viewer records "seen" for each one shown
    const teaching = !!obj.training && Array.isArray(obj.answers) && obj.answers.length === 1 && obj.answers[0] === 'seen';
    if (teaching) answers = ['seen'];
    else if (Array.isArray(obj.answers)) {
      answers = ['yes', 'no', 'unsure'].filter(a => obj.answers.includes(a));
      if (!answers.includes('yes') || !answers.includes('no')) throw new Error('The batch must offer both Yes and No ("answers").');
    }
    return {
      id: String(obj.batch_id || 'batch'),
      mode: obj.mode === 'research' ? 'research' : 'rater',
      fs, clipS, nS,
      padS: obj.pad_s != null ? Number(obj.pad_s) : 1,
      eventS: obj.event_s != null ? Number(obj.event_s) : clipS / 2,
      candidateS: obj.candidate_s != null ? Number(obj.candidate_s) : 1,
      channels,
      chIndex: new Map(channels.map((n, i) => [n, i])),
      groups,
      clips: obj.clips,
      orderSeed: obj.order_seed != null ? obj.order_seed : 0,
      display: obj.display && typeof obj.display === 'object' ? obj.display : {},
      answers,
      // what the rater is asked (kit 1.7): about the whole 10 s clip, or about the highlighted second (older batches)
      question: obj.question === 'clip' ? 'clip' : obj.question === 'alignment' ? 'alignment' : 'second',
      // a time-alignment review (alignment-review branch): the question and the three answers in words,
      // e.g. {"yes": "Time aligned", "no": "Not time aligned", "unsure": "Unsure"}
      questionText: obj.question_text ? String(obj.question_text) : '',
      answerLabels: obj.answer_labels && typeof obj.answer_labels === 'object' ? obj.answer_labels : {},
      created: obj.created || '',
      notes: obj.notes || '',
      synthetic: !!obj.synthetic,
      // a staged study (kit 1.8): which part, whether it is the UNEEG channels only, whether answers are shown
      stage: Number(obj.stage) > 0 ? Number(obj.stage) : 0,
      stageName: obj.stage_name ? String(obj.stage_name) : '',
      // both (kit 1.8.1): the scalp montage with the UNEEG channels under it, for training examples
      view: obj.view === 'subq' || obj.view === 'both' ? obj.view : 'scalp',
      training: !!obj.training,
      teaching,
      // the margin raters are told to count around the highlighted second, s (the server may set it too)
      toleranceS: Number(obj.tolerance_s) > 0 ? Math.min(1, Number(obj.tolerance_s)) : 0
    };
  }

  // The owner-only admin view of a batch (make_review_batches.admin_item): per clip, SubQ channels over the same
  // 12 s, the implant side, which draw picked it, and detector strips. Returns Map(clip id -> item).
  function parseAdmin(obj, batch) {
    if (!obj || obj.format !== 'ied-review-admin') throw new Error('This is not an admin view file.');
    if (String(obj.batch_id) !== batch.id) throw new Error('The admin view belongs to another batch.');
    if (Number(obj.fs) !== batch.fs || Number(obj.clip_s) !== batch.clipS) {
      throw new Error('The admin view was made at another sampling rate or clip length.');
    }
    const ids = new Set(batch.clips.map(c => c.id));
    const out = new Map();
    for (const it of obj.clips || []) {
      if (!it || !ids.has(it.id)) continue;
      const item = { implantSide: it.implant_side === 'L' || it.implant_side === 'R' ? it.implant_side : '',
        drawnBy: Array.isArray(it.drawn_by) ? it.drawn_by.map(String) : [], error: it.error ? String(it.error) : '',
        strips: [], subq: null, virtual: [], scalp: null };
      for (const pr of Array.isArray(it.virtual) ? it.virtual : []) {
        if (Array.isArray(pr) && pr.length === 2 && POS[pr[0]] && POS[pr[1]] && pr[0] !== pr[1]) item.virtual.push([pr[0], pr[1]]);
      }
      for (const st of Array.isArray(it.strips) ? it.strips : []) {
        const secs = (Array.isArray(st.secs) ? st.secs : []).filter(x => Array.isArray(x) && isFinite(Number(x[0])))
          .map(x => [Number(x[0]), x[1] == null || !isFinite(Number(x[1])) ? null : Number(x[1])]);
        item.strips.push({ name: String(st.name || ''), kind: st.kind === 'subq' ? 'subq' : 'scalp',
          len: Number(st.len) > 0 ? Number(st.len) : 1, secs });
      }
      const sq = it.subq;
      if (sq && Array.isArray(sq.channels) && sq.channels.length && Array.isArray(sq.scale) &&
          sq.scale.length === sq.channels.length && typeof sq.data === 'string') {
        item.subq = { channels: sq.channels.map(String), scale: sq.scale, data: sq.data };
      }
      // a UNEEG clip's 10-20 channels over the same 12 s (kit 1.8.5)
      const sc = it.scalp;
      if (sc && Array.isArray(sc.channels) && sc.channels.length && Array.isArray(sc.scale) &&
          sc.scale.length === sc.channels.length && typeof sc.data === 'string') {
        item.scalp = { channels: sc.channels.map(normalizeName), scale: sc.scale, data: sc.data,
          bad: Array.isArray(sc.bad) ? sc.bad.map(normalizeName) : [] };
      }
      out.set(it.id, item);
    }
    return out;
  }

  // Decoded signals for one clip: scalp map (null for missing or bad electrodes) plus extra channels by group.
  function clipSignals(batch, clip) {
    const arrs = decodeClip(clip.data, batch.channels.length, batch.nS, clip.scale);
    const bad = new Set((clip.bad || []).map(normalizeName));
    const scalp = new Map();
    for (const name of SCALP) {
      const i = batch.chIndex.get(name);
      scalp.set(name, i == null || bad.has(name) ? null : arrs[i]);
    }
    const extras = [];
    for (const [g, names] of Object.entries(batch.groups)) {
      for (const name of names) {
        const i = batch.chIndex.get(name);
        extras.push({ name, group: g, values: i == null || bad.has(name) ? null : arrs[i] });
      }
    }
    return { scalp, extras };
  }

  function subtract(a, b) {
    const n = a.length;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = a[i] - b[i];
    return out;
  }

  // Rows to draw for a montage: scalp derivations, then extra channel groups as recorded.
  function deriveRows(montageKey, scalp, extras, hideGroups, view) {
    const m = MONTAGES[montageKey] || MONTAGES.banana;
    const rows = [];
    let gi = 0;
    if (view === 'subq') {
      // UNEEG-only clips (kit 1.8): the channels as recorded, no scalp montage
    } else if (m.ref === 'avg') {
      const avail = SCALP.filter(n => scalp.get(n));
      let mean = null;
      if (avail.length) {
        const n = scalp.get(avail[0]).length;
        mean = new Float64Array(n);
        for (const name of avail) { const v = scalp.get(name); for (let i = 0; i < n; i++) mean[i] += v[i]; }
        for (let i = 0; i < n; i++) mean[i] /= avail.length;
      }
      for (const g of m.groups) {
        for (const name of g.chans) {
          const v = scalp.get(name);
          rows.push({ label: name + '-Avg', group: gi, side: g.side, a: name, b: null, values: v && mean ? subtract(v, mean) : null });
        }
        gi++;
      }
    } else {
      for (const g of m.groups) {
        for (const [a, b] of g.pairs) {
          const va = scalp.get(a), vb = scalp.get(b);
          rows.push({ label: a + '-' + b, group: gi, side: g.side, a, b, values: va && vb ? subtract(va, vb) : null });
        }
        gi++;
      }
    }
    let lastGroup = null;
    let egi = gi - 1;
    for (const ex of extras || []) {
      if (hideGroups && hideGroups.has(ex.group)) continue;
      if (ex.group !== lastGroup) { egi++; lastGroup = ex.group; }
      // under the scalp lanes, UNEEG channels say so (D-C alone could pass for a scalp pair)
      const label = view === 'both' && /sub\s*q|^sq/i.test(ex.group) ? 'UNEEG ' + ex.name : ex.name;
      rows.push({ label, group: egi, side: 'E', extraGroup: ex.group, a: ex.name, b: null, values: ex.values });
    }
    return rows;
  }

  /* ---------- filters: zero-phase IIR (forward and backward) ---------- */

  function highpass1(fc, fs) {
    const K = Math.tan(Math.PI * fc / fs);
    const b0 = 1 / (1 + K);
    return [b0, -b0, 0, (K - 1) / (K + 1), 0];
  }
  function lowpass2(fc, fs) {
    const w0 = 2 * Math.PI * fc / fs, cw = Math.cos(w0), alpha = Math.sin(w0) / (2 * Math.SQRT1_2);
    const a0 = 1 + alpha;
    return [(1 - cw) / 2 / a0, (1 - cw) / a0, (1 - cw) / 2 / a0, -2 * cw / a0, (1 - alpha) / a0];
  }
  function notch2(f0, fs, Q) {
    const w0 = 2 * Math.PI * f0 / fs, cw = Math.cos(w0), alpha = Math.sin(w0) / (2 * Q);
    const a0 = 1 + alpha;
    return [1 / a0, -2 * cw / a0, 1 / a0, -2 * cw / a0, (1 - alpha) / a0];
  }

  // Filter sections for the chosen display settings. Options above about 0.45 fs are dropped.
  function designFilters(fs, opts) {
    const secs = [];
    const nyq = fs / 2;
    const lff = Number(opts && opts.lff) || 0;
    const hff = Number(opts && opts.hff) || 0;
    if (lff > 0 && lff < nyq * 0.5) secs.push(highpass1(lff, fs));
    if (hff > 0 && hff < nyq * 0.9) secs.push(lowpass2(hff, fs));
    if (opts && opts.notch && nyq > 62) secs.push(notch2(60, fs, 25));
    return secs;
  }

  function runSections(y, secs) {
    for (const [b0, b1, b2, a1, a2] of secs) {
      let z1 = 0, z2 = 0;
      for (let i = 0; i < y.length; i++) {
        const x = y[i];
        const out = b0 * x + z1;
        z1 = b1 * x - a1 * out + z2;
        z2 = b2 * x - a2 * out;
        y[i] = out;
      }
    }
  }

  // Zero-phase filtering with odd reflection at both ends, as in scipy.signal.filtfilt.
  function filtfilt(x, secs, padLen) {
    const n = x.length;
    if (!secs.length || n < 4) return Float32Array.from(x);
    const p = Math.max(1, Math.min(padLen | 0, n - 1));
    const y = new Float64Array(n + 2 * p);
    const x0 = x[0], xn = x[n - 1];
    for (let i = 0; i < p; i++) y[i] = 2 * x0 - x[p - i];
    for (let i = 0; i < n; i++) y[p + i] = x[i];
    for (let i = 0; i < p; i++) y[p + n + i] = 2 * xn - x[n - 2 - i];
    runSections(y, secs);
    y.reverse();
    runSections(y, secs);
    y.reverse();
    return Float32Array.from(y.subarray(p, p + n));
  }

  function applyFilters(values, secs, fs) {
    if (!values) return null;
    let mean = 0;
    for (let i = 0; i < values.length; i++) mean += values[i];
    mean /= values.length;
    const centered = new Float32Array(values.length);
    for (let i = 0; i < values.length; i++) centered[i] = values[i] - mean;
    if (!secs.length) return centered;
    return filtfilt(centered, secs, Math.round(fs * 2));
  }

  /* ---------- seeded randomness ---------- */

  function cyrb32(str, seed) {
    let h1 = 0xdeadbeef ^ (seed || 0), h2 = 0x41c6ce57 ^ (seed || 0);
    for (let i = 0; i < str.length; i++) {
      const ch = str.charCodeAt(i);
      h1 = Math.imul(h1 ^ ch, 2654435761);
      h2 = Math.imul(h2 ^ ch, 1597334677);
    }
    h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
    h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
    return (h2 ^ h1) >>> 0;
  }

  function mulberry32(a) {
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function shuffledOrder(n, seed) {
    const rng = mulberry32(seed >>> 0);
    const idx = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }
    return idx;
  }

  function raterSeed(batch, rater) {
    return cyrb32(String(batch.orderSeed) + '|' + batch.id + '|' + String(rater).toUpperCase());
  }

  /* ---------- CSV ---------- */

  const EXPORT_COLUMNS = ['batch_id', 'clip_id', 'rater', 'order_index', 'decision', 'max_electrodes', 'side',
    'region', 'generalized', 'artifact', 'other_discharges_s', 'note', 'seconds_viewed', 'labeled_at',
    'order_seed', 'viewer_version', 'view'];

  function csvCell(v) {
    if (v == null) return '';
    const s = String(v);
    return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  }

  function toCSV(columns, rows) {
    const lines = [columns.join(',')];
    for (const r of rows) lines.push(columns.map(c => csvCell(r[c])).join(','));
    return lines.join('\n') + '\n';
  }

  function parseCSV(text) {
    const rows = [];
    let row = [], cell = '', inQ = false;
    const s = String(text).replace(/^﻿/, '');
    for (let i = 0; i < s.length; i++) {
      const ch = s[i];
      if (inQ) {
        if (ch === '"') {
          if (s[i + 1] === '"') { cell += '"'; i++; } else inQ = false;
        } else cell += ch;
      } else if (ch === '"') inQ = true;
      else if (ch === ',') { row.push(cell); cell = ''; }
      else if (ch === '\n' || ch === '\r') {
        if (ch === '\r' && s[i + 1] === '\n') i++;
        row.push(cell); cell = '';
        if (row.length > 1 || row[0] !== '') rows.push(row);
        row = [];
      } else cell += ch;
    }
    if (cell !== '' || row.length) { row.push(cell); if (row.length > 1 || row[0] !== '') rows.push(row); }
    if (!rows.length) return [];
    const head = rows[0].map(h => h.trim());
    return rows.slice(1).map(r => Object.fromEntries(head.map((h, i) => [h, r[i] != null ? r[i] : ''])));
  }

  function exportRows(batch, order, labels, rater, seed) {
    return order.map((ci, k) => {
      const clip = batch.clips[ci];
      const L = labels[clip.id] || {};
      const located = L.decision === 'yes' || L.decision === 'unsure';
      const loc = located ? locationSummary(L.max || [], !!L.generalized) : { side: '', region: '' };
      return {
        batch_id: batch.id,
        clip_id: clip.id,
        rater,
        order_index: k + 1,
        decision: L.decision || '',
        max_electrodes: located ? (L.max || []).join(';') : '',
        side: loc.side,
        region: loc.region,
        generalized: located && L.generalized ? 1 : 0,
        artifact: L.artifact ? 1 : 0,
        other_discharges_s: (L.marks || []).slice().sort((a, b) => a - b).map(t => t.toFixed(2)).join(';'),
        note: L.note || '',
        seconds_viewed: Math.round(L.secs || 0),
        labeled_at: L.t || '',
        order_seed: seed,
        viewer_version: VERSION,
        // what was on screen when the answer was given: blind, admin (SubQ shown) or superadmin (detector scores shown)
        view: L.decision ? (L.view || '') : ''
      };
    });
  }

  // Turns rows of an exported CSV back into label objects for clips in this batch.
  function importRows(batch, rows, rater) {
    const ids = new Set(batch.clips.map(c => c.id));
    const out = {};
    let skippedBatch = 0, skippedRater = 0, unknown = 0;
    for (const r of rows) {
      if (r.batch_id && r.batch_id !== batch.id) { skippedBatch++; continue; }
      if (rater && r.rater && String(r.rater).toUpperCase() !== String(rater).toUpperCase()) { skippedRater++; continue; }
      if (!ids.has(r.clip_id)) { unknown++; continue; }
      const decision = ['yes', 'no', 'unsure'].concat(batch.teaching ? ['seen'] : []).includes(r.decision) ? r.decision : null;
      out[r.clip_id] = {
        decision,
        max: r.max_electrodes ? r.max_electrodes.split(';').map(normalizeName).filter(Boolean) : [],
        generalized: r.generalized === '1' || r.generalized === 'true',
        artifact: r.artifact === '1' || r.artifact === 'true',
        marks: r.other_discharges_s ? r.other_discharges_s.split(';').map(Number).filter(isFinite) : [],
        note: r.note || '',
        secs: Number(r.seconds_viewed) || 0,
        t: r.labeled_at || '',
        view: ['blind', 'admin', 'superadmin'].includes(r.view) ? r.view : ''
      };
    }
    return { labels: out, skippedBatch, skippedRater, unknown };
  }

  /* ---------- synthetic EEG for the demo batch ---------- */

  function gauss(rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // Roughly 1/f noise from a sum of AR(1) processes, unit standard deviation.
  function pinkNoise(n, rng) {
    const poles = [0.995, 0.98, 0.94, 0.85, 0.6];
    const gains = [0.55, 0.65, 0.7, 0.55, 0.4];
    const st = poles.map((p, k) => gains[k] * gauss(rng));
    const out = new Float64Array(n);
    let ss = 0;
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let k = 0; k < poles.length; k++) {
        st[k] = poles[k] * st[k] + gains[k] * Math.sqrt(1 - poles[k] * poles[k]) * gauss(rng);
        s += st[k];
      }
      out[i] = s; ss += s * s;
    }
    const sd = Math.sqrt(ss / n) || 1;
    for (let i = 0; i < n; i++) out[i] /= sd;
    return out;
  }

  function dist(a, b) { const p = POS[a], q = POS[b]; return Math.hypot(p[0] - q[0], p[1] - q[1]); }
  function fieldWeight(name, at, width) { const d = dist(name, at); return Math.exp(-(d * d) / (2 * width * width)); }

  // Spike with aftergoing slow wave; t in seconds from the negative peak; negative peak of -A.
  function spikeShape(t, A) {
    const rise = 0.013, fall = 0.022;
    const s = t < 0 ? Math.exp(-(t * t) / (2 * rise * rise)) : Math.exp(-(t * t) / (2 * fall * fall));
    const over = 0.28 * Math.exp(-((t - 0.055) ** 2) / (2 * 0.018 * 0.018));
    const slow = 0.45 * Math.exp(-((t - 0.19) ** 2) / (2 * 0.065 * 0.065));
    return -A * s + A * over - A * slow;
  }

  function addEvent(ch, fs, t0, dur, fn) {
    const n = ch.length;
    const i0 = Math.max(0, Math.floor((t0 - 0.2) * fs));
    const i1 = Math.min(n, Math.ceil((t0 + dur) * fs));
    for (let i = i0; i < i1; i++) ch[i] += fn(i / fs - t0);
  }

  function synthClip(spec, seed, fs, clipS, eventS) {
    const rng = mulberry32(seed >>> 0);
    const n = Math.round(fs * clipS);
    const ch = {};
    for (const name of SCALP) ch[name] = new Float64Array(n);
    const ekg = new Float64Array(n);

    // Background: smooth latent sources, plus independent electrode noise.
    for (let k = 0; k < 7; k++) {
      const cx = (rng() * 2 - 1) * 0.7, cy = (rng() * 2 - 1) * 0.7;
      const w = 0.35 + rng() * 0.35, amp = 5 + rng() * 6;
      const src = pinkNoise(n, rng);
      for (const name of SCALP) {
        const [x, y] = POS[name];
        const g = amp * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * w * w));
        if (g < 0.05) continue;
        const a = ch[name];
        for (let i = 0; i < n; i++) a[i] += g * src[i];
      }
    }
    for (const name of SCALP) { const a = ch[name]; for (let i = 0; i < n; i++) a[i] += 1.0 * gauss(rng); }

    const state = spec.state || 'awake';
    if (state === 'awake') {
      const f = 9 + rng() * 1.4, ph = rng() * 6.283, envPh = rng() * 6.283;
      const w = { O1: 30, O2: 30, P3: 19, P4: 19, Pz: 17, T5: 17, T6: 17, C3: 7, C4: 7, Cz: 7 };
      for (const name of SCALP) {
        const amp = w[name] || 3;
        const lag = (POS[name][1] + 0.76) * 0.25;
        const a = ch[name];
        for (let i = 0; i < n; i++) {
          const t = i / fs;
          const env = 0.55 + 0.45 * Math.sin(2 * Math.PI * 0.33 * t + envPh);
          a[i] += amp * env * Math.sin(2 * Math.PI * f * t + ph + lag);
        }
      }
    } else {
      const f = 5 + rng() * 1.2, ph = rng() * 6.283;
      for (const name of SCALP) {
        const a = ch[name];
        const amp = 6 + 4 * Math.exp(-(POS[name][0] ** 2 + POS[name][1] ** 2) / 0.5);
        for (let i = 0; i < n; i++) {
          const t = i / fs;
          a[i] += amp * (0.6 + 0.4 * Math.sin(2 * Math.PI * 0.21 * t + ph)) * Math.sin(2 * Math.PI * f * t + ph * 1.7);
        }
      }
    }

    // EKG: a QRS complex every ~0.86 s, faintly visible on left temporal-occipital electrodes.
    const hr = 0.82 + rng() * 0.12;
    for (let tb = rng() * hr; tb < clipS; tb += hr * (0.97 + rng() * 0.06)) {
      addEvent(ekg, fs, tb, 0.5, t => 900 * Math.exp(-(t * t) / (2 * 0.012 * 0.012)) - 160 * Math.exp(-((t - 0.03) ** 2) / (2 * 0.012 * 0.012)) + 130 * Math.exp(-((t - 0.25) ** 2) / (2 * 0.05 * 0.05)));
      const k = spec.kind === 'ekg' ? 1 : 0.25;
      for (const [name, g] of [['T3', 9], ['T5', 11], ['O1', 7], ['P3', 4]]) {
        addEvent(ch[name], fs, tb, 0.08, t => k * g * (Math.exp(-(t * t) / (2 * 0.011 * 0.011)) - 0.25 * Math.exp(-((t - 0.028) ** 2) / (2 * 0.01 * 0.01))));
      }
    }

    const t0 = eventS + (spec.dt || 0);
    const placeSpike = (at, tt, A, width) => {
      for (const name of SCALP) {
        const g = fieldWeight(name, at, width || 0.36);
        if (g < 0.03) continue;
        addEvent(ch[name], fs, tt, 0.5, t => spikeShape(t, A * g));
      }
    };

    switch (spec.kind) {
      case 'spike':
        placeSpike(spec.at, t0, spec.amp, spec.width);
        break;
      case 'sharp_small':
        placeSpike(spec.at, t0, spec.amp, 0.26);
        break;
      case 'gsw': {
        const dur = spec.dur || 2.2;
        for (let tt = t0 - 0.35; tt < t0 - 0.35 + dur; tt += 1 / 3.1) {
          for (const name of SCALP) {
            const g = fieldWeight(name, spec.at || 'Fz', 0.85);
            addEvent(ch[name], fs, tt, 0.45, t => spikeShape(t, spec.amp * 0.55 * g) - spec.amp * 0.65 * g * Math.exp(-((t - 0.16) ** 2) / (2 * 0.06 * 0.06)));
          }
        }
        break;
      }
      case 'blink': {
        const fp = [0, 1.0];
        for (const name of SCALP) {
          const [x, y] = POS[name];
          const g = Math.exp(-((x - fp[0]) ** 2 * 0.6 + (y - fp[1]) ** 2) / (2 * 0.16));
          if (g < 0.03) continue;
          addEvent(ch[name], fs, t0 - 0.12, 0.5, t => (t >= 0 && t <= 0.36) ? 140 * g * Math.sin(Math.PI * t / 0.36) ** 1.5 : 0);
        }
        break;
      }
      case 'muscle': {
        const names = spec.side === 'R' ? [['F8', 1], ['T4', 0.9], ['T6', 0.55], ['Fp2', 0.35]] : [['F7', 1], ['T3', 0.9], ['T5', 0.55], ['Fp1', 0.35]];
        const dur = 1.6;
        const start = t0 - 0.6;
        for (const [name, g] of names) {
          let prev = 0;
          const a = ch[name];
          for (let i = Math.max(0, Math.floor(start * fs)); i < Math.min(n, Math.ceil((start + dur) * fs)); i++) {
            const t = i / fs - start;
            const env = Math.sin(Math.PI * t / dur) ** 0.6;
            const w = gauss(rng);
            a[i] += 22 * g * env * (w - 0.6 * prev);
            prev = w;
          }
        }
        break;
      }
      case 'pop': {
        const sgn = rng() < 0.5 ? 1 : -1;
        addEvent(ch[spec.at], fs, t0, 3.0, t => t < 0 ? 0 : sgn * 160 * (1 - Math.exp(-t / 0.004)) * Math.exp(-t / 0.28));
        break;
      }
      case 'vertex': {
        for (const name of SCALP) {
          const g = fieldWeight(name, 'Cz', 0.42);
          if (g < 0.03) continue;
          addEvent(ch[name], fs, t0, 0.5, t => -75 * g * Math.exp(-(t * t) / (2 * 0.048 * 0.048)) + 24 * g * Math.exp(-((t - 0.13) ** 2) / (2 * 0.05 * 0.05)));
        }
        break;
      }
      default:
        break;
    }
    for (const extra of spec.also || []) placeSpike(extra.at, eventS + extra.t, extra.amp);

    // Subcutaneous channels for the research view: a lead over the left temporal region
    // (central contact near T3), sampled through the same scalp field with its own noise
    // and a 20 ms residual misalignment.
    const lag = Math.round(0.02 * fs);
    const sq = [new Float64Array(n), new Float64Array(n)];
    for (let i = 0; i < n; i++) {
      const j = Math.min(n - 1, Math.max(0, i - lag));
      sq[0][i] = 0.75 * (ch.T5[j] - ch.T3[j]) + 1.6 * gauss(rng);
      sq[1][i] = 0.75 * (ch.F7[j] - ch.T3[j]) + 1.6 * gauss(rng);
    }
    return { scalp: SCALP.map(name => ch[name]), ekg, sq };
  }

  const DEMO_SPECS = [
    { kind: 'spike', at: 'T3', amp: 115, dt: 0.06, p: 0.93, cy: 0.91, what: 'Spike, maximum T3' },
    { kind: 'none', p: 0.04, what: 'Background, awake' },
    { kind: 'blink', p: 0.21, what: 'Eye blink' },
    { kind: 'spike', at: 'F8', amp: 95, dt: -0.18, p: 0.86, what: 'Spike, maximum F8' },
    { kind: 'vertex', state: 'drowsy', p: 0.47, what: 'Vertex wave in drowsiness' },
    { kind: 'spike', at: 'T4', amp: 125, dt: 0.12, also: [{ at: 'T4', t: 2.7, amp: 105 }], p: 0.95, what: 'Spike, maximum T4, second spike at +2.7 s' },
    { kind: 'pop', at: 'T5', p: 0.38, what: 'Electrode pop at T5' },
    { kind: 'gsw', at: 'Fz', amp: 230, p: 0.98, cy: 0.94, what: 'Generalized 3 Hz spike-and-wave' },
    { kind: 'muscle', side: 'L', p: 0.27, cy: 0.58, what: 'Left temporal muscle artifact' },
    { kind: 'spike', at: 'F7', amp: 75, dt: 0, state: 'drowsy', p: 0.81, cy: 0.84, what: 'Spike, maximum F7' },
    { kind: 'ekg', p: 0.12, what: 'EKG artifact' },
    { kind: 'none', state: 'drowsy', p: 0.09, what: 'Background, drowsy' },
    { kind: 'sharp_small', at: 'T5', amp: 38, dt: 0.2, p: 0.55, cy: 0.49, what: 'Low-amplitude sharp transient at T5' },
    { kind: 'spike', at: 'T6', amp: 100, dt: -0.1, p: 0.88, what: 'Spike, maximum T6' },
    { kind: 'spike', at: 'F3', amp: 95, dt: 0.15, p: 0.84, what: 'Spike, maximum F3' },
    { kind: 'none', p: 0.02, what: 'Background, awake' },
    { kind: 'spike', at: 'T3', amp: 150, dt: -0.05, also: [{ at: 'T3', t: -3.1, amp: 120 }], p: 0.97, cy: 0.95, what: 'Spike, maximum T3, second spike at -3.1 s' },
    { kind: 'muscle', side: 'R', p: 0.19, what: 'Right temporal muscle artifact' },
    { kind: 'spike', at: 'O1', amp: 85, dt: 0.25, p: 0.72, what: 'Spike, maximum O1' },
    { kind: 'vertex', state: 'drowsy', p: 0.41, what: 'Vertex wave in drowsiness' },
    { kind: 'spike', at: 'T4', amp: 62, dt: 0.08, state: 'drowsy', p: 0.66, what: 'Spike, maximum T4' },
    { kind: 'none', p: 0.07, what: 'Background, awake' },
    { kind: 'gsw', at: 'Fz', amp: 180, dur: 1.2, p: 0.92, cy: 0.88, what: 'Generalized spike-and-wave, short run' },
    { kind: 'spike', at: 'T5', amp: 90, dt: -0.22, p: 0.89, cy: 0.9, what: 'Spike, maximum T5' }
  ];

  function binLabel(p) {
    const lo = Math.min(9, Math.floor(p * 10)) * 10;
    return lo + '-' + (lo + 10) + '%';
  }

  // Builds the demo batch in the same file format the packager writes, so it runs through parseBatch.
  function makeDemoBatch(view) {
    const fs = 200, clipS = 12, eventS = 6;
    const research = view === 'research';
    const channels = SCALP.concat(['EKG'], research ? ['SQ D-C', 'SQ P-C'] : []);
    const patients = ['SYN-01', 'SYN-02', 'SYN-03'];
    const clips = DEMO_SPECS.map((spec, i) => {
      const sig = synthClip(spec, 9001 + i * 7919, fs, clipS, eventS);
      const rows = sig.scalp.concat([sig.ekg], research ? sig.sq : []);
      const enc = encodeClip(rows);
      const id = 'd' + cyrb32('demo-clip-' + i).toString(16).padStart(8, '0').slice(0, 7);
      const clip = { id, event_s: eventS, scale: enc.scale, data: enc.data };
      if (research) {
        const hh = 9 + ((i * 5) % 12), mm = (i * 17) % 60, ss = (i * 29) % 60;
        clip.meta = {
          Patient: patients[i % 3],
          'Window start': 'Day ' + (1 + (i % 3)) + ', ' + String(hh).padStart(2, '0') + ':' + String(mm).padStart(2, '0') + ':' + String(ss).padStart(2, '0'),
          'MORGOTH p': spec.p.toFixed(2),
          Bin: binLabel(spec.p),
          Planted: spec.what
        };
        const spikeT = eventS + (spec.dt || 0);
        clip.markers = [{ t: eventS, kind: 'morgoth', label: 'MORGOTH ' + spec.p.toFixed(2) }];
        if (spec.cy) clip.markers.push({ t: spikeT + 0.02, kind: 'cyclops', label: 'Cyclops ' + spec.cy.toFixed(2) });
      }
      return clip;
    });
    return {
      format: FORMAT, version: FORMAT_VERSION, batch_id: 'demo', mode: research ? 'research' : 'rater',
      synthetic: true, created: '2026-10-09T00:00:00Z', fs, clip_s: clipS, pad_s: 1, event_s: eventS,
      candidate_s: 1, channels, groups: research ? { EKG: ['EKG'], SubQ: ['SQ D-C', 'SQ P-C'] } : { EKG: ['EKG'] },
      order_seed: 77, answers: ['yes', 'no'], notes: 'Synthetic EEG generated in the browser.', clips
    };
  }

  // B, T and A pick the montage (double banana, transverse, average reference), in a blinded Yes/No batch and in
  // the time-alignment review. Returns the montage for a keydown's e.key, either case, or null.
  const MONTAGE_KEYS = { b: 'banana', t: 'transverse', a: 'average' };
  function montageKey(key) {
    if (typeof key !== 'string' || key.length !== 1) return null;
    return MONTAGE_KEYS[key.toLowerCase()] || null;
  }

  // Keys in a blinded Yes/No batch (1.6.3; arrows from 1.7.5). Returns the action for a keydown's e.key, or null
  // when the key does nothing. Y and N answer and move on; B, T, A switch the montage; Up and Down step the
  // sensitivity one rung (-1 = fewer uV/mm, larger traces); Left and Right move one clip without answering;
  // S and D toggle the admin and superadmin views, only when the server sent an admin table.
  const SIMPLE_ARROWS = { ArrowUp: ['sens', -1], ArrowDown: ['sens', 1], ArrowLeft: ['go', -1], ArrowRight: ['go', 1] };
  function simpleKey(key, adminAvailable) {
    if (typeof key !== 'string') return null;
    if (SIMPLE_ARROWS[key]) return { act: SIMPLE_ARROWS[key][0], value: SIMPLE_ARROWS[key][1] };
    if (key.length !== 1) return null;
    const k = key.toLowerCase();
    if (k === 'y') return { act: 'decide', value: 'yes' };
    if (k === 'n') return { act: 'decide', value: 'no' };
    if (montageKey(k)) return { act: 'montage', value: montageKey(k) };
    if (adminAvailable && k === 's') return { act: 'admin', value: 'subq' };
    if (adminAvailable && k === 'd') return { act: 'admin', value: 'strips' };
    return null;
  }

  const Core = {
    VERSION, FORMAT, FORMAT_VERSION, SCALP, POS, REGION, MONTAGES, MONTAGE_ORDER, EXPORT_COLUMNS, DEMO_SPECS, parseAdmin,
    normalizeName, sideOf, locationSummary, b64ToBytes, bytesToB64, decodeClip, encodeClip, parseBatch, decryptBatchBytes,
    clipSignals, deriveRows, designFilters, filtfilt, applyFilters, cyrb32, mulberry32, shuffledOrder,
    raterSeed, csvCell, toCSV, parseCSV, exportRows, importRows, spikeShape, synthClip, makeDemoBatch, binLabel, simpleKey,
    MONTAGE_KEYS, montageKey
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = Core;
  else root.Core = Core;
})(typeof globalThis !== 'undefined' ? globalThis : this);
