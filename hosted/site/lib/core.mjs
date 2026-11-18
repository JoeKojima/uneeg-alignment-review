// The review site's server logic: the same API serve_review.py gives on Borel, answered from the batch
// files in this repo and a small labels database. make_site.py writes content/site.json; nothing here
// knows which recordings the clips came from.
import { createHash, scryptSync, timingSafeEqual } from 'node:crypto';
import { readFileSync } from 'node:fs';
import path from 'node:path';

// Must match EXPORT_COLUMNS in the viewer and in serve_review.py.
export const EXPORT_COLUMNS = ['batch_id', 'clip_id', 'rater', 'order_index', 'decision', 'max_electrodes', 'side',
  'region', 'generalized', 'artifact', 'other_discharges_s', 'note', 'seconds_viewed', 'labeled_at',
  'order_seed', 'viewer_version', 'view'];
// labels saved by viewers before kit 1.7, which had no view column, are still accepted
const OLD_COLUMNS = EXPORT_COLUMNS.slice(0, -1);
const RATER_RE = /^[A-Z0-9_-]{1,12}$/;
const BATCH_RE = /^[A-Za-z0-9_-]{1,40}$/;
// A 100-clip labels file is about 15 KB; this keeps every save far below Vercel's 4.5 MB body limit and
// the labels database's request limit (the file is stored twice per save: current copy and history)
export const MAX_LABELS_BYTES = 512 * 1024;
const HISTORY_EVERY_MIN = 10;
const SITE_FORMAT = 'ied-review-site';

export function loadSite(root) {
  const site = JSON.parse(readFileSync(path.join(root, 'content', 'site.json'), 'utf8'));
  if (site.format !== SITE_FORMAT) throw new Error('content/site.json is not a review site description');
  site.byId = new Map(site.batches.map(b => [b.batch_id, Object.assign({}, b, { clipIds: new Set(b.clip_ids) })]));
  site.root = root;
  return site;
}

const sha256 = s => createHash('sha256').update(String(s)).digest('hex');

// The results page has its own sign-in: a username and password sent by the browser's own login box (HTTP
// Basic, over HTTPS), checked against the scrypt hash make_site.py put in site.json. Link tokens don't open it.
export function resultsLogin(site, authorization) {
  const login = site.results_login;
  const m = /^Basic\s+([A-Za-z0-9+/=]+)\s*$/i.exec(authorization || '');
  if (!login || !m) return false;
  const text = Buffer.from(m[1], 'base64').toString('utf8');
  const cut = text.indexOf(':');
  if (cut < 0) return false;
  const want = Buffer.from(login.hash, 'hex');
  const got = scryptSync(text.slice(cut + 1), Buffer.from(login.salt, 'hex'), want.length,
    { N: login.n, r: login.r, p: login.p, maxmem: 64 * 1024 * 1024 });
  return timingSafeEqual(got, want) && text.slice(0, cut).trim().toUpperCase() === login.user;
}

function resultsPage(site, authorization) {
  if (!site.results_login) return text(404, 'This site has no results page.\n');
  if (!resultsLogin(site, authorization)) {
    return { status: 401, type: 'text/plain; charset=utf-8', body: 'Sign in with the results username and password.\n',
      headers: { 'WWW-Authenticate': 'Basic realm="IED review results", charset="UTF-8"' } };
  }
  const res = JSON.parse(readFileSync(path.join(site.root, 'content', 'results.json'), 'utf8'));
  const updated = String(res.created).replace(/[<>&"]/g, '');
  return { status: 200, type: 'text/html; charset=utf-8', body: res.html.split('{{updated}}').join(updated) };
}

// "" for the owner's link, the rater's initials for a rater's link, null for anything else.
export function identity(site, token) {
  if (!token) return null;
  const given = Buffer.from(sha256(token), 'hex');
  let who = null;
  for (const [name, hash] of Object.entries(site.tokens)) {
    const want = Buffer.from(hash, 'hex');
    if (want.length === given.length && timingSafeEqual(want, given)) who = name;
  }
  return who;
}

// Whether a rater may see a batch. The owner and studies without a split see everything.
export function visible(site, who, bid) {
  const b = site.byId.get(bid);
  if (!b) return false;
  if (!who || !site.assignment) return true;
  return !!b.shared || (site.assignment.by_rater[who] || []).includes(bid);
}

// The staged-study fields of a batch (kit 1.8), as the listing gives them to the viewer.
function stageInfo(b) {
  const out = {};
  if (b.stage) { out.stage = Number(b.stage); out.stage_name = String(b.stage_name || ''); }
  if (b.view === 'subq' || b.view === 'both') out.view = b.view;
  if (b.training) out.training = true;
  return out;
}

// Batch IDs a rater cannot open yet: in a staged study each part opens once the rater has answered every clip of
// the earlier parts they can see (as serve_review.locked_batches). answered: Map(batch id -> count).
export function lockedBatches(site, who, answered) {
  const locked = new Set();
  if (!who) return locked;
  const list = site.batches.filter(x => x.stage && visible(site, who, x.batch_id));
  const stages = [...new Set(list.map(x => Number(x.stage)))].sort((a, b) => a - b);
  let open = true;
  for (const st of stages) {
    const ids = list.filter(x => Number(x.stage) === st);
    if (!open) { ids.forEach(x => locked.add(x.batch_id)); continue; }
    if (ids.some(x => (answered.get(x.batch_id) || 0) < x.n_clips)) open = false;
  }
  return locked;
}

async function lockedFor(site, store, who) {
  if (!who || !site.batches.some(x => x.stage)) return new Set();
  const list = site.batches.filter(x => x.stage && visible(site, who, x.batch_id));
  // HGETALL, like the listing: the store answers only GET, SET, HSET, HGETALL and HKEYS
  const got = await store.run(list.map(x => ['HGETALL', keyOf(site, 'progress', x.batch_id)]));
  const mine = flat => { for (let k = 0; k + 1 < (flat || []).length; k += 2) if (flat[k] === who) return Number(flat[k + 1]); return 0; };
  return lockedBatches(site, who, new Map(list.map((x, i) => [x.batch_id, mine(got[i])])));
}

const LOCKED_MSG = 'This part opens once you have answered every clip in the parts before it.\n';

export function parseCSV(text) {
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
  return rows;
}

// Checks an exported-labels CSV the way serve_review.py does. Returns { answered, total } or throws.
export function checkLabels(batch, rater, text) {
  const head = text.split('\n', 1)[0].trim().replace(/^﻿/, '').split(',');
  const cols = head.join(',') === EXPORT_COLUMNS.join(',') ? EXPORT_COLUMNS :
    head.join(',') === OLD_COLUMNS.join(',') ? OLD_COLUMNS : null;
  if (!cols) throw new Error("columns do not match the viewer's export format");
  const rows = parseCSV(text).slice(1);
  let answered = 0;
  for (const cells of rows) {
    const r = Object.fromEntries(cols.map((c, i) => [c, cells[i] != null ? cells[i] : '']));
    if (r.batch_id !== batch.batch_id || r.rater.toUpperCase() !== rater) throw new Error('a row names another batch or rater');
    if (!batch.clipIds.has(r.clip_id)) throw new Error('a row names a clip that is not in this batch');
    if (r.decision && !batch.answers.includes(r.decision)) throw new Error(`a row answers '${r.decision}', which this study does not offer`);
    if (r.decision) answered++;
  }
  return { answered, total: rows.length };
}

const keyOf = (site, ...parts) => ['ied', site.study_id, ...parts].join(':');
const text = (status, body) => ({ status, type: 'text/plain; charset=utf-8', body });
const json = (status, obj) => ({ status, type: 'application/json', body: JSON.stringify(obj) });

function stamp10(d) {
  const p = n => String(n).padStart(2, '0');
  const m = Math.floor(d.getUTCMinutes() / HISTORY_EVERY_MIN) * HISTORY_EVERY_MIN;
  return `${d.getUTCFullYear()}${p(d.getUTCMonth() + 1)}${p(d.getUTCDate())}-${p(d.getUTCHours())}${p(m)}`;
}

/**
 * One request. req = { method, route: ['batches'] | ['batch', id] | ['labels', batch, rater] | ['export'],
 * token, body (text), query, authorization }. store is from store.mjs. Returns { status, type, body, headers? }.
 */
// the owner, and raters the owner gave the admin view (make_site.py --admin-raters)
function canAdmin(site, who) {
  return who === '' || (Array.isArray(site.admin_raters) && site.admin_raters.includes(who));
}

export async function handle(site, store, req) {
  if (req.route[0] === 'results' && req.route.length === 1 && (req.method === 'GET' || req.method === 'HEAD')) {
    return resultsPage(site, req.authorization);
  }
  const who = identity(site, req.token);
  if (who === null) return text(403, 'Forbidden. Open your own link, the one that ends in ?token=...\n');
  const [what, a, b] = req.route;
  const method = req.method === 'HEAD' ? 'GET' : req.method;

  if (method === 'GET' && what === 'batches' && req.route.length === 1) {
    const list = site.batches.filter(x => visible(site, who, x.batch_id));
    const prog = await store.run(list.map(x => ['HGETALL', keyOf(site, 'progress', x.batch_id)]));
    const mine = new Map();
    const items = list.map((x, i) => {
      let p = {};
      const flat = prog[i] || [];
      for (let k = 0; k + 1 < flat.length; k += 2) p[flat[k]] = Number(flat[k + 1]);
      if (who) p = { [who]: p[who] || 0 };  // a rater sees only their own progress
      if (who) mine.set(x.batch_id, p[who]);
      return Object.assign({ file: x.file, batch_id: x.batch_id, mode: 'rater', synthetic: !!x.synthetic,
        n_clips: x.n_clips, shared: !!x.shared, progress: p }, stageInfo(x));
    });
    const locked = lockedBatches(site, who, mine);
    for (const it of items) if (locked.has(it.batch_id)) it.locked = true;
    const listing = { batches: items, rater: who || null };
    if (Number(site.tolerance_s) > 0) listing.tolerance_s = Number(site.tolerance_s);
    // the rater manuals, public/manual_scalp.html and public/manual_uneeg.html (kit 1.8.4)
    const manuals = {};
    for (const k of ['scalp', 'uneeg']) {
      if (site.manuals && site.manuals[k] === `manual_${k}.html`) manuals[k] = `manual_${k}.html`;
    }
    if (Object.keys(manuals).length) listing.manuals = manuals;
    // the owner's link is told which initials have their own link, so it never labels under them (1.7.7)
    if (who === '') listing.rater_links = Object.keys(site.tokens).filter(k => k).sort();
    if (canAdmin(site, who)) listing.admin = site.batches.some(x => x.admin_file);
    return json(200, listing);
  }

  if (method === 'GET' && what === 'batch' && req.route.length === 2) {
    if (!BATCH_RE.test(a || '') || !visible(site, who, a)) return text(404, 'No such batch.\n');
    if ((await lockedFor(site, store, who)).has(a)) return text(403, LOCKED_MSG);
    const bytes = readFileSync(path.join(site.root, 'content', 'batches', site.byId.get(a).file));
    return { status: 200, type: 'application/octet-stream', body: bytes };
  }

  if (method === 'GET' && what === 'admin' && req.route.length === 2) {
    if (!canAdmin(site, who)) return text(403, 'Only the owner link opens the admin view.\n');
    const b = BATCH_RE.test(a || '') && visible(site, who, a) ? site.byId.get(a) : null;
    if (!b || !b.admin_file) return text(404, 'No admin view for this batch.\n');
    const bytes = readFileSync(path.join(site.root, 'content', 'admin', b.admin_file));
    return { status: 200, type: 'application/octet-stream', body: bytes };
  }

  if (what === 'labels' && req.route.length === 3) {
    const rater = String(b || '').toUpperCase();
    if (method === 'GET') {
      if (!site.byId.has(a) || !RATER_RE.test(rater) || !visible(site, who, a)) return text(404, 'No such batch or rater.\n');
      if (who && rater !== who) return text(403, 'This link belongs to another rater.\n');
      const got = (await store.run([['GET', keyOf(site, 'labels', a, rater)]]))[0];
      return got == null ? text(404, 'No labels saved yet.\n') : { status: 200, type: 'text/csv; charset=utf-8', body: got };
    }
    if (method === 'PUT' || method === 'POST') {
      if (!RATER_RE.test(rater)) return text(400, 'Initials must be 1 to 12 letters or digits.\n');
      if (who && rater !== who) return text(403, 'This link belongs to another rater.\n');
      if (who === '' && Object.prototype.hasOwnProperty.call(site.tokens, rater)) {
        return text(403, `Not saved: ${rater} has their own link. On your link, use your own initials (the Rater button).\n`);
      }
      const batch = site.byId.get(a);
      if (!batch) return text(400, 'Not saved: unknown batch.\n');
      if (!visible(site, who, a)) return text(403, 'This batch is not one of yours.\n');
      if ((await lockedFor(site, store, who)).has(a)) return text(403, 'Not saved: t' + LOCKED_MSG.slice(1));
      const body = req.body || '';
      if (!body.length || Buffer.byteLength(body) > MAX_LABELS_BYTES) return text(413, 'Labels file is empty or too large.\n');
      let res;
      try { res = checkLabels(batch, rater, body); } catch (e) { return text(400, `Not saved: ${e.message}.\n`); }
      const now = new Date();
      await store.run([
        ['SET', keyOf(site, 'labels', a, rater), body],
        ['HSET', keyOf(site, 'progress', a), rater, String(res.answered)],
        // a copy at most every 10 minutes, like history/ on Borel
        ['SET', keyOf(site, 'history', a, rater, stamp10(now)), body, 'NX'],
      ]);
      return json(200, { saved: true, answered: res.answered, total: res.total, at: now.toISOString() });
    }
  }

  if (method === 'GET' && what === 'export' && req.route.length === 1) {
    if (who !== '') return text(403, 'Only the owner link can export labels.\n');
    const want = req.query && req.query.batch ? [req.query.batch] : site.batches.map(x => x.batch_id);
    const ids = want.filter(x => site.byId.has(x));
    const raters = await store.run(ids.map(x => ['HKEYS', keyOf(site, 'progress', x)]));
    const pairs = [];
    ids.forEach((x, i) => (raters[i] || []).forEach(r => pairs.push([x, r])));
    const got = await store.run(pairs.map(([x, r]) => ['GET', keyOf(site, 'labels', x, r)]));
    const files = {};
    pairs.forEach(([x, r], i) => { if (got[i] != null) files[`labels_${x}_${r}.csv`] = got[i]; });
    return json(200, { files });
  }

  return text(404, 'Not found.\n');
}

// The route from a request URL: /api/batch/b01, or the query Vercel's rewrites add (?route=batch&id=b01).
export function routeOf(url) {
  const q = url.searchParams;
  const r = q.get('route');
  if (r === 'batches' || r === 'export' || r === 'results') return [r];
  if (r === 'batch') return ['batch', q.get('id') || ''];
  if (r === 'admin') return ['admin', q.get('id') || ''];
  if (r === 'labels') return ['labels', q.get('batch') || '', q.get('rater') || ''];
  const parts = url.pathname.split('/').filter(Boolean).map(decodeURIComponent);
  return parts[0] === 'api' ? parts.slice(1) : parts;
}
