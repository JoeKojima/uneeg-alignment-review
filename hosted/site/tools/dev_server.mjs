// A local stand-in for Vercel, for a preview before pushing and for the kit's tests: serves public/, applies
// the rewrites and headers in vercel.json, and answers /api/ with api/review.mjs. Labels go to a JSON file.
//
//   node tools/dev_server.mjs --port 8780 --store /tmp/ied-site-labels
//
// Listens on 127.0.0.1 only.
import { createServer } from 'node:http';
import { readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const arg = (name, dflt) => { const i = process.argv.indexOf(name); return i > 0 ? process.argv[i + 1] : dflt; };
const port = Number(arg('--port', 8780));
const env = Object.assign({}, process.env, { LABELS_STORE_DIR: arg('--store', path.join(root, '.vercel', 'dev-labels')) });
const config = JSON.parse(readFileSync(path.join(root, 'vercel.json'), 'utf8'));
const { respond } = await import(path.join(root, 'api', 'review.mjs'));

// "/api/batch/:id" -> /^\/api\/batch\/([^/]+)$/ with names ['id']; "/(.*)" -> /^\/(.*)$/ (every path; before
// kit 1.8.4 the dot was escaped, so the headers reached "/" only in this preview, never public/manual.html)
function pattern(source) {
  const names = [];
  const re = source.split('(.*)').map(part => part.replace(/[.+?^${}|[\]\\]/g, '\\$&')
    .replace(/:(\w+)/g, (_m, n) => { names.push(n); return '([^/]+)'; })).join('(.*)');
  return { re: new RegExp('^' + re + '$'), names };
}
const rewrites = (config.rewrites || []).map(r => Object.assign(pattern(r.source), r));
const headerRules = (config.headers || []).map(h => Object.assign(pattern(h.source), h));
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml' };

createServer(async (req, res) => {
  const url = new URL(req.url, `http://127.0.0.1:${port}`);
  for (const h of headerRules) if (h.re.test(url.pathname)) for (const { key, value } of h.headers) res.setHeader(key, value);
  let target = url;
  for (const r of rewrites) {
    const m = url.pathname.match(r.re);
    if (!m) continue;
    let dest = r.destination;
    r.names.forEach((n, i) => { dest = dest.replace(':' + n, encodeURIComponent(decodeURIComponent(m[i + 1]))); });
    target = new URL(dest, url);
    for (const [k, v] of url.searchParams) if (!target.searchParams.has(k)) target.searchParams.set(k, v);
    break;
  }
  if (target.pathname === '/api/review') {
    const chunks = [];
    for await (const c of req) chunks.push(c);
    const request = new Request(target, { method: req.method, headers: req.headers,
      body: ['GET', 'HEAD'].includes(req.method) ? undefined : Buffer.concat(chunks) });
    const out = await respond(request, env, root);
    for (const [k, v] of out.headers) res.setHeader(k, v);
    res.writeHead(out.status);
    res.end(Buffer.from(await out.arrayBuffer()));
    return;
  }
  if (target.pathname.startsWith('/api/')) { res.writeHead(404); res.end('Not found.\n'); return; }
  const rel = target.pathname === '/' ? 'index.html' : decodeURIComponent(target.pathname).replace(/^\/+/, '');
  const file = path.join(root, 'public', rel);
  if (!file.startsWith(path.join(root, 'public') + path.sep) || !statSync(file, { throwIfNoEntry: false })?.isFile()) {
    res.writeHead(404); res.end('Not found.\n'); return;
  }
  res.setHeader('Content-Type', TYPES[path.extname(file)] || 'application/octet-stream');
  res.writeHead(200);
  res.end(readFileSync(file));
}).listen(port, '127.0.0.1', () => console.log(`review site preview on http://127.0.0.1:${port}/ (labels in ${env.LABELS_STORE_DIR})`));
