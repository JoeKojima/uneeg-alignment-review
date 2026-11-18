// The one server function. vercel.json sends /api/batches, /api/batch/<id>, /api/admin/<id> (the owner link
// only), /api/labels/<batch>/<rater> and /api/export here, and every one of those must carry a link token
// (X-Review-Token header or ?token=); /results
// comes here too and asks for the results username and password instead.
import { MAX_LABELS_BYTES, handle, loadSite, routeOf } from '../lib/core.mjs';
import { openStore } from '../lib/store.mjs';

let site = null;
let store = null;

export async function respond(request, env = process.env, root = process.cwd()) {
  const headers = { 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' };
  try {
    site = site || loadSite(root);
    store = store || openStore(env);
    const url = new URL(request.url);
    const token = request.headers.get('x-review-token') || url.searchParams.get('token') || '';
    let body = '';
    if (request.method === 'PUT' || request.method === 'POST') {
      if (Number(request.headers.get('content-length') || 0) > MAX_LABELS_BYTES) {
        return new Response('Labels file is too large.\n', { status: 413, headers });
      }
      body = await request.text();
    }
    const query = Object.fromEntries(url.searchParams);
    const out = await handle(site, store, { method: request.method, route: routeOf(url), token, body, query,
      authorization: request.headers.get('authorization') || '' });
    Object.assign(headers, out.headers || {});
    headers['Content-Type'] = out.type;
    return new Response(request.method === 'HEAD' ? null : out.body, { status: out.status, headers });
  } catch (e) {
    console.error(e);  // shown only in the Vercel project's logs
    headers['Content-Type'] = 'text/plain; charset=utf-8';
    return new Response('The review site could not answer just now. Labels stay in this browser until it can.\n',
      { status: 503, headers });
  }
}

export default { fetch: request => respond(request) };
