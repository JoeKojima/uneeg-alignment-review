// Where labels are kept. On Vercel: an Upstash Redis database, added from the Vercel Marketplace, which sets
// KV_REST_API_URL and KV_REST_API_TOKEN (or UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN). For a local
// test: a JSON file in LABELS_STORE_DIR. Both answer run([[command, ...args], ...]) with one result per command,
// like a Redis pipeline. Only GET, SET (with NX), HSET, HGETALL and HKEYS are used.
import { mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import path from 'node:path';

export function openStore(env) {
  const url = env.KV_REST_API_URL || env.UPSTASH_REDIS_REST_URL;
  const token = env.KV_REST_API_TOKEN || env.UPSTASH_REDIS_REST_TOKEN;
  if (url && token) return upstash(url, token);
  if (env.LABELS_STORE_DIR) return fileStore(env.LABELS_STORE_DIR);
  return {
    async run() {
      throw new Error('no labels database: connect Upstash Redis to this Vercel project (README, step 4)');
    }
  };
}

function upstash(url, token) {
  const endpoint = url.replace(/\/+$/, '') + '/pipeline';
  return {
    async run(cmds) {
      if (!cmds.length) return [];
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify(cmds)
      });
      if (!res.ok) throw new Error(`labels database answered ${res.status}`);
      const out = await res.json();
      if (!Array.isArray(out) || out.length !== cmds.length) throw new Error('labels database gave an unexpected answer');
      return out.map((r, i) => {
        if (r && r.error) throw new Error(`labels database refused ${cmds[i][0]}: ${r.error}`);
        return r ? r.result : null;
      });
    }
  };
}

// For tests and a local preview only: one process, one JSON file.
function fileStore(dir) {
  mkdirSync(dir, { recursive: true });
  const file = path.join(dir, 'labels-store.json');
  const load = () => { try { return JSON.parse(readFileSync(file, 'utf8')); } catch (e) { return { str: {}, hash: {} }; } };
  return {
    async run(cmds) {
      const db = load();
      const out = cmds.map(([cmd, key, ...args]) => {
        switch (cmd) {
          case 'GET': return key in db.str ? db.str[key] : null;
          case 'SET':
            if (args.includes('NX') && key in db.str) return null;
            db.str[key] = String(args[0]);
            return 'OK';
          case 'HSET': {
            const h = db.hash[key] || (db.hash[key] = {});
            let added = 0;
            for (let i = 0; i + 1 < args.length; i += 2) { if (!(args[i] in h)) added++; h[args[i]] = String(args[i + 1]); }
            return added;
          }
          case 'HGETALL': return Object.entries(db.hash[key] || {}).flat();
          case 'HKEYS': return Object.keys(db.hash[key] || {});
          default: throw new Error(`unsupported command ${cmd}`);
        }
      });
      writeFileSync(file + '.tmp', JSON.stringify(db));
      renameSync(file + '.tmp', file);
      return out;
    }
  };
}
