# IED clip review site

Rater links for the IED clip review. Each rater opens their own link and labels blinded 10 s EEG clips
(Yes or No); answers save to a small labels database as they go. Written by `make_site.py` from the IED
review kit; do not edit files here by hand, run `make_site.py` again instead.

## What is here, and what never is

| Path | What it is |
|---|---|
| `public/index.html` | The viewer (the same page the Borel server gives) |
| `api/review.mjs`, `lib/` | The one server function: checks the link, sends batches, saves labels |
| `content/batches/*.json.gz` | Blinded rater batches: the signal and random clip IDs, nothing else |
| `content/admin/*.json.gz` | Only with `make_site.py --admin`: the owner-only admin view of each batch (SubQ channels, implant side, which draw picked each clip, each detector's per-second scores), keyed by the same random clip IDs. Only your own link can fetch it; rater links get 403 |
| `content/site.json` | Batch list, which rater gets which batch, and SHA-256 hashes of the links |
| `content/results.json` | The results page at `/results` (only once `make_site.py --results-login` turned it on): counts and rates, patients as P01, P02, ..., behind its own username and a scrypt hash of its password |
| `tools/` | The upload check (also the pre-commit hook) and a local preview server |

Never here: EDF files, `key.csv`, clip tables, study plans, raw link tokens, label files. Those stay on
Borel. `tools/check_upload.py` refuses a commit that holds any of them, or any text that looks like a
subject ID, HUP number, EEG file name, Borel path or calendar date. Keep this repository private.

## One-time setup

1. GitHub: create a **private** repository under your personal account (Vercel's free plan cannot deploy a
   private repository owned by an organization). Leave it empty: no README, licence or .gitignore.
2. Push this folder to it. The pre-commit hook runs when `git config core.hooksPath .githooks` is set.
3. Vercel: Add New, Project, import the repository. Framework preset: Other. Leave build settings empty.
   Deploy.
4. Labels database: in the Vercel project, Storage (or Marketplace), add **Upstash for Redis** (free plan),
   and connect it to this project with the default environment variable prefix. Then Deployments,
   the latest one, Redeploy, so the function sees the new variables.
5. Your link is `https://<project>.vercel.app/?token=...`; `make_site.py --url https://<project>.vercel.app
   --links` prints it and every rater's.

## Day to day

* New or removed raters, or new links: run `make_site.py` on Borel with `--raters`, `--revoke` or
  `--new-token`, then commit and push. Vercel redeploys on every push to the main branch.
* Labels back on Borel: `python hosted/pull_labels.py --labels <labels folder>`. It writes the same
  `labels_<batch>_<RATER>.csv` files serve_review.py saves.
* Progress: open your own link, or `pull_labels.py --status`.
* Results: `pull_labels.py`, then `make_site.py` again, then commit and push; `/results` shows every answer
  pulled so far. Rater links don't open it.

## Local preview

`node tools/dev_server.mjs --port 8780 --store /tmp/ied-site-labels` serves the site on
http://127.0.0.1:8780 with labels in a local file, so a link can be tried before pushing.
