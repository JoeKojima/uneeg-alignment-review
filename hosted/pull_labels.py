#!/usr/bin/env python3
"""Copies the labels saved on the hosted review site into the labels folder on Borel.

    python hosted/pull_labels.py --labels review_labels            (the site address make_site.py remembered)
    python hosted/pull_labels.py --labels review_labels --status   (each rater's progress, nothing written)

Uses your own link (the token in <labels>/.token), so run it where the labels folder is. It writes
labels_<batch>_<RATER>.csv, the same files serve_review.py saves, so merge_labels.py and report_labels.py read
them as they are. A file that changes is first copied to <labels>/history/. A local file with more answers
than the site's copy is kept (it was labelled through the Borel tunnel) unless --force. Standard library only.
"""
import argparse
import csv
import datetime as dt
import io
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import serve_review as sr  # noqa: E402

NAME_RE = re.compile(r"^labels_([A-Za-z0-9_-]{1,40})_([A-Z0-9_-]{1,12})\.csv$")


def get(url, token, what):
    req = urllib.request.Request(url, headers={"X-Review-Token": token, "User-Agent": "ied-review-pull/1.6"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 403:
            sys.exit("the site does not know your link. If you made new links (--new-token), run make_site.py and "
                     "push again; otherwise check the address.")
        sys.exit(f"the site answered {e.code} for {what}: {e.read().decode('utf-8', 'replace').strip()[:200]}")
    except urllib.error.URLError as e:
        sys.exit(f"could not reach the site ({e.reason}). Check the address and that this computer can reach it.")


def answered(text):
    return sum(1 for r in csv.DictReader(io.StringIO(text)) if r.get("decision"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", required=True, help="the labels folder (holds your link token)")
    ap.add_argument("--site", help="the site's address, e.g. https://ied-review.vercel.app (default: the one "
                                   "make_site.py --url remembered)")
    ap.add_argument("--status", action="store_true", help="print progress and exit")
    ap.add_argument("--force", action="store_true", help="replace local files even when they have more answers")
    args = ap.parse_args(argv)

    site = args.site
    if not site and os.path.exists(os.path.join(args.labels, ".site_url")):
        with open(os.path.join(args.labels, ".site_url"), encoding="utf-8") as f:
            site = f.read().strip()
    if not site:
        sys.exit("give --site https://... (or run make_site.py --url once so it is remembered)")
    site = site.rstrip("/")
    tok_path = os.path.join(args.labels, ".token")
    if not os.path.exists(tok_path):
        sys.exit(f"no link token in {args.labels}; give the labels folder make_site.py used")
    with open(tok_path, encoding="ascii") as f:
        token = f.read().strip()

    batches = get(site + "/api/batches", token, "the batch list").get("batches", [])
    if args.status:
        raters = sorted({r for b in batches for r in b.get("progress", {})})
        print(f"{'batch':<10} {'clips':>5}  {'shared':<6}" + "  ".join(f"{r:>8}" for r in raters))
        for b in batches:
            p = b.get("progress", {})
            print(f"{b['batch_id']:<10} {b['n_clips']:>5}  {'yes' if b.get('shared') else '':<6}" +
                  "  ".join(f"{p.get(r, 0):>8}" for r in raters))
        return

    hist = os.path.join(args.labels, "history")
    os.makedirs(hist, mode=0o700, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    new = same = kept = 0
    for b in batches:
        if not any(b.get("progress", {}).values()):
            continue
        q = urllib.parse.urlencode({"batch": b["batch_id"]})
        files = get(f"{site}/api/export?{q}", token, f"batch {b['batch_id']}").get("files", {})
        for name, text in sorted(files.items()):
            m = NAME_RE.match(name)
            if not m:
                print(f"skipped {name!r}: not a labels file name")
                continue
            if m.group(1) != b["batch_id"]:  # each batch is asked for on its own
                continue
            if text.split("\n", 1)[0].strip().lstrip("﻿").split(",") not in sr.ACCEPTED_COLUMNS:
                print(f"skipped {name}: columns do not match the viewer's export format")
                continue
            path = os.path.join(args.labels, name)
            if os.path.exists(path):
                with open(path, encoding="utf-8", newline="") as f:
                    local = f.read()
                if local == text:
                    same += 1
                    continue
                if answered(local) > answered(text) and not args.force:
                    print(f"kept {name}: this copy has {answered(local)} answers, the site's {answered(text)} "
                          f"(--force replaces it)")
                    kept += 1
                    continue
                shutil.copy2(path, os.path.join(hist, f"{name[:-4]}_{stamp}_before_pull.csv"))
            tmp = path + ".tmp"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            os.replace(tmp, path)
            new += 1
            print(f"pulled {name}: {answered(text)} answers")
    print(f"done: {new} file(s) written, {same} already up to date, {kept} kept; labels are in {args.labels}")


if __name__ == "__main__":
    main()
