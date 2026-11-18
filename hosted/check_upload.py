#!/usr/bin/env python3
"""Checks a review-site folder before anything in it is committed or pushed.

    python3 tools/check_upload.py .                      (the site's pre-commit hook runs this)
    python3 check_upload.py SITE --labels review_labels   (make_site.py also checks no raw link token leaked)

The site may hold only the viewer, the server code, content/site.json, blinded rater batches, the owner-only
admin view (content/admin/, SubQ signal, implant sides and detector scores), the results page
(content/results.json, counts and rates only) and the rater manuals (public/manual_scalp.html and
public/manual_uneeg.html, text only). It fails on:
  * any other file, and by name on EEG files, key.csv, clip tables, study plans, *id_map* and token files;
  * a batch that is not a blinded rater batch: unknown keys, research mode, notes, clip details beyond the
    signal, or a clip ID that is not a kit ID;
  * an admin file with anything beyond SubQ signal, implant side (L or R), draw names and per-second scores;
  * text that looks like a source: EMU or UNEEG subject IDs, HUP numbers, EDF names, Borel paths, or a
    calendar date anywhere but a 'created' stamp;
  * site.json token entries that are not SHA-256 hashes, a results login that is not a salted scrypt hash, or
    (with --labels) a raw link token anywhere;
  * a results page with anything but its page, or with a patient's study ID, a path or a date in it;
  * a batch over 4,000,000 bytes (Vercel answers at most 4.5 MB) or a site over 230 MB of batches and admin files.

Exit status 0 when the folder is fine, 1 with one line per problem otherwise. Standard library only.
"""
import argparse
import gzip
import json
import os
import re
import sys

SKIP_DIRS = {".git", ".vercel", "node_modules"}
FIXED_FILES = {"public/index.html", "api/review.mjs", "lib/core.mjs", "lib/store.mjs", "content/site.json",
               "tools/check_upload.py", "tools/dev_server.mjs", ".githooks/pre-commit", "vercel.json",
               "package.json", "README.md", ".gitignore"}
BATCH_PATH_RE = re.compile(r"^content/batches/[A-Za-z0-9_-]{1,40}\.json\.gz$")
ADMIN_PATH_RE = re.compile(r"^content/admin/[A-Za-z0-9_-]{1,40}\.json\.gz$")
ADMIN_KEYS = {"format", "version", "batch_id", "fs", "clip_s", "clips"}
ADMIN_CLIP_KEYS = {"id", "implant_side", "drawn_by", "strips", "subq", "error", "virtual", "scalp"}
ELECTRODES = {"Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz",
              "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2"}
STRIP_KEYS = {"name", "kind", "len", "secs"}
NAME_RE = re.compile(r"^[A-Za-z0-9 _-]{1,24}$")
FORBIDDEN_RE = re.compile(r"(\.(edf|bdf|edf\.gz)$|(^|/)(key|clips|seconds)\.csv$|(^|/)plan\.json$|id_map|"
                          r"(^|/)\.token$|(^|/)\.raters\.json$|(^|/)\.assignment\.json$|(^|/)labels_.*\.csv$|"
                          r"(^|/)\.results\.json$|results_patients)", re.I)
BATCH_KEYS = {"format", "version", "batch_id", "mode", "synthetic", "created", "packager_version", "fs", "clip_s",
              "pad_s", "event_s", "candidate_s", "channels", "groups", "order_seed", "dc_removed", "display",
              "answers", "notes", "clips", "question", "part", "stage", "stage_name", "view", "training"}
CLIP_KEYS = {"id", "event_s", "scale", "data", "bad"}
# a training example (kit 1.8) also carries the expert's answer and a short note, shown after the rater answers
TRAINING_CLIP_KEYS = {"ref", "ref_note", "ref_type"}
REF_TYPES = ("generalized", "focal left", "focal right")  # the kind of spike an example shows (kit 1.8.5)
MAX_REF_NOTE = 300
SITE_KEYS = {"format", "version", "study_id", "created", "kit_version", "batches", "assignment", "tokens",
             "results_login", "admin_raters", "tolerance_s", "manuals"}
LOGIN_KEYS = {"user", "salt", "hash", "n", "r", "p"}
RESULTS_KEYS = {"format", "version", "created", "html"}
SITE_BATCH_KEYS = {"batch_id", "file", "n_clips", "shared", "synthetic", "answers", "sha256", "clip_ids", "admin_file",
                   "stage", "stage_name", "view", "training"}
CLIP_ID_RE = re.compile(r"^c[0-9a-f]{7}$")
CHANNEL_RE = re.compile(r"^[A-Za-z0-9-]{1,12}$")
SYNTHETIC_NOTE = "Synthetic EEG with planted events."
SOURCE_RES = [
    (re.compile(r"EMU\s*\d{3,}", re.I), "an EMU subject ID"),
    (re.compile(r"\bUS\d{2}[-_]\d{2}\b", re.I), "a UNEEG subject ID"),
    (re.compile(r"\bHUP\s*\d{2,}", re.I), "a HUP number"),
    (re.compile(r"\.edf\b|\.bdf\b", re.I), "an EEG file name"),
    (re.compile(r"/mnt/|/users/|/lcars/|littlab|Human_Data", re.I), "a Borel path"),
]
DATE_RE = re.compile(r"\b(19|20)\d{2}[-/.]?(0[1-9]|1[0-2])[-/.]?(0[1-9]|[12]\d|3[01])\b")
MAX_BATCH_BYTES = 4_000_000
MAX_TOTAL_BYTES = 230_000_000  # a Vercel function holds 250 MB; the rest is the viewer and code
MANUAL_PARTS = ("scalp", "uneeg")
MANUAL_FILES = {f"public/manual_{k}.html" for k in MANUAL_PARTS}


def strings(obj, where, skip=("data",)):
    """(path, key, text) for every string in a JSON value, leaving out the encoded signal."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in skip:
                continue
            yield from strings(v, f"{where}.{k}", skip) if not isinstance(v, str) else [(f"{where}.{k}", k, v)]
            if isinstance(k, str):
                yield (f"{where} (key)", "", k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from strings(v, f"{where}[{i}]", skip) if not isinstance(v, str) else [(f"{where}[{i}]", "", v)]
    elif isinstance(obj, str):
        yield (where, "", obj)


def scan_text(text, where, key=""):
    out = [f"{where}: looks like {what}" for rx, what in SOURCE_RES if rx.search(text)]
    if key != "created" and DATE_RE.search(text):
        out.append(f"{where}: holds a date")
    return out


def check_batch(path, rel):
    problems = []
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) > MAX_BATCH_BYTES:
        problems.append(f"{rel}: {len(raw):,} bytes; a hosted batch must be under {MAX_BATCH_BYTES:,} "
                        f"(plan the study with --batch-size 25)")
    try:
        obj = json.loads(gzip.decompress(raw).decode("utf-8"))
    except Exception:
        return problems + [f"{rel}: not a gzipped batch file"]
    if obj.get("format") != "ied-review-batch":
        return problems + [f"{rel}: not a review batch"]
    extra = set(obj) - BATCH_KEYS
    if extra:
        problems.append(f"{rel}: unexpected batch fields {sorted(extra)}")
    if obj.get("mode") != "rater":
        problems.append(f"{rel}: not a blinded rater batch (mode {obj.get('mode')!r})")
    if obj.get("notes") not in ("", None) and not (obj.get("synthetic") and obj.get("notes") == SYNTHETIC_NOTE):
        problems.append(f"{rel}: the batch carries notes; hosted batches must have none")
    if os.path.basename(rel) != f"{obj.get('batch_id')}.json.gz":
        problems.append(f"{rel}: file name does not match batch_id")
    for ch in list(obj.get("channels") or []) + [c for g in (obj.get("groups") or {}).values() for c in g]:
        if not isinstance(ch, str) or not CHANNEL_RE.match(ch):
            problems.append(f"{rel}: channel name {str(ch)[:20]!r} is not a plain electrode name")
            break
    training = obj.get("training") is True
    for i, clip in enumerate(obj.get("clips") or []):
        extra = set(clip) - CLIP_KEYS - (TRAINING_CLIP_KEYS if training else set())
        if extra:
            problems.append(f"{rel}: clip {i + 1} has fields {sorted(extra)}; rater batches carry only the signal")
        if training:
            if clip.get("ref") not in ("yes", "no"):
                problems.append(f"{rel}: training clip {i + 1} has no expert answer (ref yes or no)")
            if clip.get("ref_type") not in (None,) + REF_TYPES:
                problems.append(f"{rel}: training clip {i + 1} has a spike type that is not one of {', '.join(REF_TYPES)}")
            note = clip.get("ref_note", "")
            if not isinstance(note, str) or len(note) > MAX_REF_NOTE:
                problems.append(f"{rel}: training clip {i + 1} has a note that is not short text")
            else:
                problems += scan_text(note, f"{rel}: clip {i + 1} ref_note")
        if not CLIP_ID_RE.match(str(clip.get("id", ""))):
            problems.append(f"{rel}: clip {i + 1} has an ID that is not a kit clip ID")
        for b in clip.get("bad") or []:
            if not isinstance(b, str) or not CHANNEL_RE.match(b):
                problems.append(f"{rel}: clip {i + 1} lists a bad channel that is not an electrode name")
                break
    head = {k: v for k, v in obj.items() if k != "clips"}
    for where, key, text in strings(head, rel):
        problems += scan_text(text, where, key)
    return problems


def check_admin(path, rel):
    """The owner-only admin view: SubQ signal, implant side, which draw picked each clip, per-second scores."""
    problems = []
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) > MAX_BATCH_BYTES:
        problems.append(f"{rel}: {len(raw):,} bytes; an admin file must be under {MAX_BATCH_BYTES:,}")
    try:
        obj = json.loads(gzip.decompress(raw).decode("utf-8"))
    except Exception:
        return problems + [f"{rel}: not a gzipped admin file"]
    if obj.get("format") != "ied-review-admin":
        return problems + [f"{rel}: not an admin view file"]
    if set(obj) - ADMIN_KEYS:
        problems.append(f"{rel}: unexpected fields {sorted(set(obj) - ADMIN_KEYS)}")
    if os.path.basename(rel) != f"{obj.get('batch_id')}.json.gz":
        problems.append(f"{rel}: file name does not match batch_id")
    for i, clip in enumerate(obj.get("clips") or []):
        where = f"{rel}: clip {i + 1}"
        if set(clip) - ADMIN_CLIP_KEYS:
            problems.append(f"{where} has fields {sorted(set(clip) - ADMIN_CLIP_KEYS)}")
        if not CLIP_ID_RE.match(str(clip.get("id", ""))):
            problems.append(f"{where} has an ID that is not a kit clip ID")
        if clip.get("implant_side") not in (None, "L", "R"):
            problems.append(f"{where}: implant side must be L or R")
        for d in clip.get("drawn_by") or []:
            if not isinstance(d, str) or not NAME_RE.match(d):
                problems.append(f"{where}: a draw name that is not a plain name")
        if clip.get("error") not in (None, "SubQ could not be read"):
            problems.append(f"{where}: an error message beyond the fixed one")
        sc = clip.get("scalp")  # a UNEEG clip's 10-20 channels for the admin view (kit 1.8.5)
        if sc is not None:
            if (not isinstance(sc, dict) or set(sc) - {"channels", "scale", "data", "bad"}
                    or not {"channels", "scale", "data"} <= set(sc)
                    or any(not isinstance(c, str) or not CHANNEL_RE.match(c) for c in sc.get("channels") or [])
                    or any(not isinstance(c, str) or not CHANNEL_RE.match(c) for c in sc.get("bad") or [])):
                problems.append(f"{where}: the 10-20 view must be plain channel names, scales and signal only")
        sq = clip.get("subq")
        if sq is not None:
            if set(sq) != {"channels", "scale", "data"} or any(not isinstance(c, str) or not CHANNEL_RE.match(c)
                                                               for c in sq.get("channels") or []):
                problems.append(f"{where}: SubQ must be plain channel names, scales and signal only")
        if any(not (isinstance(p, list) and len(p) == 2 and all(e in ELECTRODES for e in p))
               for p in clip.get("virtual") or []):
            problems.append(f"{where}: virtual UNEEG must be pairs of 10-20 electrode names")
        for st in clip.get("strips") or []:
            if set(st) - STRIP_KEYS or not NAME_RE.match(str(st.get("name", ""))) or st.get("kind") not in ("scalp", "subq"):
                problems.append(f"{where}: a strip with fields or a name it should not have")
                break
            if any(not (isinstance(x, list) and len(x) == 2 and all(v is None or isinstance(v, (int, float)) for v in x))
                   for x in st.get("secs") or []):
                problems.append(f"{where}: a strip holds something other than times and scores")
                break
    for where, key, text in strings(obj, rel):
        problems += scan_text(text, where, key)
    return problems


def check_site_json(path, rel):
    problems = []
    try:
        with open(path, encoding="utf-8") as f:
            site = json.load(f)
    except Exception:
        return [f"{rel}: not JSON"]
    if site.get("format") != "ied-review-site":
        problems.append(f"{rel}: not a review site description")
    extra = set(site) - SITE_KEYS | {k for b in site.get("batches") or [] for k in b} - SITE_BATCH_KEYS
    if extra:
        problems.append(f"{rel}: unexpected fields {sorted(extra)}")
    for who, h in (site.get("tokens") or {}).items():
        if not re.fullmatch(r"[0-9a-f]{64}", str(h)):
            problems.append(f"{rel}: the link for {who or 'the owner'} is not stored as a SHA-256 hash")
        if who and not re.fullmatch(r"[A-Z0-9_-]{1,12}", who):
            problems.append(f"{rel}: rater {who[:20]!r} is not initials")
    for who in site.get("admin_raters") or []:
        if who not in (site.get("tokens") or {}) or not who:
            problems.append(f"{rel}: admin view given to {str(who)[:20]!r}, who has no rater link")
    login = site.get("results_login")
    if login is not None:
        if (set(login) != LOGIN_KEYS or not re.fullmatch(r"[0-9a-f]{64}", str(login.get("hash")))
                or not re.fullmatch(r"[0-9a-f]{32}", str(login.get("salt"))) or int(login.get("n") or 0) < 2 ** 14
                or not re.fullmatch(r"[A-Z0-9_-]{1,12}", str(login.get("user")))):
            problems.append(f"{rel}: the results login is not a username with a salted scrypt hash")
    manuals = site.get("manuals")
    if manuals is not None:
        if (not isinstance(manuals, dict) or not manuals
                or any(k not in MANUAL_PARTS or v != f"manual_{k}.html" for k, v in manuals.items())):
            problems.append(f"{rel}: manuals must map scalp and/or uneeg to manual_scalp.html / manual_uneeg.html")
        else:
            for k, v in manuals.items():
                if not os.path.isfile(os.path.join(os.path.dirname(os.path.dirname(path)), "public", v)):
                    problems.append(f"{rel}: names the {k} manual but public/{v} is missing")
    tol = site.get("tolerance_s")
    if tol is not None and not (isinstance(tol, (int, float)) and 0 <= tol <= 1):
        problems.append(f"{rel}: tolerance_s is not a number of seconds between 0 and 1")
    for b in site.get("batches") or []:
        bad_ids = [c for c in b.get("clip_ids") or [] if not CLIP_ID_RE.match(str(c))]
        if bad_ids:
            problems.append(f"{rel}: batch {b.get('batch_id')} lists {len(bad_ids)} clip ID(s) that are not kit IDs")
    for where, key, text in strings(site, rel, skip=("tokens", "clip_ids", "sha256", "study_id", "results_login")):
        problems += scan_text(text, where, key)
    return problems


def check_results_json(path, rel):
    try:
        with open(path, encoding="utf-8") as f:
            page = json.load(f)
    except Exception:
        return [f"{rel}: not JSON"]
    problems = []
    if page.get("format") != "ied-review-results":
        problems.append(f"{rel}: not a results page")
    extra = set(page) - RESULTS_KEYS
    if extra:
        problems.append(f"{rel}: unexpected fields {sorted(extra)}")
    for where, key, text in strings(page, rel):
        problems += scan_text(text, where, key)
    return problems


def raw_tokens(labels_dir):
    toks = []
    p = os.path.join(labels_dir, ".token")
    if os.path.exists(p):
        with open(p, encoding="ascii") as f:
            toks.append(f.read().strip())
    p = os.path.join(labels_dir, ".raters.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            toks += list(json.load(f).values())
    return [t for t in toks if len(t) >= 20]


def check(site_dir, labels_dir=None):
    problems, total = [], 0
    tokens = [t.encode("ascii") for t in raw_tokens(labels_dir)] if labels_dir else []
    for top, dirs, files in os.walk(site_dir):
        dirs[:] = sorted(d for d in dirs if not (top == site_dir and d in SKIP_DIRS))
        for name in sorted(files):
            path = os.path.join(top, name)
            rel = os.path.relpath(path, site_dir).replace(os.sep, "/")
            if os.path.islink(path):
                problems.append(f"{rel}: a link; the site holds only plain files")
                continue
            if FORBIDDEN_RE.search(rel):
                problems.append(f"{rel}: must never be in the site (it can tie clips to patients or open links)")
                continue
            if BATCH_PATH_RE.match(rel):
                total += os.path.getsize(path)
                problems += check_batch(path, rel)
            elif ADMIN_PATH_RE.match(rel):
                total += os.path.getsize(path)
                problems += check_admin(path, rel)
            elif rel == "content/site.json":
                problems += check_site_json(path, rel)
            elif rel == "content/results.json":
                problems += check_results_json(path, rel)
            elif rel in MANUAL_FILES:  # the rater manuals (kit 1.8.4): text only, no source or date in them
                with open(path, encoding="utf-8", errors="replace") as f:
                    text = f.read()
                problems += scan_text(text, rel)
                if re.search(r"<script|<iframe|<object|<embed|\bon[a-z]+\s*=|https?://", text, re.I):
                    problems.append(f"{rel}: holds a script, frame or outside link; the manual is plain text")
            elif rel not in FIXED_FILES:
                problems.append(f"{rel}: not a file the site needs; remove it")
                continue
            elif rel in ("public/index.html", "README.md"):
                with open(path, encoding="utf-8", errors="replace") as f:
                    text = f.read()
                problems += [p for p in scan_text(text, rel, "created") if "date" not in p]
            if tokens:
                with open(path, "rb") as f:
                    raw = f.read()
                if rel.endswith(".gz"):
                    raw = gzip.decompress(raw)
                if any(t in raw for t in tokens):
                    problems.append(f"{rel}: holds a raw link token")
    if total > MAX_TOTAL_BYTES:
        problems.append(f"batches and admin files add up to {total / 1e6:.0f} MB; a Vercel function holds at most "
                        f"250 MB, keep "
                        f"the site under {MAX_TOTAL_BYTES / 1e6:.0f} MB")
    if not os.path.isfile(os.path.join(site_dir, "content", "site.json")):
        problems.append("content/site.json is missing; run make_site.py")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("site", help="the site folder (the git repository's top folder)")
    ap.add_argument("--labels", help="labels folder on Borel; also checks no raw link token is in the site")
    args = ap.parse_args(argv)
    problems = check(os.path.abspath(args.site), args.labels)
    if problems:
        print("upload check FAILED; nothing should be committed until these are fixed:", file=sys.stderr)
        for p in problems[:50]:
            print("  " + p, file=sys.stderr)
        if len(problems) > 50:
            print(f"  ... and {len(problems) - 50} more", file=sys.stderr)
        return 1
    bdir = os.path.join(args.site, "content", "batches")
    n = sum(1 for _ in os.scandir(bdir)) if os.path.isdir(bdir) else 0
    print(f"upload check OK: {n} blinded batch file(s), no source identifiers" + (", no raw tokens" if args.labels else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
