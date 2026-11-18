#!/usr/bin/env python3
"""Writes the hosted review site: a folder you push to a private GitHub repository that Vercel serves.

    python hosted/make_site.py --clips clips.csv --study review_study --labels review_labels \\
        --out /path/to/site --raters EC,JK --batch-size 25 --shared-clips 200

takes the same study and labels folders as serve_review.py, so the study is planned once and the links are
the same on both: a rater's link works on the site and, through the tunnel, on Borel. It writes

    content/batches/       blinded rater batches, one per batch in the study (signal and clip IDs only)
    content/site.json      batch list, the shared/split assignment, and SHA-256 hashes of the link tokens
    content/admin/         with --admin: the owner-only admin view of each batch (SubQ lanes, implant side,
                           detector strips; signal, sides and scores only), served to your own link only
    public/index.html      the viewer
    public/manual_*.html   with --manual scalp=FILE.md / uneeg=FILE.md: the 10-20 and UNEEG rater manuals,
                           behind the viewer's Manual button (it opens the one for the rater's part)
    api/, lib/             the server function (Node, no dependencies)
    tools/                 the upload check (also the pre-commit hook) and a local preview server

and nothing else: key.csv, the clip table, the plan, EEG paths and raw tokens stay on Borel. The upload
check runs at the end and the command fails if anything in the folder could tie a clip to a patient.

Run it again after --raters, --revoke or --reassign; batches already written are kept unless --rebuild.

The results page: `--results-login SJ` asks for a password (12 characters or more) and stores only its scrypt
hash. From then on every run also writes the study's results page (content/results.json) from key.csv and the
labels pulled into the labels folder (run pull_labels.py first), so it shows whatever has come in so far; it
opens at <site>/results with that username and password. Patients appear on it as P01, P02, ...; the key to
those names stays on Borel in the labels folder (results_patients.csv). `--expect-scorings` names the scorings
the page waits for, and `--extra-scores name=seconds.csv` adds a scoring that arrived after the clips were drawn.
Hosted batches must be under 4 MB (Vercel's limit), about 40 clips; plan the study with --batch-size 25.
Live mode needs numpy, scipy and pyedflib, like serve_review.py.
"""
import argparse
import datetime as dt
import getpass
import glob
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.dirname(HERE)
sys.path.insert(0, KIT)
sys.path.insert(0, HERE)
import serve_review as sr  # noqa: E402
import manual_page  # noqa: E402
import check_upload  # noqa: E402

VERSION = "1.8.6"
SITE_FORMAT = "ied-review-site"
TEMPLATE = os.path.join(HERE, "site")
SITE_MARK = b'<meta name="ied-review-site" content="1">'
RESULTS_FORMAT = "ied-review-results"
SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1}
MIN_PASSWORD = 12
DEFAULT_EXPECT = ["morgoth", "cyclops", "cyclops_sd", "spikenet1"]
# template path -> path in the site
COPIES = {"api/review.mjs": "api/review.mjs", "lib/core.mjs": "lib/core.mjs", "lib/store.mjs": "lib/store.mjs",
          "tools/dev_server.mjs": "tools/dev_server.mjs", "vercel.json": "vercel.json",
          "package.json": "package.json", "README.md": "README.md", "gitignore": ".gitignore",
          "githooks/pre-commit": ".githooks/pre-commit"}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def write_if_changed(path, data):
    """Writes bytes atomically unless the file already holds them. Returns True when written."""
    if os.path.isfile(path):
        with open(path, "rb") as f:
            if f.read() == data:
                return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return True


def hosted_bytes(raw):
    """Batch bytes as the site stores them: gzipped with no file name or time in the gzip header."""
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return gzip.compress(raw, 6, mtime=0)


def site_viewer(html):
    """The viewer with the mark that tells it it is the review site, not the synthetic demo."""
    if SITE_MARK in html:
        return html
    head = html.find(b"<head>")
    if head < 0:
        sys.exit("the viewer has no <head>; run python build.py again")
    return html[:head + 6] + b"\n" + SITE_MARK + html[head + 6:]


def inside(path, folder):
    path, folder = os.path.realpath(path), os.path.realpath(folder)
    return path == folder or path.startswith(folder + os.sep)


def site_url(args):
    """--url, else the one given last time (kept in the labels folder)."""
    path = os.path.join(args.labels, ".site_url")
    if args.url:
        url = args.url.rstrip("/")
        if not url.startswith("https://"):
            sys.exit("--url must start with https://")
        sr._write_private(path, url + "\n")
        return url
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    return ""


def ask_password():
    if not sys.stdin.isatty():
        sys.exit("give the results password at a terminal (or in IED_RESULTS_PASSWORD)")
    first = getpass.getpass(f"results password ({MIN_PASSWORD} characters or more): ")
    if getpass.getpass("again: ") != first:
        sys.exit("the two passwords differ; nothing changed")
    return first


def results_config(args):
    """The results page's settings, kept in the labels folder (.results.json): the username and a scrypt hash of
    the password (never the password), the scorings the page waits for, and scorings added after the draw.
    Returns None while no login is set."""
    path = os.path.join(args.labels, ".results.json")
    cfg = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    changed = False
    if args.results_login:
        user = args.results_login.strip().upper()
        if not re.fullmatch(r"[A-Z0-9_-]{1,12}", user):
            sys.exit("--results-login takes initials, e.g. SJ")
        if not hasattr(hashlib, "scrypt"):
            sys.exit("this Python has no hashlib.scrypt (it needs OpenSSL 1.1 or later); run make_site.py from the thesis env")
        password = os.environ.get("IED_RESULTS_PASSWORD") or ask_password()
        if len(password) < MIN_PASSWORD:
            sys.exit(f"the results password must be {MIN_PASSWORD} characters or more; the site's address is public")
        salt = os.urandom(16)
        digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, dklen=32, **SCRYPT)
        cfg.update(user=user, salt=salt.hex(), hash=digest.hex(), **SCRYPT)
        changed = True
    if args.expect_scorings is not None:
        cfg["expect"] = [x.strip().lower() for x in args.expect_scorings.split(",") if x.strip()]
        changed = True
    for spec in args.extra_scores or []:
        name, _, src = spec.partition("=")
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,30}", name) or not os.path.isfile(src):
            sys.exit(f"--extra-scores takes name=seconds.csv with a file that exists, e.g. spikenet1_sd=sd.csv ({spec!r})")
        cfg.setdefault("extra", {})[name] = os.path.abspath(src)
        changed = True
    if args.threshold is not None:
        cfg["threshold"] = args.threshold
        changed = True
    if changed:
        if not cfg.get("hash"):
            sys.exit("set the results login first: --results-login SJ")
        sr._write_private(path, json.dumps(cfg, indent=1) + "\n")
    return cfg if cfg.get("hash") else None


def write_results(args, cfg, batches, assignment, out):
    """content/results.json: the results page from key.csv and every labels file pulled so far."""
    import pandas as pd
    import report_labels
    key_path = os.path.join(args.study, "key.csv") if args.study else os.path.join(args.batches or "", "key.csv")
    if not os.path.isfile(key_path):
        print(f"note: no key.csv at {key_path}, so no results page this time")
        return None
    key = pd.read_csv(key_path, dtype=str, keep_default_na=False)
    labels = sorted(glob.glob(os.path.join(args.labels, "labels_*.csv")))
    extra = {name: pd.read_csv(src) for name, src in (cfg.get("extra") or {}).items()}
    html, names = report_labels.results_page(key, labels, assignment, {b: i["n_clips"] for b, i in batches.items()},
                                             cfg.get("expect", DEFAULT_EXPECT), extra, cfg.get("threshold", 0.5))
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    page = {"format": RESULTS_FORMAT, "version": 1, "created": stamp, "html": html}
    write_if_changed(os.path.join(out, "content", "results.json"), (json.dumps(page) + "\n").encode("utf-8"))
    sr._write_private(os.path.join(args.labels, "results_patients.csv"),
                      "patient,shown_as\n" + "".join(f"{p},{n}\n" for p, n in sorted(names.items(), key=lambda x: x[1])))
    return len(labels)


def build(args, ap):
    out = os.path.abspath(args.out)
    for name, folder in (("labels", args.labels), ("study", args.study), ("batches", args.batches),
                         ("kit", KIT)):
        if folder and (inside(out, folder) or inside(folder, out)):
            sys.exit(f"--out must be a folder of its own, not inside (or around) the {name} folder")
    store = sr.open_store(ap, args)
    owner = sr.load_token(args.labels, args.new_token)
    raters = sr.load_rater_tokens(args.labels, args.new_token, args.raters.split(","), args.revoke.split(","))
    batches = store.batches()
    if not batches:
        sys.exit("no rater batches to put on the site")
    assignment = sr.settle_assignment(store, args.labels, raters, args.reassign)
    admin_raters = sr.load_admin_raters(args.labels, args.admin_raters, raters)

    all_ids = sorted(c for info in batches.values() for c in info["clip_ids"])
    study_id = sha256_bytes("\n".join(all_ids).encode("ascii"))[:16]
    old = {}
    old_path = os.path.join(out, "content", "site.json")
    if os.path.isfile(old_path):
        with open(old_path, encoding="utf-8") as f:
            old = json.load(f)
        if old.get("format") != SITE_FORMAT:
            sys.exit(f"{out} holds a content/site.json that make_site.py did not write")
        if old.get("study_id") != study_id:
            print("note: this folder held a different study; its batches are replaced and its labels database "
                  "keys are not reused")
    elif os.path.isdir(out) and any(n not in (".git", "README.md", ".gitignore") for n in os.listdir(out)):
        sys.exit(f"{out} is not empty and is not a review site; give a new folder")
    same_study = old.get("study_id") == study_id
    old_sha = {b["batch_id"]: b.get("sha256") for b in old.get("batches", [])} if same_study else {}

    print(f"writing {len(batches)} batch(es) to {out}/content/batches ...", flush=True)
    entries, wrote, total = [], 0, 0
    for bid in sorted(batches, key=lambda b: sr.listing_order(b, batches[b])):
        info = batches[bid]
        if info.get("mode") != "rater":
            sys.exit(f"batch {bid} is not a blinded rater batch; research batches never go on the site")
        path = os.path.join(out, "content", "batches", f"{bid}.json.gz")
        data = None
        if not args.rebuild and old_sha.get(bid) and os.path.isfile(path):
            with open(path, "rb") as f:
                data = f.read()
            if sha256_bytes(data) != old_sha[bid]:
                data = None
        if data is None:
            data = hosted_bytes(store.batch_bytes(bid))
            if len(data) > check_upload.MAX_BATCH_BYTES:
                sys.exit(f"batch {bid} packs to {len(data) / 1e6:.1f} MB; hosted batches must be under "
                         f"{check_upload.MAX_BATCH_BYTES / 1e6:.0f} MB. Plan the study with --batch-size 25 "
                         f"(a new --study and --labels folder).")
            wrote += write_if_changed(path, data)
            print(f"  {bid}: {info['n_clips']} clips, {len(data) / 1e6:.1f} MB", flush=True)
        total += len(data)
        entries.append(dict({"batch_id": bid, "file": f"{bid}.json.gz", "n_clips": info["n_clips"],
                             "shared": bool(info.get("shared")), "synthetic": bool(info.get("synthetic")),
                             "answers": list(info.get("answers") or sr.OLD_ANSWERS), "sha256": sha256_bytes(data),
                             "clip_ids": sorted(info["clip_ids"])}, **sr.stage_info(info)))
    keep = {e["file"] for e in entries}
    for name in sorted(os.listdir(os.path.join(out, "content", "batches"))):
        if name not in keep:
            os.remove(os.path.join(out, "content", "batches", name))
            print(f"  removed {name} (not in the study any more)")

    # the owner-only admin view: SubQ lanes, implant side and detector strips, served to your own link only
    admin_dir = os.path.join(out, "content", "admin")
    n_admin = 0
    if store.has_admin():
        print(f"writing the admin view to {admin_dir} ...", flush=True)
        for e in entries:
            data = store.admin_bytes(e["batch_id"])
            if data is None:
                continue
            data = hosted_bytes(data)
            if len(data) > check_upload.MAX_BATCH_BYTES:
                sys.exit(f"the admin view of batch {e['batch_id']} packs to {len(data) / 1e6:.1f} MB; it must be under "
                         f"{check_upload.MAX_BATCH_BYTES / 1e6:.0f} MB")
            write_if_changed(os.path.join(admin_dir, e["file"]), data)
            e["admin_file"] = e["file"]
            n_admin += 1
    if os.path.isdir(admin_dir):
        for name in sorted(os.listdir(admin_dir)):
            if name not in {e.get("admin_file") for e in entries}:
                os.remove(os.path.join(admin_dir, name))

    site = {"format": SITE_FORMAT, "version": 1, "study_id": study_id,
            "created": old.get("created") if same_study and old.get("created") else
            dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "kit_version": VERSION, "batches": entries,
            "assignment": None if assignment is None else
            {"shared": assignment["shared"], "by_rater": assignment["by_rater"]},
            "tokens": {who: sha256_bytes(tok.encode("ascii")) for who, tok in {"": owner, **raters}.items()}}
    if admin_raters:
        site["admin_raters"] = list(admin_raters)
    tolerance_ms = sr.load_tolerance(args.labels, args.tolerance_ms)
    if tolerance_ms:
        site["tolerance_s"] = tolerance_ms / 1000
    # the rater manuals (kit 1.8.4): public pages, so each is checked like everything else before it is written
    manual_text = manual_page.load(args.labels, args.manual)
    for part in manual_page.PARTS:
        path = os.path.join(out, "public", manual_page.file_name(part))
        if part in manual_text:
            found = check_upload.scan_text(manual_text[part], f"the {part} manual")
            if found:
                sys.exit("--manual: " + "; ".join(found) + ". Take it out of the manual and run again.")
            write_if_changed(path, manual_page.page(manual_text[part]))
        elif os.path.exists(path):
            os.remove(path)
    if os.path.exists(os.path.join(out, "public", "manual.html")):  # a single manual from an earlier 1.8.4 build
        os.remove(os.path.join(out, "public", "manual.html"))
    if manual_text:
        site["manuals"] = {part: manual_page.file_name(part) for part in manual_text}
    cfg = results_config(args)
    if cfg:
        site["results_login"] = {k: cfg[k] for k in ("user", "salt", "hash", "n", "r", "p")}
    write_if_changed(old_path, (json.dumps(site, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    n_results = write_results(args, cfg, batches, assignment, out) if cfg else None
    if not cfg and os.path.exists(os.path.join(out, "content", "results.json")):
        os.remove(os.path.join(out, "content", "results.json"))

    for src, dst in COPIES.items():
        with open(os.path.join(TEMPLATE, src), "rb") as f:
            write_if_changed(os.path.join(out, dst), f.read())
    with open(os.path.join(HERE, "check_upload.py"), "rb") as f:
        write_if_changed(os.path.join(out, "tools", "check_upload.py"), f.read())
    with open(args.viewer, "rb") as f:
        write_if_changed(os.path.join(out, "public", "index.html"), site_viewer(f.read()))
    os.chmod(os.path.join(out, ".githooks", "pre-commit"), 0o755)
    if os.path.isdir(os.path.join(out, ".git")):
        subprocess.run(["git", "-C", out, "config", "core.hooksPath", ".githooks"], check=False)

    problems = check_upload.check(out, args.labels)
    if problems:
        print("upload check FAILED; do not push this folder:", file=sys.stderr)
        for p in problems[:50]:
            print("  " + p, file=sys.stderr)
        sys.exit(1)
    print(f"site ready in {out}: {len(entries)} batch(es), {total / 1e6:.0f} MB, {wrote} batch file(s) new or "
          f"changed; upload check OK")
    if n_admin:
        print(f"admin view: {n_admin} batch(es); your own link opens it (S: SubQ, D: detector strips)" +
              (f", and so do {', '.join(admin_raters)}'s" if admin_raters else ""))
    sr.describe_assignment(assignment, batches, raters)
    sr.describe_stages(batches)
    if tolerance_ms:
        print(f"raters are told to count a spike whose peak is within {tolerance_ms} ms of the highlighted second")
    if manual_text:
        print("raters see a Manual button: " + sr.describe_manuals(manual_text) + " (kept in the labels folder; "
              "give --manual again after editing one, or --manual none)")
    print_links(site_url(args), owner, raters, admin_raters)
    if n_results is not None:
        print(f"results page (from {n_results} labels file(s) in {args.labels}; run pull_labels.py first for the latest):")
        print(f"    {site_url(args) or 'https://YOUR-SITE.vercel.app'}/results   username {cfg['user']}")


def print_links(url, owner, raters, admin_raters=()):
    base = url or "https://YOUR-SITE.vercel.app"
    if not url:
        print("(give --url https://... once the site is on Vercel to print the real links)")
    print("your link (every batch, everyone's progress, label export):")
    print(f"    {base}/?token={owner}")
    if raters:
        print("rater links (each opens signed in as that rater; send each person only their own):")
        for r, tok in sorted(raters.items()):
            print(f"    {r:<6} {base}/?token={tok}" + ("   (also opens the admin view)" if r in admin_raters else ""))



def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("batches", nargs="?", help="folder of rater batch files from make_review_batches.py pack")
    ap.add_argument("--labels", required=True, help="the labels folder serve_review.py uses (link tokens live here)")
    ap.add_argument("--out", help="the site folder to write (your clone of the private repository)")
    ap.add_argument("--viewer", default=os.path.join(KIT, "ied_review.html"), help="viewer HTML")
    ap.add_argument("--raters", default="", help="comma-separated initials that get their own link, e.g. EC,JK")
    ap.add_argument("--revoke", default="", help="comma-separated initials whose links stop working")
    ap.add_argument("--new-token", action="store_true", help="replace every link; old links stop working")
    ap.add_argument("--reassign", action="store_true",
                    help="deal the split batches nobody has started again (run pull_labels.py first)")
    ap.add_argument("--rebuild", action="store_true", help="rewrite every batch file, e.g. after changing --notch")
    ap.add_argument("--url", help="the site's address on Vercel, e.g. https://ied-review.vercel.app (remembered)")
    ap.add_argument("--links", action="store_true", help="print the links and exit")
    ap.add_argument("--results-login", metavar="INITIALS",
                    help="turn on the results page with this username; asks for a password (12+ characters)")
    ap.add_argument("--expect-scorings", help="scorings the results page waits for (default " + ",".join(DEFAULT_EXPECT) + ")")
    ap.add_argument("--extra-scores", action="append", metavar="NAME=SECONDS.CSV",
                    help="a scoring that arrived after the clips were drawn, as a seconds table (repeatable)")
    ap.add_argument("--threshold", type=float, help="score at which the results page counts a second as flagged (0.5)")
    sr.add_live_args(ap)
    ap.set_defaults(allow_research=False, status=False)
    args = ap.parse_args(argv)
    if args.links:
        if not os.path.exists(os.path.join(args.labels, ".token")):
            sys.exit(f"no links in {args.labels} yet; give the labels folder make_site.py wrote the site from")
        owner = sr.load_token(args.labels, False)
        raters = sr.load_rater_tokens(args.labels, False)
        print_links(site_url(args), owner, raters, sr.load_admin_raters(args.labels, None, raters))
        return
    if not args.out:
        ap.error("--out is required (the site folder)")
    if not os.path.isfile(args.viewer):
        sys.exit(f"viewer not found at {args.viewer}; run python build.py first")
    build(args, ap)


if __name__ == "__main__":
    main()
