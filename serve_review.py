#!/usr/bin/env python3
"""Serve ied_review.html and EEG clips from Borel, and save labels back to Borel.

Two ways to give it clips:

    python serve_review.py --clips clips.csv --study review_study --labels review_labels --raters EC,JK

reads every clip straight from the EEG files on Borel when a rater opens a batch (live
mode; nothing is packed or copied). On first start it checks that every clip in clips.csv
can be read, fixes the clip IDs, order, repeats and batches, and writes the plan and
key.csv into the study folder; later starts reuse that plan, so batch b03 is always the
same clips. EDF+D files are read on their own clock and a clip that crosses a gap is left
out at planning time.

    python serve_review.py review_batches --labels review_labels

serves batch files already written by make_review_batches.py pack.

Raters reach it through an SSH tunnel, the way Jupyter on finkel works, so EEG is drawn in
their browser but no batch file or label file is ever written to their computer:

    ssh -N -L 8765:localhost:8765 PENNKEY@borel.seas.upenn.edu
    open http://localhost:8765/?token=...   (the full link is printed at start)

Labels are saved to the labels folder as the rater works (labels_<batch>_<RATER>.csv, the
same columns as Export labels), with a timestamped copy in history/ at most every 10
minutes. merge_labels.py reads them as they are.

Rater logins: --raters EC,JK gives each rater a link of their own (tokens kept in
<labels>/.raters.json, readable only by you). A rater's link opens the viewer signed in
as that rater: it can read and save only that rater's labels and sees only their own
progress. Your own (owner) link asks for your initials once per browser and may not save
under initials that have their own link (1.7.7). --revoke EC turns one link off; --links
prints the links again.

Shared and split batches: a study planned with --shared-clips 200 has shared batches
(s01, s02, ...) that every rater labels, for agreement between raters, and the rest
(b01, b02, ...) split evenly between the raters named when the split is first made. The
split is kept in <labels>/.assignment.json so it never changes under anyone. A rater
added later gets the shared batches only, until --reassign hands out the batches nobody
has started. Your own link always sees every batch.

Safety:
  * Listens on 127.0.0.1 only. Borel is shared, so every request must also carry a
    token: your own (kept in <labels>/.token, readable only by you, so the link survives
    a restart) or a rater's. Requests without one get 403.
  * Serves only the viewer and the batches. key.csv, the study plan, manifest.json, the
    clip table and the EEG file paths are never served; a clip that fails to read is
    reported in your terminal, not to the browser.
  * Refuses research (unblinded) batch files unless --allow-research is given; live mode
    only makes blinded batches.

    python serve_review.py review_batches --labels review_labels --status

prints how far each rater has got without starting the server.

Batch-file mode needs only the standard library; live mode also needs numpy, scipy and
pyedflib, like make_review_batches.py pack.
"""
import argparse
import csv
import datetime as dt
import getpass
import gzip
import hmac
import io
import json
import os
import re
import secrets
import shutil
import sys
import threading
from collections import OrderedDict
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

import manual_page  # noqa: E402  (the rater manual page, kit 1.8.4)

HERE = os.path.dirname(os.path.abspath(__file__))
# Must match EXPORT_COLUMNS in src/core.js (tests/test_server.py checks).
EXPORT_COLUMNS = ["batch_id", "clip_id", "rater", "order_index", "decision", "max_electrodes", "side",
                  "region", "generalized", "artifact", "other_discharges_s", "note", "seconds_viewed", "labeled_at",
                  "order_seed", "viewer_version", "view"]
# labels saved by viewers before kit 1.7, which had no view column, are still accepted
ACCEPTED_COLUMNS = (EXPORT_COLUMNS, EXPORT_COLUMNS[:-1])
RATER_RE = re.compile(r"^[A-Z0-9_-]{1,12}$")
BATCH_FILE_RE = re.compile(r"^[A-Za-z0-9_.-]+\.json(\.gz)?$")
MAX_LABELS_BYTES = 4 * 1024 * 1024
HISTORY_EVERY_S = 600
PLAN_FORMAT = "ied-review-live-plan"
ASSIGN_FORMAT = "ied-review-assignment"
# answers a batch allows when it does not say (batches and plans made before kit 1.6)
OLD_ANSWERS = ["yes", "no", "unsure"]


def read_batch(path):
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def stage_info(obj):
    """The staged-study fields of a batch header or plan entry (kit 1.8): which part, and how it is shown."""
    out = {}
    if obj.get("stage"):
        out["stage"] = int(obj["stage"])
        out["stage_name"] = str(obj.get("stage_name") or "")
    if obj.get("view") in ("subq", "both"):
        out["view"] = obj["view"]
    if obj.get("training"):
        out["training"] = True
    return out


def locked_batches(batches, who, answered, visible_ids):
    """Batch IDs a rater cannot open yet. In a staged study each part opens once the rater has answered every
    clip of the earlier parts they can see; answered(bid) is that rater's count. The owner ("") opens all."""
    if not who:
        return set()
    stages = sorted({batches[b]["stage"] for b in visible_ids if batches[b].get("stage")})
    locked, open_ = set(), True
    for st in stages:
        ids = [b for b in visible_ids if batches[b].get("stage") == st]
        if not open_:
            locked.update(ids)
            continue
        if any(answered(b) < batches[b]["n_clips"] for b in ids):
            open_ = False
    return locked


LOCKED_MSG = "This part opens once you have answered every clip in the parts before it.\n"


def describe_manuals(manuals):
    """One line on which manual each part gets."""
    names = {k[len("manual_"):-len(".html")] if k.startswith("manual_") else k for k in manuals}
    out = []
    if "scalp" in names:
        out.append("the 10-20 manual for the 10-20 clips" + ("" if "uneeg" in names else " and every other part"))
    if "uneeg" in names:
        out.append("the UNEEG manual for the training examples and UNEEG clips" + ("" if "scalp" in names else
                                                                                   " and every other part"))
    return "; ".join(out)


def load_tolerance(labels_dir, asked):
    """The margin around the highlighted second that raters are told to count (kit 1.8), in ms, kept in
    <labels>/.tolerance_ms so the site keeps it between runs. asked is --tolerance-ms: None keeps the saved value
    (0 when none was saved), any number replaces it (0 turns it off)."""
    path = os.path.join(labels_dir, ".tolerance_ms")
    if asked is None:
        if os.path.exists(path):
            with open(path, encoding="ascii") as f:
                return int(f.read().strip() or 0)
        return 0
    ms = int(asked)
    if ms < 0 or ms > 1000:
        sys.exit("--tolerance-ms must be between 0 and 1000")
    _write_private(path, f"{ms}\n")
    return ms


class Store:
    """Batch headers (cached by mtime) and label files."""

    def __init__(self, batch_dir, labels_dir, allow_research):
        self.batch_dir = os.path.abspath(batch_dir)
        self.labels_dir = os.path.abspath(labels_dir)
        self.allow_research = allow_research
        self._cache = {}
        self._lock = threading.Lock()
        os.makedirs(os.path.join(self.labels_dir, "history"), mode=0o700, exist_ok=True)
        self.shared_ids = set()
        manifest = os.path.join(self.batch_dir, "manifest.json")
        if os.path.isfile(manifest):
            with open(manifest, encoding="utf-8") as f:
                self.shared_ids = {str(b.get("batch_id")) for b in json.load(f).get("batches", []) if b.get("shared")}

    def batches(self):
        """{batch_id: info} for every servable batch file in the folder."""
        out = {}
        for name in sorted(os.listdir(self.batch_dir)):
            if not BATCH_FILE_RE.match(name):
                continue
            path = os.path.join(self.batch_dir, name)
            if not os.path.isfile(path) or os.path.islink(path):
                continue
            mtime = os.path.getmtime(path)
            hit = self._cache.get(name)
            if not hit or hit[0] != mtime:
                try:
                    obj = read_batch(path)
                except Exception:
                    continue
                if obj.get("format") != "ied-review-batch":
                    continue
                info = {"file": name, "batch_id": str(obj.get("batch_id", "")), "mode": obj.get("mode", "rater"),
                        "synthetic": bool(obj.get("synthetic")), "n_clips": len(obj.get("clips", [])),
                        "clip_ids": {c.get("id") for c in obj.get("clips", [])},
                        "answers": list(obj.get("answers") or OLD_ANSWERS)}
                info.update(stage_info(obj))
                info["shared"] = info["batch_id"] in self.shared_ids
                hit = (mtime, info)
                self._cache[name] = hit
            info = hit[1]
            if info["mode"] == "research" and not self.allow_research:
                continue
            if info["batch_id"] in out:
                print(f"WARNING: {name} repeats batch id {info['batch_id']}; serving only the first", file=sys.stderr)
                continue
            out[info["batch_id"]] = info
        return out

    def batch_bytes(self, batch_id):
        info = self.batches().get(batch_id)
        if info is None:
            return None
        with open(os.path.join(self.batch_dir, info["file"]), "rb") as f:
            return f.read()

    def has_admin(self):
        return False

    def admin_bytes(self, batch_id):
        return None

    def labels_path(self, batch_id, rater):
        return os.path.join(self.labels_dir, f"labels_{batch_id}_{rater}.csv")

    def progress(self, batch_id):
        """{rater: answered count} from saved label files for one batch."""
        out = {}
        prefix = f"labels_{batch_id}_"
        for name in os.listdir(self.labels_dir):
            if name.startswith(prefix) and name.endswith(".csv"):
                rater = name[len(prefix):-4]
                if not RATER_RE.match(rater):
                    continue
                with open(os.path.join(self.labels_dir, name), newline="", encoding="utf-8") as f:
                    out[rater] = sum(1 for r in csv.DictReader(f) if r.get("decision"))
        return out

    def save_labels(self, batch_id, rater, body):
        """Validates an exported-labels CSV and writes it atomically. Returns (answered, total)."""
        info = self.batches().get(batch_id)
        if info is None:
            raise ValueError("unknown batch")
        text = body.decode("utf-8")
        rows = list(csv.DictReader(io.StringIO(text)))
        head = text.split("\n", 1)[0].strip().lstrip("﻿").split(",")
        if head not in ACCEPTED_COLUMNS:
            raise ValueError("columns do not match the viewer's export format")
        for r in rows:
            if r["batch_id"] != batch_id or r["rater"].upper() != rater:
                raise ValueError("a row names another batch or rater")
            if r["clip_id"] not in info["clip_ids"]:
                raise ValueError("a row names a clip that is not in this batch")
            if r["decision"] and r["decision"] not in info["answers"]:
                raise ValueError(f"a row answers {r['decision']!r}, which this study does not offer")
        path = self.labels_path(batch_id, rater)
        with self._lock:
            if os.path.exists(path):
                hist = os.path.join(self.labels_dir, "history")
                stamps = sorted(n for n in os.listdir(hist) if n.startswith(f"labels_{batch_id}_{rater}_"))
                last = os.path.getmtime(os.path.join(hist, stamps[-1])) if stamps else 0
                if dt.datetime.now().timestamp() - last > HISTORY_EVERY_S:
                    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
                    shutil.copy2(path, os.path.join(hist, f"labels_{batch_id}_{rater}_{stamp}.csv"))
            tmp = path + ".tmp"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            os.replace(tmp, path)
        return sum(1 for r in rows if r["decision"]), len(rows)


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_private(path, text):
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, path)


class LiveStore(Store):
    """Blinded batches planned from a clip table and read from the EEG files when opened; nothing is packed."""

    SETTINGS = ("fs", "clip_s", "pad_s", "candidate_s", "batch_size", "repeat_frac", "seed", "batch_prefix",
                "extras", "reader", "answers", "shared_clips", "question")

    def __init__(self, clips_csv, study_dir, labels_dir, opts, prepare=True):
        self.clips_csv = os.path.abspath(clips_csv)
        self.study_dir = self.batch_dir = os.path.abspath(study_dir)
        self.labels_dir = os.path.abspath(labels_dir)
        self.allow_research = False
        self._lock = threading.Lock()
        self._cache = OrderedDict()
        os.makedirs(os.path.join(self.labels_dir, "history"), mode=0o700, exist_ok=True)
        sys.path.insert(0, HERE)
        import make_review_batches as mrb
        self.mrb = mrb
        if not os.path.isfile(self.clips_csv):
            sys.exit(f"{clips_csv} not found")
        with open(self.clips_csv, newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        plan_path = os.path.join(self.study_dir, "plan.json")
        if os.path.exists(plan_path):
            with open(plan_path, encoding="utf-8") as f:
                self.plan = json.load(f)
            if self.plan.get("format") != PLAN_FORMAT:
                sys.exit(f"{plan_path} is not a review study plan")
            if self.plan["clips_sha256"] != _sha256(self.clips_csv):
                sys.exit(f"{clips_csv} has changed since this study was planned from it. A plan is fixed once raters "
                         f"start; to plan a new study from the new table, use a new --study folder (and --labels folder).")
        elif prepare:
            self.plan = self.prepare(opts)
        else:
            sys.exit(f"no study planned in {self.study_dir} yet; start the server once with --clips to plan it")
        st = self.plan["settings"]
        # plans made before kit 1.6 keep Yes, No and Unsure and have no shared batches
        answers = st.get("answers", "ynu")
        # plans made before kit 1.7 asked about the highlighted second
        question = st.get("question", "second")
        for k, planned in (("answers", answers), ("shared_clips", st.get("shared_clips", 0)), ("question", question)):
            asked = getattr(opts, k, None)
            if asked is not None and asked != planned and not (getattr(opts, "status", False) or getattr(opts, "links", False)):
                print(f"note: this study was planned with --{k.replace('_', '-')} {planned}; keeping that "
                      f"(a plan is fixed once raters start)")
        self.ns = SimpleNamespace(fs=st["fs"], clip_s=st["clip_s"], pad_s=st["pad_s"], candidate_s=st["candidate_s"],
                                  mode="rater", extras=st["extras"], subq_names=[], extras_all=st["extras"],
                                  notch=getattr(opts, "notch", "on"), answers=answers, question=question,
                                  subq_channels=st.get("subq_channels") or [])
        self.reader_name = st["reader"]
        # the owner-only extras (select_clips.py admin): not part of the plan, so they can be remade any time
        self.admin_rows = None
        if getattr(opts, "admin", None):
            with open(opts.admin, newline="", encoding="utf-8") as f:
                self.admin_rows = {(r["source_path"], round(float(r["center_s"]), 3)): r for r in csv.DictReader(f)}
            n_match = sum((r["source_path"], round(float(r["center_s"]), 3)) in self.admin_rows for r in self.rows)
            if not n_match:
                sys.exit(f"{opts.admin} matches no clip in {clips_csv}; make it from this clip table with "
                         "select_clips.py admin")
        self._admin_cache = OrderedDict()
        self._batches = {}
        for b in self.plan["batches"]:
            self._batches[b["batch_id"]] = {"file": b["batch_id"] + ".json.gz", "batch_id": b["batch_id"], "mode": "rater",
                                            "synthetic": False, "n_clips": len(b["members"]),
                                            "clip_ids": {m["id"] for m in b["members"]}, "plan": b,
                                            "answers": mrb.ANSWER_SETS[answers], "shared": bool(b.get("shared"))}
            if b.get("part") and mrb.PARTS[b["part"]]["training"]:
                self._batches[b["batch_id"]]["answers"] = list(mrb.TEACHING_ANSWERS)  # looked at, not answered (1.8.5)
            if b.get("part"):
                p = mrb.PARTS[b["part"]]
                self._batches[b["batch_id"]].update(stage_info({"stage": p["stage"], "stage_name": p["name"],
                                                                "view": p["view"], "training": p["training"]}))

    def prepare(self, opts):
        """Reads every clip once (to prove it can be read), then fixes IDs, order, repeats and batches."""
        import numpy as np
        mrb = self.mrb
        for col in ("source_path", "center_s"):
            if self.rows and col not in self.rows[0]:
                sys.exit(f"{self.clips_csv} needs a '{col}' column")
        if not self.rows:
            sys.exit(f"{self.clips_csv} has no clips")
        staged = any(mrb.part_of(r) for r in self.rows)
        if staged:
            problems = mrb.check_parts(self.rows)
            if problems:
                sys.exit(f"{self.clips_csv} is a staged study table (a part column) with problems:\n  " +
                         "\n  ".join(problems[:20]))
        opts.answers = getattr(opts, "answers", None) or "yn"
        opts.shared_clips = getattr(opts, "shared_clips", None) or 0
        opts.question = getattr(opts, "question", None) or "clip"
        os.makedirs(self.study_dir, mode=0o700, exist_ok=True)
        extras = [mrb.normalize_name(e) for e in opts.extras]
        ns = SimpleNamespace(fs=opts.fs, clip_s=opts.clip_s, pad_s=opts.pad_s, candidate_s=opts.candidate_s,
                             mode="rater", extras=extras, subq_names=[], extras_all=extras)
        reader = mrb.READERS[opts.reader]()
        good, bad_by_row, amps, failures = [], {}, [], []
        subq_channels, subq_amps = None, []
        print(f"checking {len(self.rows)} clips in {os.path.basename(self.clips_csv)} against the EEG files ...", flush=True)
        for i, row in enumerate(self.rows):
            try:
                if mrb.part_of(row) == "uneeg":
                    names, rows, bad = mrb.read_subq_clip(reader, row, ns, subq_channels)
                    if subq_channels is None:
                        subq_channels = names
                        ns.subq_channels = names
                elif mrb.part_of(row) == "examples":
                    # the scalp clip with the UNEEG channels under it, when the example has them (kit 1.8.1)
                    rows, bad, names = mrb.read_example(reader, row, ns, subq_channels)
                    if subq_channels is None and names:
                        subq_channels = names
                        ns.subq_channels = names
                else:
                    rows, bad, _units = mrb.read_clip(reader, row["source_path"], float(row["center_s"]), ns, extras,
                                                      row.get("edf_clock"))
            except Exception as e:  # keep going; every failure is listed at the end
                failures.append({"row": i + 2, "source_path": row.get("source_path", ""), "why": str(e)})
                continue
            good.append(i)
            bad_by_row[i] = bad
            if mrb.part_of(row) == "uneeg":
                subq_amps.append(float(np.median(mrb.robust_amp(rows))))
            else:
                amps.append(float(np.median(mrb.robust_amp(rows[:len(mrb.SCALP)]))))
            if len(good) % 200 == 0:
                print(f"  {len(good)} read", flush=True)
                reader.close()
        reader.close()
        if failures:
            print(f"{len(failures)} clip(s) could not be read and are left out of the study:")
            for f_ in failures[:20]:
                print(f"  row {f_['row']}: {f_['source_path']}: {f_['why']}")
        if not good:
            sys.exit("no clip could be read")
        med = float(np.median(amps)) if amps else float("nan")
        if amps:
            print(f"median robust scalp amplitude {med:.1f} µV across {len(amps)} clips")
        if subq_amps:
            print(f"median robust UNEEG amplitude {float(np.median(subq_amps)):.1f} µV across {len(subq_amps)} clips "
                  f"({', '.join(subq_channels)})")
        from collections import Counter
        badc = Counter(n for i in good for n in bad_by_row[i])
        if badc:
            print("channels missing, flat or noisy (drawn dashed for raters): " +
                  ", ".join(f"{n} in {k}" for n, k in badc.most_common(12)))
        if amps and (med < 0.5 or med > 2000):
            print("WARNING: that amplitude is implausible for scalp EEG in microvolts. Check the units before "
                  "anyone labels (a factor of 1e6 means volts were read as microvolts or the reverse).")

        rng = np.random.default_rng([opts.seed, mrb.table_salt([self.rows[i] for i in good])])
        if staged:
            # one set of batches per part (e01, s01.., u01..), every one shared: each rater answers every clip, part
            # by part, so agreement covers the whole study and the UNEEG part comes after the scalp seconds
            groups, taken = [], set()
            for part, spec in mrb.PARTS.items():
                idx = [r for r in good if mrb.part_of(self.rows[r]) == part]
                if not idx:
                    continue
                size = len(idx) if part == "examples" and len(idx) <= 40 else opts.batch_size
                ids, plan = mrb.plan_batches(len(idx), "rater", size, opts.repeat_frac if part == "scalp" else 0.0,
                                             spec["prefix"] + "x", rng, shared=10 ** 9, shared_prefix=spec["prefix"],
                                             taken=taken)  # shared past the end: the repeats are shared too
                taken |= {m["id"] for pb in plan for m in pb["members"]}
                groups.append((part, idx, plan))
        else:
            _ids, plan = mrb.plan_batches(len(good), "rater", opts.batch_size, opts.repeat_frac, opts.batch_prefix,
                                          rng, shared=opts.shared_clips)
            groups = [(None, good, plan)]
        key_rows, batches = [], []
        for part, idx, plan in groups:
            for pb in plan:
                members = []
                for pos, m in enumerate(pb["members"]):
                    r = idx[m["src"]]
                    members.append({"id": m["id"], "row": r, "repeat_of": m["repeat_of"]})
                    key_rows.append(mrb.key_row(m, pb["batch_id"], pb["batch_id"] + ".json.gz", pos, bad_by_row[r],
                                                dict(self.rows[r], question=opts.question)))
                entry = {"batch_id": pb["batch_id"], "order_seed": pb["order_seed"], "shared": pb["shared"],
                         "members": members}
                if part:
                    entry["part"] = part
                batches.append(entry)
        mrb.write_key(os.path.join(self.study_dir, "key.csv"), key_rows)
        settings = {k: getattr(opts, k) for k in self.SETTINGS}
        settings["extras"] = extras
        if staged:
            settings["staged"] = True
            settings["subq_channels"] = subq_channels or []
        plan_obj = {"format": PLAN_FORMAT, "version": 1, "created": mrb.utc_stamp(), "clips_csv": self.clips_csv,
                    "clips_sha256": _sha256(self.clips_csv), "n_rows": len(self.rows), "settings": settings,
                    "left_out": failures, "batches": batches}
        _write_private(os.path.join(self.study_dir, "plan.json"), json.dumps(plan_obj, indent=1))
        n_slots = sum(len(b["members"]) for b in batches)
        n_shared = sum(len(b["members"]) for b in batches if b["shared"])
        print(f"planned {len(batches)} batch(es): {len(good)} clips plus {n_slots - len(good)} repeats; "
              f"key.csv and plan.json are in {self.study_dir} (readable only by you)")
        if staged:
            for part, idx, plan in groups:
                print(f"  part {mrb.PARTS[part]['stage']}, {mrb.PARTS[part]['name']}: {len(idx)} clips in "
                      f"{len(plan)} batch(es), {plan[0]['batch_id']} to {plan[-1]['batch_id']}; every rater answers all")
        elif n_shared:
            print(f"{n_shared} clips in shared batches for every rater; {n_slots - n_shared} split between raters")
        print(f"answers: {' / '.join(mrb.ANSWER_SETS[opts.answers])}; question: "
              + ("an IED anywhere in the 10 s shown" if opts.question == "clip" else "an IED in the highlighted second"))
        return plan_obj

    def batches(self):
        return self._batches

    def batch_bytes(self, batch_id):
        info = self._batches.get(batch_id)
        if info is None:
            return None
        with self._lock:
            if batch_id in self._cache:
                self._cache.move_to_end(batch_id)
                return self._cache[batch_id]
            mrb = self.mrb
            b = info["plan"]
            part = b.get("part")
            obj = mrb.batch_header(self.ns, batch_id, b["order_seed"], False, self.plan["created"], part=part)
            reader = mrb.READERS[self.reader_name]()
            try:
                for m in b["members"]:
                    row = self.rows[m["row"]]
                    try:
                        if part == "uneeg":
                            _n, rows, bad = mrb.read_subq_clip(reader, row, self.ns, self.ns.subq_channels)
                        elif part == "examples":
                            rows, bad, _n = mrb.read_example(reader, row, self.ns, self.ns.subq_channels)
                        else:
                            rows, bad, _u = mrb.read_clip(reader, row["source_path"], float(row["center_s"]), self.ns,
                                                          self.ns.extras, row.get("edf_clock"))
                    except Exception as e:
                        # details go to the owner's terminal only; the browser must not learn the source
                        print(f"ERROR: batch {batch_id}, clip {m['id']} (clip table row {m['row'] + 2}) could not be "
                              f"read: {e}", file=sys.stderr, flush=True)
                        raise RuntimeError("a clip in this batch could not be read from the EEG files") from None
                    item, _c, _s = mrb.clip_item({"rows": rows, "bad": bad, "event_s": self.ns.clip_s / 2},
                                                 m["id"], "rater")
                    if part == "examples":
                        item.update(mrb.ref_fields(row))
                    obj["clips"].append(item)
            finally:
                reader.close()
            # mtime=0: the same batch always packs to the same bytes (the hosted site commits them)
            data = gzip.compress(json.dumps(obj, separators=(",", ":")).encode("utf-8"), 6, mtime=0)
            self._cache[batch_id] = data
            while len(self._cache) > 3:
                self._cache.popitem(last=False)
            return data


    def has_admin(self):
        return self.admin_rows is not None

    def admin_bytes(self, batch_id):
        """The owner-only extras for a batch (SubQ lanes, implant side, detector strips), or None."""
        info = self._batches.get(batch_id)
        if info is None or self.admin_rows is None:
            return None
        with self._lock:
            if batch_id in self._admin_cache:
                return self._admin_cache[batch_id]
            mrb = self.mrb
            obj = mrb.admin_header(self.ns, batch_id)
            reader = mrb.READERS[self.reader_name]()
            try:
                for m in info["plan"]["members"]:
                    row = self.rows[m["row"]]
                    arow = self.admin_rows.get((row["source_path"], round(float(row["center_s"]), 3)))
                    if arow is None:
                        continue
                    try:
                        obj["clips"].append(mrb.admin_item(reader, arow, m["id"], self.ns,
                                                           row if info["plan"].get("part") == "uneeg" else None))
                    except Exception as e:
                        print(f"ERROR: admin view, batch {batch_id}, clip {m['id']} (clip table row {m['row'] + 2}): "
                              f"{e}", file=sys.stderr, flush=True)
                        obj["clips"].append({"id": m["id"], "error": "SubQ could not be read"})
            finally:
                reader.close()
            data = gzip.compress(json.dumps(obj, separators=(",", ":")).encode("utf-8"), 6, mtime=0)
            self._admin_cache[batch_id] = data
            while len(self._admin_cache) > 3:
                self._admin_cache.popitem(last=False)
            return data


def make_handler(store, tokens, viewer_path, port, assignment=None, admin_raters=(), tolerance_ms=0, manuals=None):
    manuals = {manual_page.file_name(k): v for k, v in (manuals or {}).items()}  # page name -> HTML (kit 1.8.4)
    admin_ok = {""} | set(admin_raters)  # the owner, and raters given the admin view (an advisor's first look)

    def locked_for(who):
        batches = store.batches()
        ids = [b for b in batches if visible(assignment, batches, who, b)]
        return locked_batches(batches, who, lambda b: store.progress(b).get(who, 0), ids)
    allowed_hosts = {f"localhost:{port}", f"127.0.0.1:{port}", f"[::1]:{port}"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "ied-review"
        sys_version = ""

        def log_message(self, fmt, *a):  # never log the query string, which carries the token
            path = urlsplit(self.path).path
            sys.stderr.write(f"{self.log_date_time_string()} {self.command} {path} {a[1] if len(a) > 1 else ''}\n")

        def send(self, code, body=b"", ctype="text/plain; charset=utf-8", extra=None):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def identity(self):
            """None (refused), "" (the owner's token) or the rater a rater token belongs to."""
            if self.headers.get("Host", "") not in allowed_hosts:
                return None
            given = (self.headers.get("X-Review-Token") or parse_qs(urlsplit(self.path).query).get("token", [""])[0]).encode()
            who = None
            for name, tok in tokens.items():
                if hmac.compare_digest(given, tok.encode()):
                    who = name
            return who

        def authorized(self):
            self.who = self.identity()
            return self.who is not None

        def route(self):
            parts = [unquote(p) for p in urlsplit(self.path).path.split("/") if p]
            return parts

        def do_GET(self):
            # the rater manuals (kit 1.8.4) hold no study data, so they open without a token, from this machine only
            parts = self.route()
            if len(parts) == 1 and parts[0] in manuals and self.headers.get("Host", "") in allowed_hosts:
                return self.send(200, manuals[parts[0]], "text/html; charset=utf-8")
            if not self.authorized():
                return self.send(403, "Forbidden. Open the link with ?token= that serve_review.py printed.\n")
            parts = self.route()
            if parts in ([], ["index.html"]):
                with open(viewer_path, "rb") as f:
                    return self.send(200, f.read(), "text/html; charset=utf-8")
            if parts == ["api", "batches"]:
                items = []
                batches = store.batches()
                locked = locked_for(self.who)
                for bid, info in sorted(batches.items(), key=lambda kv: listing_order(kv[0], kv[1])):
                    if not visible(assignment, batches, self.who, bid):
                        continue
                    prog = store.progress(bid)
                    if self.who:  # a rater sees only their own progress
                        prog = {self.who: prog.get(self.who, 0)}
                    item = {"file": info["file"], "batch_id": bid, "mode": info["mode"],
                            "synthetic": info["synthetic"], "n_clips": info["n_clips"],
                            "shared": bool(info.get("shared")), "progress": prog}
                    item.update(stage_info(info))
                    if bid in locked:
                        item["locked"] = True
                    items.append(item)
                listing = {"batches": items, "rater": self.who or None}
                if tolerance_ms:
                    listing["tolerance_s"] = tolerance_ms / 1000
                if manuals:
                    listing["manuals"] = {k[len("manual_"):-len(".html")]: k for k in manuals}
                if self.who == "":  # the owner's link never labels under a rater's own initials (1.7.7)
                    listing["rater_links"] = sorted(n for n in tokens if n)
                if self.who in admin_ok:
                    listing["admin"] = store.has_admin()
                return self.send(200, json.dumps(listing), "application/json")
            if len(parts) == 3 and parts[:2] == ["api", "admin"]:
                if self.who not in admin_ok:
                    return self.send(403, "Only the owner link opens the admin view.\n")
                if not visible(assignment, store.batches(), self.who, parts[2]):
                    return self.send(404, "No such batch.\n")
                try:
                    data = store.admin_bytes(parts[2])
                except Exception as e:
                    return self.send(500, f"{e}. Tell whoever runs the server.\n")
                if data is None:
                    return self.send(404, "No admin view for this batch.\n")
                return self.send(200, data, "application/octet-stream")
            if len(parts) == 3 and parts[:2] == ["api", "batch"]:
                if not visible(assignment, store.batches(), self.who, parts[2]):
                    return self.send(404, "No such batch.\n")
                if parts[2] in locked_for(self.who):
                    return self.send(403, LOCKED_MSG)
                try:
                    data = store.batch_bytes(parts[2])
                except Exception as e:
                    return self.send(500, f"{e}. Tell whoever runs the server.\n")
                if data is None:
                    return self.send(404, "No such batch.\n")
                return self.send(200, data, "application/octet-stream")
            if len(parts) == 4 and parts[:2] == ["api", "labels"]:
                bid, rater = parts[2], parts[3].upper()
                if bid not in store.batches() or not RATER_RE.match(rater) or \
                        not visible(assignment, store.batches(), self.who, bid):
                    return self.send(404, "No such batch or rater.\n")
                if self.who and rater != self.who:
                    return self.send(403, "This link belongs to another rater.\n")
                path = store.labels_path(bid, rater)
                if not os.path.exists(path):
                    return self.send(404, "No labels saved yet.\n")
                with open(path, "rb") as f:
                    return self.send(200, f.read(), "text/csv; charset=utf-8")
            return self.send(404, "Not found.\n")

        do_HEAD = do_GET

        def do_PUT(self):
            if not self.authorized():
                return self.send(403, "Forbidden.\n")
            parts = self.route()
            if not (len(parts) == 4 and parts[:2] == ["api", "labels"]):
                return self.send(404, "Not found.\n")
            bid, rater = parts[2], parts[3].upper()
            if not RATER_RE.match(rater):
                return self.send(400, "Initials must be 1 to 12 letters or digits.\n")
            if self.who and rater != self.who:
                return self.send(403, "This link belongs to another rater.\n")
            if self.who == "" and rater in tokens:
                return self.send(403, f"Not saved: {rater} has their own link. On your link, use your own initials "
                                      "(the Rater button).\n")
            if bid in store.batches() and not visible(assignment, store.batches(), self.who, bid):
                return self.send(403, "This batch is not one of yours.\n")
            if bid in locked_for(self.who):
                return self.send(403, "Not saved: " + LOCKED_MSG[0].lower() + LOCKED_MSG[1:])
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_LABELS_BYTES:
                return self.send(413, "Labels file is empty or too large.\n")
            body = self.rfile.read(n)
            try:
                answered, total = store.save_labels(bid, rater, body)
            except (ValueError, UnicodeDecodeError, KeyError, csv.Error) as e:
                return self.send(400, f"Not saved: {e}.\n")
            return self.send(200, json.dumps({"saved": True, "answered": answered, "total": total,
                                              "at": dt.datetime.now(dt.timezone.utc).isoformat()}),
                             "application/json")

        def do_POST(self):
            return self.do_PUT()

    return Handler


def load_token(labels_dir, new):
    path = os.path.join(labels_dir, ".token")
    if not new and os.path.exists(path):
        with open(path, encoding="ascii") as f:
            tok = f.read().strip()
        if len(tok) >= 20:
            return tok
    tok = secrets.token_urlsafe(24)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="ascii") as f:
        f.write(tok + "\n")
    return tok


def load_rater_tokens(labels_dir, new, add=(), revoke=()):
    """{RATER: token} from <labels>/.raters.json, adding and removing raters as asked."""
    path = os.path.join(labels_dir, ".raters.json")
    raters = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            raters = json.load(f)
    changed = False
    for r in add:
        r = r.strip().upper()
        if not r:
            continue
        if not RATER_RE.match(r):
            sys.exit(f"rater initials {r!r} must be 1 to 12 letters or digits")
        if r not in raters:
            raters[r] = secrets.token_urlsafe(24)
            changed = True
    for r in revoke:
        if raters.pop(r.strip().upper(), None) is not None:
            changed = True
    if new and raters:  # every link changes, the same raters stay
        raters = {r: secrets.token_urlsafe(24) for r in raters}
        changed = True
    if changed:
        _write_private(path, json.dumps(raters, indent=1))
    return raters


def load_admin_raters(labels_dir, asked, raters):
    """Raters whose own links also open the admin view, kept in <labels>/.admin_raters.json. asked is the
    --admin-raters value: None keeps the saved list, "none" clears it, else it replaces it."""
    path = os.path.join(labels_dir, ".admin_raters.json")
    saved = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
    if asked is None:
        return sorted(r for r in saved if r in raters)
    names = [] if asked.strip().lower() == "none" else sorted({r.strip().upper() for r in asked.split(",") if r.strip()})
    missing = [r for r in names if r not in raters]
    if missing:
        sys.exit(f"--admin-raters: {', '.join(missing)} has no rater link; add them with --raters first")
    if names != saved:
        _write_private(path, json.dumps(names))
    return names


def assignment_path(labels_dir):
    return os.path.join(labels_dir, ".assignment.json")


def load_assignment(labels_dir):
    path = assignment_path(labels_dir)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        a = json.load(f)
    if a.get("format") != ASSIGN_FORMAT:
        sys.exit(f"{path} is not a batch assignment")
    return a


def split_batches(batches, raters, keep=None):
    """{rater: [batch ids]} for the batches that are not shared, dealt so every rater ends with about the same
    number of clips. keep = {batch: rater} for batches that stay where they are (someone has started them)."""
    raters = sorted(raters)
    by = {r: [] for r in raters}
    load = {r: 0 for r in raters}
    for bid, r in (keep or {}).items():
        by.setdefault(r, []).append(bid)
        load[r] = load.get(r, 0) + batches[bid]["n_clips"]
    for bid, info in batches.items():
        if info.get("shared") or bid in (keep or {}) or not raters:
            continue
        r = min(raters, key=lambda x: (load[x], x))
        by[r].append(bid)
        load[r] += info["n_clips"]
    return {r: sorted(b) for r, b in by.items()}


def make_assignment(store, raters, old=None, reassign=False):
    """The batch split for a study with shared batches, or None when every rater sees every batch.
    An existing split is kept unless reassign is set; then only batches nobody has started move."""
    batches = store.batches()
    shared = sorted(b for b, i in batches.items() if i.get("shared"))
    if not shared:
        return None
    if old and not reassign:
        return old
    raters = [r for r in raters if r]
    if not raters:
        return old
    keep = {}
    for r, bids in (old or {}).get("by_rater", {}).items():
        for bid in bids:
            if bid in batches and any(store.progress(bid).values()):
                keep[bid] = r
    return {"format": ASSIGN_FORMAT, "version": 1,
            "created": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "shared": shared, "by_rater": split_batches(batches, raters, keep)}


def listing_order(bid, info):
    """The parts of a staged study in order; within each, shared batches first (everyone's agreement rests on
    them), then by number: b2 before b10."""
    return (info.get("stage") or 0, not info.get("shared"), len(bid), bid)


def visible(assignment, batches, who, bid):
    """Whether a rater (who) may see a batch. The owner ("") and studies without a split see everything."""
    if not who or assignment is None:
        return True
    info = batches.get(bid)
    return info is not None and (bool(info.get("shared")) or bid in assignment.get("by_rater", {}).get(who, []))


def describe_assignment(assignment, batches, raters):
    if assignment is None:
        return
    shared = assignment["shared"]
    n_shared = sum(batches[b]["n_clips"] for b in shared if b in batches)
    print(f"shared batches {', '.join(shared)} ({n_shared} clips) go to every rater")
    for r, bids in sorted(assignment["by_rater"].items()):
        n = sum(batches[b]["n_clips"] for b in bids if b in batches)
        gone = "" if r in raters else " (link revoked)"
        print(f"    {r:<6} {len(bids)} more batch(es), {n} clips{gone}")
    late = sorted(r for r in raters if r and r not in assignment["by_rater"])
    if late:
        print(f"note: {', '.join(late)} joined after the split and see the shared batches only; "
              f"--reassign hands them a share of the batches nobody has started")


def describe_stages(batches):
    """One line per part of a staged study (kit 1.8)."""
    parts = {}
    for info in batches.values():
        if info.get("stage"):
            p = parts.setdefault(info["stage"], [info.get("stage_name") or "", 0, 0])
            p[1] += 1
            p[2] += info["n_clips"]
    if parts:
        print("parts, in the order every rater meets them (each opens once the one before is answered):")
        for st, (name, nb, nc) in sorted(parts.items()):
            print(f"    {st}. {name}: {nb} batch(es), {nc} clips")


def print_links(port, owner, raters, ssh_host, admin_raters=()):
    print("on each rater's own computer, open the tunnel first (leave it running), with their own Borel login:")
    print(f"    ssh -N -L {port}:localhost:{port} {getpass.getuser()}@{ssh_host}")
    print("your link (any initials, all progress):")
    print(f"    http://localhost:{port}/?token={owner}")
    if raters:
        print("rater links (each opens signed in as that rater; send each person only their own):")
        for r, tok in sorted(raters.items()):
            print(f"    {r:<6} http://localhost:{port}/?token={tok}" + ("   (also opens the admin view)" if r in admin_raters else ""))


def print_status(store, assignment=None):
    batches = store.batches()
    if not batches:
        print(f"no servable batches in {store.batch_dir}")
        return
    raters = sorted({r for bid in batches for r in store.progress(bid)})
    owner_of = {b: r for r, bids in (assignment or {}).get("by_rater", {}).items() for b in bids}
    print(f"{'batch':<10} {'clips':>5}  {'for':<8}" + "  ".join(f"{r:>8}" for r in raters))
    for bid, info in batches.items():
        prog = store.progress(bid)
        who = "everyone" if assignment is None or info.get("shared") else owner_of.get(bid, "-")
        print(f"{bid:<10} {info['n_clips']:>5}  {who:<8}" + "  ".join(f"{prog.get(r, 0):>8}" for r in raters))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("batches", nargs="?", help="folder of batch files from make_review_batches.py pack")
    ap.add_argument("--labels", required=True, help="folder where labels are saved (created if missing)")
    ap.add_argument("--port", type=int, default=8765, help="port on 127.0.0.1 (default 8765)")
    ap.add_argument("--viewer", default=os.path.join(HERE, "ied_review.html"), help="viewer HTML to serve")
    ap.add_argument("--allow-research", action="store_true", help="also serve research (unblinded) batch files")
    ap.add_argument("--new-token", action="store_true", help="replace your token and every rater's; old links stop working")
    ap.add_argument("--raters", default="", help="comma-separated initials that get their own link, e.g. EC,JK")
    ap.add_argument("--revoke", default="", help="comma-separated initials whose links stop working")
    ap.add_argument("--links", action="store_true", help="print the links and exit")
    ap.add_argument("--ssh-host", default="borel.seas.upenn.edu", help="host name printed in the tunnel command")
    ap.add_argument("--status", action="store_true", help="print each rater's progress and exit")
    ap.add_argument("--reassign", action="store_true",
                    help="deal the split batches nobody has started again, over the raters with links now")
    add_live_args(ap)
    args = ap.parse_args(argv)
    return serve(ap, args)


def add_live_args(ap):
    live = ap.add_argument_group("live mode: read clips straight from the EEG files (used when the study is first planned)")
    live.add_argument("--clips", help="clip table from select_clips.py sample (source_path, center_s, ...)")
    live.add_argument("--study", help="folder for the study plan and key.csv (created on first start)")
    live.add_argument("--batch-size", type=int, default=100, help="clips per batch (default 100)")
    live.add_argument("--repeat-frac", type=float, default=0.0, help="fraction of clips shown twice under new IDs")
    live.add_argument("--seed", type=int, default=2026, help="seed for clip IDs, order and repeats (default 2026)")
    live.add_argument("--batch-prefix", default="b", help="batch IDs are this prefix plus a number (default b)")
    live.add_argument("--fs", type=float, default=200.0, help="sampling rate sent to the viewer, Hz (default 200)")
    live.add_argument("--clip-s", type=float, default=12.0, help="clip length including padding, s (default 12)")
    live.add_argument("--pad-s", type=float, default=1.0, help="padding on each side, s (default 1)")
    live.add_argument("--candidate-s", type=float, default=1.0, help="width of the highlighted window, s (default 1)")
    live.add_argument("--extras", nargs="*", default=["EKG"], help="non-scalp channels to show (default EKG)")
    live.add_argument("--reader", default="edf", help="file reader from make_review_batches.py (default edf)")
    live.add_argument("--notch", choices=["on", "off"], default="on",
                      help="60 Hz notch when a rater opens a batch (default on; they can switch it). Not part of the "
                           "plan, so it can change between starts")
    live.add_argument("--answers", choices=["yn", "ynu"], default=None,
                      help="yn: Yes or No, each moves to the next clip (default for a new study). ynu: also Unsure")
    live.add_argument("--question", choices=["clip", "second"], default=None,
                      help="what raters are asked, fixed when the study is planned: clip (an IED anywhere in the 10 s "
                           "shown; default for a new study) or second (in the highlighted second, as before kit 1.7)")
    live.add_argument("--admin", help="the admin table from select_clips.py admin: SubQ lanes, implant side and detector "
                                     "strips that only your own link can open (S and D in the viewer)")
    live.add_argument("--admin-raters", default=None, metavar="INITIALS",
                      help="raters whose own links also open the admin view (S and D), e.g. ADV for an advisor's "
                           "first look; remembered in the labels folder; none takes it away again")
    page = ap.add_argument_group("rater page")
    page.add_argument("--tolerance-ms", type=int, default=None, metavar="MS",
                      help="tell raters to count a spike whose peak is within this many ms of the highlighted second, "
                           "and draw that margin (kit 1.8; e.g. 250). Not part of the plan; remembered in the labels "
                           "folder until changed; 0 turns it off")
    page.add_argument("--manual", action="append", default=None, metavar="PART=FILE.md",
                      help="a rater manual as a Markdown file (kit 1.8.4): scalp=FILE for the 10-20 clips, uneeg=FILE "
                           "for the training examples and UNEEG clips (a bare FILE is the 10-20 one); give it twice "
                           "for both. Raters get a Manual button that opens the one for their part, and it opens by "
                           "itself when they start a part. Remembered in the labels folder; give it again after "
                           "editing; none removes both")
    live.add_argument("--shared-clips", type=int, default=None,
                      help="clips every rater labels, in batches s01, s02, ... (default 0); the rest are split "
                           "between raters")


def open_store(ap, args):
    """The store for a batch folder or a live study; plans the study on first use unless only reporting."""
    if bool(args.batches) == bool(args.clips):
        ap.error("give either a folder of batch files or --clips (with --study), not both")
    if args.clips and not args.study:
        ap.error("--clips needs --study, the folder that holds the plan and key.csv")
    if args.shared_clips is not None and args.shared_clips < 0:
        ap.error("--shared-clips cannot be negative")
    if args.shared_clips and args.batch_prefix == "s":
        ap.error("batch prefix s is kept for shared batches; pick another --batch-prefix")
    os.makedirs(args.labels, mode=0o700, exist_ok=True)
    quiet = getattr(args, "status", False) or getattr(args, "links", False)
    if args.clips:
        return LiveStore(args.clips, args.study, args.labels, args, prepare=not quiet)
    if not os.path.isdir(args.batches):
        sys.exit(f"{args.batches} is not a folder")
    return Store(args.batches, args.labels, args.allow_research)


def settle_assignment(store, labels_dir, raters, reassign):
    """Loads the batch split, makes it the first time raters exist, and saves any change."""
    old = load_assignment(labels_dir)
    assignment = make_assignment(store, list(raters), old, reassign)
    if assignment is not None and assignment is not old:
        _write_private(assignment_path(labels_dir), json.dumps(assignment, indent=1))
    if assignment is None and any(i.get("shared") for i in store.batches().values()):
        print("note: this study has shared batches but no raters yet; the split is made once --raters names them")
    return assignment


def serve(ap, args):
    if not os.path.isfile(args.viewer):
        sys.exit(f"viewer not found at {args.viewer}; run python build.py first")
    store = open_store(ap, args)
    if args.status:
        print_status(store, load_assignment(args.labels))
        return
    owner = load_token(args.labels, args.new_token)
    raters = load_rater_tokens(args.labels, args.new_token, args.raters.split(","), args.revoke.split(","))
    admin_raters = load_admin_raters(args.labels, args.admin_raters, raters)
    if args.links:
        print_links(args.port, owner, raters, args.ssh_host, admin_raters)
        return
    batches = store.batches()
    if not batches:
        sys.exit(f"no servable batch files in {args.batches}" +
                 ("" if args.allow_research else " (research batches need --allow-research)"))
    assignment = settle_assignment(store, args.labels, raters, args.reassign)
    tolerance_ms = load_tolerance(args.labels, args.tolerance_ms)
    manuals = {part: manual_page.page(text) for part, text in manual_page.load(args.labels, args.manual).items()}
    tokens = {"": owner, **raters}
    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                 make_handler(store, tokens, args.viewer, args.port, assignment, admin_raters,
                                              tolerance_ms, manuals))
    port = server.server_address[1]
    server.RequestHandlerClass = make_handler(store, tokens, args.viewer, port, assignment, admin_raters, tolerance_ms,
                                              manuals)
    what = "live from the EEG files" if args.clips else f"from {store.batch_dir}"
    print(f"serving {len(batches)} batch(es) {what}; labels go to {store.labels_dir}")
    describe_assignment(assignment, batches, raters)
    describe_stages(batches)
    if tolerance_ms:
        print(f"raters are told to count a spike whose peak is within {tolerance_ms} ms of the highlighted second")
    if manuals:
        print("raters see a Manual button: " + describe_manuals(manuals) + " (kept in the labels folder; "
              "--manual none removes them)")
    print_links(port, owner, raters, args.ssh_host, admin_raters)
    print("Ctrl+C stops the server; labels already saved stay saved.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
