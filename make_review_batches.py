#!/usr/bin/env python3
"""Package EEG clips into batch files for ied_review.html.

Each clip is a 12 s window of referential scalp EEG centered on a candidate second
(for example, one 1 s MORGOTH window). The viewer shows the middle 10 s and highlights
the candidate second; the 1 s of padding on each side keeps display filters from
ringing at the edges.

Two commands:

  synthetic   Writes batches of synthetic EEG with planted spikes, for testing the
              viewer and the label workflow without patient data.

      python make_review_batches.py synthetic --out demo_batches

  pack        Reads clip windows from EEG files listed in a CSV and writes batches.

      python make_review_batches.py pack clips.csv --out review_batches --mode rater

The clip table needs two columns: source_path (an EEG file) and center_s (the candidate
center, in seconds from the start of that file). Every other column is copied into
key.csv. In research mode those columns are also shown in the viewer as clip details,
and two optional columns add more: subq_path with subq_center_s (subcutaneous channels,
time already aligned to the scalp clock) and markers (a JSON list such as
[{"t": 6.02, "kind": "cyclops", "label": "Cyclops 0.91"}], with t in seconds from the
start of the 12 s clip).

Blinding: rater batches hold only random clip IDs and waveforms. Patient, time,
probabilities and the source file stay in key.csv, which should never leave the cluster.
"""
import argparse
import base64
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
import os
import re
import sys
from fractions import Fraction

import numpy as np

FORMAT = "ied-review-batch"
FORMAT_VERSION = 1
# batches of a staged study (kit 1.8: training examples, UNEEG-only clips) are version 2, so a viewer from before
# 1.8 refuses them instead of drawing a UNEEG clip as an empty scalp page
FORMAT_VERSION_STAGED = 2
PACKAGER_VERSION = "1.8.6"
# answers a rater can give; a study picks one set and every batch in it carries it
ANSWER_SETS = {"yn": ["yes", "no"], "ynu": ["yes", "no", "unsure"]}
# a training example is shown, not asked (kit 1.8.5, Svanik November 10: every example is a spike, so raters look
# through them); the viewer records "seen" for each one it shows, which opens the next part and is never scored
TEACHING_ANSWERS = ["seen"]

SCALP = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz",
         "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2"]
_SCALP_UPPER = {n.upper(): n for n in SCALP}
_ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
_SUFFIXES = ("REF", "LE", "AVG", "AV", "CAR", "A1", "A2", "M1", "M2", "A1A2", "LINKED")
UNIT_TO_UV = {"uv": 1.0, "µv": 1.0, "μv": 1.0, "microvolt": 1.0, "mv": 1e3, "v": 1e6, "nv": 1e-3}
RESERVED = {"source_path", "subq_path", "markers", "edf_clock"}
# The parts of a staged study (kit 1.8), from a clip table's part column, in the order raters meet them: each
# part opens once the rater has answered every clip of the parts before it. Since 1.8.4 the order is scalp, then
# examples, then uneeg (Svanik, November 10: the 10-20 clips first, then the examples as the lesson right before
# the UNEEG clips). scalp: the full 10-20 clips. examples: clips with the expert's answer (ref_answer, and
# ref_note if any) shown after the rater answers, not scored; since 1.8.1 they show the 10-20 channels with the
# UNEEG channels under them (view both) when the study has UNEEG clips, so raters see the comparison. uneeg: the
# same kind of second shown on the UNEEG channels only (subq_path at subq_center_s).
PARTS = {
    "scalp": {"stage": 1, "name": "10-20 clips", "prefix": "s", "view": "scalp", "training": False},
    "examples": {"stage": 2, "name": "UNEEG training examples", "prefix": "e", "view": "both", "training": True},
    "uneeg": {"stage": 3, "name": "UNEEG clips", "prefix": "u", "view": "subq", "training": False},
}
REF_ANSWERS = ("yes", "no")
MAX_REF_NOTE = 300
# the kind of spike a training example shows (kit 1.8.5), shown with it: generalized, focal left or focal right
REF_TYPES = ("generalized", "focal left", "focal right")


def ref_type(value):
    """A ref_type cell as one of REF_TYPES ("" when empty, None when it is something else)."""
    v = " ".join(re.sub(r"[^a-z]+", " ", str(value or "").lower()).split())
    if not v or v in ("nan", "none", "null"):
        return ""
    v = {"focal l": "focal left", "focal r": "focal right", "left focal": "focal left",
         "right focal": "focal right", "left": "focal left", "right": "focal right", "gen": "generalized",
         "generalised": "generalized"}.get(v, v)
    return v if v in REF_TYPES else None


def part_of(row):
    """The part of a staged study a clip-table row belongs to ("" when the table has no part column)."""
    return str(row.get("part") or "").strip().lower()


def check_parts(rows):
    """Problems with a staged clip table (a part column), one line each; empty when it is fine."""
    problems = []
    for i, row in enumerate(rows):
        part = part_of(row)
        where = f"row {i + 2}"
        if part not in PARTS:
            problems.append(f"{where}: part {part!r} is not one of {', '.join(PARTS)}")
            continue
        if part == "examples" and str(row.get("ref_answer") or "").strip().lower() not in REF_ANSWERS:
            problems.append(f"{where}: a training example needs ref_answer yes or no")
        if part == "examples" and len(str(row.get("ref_note") or "")) > MAX_REF_NOTE:
            problems.append(f"{where}: ref_note is over {MAX_REF_NOTE} characters")
        if part == "examples" and ref_type(row.get("ref_type")) is None:
            problems.append(f"{where}: ref_type must be one of {', '.join(REF_TYPES)} (or empty)")
        if part == "uneeg" and not (row.get("subq_path") and row.get("subq_center_s")):
            problems.append(f"{where}: a UNEEG clip needs subq_path and subq_center_s")
    if rows and not any(part_of(r) == "scalp" for r in rows):
        problems.append("no scalp clips (part scalp)")
    return problems


def ref_fields(row):
    """The expert's read of a training example (kit 1.8.5: shown with it): the answer, the kind of spike, a note."""
    out = {"ref": str(row.get("ref_answer") or "").strip().lower()}
    kind = ref_type(row.get("ref_type"))
    if kind:
        out["ref_type"] = kind
    note = " ".join(str(row.get("ref_note") or "").split())[:MAX_REF_NOTE]
    if note:
        out["ref_note"] = note
    return out

POS = {
    "Fp1": (-0.247, 0.761), "Fp2": (0.247, 0.761), "F7": (-0.647, 0.470), "F3": (-0.320, 0.420),
    "Fz": (0.0, 0.400), "F4": (0.320, 0.420), "F8": (0.647, 0.470), "T3": (-0.800, 0.0),
    "C3": (-0.400, 0.0), "Cz": (0.0, 0.0), "C4": (0.400, 0.0), "T4": (0.800, 0.0),
    "T5": (-0.647, -0.470), "P3": (-0.320, -0.420), "Pz": (0.0, -0.400), "P4": (0.320, -0.420),
    "T6": (0.647, -0.470), "O1": (-0.247, -0.761), "O2": (0.247, -0.761),
}


def normalize_name(raw):
    """Maps EDF-style labels ("EEG FP1-REF", "T7", "ECG EKG-REF") to classic 10-20 names."""
    s = str(raw).strip()
    if s.upper().startswith("EEG"):
        s = s[3:].lstrip(" _:-")
    for suf in _SUFFIXES:
        for sep in ("-", "_", " "):
            if s.upper().endswith(sep + suf):
                s = s[: -(len(suf) + 1)].rstrip(" _-")
                break
    up = s.upper().replace(" ", "")
    if up in _SCALP_UPPER:
        return _SCALP_UPPER[up]
    if up in _ALIASES:
        return _ALIASES[up]
    if "EKG" in up or "ECG" in up:
        return "EKG"
    return s


def resample_to(x, fs_in, fs_out, n_out):
    """Polyphase resampling to fs_out, trimmed or edge-padded to exactly n_out samples."""
    x = np.asarray(x, dtype=np.float64)
    if abs(fs_in - fs_out) > 1e-9:
        from scipy.signal import resample_poly
        frac = Fraction(fs_out / fs_in).limit_denominator(1000)
        x = resample_poly(x, frac.numerator, frac.denominator)
    if len(x) >= n_out:
        return x[:n_out]
    return np.pad(x, (0, n_out - len(x)), mode="edge")


# ---------------------------------------------------------------- readers

def _edf_reserved_field(path):
    with open(path, "rb") as f:
        head = f.read(256)
    return head[192:236].decode("ascii", "replace").strip()


class EDFReader:
    """Reads windows from EDF or BDF files. Returns microvolts.

    EDF+D (discontinuous) files go through edfplus.EDFPlus, which places every data record
    at its own onset: t0 and t1 are then seconds on the recording clock (the same clock as
    the file's annotation onsets), and a window that touches a gap is refused. Contiguous
    files are read with pyedflib, or with EDFPlus when pyedflib refuses them (a short last
    record). With clock="onsets" (a clip table's edf_clock column) every EDF+ file is read on
    its record onsets, EDF+C included, the clock the UNEEG detector runs use."""

    def __init__(self):
        self._files = {}

    def _open(self, path, clock=None):
        key = (path, clock or "auto")
        if key not in self._files:
            kind = _edf_reserved_field(path)
            if kind.startswith("BDF+D"):
                raise ValueError(f"{path} is a discontinuous BDF file, which this reader does not handle")
            if clock not in (None, "", "auto", "samples", "onsets"):
                raise ValueError(f"unknown edf_clock {clock!r}; use samples or onsets")
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from edfplus import EDFPlus
            if kind.startswith("EDF+D") or clock == "onsets":
                self._files[key] = EDFPlus(path, clock="onsets" if clock == "onsets" else "auto")
            else:
                import pyedflib
                # annotations are not needed for a contiguous file; skipping them keeps an open to its header
                mode = getattr(pyedflib, "DO_NOT_READ_ANNOTATIONS", None)
                try:
                    self._files[key] = pyedflib.EdfReader(path) if mode is None else pyedflib.EdfReader(path, mode)
                except (OSError, ValueError):
                    with open(path, "rb") as f:
                        if f.read(1) == b"\xff":
                            raise
                    self._files[key] = EDFPlus(path)  # same clock: 0 is the first sample
        return self._files[key]

    def read(self, path, t0, t1, clock=None):
        f = self._open(path, clock)
        if hasattr(f, "segments"):
            return f.read(t0, t1)
        nsamp = f.getNSamples()
        out = []
        for i, lab in enumerate(f.getSignalLabels()):
            fs_i = float(f.getSampleFrequency(i))
            start = int(round(t0 * fs_i))
            n = int(round((t1 - t0) * fs_i))
            if start < 0 or start + n > nsamp[i]:
                raise ValueError(f"window {t0:.2f}-{t1:.2f} s falls outside {path} "
                                 f"({nsamp[i] / fs_i:.1f} s long)")
            x = f.readSignal(i, start=start, n=n)
            unit = f.getPhysicalDimension(i).strip().lower()
            out.append((lab, fs_i, x * UNIT_TO_UV.get(unit, 1.0), unit))
        return out

    def close(self):
        for f in self._files.values():
            f.close()
        self._files = {}


class MNEReader:
    """Reads windows from anything mne.io.read_raw opens (EDF, BDF, FIF, BrainVision, ...)."""

    def __init__(self):
        self._raws = {}

    def read(self, path, t0, t1, clock=None):
        if clock == "onsets":
            raise ValueError("the mne reader does not follow EDF record onsets; use --reader edf for an onsets clip table")
        import mne
        if path not in self._raws:
            self._raws[path] = mne.io.read_raw(path, preload=False, verbose="ERROR")
        raw = self._raws[path]
        fs = float(raw.info["sfreq"])
        start = int(round(t0 * fs))
        stop = start + int(round((t1 - t0) * fs))
        if start < 0 or stop > raw.n_times:
            raise ValueError(f"window {t0:.2f}-{t1:.2f} s falls outside {path}")
        data = raw.get_data(start=start, stop=stop) * 1e6  # MNE works in volts
        return [(ch, fs, data[i], "uv") for i, ch in enumerate(raw.ch_names)]

    def close(self):
        self._raws = {}


READERS = {"edf": EDFReader, "mne": MNEReader}
# To read another format (for example the arrays the seizure-paper pipeline wrote), add a
# class with read(path, t0, t1) -> [(label, fs, data_in_uV, unit), ...] and register it here.


# ---------------------------------------------------------------- clip assembly

def stuck(x):
    """True when one value fills over a quarter of a channel's samples: a contact that is mostly a constant, such
    as every other sample stuck at one value, which no EEG does at its own sampling rate."""
    x = np.asarray(x)
    if x.size < 16:
        return False
    return np.unique(x, return_counts=True)[1].max() > 0.25 * x.size


def assemble(signals, fs_out, n_out, extras):
    """Returns (rows, names, bad): scalp channels in SCALP order, then the requested extras. Bad channels are
    missing, not finite, flat, or stuck (judged at the file's own rate, before resampling)."""
    by_name = {}
    for lab, fs_i, x, _unit in signals:
        name = normalize_name(lab)
        if name not in by_name:
            by_name[name] = (fs_i, x)
    rows, bad = [], []
    names = SCALP + list(extras)
    for name in names:
        if name in by_name:
            fs_i, x = by_name[name]
            y = resample_to(x, fs_i, fs_out, n_out)
            if not np.all(np.isfinite(y)):
                y = np.nan_to_num(y)
                bad.append(name)
            elif np.std(y) < 0.05 or stuck(x):
                bad.append(name)
        else:
            y = np.zeros(n_out)
            bad.append(name)
        rows.append(y)
    return np.vstack(rows), names, bad


NOISY_RATIO, NOISY_FLOOR_UV = 8.0, 100.0


def noisy_channels(rows, names, fs):
    """Scalp electrodes whose EEG-band amplitude (1-70 Hz, 60 Hz notched) is over 8 times the clip's median
    electrode and over 100 µV: a loose or broken contact. Drawn dashed like a missing channel, because one such
    trace scribbles over the whole page. A discharge lasts a fraction of the clip and barely moves this robust
    measure, and a real field spans several electrodes."""
    try:
        from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos
    except ImportError:
        return []
    sos = butter(4, [1.0, min(70.0, 0.45 * fs)], btype="bandpass", fs=fs, output="sos")
    if fs / 2 > 62:
        sos = np.vstack([sos, tf2sos(*iirnotch(60.0, 30.0, fs=fs))])
    amp = {}
    for name, y in zip(names, rows):
        if name in SCALP and np.std(y) >= 0.05:
            amp[name] = float(robust_amp(sosfiltfilt(sos, y)[None, :])[0])
    if len(amp) < 5:
        return []
    limit = max(NOISY_RATIO * float(np.median(list(amp.values()))), NOISY_FLOOR_UV)
    return [n for n, a in amp.items() if a > limit]


def read_clip(reader, source_path, center_s, args, extras, clock=None):
    """One clip's scalp channels (and extras) at args.fs. Returns (rows in µV, bad channel names, source units).
    Bad channels are missing, flat or noisy (see noisy_channels); the viewer draws them dashed.
    clock is the clip table's edf_clock (samples, the default, or onsets)."""
    half = args.clip_s / 2
    sig = reader.read(source_path, center_s - half, center_s + half, clock=clock) if clock else \
        reader.read(source_path, center_s - half, center_s + half)
    rows, names, bad = assemble(sig, args.fs, int(round(args.clip_s * args.fs)), extras)
    bad += [n for n in noisy_channels(rows, names, args.fs) if n not in bad]
    return rows, bad, {u for *_x, u in sig}


# the finest step stored, µV: far below what any display sensitivity shows, and coarse enough that batches
# compress to about three quarters of full int16 precision (kit 1.7, so ~2,200 clips fit on the hosted site)
MIN_STEP_UV = 0.1


def encode(rows):
    """Removes each row's mean and stores it as little-endian int16 with a per-row scale (µV per step): the
    peak over 32000, but never finer than MIN_STEP_UV."""
    rows = np.asarray(rows, dtype=np.float64)
    rows = rows - rows.mean(axis=1, keepdims=True)
    peak = np.max(np.abs(rows), axis=1)
    scale = np.array([float(f"{max(p / 32000.0, MIN_STEP_UV):.6g}") if p > 0 else 1.0 for p in peak])
    q = np.round(rows / scale[:, None]).astype("<i2")
    return base64.b64encode(q.tobytes()).decode("ascii"), [float(s) for s in scale], rows


def decode(b64, n_ch, n_s, scale):
    q = np.frombuffer(base64.b64decode(b64), dtype="<i2").reshape(n_ch, n_s)
    return q.astype(np.float64) * np.asarray(scale)[:, None]


def robust_amp(rows):
    med = np.median(rows, axis=1, keepdims=True)
    return np.median(np.abs(rows - med), axis=1) * 1.4826


def write_batch(path, obj, gz):
    data = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    if gz:
        with gzip.open(path, "wb", compresslevel=6) as f:
            f.write(data)
    else:
        with open(path, "wb") as f:
            f.write(data)


def read_batch(path):
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def table_salt(rows):
    """A number fixed by the clip table's contents, mixed into the seed so two studies drawn from different
    tables never share clip IDs (a browser keeps answers by clip ID), while the same table and seed always
    give the same IDs."""
    h = hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little")


def unique_ids(rng, n, taken=()):
    ids, seen = [], set(taken)
    while len(ids) < n:
        cid = "c%07x" % int(rng.integers(0, 16 ** 7))
        if cid not in seen:
            seen.add(cid)
            ids.append(cid)
    return ids


def place_repeats(order, originals, rng, min_gap):
    """Inserts repeated clips at least min_gap positions away from their originals."""
    order = list(order)
    for rep, orig in originals:
        pos = order.index(orig)
        allowed = [i for i in range(len(order) + 1) if abs(i - pos) > min_gap]
        order.insert(int(rng.choice(allowed)) if allowed else len(order), rep)
    return order


# ---------------------------------------------------------------- packing

def plan_batches(n, mode, batch_size, repeat_frac, batch_prefix, rng, shared=0, shared_prefix="s", taken=()):
    """Clip IDs, order, repeats and batch membership, decided before any signal is read.

    Returns (ids, batches): ids[i] is the ID of input clip i; each batch is
    {"batch_id", "order_seed", "shared", "members": [{"id", "src", "repeat_of"}]} where src indexes the input
    clips. serve_review.py --clips uses the same plan, so a live study and a packed one match.

    shared > 0 puts the first `shared` clips of the shuffled order into their own batches (s01, s02, ...)
    that every rater labels, for agreement between raters; the rest go into b01, b02, ... to be split
    between raters. With shared = 0 the plan is exactly what earlier versions made. taken holds clip IDs already
    used (by other parts of a staged study), which no new clip gets."""
    ids = unique_ids(rng, n, taken=taken)
    slots = [{"id": ids[i], "src": i, "repeat_of": ""} for i in range(n)]
    order = list(range(n))
    if mode == "rater":
        order = [int(i) for i in rng.permutation(n)]
        n_rep = int(round(repeat_frac * n))
        if n_rep:
            rep_src = [int(i) for i in rng.choice(n, size=n_rep, replace=False)]
            rep_ids = unique_ids(rng, n_rep, taken=set(ids) | set(taken))
            reps = []
            for src, rid in zip(rep_src, rep_ids):
                slots.append({"id": rid, "src": src, "repeat_of": ids[src]})
                reps.append((len(slots) - 1, src))
            order = place_repeats(order, reps, rng, min_gap=max(10, batch_size // 3))
    shared = max(0, min(int(shared), len(order))) if mode == "rater" else 0
    if shared and shared_prefix == batch_prefix:
        raise ValueError("shared batches need a prefix of their own")
    batches = []
    for prefix, part, is_shared in ((shared_prefix, order[:shared], True), (batch_prefix, order[shared:], False)):
        if not part and (is_shared or batches):
            continue
        for b in range(max(1, math.ceil(len(part) / batch_size))):
            members = [slots[i] for i in part[b * batch_size:(b + 1) * batch_size]]
            batches.append({"batch_id": f"{prefix}{b + 1:02d}", "order_seed": int(rng.integers(0, 2 ** 31 - 1)),
                            "shared": is_shared, "members": members})
    return ids, batches


def batch_groups(args):
    groups = {}
    for name in args.extras:
        groups.setdefault("EKG" if name == "EKG" else "Other", []).append(name)
    if args.subq_names:
        groups["SubQ"] = list(args.subq_names)
    return groups


def batch_header(args, batch_id, order_seed, synthetic, stamp, part=None):
    """Everything in a batch file except its clips. part names the part of a staged study (PARTS) it belongs to."""
    obj = {
        "format": FORMAT, "version": FORMAT_VERSION, "batch_id": batch_id, "mode": args.mode,
        "synthetic": bool(synthetic), "created": stamp, "packager_version": PACKAGER_VERSION,
        "fs": args.fs, "clip_s": args.clip_s, "pad_s": args.pad_s, "event_s": args.clip_s / 2,
        "candidate_s": args.candidate_s, "channels": SCALP + list(args.extras_all), "groups": batch_groups(args),
        "order_seed": order_seed, "dc_removed": True,
        # where the viewer starts when a batch opens; raters can change it while they review
        "display": {"notch": getattr(args, "notch", "on") == "on"},
        "answers": ANSWER_SETS[getattr(args, "answers", "yn")],
        # clip: is there an IED anywhere in the 10 s shown (kit 1.7); second: in the highlighted second (before)
        "question": getattr(args, "question", "second"),
        "notes": "Synthetic EEG with planted events." if synthetic else "",
        "clips": [],
    }
    if part:
        p = PARTS[part]
        obj.update({"version": FORMAT_VERSION_STAGED, "part": part, "stage": p["stage"], "stage_name": p["name"]})
        if p["view"] == "subq":
            # the UNEEG channels only: the viewer draws them as they are, with no scalp montage
            names = list(args.subq_channels)
            obj.update({"view": "subq", "channels": names, "groups": {"SubQ": names}})
        elif p["view"] == "both" and getattr(args, "subq_channels", None):
            # the scalp montage with the study's UNEEG channels under it (training examples, kit 1.8.1)
            names = list(args.subq_channels)
            obj.update({"view": "both", "channels": SCALP + list(args.extras_all) + names,
                        "groups": dict(batch_groups(args), SubQ=names)})
        if p["training"]:
            obj["training"] = True
            obj["answers"] = list(TEACHING_ANSWERS)
    return obj


def clip_item(c, cid, mode):
    """One clip as stored in a batch file. Returns (item, mean-removed rows, scale)."""
    b64, scale, centered = encode(c["rows"])
    item = {"id": cid, "event_s": round(float(c["event_s"]), 4), "scale": scale, "data": b64}
    if c["bad"]:
        item["bad"] = c["bad"]
    if mode == "research":
        if c.get("meta"):
            item["meta"] = c["meta"]
        if c.get("markers"):
            item["markers"] = c["markers"]
    return item, centered, scale


ADMIN_FORMAT = "ied-review-admin"
_LABEL_RE = re.compile(r"[^A-Za-z0-9-]")


def admin_item(reader, arow, cid, args, scalp_row=None):
    """The owner-only extras for one clip (the admin and superadmin views), from a row of the admin table that
    select_clips.py admin writes: the SubQ channels over the same 12 s, the implant side, which draw picked the
    clip, and each detector's per-second scores (strips). Holds no patient, file or time. scalp_row (kit 1.8.5) is
    the clip-table row of a UNEEG clip: its admin view carries the 10-20 channels of the same 12 s instead of the
    SubQ ones the rater already sees."""
    item = {"id": cid}
    if scalp_row is not None:
        rows, bad, _u = read_clip(reader, scalp_row["source_path"], float(scalp_row["center_s"]), args, args.extras,
                                  scalp_row.get("edf_clock") or None)
        b64, scale, _c = encode(rows)
        item["scalp"] = {"channels": SCALP + list(args.extras), "scale": scale, "data": b64}
        if bad:
            item["scalp"]["bad"] = list(bad)
    side = str(arow.get("implant_side") or "").strip().upper()[:1]
    if side in ("L", "R"):
        item["implant_side"] = side
    drawn = [d for d in str(arow.get("draws") or "").split(";") if d]
    if drawn:
        item["drawn_by"] = drawn
    if arow.get("strips"):
        item["strips"] = json.loads(arow["strips"])
    virtual = []
    for pair in str(arow.get("virtual") or "").split(";"):
        a, _d, b = pair.strip().partition("-")
        if a in SCALP and b in SCALP and a != b:
            virtual.append([a, b])
    if virtual:
        item["virtual"] = virtual  # the viewer derives these from the scalp channels it already has
    if scalp_row is None and arow.get("subq_path") and arow.get("subq_center_s"):
        names, rows, _raw = read_subq(reader, arow["subq_path"], float(arow["subq_center_s"]), args,
                                      arow.get("edf_clock") or None)
        if rows:
            b64, scale, _c = encode(np.vstack(rows))
            item["subq"] = {"channels": names, "scale": scale, "data": b64}
    return item


def read_subq(reader, path, center_s, args, clock=None):
    """The SubQ channels of one 12 s clip at args.fs: (names, rows, raw), with the outer pair added the way
    admin_item has shown it since 1.7.6 (raw is each channel at its own rate, None for the added pair)."""
    half = args.clip_s / 2
    sig = reader.read(path, center_s - half, center_s + half, clock=clock) if clock else \
        reader.read(path, center_s - half, center_s + half)
    n_out = int(round(args.clip_s * args.fs))
    names, rows, raw = [], [], []
    for lab, fs_i, x, _u in sig:
        name = _LABEL_RE.sub("", str(lab).strip().upper().replace("EEG", "", 1))[:12] or f"CH{len(names) + 1}"
        names.append(name)
        rows.append(resample_to(x, fs_i, args.fs, n_out))
        raw.append(x)
    outer = outer_pair(names)
    if outer:
        # the distant pair from the two bipolar channels (UNEEG D-C and P-C give D-P)
        label, sign = outer
        names.append(label)
        rows.append(rows[0] + sign * rows[1])
        raw.append(None)
    return names, rows, raw


_SQ_PAIR_RE = re.compile(r"^SQ([A-Z0-9]{1,4})-SQ([A-Z0-9]{1,4})$")


def short_subq(name):
    """UNEEG channel names as raters read them: SQD-SQC becomes D-C. Other names stay as they are."""
    m = _SQ_PAIR_RE.match(name)
    return f"{m.group(1)}-{m.group(2)}" if m else name


def read_subq_clip(reader, row, args, names=None):
    """A UNEEG-only clip (the uneeg part of a staged study) from a clip-table row's subq_path and subq_center_s.
    Returns (names, rows in µV, bad names). names fixes the channels and their order (the study's, from its
    first clip); a channel this clip lacks is zeros and listed as bad, like a missing scalp electrode."""
    if not row.get("subq_path") or not row.get("subq_center_s"):
        raise ValueError("no subq_path and subq_center_s for this UNEEG clip")
    got, rows, raw = read_subq(reader, row["subq_path"], float(row["subq_center_s"]), args, row.get("edf_clock") or None)
    if not got:
        raise ValueError("the SubQ file has no channels")
    by = {}
    for n, y, x in zip((short_subq(n) for n in got), rows, raw):
        by.setdefault(n, (y, x))
    names = list(names) if names else list(by)
    n_out = int(round(args.clip_s * args.fs))
    out, bad = [], []
    for n in names:
        if n not in by:
            out.append(np.zeros(n_out))
            bad.append(n)
            continue
        y, x = by[n]
        if not np.all(np.isfinite(y)):
            y = np.nan_to_num(y)
            bad.append(n)
        elif np.std(y) < 0.05 or (x is not None and stuck(x)):
            bad.append(n)
        out.append(y)
    return names, np.vstack(out), bad


def read_example(reader, row, args, names=None):
    """A training example (kit 1.8.1): the scalp clip with the UNEEG channels of the same seconds under it.
    Returns (rows, bad, names): names are the UNEEG channels read (the study's when given), [] when the study has
    none. An example with no subq_path in a study that has UNEEG channels gets zeros there, listed as bad (drawn
    dashed), like a missing electrode."""
    rows, bad, _u = read_clip(reader, row["source_path"], float(row["center_s"]), args, args.extras,
                              row.get("edf_clock"))
    if row.get("subq_path") and row.get("subq_center_s"):
        names, sq, sq_bad = read_subq_clip(reader, row, args, names or None)
        return np.vstack([rows, sq]), list(bad) + list(sq_bad), names
    names = list(names or [])
    if names:
        n_out = int(round(args.clip_s * args.fs))
        rows = np.vstack([rows, np.zeros((len(names), n_out))])
        bad = list(bad) + names
    return rows, bad, names


def outer_pair(names):
    """For two bipolar channels that share one contact, the third pair they make: (label, s) with the new
    channel = channel 1 + s * channel 2. UNEEG's D-C and P-C give D-P (s = -1); C-P and C-D also give D-P.
    None unless there are exactly two channels named X-Y that share exactly one contact and the pair is new."""
    if len(names) != 2:
        return None
    pairs = []
    for n in names:
        a, sep, b = n.partition("-")
        if not sep or not a or not b or "-" in b or a == b:
            return None
        pairs.append((a, b))
    for s in (-1, 1):
        coef = {}
        for (a, b), k in zip(pairs, (1, s)):
            coef[a] = coef.get(a, 0) + k
            coef[b] = coef.get(b, 0) - k
        live = {c: v for c, v in coef.items() if v}
        pos = [c for c, v in live.items() if v == 1]
        neg = [c for c, v in live.items() if v == -1]
        if len(live) == 2 and len(pos) == 1 and len(neg) == 1:
            label = f"{pos[0]}-{neg[0]}"[:12]
            if label in names or f"{neg[0]}-{pos[0]}" in names:
                return None
            return label, s
    return None


def admin_header(args, batch_id):
    return {"format": ADMIN_FORMAT, "version": 1, "batch_id": batch_id, "fs": args.fs, "clip_s": args.clip_s,
            "clips": []}


def utc_stamp():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def key_row(member, batch_id, fname, pos, bad, fields):
    row = {"clip_id": member["id"], "batch_id": batch_id, "batch_file": fname, "packed_position": pos + 1,
           "repeat_of": member["repeat_of"], "bad_channels": ";".join(bad)}
    row.update(fields)
    return row


def write_key(path, key_rows):
    fields = []
    for r in key_rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(key_rows)


def build_batches(clips, args, synthetic=False):
    """clips: list of dicts with 'rows' (n_ch x n_s µV), 'bad', 'event_s', 'row' (key fields),
    and in research mode optional 'meta' and 'markers'. Writes batches, key.csv, manifest.json."""
    rng = np.random.default_rng([args.seed, table_salt([c["row"] for c in clips])])
    os.makedirs(args.out, exist_ok=True)
    n = len(clips)
    _ids, plan = plan_batches(n, args.mode, args.batch_size, args.repeat_frac, args.batch_prefix, rng,
                              shared=getattr(args, "shared_clips", 0))

    stamp = utc_stamp()
    ext = ".json.gz" if args.gzip else ".json"
    channels = SCALP + list(args.extras_all)
    key_rows, manifest_batches, checks = [], [], []
    for pb in plan:
        batch_id = pb["batch_id"]
        fname = batch_id + ext
        obj = batch_header(args, batch_id, pb["order_seed"], synthetic, stamp)
        members = pb["members"]
        for pos, m in enumerate(members):
            c = clips[m["src"]]
            item, centered, scale = clip_item(c, m["id"], args.mode)
            obj["clips"].append(item)
            if pos in (0, len(members) - 1):
                checks.append((fname, m["id"], centered, scale))
            key_rows.append(key_row(m, batch_id, fname, pos, c["bad"],
                                    dict(c["row"], question=getattr(args, "question", "second"))))
        path = os.path.join(args.out, fname)
        write_batch(path, obj, args.gzip)
        manifest_batches.append({"file": fname, "batch_id": batch_id, "n_clips": len(members),
                                 "shared": pb["shared"], "bytes": os.path.getsize(path), "sha256": sha256(path)})
    order = [m for pb in plan for m in pb["members"]]

    # round-trip check: re-read files and compare decoded clips with what was encoded
    worst = 0.0
    for fname, cid, centered, scale in checks:
        obj = read_batch(os.path.join(args.out, fname))
        item = next(c for c in obj["clips"] if c["id"] == cid)
        n_s = int(round(obj["clip_s"] * obj["fs"]))
        back = decode(item["data"], len(obj["channels"]), n_s, item["scale"])
        err = np.max(np.abs(back - centered) / (np.asarray(scale)[:, None]))
        worst = max(worst, float(err))
    if worst > 0.51:
        raise RuntimeError(f"round-trip error {worst:.3f} quantization steps; batch files are not trustworthy")

    write_key(os.path.join(args.out, "key.csv"), key_rows)

    manifest = {
        "packager_version": PACKAGER_VERSION, "created": stamp, "mode": args.mode, "synthetic": bool(synthetic),
        "fs": args.fs, "clip_s": args.clip_s, "pad_s": args.pad_s, "candidate_s": args.candidate_s,
        "seed": args.seed, "n_clips": n, "n_repeats": len(order) - n, "channels": channels,
        "answers": ANSWER_SETS[getattr(args, "answers", "yn")],
        "shared_clips": sum(b["n_clips"] for b in manifest_batches if b["shared"]),
        "batches": manifest_batches, "key_file": "key.csv",
        "args": {k: v for k, v in vars(args).items() if k not in ("func",) and isinstance(v, (str, int, float, bool, list, type(None)))},
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total = sum(b["bytes"] for b in manifest_batches)
    print(f"wrote {len(manifest_batches)} batch file(s) to {args.out}: {len(order)} clips "
          f"({len(order) - n} repeats), {total / 1e6:.1f} MB")
    print(f"round trip ok: worst error {worst:.3f} quantization steps")
    if args.mode == "rater":
        print("key.csv maps clip IDs to sources and stays on the cluster; send raters only the batch files and ied_review.html")


def cmd_pack(args):
    import pandas as pd
    table = pd.read_csv(args.table, dtype=str, keep_default_na=False)
    for col in ("source_path", "center_s"):
        if col not in table.columns:
            sys.exit(f"{args.table} needs a '{col}' column")
    reader = READERS[args.reader]()
    n_out = int(round(args.clip_s * args.fs))
    half = args.clip_s / 2
    extras = [normalize_name(e) for e in args.extras]
    args.extras = extras
    clips, amps, failures = [], [], []
    subq_names = None
    for idx, rec in table.iterrows():
        row = dict(rec)
        try:
            rows, bad, units = read_clip(reader, row["source_path"], float(row["center_s"]), args, extras,
                                         row.get("edf_clock"))
            amps.append(float(np.median(robust_amp(rows[:len(SCALP)]))))
            c = {"rows": rows, "bad": bad, "event_s": half, "row": row}
            if args.mode == "research":
                meta = {k: v for k, v in row.items() if k not in RESERVED and v != ""}
                if units - {"uv", "µv", "μv"}:
                    meta["Source units"] = ",".join(sorted(units))
                c["meta"] = meta
                if row.get("markers"):
                    c["markers"] = json.loads(row["markers"])
                if row.get("subq_path") and row.get("subq_center_s"):
                    sc = float(row["subq_center_s"])
                    sq = reader.read(row["subq_path"], sc - half, sc + half, clock=row["edf_clock"]) \
                        if row.get("edf_clock") else reader.read(row["subq_path"], sc - half, sc + half)
                    labels = ["SQ " + str(lab).strip() for lab, *_r in sq]
                    if subq_names is None:
                        subq_names = labels
                    sq_rows = []
                    for name in subq_names:
                        hit = [(fs_i, x) for (lab, fs_i, x, _u), l2 in zip(sq, labels) if l2 == name]
                        sq_rows.append(resample_to(hit[0][1], hit[0][0], args.fs, n_out) if hit else np.zeros(n_out))
                    c["rows"] = np.vstack([rows] + sq_rows)
            clips.append(c)
        except Exception as e:  # keep going; report every failure at the end
            failures.append((idx, row.get("source_path", ""), str(e)))
    if hasattr(reader, "close"):
        reader.close()
    if args.mode == "research" and subq_names:
        for c in clips:
            if c["rows"].shape[0] == len(SCALP) + len(extras):
                c["rows"] = np.vstack([c["rows"], np.zeros((len(subq_names), n_out))])
                c["bad"] = c["bad"] + subq_names
    args.subq_names = subq_names or []
    args.extras_all = extras + args.subq_names
    if failures:
        print(f"{len(failures)} clip(s) could not be read:")
        for idx, path, msg in failures[:20]:
            print(f"  row {idx + 2}: {path}: {msg}")
        if not clips:
            sys.exit("no clips were read")
    if amps:
        med = float(np.median(amps))
        print(f"median robust scalp amplitude {med:.1f} µV across {len(amps)} clips")
        if med < 0.5 or med > 2000:
            print("WARNING: that amplitude is implausible for scalp EEG in microvolts. Check the units the "
                  "reader returns (a factor of 1e6 means volts were read as microvolts or the reverse).")
    build_batches(clips, args)


# ---------------------------------------------------------------- synthetic data

def _pink(n, rng):
    poles = np.array([0.995, 0.98, 0.94, 0.85, 0.6])
    gains = np.array([0.55, 0.65, 0.7, 0.55, 0.4])
    from scipy.signal import lfilter
    out = np.zeros(n)
    for p, g in zip(poles, gains):
        w = rng.standard_normal(n + 400) * g * math.sqrt(1 - p * p)
        out += lfilter([1.0], [1.0, -p], w)[400:]
    return out / (out.std() or 1.0)


def _spike(t, a):
    rise, fall = 0.013, 0.022
    s = np.where(t < 0, np.exp(-t ** 2 / (2 * rise ** 2)), np.exp(-t ** 2 / (2 * fall ** 2)))
    return -a * s + a * 0.28 * np.exp(-(t - 0.055) ** 2 / (2 * 0.018 ** 2)) - a * 0.45 * np.exp(-(t - 0.19) ** 2 / (2 * 0.065 ** 2))


def synth_clip(rng, kind, at, amp, fs, clip_s, dt_s):
    n = int(round(fs * clip_s))
    t = np.arange(n) / fs
    data = {name: np.zeros(n) for name in SCALP}
    for _ in range(7):
        cx, cy = rng.uniform(-0.7, 0.7, 2)
        w, a = rng.uniform(0.35, 0.7), rng.uniform(5, 11)
        src = _pink(n, rng)
        for name, (x, y) in POS.items():
            g = a * math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * w * w))
            if g > 0.05:
                data[name] += g * src
    f_alpha, ph = rng.uniform(9, 10.4), rng.uniform(0, 2 * math.pi)
    env = 0.55 + 0.45 * np.sin(2 * math.pi * 0.33 * t + rng.uniform(0, 6.28))
    alpha_w = {"O1": 30, "O2": 30, "P3": 19, "P4": 19, "Pz": 17, "T5": 17, "T6": 17, "C3": 7, "C4": 7, "Cz": 7}
    for name in SCALP:
        data[name] += alpha_w.get(name, 3) * env * np.sin(2 * math.pi * f_alpha * t + ph + (POS[name][1] + 0.76) * 0.25)
        data[name] += rng.standard_normal(n)
    ekg = np.zeros(n)
    tb = rng.uniform(0, 0.8)
    while tb < clip_s:
        ekg += 900 * np.exp(-(t - tb) ** 2 / (2 * 0.012 ** 2)) + 130 * np.exp(-(t - tb - 0.25) ** 2 / (2 * 0.05 ** 2))
        tb += rng.uniform(0.8, 0.95)
    t0 = clip_s / 2 + dt_s
    if kind == "spike":
        for name, (x, y) in POS.items():
            d2 = (x - POS[at][0]) ** 2 + (y - POS[at][1]) ** 2
            g = math.exp(-d2 / (2 * 0.36 ** 2))
            if g > 0.03:
                data[name] += _spike(t - t0, amp * g)
    elif kind == "blink":
        for name, (x, y) in POS.items():
            g = math.exp(-((x ** 2) * 0.6 + (y - 1.0) ** 2) / (2 * 0.16))
            bump = np.where((t >= t0 - 0.12) & (t <= t0 + 0.24), np.sin(np.clip((t - t0 + 0.12) / 0.36, 0, 1) * math.pi) ** 1.5, 0)
            data[name] += 140 * g * bump
    return np.vstack([data[name] for name in SCALP] + [ekg])


def cmd_synthetic(args):
    rng = np.random.default_rng(args.seed)
    sites = ["T3", "F7", "T5", "T4", "F8", "T6", "F3", "C4", "O1"]
    clips = []
    for i in range(args.n):
        u = rng.random()
        kind = "spike" if u < 0.45 else ("blink" if u < 0.6 else "none")
        at = str(rng.choice(sites)) if kind == "spike" else ""
        amp = float(rng.uniform(55, 150)) if kind == "spike" else 0.0
        dts = float(rng.uniform(-0.3, 0.3))
        rows = synth_clip(rng, kind, at, amp, args.fs, args.clip_s, dts)
        p = float(np.clip(rng.normal(0.85, 0.1), 0, 0.999)) if kind == "spike" else float(np.clip(rng.beta(1.2, 6), 0, 0.999))
        lo = min(9, int(p * 10)) * 10
        row = {"patient": f"SYN-{1 + i % 4:02d}", "center_s": f"{600 + 37.5 * i:.2f}",
               "morgoth_prob": f"{p:.3f}", "bin": f"{lo}-{lo + 10}%",
               "truth": kind, "truth_max": at, "truth_amp_uv": f"{amp:.0f}", "truth_dt_s": f"{dts:.3f}"}
        c = {"rows": rows, "bad": [], "event_s": args.clip_s / 2, "row": row}
        if args.mode == "research":
            c["meta"] = {"Patient": row["patient"], "MORGOTH p": row["morgoth_prob"], "Bin": row["bin"],
                         "Planted": f"{kind} {at}".strip()}
            c["markers"] = [{"t": args.clip_s / 2, "kind": "morgoth", "label": f"MORGOTH {p:.2f}"}]
        clips.append(c)
    args.extras = ["EKG"]
    args.extras_all = ["EKG"]
    args.subq_names = []
    build_batches(clips, args, synthetic=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--out", required=True, help="output folder for batch files, key.csv and manifest.json")
        p.add_argument("--mode", choices=["rater", "research"], default="rater",
                       help="rater: blinded, shuffled, random IDs only (default). research: unblinded, keeps order, adds details")
        p.add_argument("--fs", type=float, default=200.0, help="sampling rate written to the batch, Hz (default 200)")
        p.add_argument("--clip-s", type=float, default=12.0, help="clip length including padding, s (default 12)")
        p.add_argument("--pad-s", type=float, default=1.0, help="padding on each side, s (default 1)")
        p.add_argument("--candidate-s", type=float, default=1.0, help="width of the highlighted window, s (default 1)")
        p.add_argument("--batch-size", type=int, default=100, help="clips per batch file (default 100)")
        p.add_argument("--batch-prefix", default="b", help="batch IDs are this prefix plus a number (default b)")
        p.add_argument("--seed", type=int, default=2026, help="random seed for order, IDs and repeats")
        p.add_argument("--repeat-frac", type=float, default=0.0,
                       help="fraction of clips repeated under new IDs to measure test-retest agreement (rater mode)")
        p.add_argument("--no-gzip", dest="gzip", action="store_false", help="write plain .json instead of .json.gz")
        p.add_argument("--notch", choices=["on", "off"], default="on",
                       help="60 Hz notch when the viewer opens a batch (default on; raters can switch it)")
        p.add_argument("--answers", choices=sorted(ANSWER_SETS), default="yn",
                       help="yn: Yes or No, each moves to the next clip (default). ynu: also Unsure")
        p.add_argument("--question", choices=["clip", "second"], default="clip",
                       help="clip: an IED anywhere in the 10 s shown (default; no second is highlighted for raters). "
                            "second: an IED in the highlighted second (kit 1.6 and before)")
        p.add_argument("--shared-clips", type=int, default=0,
                       help="clips every rater labels (batches s01, s02, ...); the rest are split between raters")

    p_pack = sub.add_parser("pack", help="package clips listed in a CSV")
    p_pack.add_argument("table", help="CSV with source_path and center_s columns")
    p_pack.add_argument("--reader", choices=sorted(READERS), default="edf", help="file reader (default edf)")
    p_pack.add_argument("--extras", nargs="*", default=["EKG"], help="non-scalp channels to include (default EKG)")
    common(p_pack)
    p_pack.set_defaults(func=cmd_pack)

    p_syn = sub.add_parser("synthetic", help="write batches of synthetic EEG with planted spikes")
    p_syn.add_argument("--n", type=int, default=60, help="number of clips (default 60)")
    common(p_syn)
    p_syn.set_defaults(func=cmd_synthetic, batch_size=30)

    args = ap.parse_args(argv)
    if args.clip_s <= 2 * args.pad_s:
        ap.error("--clip-s must be longer than twice --pad-s")
    if args.shared_clips < 0:
        ap.error("--shared-clips cannot be negative")
    if args.shared_clips and args.batch_prefix == "s":
        ap.error("batch prefix s is kept for shared batches; pick another --batch-prefix")
    args.func(args)


if __name__ == "__main__":
    main()
