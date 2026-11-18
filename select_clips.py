#!/usr/bin/env python3
"""Draw a stratified, weighted clip table from per-second MORGOTH probabilities.

This is the step between Phase 4 (a spike probability for every second of scalp EEG) and
make_review_batches.py pack. It reads one row per scored second and writes clips.csv,
which pack reads as is.

    python select_clips.py sample morgoth_seconds.csv --out clips.csv --n 1000 --pilot 25

Design:
  * Every scored second is a candidate. Seconds too close to the start or end of the
    unbroken stretch of scored seconds they sit in (the file, or one segment of an EDF+D
    file) to fit a whole clip are dropped first (and counted).
  * Candidates fall into probability bins (--bins, deciles by default). Each bin gets an
    equal share of --n (--alloc equal) or a share proportional to its size (--alloc
    proportional); a bin with fewer candidates than its share gives the rest to the others.
  * Inside a bin, the share is split across patients so no patient dominates (--within
    balanced, the default: equal shares, capped by what each patient has) or in
    proportion to their candidates (--within proportional). When a bin has fewer clips
    than patients, the patients that get one are drawn at random (systematic sampling,
    each with a chance equal to its share, cell_p), never by ID order.
  * --min-gap-s keeps clips from the same file apart, so one discharge does not show up as
    two clips. It is off by default; with it on, weights still add up per cell, but each
    clip's chance of selection is only approximately n_cell / N_cell.
  * Sampling is without replacement inside each bin x patient cell, so every candidate in a
    cell had the same chance, n_cell / N_cell. When every patient in a bin gets a clip,
    sampling_weight is N_cell / n_cell, the number of candidate seconds each clip stands
    for, and weights in a cell add back up to the cell's candidate count. When patients
    were drawn, a clip's weight is N_cell / (n_cell * cell_p), scaled so the bin's weights
    add up to its candidate count (a ratio estimate): the drawn patients stand for the
    ones not drawn. Each bin's weights add up to its candidate count exactly; the script
    checks this and stops if it fails. Per-patient shares use N_cell / n_cell only.

Input columns (rename with the --*-col options): patient, source_path (the contiguous file
pack will read), start_s (start of the scored second, seconds from the start of that file)
and prob. Parquet works too if pandas can read it.

    python select_clips.py from-runs out/cyclops_emu --condition full19 --detector cyclops \
        --clip-root /path/to/edf/clips --admissions data_derived/emu_admissions.csv --out seconds.csv

turns a detector's output folder (one table per clip, or long tables with a clip column)
into that per-second table. Column names are found automatically (inspect shows what it
finds, with no values); times are moved to the start of each scored second on the EDF's
own clock and checked against each EDF's length; EMU IDs are mapped to patients through
the admissions table, so a patient with two admissions counts once.

    python select_clips.py join --out seconds.csv cyclops=cyclops_seconds.csv cyclops_sd=cyclops_sd_seconds.csv
    python select_clips.py sample seconds.csv --prob-cols cyclops_prob,cyclops_sd_prob --out clips.csv --n 1000

puts several scorings of the same seconds side by side (raw and SD-matched, or several
detectors), one <name>_prob column each, and samples with each second's stratum set by the
highest score any of them gives it. One set of expert labels then serves every scoring: the
weights make each one's estimates unbiased whatever the strata, and stratifying on all of
them keeps each one's high-score seconds well sampled, so none is estimated from a handful
of clips.

    python select_clips.py inspect out/cyclops_emu

    python select_clips.py from-edf --clip-root /path/to/edf/clips --pattern '*iic*' \
        --admissions data_derived/emu_admissions.csv --out seconds.csv

writes every whole second of the files with no score, for drawing clips before any detector
has run (sample then uses one bin: a simple random sample within each patient).

    python select_clips.py check-data --clip-root /path/to/edf/clips --n 20

reads one 12 s window from 20 of the files exactly as the review server will and prints
only aggregates: file types, gaps, sampling rates, units, which 10-20 channels were found,
and the median amplitude. It ends with OK or PROBLEM.

    python select_clips.py fake-morgoth --edf-dir /tmp/edf --out fake_morgoth.csv

writes made-up probabilities for every second of the test EDFs, for trying the loop
without patient data.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

DEFAULT_BINS = "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"


def read_table(path):
    if path.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def bin_labels(edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append(f"{lo * 100:g}-{hi * 100:g}%")
    return out


def share(total, caps, mode, sizes=None, rng=None):
    """Splits total across slots, never giving a slot more than its cap; leftovers go to the rest.

    mode 'equal' gives each slot the same share; 'proportional' follows sizes. Whole shares are
    handed out first. The last few clips, when there are fewer of them than slots that could take
    one (fewer clips than patients, say), go to slots drawn at random by systematic sampling, each
    with a chance equal to its fractional share; without rng they go to the first slots instead.

    Returns (got, p): clips per slot, and each slot's chance of getting at least one."""
    caps = np.asarray(caps, dtype=int)
    got = np.zeros(len(caps), dtype=int)
    p = np.zeros(len(caps))
    remaining = int(min(total, caps.sum()))
    open_ = caps > 0
    while remaining > 0 and open_.any():
        if mode == "proportional":
            w = np.where(open_, np.asarray(sizes, dtype=float), 0.0)
        else:
            w = open_.astype(float)
        ideal = remaining * w / w.sum()
        add = np.minimum(np.floor(ideal).astype(int), caps - got)
        if add.sum() == 0:
            # every open slot's share is under one clip: draw which slots get one
            idx = np.flatnonzero(open_)
            q = ideal[idx]
            if rng is None:
                pick = idx[np.argsort(-q, kind="stable")[:remaining]]
            else:
                order = rng.permutation(len(idx))
                cum = np.cumsum(q[order])
                cum *= remaining / cum[-1]
                hits = np.searchsorted(cum, rng.random() + np.arange(remaining), side="right")
                pick = idx[order[np.minimum(hits, len(idx) - 1)]]
            fresh = idx[got[idx] == 0]
            p[fresh] = np.minimum(1.0, ideal[fresh])
            got[pick] += 1
            remaining = 0
        else:
            got += add
            remaining -= int(add.sum())
        open_ = got < caps
    p[(got > 0) & (p == 0)] = 1.0
    if rng is None:
        p = (got > 0).astype(float)
    return got, p


def _bins(score, edges, labels):
    b = pd.cut(score, bins=edges, labels=labels, include_lowest=True, right=False).astype(str)
    b[score >= edges[-1]] = labels[-1]
    return b


def _draw(df, score, n, edges, args, rng):
    """One stratified draw: n clips over the bins of score (edges), each bin shared across patients.

    Returns (picks, cells, bin_sizes, labels): picks keep df's index and carry bin, sampling_weight, cell_N, cell_n
    and cell_p; cells is one row per bin x patient with N (candidates), n (clips) and p (the patient's chance of
    being drawn in that bin, 1 when every patient got its share)."""
    labels = bin_labels(edges)
    df = df.assign(bin=_bins(score, edges, labels))
    bin_sizes = df.groupby("bin").size().reindex(labels, fill_value=0)
    per_bin, _p = share(n, bin_sizes.values, args.alloc, bin_sizes.values, rng)
    picks, cells, taken = [], [], {}
    for b, nb in zip(labels, per_bin):
        if nb == 0:
            cells.extend({"bin": b, "patient": p, "N": int(N), "n": 0, "p": 0.0}
                         for p, N in df[df["bin"] == b].groupby("patient").size().items())
            continue
        sub = df[df["bin"] == b]
        sizes = sub.groupby("patient").size()
        cap = sizes.values
        if args.within == "balanced" and args.max_patient_frac < 1:
            cap = np.minimum(cap, max(1, int(np.floor(args.max_patient_frac * nb))))
        per_pat, p_pat = share(nb, cap, "equal" if args.within == "balanced" else "proportional", sizes.values, rng)
        if per_pat.sum() < nb:
            print(f"note: bin {b} wanted {nb} clips but the patient cap allows {per_pat.sum()}; "
                  "raise --max-patient-frac to fill it")
        bin_picks = []
        for (pat, N), k, pp in zip(sizes.items(), per_pat, p_pat):
            cells.append({"bin": b, "patient": pat, "N": int(N), "n": int(k), "p": float(pp)})
            if k == 0:
                continue
            g = sub[sub["patient"] == pat]
            if args.min_gap_s > 0:
                idx = []
                for i in rng.permutation(len(g)):
                    path, c = g["source_path"].iat[i], g["center_s"].iat[i]
                    near = taken.setdefault(path, [])
                    if any(abs(c - x) < args.min_gap_s for x in near):
                        continue
                    near.append(c)
                    idx.append(i)
                    if len(idx) == k:
                        break
                k = len(idx)
                cells[-1]["n"] = k
                if k == 0:
                    continue
            else:
                idx = rng.choice(len(g), size=int(k), replace=False)
            chosen = g.iloc[np.sort(idx)].copy()
            chosen["sampling_weight"] = N / k / cells[-1]["p"]
            chosen["cell_N"] = int(N)
            chosen["cell_n"] = int(k)
            chosen["cell_p"] = cells[-1]["p"]
            bin_picks.append(chosen)
        if bin_picks:
            # patients drawn at random stand for the ones not drawn: scale so the bin's weights add up to
            # its candidate count (a ratio estimate; a no-op when every patient in the bin has clips)
            got = sum(c["sampling_weight"].sum() for c in bin_picks)
            for c in bin_picks:
                c["sampling_weight"] *= len(sub) / got
            picks.extend(bin_picks)
    out = pd.concat(picks) if picks else df.iloc[:0].assign(sampling_weight=[], cell_N=[], cell_n=[], cell_p=[])
    cells = pd.DataFrame(cells, columns=["bin", "patient", "N", "n", "p"])
    # checks that must hold before anyone labels a clip
    wsum = out.groupby("bin")["sampling_weight"].sum()
    for b, got in wsum.items():
        if abs(got - bin_sizes[b]) > 1e-6 * max(1, bin_sizes[b]):
            sys.exit(f"weights in bin {b} add to {got}, not {bin_sizes[b]}")
    return out, cells, bin_sizes, labels


def _inclusion(df, score, edges, labels, cells):
    """Each candidate's chance of being drawn by one draw, overall and within its patient (from the cells)."""
    tab = cells.set_index(["bin", "patient"])
    key = pd.MultiIndex.from_arrays([_bins(score, edges, labels).values, df["patient"].values])
    N = tab["N"].reindex(key).to_numpy(float)
    n = tab["n"].reindex(key).fillna(0).to_numpy(float)
    p = tab["p"].reindex(key).fillna(0).to_numpy(float)
    # a patient drawn at random in a bin (p < 1) gets one clip there, so each of its seconds had p / N
    k = np.where((p > 0) & (p < 1), np.maximum(n, 1), n)
    within = np.where(N > 0, k / N, 0.0)
    return np.minimum(1.0, within * np.where(p > 0, p, 0.0)), np.minimum(1.0, within)


def _clip_max(clips, seconds, cols, shown_s, candidate_s):
    """{col: per-clip max} over the seconds whose middle lies in the 10 s a rater sees (the clip-level rule:
    a detector finds a clip when any second shown reaches its threshold)."""
    out = {c: np.full(len(clips), np.nan) for c in cols}
    s = seconds.assign(_mid=seconds["start_s"].astype(float) + candidate_s / 2)
    groups = {p: g.sort_values("_mid") for p, g in s[s["source_path"].isin(set(clips["source_path"]))].groupby("source_path")}
    for i, (path, c) in enumerate(zip(clips["source_path"], clips["center_s"].astype(float))):
        g = groups.get(path)
        if g is None:
            continue
        mids = g["_mid"].to_numpy()
        lo, hi = np.searchsorted(mids, c - shown_s / 2, "left"), np.searchsorted(mids, c + shown_s / 2, "left")
        for col in cols:
            v = pd.to_numeric(g[col].iloc[lo:hi], errors="coerce").to_numpy(float)
            if len(v) and not np.isnan(v).all():
                out[col][i] = np.nanmax(v)
    return out


def _fmt(v, d=4):
    return "" if pd.isna(v) else f"{v:.{d}f}"


def _grid(df, args):
    """--grid-s: keeps one candidate second in every grid-s of each file, at an offset drawn per file, so every second
    has the same 1 in G chance of being a candidate and no two candidates in a file are closer than grid-s minus one
    second. Clips then never share the EEG a rater sees, with no --min-gap-s, and every chance stays exact."""
    if args.min_gap_s > 0:
        sys.exit("--grid-s already keeps the candidates in a file apart; leave --min-gap-s out")
    g = int(round(args.grid_s / args.candidate_s))
    if g < 2 or abs(g * args.candidate_s - args.grid_s) > 1e-9:
        sys.exit("--grid-s must be a whole number (2 or more) of --candidate-s")
    files = sorted(df["source_path"].astype(str).unique())
    offset = dict(zip(files, np.random.default_rng([args.seed, 977]).integers(0, g, len(files))))
    slot = np.floor(df["start_s"].to_numpy(float) / args.candidate_s + 1e-9).astype(np.int64) % g
    n_before = len(df)
    df = df[slot == df["source_path"].astype(str).map(offset).to_numpy()].copy()
    # two seconds of one file in the same slot less than a step apart (records that start off the whole second):
    # keep the first
    srt = df.sort_values(["source_path", "start_s"])
    close = (srt.groupby("source_path")["start_s"].diff() < args.grid_s - args.candidate_s).to_numpy()
    if close.any():
        df = df.drop(index=srt.index[close])
        print(f"note: --grid-s left out {int(close.sum())} seconds that sat less than "
              f"{args.grid_s - args.candidate_s:g} s after another kept second (records starting off the whole second)")
    print(f"--grid-s {args.grid_s:g}: kept {len(df)} of {n_before} candidates, one in every {g} seconds of each file at "
          f"an offset drawn with the seed; weights stand for the kept seconds (each second had a 1 in {g} chance, "
          f"so multiply totals by {g})")
    if df.empty:
        sys.exit("--grid-s left no candidates")
    return df


def cmd_sample(args):
    df = read_table(args.table)
    draws = []
    for spec in args.draw or []:
        col, sep, n = spec.partition("=")
        if not sep or not n.strip().isdigit() or int(n) < 1 or not col.strip().endswith("_prob"):
            sys.exit(f"--draw {spec!r}: give a score column from join and a clip count, e.g. spikenet2_prob=1000")
        draws.append((col.strip(), int(n)))
    if args.random < 0:
        sys.exit("--random cannot be negative")
    designs = bool(draws or args.random)
    if args.disjoint and len(draws) + bool(args.random) < 2:
        sys.exit("--disjoint keeps several draws apart; give two or more (--draw ... --draw ..., or --random)")
    if designs and args.prob_cols:
        sys.exit("--draw sets each draw's own score column; leave --prob-cols out")
    multi = [c for c, _n in draws] if designs else [c.strip() for c in (args.prob_cols or "").split(",") if c.strip()]
    cols = {"patient": args.patient_col, "source_path": args.path_col, "start_s": args.start_col}
    if not multi and not designs:
        cols["prob"] = args.prob_col
    missing = [c for c in list(cols.values()) + multi if c not in df.columns]
    if missing:
        sys.exit(f"{args.table} is missing column(s) {missing}; it has {list(df.columns)[:12]}. "
                 "Point the --*-col options at the right names.")
    df = df.rename(columns={v: k for k, v in cols.items()})[list(cols) + [c for c in df.columns if c not in cols.values()]]
    df["start_s"] = pd.to_numeric(df["start_s"], errors="coerce")
    for c in multi:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].lt(0).any() or df[c].gt(1).any():
            sys.exit(f"{c} has values outside 0 to 1; is that column a probability?")
    if multi:
        # a second's stratum is the highest score any of the scorings gives it, so each one's high seconds are
        # sampled densely; the weights keep every scoring's estimates unbiased either way
        df["prob"] = df[multi].max(axis=1)
    elif designs:
        df["prob"] = 0.0  # random draw only
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce")
    n_in = len(df)
    # a draw bins every candidate on its own score, so each draw's score must be there
    df = df.dropna(subset=["start_s", "prob"] + (multi if designs else []))
    n_nan = n_in - len(df)
    if df["prob"].lt(0).any() or df["prob"].gt(1).any():
        sys.exit("prob has values outside 0 to 1; is that column a probability?")
    dup = df.duplicated(["source_path", "start_s"]).sum()
    if dup:
        sys.exit(f"{dup} rows repeat the same source_path and start_s; each scored second should appear once")
    every = df  # every scored second, for the clip-level maxima

    # edges: the clip must fit inside one unbroken stretch of scored seconds in its file, so a clip never
    # crosses the end of a recording or a gap in an EDF+D file (where the detector scored nothing)
    half = args.clip_s / 2
    df = df.assign(center_s=df["start_s"] + args.candidate_s / 2)
    srt = df.sort_values(["source_path", "start_s"])
    step = srt.groupby("source_path")["start_s"].diff()
    typical = srt.groupby("source_path")["start_s"].transform(lambda x: x.diff().median()).fillna(args.candidate_s)
    brk = step.isna() | (step > np.maximum(1.5 * args.candidate_s, 1.5 * typical))
    df.loc[srt.index, "_stretch"] = brk.cumsum().values
    lo = df.groupby("_stretch")["start_s"].transform("min")
    hi = df.groupby("_stretch")["start_s"].transform("max") + args.candidate_s
    n_split = int((df.groupby("source_path")["_stretch"].nunique() > 1).sum())
    if n_split:
        print(f"note: {n_split} file(s) have gaps in their scored seconds; clips are kept inside each unbroken stretch")
    fits = (df["center_s"] - half >= lo) & (df["center_s"] + half <= hi)
    n_edge = int((~fits).sum())
    df = df[fits].copy()
    if df.empty:
        sys.exit("no candidates left after dropping seconds too close to a file edge")
    if args.grid_s > 0:
        df = _grid(df, args)
    if "subq_start_s" in df.columns:
        # the middle of the drawn second on the SubQ file's clock, for the admin view's SubQ lanes
        rate = pd.to_numeric(df["subq_rate"], errors="coerce").fillna(1.0) if "subq_rate" in df.columns else 1.0
        df["subq_center_s"] = pd.to_numeric(df["subq_start_s"], errors="coerce") + rate * args.candidate_s / 2

    edges = [float(x) for x in args.bins.split(",")]
    if edges != sorted(edges) or edges[0] > 0 or edges[-1] < 1:
        sys.exit("--bins must be increasing and cover 0 to 1, for example 0,0.5,0.9,1")
    shown = args.clip_s - 2 * args.pad_s
    score_cols = [c for c in df.columns if c.endswith("_prob") and c != "strata_prob"]
    if designs:
        return _sample_designs(args, df, every, draws, edges, score_cols, shown, n_in, n_nan, n_edge)

    if "detector" in df.columns and (df["detector"].astype(str) == "none").all() and len(edges) > 2:
        edges = [0.0, 1.0]  # from-edf: no score, so one bin (a simple random sample within each patient)
        print("no detector score in this table, so all seconds go in one bin")
    rng = np.random.default_rng(args.seed)
    out, cell_df, bin_sizes, labels = _draw(df, df["prob"], args.n, edges, args, rng)
    if out.empty:
        sys.exit("nothing was sampled")
    out = out.reset_index(drop=True)
    unsampled = cell_df[(cell_df["n"] == 0) & (cell_df["p"] == 0)]["N"].sum()
    if unsampled:
        print(f"WARNING: {unsampled} candidate seconds sit in bins with no sampled clip, so the weighted "
              "estimate cannot represent them (raise --n)")
    drawn = cell_df[(cell_df["p"] > 0) & (cell_df["p"] < 1)]
    if len(drawn):
        nbins = drawn["bin"].nunique()
        print(f"note: {nbins} bin(s) had fewer clips than patients, so clips went to {int((drawn['n'] > 0).sum())} of "
              f"{len(drawn)} patients there, drawn at random; their weights also stand for the patients not drawn")
    lost = cell_df[(cell_df["p"] == 1) & (cell_df["n"] == 0)]
    if len(lost):
        print(f"note: {len(lost)} patient(s) lost every clip in a bin to --min-gap-s; the others in that bin stand for them")
    total_w = out["sampling_weight"].sum()

    # the score column is named for the detector that produced it (cyclops_prob, ...); morgoth_prob by default
    dets = out["detector"].dropna().astype(str).unique() if "detector" in out.columns else []
    score_col = f"{dets[0].lower()}_prob" if len(dets) == 1 and dets[0] else "morgoth_prob"
    if multi:
        score_col = "strata_prob"  # the scorings' own columns follow it, unchanged
        for c in multi:
            out[c] = out[c].map(_fmt)
        out[score_col] = out["prob"].map(lambda p: f"{p:.4f}")
    elif len(dets) == 1 and dets[0] == "none":
        score_col = None  # drawn from select_clips.py from-edf: there is no score to keep
    else:
        out[score_col] = out["prob"].map(lambda p: f"{p:.4f}")
    # clip-level maxima: the highest score of each scoring over the seconds a rater sees
    maxed = multi if multi else ([] if score_col is None else ["prob"])
    for c, v in _clip_max(out, every, maxed, shown, args.candidate_s).items():
        out[(c[:-5] if multi else score_col[:-5]) + "_clipmax"] = [_fmt(x) for x in v]
    clipmax = [c for c in out.columns if c.endswith("_clipmax")]
    out["center_s"] = out["center_s"].map(lambda t: f"{t:.3f}")
    out["sampling_weight"] = out["sampling_weight"].map(lambda w: f"{w:.6f}")
    out["cell_p"] = out["cell_p"].map(lambda q: f"{q:.6f}")
    front = ["source_path", "center_s", "patient", "start_s"] + ([score_col] if score_col else []) + multi + clipmax + \
        ["bin", "sampling_weight", "cell_N", "cell_n", "cell_p"]
    rest = [c for c in out.columns if c not in front + ["prob", "_stretch"]]
    out = out[front + rest]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)

    # overlapping clips from the same file show the same EEG twice under two IDs
    ov = 0
    for _, g in out.assign(c=out["center_s"].astype(float)).groupby("source_path"):
        c = np.sort(g["c"].values)
        ov += int((np.diff(c) < args.clip_s).sum())

    print(f"read {n_in} scored seconds; dropped {n_nan} with no time or probability and {n_edge} too close "
          f"to a file edge; {len(df)} candidates from {df['patient'].nunique()} patients")
    print(f"wrote {args.out}: {len(out)} clips; weights add to {total_w:.0f} of {len(df)} candidates")
    if multi:
        print(f"strata: the highest of {', '.join(multi)} for each second; clips each scoring calls high:")
        for c in multi:
            hi_c = int((pd.to_numeric(out[c], errors="coerce") >= 0.5).sum())
            print(f"  {c:<22} {hi_c:>5} clips at 0.5 or more, of {int((df[c] >= 0.5).sum())} candidate seconds")
    _bin_table(out, bin_sizes, labels)
    if ov:
        print(f"\nnote: {ov} pairs of clips from the same file overlap (centers closer than {args.clip_s:g} s); "
              "raters will see some EEG twice")
    _pilot(out, labels, args)


def _bin_table(out, bin_sizes, labels, bin_col="bin"):
    print(f"\n  {'bin':>9} {'candidates':>11} {'clips':>6} {'patients':>8} {'top patient share':>18}")
    for b in labels:
        g = out[out[bin_col] == b]
        top = g["patient"].value_counts(normalize=True).max() if len(g) else float("nan")
        print(f"  {b:>9} {int(bin_sizes[b]):>11} {len(g):>6} {g['patient'].nunique():>8} {top:>18.2f}")


def _pilot(out, labels, args):
    if not args.pilot:
        return
    # spread the pilot across bins, then across patients inside each bin
    prng = np.random.default_rng(args.seed + 1)
    by_bin = {b: list(prng.permutation(out.index[out["bin"] == b])) for b in labels}
    chosen, used = [], {}
    while len(chosen) < min(args.pilot, len(out)):
        moved = False
        for b in labels:
            if len(chosen) >= args.pilot or not by_bin[b]:
                continue
            pool = by_bin[b]
            pool.sort(key=lambda i: used.get(out.at[i, "patient"], 0))
            i = pool.pop(0)
            chosen.append(i)
            used[out.at[i, "patient"]] = used.get(out.at[i, "patient"], 0) + 1
            moved = True
        if not moved:
            break
    pilot_path = os.path.splitext(args.out)[0] + "_pilot.csv"
    out.loc[sorted(chosen)].to_csv(pilot_path, index=False)
    print(f"\nwrote {pilot_path}: {len(chosen)} clips across {out.loc[chosen, 'bin'].nunique()} bins "
          f"and {out.loc[chosen, 'patient'].nunique()} patients (also in {os.path.basename(args.out)})")


def _near_picks(df, picks, gap):
    """True for each candidate whose center lies closer than gap to a picked clip's center in the same file."""
    near = np.zeros(len(df), dtype=bool)
    if not picks:
        return near
    centers = df["center_s"].to_numpy(float)
    rows_of = df.groupby("source_path", sort=False).indices
    for path, cs in picks.items():
        rows = rows_of.get(path)
        if rows is None:
            continue
        cs = np.sort(np.asarray(cs, dtype=float))
        c = centers[rows]
        j = np.searchsorted(cs, c)
        left = np.abs(c - cs[np.maximum(j - 1, 0)])
        right = np.abs(cs[np.minimum(j, len(cs) - 1)] - c)
        near[rows] = np.minimum(left, right) < gap
    return near


def _disjoint_chances(args, df, specs, pos, gap):
    """--disjoint: for the drawn clips (positions pos in df), each draw's chance of being open to the clip and picking
    it, overall and within the patient, averaged over --disjoint-reps re-runs of the whole draw with other seeds."""
    import contextlib
    import io
    reps = max(1, int(getattr(args, "disjoint_reps", 100)))
    pi = [np.zeros(len(pos)) for _ in specs]
    pi_w = [np.zeros(len(pos)) for _ in specs]
    # run -1 repeats the draw that was made (same seeds), so every drawn clip has a chance above 0
    for r in range(-1, reps):
        taken = {}
        for k, (name, score, n, ed, col) in enumerate(specs):
            open_ = ~_near_picks(df, taken, gap) if k else np.ones(len(df), dtype=bool)
            pool = df[open_]
            with contextlib.redirect_stdout(io.StringIO()):
                out, cells, _sizes, labels = _draw(pool, score.loc[pool.index], n, ed, args,
                                                   np.random.default_rng([args.seed, k] if r < 0 else
                                                                         [args.seed, k, r + 1]))
            p_all, p_pat = _inclusion(df.iloc[pos], score.iloc[pos], ed, labels, cells)
            pi[k] += open_[pos] * p_all
            pi_w[k] += open_[pos] * p_pat
            if k < len(specs) - 1:
                for path, c in zip(out["source_path"], out["center_s"].astype(float)):
                    taken.setdefault(path, []).append(c)
    print(f"--disjoint: each clip's chance found from the draw made and {reps} re-runs of it")
    return [p / (reps + 1) for p in pi], [p / (reps + 1) for p in pi_w]


def _sample_designs(args, df, every, draws, edges, score_cols, shown, n_in, n_nan, n_edge):
    """Several independent draws from the same candidates (--draw per score, --random), merged into one table.

    Each draw is stratified on its own score exactly as a single draw is, with its own seed. A clip two draws both
    pick is shown once and belongs to both. Every clip keeps each draw's own weight (w_<draw>, blank when that draw
    did not pick it), so each draw can be analysed alone, and gets sampling_weight = 1 / its chance of being in the
    merged table, 1 - (1 - p1)(1 - p2)..., where p is its chance under each draw (Horvitz-Thompson); with
    --min-gap-s those chances are approximate.

    With --disjoint the draws run in the order given and each one leaves out every candidate closer than the gap
    (--min-gap-s, or --clip-s when that is 0) to a clip an earlier draw picked, so no two clips share any EEG. A clip's
    chance of being in the merged table is then the sum over draws of the chance that the draw was still open to it
    (no earlier draw picked it or anything within the gap) times the draw's chance of picking it when open. Which
    candidates are still open depends on where the earlier draws happened to land, so that sum is found by running
    the whole draw again --disjoint-reps times with other seeds and averaging each clip's open-times-chance over the
    runs (the chance under each draw's own cells, as without --disjoint)."""
    names, specs, col_of = [], [], {}
    for col, n in draws:
        name = col[:-5] if col.endswith("_prob") else col
        if name in names or name == "random":
            sys.exit(f"--draw {col}: each draw needs its own score column")
        names.append(name)
        col_of[name] = col
        specs.append((name, df[col], n, edges, col))
    if args.random:
        names.append("random")
        specs.append(("random", pd.Series(0.0, index=df.index), args.random, [0.0, 1.0], None))
    picked, pi, pi_w, summary, designs = {}, [], [], [], []
    disjoint = getattr(args, "disjoint", False)
    # with --grid-s every two candidates are already apart, so --disjoint only has to leave out the seconds drawn
    gap = args.min_gap_s if args.min_gap_s > 0 else (args.candidate_s / 2 if args.grid_s > 0 else args.clip_s)
    taken, removed = {}, []
    for k, (name, score, n, ed, col) in enumerate(specs):
        rng = np.random.default_rng([args.seed, k])
        pool = df
        if disjoint and k:
            near = _near_picks(df, taken, gap)
            pool = df[~near]
            lab = bin_labels(ed)
            b_all = _bins(score, ed, lab)
            removed.append((name, gap, b_all[near].value_counts().reindex(lab, fill_value=0),
                            b_all.value_counts().reindex(lab, fill_value=0)))
        out, cells, sizes, labels = _draw(pool, score.loc[pool.index], n, ed, args, rng)
        if out.empty:
            sys.exit(f"draw {name}: nothing was sampled")
        if disjoint:
            for path, c in zip(out["source_path"], out["center_s"].astype(float)):
                taken.setdefault(path, []).append(c)
        else:
            p_all, p_pat = _inclusion(df, score, ed, labels, cells)
            pi.append(p_all)
            pi_w.append(p_pat)
        for i, row in out.iterrows():
            rec = picked.setdefault(i, {})
            rec[name] = (f"{name} {row['bin']}" if col else "random", row["sampling_weight"])
        summary.append((name, col, out, sizes, labels))
        designs.append(cells.assign(draw=name))
    idx = sorted(picked)
    pos = df.index.get_indexer(idx)
    if disjoint:
        if any(len(picked[i]) > 1 for i in idx):
            sys.exit("--disjoint: two draws picked the same second; this is a bug, please report it")
        pi, pi_w = _disjoint_chances(args, df, specs, pos, gap)
        p_merged = np.minimum(1.0, np.sum(pi, axis=0))
        p_merged_pat = np.minimum(1.0, np.sum(pi_w, axis=0))
    else:
        p_merged = 1 - np.prod([1 - p[pos] for p in pi], axis=0)
        p_merged_pat = 1 - np.prod([1 - p[pos] for p in pi_w], axis=0)
    if (p_merged <= 0).any():
        sys.exit("a drawn clip has no chance of selection under any draw; this is a bug, please report it")
    out = df.loc[idx].copy()
    out["draws"] = [";".join(n for n in names if n in picked[i]) for i in idx]
    out["bin"] = [picked[i][next(n for n in names if n in picked[i])][0] for i in idx]
    first = [next(n for n in names if n in picked[i]) for i in idx]
    out["strata_prob"] = [_fmt(df.at[i, col_of[n]]) if n != "random" else "" for i, n in zip(idx, first)]
    out["sampling_weight"] = [f"{1 / p:.6f}" for p in p_merged]
    out["patient_weight"] = [f"{1 / p:.6f}" for p in p_merged_pat]
    for name in names:
        out[f"w_{name}"] = [f"{picked[i][name][1]:.6f}" if name in picked[i] else "" for i in idx]
    if disjoint:
        # each draw's own chance of picking each clip, for re-weighting later (they add to the merged chance)
        for name, p in zip(names, pi):
            out[f"p_{name}"] = [f"{v:.8f}" for v in p]
    for c, v in _clip_max(out, every, score_cols, shown, args.candidate_s).items():
        out[c[:-5] + "_clipmax"] = [_fmt(x) for x in v]
    for c in score_cols:
        out[c] = out[c].map(_fmt)
    out = out.reset_index(drop=True)
    clipmax = [c[:-5] + "_clipmax" for c in score_cols]
    out["center_s"] = out["center_s"].map(lambda t: f"{t:.3f}")
    if "subq_center_s" in out.columns:
        out["subq_center_s"] = out["subq_center_s"].map(lambda t: _fmt(t, 3))
    front = ["source_path", "center_s", "patient", "start_s", "draws", "bin", "strata_prob"] + score_cols + clipmax + \
        ["sampling_weight", "patient_weight"] + [f"w_{n}" for n in names] + \
        ([f"p_{n}" for n in names] if disjoint else [])
    rest = [c for c in out.columns if c not in front + ["prob", "_stretch", "cell_N", "cell_n", "cell_p"]]
    out = out[front + rest]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)
    # the design itself, for weighting later: every draw's bin x patient cells with candidates (N), clips (n),
    # the chance the patient was drawn in that bin at all (p_patient; below 1 only when a bin has fewer clips
    # than patients) and each candidate second's chance of selection (p_second, as _inclusion computes it)
    design_path = os.path.splitext(args.out)[0] + "_design.csv"
    design = pd.concat(designs, ignore_index=True).rename(columns={"p": "p_patient"})
    pp, n_, N_ = design["p_patient"].astype(float), design["n"].astype(float), design["N"].astype(float)
    k = np.where((pp > 0) & (pp < 1), np.maximum(n_, 1), n_)
    design["p_second"] = np.minimum(1.0, np.where(N_ > 0, k / N_, 0.0) * pp.clip(lower=0)).round(8)
    design = design[["draw"] + [c for c in design.columns if c != "draw"]]
    design.to_csv(design_path, index=False)

    print(f"read {n_in} scored seconds; dropped {n_nan} with no time or score and {n_edge} too close "
          f"to a file edge; {len(df)} candidates from {df['patient'].nunique()} patients")
    gone = {r[0]: r for r in removed}
    for name, col, got, sizes, labels in summary:
        print(f"\ndraw {name}: {len(got)} clips" + (f", binned on {col}" if col else ", one bin (random within each patient)"))
        if name in gone:
            _n, g, cut, tot = gone[name]
            what = f"closer than {g:g} s to an earlier draw's clip" if g >= args.candidate_s else \
                "an earlier draw had picked"
            print(f"  --disjoint: left out {int(cut.sum())} of {int(tot.sum())} candidates {what}; "
                  "by bin (left out / candidates):")
            print("  " + "  ".join(f"{b} {int(cut[b])}/{int(tot[b])}" for b in cut.index))
        _bin_table(got, sizes, labels)
    multi_hit = sum(1 for i in idx if len(picked[i]) > 1)
    total = sum(len(s[2]) for s in summary)
    if disjoint:
        apart = f"on the {args.grid_s:g} s grid, so no two share the EEG a rater sees" if args.grid_s > 0 else \
            f"no two closer than {gap:g} s in one file"
        print(f"\nwrote {args.out}: {len(out)} clips from {len(names)} draws that left out each other's picks, {apart}")
    else:
        print(f"\nwrote {args.out}: {len(out)} clips from {total} drawn; {multi_hit} drawn by more than one draw "
              "(shown once, counted in each)")
    for combo, k in pd.Series(out["draws"]).value_counts().items():
        if ";" in combo:
            print(f"  {combo.replace(';', ' and '):<30} {k}")
    print(f"merged weights add to {(1 / p_merged).sum():.0f} of {len(df)} candidates "
          "(close, not exact: each draw's own w_ columns add up exactly)")
    ov = 0
    for _, g in out.assign(c=out["center_s"].astype(float)).groupby("source_path"):
        c = np.sort(g["c"].values)
        ov += int((np.diff(c) < args.clip_s).sum())
    if ov:
        print(f"note: {ov} pairs of clips from the same file overlap (centers closer than {args.clip_s:g} s); "
              "raters will see some EEG twice")
    print(f"wrote {design_path}: candidates, clips and selection chance for every bin and patient of each draw")
    if args.pilot:
        _pilot(out, sorted(out["bin"].unique()), args)


def cmd_fake_morgoth(args):
    """Made-up per-second probabilities for every second of each EDF in a folder."""
    import pyedflib
    rng = np.random.default_rng(args.seed)
    rows = []
    edfs = sorted(f for f in os.listdir(args.edf_dir) if f.lower().endswith((".edf", ".bdf")))
    if not edfs:
        sys.exit(f"no EDF files in {args.edf_dir}")
    for k, name in enumerate(edfs):
        path = os.path.join(os.path.abspath(args.edf_dir), name)
        with pyedflib.EdfReader(path) as f:
            dur = f.getFileDuration()
        for s in range(int(dur)):
            p = float(np.clip(rng.beta(0.6, 6.0), 0, 0.999))
            if rng.random() < 0.04:
                p = float(rng.uniform(0.5, 0.99))
            rows.append({"patient": f"FAKE-{k + 1:02d}", "source_path": path, "start_s": s, "prob": round(p, 4)})
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(rows)} fake scored seconds from {len(edfs)} files")


# ---------------------------------------------------------------- run outputs -> per-second table

TIME_START = ["start_s", "second_start_s", "window_start_s", "t_start_s", "start", "second", "sec", "t_start",
              "onset_s", "onset"]
TIME_CENTRE = ["center_s", "centre_s", "window_center_s", "window_centre_s", "t_center", "t_centre", "center",
               "centre", "mid_s", "t_mid"]
TIME_OTHER = ["time_s", "t_s", "time_in_clip_s", "time", "t", "seconds"]
SCORE = ["prob", "probability", "p_spike", "spike_prob", "prob_spike", "p_ied", "ied_prob", "morgoth_prob",
         "cyclops_prob", "max_prob", "max_score", "score", "p", "pred", "y_prob"]
CONDITION = ["condition", "montage", "cond", "channel_set", "variant"]
CLIP = ["clip", "clip_id", "clip_name", "edf_name", "file", "filename", "recording", "record", "source"]
PATH = ["source_path", "edf_path", "file_path", "filepath", "path", "edf_file"]
PATIENT = ["patient_anon_id", "patient", "patient_id", "subject", "subject_id"]
EDGE = ["edge", "edge_flag", "near_edge", "is_edge"]
TABLE_EXT = (".csv", ".csv.gz", ".tsv", ".parquet", ".pq")
EMU_RE = r"(EMU\d+)"


def _pick(cols, candidates, override=None):
    """The first candidate present (case-insensitive), or the override, which must exist."""
    lower = {c.lower(): c for c in cols}
    if override:
        if override.lower() not in lower:
            return None
        return lower[override.lower()]
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def _stem(path):
    name = os.path.basename(path)
    for ext in TABLE_EXT:
        if name.lower().endswith(ext):
            return name[:-len(ext)]
    return os.path.splitext(name)[0]


def _clip_name(path, root):
    """The table's clip: its file name, or the nearest folder name holding an EMU ID when the file name has none."""
    import re
    stem = _stem(path)
    if re.search(EMU_RE, stem):
        return stem
    d = os.path.dirname(os.path.abspath(path))
    root = os.path.abspath(root)
    while d.startswith(root) and d != root:
        if re.search(EMU_RE, os.path.basename(d)):
            return os.path.basename(d)
        d = os.path.dirname(d)
    return stem


def _tables(run_dir, pattern):
    import fnmatch
    if os.path.isfile(run_dir):
        return [run_dir]
    found = []
    for d, dirs, files in os.walk(run_dir):
        dirs.sort()
        for f in sorted(files):
            if f.lower().endswith(TABLE_EXT) and (not pattern or fnmatch.fnmatch(f, pattern)):
                found.append(os.path.join(d, f))
    return found


def _read_any(path):
    if path.lower().endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t" if path.lower().endswith(".tsv") else ",")


def _roles(cols, args):
    r = {"time": _pick(cols, TIME_START + TIME_CENTRE + TIME_OTHER, args.time_col),
         "score": _pick(cols, SCORE, args.score_col),
         "condition": _pick(cols, CONDITION, args.condition_col),
         "clip": _pick(cols, CLIP, args.clip_col),
         "path": _pick(cols, PATH, args.path_col),
         "patient": _pick(cols, PATIENT, args.patient_col),
         "edge": _pick(cols, EDGE)}
    if r["score"] is None and args.condition and r["condition"] is None:
        # wide tables: one score column per condition
        r["score"] = _pick(cols, [], args.condition)
        r["wide"] = r["score"] is not None
    return r


def _time_kind(name, values, forced):
    """'start' or 'centre', from --time-is, the column name, or the values' fractional parts."""
    if forced != "auto":
        return forced, f"--time-is {forced}"
    frac = np.mod(np.asarray(values, dtype=float), 1.0)
    frac = frac[np.isfinite(frac)]
    if len(frac) and np.mean(np.abs(frac - 0.5) < 0.01) > 0.95:
        return "centre", "times end in .5, so they are window centres"
    low = name.lower()
    if low in TIME_CENTRE:
        return "centre", f"column '{name}' names window centres"
    if low in TIME_START:
        return "start", f"column '{name}' names window starts"
    return "start", f"column '{name}' is read as window starts (use --time-is centre if it holds centres)"


def _admission_map(path):
    """EMU ID -> patient from the admissions table (data_derived/emu_admissions.csv)."""
    if "id_map" in os.path.basename(path).lower():
        sys.exit("refusing to read an id_map file; pass data_derived/emu_admissions.csv")
    adm = pd.read_csv(path, dtype=str, keep_default_na=False)
    pat = _pick(adm.columns, PATIENT)
    emu = None
    for c in adm.columns:
        v = adm[c].astype(str).str.strip()
        if len(v) and v.str.fullmatch(r"EMU\d+").mean() > 0.8 and c != pat:
            emu = c
            break
    if pat is None or emu is None:
        sys.exit(f"{path}: need a patient column ({', '.join(PATIENT)}) and a column of EMU IDs; "
                 f"it has {list(adm.columns)}")
    m = dict(zip(adm[emu].str.strip(), adm[pat].str.strip()))
    return m, emu, pat


def _edf_index(root):
    """EDF file stem -> [paths] under root (names only; no file is opened)."""
    idx = {}
    for d, dirs, files in os.walk(root):
        dirs.sort()
        for f in files:
            if f.lower().endswith((".edf", ".bdf")):
                idx.setdefault(os.path.splitext(f)[0], []).append(os.path.join(d, f))
    return idx


def _match_edf(clip, idx):
    """Exact stem, else the longest EDF stem that the clip name starts with plus a separator."""
    if clip in idx:
        return idx[clip]
    best = None
    for stem in idx:
        if clip.startswith(stem) and len(clip) > len(stem) and clip[len(stem)] in "_-." and \
                (best is None or len(stem) > len(best)):
            best = stem
    return idx[best] if best else []


def _edf_end_s(path, clock="samples"):
    """Where the recording ends on its own clock, from the header (and record onsets for EDF+D, or for every EDF+
    file on the onsets clock), leaving out records that run backward the way the reader does."""
    from edfplus import EDFPlus
    f = EDFPlus(path, clock=_edf_clock(clock))
    try:
        return f.end_s
    finally:
        f.close()


def _edf_clock(clock):
    """edfplus' name for a table's edf_clock: samples (0 is the first sample; EDF+D on its onsets) or onsets."""
    return "onsets" if clock == "onsets" else "auto"


def _load_runs(args):
    tables = _tables(args.run_dir, args.files)
    if not tables:
        sys.exit(f"no CSV, TSV or Parquet tables under {args.run_dir}")
    frames, skipped, schemas = [], {}, {}
    for t in tables:
        try:
            df = _read_any(t)
        except Exception as e:  # unreadable table: count it, never print its contents
            skipped[f"unreadable ({type(e).__name__})"] = skipped.get(f"unreadable ({type(e).__name__})", 0) + 1
            continue
        r = _roles(list(df.columns), args)
        if r["time"] is None or r["score"] is None:
            key = "no time or score column: " + ", ".join(map(str, list(df.columns)[:10]))
            skipped[key] = skipped.get(key, 0) + 1
            continue
        sig = tuple(sorted((k, v) for k, v in r.items() if v and k != "wide"))
        schemas[sig] = schemas.get(sig, 0) + 1
        out = pd.DataFrame({"t": pd.to_numeric(df[r["time"]], errors="coerce"),
                            "prob": pd.to_numeric(df[r["score"]], errors="coerce")})
        if r["clip"]:
            out["clip"] = df[r["clip"]].astype(str).map(_stem)
        elif r["path"]:  # one clip per file, named by the file
            out["clip"] = df[r["path"]].astype(str).map(_stem)
        else:
            out["clip"] = _clip_name(t, args.run_dir if os.path.isdir(args.run_dir) else os.path.dirname(t))
        out["condition"] = df[r["condition"]].astype(str) if r["condition"] else (args.condition if r.get("wide") else "")
        if r["path"]:
            out["source_path"] = df[r["path"]].astype(str)
        if r["patient"]:
            out["run_patient"] = df[r["patient"]].astype(str)
        if r["edge"]:
            out["edge"] = df[r["edge"]]
        out["_time_col"] = r["time"]
        frames.append(out)
    if not frames:
        print(f"read {len(tables)} table(s); none had a time and a score column:")
        for k, n in skipped.items():
            print(f"  {n} x {k}")
        sys.exit("point --time-col and --score-col at the right columns")
    return pd.concat(frames, ignore_index=True), len(tables), skipped, schemas


def cmd_inspect(args):
    """Column names, row counts and detected roles of every table under the run folder; no values."""
    tables = _tables(args.run_dir, args.files)
    if not tables:
        sys.exit(f"no CSV, TSV or Parquet tables under {args.run_dir}")
    shapes = {}
    for t in tables:
        try:
            df = _read_any(t)
        except Exception as e:
            shapes.setdefault(("(unreadable: " + type(e).__name__ + ")",), []).append((t, 0, None))
            continue
        shapes.setdefault(tuple(map(str, df.columns)), []).append((t, len(df), df))
    print(f"{len(tables)} table(s) under {args.run_dir}, {len(shapes)} distinct column layout(s)\n")
    for cols, items in shapes.items():
        rows = [n for _t, n, _d in items]
        print(f"layout seen in {len(items)} table(s), {min(rows)}-{max(rows)} rows each; "
              f"example {os.path.relpath(items[0][0], args.run_dir if os.path.isdir(args.run_dir) else os.path.dirname(items[0][0]))}")
        df = items[0][2]
        if df is None:
            print("  (could not be read)\n")
            continue
        print("  columns: " + ", ".join(f"{c} ({df[c].dtype})" for c in df.columns))
        r = _roles(list(df.columns), args)
        print("  detected: " + ", ".join(f"{k}={v}" for k, v in r.items() if v and k != "wide") or "  detected: nothing")
        if r["condition"]:
            vals = pd.concat([d[r["condition"]] for _t, _n, d in items if d is not None]).astype(str).value_counts()
            print("  conditions: " + ", ".join(f"{k} ({v} rows)" for k, v in vals.head(30).items()))
        if r["time"]:
            kind, why = _time_kind(r["time"], df[r["time"]].head(500), args.time_is)
            print(f"  time: {kind} ({why})")
        if r["time"] is None or r["score"] is None:
            print("  not usable as is: no time or score column; from-runs skips tables like this")
        print()


def cmd_from_runs(args):
    df, n_tables, skipped, schemas = _load_runs(args)
    n_read = len(df)
    print(f"read {n_tables} table(s); used {sum(schemas.values())}; {n_read} rows")
    for k, n in skipped.items():
        print(f"  skipped {n} table(s): {k}")
    for sig, n in schemas.items():
        print(f"  {n} table(s) read as " + ", ".join(f"{k}={v}" for k, v in sig))

    conds = df["condition"].replace("", np.nan).dropna().unique()
    if args.condition:
        df = df[df["condition"].str.lower() == args.condition.lower()]
        if df.empty:
            sys.exit(f"no rows for condition {args.condition}; conditions present: {sorted(conds)}")
    elif len(conds) > 1:
        counts = df["condition"].value_counts()
        sys.exit("these tables hold several conditions; pick one with --condition:\n" +
                 "\n".join(f"  {k} ({v} rows)" for k, v in counts.items()))
    condition = args.condition or (conds[0] if len(conds) else "")

    # time on the EDF's clock, as the start of the scored second
    kinds = {}
    for col, g in df.groupby("_time_col"):
        kinds[col] = _time_kind(col, g["t"].values, args.time_is)
    for col, (kind, why) in kinds.items():
        print(f"time column {col}: {why}")
    shift = df["_time_col"].map(lambda c: args.candidate_s / 2 if kinds[c][0] == "centre" else 0.0)
    df["start_s"] = df["t"] - shift
    bad_t = df["start_s"].isna() | df["prob"].isna()
    if args.score_range == "probability" and (df["prob"].lt(0).any() or df["prob"].gt(1).any()):
        sys.exit("scores fall outside 0 to 1; if they are logits or another score, pass --score-range any "
                 "and give select_clips.py sample bins that fit them")
    df = df[~bad_t]

    # clip -> EDF path
    if "source_path" in df.columns and args.clip_root:
        # a path column may hold paths relative to --clip-root (the same name can sit in two folders)
        rel = ~df["source_path"].map(os.path.isabs)
        df.loc[rel, "source_path"] = df.loc[rel, "source_path"].map(
            lambda p: os.path.abspath(os.path.join(args.clip_root, p)))
    if "source_path" not in df.columns:
        if not args.clip_root:
            sys.exit("the tables have no path column; pass --clip-root (the folder holding the EDFs) so clips can be found")
        idx = _edf_index(args.clip_root)
        clips = df["clip"].unique()
        where, none, many = {}, [], []
        for c in clips:
            hits = _match_edf(c, idx)
            if len(hits) == 1:
                where[c] = os.path.abspath(hits[0])
            elif hits:
                many.append(c)
            else:
                none.append(c)
        if none:
            print(f"{len(none)} clip(s) have no EDF under {args.clip_root} and are left out, e.g. {', '.join(sorted(none)[:5])}")
        if many:
            print(f"{len(many)} clip(s) match more than one EDF under {args.clip_root} and are left out, "
                  f"e.g. {', '.join(sorted(many)[:5])}; point --clip-root at a narrower folder")
        df = df[df["clip"].isin(where)].copy()
        df["source_path"] = df["clip"].map(where)
    if df.empty:
        sys.exit("no rows left after matching clips to EDF files")

    # patient
    df["admission"] = df["clip"].str.extract(EMU_RE, expand=False)
    miss = df["admission"].isna()
    if miss.any():
        df.loc[miss, "admission"] = df.loc[miss, "source_path"].map(os.path.basename).str.extract(EMU_RE, expand=False)
    if args.admissions:
        m, emu_col, pat_col = _admission_map(args.admissions)
        df["patient"] = df["admission"].map(m)
        lost = df[df["patient"].isna()]
        if len(lost):
            print(f"{lost['admission'].nunique(dropna=False)} admission(s) ({len(lost)} rows) are not in "
                  f"{os.path.basename(args.admissions)} and are left out (the extended/subset IDs, or clips with no EMU ID)")
        df = df[df["patient"].notna()].copy()
        print(f"patients from {os.path.basename(args.admissions)} ({emu_col} -> {pat_col})")
    elif "run_patient" in df.columns:
        df["patient"] = df["run_patient"]
        print("patients from the run tables' own patient column")
    else:
        df["patient"] = df["admission"]
        if df["patient"].isna().any():
            sys.exit("some clips carry no EMU ID; pass --admissions or add a patient column")
        print("WARNING: no --admissions table, so each EMU ID (one admission) counts as a patient. Pass "
              "--admissions data_derived/emu_admissions.csv so a patient with two admissions counts once.")

    # duplicates: the same second scored twice
    df = df.drop_duplicates(["source_path", "start_s", "prob"])
    clash = df.duplicated(["source_path", "start_s"], keep=False)
    if clash.any():
        sys.exit(f"{int(clash.sum())} rows give different scores for the same file and second; is more than one "
                 "condition or detector mixed in? Narrow with --condition or --files")

    # times must fall inside the recording
    if not args.no_check_edf:
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        ends = {}
        for p in df["source_path"].unique():
            try:
                ends[p] = _edf_end_s(p, args.edf_clock)
            except Exception as e:
                sys.exit(f"could not read the header of {p}: {e}")
        end = df["source_path"].map(ends)
        over = (df["start_s"] + args.candidate_s > end + 1e-3) | (df["start_s"] < -1e-3)
        if over.mean() > 0.05:
            sys.exit(f"{int(over.sum())} of {len(df)} scored seconds fall outside their recording. Are the times in "
                     "seconds from the start of the file? (--time-col, --time-is; --edf-clock onsets if they follow "
                     "the files' record onsets)")
        if over.any():
            print(f"{int(over.sum())} scored seconds fall outside their recording and are left out")
            df = df[~over]

    out = pd.DataFrame({"patient": df["patient"], "source_path": df["source_path"], "start_s": df["start_s"].round(4),
                        "prob": df["prob"], "clip": df["clip"], "admission": df["admission"],
                        "detector": args.detector, "condition": condition, "edf_clock": args.edf_clock})
    if "edge" in df.columns:
        out["edge"] = df["edge"]
    out = out.sort_values(["patient", "source_path", "start_s"])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(out)} scored seconds from {out['clip'].nunique()} clips, "
          f"{out['admission'].nunique()} admissions, {out['patient'].nunique()} patients "
          f"({args.detector}{', ' + condition if condition else ''}); scores {out['prob'].min():.3f} to {out['prob'].max():.3f}")
    per = out.groupby("clip").size()
    print(f"seconds per clip: median {per.median():.0f}, range {per.min()}-{per.max()}")


# ---------------------------------------------------------------- EEG files with no detector output

def _pick_files(args):
    """EDF files to use: those named in --clip-list (paths, or clip names found under --clip-root), else every
    file under --clip-root whose name matches --pattern."""
    import fnmatch
    if args.clip_list:
        if args.clip_list.lower().endswith(TABLE_EXT):
            t = _read_any(args.clip_list)
            col = _pick(t.columns, PATH + CLIP, args.list_col)
            if col is None:
                sys.exit(f"{args.clip_list}: no path or clip column; it has {list(t.columns)[:12]} (use --list-col)")
            names = t[col].astype(str).tolist()
        else:
            with open(args.clip_list, encoding="utf-8") as f:
                names = [ln.strip() for ln in f if ln.strip()]
        idx = _edf_index(args.clip_root) if args.clip_root else {}
        files, lost = [], []
        for n in dict.fromkeys(names):
            cands = [n] + ([os.path.join(args.clip_root, n)] if args.clip_root else [])
            path = next((c for c in cands if os.path.isfile(c)), None)  # a path, absolute or relative to --clip-root
            if path:
                files.append(os.path.abspath(path))
                continue
            hits = _match_edf(_stem(n), idx)
            if len(hits) == 1:
                files.append(os.path.abspath(hits[0]))
            else:
                lost.append(n)
        if lost:
            print(f"{len(lost)} of {len(names)} listed clips matched no single EDF file and are left out, "
                  f"e.g. {', '.join(os.path.basename(x) for x in lost[:5])}")
        return sorted(set(files))
    if not args.clip_root:
        sys.exit("pass --clip-root (a folder of EDF files) or --clip-list")
    files = []
    for d, dirs, names in os.walk(args.clip_root):
        dirs.sort()
        for n in sorted(names):
            if n.lower().endswith((".edf", ".bdf")) and fnmatch.fnmatch(n, args.pattern):
                files.append(os.path.abspath(os.path.join(d, n)))
    return files


def _segments(path, clock="samples"):
    """[(start_s, end_s)] of a file's contiguous stretches on its own clock."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    with open(path, "rb") as f:
        bdf = f.read(1) == b"\xff"
    if bdf:
        import pyedflib
        mode = getattr(pyedflib, "DO_NOT_READ_ANNOTATIONS", None)
        with (pyedflib.EdfReader(path) if mode is None else pyedflib.EdfReader(path, mode)) as r:
            return [(0.0, float(r.getFileDuration()))]
    from edfplus import EDFPlus
    e = EDFPlus(path, clock=_edf_clock(clock))
    try:
        return [(s[2], s[3]) for s in e.segments]
    finally:
        e.close()


def cmd_from_edf(args):
    """Every whole second of every file, with no score, for drawing clips before any detector has run."""
    import re
    files = _pick_files(args)
    if not files:
        sys.exit("no EDF files found")
    m = None
    if args.admissions:
        m, _emu, _pat = _admission_map(args.admissions)
    rows, n_unmapped, n_bad = [], 0, []
    for path in files:
        clip = os.path.splitext(os.path.basename(path))[0]
        emu = re.search(EMU_RE, clip)
        adm = emu.group(1) if emu else clip
        pat = m.get(adm) if m is not None else adm
        if pat is None:
            n_unmapped += 1
            continue
        try:
            segs = _segments(path, args.edf_clock)
        except Exception as e:
            n_bad.append((path, str(e)))
            continue
        for s0, s1 in segs:
            for t in np.arange(np.ceil(s0), np.floor(s1) - args.candidate_s + 1e-9, args.candidate_s):
                rows.append((pat, path, float(t), clip, adm))
    if n_bad:
        print(f"{len(n_bad)} file(s) could not be read and are left out, e.g. {n_bad[0][0]}: {n_bad[0][1]}")
    if n_unmapped:
        print(f"{n_unmapped} file(s) belong to admissions not in {os.path.basename(args.admissions)} and are left out")
    if not rows:
        sys.exit("no seconds to write")
    out = pd.DataFrame(rows, columns=["patient", "source_path", "start_s", "clip", "admission"])
    out.insert(3, "prob", 0.0)
    out["detector"] = "none"
    out["edf_clock"] = args.edf_clock
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(out)} seconds from {out['clip'].nunique()} files, {out['admission'].nunique()} "
          f"admissions, {out['patient'].nunique()} patients, with no detector score")
    if m is None:
        print("WARNING: no --admissions table, so each EMU ID (one admission) counts as a patient")
    print("select_clips.py sample puts these in one bin: a simple random sample within each patient")


def cmd_check_data(args):
    """Reads a 12 s window from a sample of the files the way the review server does; prints aggregates only."""
    from collections import Counter
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import make_review_batches as mrb
    files = _pick_files(args)
    if not files:
        sys.exit("no EDF files found")
    rng = np.random.default_rng(args.seed)
    pick = files if len(files) <= args.n else [files[i] for i in sorted(rng.choice(len(files), args.n, replace=False))]
    reader = mrb.EDFReader()
    kinds, rates, units, missing, flat, other_n, other_names = Counter(), Counter(), Counter(), Counter(), Counter(), [], Counter()
    noisy = Counter()
    amps, fails, ekg, n_ok, gaps = [], [], 0, 0, 0
    for path in pick:
        try:
            kinds[mrb._edf_reserved_field(path) or "EDF"] += 1
            segs = _segments(path, args.edf_clock)
            gaps += len(segs) > 1
            s0, s1 = max(segs, key=lambda x: x[1] - x[0])
            if s1 - s0 < 12:
                raise ValueError("no stretch of 12 s or more")
            c = (s0 + s1) / 2
            sig = reader.read(path, c - 6, c + 6, clock=args.edf_clock)
            rows, names, bad = mrb.assemble(sig, 200.0, 2400, ["EKG"])
            loud = [n for n in mrb.noisy_channels(rows, names, 200.0) if n not in bad]
        except Exception as e:
            fails.append((os.path.basename(path), str(e)))
            continue
        finally:
            reader.close()
        n_ok += 1
        found = {}
        for lab, fs_i, _x, unit in sig:
            found.setdefault(mrb.normalize_name(lab), (fs_i, unit, lab))
        for name in mrb.SCALP:
            if name not in found:
                missing[name] += 1
            elif name in bad:
                flat[name] += 1
        noisy.update(loud)
        ekg += "EKG" in found and "EKG" not in bad
        scalp = [found[n] for n in mrb.SCALP if n in found]
        for fs_i, unit, _lab in scalp:
            rates[f"{fs_i:g} Hz"] += 1
            units[unit or "(blank)"] += 1
        others = [lab for name, (_f, _u, lab) in found.items() if name not in mrb.SCALP and name != "EKG"
                  and lab != "EDF Annotations"]
        other_n.append(len(others))
        other_names.update(others)
        have = [i for i, n in enumerate(mrb.SCALP) if n in found]
        if have:  # the same amplitude the server prints when it plans a study
            amps.append(float(np.median(mrb.robust_amp(rows[have]))))
    print(f"checked {len(pick)} of {len(files)} file(s), reading one 12 s window from the middle of each the way "
          f"the review server does")
    print("  file types: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common()) +
          (f"; {gaps} with gaps (read one segment at a time)" if gaps else ""))
    if n_ok:
        print("  sampling rates (per scalp channel): " + ", ".join(f"{k} x{v}" for k, v in rates.most_common()) +
              "; every clip is resampled to 200 Hz for the viewer")
        print("  units: " + ", ".join(f"{k} x{v}" for k, v in units.most_common()))
        if missing:
            print("  10-20 scalp channels missing: " + ", ".join(f"{k} in {v}" for k, v in missing.most_common()))
        else:
            print(f"  10-20 scalp channels: all 19 found in every file read ({n_ok})")
        if flat:
            print("  flat, stuck or empty in the window read: " + ", ".join(f"{k} in {v}" for k, v in flat.most_common()))
        if noisy:
            print("  noisy in the window read (EEG-band amplitude over 8x the clip's median electrode; drawn dashed): "
                  + ", ".join(f"{k} in {v}" for k, v in noisy.most_common()))
        print(f"  EKG found in {ekg} of {n_ok}; other channels per file: median {int(np.median(other_n))}"
              + (" (" + ", ".join(sorted(other_names)[:30]) + ")" if args.show_labels and other_names else ""))
        med = float(np.median(amps)) if amps else float("nan")
        print(f"  median robust scalp amplitude {med:.1f} µV")
    if fails:
        print(f"  {len(fails)} file(s) could not be read:")
        for name, why in fails[:10]:
            print(f"    {name}: {why}")
    problems = []
    if fails:
        problems.append(f"{len(fails)} file(s) unreadable")
    if n_ok and missing and missing.most_common(1)[0][1] > n_ok / 2:
        problems.append("a scalp channel is missing in most files, so channel names may not be mapping to 10-20 "
                        "(rerun with --show-labels)")
    if n_ok and flat and sum(flat.values()) > 0.5 * 19 * n_ok:
        problems.append("most scalp channels are flat (near-zero amplitude: volts read as µV?)")
    if amps and not (0.5 <= float(np.median(amps)) <= 2000):
        problems.append("implausible amplitude for scalp EEG in µV (volts read as µV, or the reverse)")
    if not n_ok:
        problems.append("nothing could be read")
    print("PROBLEM: " + "; ".join(problems) if problems else "OK: the review server can read these files")
    if problems:
        sys.exit(1)


def _nearest(left, right, cols, tol, what):
    """For each row of left, the right row of the same source_path whose start_s is nearest, within tol seconds
    (tables written by different runs can differ in the last digits of start_s), as right's cols beside left's own
    columns, in left's order; NaN where none is that close."""
    lk = left[["source_path", "start_s"]].copy()
    lk["source_path"] = lk["source_path"].astype(str)
    lk["start_s"] = lk["start_s"].astype(float)
    lk["_row"] = np.arange(len(lk))
    lk = lk.dropna(subset=["start_s"]).sort_values("start_s", kind="stable")
    rk = right[["source_path", "start_s"] + cols].copy()
    rk["source_path"] = rk["source_path"].astype(str)
    rk["start_s"] = rk["start_s"].astype(float)
    rk = rk.dropna(subset=["start_s"]).sort_values("start_s", kind="stable")
    rk["_hit"] = np.arange(len(rk))
    m = pd.merge_asof(lk, rk, on="start_s", by="source_path", direction="nearest", tolerance=tol)
    hit = m["_hit"].dropna()
    if hit.duplicated().any():
        sys.exit(f"{what}: one row matches two seconds within {tol:g} s; use a smaller --match-tol-s")
    rows = m["_row"].to_numpy()
    res = {}
    for c in cols:
        arr = np.full(len(left), np.nan, dtype=object)
        arr[rows] = m[c].to_numpy(dtype=object)
        res[c] = arr
    return pd.DataFrame(res, index=left.index)


def _path_col(t, what):
    for c in ("source_path", "path"):
        if c in t.columns:
            return t.rename(columns={c: "source_path"}) if c != "source_path" else t
    sys.exit(f"{what} needs a source_path (or path) column; it has {list(t.columns)[:12]}")


def _drop_flagged(joined, args):
    """Leaves out the seconds a flag table marks (Svanik's artifact flag: any scored channel past ±1,000 µV)."""
    f = _path_col(read_table(args.drop_flagged), args.drop_flagged)
    col = args.flag_col or next((c for c in ("artifact", "flag", "flagged") if c in f.columns), None)
    if col is None or col not in f.columns or "start_s" not in f.columns:
        sys.exit(f"{args.drop_flagged} needs start_s and a flag column (artifact, flag or flagged; or --flag-col)")
    f = f[["source_path", "start_s", col]].copy()
    f["start_s"] = pd.to_numeric(f["start_s"], errors="coerce").round(6)
    v = f[col]
    f["_flag"] = v.astype(str).str.lower().isin(["1", "1.0", "true", "yes"]) if v.dtype == object else \
        pd.to_numeric(v, errors="coerce").fillna(0).astype(float) > 0
    if f.duplicated(["source_path", "start_s"]).any():
        sys.exit(f"{args.drop_flagged} repeats a source_path and start_s")
    m = joined.copy()
    m["_flag"] = _nearest(joined, f, ["_flag"], args.match_tol_s, args.drop_flagged)["_flag"]
    flagged, unknown = int(m["_flag"].eq(True).sum()), int(m["_flag"].isna().sum())
    drop_unknown = getattr(args, "drop_unflagged", False)
    keep = m["_flag"].isna() if not drop_unknown else pd.Series(False, index=m.index)
    kept = m[m["_flag"].eq(False) | keep].drop(columns="_flag")
    note = f"--drop-flagged: left out {flagged} flagged seconds ({col})"
    if unknown:
        note += (f"; {unknown} seconds are not in the flag table and are left out too (--drop-unflagged)" if drop_unknown
                 else f"; {unknown} seconds are not in the flag table and stay in")
    return kept, note


def _apply_pool(joined, args, clocks):
    """Keeps only seconds wholly inside a pool interval and carries its SubQ map: the SubQ file and the time of
    each second there, subq_start_s = subq_t0_s + (start_s - interval start) x subq_rate."""
    pool = _path_col(read_table(args.pool), args.pool)
    for c in ("start_s", "end_s"):
        if c not in pool.columns:
            sys.exit(f"{args.pool} needs start_s and end_s (each interval on the scalp file's clock)")
    if "edf_clock" in pool.columns:
        found = set(pool["edf_clock"].dropna().astype(str))
        if found and found != set(clocks.values()):
            sys.exit(f"{args.pool} is on the {', '.join(sorted(found))} clock and the tables on "
                     f"{', '.join(sorted(set(clocks.values())))}; they must count time the same way")
    pool = pool.copy()
    pool["iv_start"] = pd.to_numeric(pool["start_s"], errors="coerce")
    pool["iv_end"] = pd.to_numeric(pool["end_s"], errors="coerce")
    if pool[["iv_start", "iv_end"]].isna().any().any() or (pool["iv_end"] <= pool["iv_start"]).any():
        sys.exit(f"{args.pool} has an interval with no start or end, or one that ends before it starts")
    srt = pool.sort_values(["source_path", "iv_start"])
    prev_end = srt.groupby("source_path")["iv_end"].shift()
    if (srt["iv_start"] < prev_end - 1e-6).any():
        sys.exit(f"{args.pool} has overlapping intervals in one file; each scalp second may map to one SubQ time only")
    carry = [c for c in ("subq_path", "subq_t0_s", "subq_rate", "implant_side", "patient") if c in pool.columns]
    right = srt[["source_path", "iv_start", "iv_end"] + carry].rename(columns={"patient": "_pool_patient"})
    left = joined.assign(_t=joined["start_s"].astype(float)).sort_values("_t")
    m = pd.merge_asof(left, right.sort_values("iv_start"), left_on="_t", right_on="iv_start", by="source_path",
                      direction="backward")
    inside = m["iv_start"].notna() & (m["_t"] + args.candidate_s <= m["iv_end"] + 1e-6)
    kept = m[inside].copy()
    if "_pool_patient" in kept.columns:
        bad = (kept["_pool_patient"].astype(str) != kept["patient"].astype(str)).sum()
        if bad:
            sys.exit(f"{bad} seconds have a different patient in {args.pool} than in the scored tables")
        kept = kept.drop(columns="_pool_patient")
    if "subq_t0_s" in kept.columns:
        rate = pd.to_numeric(kept["subq_rate"], errors="coerce").fillna(1.0) if "subq_rate" in kept.columns else 1.0
        kept["subq_start_s"] = (pd.to_numeric(kept["subq_t0_s"], errors="coerce") +
                                (kept["_t"] - kept["iv_start"]) * rate).round(6)
        if "subq_rate" not in kept.columns:
            kept["subq_rate"] = 1.0
        kept = kept.drop(columns="subq_t0_s")
    kept = kept.drop(columns=["_t", "iv_start", "iv_end"])
    hours = len(kept) * args.candidate_s / 3600
    pats = kept["patient"].nunique() if "patient" in kept.columns else 0
    return kept, (f"--pool: kept {len(kept)} seconds ({hours:.1f} h, {pats} patients) inside "
                  f"{len(pool)} intervals; left out {len(joined) - len(kept)} outside them")


NAMES = {"morgoth": "MORGOTH", "cyclops": "Cyclops", "cyclops_sd": "Cyclops SD", "spikenet1": "SpikeNet1",
         "spikenet2": "SpikeNet2", "linelength": "Line length", "persyst": "Persyst"}


def _scores_near(spec, what, windows):
    """(name, {file: rows}) from name=path, keeping only rows whose start_s falls in one of windows[file] (a sorted
    list of (lo, hi)); the table is read in pieces, so a SubQ table of tens of millions of seconds fits in memory.
    The score column may be prob, score or <name>_prob; the file column source_path or path."""
    name, sep, path = spec.partition("=")
    name = name.strip().lower()
    if not sep or not name or not path or not all(ch.isalnum() or ch == "_" for ch in name):
        sys.exit(f"{what} {spec!r}: give name=path, e.g. spikenet2=out/phase5/spikenet2_seconds.csv")
    wanted = ("source_path", "path", "start_s", "prob", "score", f"{name}_prob")
    if path.lower().endswith((".parquet", ".pq")):
        try:
            import pyarrow.parquet as pq
        except ImportError:  # another parquet engine: read it whole
            t = pd.read_parquet(path)
            cols, pieces = list(t.columns), [t]
        else:
            pf = pq.ParquetFile(path)
            cols = pf.schema_arrow.names
            pieces = (b.to_pandas() for b in pf.iter_batches(batch_size=2_000_000, columns=[c for c in cols if c in wanted]))
    else:
        cols = list(pd.read_csv(path, nrows=0).columns)
        pieces = pd.read_csv(path, chunksize=2_000_000)
    fcol = next((c for c in ("source_path", "path") if c in cols), None)
    scol = next((c for c in ("prob", f"{name}_prob", "score") if c in cols), None)
    if fcol is None or scol is None or "start_s" not in cols:
        sys.exit(f"{path} needs source_path (or path), start_s and a score column (prob, score or {name}_prob)")
    bounds = {f: (np.array([w[0] for w in ws]), np.array([w[1] for w in ws])) for f, ws in windows.items()}
    kept = []
    for t in pieces:
        t = t[t[fcol].isin(bounds.keys())]
        if t.empty:
            continue
        st = pd.to_numeric(t["start_s"], errors="coerce").to_numpy(float)
        ok = np.zeros(len(t), dtype=bool)
        for f, ix in t.groupby(fcol).indices.items():
            lo, hi = bounds[f]
            k = np.searchsorted(lo, st[ix], "right") - 1
            ok[ix] = (k >= 0) & (st[ix] < hi[np.maximum(k, 0)])
        if ok.any():
            kept.append(pd.DataFrame({"file": t[fcol].to_numpy()[ok], "start_s": st[ok],
                                      "prob": pd.to_numeric(t[scol], errors="coerce").to_numpy(float)[ok]}))
    allrows = pd.concat(kept) if kept else pd.DataFrame(columns=["file", "start_s", "prob"])
    return name, {f: g.sort_values("start_s") for f, g in allrows.groupby("file")}


def _windows(files, anchors, reach_lo, reach_hi):
    """{file: sorted, merged (lo, hi) windows} around each clip's anchor time."""
    by = {}
    for f, a, lo, hi in zip(files, anchors, reach_lo, reach_hi):
        if f:
            by.setdefault(f, []).append((a - lo, a + hi))
    out = {}
    for f, ws in by.items():
        ws.sort()
        merged = [list(ws[0])]
        for lo, hi in ws[1:]:
            if lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        out[f] = [tuple(w) for w in merged]
    return out


VIRTUAL_DEFAULT = ["L=F7-T3", "R=F8-T4"]  # lab meeting 2026-10-29; the montage thread may change it


def _virtual_pairs(specs):
    """{"L": ["F7-T3", ...], "R": [...]} from --virtual L=F7-T3,T3-T5 R=F8-T4 (classic 10-20 names)."""
    import make_review_batches as mrb
    out = {}
    for spec in specs or VIRTUAL_DEFAULT:
        side, sep, rest = spec.partition("=")
        side = side.strip().upper()[:1]
        if not sep or side not in ("L", "R"):
            sys.exit(f"--virtual {spec!r}: give a side and bipolar pairs, e.g. L=F7-T3,T3-T5")
        for pair in rest.split(","):
            a, dash, b = pair.strip().partition("-")
            if not dash or a not in mrb.SCALP or b not in mrb.SCALP or a == b:
                sys.exit(f"--virtual {spec!r}: {pair.strip()!r} is not a pair of 10-20 electrodes (names like F7, T3)")
            out.setdefault(side, []).append(f"{a}-{b}")
    return out


def cmd_admin(args):
    """The owner-only extras for each clip: SubQ map, implant side, which draw picked it, and every detector's
    score for every second of the clip (the superadmin strips), scalp and SubQ."""
    import json
    clips = pd.read_csv(args.clips, dtype=str, keep_default_na=False)
    for c in ("source_path", "center_s"):
        if c not in clips.columns:
            sys.exit(f"{args.clips} needs a {c} column (the clip table sample wrote)")
    centers = clips["center_s"].astype(float).to_numpy()
    half, shown, cand = args.clip_s / 2, args.clip_s - 2 * args.pad_s, args.candidate_s
    strips = [[] for _ in range(len(clips))]
    maxima = {}
    has_subq = "subq_path" in clips.columns and "subq_center_s" in clips.columns
    if args.subq and not has_subq:
        sys.exit(f"{args.clips} has no subq_path and subq_center_s; draw from a table joined with --pool to map SubQ")
    rates = clips["subq_rate"].replace("", "1").astype(float).to_numpy() if "subq_rate" in clips.columns else \
        np.ones(len(clips))
    reach = {"scalp": _windows(clips["source_path"], centers, np.full(len(clips), half + cand), np.full(len(clips), half))}
    if has_subq:
        sc = pd.to_numeric(clips["subq_center_s"], errors="coerce").to_numpy(float)
        ok = (clips["subq_path"] != "").to_numpy() & ~np.isnan(sc)
        reach["subq"] = _windows(clips["subq_path"].where(ok, ""), np.nan_to_num(sc), (half + cand) * rates, half * rates)
    for kind, specs in (("scalp", args.scalp or []), ("subq", args.subq or [])):
        for spec in specs:
            name, by_file = _scores_near(spec, f"--{kind}", reach[kind])
            shown_name = NAMES.get(name, name[:1].upper() + name[1:])
            col = (f"subq_{name}" if kind == "subq" else name) + "_clipmax"
            maxima[col] = np.full(len(clips), np.nan)
            n_hit = 0
            for i in range(len(clips)):
                c = centers[i]
                if kind == "scalp":
                    g, anchor, rate = by_file.get(clips.at[i, "source_path"]), c, 1.0
                else:
                    path, sc = clips.at[i, "subq_path"], clips.at[i, "subq_center_s"]
                    if not path or not sc:
                        continue
                    rate = rates[i]
                    g, anchor = by_file.get(path), float(sc)
                if g is None:
                    continue
                starts, probs = g["start_s"].to_numpy(), g["prob"].to_numpy()
                lo = np.searchsorted(starts, anchor - (half + cand) * rate, "left")
                hi = np.searchsorted(starts, anchor + half * rate, "left")
                secs, best = [], np.nan
                for t, pr in zip(starts[lo:hi], probs[lo:hi]):
                    rel = (t - anchor) / rate + c - (c - half)  # seconds from the clip's start
                    if rel + cand <= 0 or rel >= args.clip_s:
                        continue
                    secs.append([round(float(rel), 3), None if np.isnan(pr) else round(float(pr), 3)])
                    mid = rel + cand / 2
                    if half - shown / 2 <= mid < half + shown / 2 and not np.isnan(pr):
                        best = pr if np.isnan(best) else max(best, pr)
                if secs:
                    n_hit += 1
                    strips[i].append({"name": shown_name, "kind": kind, "len": cand, "secs": secs})
                maxima[col][i] = best
            print(f"  {kind} {name:<12} scores for {n_hit} of {len(clips)} clips")
    keep = [c for c in ("source_path", "center_s", "patient", "draws", "implant_side", "subq_path", "subq_center_s",
                        "subq_rate", "edf_clock") if c in clips.columns]
    out = clips[keep].copy()
    out["strips"] = [json.dumps(s, separators=(",", ":")) if s else "" for s in strips]
    # virtual UNEEG: scalp bipolar pairs on the implant side, drawn beside the real SubQ channels
    pairs = _virtual_pairs(args.virtual)
    sides = clips["implant_side"].str.strip().str.upper().str[:1] if "implant_side" in clips.columns else \
        pd.Series("", index=clips.index)
    out["virtual"] = [";".join(pairs[s] if s in pairs else pairs.get("L", []) + pairs.get("R", [])) for s in sides]
    for col, v in maxima.items():
        out[col] = [_fmt(x) for x in v]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)
    n_subq = int((out["subq_path"] != "").sum()) if "subq_path" in out.columns else 0
    print(f"wrote {args.out}: {len(out)} clips, {n_subq} with a SubQ file to show; give it to serve_review.py or "
          "make_site.py as --admin (only your own link sees it)")


PART_COLS = ("part", "ref_answer", "ref_type", "ref_note")


def _clip_key(df):
    return list(zip(df["source_path"], pd.to_numeric(df["center_s"], errors="coerce").round(3)))


def _score_bin(v):
    """The 0.1-wide score bin of each value (0 to 9; 1.0 is in the top bin), -1 where there is no score."""
    x = pd.to_numeric(v, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(x), -1, dtype=int)
    ok = np.isfinite(x) & (x >= 0) & (x <= 1)
    out[ok] = np.minimum((x[ok] * 10).astype(int), 9)
    return out


def pick_by_bins(bins, n, rng):
    """Picks n rows so that each score column has as close to n/10 rows in each of its ten bins as the rows allow.
    bins: (rows x columns) bin numbers 0-9. Each step takes a random row from the bin combination (cell) whose bins
    are furthest below their share, summed over the columns; among those, the cell with the fewest picks so far, so
    the picks also spread over the cells (250 over a 10 x 10 grid: 2 or 3 a cell); then at random. Returns row
    positions, sorted."""
    bins = np.asarray(bins, dtype=int)
    n = min(n, len(bins))
    k = bins.shape[1]
    target = n / 10
    counts = np.zeros((k, 10))
    cells = {}
    for i, key in enumerate(map(tuple, bins)):
        cells.setdefault(key, []).append(i)
    for key in cells:
        rng.shuffle(cells[key])
    keys = list(cells)
    picked, taken = [], {c: 0 for c in keys}
    for _ in range(n):
        live = [c for c in keys if cells[c]]
        need = np.array([sum(target - counts[j, c[j]] for j in range(k)) for c in live])
        best = [i for i in np.flatnonzero(need >= need.max() - 1e-9)]
        fewest = min(taken[live[i]] for i in best)
        best = [i for i in best if taken[live[i]] == fewest]
        c = live[best[rng.integers(len(best))]]
        picked.append(cells[c].pop())
        taken[c] += 1
        for j in range(k):
            counts[j, c[j]] += 1
    return sorted(picked)


def cmd_parts(args):
    """The clip table of a staged study (kit 1.8), for serve_review.py and make_site.py --clips: every scalp clip,
    then the training examples, then UNEEG-only clips at seconds from the scalp part that have a SubQ file (the
    order raters meet them since kit 1.8.4).
    Examples: the rows of --examples-table (kit 1.8.1; chosen by hand, each with the expert's answer and a note),
    or else --examples-n seconds a reference rater called Yes; either way none is in the scalp part or within
    --gap-s of one. UNEEG clips: --uneeg-pick bins (kit 1.8.1, the default) spreads them evenly over the 0.1-wide
    bins of each --uneeg-bins-by score; ref takes --uneeg-yes-frac reference Yes seconds and the rest No.
    Prints counts only."""
    # empty cells stay empty text (kit 1.8.3): astype(str) alone turned them into "nan", which passed as a SubQ time
    scalp = read_table(args.scalp).fillna("").astype(str)
    for c in ("source_path", "center_s"):
        if c not in scalp.columns:
            sys.exit(f"{args.scalp} needs a {c} column (the clip table sample wrote)")
    scalp = scalp.drop(columns=[c for c in PART_COLS if c in scalp.columns])
    cols = list(scalp.columns)
    need_ref = (args.examples_n and not args.examples_table) or args.uneeg_pick == "ref"
    answer, n_dup, ref = {}, 0, None
    if need_ref and not args.ref:
        sys.exit("--ref is needed to draw examples from the reference's Yes seconds or to pick UNEEG clips by "
                 "the reference's answers; give --examples-table and --uneeg-pick bins to do without it")
    if args.ref:
        ref = pd.read_csv(args.ref, dtype=str, keep_default_na=False)
        for c in ("source_path", "center_s", "decision"):
            if c not in ref.columns:
                sys.exit(f"{args.ref} needs source_path, center_s and decision columns (merge_labels.py writes them)")
        if args.ref_rater:
            if "rater" not in ref.columns:
                sys.exit(f"{args.ref} has no rater column for --ref-rater")
            ref = ref[ref["rater"].str.upper() == args.ref_rater.strip().upper()]
        ref = ref[ref["decision"].isin(["yes", "no"])].copy()
        if not len(ref) and need_ref:
            sys.exit("the reference has no Yes or No answers" + (f" from {args.ref_rater}" if args.ref_rater else ""))
        ref["_k"] = _clip_key(ref)
        n_dup = int(ref["_k"].duplicated().sum())
        ref = ref.drop_duplicates("_k", keep="last")
        answer = dict(zip(ref["_k"], ref["decision"]))
    rng = np.random.default_rng(args.seed)
    scalp_keys = set(_clip_key(scalp))
    centres = {}
    for path, c in scalp_keys:
        centres.setdefault(path, []).append(c)

    def far(k):
        return all(abs(k[1] - c) >= args.gap_s for c in centres.get(k[0], []))

    # training examples: chosen by hand (--examples-table), or reference Yes seconds; away from every scalp clip
    if args.examples_table:
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        import make_review_batches as mrb
        ex = pd.read_csv(args.examples_table, dtype=str, keep_default_na=False)
        for c in ("source_path", "center_s"):
            if c not in ex.columns:
                sys.exit(f"{args.examples_table} needs source_path and center_s columns")
        if "ref_answer" not in ex.columns:
            ex["ref_answer"] = "yes"
        ex["ref_answer"] = ex["ref_answer"].str.strip().str.lower().replace("", "yes")
        if "ref_note" not in ex.columns:
            ex["ref_note"] = ""
        if "ref_type" not in ex.columns:
            ex["ref_type"] = ""
        # the kind of spike (kit 1.8.5): generalized, focal left or focal right, shown with the example
        kinds = [mrb.ref_type(v) for v in ex["ref_type"]]
        bad = [i + 2 for i, k in enumerate(kinds) if k is None]
        if bad:
            sys.exit(f"{args.examples_table}: ref_type must be one of {', '.join(mrb.REF_TYPES)} or empty "
                     f"(rows {', '.join(map(str, bad[:10]))})")
        ex["ref_type"] = kinds
        bad = [i + 2 for i, a in enumerate(ex["ref_answer"]) if a not in ("yes", "no")]
        if bad:
            sys.exit(f"{args.examples_table}: ref_answer must be yes or no (rows {', '.join(map(str, bad[:10]))})")
        keys = _clip_key(ex)
        near = [i + 2 for i, k in enumerate(keys) if k in scalp_keys or not far(k)]
        if near:
            sys.exit(f"{args.examples_table}: {len(near)} example(s) lie in the scalp part or within {args.gap_s:g} s "
                     f"of a scalp clip (rows {', '.join(map(str, near[:10]))}); raters would meet them twice")
        if len(set(keys)) < len(keys):
            sys.exit(f"{args.examples_table} lists the same second twice")
        for c in ("subq_path", "subq_center_s", "subq_rate", "edf_clock"):
            if c in ex.columns and c not in cols:  # the examples' UNEEG file, even if the scalp table has none
                scalp[c] = ""
                cols.append(c)
        examples = pd.DataFrame({c: ex[c] if c in ex.columns else "" for c in cols}).reset_index(drop=True)
        examples["ref_answer"] = ex["ref_answer"].values
        examples["ref_type"] = ex["ref_type"].values
        examples["ref_note"] = ex["ref_note"].values
    elif args.examples_n:
        pool = ref[(ref["decision"] == "yes") & ref["_k"].map(lambda k: k not in scalp_keys and far(k))]
        if len(pool) < args.examples_n:
            print(f"note: only {len(pool)} reference Yes seconds lie outside the scalp part; using all of them")
        pick = pool.iloc[np.sort(rng.choice(len(pool), size=min(args.examples_n, len(pool)), replace=False))] \
            if len(pool) else pool
        examples = pd.DataFrame({c: pick[c] if c in pick.columns else "" for c in cols}).reset_index(drop=True)
        examples["ref_answer"] = "yes"
        examples["ref_type"] = ""
        examples["ref_note"] = ""
    else:
        examples = pd.DataFrame(columns=cols + ["ref_answer", "ref_type", "ref_note"])

    # UNEEG part: scalp seconds with a SubQ file
    def _filled(c):
        v = scalp.get(c, pd.Series("", index=scalp.index)).str.strip()
        return ~v.str.lower().isin(["", "nan", "none", "null"])

    has_sq = _filled("subq_path") & _filled("subq_center_s")
    cand = scalp[has_sq].copy()
    cand["_a"] = [answer.get(k, "") for k in _clip_key(cand)]
    by = []
    if args.uneeg_pick == "bins":
        by = [c.strip() for c in args.uneeg_bins_by.split(",") if c.strip()]
        miss = [c for c in by if c not in scalp.columns]
        if not by or miss:
            sys.exit(f"--uneeg-bins-by: {args.scalp} has no column {', '.join(miss) or '(none given)'}; its score "
                     f"columns are {', '.join(c for c in cols if c.endswith('_prob')) or 'none'}")
        bins = np.column_stack([_score_bin(cand[c]) for c in by])
        ok = (bins >= 0).all(axis=1)
        if not ok.all():
            print(f"note: {int((~ok).sum())} scalp clips with a SubQ file have no score in {', '.join(by)} and "
                  "are left out of the UNEEG part")
        cand, bins = cand[ok], bins[ok]
        if len(cand) < args.uneeg_n:
            print(f"note: only {len(cand)} scalp clips with a SubQ file for the UNEEG part (wanted {args.uneeg_n})")
        uneeg = cand.iloc[pick_by_bins(bins, args.uneeg_n, rng)][cols] if len(cand) else pd.DataFrame(columns=cols)
    else:
        n_yes = int(round(args.uneeg_n * args.uneeg_yes_frac))
        picks = []
        for a, want in (("yes", n_yes), ("no", args.uneeg_n - n_yes)):
            pool = cand[cand["_a"] == a]
            if len(pool) < want:
                print(f"note: only {len(pool)} {a.title()} seconds with a SubQ file for the UNEEG part (wanted {want})")
            if len(pool):
                picks.append(pool.iloc[np.sort(rng.choice(len(pool), size=min(want, len(pool)), replace=False))])
        uneeg = pd.concat(picks)[cols] if picks else pd.DataFrame(columns=cols)

    # rows in the order raters meet the parts (kit 1.8.4): the 10-20 clips, the examples, the UNEEG clips
    blank = {"ref_answer": "", "ref_type": "", "ref_note": ""}
    out = pd.concat([scalp.assign(part="scalp", **blank), examples.assign(part="examples"),
                     uneeg.assign(part="uneeg", **blank)], ignore_index=True)
    out = out[cols + ["part", "ref_answer", "ref_type", "ref_note"]].fillna("")
    # every clip of one study on one EDF clock (kit 1.8.2): a row with no edf_clock is read on the samples clock,
    # which can sit up to 1 s from the onsets clock on EDF+D files, so a table that mixes the two is refused
    if args.edf_clock:
        if "edf_clock" not in out.columns:
            out["edf_clock"] = ""
            cols.append("edf_clock")
        out.loc[out["edf_clock"].astype(str).str.strip() == "", "edf_clock"] = args.edf_clock
        out = out[[c for c in out.columns if c not in PART_COLS] + list(PART_COLS)]
    if "edf_clock" in out.columns:
        found = set(out["edf_clock"].astype(str).str.strip().replace("", "samples"))
        if len(found) > 1:
            n_none = int((out["edf_clock"].astype(str).str.strip() == "").sum())
            if n_none:
                sys.exit(f"the clips are on different EDF clocks ({', '.join(sorted(found))}; {n_none} rows name none, "
                         "which means samples). Give --edf-clock with the clock the scalp table's times are on "
                         "(onsets for tables from the onsets-clock seconds) to fill the rows that name none")
            sys.exit(f"the clips name different EDF clocks ({', '.join(sorted(found))}); give the examples on the "
                     "scalp table's clock")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)
    n_ex_sq = int(((examples.get("subq_path", pd.Series("", index=examples.index)).astype(str) != "") &
                   (examples.get("subq_center_s", pd.Series("", index=examples.index)).astype(str) != "")).sum())
    print(f"wrote {args.out}: {len(examples)} training examples ({n_ex_sq} with UNEEG channels), {len(scalp)} scalp "
          f"clips, {len(uneeg)} UNEEG clips")
    if args.uneeg_pick == "bins":
        for c in by:
            got = np.bincount(_score_bin(uneeg[c]), minlength=10) if len(uneeg) else np.zeros(10, int)
            have = np.bincount(_score_bin(cand[c]), minlength=10) if len(cand) else np.zeros(10, int)
            print(f"  UNEEG clips per 0.1 bin of {c}: {', '.join(map(str, got))} (of {', '.join(map(str, have))} "
                  "with a SubQ file)")
    else:
        print(f"  {int((uneeg.index.isin(cand.index[cand['_a'] == 'yes'])).sum())} reference Yes, "
              f"{int((uneeg.index.isin(cand.index[cand['_a'] == 'no'])).sum())} reference No")
    n_no_ref = int((cand["_a"] == "").sum())
    print(f"  scalp clips with a SubQ file: {int(has_sq.sum())}"
          + (f"; with no reference answer: {n_no_ref}" if ref is not None else "")
          + (f"; {n_dup} repeated reference rows (the last kept)" if n_dup else ""))


def cmd_join(args):
    """Several scorings of the same seconds side by side, one <name>_prob column each."""
    pairs = []
    for spec in args.tables:
        name, sep, path = spec.partition("=")
        name = name.strip().lower()
        if not sep or not name or not path:
            sys.exit(f"{spec!r}: give each table as name=path, e.g. cyclops_sd=cyclops_sd_seconds.csv")
        if not all(ch.isalnum() or ch == "_" for ch in name):
            sys.exit(f"{name!r}: names are letters, digits and _ (they become column names)")
        pairs.append((name, path))
    if len(pairs) < 2 and not (args.pool or args.drop_flagged):
        sys.exit("join needs two or more tables (or one, with --pool or --drop-flagged)")
    if len({n for n, _ in pairs}) < len(pairs):
        sys.exit("each table needs its own name")
    key = ["source_path", "start_s"]
    joined, counts, clocks = None, [], {}
    for name, path in pairs:
        t = read_table(path)
        need = ["patient", "source_path", "start_s", "prob"]
        miss = [c for c in need if c not in t.columns]
        if miss:
            sys.exit(f"{path} is missing {miss}; join reads the per-second tables from-runs writes")
        # the same second means the same samples only if every table counts time on the same clock
        found = set(t["edf_clock"].dropna().astype(str)) if "edf_clock" in t.columns else {"samples"}
        if len(found) != 1:
            sys.exit(f"{path} mixes EDF clocks ({', '.join(sorted(found))}); remake it with one --edf-clock")
        clocks[name] = found.pop()
        if len(set(clocks.values())) > 1:
            sys.exit("these tables count time on different EDF clocks (" +
                     ", ".join(f"{n}: {c}" for n, c in clocks.items()) +
                     "); a table with no edf_clock column is on the samples clock. Remake them on one clock "
                     "(from-runs --edf-clock), or add the column if a table was written another way")
        t = t.copy()
        t["edf_clock"] = clocks[name]
        t["start_s"] = pd.to_numeric(t["start_s"], errors="coerce").round(6)
        if t.duplicated(key).any():
            sys.exit(f"{path} repeats a source_path and start_s; each scored second should appear once")
        counts.append((name, len(t)))
        t = t.rename(columns={"prob": f"{name}_prob"})
        if joined is None:
            joined = t.drop(columns=[c for c in ("detector",) if c in t.columns])
            joined["detector"] = name
            continue
        t = t[key + ["patient", f"{name}_prob"]].rename(columns={"patient": "_patient"})
        got = _nearest(joined, t, ["_patient", f"{name}_prob"], args.match_tol_s, path)
        joined = joined.assign(_patient=got["_patient"], **{f"{name}_prob": got[f"{name}_prob"]})
        joined = joined[joined["_patient"].notna()].copy()
        joined[f"{name}_prob"] = pd.to_numeric(joined[f"{name}_prob"])
        bad = (joined["_patient"].astype(str) != joined["patient"].astype(str)).sum()
        if bad:
            sys.exit(f"{bad} seconds have a different patient in {path} than in {pairs[0][1]}; "
                     "were both tables made with the same admissions table?")
        joined = joined.drop(columns=["_patient"])
    if joined.empty:
        sys.exit("the tables share no scored seconds (same --clip-root and clock for each from-runs run?)")
    n_joined = len(joined)
    notes = []
    if args.drop_flagged:
        joined, note = _drop_flagged(joined, args)
        notes.append(note)
    if args.pool:
        joined, note = _apply_pool(joined, args, clocks)
        notes.append(note)
    if joined.empty:
        sys.exit("no seconds left after --pool and --drop-flagged")
    joined = joined.sort_values(key, kind="stable")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    joined.to_csv(args.out, index=False)
    for name, n in counts:
        left = n - n_joined
        print(f"  {name:<14} {n:>10} scored seconds" + (f", {left} not scored by every table (left out)" if left else ""))
    for note in notes:
        print("  " + note)
    print(f"wrote {args.out}: {len(joined)} seconds scored by all {len(pairs)} tables, columns "
          f"{', '.join(n + '_prob' for n, _ in pairs)}; sample it with --prob-cols "
          f"{','.join(n + '_prob' for n, _ in pairs)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sample", help="draw the weighted clip table")
    p.add_argument("table", help="per-second MORGOTH output, CSV or Parquet")
    p.add_argument("--out", required=True, help="clip table to write (CSV), read by make_review_batches.py pack")
    p.add_argument("--n", type=int, default=1000, help="clips to draw (default 1000)")
    p.add_argument("--bins", default=DEFAULT_BINS, help="probability bin edges (default deciles)")
    p.add_argument("--alloc", choices=["equal", "proportional"], default="equal",
                   help="how clips are shared across bins (default equal)")
    p.add_argument("--within", choices=["balanced", "proportional"], default="balanced",
                   help="how a bin's clips are shared across patients (default balanced)")
    p.add_argument("--max-patient-frac", type=float, default=0.5,
                   help="with --within balanced, the most of one bin a single patient may take (default 0.5)")
    p.add_argument("--min-gap-s", type=float, default=0.0,
                   help="keep sampled centers in one file at least this far apart, s (default 0, off). Stops one spike "
                        "from appearing in two clips; weights then hold for each cell's total but are approximate per clip")
    p.add_argument("--pilot", type=int, default=0, help="also write a pilot table of this many clips, spread across bins")
    p.add_argument("--clip-s", type=float, default=12.0, help="clip length including padding, s (match pack; default 12)")
    p.add_argument("--candidate-s", type=float, default=1.0, help="length of one scored window, s (default 1)")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--patient-col", default="patient")
    p.add_argument("--path-col", default="source_path")
    p.add_argument("--start-col", default="start_s")
    p.add_argument("--prob-col", default="prob")
    p.add_argument("--prob-cols", help="comma-separated score columns from join (e.g. cyclops_prob,cyclops_sd_prob); "
                                       "each second's bin is set by the highest of them")
    p.add_argument("--draw", action="append", metavar="COLUMN=N",
                   help="a draw of N clips binned on one score column from join, e.g. spikenet2_prob=1000 "
                        "(repeatable: each draw is made on its own and the clips merged; --n is then not used)")
    p.add_argument("--random", type=int, default=0,
                   help="also draw this many clips with no score, shared evenly across patients (with --draw, or alone)")
    p.add_argument("--pad-s", type=float, default=1.0,
                   help="padding on each side of the clip, s (match pack; default 1): the seconds a rater sees, "
                        "clip-s minus twice this, set each scoring's <name>_clipmax")
    p.add_argument("--grid-s", type=float, default=0.0,
                   help="keep one candidate second in every this many seconds of each file, at an offset drawn per "
                        "file (default 0, off); with 12, no two clips share the 10 s a rater sees, without --min-gap-s, "
                        "and the weights stay exact")
    p.add_argument("--disjoint", action="store_true",
                   help="with several draws: run them in the order given, each leaving out the candidates closer than "
                        "--min-gap-s (or --clip-s) to an earlier draw's clips, so no two clips share EEG; the merged "
                        "weights follow from re-running the draw (--disjoint-reps), and p_<draw> columns keep each "
                        "draw's chance")
    p.add_argument("--disjoint-reps", type=int, default=100,
                   help="with --disjoint, how many re-runs of the draw set each clip's chance (default 100)")
    p.set_defaults(func=cmd_sample)

    j = sub.add_parser("join", help="put several scorings of the same seconds side by side (raw and SD-matched, ...)")
    j.add_argument("tables", nargs="+", help="name=path for each per-second table from from-runs, e.g. cyclops=a.csv")
    j.add_argument("--out", required=True, help="joined table to write (CSV), for sample --prob-cols")
    j.add_argument("--pool", help="keep only seconds inside these intervals: a table with source_path (or path), "
                                  "start_s and end_s, optionally subq_path, subq_t0_s, subq_rate and implant_side "
                                  "(the SubQ-overlap pool); its SubQ columns are carried to every second")
    j.add_argument("--drop-flagged", metavar="TABLE",
                   help="leave out seconds this table flags: source_path (or path), start_s and a 0/1 column "
                        "(artifact, flag or flagged; or --flag-col)")
    j.add_argument("--flag-col", help="the flag column of --drop-flagged, if not found automatically")
    j.add_argument("--drop-unflagged", action="store_true",
                   help="with --drop-flagged, also leave out seconds the flag table has no row for")
    j.add_argument("--candidate-s", type=float, default=1.0, help="length of one scored window, s (default 1)")
    j.add_argument("--match-tol-s", type=float, default=0.001,
                   help="the same second in two tables (or the flag table) may differ by up to this much in start_s, "
                        "from rounding (default 0.001 s)")
    j.set_defaults(func=cmd_join)

    pt = sub.add_parser("parts", help="the clip table of a staged study: training examples, scalp clips, UNEEG clips")
    pt.add_argument("--scalp", required=True, help="the clip table of the scalp part (from sample)")
    pt.add_argument("--ref", help="reference answers with source_path, center_s and decision (merged_labels.csv "
                                  "from merge_labels.py); needed unless --examples-table and --uneeg-pick bins")
    pt.add_argument("--ref-rater", help="use only this rater's answers from --ref, e.g. ADV")
    pt.add_argument("--examples-table", help="the training examples, chosen by hand (CSV): source_path, center_s, "
                                             "and subq_path, subq_center_s to show the UNEEG channels too; ref_answer "
                                             "(yes or no, default yes), ref_type (generalized, focal left or focal "
                                             "right) and ref_note, shown with each example (kit 1.8.5)")
    pt.add_argument("--examples-n", type=int, default=10,
                    help="without --examples-table: training examples, reference Yes seconds (default 10)")
    pt.add_argument("--gap-s", type=float, default=12.0,
                    help="a training example lies at least this far from every scalp clip in its file (default 12)")
    pt.add_argument("--uneeg-n", type=int, default=250, help="UNEEG-only clips (default 250)")
    pt.add_argument("--uneeg-pick", choices=("bins", "ref"), default="bins",
                    help="bins: spread the UNEEG clips evenly over the score bins of --uneeg-bins-by (default); "
                         "ref: by the reference's answers (--uneeg-yes-frac)")
    pt.add_argument("--uneeg-bins-by", default="morgoth_prob,spikenet2_prob",
                    help="score columns of --scalp whose 0.1-wide bins the UNEEG clips fill evenly, n/10 per bin of "
                         "each (default morgoth_prob,spikenet2_prob)")
    pt.add_argument("--uneeg-yes-frac", type=float, default=0.5,
                    help="with --uneeg-pick ref: share of the UNEEG clips the reference called Yes; the rest it "
                         "called No (default 0.5)")
    pt.add_argument("--edf-clock", choices=("samples", "onsets"),
                    help="the EDF clock of rows that name none (no edf_clock value): set it when the scalp table has "
                         "no edf_clock column but its times are on the onsets clock. The study must end up on one clock")
    pt.add_argument("--seed", type=int, default=2026)
    pt.add_argument("--out", required=True, help="the staged clip table to write (CSV)")
    pt.set_defaults(func=cmd_parts)

    a = sub.add_parser("admin", help="the owner-only view's extras for each clip: SubQ map and detector strips")
    a.add_argument("clips", help="the clip table sample wrote")
    a.add_argument("--out", required=True, help="admin table to write (CSV), for --admin in serve_review.py and make_site.py")
    a.add_argument("--scalp", action="append", metavar="NAME=TABLE",
                   help="a detector's per-second scalp scores (a from-runs table, or source_path/path, start_s, prob); "
                        "repeatable")
    a.add_argument("--subq", action="append", metavar="NAME=TABLE",
                   help="a detector's per-second scores on the SubQ files (path = the SubQ EDF), mapped onto each "
                        "clip through subq_center_s and subq_rate; repeatable")
    a.add_argument("--clip-s", type=float, default=12.0, help="clip length including padding, s (default 12)")
    a.add_argument("--pad-s", type=float, default=1.0, help="padding on each side, s (default 1)")
    a.add_argument("--candidate-s", type=float, default=1.0, help="length of one scored window, s (default 1)")
    a.add_argument("--virtual", action="append", metavar="SIDE=PAIRS",
                   help="virtual UNEEG lanes: scalp bipolar pairs for a left or right implant, e.g. L=F7-T3 or "
                        "R=F8-T4,T4-T6 (repeatable; default L=F7-T3 and R=F8-T4; a clip with no known side shows both)")
    a.set_defaults(func=cmd_admin)

    f = sub.add_parser("fake-morgoth", help="write made-up per-second probabilities for test EDFs")
    f.add_argument("--edf-dir", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--seed", type=int, default=11)
    f.set_defaults(func=cmd_fake_morgoth)

    def clock_opt(q):
        q.add_argument("--edf-clock", choices=["samples", "onsets"], default="samples",
                       help="the clock the times are on: samples (0 is a file's first sample; EDF+D files on their "
                            "record onsets; the EMU runs) or onsets (every EDF+ file on its record onsets, EDF+C "
                            "included; the thesis segio reader and the UNEEG runs). Written to the table, and the "
                            "review server reads the clips on the same clock")

    def run_opts(q):
        q.add_argument("run_dir", help="a detector's output folder (searched recursively) or one table")
        q.add_argument("--files", help="only tables whose file name matches this pattern, e.g. '*seconds*.csv'")
        q.add_argument("--condition", help="keep only this condition (montage), e.g. full19")
        q.add_argument("--time-is", choices=["auto", "start", "centre"], default="auto",
                       help="whether the time column holds window starts or centres (default: work it out)")
        for name in ("time", "score", "condition", "clip", "path", "patient"):
            q.add_argument(f"--{name}-col", help=f"name of the {name} column, if it is not found automatically")

    i = sub.add_parser("inspect", help="show the column layout of a detector's output folder (no values)")
    run_opts(i)
    i.set_defaults(func=cmd_inspect)

    r = sub.add_parser("from-runs", help="turn a detector's per-clip output into the per-second table sample reads")
    run_opts(r)
    r.add_argument("--out", required=True, help="per-second table to write (CSV)")
    r.add_argument("--detector", default="morgoth", help="name written to the detector column (default morgoth)")
    r.add_argument("--clip-root", help="folder holding the EDF clips, when the tables carry no path column")
    r.add_argument("--admissions", help="admission-to-patient table (data_derived/emu_admissions.csv)")
    r.add_argument("--candidate-s", type=float, default=1.0, help="length of one scored window, s (default 1)")
    r.add_argument("--score-range", choices=["probability", "any"], default="probability",
                   help="refuse scores outside 0 to 1 unless 'any' (default probability)")
    r.add_argument("--no-check-edf", action="store_true", help="skip checking times against each EDF's length")
    clock_opt(r)
    r.set_defaults(func=cmd_from_runs)

    def file_opts(q):
        clock_opt(q)
        q.add_argument("--clip-root", help="folder holding the EDF clips (searched recursively)")
        q.add_argument("--pattern", default="*", help="only files whose name matches, e.g. '*iic*' (default all)")
        q.add_argument("--clip-list", help="only these clips: a table with a path or clip column, or one per line "
                                              "(paths may be relative to --clip-root; bare clip names are matched by file name)")
        q.add_argument("--list-col", help="the column of --clip-list to use, if not found automatically")

    e = sub.add_parser("from-edf", help="every second of the EDF files, with no score (before any detector has run)")
    file_opts(e)
    e.add_argument("--out", required=True, help="per-second table to write (CSV)")
    e.add_argument("--admissions", help="admission-to-patient table (data_derived/emu_admissions.csv)")
    e.add_argument("--candidate-s", type=float, default=1.0, help="length of one candidate window, s (default 1)")
    e.set_defaults(func=cmd_from_edf)

    c = sub.add_parser("check-data", help="read a sample of the EDF files the way the review server does (aggregates only)")
    file_opts(c)
    c.add_argument("--n", type=int, default=20, help="files to check (default 20)")
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--show-labels", action="store_true", help="also list the names of channels that are not 10-20 or EKG")
    c.set_defaults(func=cmd_check_data)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
