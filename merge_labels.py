#!/usr/bin/env python3
"""Join exported label CSVs with key.csv and summarize them.

    python merge_labels.py --key review_batches/key.csv labels_b01_EC_*.csv --out merged_labels.csv

Writes one row per (clip, rater) with every key column attached, then prints:
decision counts per rater, the Yes rate per probability bin when key.csv has a 'bin'
column, test-retest agreement on repeated clips, agreement between raters who labeled
the same clips (Cohen's kappa for each pair; Fleiss' kappa and Gwet's AC1 across all raters),
a weighted prevalence when key.csv has a 'sampling_weight' column, and accuracy against
planted truth for synthetic batches.
"""
import argparse
import sys

import pandas as pd

DECISIONS = ["yes", "no", "unsure"]


def cohen_kappa(a, b, cats=DECISIONS):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((list(a).count(c) / n) * (list(b).count(c) / n) for c in cats)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def rater_counts(df, cats=DECISIONS):
    """Clips x answers: how many raters gave each answer, for clips that two or more raters answered."""
    ans = df[df["decision"].isin(cats)]
    if ans.empty:
        return pd.DataFrame(columns=cats)
    t = pd.crosstab(ans["clip_id"], ans["decision"]).reindex(columns=cats, fill_value=0)
    return t[t.sum(axis=1) >= 2]


def fleiss_ac1(counts, q):
    """Fleiss' kappa and Gwet's AC1 from a clips x answers count table with q possible answers. The number
    of raters may differ between clips (two or more each). Both share the observed agreement; they differ in
    the chance term, and AC1's stays small when nearly every answer is the same (as No will be), where
    kappa can fall near zero even with high agreement."""
    import numpy as np
    t = np.asarray(counts, float)
    if t.ndim != 2 or len(t) == 0 or q < 2:
        return float("nan"), float("nan")
    r = t.sum(1)
    pa = float(np.mean((t * (t - 1)).sum(1) / (r * (r - 1))))
    pi = (t / r[:, None]).mean(0)
    pe_k = float((pi ** 2).sum())
    pe_g = float((pi * (1 - pi)).sum() / (q - 1))
    kappa = (pa - pe_k) / (1 - pe_k) if pe_k < 1 else float("nan")
    ac1 = (pa - pe_g) / (1 - pe_g) if pe_g < 1 else float("nan")
    return kappa, ac1


def agreement_all(df, n_boot=1000, seed=0):
    """Fleiss' kappa and Gwet's AC1 over every clip that two or more raters answered, with 95% intervals from
    resampling clips (needs 10 or more). Unsure is an answer of its own when anyone gave it; otherwise the
    answers are Yes and No."""
    import numpy as np
    cats = DECISIONS if (df["decision"] == "unsure").any() else ["yes", "no"]
    t = rater_counts(df, cats).to_numpy(float)
    out = {"n_clips": int(len(t)), "answers": cats, "fleiss_ci": (float("nan"),) * 2, "ac1_ci": (float("nan"),) * 2}
    out["fleiss"], out["ac1"] = fleiss_ac1(t, len(cats))
    if len(t) >= 10:
        rng = np.random.default_rng(seed)
        boots = np.array([fleiss_ac1(t[rng.integers(0, len(t), len(t))], len(cats)) for _ in range(n_boot)])
        for col, name in ((0, "fleiss_ci"), (1, "ac1_ci")):
            v = boots[:, col][~np.isnan(boots[:, col])]
            if len(v):
                out[name] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return out


def load_labels(paths):
    frames = []
    for p in paths:
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        if "clip_id" not in df.columns or "decision" not in df.columns:
            sys.exit(f"{p} does not look like an exported labels file (no clip_id or decision column)")
        df["source_file"] = p
        frames.append(df)
    lab = pd.concat(frames, ignore_index=True)
    lab["rater"] = lab["rater"].str.upper()
    # one row per clip and rater: prefer rows with an answer, then the latest labeled_at
    lab["_has"] = lab["decision"].isin(DECISIONS)
    lab = lab.sort_values(["_has", "labeled_at"]).drop_duplicates(["clip_id", "rater"], keep="last")
    return lab.drop(columns="_has")


def merge(key, lab, key_name="key.csv"):
    """Label rows joined to key.csv's columns (patient, scores, sampling weight), one row per clip and rater."""
    unknown = sorted(set(lab["clip_id"]) - set(key["clip_id"]))
    if unknown:
        print(f"WARNING: {len(unknown)} labeled clip IDs are not in {key_name} (first: {unknown[:3]})")
    merged = lab.merge(key, on="clip_id", how="inner", suffixes=("", "_key"))
    if "batch_id_key" in merged.columns:
        clash = merged[merged["batch_id"] != merged["batch_id_key"]]
        if len(clash):
            print(f"WARNING: {len(clash)} rows name a different batch than {key_name} says")
        merged = merged.drop(columns="batch_id_key")
    return merged


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("labels", nargs="+", help="label CSVs exported from ied_review.html")
    ap.add_argument("--key", required=True, help="key.csv written by make_review_batches.py")
    ap.add_argument("--out", default="merged_labels.csv", help="output CSV (default merged_labels.csv)")
    args = ap.parse_args(argv)

    key = pd.read_csv(args.key, dtype=str, keep_default_na=False)
    merged = merge(key, load_labels(args.labels), args.key)
    merged.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(merged)} rows, {merged['rater'].nunique()} rater(s)\n")

    parts = [p for p in merged["part"].unique() if p] if "part" in merged.columns else []
    if not parts:
        summarize(merged)
        return
    # a staged study (kit 1.8): training answers are checked against the expert's, and each scored part is
    # summarized on its own (the UNEEG part asks the same question about the same kind of second on other channels)
    names = {"examples": "training examples", "scalp": "scalp clips", "uneeg": "UNEEG clips"}
    seen = merged[(merged["part"] == "examples") & (merged["decision"] == "seen")]
    if len(seen):  # kit 1.8.5: examples are looked at, not answered
        print("training examples (shown, not asked):")
        for rater, g in seen.groupby("rater"):
            print(f"  rater {rater}: {len(g)} seen")
    ex = merged[(merged["part"] == "examples") & merged["decision"].isin(DECISIONS)]
    if len(ex):
        print("training examples (not scored):")
        for rater, g in ex.groupby("rater"):
            ref = g["ref_answer"].str.strip().str.lower() if "ref_answer" in g.columns else ""
            same = int((g["decision"] == ref).sum())
            print(f"  rater {rater}: {len(g)} answered, {same} matched the expert's answer")
    for part in ("scalp", "uneeg"):
        sub = merged[merged["part"] == part]
        if len(sub):
            print(f"\n==== {names[part]} ====")
            summarize(sub)


def summarize(merged):
    """The printed summary of one set of merged labels."""
    answered = merged[merged["decision"].isin(DECISIONS)]
    for rater, g in merged.groupby("rater"):
        counts = g["decision"].value_counts()
        secs = pd.to_numeric(g["seconds_viewed"], errors="coerce")
        print(f"rater {rater}: {int(g['decision'].isin(DECISIONS).sum())} of {len(g)} answered "
              f"(yes {counts.get('yes', 0)}, no {counts.get('no', 0)}, unsure {counts.get('unsure', 0)}), "
              f"median {secs.median():.0f} s per clip")

    if "bin" in merged.columns and len(answered):
        print("\nYes rate by bin (unsure counted separately):")
        rows = []
        for b, g in answered.groupby("bin"):
            yes, no, uns = (g["decision"] == "yes").sum(), (g["decision"] == "no").sum(), (g["decision"] == "unsure").sum()
            rate = yes / (yes + no) if yes + no else float("nan")
            rows.append((b, len(g), yes, no, uns, rate))
        rows.sort(key=lambda r: float(str(r[0]).split("-")[0]) if str(r[0]).split("-")[0].replace(".", "").isdigit() else 0)
        print(f"  {'bin':>9} {'n':>4} {'yes':>4} {'no':>4} {'uns':>4} {'yes/(yes+no)':>13}")
        for b, n, y, no, u, r in rows:
            print(f"  {b:>9} {n:>4} {y:>4} {no:>4} {u:>4} {r:>13.2f}")

    if "sampling_weight" in merged.columns and len(answered):
        w = pd.to_numeric(answered["sampling_weight"], errors="coerce")
        decided = answered["decision"].isin(["yes", "no"]) & w.notna()
        if decided.any():
            est = (w[decided] * (answered.loc[decided, "decision"] == "yes")).sum() / w[decided].sum()
            print(f"\nweighted Yes prevalence (unsure excluded): {est:.3f}")

    if "repeat_of" in merged.columns:
        reps = merged[merged["repeat_of"] != ""]
        if len(reps):
            pairs = reps.merge(merged, left_on=["repeat_of", "rater"], right_on=["clip_id", "rater"], suffixes=("_rep", "_orig"))
            pairs = pairs[pairs["decision_rep"].isin(DECISIONS) & pairs["decision_orig"].isin(DECISIONS)]
            if len(pairs):
                for rater, g in pairs.groupby("rater"):
                    agree = (g["decision_rep"] == g["decision_orig"]).mean()
                    print(f"\ntest-retest, rater {rater}: {len(g)} repeated clips, agreement {agree:.2f}, "
                          f"kappa {cohen_kappa(g['decision_orig'], g['decision_rep']):.2f}")

    raters = sorted(answered["rater"].unique())
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            a = answered[answered["rater"] == raters[i]].set_index("clip_id")["decision"]
            b = answered[answered["rater"] == raters[j]].set_index("clip_id")["decision"]
            both = a.index.intersection(b.index)
            if len(both):
                print(f"\nraters {raters[i]} and {raters[j]}: {len(both)} shared clips, "
                      f"agreement {(a[both] == b[both]).mean():.2f}, kappa {cohen_kappa(a[both], b[both]):.2f}")
    if len(raters) > 1:
        main_rows = answered[answered["repeat_of"] == ""] if "repeat_of" in answered.columns else answered
        ag = agreement_all(main_rows)
        if ag["n_clips"]:
            ci = lambda v: "" if v[0] != v[0] else f" (95% CI {v[0]:.2f} to {v[1]:.2f})"
            print(f"\nall raters, {ag['n_clips']} clips answered by two or more: Fleiss' kappa {ag['fleiss']:.2f}"
                  f"{ci(ag['fleiss_ci'])}, Gwet's AC1 {ag['ac1']:.2f}{ci(ag['ac1_ci'])}")

    if "truth" in merged.columns and len(answered):
        print("\nagainst planted truth (synthetic batches):")
        t = answered.assign(planted=answered["truth"].map(lambda s: "spike" if s == "spike" else "no spike"))
        print(pd.crosstab(t["planted"], t["decision"]).to_string())
        if "truth_max" in t.columns:
            located = t[(t["truth"] == "spike") & (t["decision"] == "yes") & (t["max_electrodes"] != "")]
            if len(located):
                hit = located.apply(lambda r: r["truth_max"] in r["max_electrodes"].split(";"), axis=1).mean()
                print(f"maximum electrode matched the planted site in {hit:.0%} of {len(located)} located spikes")


if __name__ == "__main__":
    main()
