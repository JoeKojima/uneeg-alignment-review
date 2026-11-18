#!/usr/bin/env python3
"""One-page results report for returned expert labels.

    python report_labels.py merged_labels.csv --out label_report.html

Reads the merged_labels.csv that merge_labels.py writes (one row per clip and rater, with
key.csv's columns attached) and writes a single HTML file that opens offline. It shows:

  * the weighted share of recorded seconds that experts call an IED, with a 95% interval
    from resampling patients (the patient is the unit of analysis),
  * for every detector scoring in the clips (each <name>_prob column select_clips.py join wrote,
    or the one detector they name): weighted precision, sensitivity and F1 at a threshold
    (0.5 unless --threshold), with 95% intervals from resampling patients; F1 across
    thresholds; and calibration, the weighted expert Yes rate in each tenth of that score,
  * agreement between raters (Cohen's kappa per pair; Fleiss' kappa and Gwet's AC1 across all
    raters, with 95% intervals from resampling clips; each rater against the others' majority
    on shared clips), and test-retest when clips were repeated,
  * the weighted IED share per patient, and the field maximum when raters marked one.

The report holds aggregates only: counts, rates and patient study IDs. It never contains
signal values, file paths or times.

    python report_labels.py demo --out label_report_demo.html

simulates a 1,000-clip study as kit 1.6 runs it (made-up patients, four detector scorings
joined and drawn with select_clips.py, 200 clips every one of four raters labels and the rest
split between them, Yes or No only) and reports it, with the simulated truth beside each
estimate. The page is marked as simulated throughout.
"""
import argparse
import datetime as dt
import html
import math
import os
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DECISIONS = ["yes", "no", "unsure"]
SCALP = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz",
         "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2"]
POS = {
    "Fp1": (-0.247, 0.761), "Fp2": (0.247, 0.761), "F7": (-0.647, 0.470), "F3": (-0.320, 0.420),
    "Fz": (0.0, 0.400), "F4": (0.320, 0.420), "F8": (0.647, 0.470), "T3": (-0.800, 0.0),
    "C3": (-0.400, 0.0), "Cz": (0.0, 0.0), "C4": (0.400, 0.0), "T4": (0.800, 0.0),
    "T5": (-0.647, -0.470), "P3": (-0.320, -0.420), "Pz": (0.0, -0.400), "P4": (0.320, -0.420),
    "T6": (0.647, -0.470), "O1": (-0.247, -0.761), "O2": (0.247, -0.761),
}
# Sequential ramp (one hue, light to dark) for electrode counts; first step means "none".
RAMP_LIGHT = ["#e6ece9", "#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
RAMP_DARK = ["#1f2a25", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#5598e7", "#9ec5f4"]


# ---------------------------------------------------------------- statistics

def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def kappa(a, b, cats=DECISIONS):
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def weighted_share(df, col="w"):
    w = df[col].to_numpy(float)
    y = df["cons_yes"].to_numpy(float)
    return float((w * y).sum() / w.sum()) if w.sum() > 0 else float("nan")


def cluster_ci(df, reps=1000, seed=7):
    """95% interval for the weighted share, resampling patients with replacement."""
    if "patient" not in df.columns or df["patient"].nunique() < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("patient")]
    sums = np.array([[(g["w"] * g["cons_yes"]).sum(), g["w"].sum()] for g in groups])
    est = []
    for _ in range(reps):
        pick = sums[rng.integers(0, len(groups), len(groups))]
        est.append(pick[:, 0].sum() / pick[:, 1].sum())
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def detector_of(merged):
    """(score column, display name): <detector>_prob when the clips name one detector, else morgoth_prob."""
    det = "morgoth"
    if "detector" in merged.columns:
        vals = [v for v in merged["detector"].dropna().astype(str).unique() if v]
        if len(vals) == 1:
            det = vals[0]
    col = f"{det.lower()}_prob" if f"{det.lower()}_prob" in merged.columns else "morgoth_prob"
    return col, ("MORGOTH" if det.lower() == "morgoth" else det[:1].upper() + det[1:])


NAMES = {"morgoth": "MORGOTH", "cyclops": "Cyclops", "spikenet1": "SpikeNet1", "spikenet": "SpikeNet",
         "spikenet2": "SpikeNet2", "persyst": "Persyst"}
THRESHOLDS = [round(0.05 * k, 2) for k in range(1, 20)]


def display_name(name):
    base, sd = (name[:-3], True) if name.endswith("_sd") else (name, False)
    shown = NAMES.get(base.lower(), base[:1].upper() + base[1:])
    return shown + (" SD-matched" if sd else "")


def scorings_of(merged):
    """[(score column, display name)]: each <name>_prob column select_clips.py join wrote (strata_prob, the
    highest of them, sets the sampling bin and is not a scoring), else the one detector the clips name."""
    cols = [c for c in merged.columns if c.endswith("_prob") and c != "strata_prob"
            and pd.to_numeric(merged[c], errors="coerce").notna().any()]
    if len(cols) > 1 or (len(cols) == 1 and "detector" not in merged.columns):
        return [(c, display_name(c[:-5])) for c in cols]
    col, name = detector_of(merged)
    if col in merged.columns and pd.to_numeric(merged[col], errors="coerce").notna().any():
        return [(col, name)]
    return []


def wilson_weighted(y, w):
    """Weighted rate with a Wilson interval on the effective sample size (Kish), for clips that stand for
    different numbers of seconds."""
    if len(y) == 0 or w.sum() <= 0:
        return float("nan"), float("nan"), float("nan"), 0.0
    rate = float((w * y).sum() / w.sum())
    n_eff = float(w.sum() ** 2 / (w ** 2).sum())
    lo, hi = wilson(rate * n_eff, n_eff)
    return rate, lo, hi, n_eff


def confusion(s, y, w, t):
    flag = s >= t
    return np.array([(w * y * flag).sum(), (w * (1 - y) * flag).sum(), (w * y * ~flag).sum(), (w * flag).sum(), w.sum()])


def rates(c):
    tp, fp, fn, flagged, total = c
    return {"ppv": tp / (tp + fp) if tp + fp > 0 else float("nan"),
            "sens": tp / (tp + fn) if tp + fn > 0 else float("nan"),
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else float("nan"),
            "per_hour": 3600 * flagged / total if total > 0 else float("nan"),
            "flag_share": flagged / total if total > 0 else float("nan")}


def scoring_stats(decided, col, threshold, reps=1000, seed=13):
    """Weighted precision, sensitivity and F1 of one scoring against the expert consensus, with intervals from
    resampling patients; F1 across thresholds; calibration by tenths of the score."""
    d = decided.assign(_s=pd.to_numeric(decided[col], errors="coerce")).dropna(subset=["_s"])
    s, y, w = d["_s"].to_numpy(float), d["cons_yes"].to_numpy(float), d["w"].to_numpy(float)
    out = {"n": int(len(d)), "at": rates(confusion(s, y, w, threshold)), "ci": {}}
    out["curve"] = [(t, rates(confusion(s, y, w, t))["f1"]) for t in THRESHOLDS]
    groups = [g.index for _, g in d.reset_index(drop=True).groupby(d["patient"].to_numpy())]
    if len(groups) >= 3:
        per = np.array([confusion(s[ix], y[ix], w[ix], threshold) for ix in groups])
        rng = np.random.default_rng(seed)
        boots = [rates(per[rng.integers(0, len(per), len(per))].sum(axis=0)) for _ in range(reps)]
        for k in ("ppv", "sens", "f1", "per_hour", "flag_share"):
            v = np.array([b[k] for b in boots], float)
            v = v[~np.isnan(v)]
            out["ci"][k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (float("nan"),) * 2
    cal = []
    edges = np.linspace(0, 1, 11)
    for i in range(10):
        inb = (s >= edges[i]) & ((s < edges[i + 1]) if i < 9 else (s <= 1))
        if inb.sum() == 0:
            continue
        rate, lo, hi, n_eff = wilson_weighted(y[inb], w[inb])
        cal.append({"lo_edge": float(edges[i]), "p": float(s[inb].mean()), "n": int(inb.sum()), "k": int(y[inb].sum()),
                    "rate": rate, "lo": lo, "hi": hi, "n_eff": n_eff})
    out["calibration"] = cal
    return out


def analyze(merged, threshold=0.5):
    m = merged.copy()
    # answers given with the admin or superadmin view on (SubQ or detector scores shown) are not blind: left out
    n_not_blind = 0
    if "view" in m.columns:
        nb = m["view"].astype(str).str.lower().isin(["admin", "superadmin"])
        n_not_blind = int(nb.sum())
        m = m[~nb]
    # the clip-level rule (kit 1.7): a detector finds a clip when any second a rater sees reaches the threshold,
    # so each scoring is judged by its <name>_clipmax when the clip table has one
    clip_level = False
    # only in a study that asked about the whole clip (key.csv's question column, kit 1.7 on); a study that asked
    # about the highlighted second is judged on that second's own score
    asked = set(m["question"].astype(str).str.strip()) - {"", "nan"} if "question" in m.columns else set()
    for c in [c for c in m.columns if c.endswith("_prob") and c != "strata_prob" and asked == {"clip"}]:
        cm = c[:-5] + "_clipmax"
        if cm in m.columns and pd.to_numeric(m[cm], errors="coerce").notna().any():
            m[c] = m[cm]
            clip_level = True
    scorings = scorings_of(m)
    score_col, det_name = detector_of(m)
    if score_col != "morgoth_prob":
        m["morgoth_prob"] = m[score_col]
    bin_col = "strata_prob" if "strata_prob" in m.columns else "morgoth_prob"
    for col in ("repeat_of", "max_electrodes", "generalized", "seconds_viewed", "bin", "patient", "morgoth_prob",
                "sampling_weight", "cell_N", "cell_n", "patient_weight", "truth", bin_col):
        if col not in m.columns:
            m[col] = ""
    m = m.fillna("")
    m["rater"] = m["rater"].astype(str).str.upper()
    m["decision"] = m["decision"].astype(str).str.lower()
    main = m[m["repeat_of"].astype(str) == ""]
    reps = m[m["repeat_of"].astype(str) != ""]

    # consensus per clip: more Yes than No among raters who answered Yes or No
    rows = []
    for cid, g in main.groupby("clip_id"):
        yes, no = int((g["decision"] == "yes").sum()), int((g["decision"] == "no").sum())
        first = g.iloc[0]
        w = pd.to_numeric(first["sampling_weight"], errors="coerce")
        # within a patient, each clip stands for cell_N / cell_n of that patient's seconds; sampling_weight
        # also carries the chance the patient was drawn at all, which belongs in the all-patient share only
        wp = pd.to_numeric(first["patient_weight"], errors="coerce")  # several draws merged (sample --draw)
        if pd.isna(wp):
            wp = pd.to_numeric(first["cell_N"], errors="coerce") / pd.to_numeric(first["cell_n"], errors="coerce")
        rows.append({"clip_id": cid, "yes": yes, "no": no, "n_raters": int(g["decision"].isin(DECISIONS).sum()),
                     "cons_yes": 1.0 if yes > no else (0.0 if no > yes else np.nan),
                     "patient": str(first["patient"]) or "all", "bin": str(first["bin"]),
                     "p": pd.to_numeric(first[bin_col], errors="coerce"),
                     **{c: pd.to_numeric(first[c], errors="coerce") for c, _n in scorings},
                     "w": float(w) if pd.notna(w) else 1.0,
                     "w_pat": float(wp) if pd.notna(wp) else (float(w) if pd.notna(w) else 1.0),
                     "truth": str(first["truth"])})
    clips = pd.DataFrame(rows)
    decided = clips.dropna(subset=["cons_yes"])
    weighted = "sampling_weight" in merged.columns and (merged["sampling_weight"].astype(str) != "").any()

    out = {"n_clips": int(clips.shape[0]), "n_decided": int(decided.shape[0]),
           "n_ties": int(clips["cons_yes"].isna().sum()), "raters": sorted(main["rater"].unique()),
           "n_patients": int(clips["patient"].nunique()), "weighted": bool(weighted),
           "n_rows": int(m.shape[0]), "n_repeats": int(reps["clip_id"].nunique()), "detector": det_name,
           "threshold": threshold, "multi": len(scorings) > 1,
           "n_shared": int((clips["n_raters"] >= 2).sum()), "clip_level": clip_level, "n_not_blind": n_not_blind,
           "any_unsure": bool((main["decision"] == "unsure").any())}
    out["scorings"] = [{"col": c, "name": n, **scoring_stats(decided, c, threshold)} for c, n in scorings]
    out["prev"] = weighted_share(decided)
    out["prev_ci"] = cluster_ci(decided)
    out["seconds_represented"] = float(decided["w"].sum())

    # calibration by bin
    cal = []
    for b, g in decided.groupby("bin"):
        k, n = int(g["cons_yes"].sum()), int(len(g))
        lo, hi = wilson(k, n)
        per_rater = {}
        for r in out["raters"]:
            d = main[(main["bin"].astype(str) == b) & (main["rater"] == r) & main["decision"].isin(["yes", "no"])]
            per_rater[r] = (int((d["decision"] == "yes").sum()), int(len(d)))
        cal.append({"bin": b, "p": float(g["p"].mean()), "k": k, "n": n, "rate": k / n if n else float("nan"),
                    "lo": lo, "hi": hi, "seconds": float(g["w"].sum()), "per_rater": per_rater,
                    "pmin": float(g["p"].min()), "pmax": float(g["p"].max())})
    cal.sort(key=lambda c: c["p"])
    out["calibration"] = cal

    # per patient
    pats = []
    rng = np.random.default_rng(11)
    for p, g in decided.groupby("patient"):
        est = weighted_share(g, "w_pat")
        boots = []
        for _ in range(300):
            s = g.iloc[rng.integers(0, len(g), len(g))]
            boots.append(weighted_share(s, "w_pat"))
        pats.append({"patient": p, "est": est, "lo": float(np.nanpercentile(boots, 2.5)),
                     "hi": float(np.nanpercentile(boots, 97.5)), "n": int(len(g)), "k": int(g["cons_yes"].sum())})
    pats.sort(key=lambda r: -r["est"])
    out["patients"] = pats

    # field maximum counts (Yes answers, all raters)
    counts = {e: 0 for e in SCALP}
    yes_rows = main[main["decision"] == "yes"]
    n_located = 0
    for v in yes_rows["max_electrodes"].astype(str):
        els = [e for e in v.split(";") if e in counts]
        if els:
            n_located += 1
        for e in els:
            counts[e] += 1
    out["field"] = counts
    out["n_yes"] = int(len(yes_rows))
    out["n_located"] = n_located
    out["n_generalized"] = int((yes_rows["generalized"].astype(str) == "1").sum())

    # raters
    rstats = []
    for r in out["raters"]:
        g = main[main["rater"] == r]
        ans = g[g["decision"].isin(DECISIONS)]
        secs = pd.to_numeric(g["seconds_viewed"], errors="coerce")
        rr = reps[reps["rater"] == r]
        pairs = rr.merge(main[main["rater"] == r][["clip_id", "decision"]], left_on="repeat_of", right_on="clip_id",
                         suffixes=("_rep", "_orig"))
        pairs = pairs[pairs["decision_rep"].isin(DECISIONS) & pairs["decision_orig"].isin(DECISIONS)]
        rstats.append({"rater": r, "answered": int(len(ans)), "assigned": int(len(g)),
                       "yes": int((ans["decision"] == "yes").sum()), "no": int((ans["decision"] == "no").sum()),
                       "unsure": int((ans["decision"] == "unsure").sum()),
                       "median_s": float(secs.median()) if secs.notna().any() else float("nan"),
                       "retest_n": int(len(pairs)),
                       "retest_agree": float((pairs["decision_rep"] == pairs["decision_orig"]).mean()) if len(pairs) else float("nan"),
                       "retest_kappa": kappa(pairs["decision_orig"], pairs["decision_rep"]) if len(pairs) else float("nan")})
    answered = main[main["decision"].isin(["yes", "no"])]
    by_clip = answered.groupby("clip_id")
    multi = answered[answered["clip_id"].map(by_clip["rater"].nunique()) >= 3]
    for r in rstats:
        mine = multi[multi["rater"] == r["rater"]].set_index("clip_id")["decision"]
        others = multi[multi["rater"] != r["rater"]].groupby("clip_id")["decision"]
        yes, no = others.apply(lambda v: (v == "yes").sum()), others.apply(lambda v: (v == "no").sum())
        maj = pd.Series(np.where(yes > no, "yes", np.where(no > yes, "no", "")), index=yes.index)
        both = mine.index.intersection(maj[maj != ""].index)
        r["vs_n"] = int(len(both))
        r["vs_agree"] = float((mine[both] == maj[both]).mean()) if len(both) else float("nan")
        r["vs_kappa"] = kappa(mine[both], maj[both], ["yes", "no"]) if len(both) else float("nan")
        r["yes_rate"] = r["yes"] / max(1, r["yes"] + r["no"])
    out["rater_stats"] = rstats

    pk = []
    rs = out["raters"]
    for i in range(len(rs)):
        for j in range(i + 1, len(rs)):
            a = main[(main["rater"] == rs[i]) & main["decision"].isin(DECISIONS)].set_index("clip_id")["decision"]
            b = main[(main["rater"] == rs[j]) & main["decision"].isin(DECISIONS)].set_index("clip_id")["decision"]
            both = a.index.intersection(b.index)
            if len(both):
                pk.append({"a": rs[i], "b": rs[j], "n": int(len(both)), "agree": float((a[both] == b[both]).mean()),
                           "kappa": kappa(a[both], b[both])})
    out["pair_kappa"] = pk
    sys.path.insert(0, HERE)
    import merge_labels
    out["agree_all"] = merge_labels.agreement_all(main)
    ks = [p["kappa"] for p in pk if not math.isnan(p["kappa"])]
    out["mean_kappa"] = float(np.mean(ks)) if ks else float("nan")
    rt = [r["retest_agree"] for r in rstats if r["retest_n"]]
    out["mean_retest"] = float(np.mean(rt)) if rt else float("nan")
    secs_all = pd.to_numeric(main["seconds_viewed"], errors="coerce")
    out["median_s"] = float(secs_all.median()) if secs_all.notna().any() else float("nan")

    # synthetic or simulated batches carry the planted truth
    if (clips["truth"] != "").any():
        t = decided[decided["truth"] != ""]
        truth_yes = t["truth"].isin(["spike", "ied", "1", "yes"]).astype(float)
        out["truth"] = {"sens": float(((t["cons_yes"] == 1) & (truth_yes == 1)).sum() / max(1, (truth_yes == 1).sum())),
                        "spec": float(((t["cons_yes"] == 0) & (truth_yes == 0)).sum() / max(1, (truth_yes == 0).sum())),
                        "n": int(len(t))}
    return out


# ---------------------------------------------------------------- rendering helpers

def esc(s):
    return html.escape(str(s), quote=True)


def pct(x, d=1):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{d}f}%"


def num(x, d=2):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def calibration_svg(cal, det="MORGOTH"):
    W, H = 560, 380
    L, R, T, B = 58, 18, 16, 52
    pw, ph = W - L - R, H - T - B
    x = lambda v: L + v * pw
    y = lambda v: T + (1 - v) * ph
    s = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Expert Yes rate against {det} probability, by bin">']
    for t in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        s.append(f'<line class="grid" x1="{L}" x2="{L + pw}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>')
        s.append(f'<text class="tick" x="{L - 8}" y="{y(t) + 4:.1f}" text-anchor="end">{int(t * 100)}%</text>')
        s.append(f'<text class="tick" x="{x(t):.1f}" y="{T + ph + 18}" text-anchor="middle">{t:.1f}</text>')
    s.append(f'<line class="axis" x1="{L}" x2="{L + pw}" y1="{T + ph}" y2="{T + ph}"/>')
    s.append(f'<line class="ref" x1="{x(0):.1f}" y1="{y(0):.1f}" x2="{x(1):.1f}" y2="{y(1):.1f}"/>')
    s.append(f'<text class="reflabel" x="{x(0.62):.1f}" y="{y(0.62) - 8:.1f}" transform="rotate(-{math.degrees(math.atan(ph / pw)):.1f} {x(0.62):.1f} {y(0.62) - 8:.1f})">perfectly calibrated</text>')
    pts = [c for c in cal if not math.isnan(c["rate"])]
    if pts:
        s.append('<polyline class="series" points="' + " ".join(f"{x(c['p']):.1f},{y(c['rate']):.1f}" for c in pts) + '"/>')
    for c in pts:
        tip = (f"Bin {c['bin']} · mean {det} {c['p']:.2f}<br>{c['k']} of {c['n']} clips called IED by consensus: "
               f"<b>{pct(c['rate'])}</b><br>95% CI {pct(c['lo'])} to {pct(c['hi'])}")
        s.append(f'<g class="mark" tabindex="0" data-tip="{esc(tip)}">'
                 f'<line class="ci" x1="{x(c["p"]):.1f}" x2="{x(c["p"]):.1f}" y1="{y(c["lo"]):.1f}" y2="{y(c["hi"]):.1f}"/>'
                 f'<circle class="dot" cx="{x(c["p"]):.1f}" cy="{y(c["rate"]):.1f}" r="5"/>'
                 f'<rect class="hit" x="{x(c["p"]) - 12:.1f}" y="{T}" width="24" height="{ph}"/></g>')
    s.append(f'<text class="axtitle" x="{L + pw / 2}" y="{H - 10}" text-anchor="middle">{det} probability (mean of the bin)</text>')
    s.append(f'<text class="axtitle" transform="translate(14 {T + ph / 2}) rotate(-90)" text-anchor="middle">Clips experts call an IED</text>')
    s.append("</svg>")
    return "".join(s)


def lines_svg(series, xlabel, ylabel, aria, diagonal=False, vline=None, ci=False):
    """One line per scoring on a 0-1 by 0-1 plot. series: [{"name", "slot", "pts": [(x, y, tip, lo, hi)]}].
    Each line ends in its name (text ink); the legend above the chart carries the colours."""
    W, H = 600, 380
    L, R, T, B = 58, 132, 16, 52
    pw, ph = W - L - R, H - T - B
    x = lambda v: L + v * pw
    y = lambda v: T + (1 - v) * ph
    s = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(aria)}">']
    for t in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        s.append(f'<line class="grid" x1="{L}" x2="{L + pw}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>')
        s.append(f'<text class="tick" x="{L - 8}" y="{y(t) + 4:.1f}" text-anchor="end">{int(t * 100)}%</text>')
        s.append(f'<text class="tick" x="{x(t):.1f}" y="{T + ph + 18}" text-anchor="middle">{t:.1f}</text>')
    s.append(f'<line class="axis" x1="{L}" x2="{L + pw}" y1="{T + ph}" y2="{T + ph}"/>')
    if diagonal:
        s.append(f'<line class="ref" x1="{x(0):.1f}" y1="{y(0):.1f}" x2="{x(1):.1f}" y2="{y(1):.1f}"/>')
    if vline is not None:
        s.append(f'<line class="ref dash" x1="{x(vline):.1f}" x2="{x(vline):.1f}" y1="{T}" y2="{T + ph}"/>')
        s.append(f'<text class="reflabel" x="{x(vline) + 5:.1f}" y="{T + 12}">{vline:g}</text>')
    ends = []
    for ser in series:
        pts = [p for p in ser["pts"] if not math.isnan(p[1])]
        if not pts:
            continue
        cls = f"c{ser['slot']}"
        s.append(f'<polyline class="line {cls}" points="' + " ".join(f"{x(p[0]):.1f},{y(p[1]):.1f}" for p in pts) + '"/>')
        for px, py, tip, lo, hi in pts:
            band = (f'<line class="ci {cls}" x1="{x(px):.1f}" x2="{x(px):.1f}" y1="{y(lo):.1f}" y2="{y(hi):.1f}"/>'
                    if ci and not math.isnan(lo) else "")
            s.append(f'<g class="mark" tabindex="0" data-tip="{esc(tip)}">{band}'
                     f'<circle class="pt {cls}" cx="{x(px):.1f}" cy="{y(py):.1f}" r="4"/>'
                     f'<circle class="hit" cx="{x(px):.1f}" cy="{y(py):.1f}" r="11"/></g>')
        ends.append([y(pts[-1][1]), x(pts[-1][0]), ser["name"]])
    ends.sort()
    for i in range(1, len(ends)):  # keep end labels at least a line apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 14)
    for ly, lx, name in ends:
        s.append(f'<text class="endlabel" x="{lx + 8:.1f}" y="{ly + 4:.1f}">{esc(name)}</text>')
    s.append(f'<text class="axtitle" x="{L + pw / 2}" y="{H - 10}" text-anchor="middle">{esc(xlabel)}</text>')
    s.append(f'<text class="axtitle" transform="translate(14 {T + ph / 2}) rotate(-90)" text-anchor="middle">{esc(ylabel)}</text>')
    s.append("</svg>")
    return "".join(s)


def legend(series):
    return ('<div class="legend">' + "".join(f'<span><i class="sw c{x["slot"]}"></i>{esc(x["name"])}</span>' for x in series)
            + "</div>")


def headmap_svg(counts, n_located):
    mx = max(counts.values()) or 1
    S = 112
    s = ['<svg class="chart head" viewBox="-130 -128 260 262" role="img" aria-label="Field maximum marks per electrode">']
    s.append('<circle class="scalp" cx="0" cy="0" r="104"/>')
    s.append('<path class="scalp" d="M -12 -103 L 0 -118 L 12 -103"/>')
    s.append('<path class="scalp" d="M -104 -12 q -10 12 0 24 M 104 -12 q 10 12 0 24"/>')
    for e in SCALP:
        ex, ey = POS[e]
        cx, cy = ex * S, -ey * S
        c = counts[e]
        step = 0 if c == 0 else 1 + min(5, int(5 * c / mx))
        share = c / max(1, sum(counts.values()))
        tip = f"<b>{e}</b>: marked as the maximum {c} time{'s' if c != 1 else ''} ({pct(share)} of all marks)"
        ink = "ink-dark" if step >= 4 else "ink-light"
        s.append(f'<g class="mark" tabindex="0" data-tip="{esc(tip)}"><circle class="el s{step}" cx="{cx:.1f}" cy="{cy:.1f}" r="17"/>'
                 f'<text class="el-name {ink}" x="{cx:.1f}" y="{cy - 2:.1f}" text-anchor="middle">{e}</text>'
                 f'<text class="el-count {ink}" x="{cx:.1f}" y="{cy + 9:.1f}" text-anchor="middle">{c}</text></g>')
    s.append("</svg>")
    return "".join(s)


def patients_svg(pats):
    W = 960
    row = 26
    L, R, T = 92, 70, 22
    H = T + row * len(pats) + 34
    top = max([p["hi"] for p in pats if not math.isnan(p["hi"])] + [0.05])
    step = next(st for st in (0.01, 0.02, 0.05, 0.1, 0.2, 0.25) if top / st <= 5)
    vmax = math.ceil(top / step) * step
    pw = W - L - R
    x = lambda v: L + min(v, vmax) / vmax * pw
    s = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Weighted IED share per patient">']
    ticks = np.arange(0, vmax + step / 2, step)
    for t in ticks:
        s.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{T - 6}" y2="{H - 30}"/>')
        s.append(f'<text class="tick" x="{x(t):.1f}" y="{H - 14}" text-anchor="middle">{t * 100:.0f}%</text>')
    for i, p in enumerate(pats):
        cy = T + i * row + row / 2
        tip = (f"<b>{esc(p['patient'])}</b>: {p['k']} of {p['n']} clips called IED<br>weighted share {pct(p['est'])}"
               f" (95% CI {pct(p['lo'])} to {pct(p['hi'])})")
        s.append(f'<text class="rowlabel" x="{L - 10}" y="{cy + 4:.1f}" text-anchor="end">{esc(p["patient"])}</text>')
        s.append(f'<g class="mark" tabindex="0" data-tip="{esc(tip)}">'
                 f'<line class="ci" x1="{x(p["lo"]):.1f}" x2="{x(p["hi"]):.1f}" y1="{cy:.1f}" y2="{cy:.1f}"/>'
                 f'<circle class="dot" cx="{x(p["est"]):.1f}" cy="{cy:.1f}" r="5"/>'
                 f'<rect class="hit" x="{L}" y="{cy - row / 2:.1f}" width="{pw}" height="{row}"/></g>')
        s.append(f'<text class="rowvalue" x="{W - R + 10}" y="{cy + 4:.1f}">{pct(p["est"])}</text>')
    s.append("</svg>")
    return "".join(s)


CSS = """
/* Layout: one reading column of result panels on the kit's chart-paper ground; the calibration plot leads. */
:root {
  --paper: #f4f7f5; --surface: #ffffff; --surface-2: #eaf0ec; --line: #d2ddd6; --grid: #e2eae5;
  --ink: #16211c; --ink-2: #4c5d55; --ink-3: #6b7d74; --accent: #0b6e5f; --accent-soft: #d6ebe4;
  --warn: #91600b; --warn-soft: #f5e7cb; --ref: #9fb3a9;
  --s0: #e6ece9; --s1: #b7d3f6; --s2: #86b6ef; --s3: #5598e7; --s4: #2a78d6; --s5: #1c5cab; --s6: #104281;
  --on-dark: #ffffff; --on-light: #16211c;
  /* one colour per scoring, in a fixed order (validated for colour-blind separation in both modes) */
  --c1: #2a78d6; --c2: #eb6834; --c3: #1baf7a; --c4: #eda100; --c5: #e87ba4; --c6: #6250d6;
  --font-ui: "Atkinson Hyperlegible", system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --c1: #3987e5; --c2: #d95926; --c3: #199e70; --c4: #c98500; --c5: #d55181; --c6: #9085e9;
    --paper: #0d1411; --surface: #131c18; --surface-2: #1a2520; --line: #283731; --grid: #1f2b26;
    --ink: #e2ece6; --ink-2: #a2b4aa; --ink-3: #8a9d93; --accent: #4fc4aa; --accent-soft: #173b32;
    --warn: #e5b450; --warn-soft: #362a12; --ref: #4d6158;
    --s0: #1f2a25; --s1: #184f95; --s2: #1c5cab; --s3: #256abf; --s4: #2a78d6; --s5: #5598e7; --s6: #9ec5f4;
    --on-dark: #0d1411; --on-light: #e2ece6;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --c1: #3987e5; --c2: #d95926; --c3: #199e70; --c4: #c98500; --c5: #d55181; --c6: #9085e9;
  --paper: #0d1411; --surface: #131c18; --surface-2: #1a2520; --line: #283731; --grid: #1f2b26;
  --ink: #e2ece6; --ink-2: #a2b4aa; --ink-3: #8a9d93; --accent: #4fc4aa; --accent-soft: #173b32;
  --warn: #e5b450; --warn-soft: #362a12; --ref: #4d6158;
  --s0: #1f2a25; --s1: #184f95; --s2: #1c5cab; --s3: #256abf; --s4: #2a78d6; --s5: #5598e7; --s6: #9ec5f4;
  --on-dark: #0d1411; --on-light: #e2ece6;
  color-scheme: dark;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--paper); color: var(--ink); font: 15px/1.5 var(--font-ui); }
.wrap { max-width: 1120px; margin: 0 auto; padding-block: 28px 48px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: flex; flex-direction: column; gap: 20px; }
header { display: flex; flex-direction: column; gap: 6px; }
.eyebrow { font: 700 11.5px/1 var(--font-ui); letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent); display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
h1 { margin: 0; font-size: clamp(24px, 3.4vw, 32px); line-height: 1.15; text-wrap: balance; }
.sub { margin: 0; color: var(--ink-2); max-width: 75ch; }
.chip { font: 700 11px/1 var(--font-ui); letter-spacing: 0.04em; padding: 5px 8px; border-radius: 999px; background: var(--warn-soft); color: var(--warn); text-transform: uppercase; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }
.tile { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; display: flex; flex-direction: column; gap: 4px; min-width: 0; }
.tile .label { font-size: 12.5px; color: var(--ink-2); }
.tile .value { font-size: 28px; font-weight: 700; line-height: 1.1; font-variant-numeric: tabular-nums; }
.tile .note { font-size: 12.5px; color: var(--ink-3); }
.tile.lead { border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }
.tile.lead .value { color: var(--accent); }
.grid2 { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap: 16px; }
.grid2.even { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
td .ci-text { display: block; font-size: 11px; color: var(--ink-3); }
td.truth { color: var(--ink-3); }
@media (max-width: 860px) { .grid2, .grid2.even { grid-template-columns: minmax(0, 1fr); } }
.panel { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 16px 18px; display: flex; flex-direction: column; gap: 10px; min-width: 0; }
.panel h2 { margin: 0; font-size: 16px; }
.panel .lede { margin: 0; color: var(--ink-2); font-size: 13.5px; max-width: 70ch; }
.scroll { overflow-x: auto; }
.chart { display: block; width: 100%; min-width: 440px; height: auto; font-family: var(--font-ui); }
.chart.head { min-width: 260px; max-width: 380px; margin: 0 auto; }
.chart .grid { stroke: var(--grid); stroke-width: 1; }
.chart .axis { stroke: var(--line); stroke-width: 1; }
.chart .ref { stroke: var(--ref); stroke-width: 1.5; }
.chart .reflabel { fill: var(--ink-3); font-size: 11px; }
.chart .tick { fill: var(--ink-3); font-size: 11px; font-variant-numeric: tabular-nums; }
.chart .axtitle { fill: var(--ink-2); font-size: 12px; }
.chart .series { fill: none; stroke: var(--accent); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.chart .ci { stroke: var(--accent); stroke-width: 2; stroke-linecap: round; opacity: 0.45; }
.chart .dot { fill: var(--accent); stroke: var(--surface); stroke-width: 2; }
.chart .hit { fill: transparent; }
.chart .mark { cursor: default; outline: none; }
.chart .mark:hover .dot, .chart .mark:focus .dot { r: 7; }
.chart .mark:focus-visible .hit { stroke: var(--accent); stroke-width: 1.5; }
.chart .line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.chart .pt { stroke: var(--surface); stroke-width: 2; }
.chart .mark:hover .pt, .chart .mark:focus .pt { r: 6; }
.chart .ref.dash { stroke-dasharray: 4 4; }
.chart .endlabel { fill: var(--ink-2); font-size: 12px; }
.chart .ci.c1, .chart .ci.c2, .chart .ci.c3, .chart .ci.c4 { opacity: 0.5; }
.line.c1, .ci.c1 { stroke: var(--c1); } .line.c2, .ci.c2 { stroke: var(--c2); } .line.c3, .ci.c3 { stroke: var(--c3); }
.line.c4, .ci.c4 { stroke: var(--c4); } .line.c5, .ci.c5 { stroke: var(--c5); } .line.c6, .ci.c6 { stroke: var(--c6); }
.pt.c1, .sw.c1 { fill: var(--c1); background: var(--c1); } .pt.c2, .sw.c2 { fill: var(--c2); background: var(--c2); }
.pt.c3, .sw.c3 { fill: var(--c3); background: var(--c3); } .pt.c4, .sw.c4 { fill: var(--c4); background: var(--c4); }
.pt.c5, .sw.c5 { fill: var(--c5); background: var(--c5); } .pt.c6, .sw.c6 { fill: var(--c6); background: var(--c6); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12.5px; color: var(--ink-2); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i.sw { display: inline-block; width: 14px; height: 4px; border-radius: 2px; }
td .sw { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; }
.chart .rowlabel { fill: var(--ink-2); font-size: 12px; font-family: var(--font-mono); }
.chart .rowvalue { fill: var(--ink); font-size: 12px; font-variant-numeric: tabular-nums; }
.chart .scalp { fill: none; stroke: var(--line); stroke-width: 1.5; }
.chart .el { stroke: var(--surface); stroke-width: 2; }
.chart .mark:hover .el, .chart .mark:focus .el { stroke: var(--ink); }
.s0 { fill: var(--s0); } .s1 { fill: var(--s1); } .s2 { fill: var(--s2); } .s3 { fill: var(--s3); }
.s4 { fill: var(--s4); } .s5 { fill: var(--s5); } .s6 { fill: var(--s6); }
.chart .el-name { font-size: 9.5px; font-weight: 700; }
.chart .el-count { font-size: 9px; font-variant-numeric: tabular-nums; }
.ink-light { fill: var(--on-light); } .ink-dark { fill: var(--on-dark); }
.ramp { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--ink-3); justify-content: center; }
.ramp span.sw { display: inline-block; width: 22px; height: 10px; }
.sw.s0 { background: var(--s0); } .sw.s1 { background: var(--s1); } .sw.s2 { background: var(--s2); } .sw.s3 { background: var(--s3); }
.sw.s4 { background: var(--s4); } .sw.s5 { background: var(--s5); } .sw.s6 { background: var(--s6); }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; font-weight: 700; color: var(--ink-3); font-size: 11.5px; letter-spacing: 0.03em; padding: 0 10px 6px 0; border-bottom: 1px solid var(--line); white-space: nowrap; }
td { padding: 6px 10px 6px 0; border-bottom: 1px solid var(--line); font-variant-numeric: tabular-nums; white-space: nowrap; }
td.num, th.num { text-align: right; }
td.mono { font-family: var(--font-mono); }
.notes { color: var(--ink-2); font-size: 13px; display: flex; flex-direction: column; gap: 6px; max-width: 85ch; }
.notes p { margin: 0; }
#tip { position: fixed; z-index: 20; pointer-events: none; max-width: 300px; background: var(--ink); color: var(--paper); font-size: 12.5px; line-height: 1.4; padding: 8px 10px; border-radius: 6px; box-shadow: 0 6px 18px rgba(0,0,0,0.18); }
@media (prefers-reduced-motion: no-preference) { .chart .dot { transition: r 0.12s ease; } }
"""

JS = """
(function () {
  var tip = document.getElementById('tip');
  function show(el, x, y) {
    tip.innerHTML = el.getAttribute('data-tip');
    tip.hidden = false;
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var left = Math.min(window.innerWidth - w - 8, x + 14), top = y - h - 12;
    if (top < 8) top = y + 16;
    tip.style.left = Math.max(8, left) + 'px'; tip.style.top = top + 'px';
  }
  document.querySelectorAll('[data-tip]').forEach(function (el) {
    el.addEventListener('pointermove', function (e) { show(el, e.clientX, e.clientY); });
    el.addEventListener('pointerleave', function () { tip.hidden = true; });
    el.addEventListener('focus', function () { var r = el.getBoundingClientRect(); show(el, r.left + r.width / 2, r.top); });
    el.addEventListener('blur', function () { tip.hidden = true; });
  });
})();
"""


def render(a, title, subtitle, simulated=None, generated=None, status_html="", source="merged_labels.csv"):
    generated = generated or dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lo, hi = a["prev_ci"]
    t = a["threshold"]
    truth = (simulated or {}).get("scorings", {})
    sc = a["scorings"]
    for i, x in enumerate(sc):
        x["slot"] = i % 6 + 1

    # ---- tiles
    per_hour = a["prev"] * 3600 if not math.isnan(a["prev"]) else float("nan")
    tiles = []
    prev_note = (f"95% CI {pct(lo)} to {pct(hi)}, resampling patients" if not math.isnan(lo)
                 else "fewer than 3 patients, so no interval")
    if simulated:
        prev_note += f". Simulated truth: {pct(simulated['true_prev'])}"
    unit = "10 s windows" if a.get("clip_level") else "Recorded seconds"
    tiles.append(("lead", f"{unit} with an IED" + (" (weighted)" if a["weighted"] else ""), pct(a["prev"], 2), prev_note))
    if not a.get("clip_level"):
        tiles.append(("", "Seconds per hour with an IED", "–" if math.isnan(per_hour) else f"{per_hour:.0f}",
                      "the same share, read per hour of recording"))
    tiles.append(("", "Clips labeled", f"{a['n_clips']:,}",
                  f"{plural(len(a['raters']), 'rater')}, {plural(a['n_patients'], 'patient')}; {a['n_shared']:,} labeled by more than one rater"
                  + (f", {a['n_ties']} with no majority" if a["n_ties"] else "")))
    ag = a["agree_all"]
    if ag["n_clips"]:
        ci = lambda v: "" if math.isnan(v[0]) else f"95% CI {num(v[0])} to {num(v[1])}; "
        tiles.append(("", "Gwet's AC1, all raters", num(ag["ac1"]),
                      f"{ci(ag['ac1_ci'])}stays steady when most answers are No"))
        tiles.append(("", "Fleiss' kappa, all raters", num(ag["fleiss"]),
                      f"{ci(ag['fleiss_ci'])}{ag['n_clips']} clips two or more raters answered"))
    tiles.append(("", "Mean Cohen's kappa", num(a["mean_kappa"]), "over rater pairs, on the clips each pair shares"))
    if a["n_repeats"]:
        tiles.append(("", "Test-retest agreement", pct(a["mean_retest"], 0),
                      f"same rater, same clip shown twice ({a['n_repeats']} repeats)"))
    tiles_html = "".join(
        f'<div class="tile {cls}"><span class="label">{esc(lab)}</span><span class="value">{esc(val)}</span>'
        f'<span class="note">{esc(note)}</span></div>' for cls, lab, val, note in tiles)

    # ---- each scoring against the experts
    def cell(x, k, fmt=pct):
        v = x["at"][k]
        c = x["ci"].get(k)
        sub = f"<span class='ci-text'>{fmt(c[0])} to {fmt(c[1])}</span>" if c and not math.isnan(c[0]) else ""
        return f"<td class='num'>{fmt(v)}{sub}</td>"
    perh = lambda v: "–" if math.isnan(v) else f"{v:,.0f}"
    if sc:
        rows = []
        for x in sc:
            tr = truth.get(x["col"])
            true_cells = (f"<td class='num truth'>{pct(tr['ppv'])}</td><td class='num truth'>{pct(tr['sens'])}</td>"
                          if tr else "")
            best = max((p for p in x["curve"] if not math.isnan(p[1])), key=lambda p: p[1], default=None)
            rows.append(f"<tr><td><i class='sw c{x['slot']}'></i>{esc(x['name'])}</td>{cell(x, 'ppv')}{cell(x, 'sens')}"
                        f"{cell(x, 'f1', num)}{cell(x, 'flag_share', pct) if a.get('clip_level') else cell(x, 'per_hour', perh)}"
                        f"<td class='num'>{'–' if best is None else format(best[0], 'g') + ' (' + num(best[1]) + ')'}</td>"
                        f"{true_cells}</tr>")
        n_cols = 6 + (2 if truth else 0)
        rows += [f"<tr class='waiting'><td><i class='sw'></i>{esc(n)}</td><td colspan='{n_cols}'>not in yet; it is scored "
                 "on the same labels once its scores are added</td></tr>" for n in a.get("waiting", [])]
        true_head = "<th class='num'>True precision</th><th class='num'>True sensitivity</th>" if truth else ""
        det_table = ("<div class='scroll'><table><tr><th>Scoring</th><th class='num'>Precision</th><th class='num'>Sensitivity</th>"
                     "<th class='num'>F1</th><th class='num'>" + ("Windows flagged" if a.get("clip_level") else "Flagged s per hour")
                     + "</th><th class='num'>Best F1 at</th>"
                     f"{true_head}</tr>{''.join(rows)}</table></div>")
        f1_series = [{"name": x["name"], "slot": x["slot"],
                      "pts": [(th, f, f"<b>{esc(x['name'])}</b> at {th:g}: F1 {num(f)}", float("nan"), float("nan"))
                              for th, f in x["curve"]]} for x in sc]
        cal_series = [{"name": x["name"], "slot": x["slot"],
                       "pts": [(c["p"], c["rate"],
                                f"<b>{esc(x['name'])}</b> {c['lo_edge']:.1f} to {c['lo_edge'] + 0.1:.1f} (mean {c['p']:.2f})<br>"
                                f"{c['k']} of {c['n']} clips called IED; weighted <b>{pct(c['rate'])}</b><br>"
                                f"95% CI {pct(c['lo'])} to {pct(c['hi'])}", c["lo"], c["hi"]) for c in x["calibration"]]}
                      for x in sc]
        many = len(sc) > 1
        if a.get("clip_level"):
            lede = (f"Raters answer for the whole 10 s clip, and each scoring is judged against the same consensus. A scoring "
                    f"finds a clip when any second the rater sees scores {t:g} or more. Each clip is weighted by the 10 s "
                    "windows it stands for, so the numbers describe every window of the recordings. Intervals resample patients.")
        else:
            lede = (f"Every second in the review is judged once by the raters, and each scoring is scored against the same "
                    f"consensus. Precision, sensitivity and F1 count a second as flagged when its score is {t:g} or more, and "
                    "weight each clip by the recorded seconds it stands for, so they describe all recorded seconds. "
                    "Intervals resample patients.")
        if a.get("n_not_blind"):
            lede += (f" {plural(a['n_not_blind'], 'answer')} given with the admin or superadmin view on "
                     "(SubQ or detector scores shown) are left out.")
        det_panel = f"""<section class="panel">
      <h2>How each detector does against the experts</h2>
      <p class="lede">{lede}</p>
      {det_table}
    </section>
    <div class="grid2 even">
      <section class="panel">
        <h2>F1 across thresholds</h2>
        <p class="lede">Weighted F1 at each threshold, pooled over patients. This curve describes the labels; the thesis threshold itself is fit by patient-level cross-validation.</p>
        {legend(f1_series) if many else ''}
        <div class="scroll">{lines_svg(f1_series, 'Threshold', 'F1', 'Weighted F1 by threshold, one line per scoring', vline=t)}</div>
      </section>
      <section class="panel">
        <h2>Calibration</h2>
        <p class="lede">Each point is a tenth of that scoring's range: the weighted share of its clips the raters' majority called an IED{'' if many else ', with a 95% interval'}. Points below the line mean the scoring is more confident than the experts.</p>
        {legend(cal_series) if many else ''}
        <div class="scroll">{lines_svg(cal_series, 'Score (mean of each tenth)', 'Clips experts call an IED', 'Expert Yes rate against score, one line per scoring', diagonal=True, ci=not many)}</div>
      </section>
    </div>"""
    else:
        det_panel = """<section class="panel">
      <h2>No detector score to calibrate</h2>
      <p class="lede">These clips were drawn without a detector score (a simple random sample within each patient), so there is no calibration curve. The weighted share above is the estimate for all recorded seconds.</p>
    </section>"""

    # ---- by sampling bin
    cal_rows = "".join(
        f"<tr><td class='mono'>{esc(c['bin'])}</td><td class='num'>{'–' if math.isnan(c['p']) else format(c['p'], '.2f')}</td><td class='num'>{c['n']}</td>"
        f"<td class='num'>{c['k']}</td><td class='num'>{pct(c['rate'])}</td><td class='num'>{pct(c['lo'], 0)} to {pct(c['hi'], 0)}</td>"
        + "".join(f"<td class='num'>{pct(c['per_rater'][r][0] / c['per_rater'][r][1], 0) if c['per_rater'][r][1] else '–'}</td>" for r in a["raters"])
        + f"<td class='num'>{c['seconds']:,.0f}</td></tr>" for c in a["calibration"])
    cal_head = ("<tr><th>Bin</th><th class='num'>Mean score</th><th class='num'>Clips</th><th class='num'>IED</th>"
                "<th class='num'>Consensus</th><th class='num'>95% CI</th>"
                + "".join(f"<th class='num'>{esc(r)}</th>" for r in a["raters"]) + "<th class='num'>Seconds represented</th></tr>")
    bin_note = ("Bins are the sampling strata: each clip's highest score over every joined scoring." if a["multi"]
                else "Bins are the sampling strata, by the score that drove sampling.")

    # ---- raters
    uns_h = "<th class='num'>Unsure</th>" if a["any_unsure"] else ""
    ret_h = "<th class='num'>Retest</th>" if a["n_repeats"] else ""
    r_rows = "".join(
        f"<tr><td class='mono'>{esc(r['rater'])}</td><td class='num'>{r['answered']:,}</td>"
        f"<td class='num'>{pct(r['yes_rate'], 1)}</td>"
        + (f"<td class='num'>{r['unsure']}</td>" if a["any_unsure"] else "")
        + f"<td class='num'>{pct(r['vs_agree'], 0) if r['vs_n'] else '–'}</td><td class='num'>{num(r['vs_kappa']) if r['vs_n'] else '–'}</td>"
        f"<td class='num'>{num(r['median_s'], 0)} s</td>"
        + (f"<td class='num'>{pct(r['retest_agree'], 0) if r['retest_n'] else '–'}</td>" if a["n_repeats"] else "")
        + "</tr>" for r in a["rater_stats"])
    k_rows = "".join(
        f"<tr><td class='mono'>{esc(p['a'])} and {esc(p['b'])}</td><td class='num'>{p['n']}</td>"
        f"<td class='num'>{pct(p['agree'], 0)}</td><td class='num'>{num(p['kappa'])}</td></tr>" for p in a["pair_kappa"]
    ) or "<tr><td colspan='4' class='truth'>No clip has answers from two raters yet.</td></tr>"

    field_panel = ""
    if a["n_located"]:
        ramp = ('<div class="ramp"><span>0</span>' + "".join(f'<span class="sw s{i}"></span>' for i in range(7))
                + f'<span>{max(a["field"].values())} marks</span></div>')
        field_panel = f"""<section class="panel">
      <h2>Where the field maximum sits</h2>
      <p class="lede">Electrodes marked as the maximum on Yes answers, all raters. {a['n_located']} of {a['n_yes']} Yes answers marked a maximum; {a['n_generalized']} were marked generalized.</p>
      <div class="scroll">{headmap_svg(a['field'], a['n_located'])}</div>
      {ramp}
    </section>"""

    truth_note = ""
    if a.get("truth"):
        tt = a["truth"]
        truth_note = (f"<p>Against the planted truth, the expert consensus caught {pct(tt['sens'], 0)} of true IEDs and "
                      f"correctly cleared {pct(tt['spec'], 1)} of clips without one ({tt['n']} clips). The true precision and "
                      "sensitivity columns are each scoring against the planted truth over every second the sampler could draw. "
                      "They differ from the estimates for two reasons: the reference is the raters' consensus, which misses "
                      "some subtle IEDs and calls some sharp artifacts, and the few clips in the lowest bins each stand for "
                      "many seconds, which is where most of the uncertainty in sensitivity comes from.</p>")
    sim_chip = '<span class="chip">Simulated data</span>' if simulated else ""
    weighted_note = ("Each clip counts for the number of recorded seconds it stands for (its sampling weight), so the share "
                     "describes all recorded seconds, not just the clips." if a["weighted"] else
                     "This label set has no sampling_weight column, so every clip counts once and the share describes the clips only.")
    uns_note = "; Unsure answers don't vote" if a["any_unsure"] else ""
    rep_note = " Repeated clips count only toward test-retest." if a["n_repeats"] else ""

    body = f"""
<div class="wrap">
  <header>
    <div class="eyebrow">IED Clip Review · results {sim_chip}</div>
    <h1>{esc(title)}</h1>
    <p class="sub">{esc(subtitle)} Generated {esc(generated)}.</p>
  </header>
  {status_html}
  <section class="tiles" aria-label="Headline numbers">{tiles_html}</section>
  {det_panel}
  <div class="grid2 even">
    <section class="panel">
      <h2>Raters</h2>
      <p class="lede">"With the others" compares each rater with the majority of the other raters, on clips three or more raters answered.</p>
      <div class="scroll"><table><tr><th>Rater</th><th class="num">Answered</th><th class="num">Yes</th>{uns_h}<th class="num">With the others</th><th class="num">κ with the others</th><th class="num">Per clip</th>{ret_h}</tr>{r_rows}</table></div>
    </section>
    <section class="panel">
      <h2>Pairs</h2>
      <p class="lede">Cohen's kappa for each pair of raters, on the clips both labeled.</p>
      <div class="scroll"><table><tr><th>Raters</th><th class="num">Shared clips</th><th class="num">Agree</th><th class="num">κ</th></tr>{k_rows}</table></div>
    </section>
  </div>
  <section class="panel">
    <h2>IED share by patient</h2>
    <p class="lede">Weighted share of each patient's recorded seconds with an IED, with a 95% interval from resampling that patient's clips.</p>
    <div class="scroll">{patients_svg(a['patients'])}</div>
  </section>
  {field_panel}
  <section class="panel">
    <h2>By sampling bin</h2>
    <p class="lede">{bin_note}</p>
    <div class="scroll"><table>{cal_head}{cal_rows}</table></div>
  </section>
  <section class="notes">
    <p>Consensus means more raters said Yes than No{uns_note}; clips with a tie are left out ({a['n_ties']} here), and a clip one rater labeled takes that rater's answer. {esc(weighted_note)}{rep_note}</p>
    {truth_note}
    <p>Made by report_labels.py from {esc(source)}. The page holds counts and rates only, with no EEG, file paths or times.</p>
  </section>
</div>
<div id="tip" role="tooltip" hidden></div>
"""
    head = (f'<title>{esc(title)}</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=JetBrains+Mono:wght@400;600&display=swap">\n'
            f"<style>{CSS}{RESULTS_CSS}</style>\n")
    return head, body + f"<script>{JS}</script>\n"


def write(path, head, body, fragment):
    text = head + body if fragment else document(head, body)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------- the study's live results page

def plural(n, word):
    return f"{n:,} {word}" + ("" if n == 1 else "s")


def pseudonyms(patients):
    """P01, P02, ... in the order of the study IDs, so each patient keeps its name from one run to the next."""
    ids = sorted({str(p) for p in patients if str(p)})
    width = max(2, len(str(len(ids))))
    return {p: f"P{i + 1:0{width}d}" for i, p in enumerate(ids)}


def add_scores(key, extra):
    """Scorings that arrive after the clips were drawn: {name: seconds table with source_path, start_s, prob}.
    Each clip takes the score of its own second. Its sampling weight does not change, so the scoring's
    estimates stay unbiased; they are only less precise than for a scoring that helped set the bins."""
    key = key.copy()
    at = key[["source_path"]].assign(start_s=pd.to_numeric(key["start_s"], errors="coerce").round(3))
    for name, table in extra.items():
        t = table[["source_path", "start_s", "prob"]].copy()
        t["start_s"] = pd.to_numeric(t["start_s"], errors="coerce").round(3)
        t = t.drop_duplicates(["source_path", "start_s"])
        key[f"{name}_prob"] = at.merge(t, on=["source_path", "start_s"], how="left")["prob"].to_numpy()
    return key


def data_status(key, merged, assignment, batch_sizes, expect):
    """What has come in so far: each rater's answers against the clips they were given, and which of the
    expected scorings the clips carry."""
    answered = merged[merged["decision"].isin(DECISIONS)] if len(merged) else merged
    n_all = int(key["clip_id"].nunique())
    raters = sorted(set((assignment or {}).get("by_rater", {})) | set(answered["rater"] if len(answered) else []))
    rows = []
    for r in raters:
        if assignment:
            # batches handed out to this rater, plus any extra batch they answered (the owner can give out more)
            bids = set(assignment.get("shared", [])) | set(assignment.get("by_rater", {}).get(r, []))
            if len(answered) and "batch_id" in answered.columns:
                bids |= set(answered.loc[answered["rater"] == r, "batch_id"])
            given = sum(batch_sizes.get(b, 0) for b in bids)
        else:
            given = n_all
        done = int((answered["rater"] == r).sum()) if len(answered) else 0
        rows.append({"rater": r, "given": int(given), "done": done})
    present = [c[:-5] for c in key.columns if c.endswith("_prob") and c != "strata_prob"
               and pd.to_numeric(key[c], errors="coerce").notna().any()]
    expect = [e for e in expect if e]
    return {"raters": rows, "n_clips": n_all,
            "n_patients": int(key["patient"].nunique()) if "patient" in key.columns else 0,
            "clips_answered": int(answered["clip_id"].nunique()) if len(answered) else 0,
            "scorings": [(display_name(e), e in present) for e in expect] +
                        [(display_name(p), True) for p in present if p not in expect],
            "waiting": [display_name(e) for e in expect if e not in present]}


def status_panel(st):
    rows = []
    for r in st["raters"]:
        share = r["done"] / r["given"] if r["given"] else 0.0
        rows.append(f"<tr><td class='mono'>{esc(r['rater'])}</td><td class='num'>{r['done']:,}</td>"
                    f"<td class='num'>{r['given']:,}</td><td class='bar'><span class='track'><span class='fill' "
                    f"style='width:{100 * min(1.0, share):.1f}%'></span></span></td><td class='num'>{pct(share, 0)}</td></tr>")
    table = ("<div class='scroll'><table><tr><th>Rater</th><th class='num'>Answered</th><th class='num'>Given</th>"
             f"<th>Progress</th><th class='num'></th></tr>{''.join(rows)}</table></div>" if rows else
             "<p class='lede'>No rater links yet.</p>")
    chips = "".join(f"<li><span class='state {'in' if ok else 'out'}'>{'In' if ok else 'Not in yet'}</span>{esc(n)}</li>"
                    for n, ok in st["scorings"]) or "<li>No detector scores in these clips.</li>"
    return f"""<section class="panel">
      <h2>What's in so far</h2>
      <p class="lede">{st['clips_answered']:,} of {st['n_clips']:,} clips have at least one answer, from {st['n_patients']} patients. Every number below uses the answers pulled so far and settles as the rest come in.</p>
      <div class="grid2 even">
        <div class="stack">{table}</div>
        <div class="stack"><h3>Detector scorings</h3><ul class="states">{chips}</ul></div>
      </div>
    </section>"""


RESULTS_CSS = """
.stack { display: flex; flex-direction: column; gap: 8px; min-width: 0; }
.panel h3 { margin: 0; font-size: 13px; color: var(--ink-2); }
td.bar { width: 40%; }
.track { display: block; height: 8px; border-radius: 4px; background: var(--surface-2); overflow: hidden; }
.fill { display: block; height: 100%; border-radius: 4px; background: var(--accent); }
.states { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 6px; font-size: 13.5px; }
.states li { display: flex; align-items: center; gap: 10px; }
.state { font: 700 11px/1 var(--font-ui); letter-spacing: 0.04em; text-transform: uppercase; padding: 5px 8px; border-radius: 999px; min-width: 84px; text-align: center; }
.state.in { background: var(--accent-soft); color: var(--accent); }
.state.out { background: var(--surface-2); color: var(--ink-3); }
.waiting td { color: var(--ink-3); }
.soon { margin: 0; padding-left: 18px; color: var(--ink-2); font-size: 13.5px; display: flex; flex-direction: column; gap: 4px; }
"""


def results_page(key, label_paths, assignment=None, batch_sizes=None, expect=(), extra=None, threshold=0.5,
                 title="IED Review Results"):
    """The study's results page from what exists so far: key.csv (as a DataFrame), the label CSVs pulled from the
    site (any number, including none), and scorings added after the draw. Patients appear as P01, P02, ...;
    returns (html, {study ID: pseudonym}). The page holds counts and rates only."""
    sys.path.insert(0, HERE)
    import merge_labels
    key = key.copy()
    if "part" in key.columns:
        # a staged study (kit 1.8): the page scores the scalp clips; training examples and the UNEEG part are left
        # out here (merge_labels.py summarizes each part on its own)
        key = key[key["part"].str.strip().str.lower().isin(["", "scalp"])].copy()
    if extra:
        key = add_scores(key, extra)
    names = pseudonyms(key["patient"]) if "patient" in key.columns else {}
    if names:
        key["patient"] = key["patient"].astype(str).map(names)
    lab = merge_labels.load_labels(label_paths) if label_paths else pd.DataFrame(columns=["clip_id", "rater", "decision"])
    merged = merge_labels.merge(key, lab) if len(lab) else lab
    st = data_status(key, merged, assignment, batch_sizes or {}, list(expect))
    status = status_panel(st)
    if len(merged) and merged["decision"].isin(["yes", "no"]).any():
        a = analyze(merged.fillna(""), threshold)
        a["waiting"] = st["waiting"]
        n_ans = int(merged["decision"].isin(DECISIONS).sum())
        head, body = render(a, title, f"{plural(n_ans, 'answer')} from {plural(len(a['raters']), 'rater')} so far.",
                            generated="{{updated}}", status_html=status,
                            source="the label files pulled so far, with patients shown as P01, P02 and so on")
    else:
        head, body = render_waiting(title, status)
    return document(head, body), names


def render_waiting(title, status):
    body = f"""
<div class="wrap">
  <header>
    <div class="eyebrow">IED Clip Review · results</div>
    <h1>{esc(title)}</h1>
    <p class="sub">No answers yet. Updated {{{{updated}}}}.</p>
  </header>
  {status}
  <section class="panel">
    <h2>What this page will show</h2>
    <ul class="soon">
      <li>The weighted share of recorded seconds with an IED, with an interval from resampling patients.</li>
      <li>Each detector scoring against the raters' consensus: precision, sensitivity and F1 at the threshold, F1 across thresholds, and calibration.</li>
      <li>Agreement: Gwet's AC1 and Fleiss' kappa over all raters, Cohen's kappa per pair, and each rater against the others.</li>
      <li>The IED share per patient, and the answers by sampling bin.</li>
    </ul>
  </section>
</div>
"""
    head = (f'<title>{esc(title)}</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=JetBrains+Mono:wght@400;600&display=swap">\n'
            f"<style>{CSS}{RESULTS_CSS}</style>\n")
    return head, body


def document(head, body):
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + head + "</head>\n<body>\n" + body + "</body>\n</html>\n")


# ---------------------------------------------------------------- simulated study

# four scorings as kit 1.6 joins them: (name, intercept, IED effect, shared-artifact effect, noise) on the logit scale
SIM_SCORINGS = [("morgoth", -4.4, 5.4, 0.9, 0.9), ("cyclops", -3.6, 5.3, 1.3, 1.0),
                ("cyclops_sd", -4.6, 5.0, 0.8, 0.9), ("spikenet1", -3.1, 4.4, 1.5, 1.1)]
# raters: (sensitivity, specificity, median seconds per clip)
# on clean background experts almost never say Yes; their mistakes sit on subtle IEDs and sharp artifacts
SIM_RATERS = {"R1": (0.92, 0.998, 7.0), "R2": (0.85, 0.997, 5.5), "R3": (0.95, 0.999, 9.0), "R4": (0.88, 0.996, 6.0)}


def simulate(out_dir, seed=2026, n=1000, shared=200, threshold=0.5):
    """A made-up study run the way kit 1.6 runs one: 12 patients, four detector scorings of the same seconds
    joined and drawn with select_clips.py (bins by the highest score), `shared` clips that all four raters label
    and the rest split evenly, Yes or No only. Subtle IEDs and sharp artifacts are hard for every rater alike,
    so raters disagree on the same clips. Returns (merged DataFrame, meta with the true values)."""
    sys.path.insert(0, HERE)
    import contextlib
    import io
    import select_clips
    rng = np.random.default_rng(seed)
    frames = []
    for k in range(12):
        pid = f"SIM-{k + 1:02d}"
        n_s = int(rng.integers(8000, 26000))
        ied = rng.random(n_s) < rng.uniform(0.002, 0.035)
        frames.append(pd.DataFrame({"patient": pid, "source_path": f"/sim/{pid}.edf", "start_s": np.arange(n_s),
                                    "ied": ied, "sharp": rng.normal(size=n_s), "subtle": ied & (rng.random(n_s) < 0.2)}))
    secs = pd.concat(frames, ignore_index=True)
    tmp = tempfile.mkdtemp(dir=out_dir)
    paths = []
    for name, a0, b, c, sd in SIM_SCORINGS:
        signal = b * (secs["ied"] & ~secs["subtle"]) + 0.45 * b * secs["subtle"]
        logit = a0 + signal + c * secs["sharp"] + rng.normal(0, sd, len(secs))
        secs[f"{name}_p"] = np.clip(1 / (1 + np.exp(-logit)), 0, 0.999).round(4)
        path = os.path.join(tmp, f"{name}.csv")
        secs[["patient", "source_path", "start_s"]].assign(prob=secs[f"{name}_p"], detector=name).to_csv(path, index=False)
        paths.append(f"{name}={path}")
    sec_path, clip_path = os.path.join(tmp, "seconds.csv"), os.path.join(tmp, "clips.csv")
    with contextlib.redirect_stdout(io.StringIO()):
        select_clips.main(["join", "--out", sec_path] + paths)
        select_clips.main(["sample", sec_path, "--prob-cols", ",".join(f"{x[0]}_prob" for x in SIM_SCORINGS),
                           "--out", clip_path, "--n", str(n), "--min-gap-s", "12", "--seed", str(seed)])
    clips = pd.read_csv(clip_path)
    facts = secs.set_index(["source_path", "start_s"])
    look = facts.loc[list(zip(clips["source_path"], clips["start_s"]))]
    clips["truth"] = np.where(look["ied"].to_numpy(), "spike", "none")
    clips["_subtle"] = look["subtle"].to_numpy()
    clips["_mimic"] = (~look["ied"].to_numpy()) & (look["sharp"].to_numpy() > 2.3)

    # the truth over every candidate second the sampler could draw
    span = secs.groupby("source_path")["start_s"].agg(["min", "max"])
    c = secs["start_s"] + 0.5
    ok = (c - 6 >= secs["source_path"].map(span["min"])) & (c + 6 <= secs["source_path"].map(span["max"]) + 1)
    cand = secs[ok]
    y = cand["ied"].to_numpy(bool)
    truth = {}
    for name, *_ in SIM_SCORINGS:
        flag = cand[f"{name}_p"].to_numpy() >= threshold
        truth[f"{name}_prob"] = {"ppv": float((flag & y).sum() / max(1, flag.sum())), "sens": float((flag & y).sum() / max(1, y.sum()))}

    clips["clip_id"] = [f"c{int(x):07x}" for x in rng.choice(16 ** 7, len(clips), replace=False)]
    order = rng.permutation(len(clips))
    raters = list(SIM_RATERS)
    who = {}
    for pos, i in enumerate(order):
        who[i] = raters if pos < shared else [raters[(pos - shared) % len(raters)]]
    rows = []
    for i, cl in clips.iterrows():
        pos = int(np.flatnonzero(order == i)[0])
        batch = f"s{pos // 25 + 1:02d}" if pos < shared else f"b{(pos - shared) // 25 + 1:02d}"
        for r in who[i]:
            sens, spec, med = SIM_RATERS[r]
            if cl["_subtle"]:
                d = "yes" if rng.random() < 0.55 else "no"
            elif cl["_mimic"]:
                d = "yes" if rng.random() < 0.18 else "no"
            elif cl["truth"] == "spike":
                d = "yes" if rng.random() < sens else "no"
            else:
                d = "no" if rng.random() < spec else "yes"
            rows.append(_sim_row(rng, cl, batch, r, d, med))
    merged = pd.DataFrame(rows)
    return merged, {"true_prev": float(y.mean()), "n_seconds": int(len(cand)), "scorings": truth}


def _sim_row(rng, cl, batch, rater, d, med):
    secs = float(rng.lognormal(math.log(med * (1.3 if d == "yes" else 1.0)), 0.45))
    row = {"batch_id": batch, "clip_id": cl["clip_id"], "rater": rater, "decision": d, "max_electrodes": "",
           "generalized": 0, "artifact": 0, "seconds_viewed": round(secs), "repeat_of": "", "patient": cl["patient"],
           "strata_prob": cl["strata_prob"], "bin": cl["bin"], "sampling_weight": cl["sampling_weight"],
           "cell_N": cl["cell_N"], "cell_n": cl["cell_n"], "truth": cl["truth"]}
    for name, *_ in SIM_SCORINGS:
        row[f"{name}_prob"] = cl[f"{name}_prob"]
    return row


# ---------------------------------------------------------------- main

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "demo":
        ap = argparse.ArgumentParser(description="Simulate a labeled study and report it")
        ap.add_argument("demo")
        ap.add_argument("--out", required=True)
        ap.add_argument("--n", type=int, default=1000)
        ap.add_argument("--seed", type=int, default=2026)
        ap.add_argument("--fragment", action="store_true", help=argparse.SUPPRESS)
        args = ap.parse_args(argv)
        out_dir = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(out_dir, exist_ok=True)
        merged, meta = simulate(out_dir, args.seed, args.n)
        merged_path = os.path.splitext(args.out)[0] + "_merged.csv"
        merged.to_csv(merged_path, index=False)
        a = analyze(merged)
        head, body = render(a, "Simulated Fellows Study",
                            f"A made-up study of {a['n_clips']:,} clips from 12 simulated patients, run the way kit 1.6 runs one: "
                            f"four detector scorings joined and drawn with select_clips.py, {a['n_shared']} clips that all four "
                            "simulated raters label and the rest split between them, Yes or No only. Every number here comes "
                            "from simulated labels with a known truth.",
                            simulated=meta)
        write(args.out, head, body, args.fragment)
        print(f"wrote {args.out} and {merged_path}")
        print(f"weighted IED share {pct(a['prev'], 2)} (95% CI {pct(a['prev_ci'][0], 2)} to {pct(a['prev_ci'][1], 2)}); "
              f"simulated truth {pct(meta['true_prev'], 2)}")
        return

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("merged", help="merged_labels.csv from merge_labels.py")
    ap.add_argument("--out", default="label_report.html")
    ap.add_argument("--title", default="Expert label results")
    ap.add_argument("--threshold", type=float, default=0.5, help="score at which a second counts as flagged (default 0.5)")
    ap.add_argument("--fragment", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    merged = pd.read_csv(args.merged, dtype=str, keep_default_na=False)
    for col in ("clip_id", "rater", "decision"):
        if col not in merged.columns:
            sys.exit(f"{args.merged} has no '{col}' column; pass the file merge_labels.py writes")
    a = analyze(merged, args.threshold)
    head, body = render(a, args.title, f"{a['n_rows']:,} label rows from {os.path.basename(args.merged)}.")
    write(args.out, head, body, args.fragment)
    print(f"wrote {args.out}: {a['n_clips']} clips, {len(a['raters'])} raters, weighted IED share {pct(a['prev'], 2)}")


if __name__ == "__main__":
    main()
