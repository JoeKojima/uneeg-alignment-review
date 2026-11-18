"""Tests for the kit 1.7 expert-label set: several draws merged (sample --draw / --random), the SubQ-overlap pool and
artifact flags in join, the owner-only admin view (select_clips.py admin, serve_review.py and make_site.py --admin)
and clip-level scoring in the report. Run: python -m pytest -q tests/test_phase5.py. Synthetic data only."""
import csv
import gzip
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT)
sys.path.insert(0, os.path.join(KIT, "hosted"))
sys.path.insert(0, os.path.join(KIT, "tests"))
import select_clips  # noqa: E402
import serve_review  # noqa: E402
import check_upload  # noqa: E402

LABS = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz",
        "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2", "EKG"]


def two_detectors(tmp, n_pat=6, seed=4):
    """Per-second SpikeNet2- and MORGOTH-like scores that agree only loosely, and a planted truth."""
    rng = np.random.default_rng(seed)
    frames = []
    for p in range(n_pat):
        n = int(rng.integers(2500, 6000))
        truth = rng.random(n) < (0.03 if p < 2 else 0.005)
        a = np.clip(rng.beta(0.7, 7, n) + truth * rng.uniform(0.3, 0.7, n), 0, 0.999)
        b = np.clip(rng.beta(0.7, 7, n) + truth * rng.uniform(0.0, 0.8, n), 0, 0.999)
        frames.append(pd.DataFrame({"patient": f"P{p}", "source_path": f"/x/P{p}.edf", "start_s": np.arange(n),
                                    "spikenet2_prob": a.round(4), "morgoth_prob": b.round(4), "truth": truth.astype(int)}))
    df = pd.concat(frames, ignore_index=True)
    path = os.path.join(tmp, "joined.csv")
    df.to_csv(path, index=False)
    return path, df


def test_draws_merge_designs_and_stay_unbiased():
    with tempfile.TemporaryDirectory() as tmp:
        path, df = two_detectors(tmp)
        out = os.path.join(tmp, "clips.csv")
        select_clips.main(["sample", path, "--out", out, "--draw", "spikenet2_prob=300", "--draw", "morgoth_prob=300",
                           "--random", "60"])
        c = pd.read_csv(out, keep_default_na=False)
        assert not c.duplicated(["source_path", "start_s"]).any()
        n_each = {d: int(c["draws"].str.split(";").map(lambda x: d in x).sum()) for d in ("spikenet2", "morgoth", "random")}
        # a top bin with too few patients can come up a clip short (the patient cap), never over
        assert all(295 <= n_each[d] <= 300 for d in ("spikenet2", "morgoth")) and n_each["random"] == 60, n_each
        assert len(c) == sum(n_each.values()) - int((c["draws"].str.count(";")).sum())  # shown once
        # the design is saved: every draw's cells, whose clips add to the draw and whose N to the candidates
        des = pd.read_csv(os.path.join(tmp, "clips_design.csv"))
        assert set(des["draw"]) == {"spikenet2", "morgoth", "random"}
        for d, g in des.groupby("draw"):
            assert int(g["n"].sum()) == n_each[d] and int(g["N"].sum()) == len(df) - 72, d
            full = g[g["p_patient"] == 1]
            assert np.allclose(full["p_second"], full["n"] / full["N"])
            assert np.isclose((g["N"] * g["p_second"]).sum(), n_each[d], rtol=0.02)  # expected clips = clips
        # each draw's own weights add back to its candidates in every bin of its own score
        cand = df[(df["start_s"] >= 5) & (df["start_s"] <= df.groupby("source_path")["start_s"].transform("max") - 5)]
        for d in ("spikenet2", "morgoth"):
            mine = c[c[f"w_{d}"] != ""]
            bins = mine["bin"].where(mine["draws"].str.startswith(d), None)
            w = pd.to_numeric(mine[f"w_{d}"])
            assert abs(w.sum() - len(cand)) < 1e-3 * len(cand), d
            assert bins.notna().any()
        assert abs(pd.to_numeric(c[c["w_random"] != ""]["w_random"]).sum() - len(cand)) < 1e-3 * len(cand)
        # the random draw is spread evenly over patients
        per = c[c["draws"].str.contains("random")]["patient"].value_counts()
        assert per.max() - per.min() <= 1
        # merged weights stand for every candidate (close, not exact) and give an unbiased share
        w = pd.to_numeric(c["sampling_weight"])
        assert abs(w.sum() / len(cand) - 1) < 0.1
        truth = pd.to_numeric(c["truth"])  # extra input columns ride along to clips.csv
        est, true = float((w * truth).sum() / w.sum()), float(cand["truth"].mean())
        assert abs(est - true) < 0.35 * true, (est, true)
        # clip-level maxima: the highest score over the 10 s shown, never below the drawn second's own
        for d in ("spikenet2", "morgoth"):
            assert (pd.to_numeric(c[f"{d}_clipmax"]) >= pd.to_numeric(c[f"{d}_prob"]) - 1e-9).all()
        r = c.iloc[0]
        s = df[(df["source_path"] == r["source_path"]) & (df["start_s"] + 0.5 >= float(r["center_s"]) - 5)
               & (df["start_s"] + 0.5 < float(r["center_s"]) + 5)]
        assert len(s) == 10 and abs(s["morgoth_prob"].max() - float(r["morgoth_clipmax"])) < 1e-4


def test_merged_estimate_is_unbiased_over_repeated_draws():
    with tempfile.TemporaryDirectory() as tmp:
        path, df = two_detectors(tmp, seed=9)
        cand = df[(df["start_s"] >= 5) & (df["start_s"] <= df.groupby("source_path")["start_s"].transform("max") - 5)]
        true = float(cand["truth"].mean())
        ests = []
        for seed in range(12):
            out = os.path.join(tmp, f"c{seed}.csv")
            select_clips.main(["sample", path, "--out", out, "--draw", "spikenet2_prob=150", "--draw",
                               "morgoth_prob=150", "--random", "30", "--seed", str(seed)])
            c = pd.read_csv(out)
            ests.append(float((c["sampling_weight"] * c["truth"]).sum() / c["sampling_weight"].sum()))
        assert abs(np.mean(ests) - true) < 0.15 * true, (np.mean(ests), true)


def bursty(tmp, n_pat=5, seed=4):
    """Like two_detectors, but the planted spikes come in bursts of 2 to 6 s, so nearby seconds score high together."""
    rng = np.random.default_rng(seed)
    frames = []
    for p in range(n_pat):
        n = int(rng.integers(2500, 4500))
        truth = np.zeros(n, bool)
        for s in rng.choice(n - 8, size=int(n * (0.006 if p < 2 else 0.0015)), replace=False):
            truth[s:s + int(rng.integers(2, 7))] = True
        a = np.clip(rng.beta(0.7, 7, n) + truth * rng.uniform(0.3, 0.7, n), 0, 0.999)
        b = np.clip(rng.beta(0.7, 7, n) + truth * rng.uniform(0.1, 0.8, n), 0, 0.999)
        frames.append(pd.DataFrame({"patient": f"P{p}", "source_path": f"/x/P{p}.edf", "start_s": np.arange(n),
                                    "spikenet2_prob": a.round(4), "morgoth_prob": b.round(4), "truth": truth.astype(int)}))
    df = pd.concat(frames, ignore_index=True)
    path = os.path.join(tmp, "joined.csv")
    df.to_csv(path, index=False)
    return path, df


def test_disjoint_draws_on_a_grid_share_no_eeg_and_their_chances_add_up():
    import contextlib
    with tempfile.TemporaryDirectory() as tmp:
        path, df = bursty(tmp)
        out = os.path.join(tmp, "clips.csv")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            select_clips.main(["sample", path, "--out", out, "--draw", "morgoth_prob=100", "--draw",
                               "spikenet2_prob=100", "--grid-s", "12", "--disjoint", "--disjoint-reps", "8"])
        log = buf.getvalue()
        assert "--grid-s 12: kept" in log and "--disjoint: left out" in log and "8 re-runs" in log, log
        c = pd.read_csv(out, keep_default_na=False)
        assert set(c["draws"]) == {"morgoth", "spikenet2"}  # every clip belongs to exactly one draw
        n_each = c["draws"].value_counts()
        assert all(90 <= n_each[d] <= 100 for d in ("morgoth", "spikenet2")), n_each
        # every clip sits on its file's grid, so no two clips in one file share the 10 s a rater sees
        for _, g in c.groupby("source_path"):
            x = np.sort(g["start_s"].astype(float).to_numpy())
            assert (np.diff(x) % 12 == 0).all() and (np.diff(x) >= 12).all()
        # each clip's chance in the merged table is the sum of its chances under each draw
        pm, ps = pd.to_numeric(c["p_morgoth"]), pd.to_numeric(c["p_spikenet2"])
        assert np.allclose(1 / pd.to_numeric(c["sampling_weight"]), np.minimum(1, pm + ps), rtol=1e-4)
        assert (pm[c["draws"] == "morgoth"] > 0).all() and (ps[c["draws"] == "spikenet2"] > 0).all()
        # the first draw is made as without --disjoint: the same clips as a MORGOTH-only draw with the same seed
        alone = os.path.join(tmp, "alone.csv")
        with contextlib.redirect_stdout(io.StringIO()):
            select_clips.main(["sample", path, "--out", alone, "--draw", "morgoth_prob=100", "--grid-s", "12"])
        a = pd.read_csv(alone)
        m = c[c["draws"] == "morgoth"]
        assert set(zip(a["source_path"], a["start_s"])) == set(zip(m["source_path"], m["start_s"].astype(int)))
        # --grid-s and --min-gap-s together are refused
        try:
            select_clips.main(["sample", path, "--out", alone, "--draw", "morgoth_prob=10", "--grid-s", "12",
                               "--min-gap-s", "12"])
            raise AssertionError("grid and gap")
        except SystemExit as e:
            assert "leave --min-gap-s out" in str(e)


def test_grid_and_disjoint_estimates_stay_unbiased_over_repeated_draws():
    import contextlib
    with tempfile.TemporaryDirectory() as tmp:
        path, df = bursty(tmp, seed=9)
        cand = df[(df["start_s"] >= 5) & (df["start_s"] <= df.groupby("source_path")["start_s"].transform("max") - 5)]
        true = float(cand["truth"].mean())
        ests = []
        for seed in range(10):
            out = os.path.join(tmp, f"c{seed}.csv")
            with contextlib.redirect_stdout(io.StringIO()):
                select_clips.main(["sample", path, "--out", out, "--draw", "morgoth_prob=100", "--draw",
                                   "spikenet2_prob=100", "--grid-s", "12", "--disjoint", "--disjoint-reps", "6",
                                   "--seed", str(seed)])
            c = pd.read_csv(out)
            ests.append(float((c["sampling_weight"] * c["truth"]).sum() / c["sampling_weight"].sum()))
        assert abs(np.mean(ests) - true) < 0.15 * true, (np.mean(ests), true)


def test_join_one_table_with_pool_and_flags_and_drop_unflagged():
    with tempfile.TemporaryDirectory() as tmp:
        sec = pd.DataFrame({"patient": "US09-01", "source_path": "/x/a.edf", "start_s": np.arange(40) + 0.5,
                            "edf_clock": "onsets", "prob": 0.3})
        a = os.path.join(tmp, "a.csv")
        sec.to_csv(a, index=False)
        flags = os.path.join(tmp, "flags.csv")
        # rows for the first 30 seconds only; one of them flagged
        pd.DataFrame({"path": "/x/a.edf", "start_s": np.arange(30) + 0.5, "artifact": (np.arange(30) == 4).astype(int)}
                     ).to_csv(flags, index=False)
        out = os.path.join(tmp, "j.csv")
        select_clips.main(["join", "--out", out, f"morgoth={a}", "--drop-flagged", flags])
        assert len(pd.read_csv(out)) == 39  # the 10 seconds with no row stay in by default
        select_clips.main(["join", "--out", out, f"morgoth={a}", "--drop-flagged", flags, "--drop-unflagged"])
        j = pd.read_csv(out)
        assert len(j) == 29 and j["start_s"].max() == 29.5 and "morgoth_prob" in j.columns
        try:
            select_clips.main(["join", "--out", out, f"morgoth={a}"])
            raise AssertionError("one table and nothing to do")
        except SystemExit as e:
            assert "two or more" in str(e)


def test_join_and_flags_match_seconds_that_differ_only_by_rounding():
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(5)
        t0 = np.arange(200) + 0.123456
        a, b = os.path.join(tmp, "a.csv"), os.path.join(tmp, "b.csv")
        pd.DataFrame({"patient": "US09-01", "source_path": "/x/a.edf", "start_s": t0, "prob": 0.1,
                      "edf_clock": "onsets"}).to_csv(a, index=False)
        # the second table's times differ by up to 0.00005 s (another run's rounding) and it lacks 3 seconds
        jit = t0 + rng.uniform(-5e-5, 5e-5, 200)
        pd.DataFrame({"patient": "US09-01", "source_path": "/x/a.edf", "start_s": np.delete(jit, [7, 8, 9]),
                      "prob": 0.2, "edf_clock": "onsets"}).to_csv(b, index=False)
        flags = os.path.join(tmp, "flags.csv")
        pd.DataFrame({"path": "/x/a.edf", "start_s": t0[:150] - 4e-5, "artifact": (np.arange(150) % 10 == 0).astype(int)}
                     ).to_csv(flags, index=False)
        out = os.path.join(tmp, "j.csv")
        select_clips.main(["join", "--out", out, f"morgoth={a}", f"spikenet2={b}", "--drop-flagged", flags,
                           "--drop-unflagged"])
        j = pd.read_csv(out)
        # 197 matched; 15 flagged among the first 150 (seconds 7 to 9 are unflagged anyway); 50 with no flag row
        assert len(j) == 197 - 15 - 50, len(j)
        assert np.allclose(j["start_s"] % 1, 0.123456) and (j["spikenet2_prob"] == 0.2).all()
        assert not ((j["start_s"] - 0.123456).round().astype(int) % 10 == 0).any()
        # a tolerance below the rounding finds almost nothing
        try:
            select_clips.main(["join", "--out", out, f"morgoth={a}", f"spikenet2={b}", "--match-tol-s", "1e-7"])
            assert len(pd.read_csv(out)) < 20
        except SystemExit as e:
            assert "share no scored seconds" in str(e)


def test_join_keeps_pool_seconds_maps_subq_and_drops_flags():
    with tempfile.TemporaryDirectory() as tmp:
        sec = pd.DataFrame({"patient": "US09-01", "source_path": "/x/a.edf", "start_s": np.arange(100) + 0.5,
                            "edf_clock": "onsets"})
        a, b = os.path.join(tmp, "a.csv"), os.path.join(tmp, "b.csv")
        sec.assign(prob=0.1, detector="spikenet2").to_csv(a, index=False)
        sec.assign(prob=0.2, detector="morgoth").to_csv(b, index=False)
        pool = os.path.join(tmp, "pool.csv")
        pd.DataFrame({"patient": ["US09-01", "US09-01"], "source_path": ["/x/a.edf"] * 2, "start_s": [0.5, 60.0],
                      "end_s": [30.5, 80.2], "subq_path": ["/x/sq1.edf", "/x/sq2.edf"], "subq_t0_s": [1000.0, 7.25],
                      "subq_rate": [1.0, 1.0001], "implant_side": ["L", "L"], "edf_clock": "onsets"}).to_csv(pool, index=False)
        flags = os.path.join(tmp, "flags.parquet")
        pd.DataFrame({"patient": "US09-01", "path": "/x/a.edf", "start_s": [3.5, 61.5, 90.5], "artifact": [1, 1, 1]}
                     ).to_parquet(flags)
        out = os.path.join(tmp, "j.csv")
        select_clips.main(["join", "--out", out, f"spikenet2={a}", f"morgoth={b}", "--pool", pool,
                           "--drop-flagged", flags])
        j = pd.read_csv(out)
        # 30 whole seconds in the first interval, 19 in the second (79.5 + 1 > 80.2), less two flagged ones
        assert len(j) == 30 + 19 - 2
        assert not j["start_s"].isin([3.5, 61.5]).any() and j["start_s"].max() == 78.5
        first = j[j["subq_path"] == "/x/sq1.edf"]
        assert np.allclose(first["subq_start_s"], 1000 + (first["start_s"] - 0.5))
        second = j[j["subq_path"] == "/x/sq2.edf"]
        assert np.allclose(second["subq_start_s"], 7.25 + (second["start_s"] - 60.0) * 1.0001)
        assert set(j["implant_side"]) == {"L"} and {"spikenet2_prob", "morgoth_prob"} <= set(j.columns)
        # the draw carries the SubQ map to each clip, at the middle of the drawn second
        clips = os.path.join(tmp, "clips.csv")
        select_clips.main(["sample", out, "--out", clips, "--draw", "spikenet2_prob=3", "--bins", "0,1",
                           "--max-patient-frac", "1"])
        c = pd.read_csv(clips)
        m = c.merge(j[["start_s", "subq_start_s", "subq_rate"]], on="start_s", suffixes=("", "_j"))
        assert np.allclose(m["subq_center_s"], m["subq_start_s_j"] + 0.5 * m["subq_rate_j"], atol=1e-3)
        # refusals: a pool on another clock, overlapping intervals, a pool naming another patient
        for change, why in ((dict(edf_clock="samples"), "clock"), (dict(start_s=[0.5, 20.0]), "overlapping"),
                            (dict(patient=["US09-01", "US09-02"]), "different patient")):
            bad = pd.read_csv(pool).assign(**change)
            bad.to_csv(os.path.join(tmp, "bad.csv"), index=False)
            try:
                select_clips.main(["join", "--out", out, f"spikenet2={a}", f"morgoth={b}", "--pool",
                                   os.path.join(tmp, "bad.csv")])
                raise AssertionError(why)
            except SystemExit as e:
                assert why in str(e), (why, str(e))


def write_pair(tmp):
    """A scalp EDF and a SubQ EDF whose clock runs 100 s ahead, both contiguous from 0."""
    from edf_writer import write_edf
    rng = np.random.default_rng(12)
    scalp = rng.standard_normal((len(LABS), 200 * 80)) * 25
    sq = rng.standard_normal((2, 200 * 300)) * 15
    sq[:, 200 * 130:200 * 131] += 300  # a block 30 s into the scalp file, 130 s into SubQ
    root = os.path.join(tmp, "uneeg")
    os.makedirs(root)
    sp = write_edf(os.path.join(root, "scalp.edf"), LABS, 200, [(0, scalp)])
    qp = write_edf(os.path.join(root, "subq.edf"), ["D-C", "P-C"], 200, [(0, sq)])
    return sp, qp, scalp, sq


def admin_study(tmp):
    sp, qp, scalp, sq = write_pair(tmp)
    n = 80
    base = pd.DataFrame({"patient": "US09-01", "source_path": sp, "start_s": np.arange(n, dtype=float)})
    rng = np.random.default_rng(3)
    a, b = os.path.join(tmp, "a.csv"), os.path.join(tmp, "b.csv")
    base.assign(prob=rng.random(n).round(3)).to_csv(a, index=False)
    base.assign(prob=rng.random(n).round(3)).to_csv(b, index=False)
    pool = os.path.join(tmp, "pool.csv")
    pd.DataFrame({"patient": ["US09-01"], "source_path": [sp], "start_s": [0.0], "end_s": [80.0], "subq_path": [qp],
                  "subq_t0_s": [100.0], "subq_rate": [1.0], "implant_side": ["R"]}).to_csv(pool, index=False)
    joined, clips, admin = (os.path.join(tmp, f) for f in ("joined.csv", "clips.csv", "admin.csv"))
    select_clips.main(["join", "--out", joined, f"spikenet2={a}", f"morgoth={b}", "--pool", pool])
    select_clips.main(["sample", joined, "--out", clips, "--draw", "spikenet2_prob=3", "--draw", "morgoth_prob=3",
                       "--random", "2", "--bins", "0,0.5,1", "--max-patient-frac", "1", "--min-gap-s", "12"])
    sqsec = os.path.join(tmp, "sq.parquet")
    pd.DataFrame({"patient": "US09-01", "path": qp, "start_s": np.arange(300, dtype=float),
                  "prob": (np.arange(300) == 130).astype(float)}).to_parquet(sqsec)
    select_clips.main(["admin", clips, "--out", admin, "--scalp", f"spikenet2={a}", "--scalp", f"morgoth={b}",
                       "--subq", f"spikenet2={sqsec}"])
    return clips, admin, sq, a


def test_admin_table_maps_scores_and_subq_onto_each_clip():
    with tempfile.TemporaryDirectory() as tmp:
        clips, admin, sq, a = admin_study(tmp)
        c, ad = pd.read_csv(clips), pd.read_csv(admin, keep_default_na=False)
        assert len(ad) == len(c) and (ad["implant_side"] == "R").all()
        assert (ad["virtual"] == "F8-T4").all()  # the default right-side pair
        sa = pd.read_csv(a)
        for _, r in ad.iterrows():
            strips = json.loads(r["strips"])
            names = [(s["kind"], s["name"]) for s in strips]
            assert ("scalp", "SpikeNet2") in names and ("scalp", "MORGOTH") in names and ("subq", "SpikeNet2") in names
            sn2 = next(s for s in strips if s["kind"] == "scalp" and s["name"] == "SpikeNet2")
            center = float(r["center_s"])
            for t, p in sn2["secs"]:  # each box sits where its second is in the 12 s clip
                start = center - 6 + t
                assert abs(sa.loc[np.isclose(sa["start_s"], start), "prob"].iloc[0] - p) < 1e-6
            sub = next(s for s in strips if s["kind"] == "subq")
            hit = [t for t, p in sub["secs"] if p == 1]
            # SubQ second 130 is scalp second 30: it lands at 30 - (center - 6) in the clip, when the clip covers it
            if center - 6 <= 30 < center + 6:
                assert hit and abs(hit[0] - (30 - (center - 6))) < 1e-6
            else:
                assert not hit
            assert float(r["spikenet2_clipmax"]) >= float(c.loc[_, "spikenet2_prob"]) - 1e-9


def test_outer_subq_pair_follows_the_contacts():
    import make_review_batches as mrb
    assert mrb.outer_pair(["D-C", "P-C"]) == ("D-P", -1)  # UNEEG: (D-C) - (P-C) = D-P
    assert mrb.outer_pair(["C-P", "C-D"]) == ("D-P", -1)  # a rater's reading: (C-P) - (C-D) = D-P
    assert mrb.outer_pair(["SQD-SQC", "SQP-SQC"]) == ("SQD-SQP", -1)
    assert mrb.outer_pair(["D-C", "C-P"]) == ("D-P", 1)  # chained: (D-C) + (C-P) = D-P
    for names in (["D-C"], ["D-C", "P-C", "D-P"], ["D-C", "D-C"], ["D-C", "P-Q"], ["DC", "PC"], ["D-C", "C-D"]):
        assert mrb.outer_pair(names) is None, names


def test_admin_view_reaches_only_the_owner_link_on_borel_and_the_site():
    import make_review_batches as mrb
    import check_upload
    from test_tools import Server, live_opts
    with tempfile.TemporaryDirectory() as tmp:
        clips, admin, sq, _a = admin_study(tmp)
        study, labels = os.path.join(tmp, "study"), os.path.join(tmp, "labels")
        st = serve_review.LiveStore(clips, study, labels, live_opts(repeat_frac=0, batch_size=4, admin=admin))
        assert st.plan["settings"]["question"] == "clip" and st.has_admin()
        bid = st.plan["batches"][0]["batch_id"]
        batch = json.loads(gzip.decompress(st.batch_bytes(bid)))
        assert batch["question"] == "clip" and "subq" not in json.dumps(batch).lower()
        obj = json.loads(gzip.decompress(st.admin_bytes(bid)))
        assert obj["format"] == "ied-review-admin" and len(obj["clips"]) == len(batch["clips"])
        key = pd.read_csv(os.path.join(study, "key.csv"))
        for item in obj["clips"]:
            assert item["implant_side"] == "R" and item["subq"]["channels"] == ["D-C", "P-C", "D-P"] and item["strips"]
            assert item["virtual"] == [["F8", "T4"]]
            row = key[key["clip_id"] == item["id"]].iloc[0]
            got = mrb.decode(item["subq"]["data"], 3, 2400, item["subq"]["scale"])
            k = int(round((float(row["subq_center_s"]) - 6) * 200))
            want = sq[:, k:k + 2400]
            want = np.vstack([want, want[0] - want[1]])  # D-P = (D-C) - (P-C)
            err = np.abs(got - (want - want.mean(axis=1, keepdims=True))).max(axis=1)
            assert err[:2].max() < 0.14 and err[2] < 0.25, err  # 0.1 µV steps + the test EDF's 0.15 µV steps (twice for D-P)
            text = json.dumps(item)
            for leak in ("subq.edf", "scalp.edf", "US09", tmp):
                assert leak not in text, leak
        # the Borel server: the owner link gets it, a rater link never does
        s = Server(tmp, raters=["EC"], store=st)
        try:
            assert json.loads(s.req("/api/batches")[1])["admin"] is True
            assert "admin" not in json.loads(s.req("/api/batches", who="EC")[1])
            assert s.req(f"/api/admin/{bid}", who="EC")[0] == 403
            code, body = s.req(f"/api/admin/{bid}")
            assert code == 200 and json.loads(gzip.decompress(body))["batch_id"] == bid
            assert s.req("/api/admin/nope")[0] == 404
        finally:
            s.close()
        # the hosted site: admin files next to the batches, passing the upload check
        site = os.path.join(tmp, "site")
        out = subprocess.run([sys.executable, os.path.join(KIT, "hosted", "make_site.py"), "--clips", clips, "--study",
                              study, "--labels", labels, "--out", site, "--raters", "EC", "--admin", admin],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stdout + out.stderr
        assert "admin view:" in out.stdout
        sj = json.load(open(os.path.join(site, "content", "site.json")))
        assert all(b["admin_file"] == b["file"] for b in sj["batches"])
        assert check_upload.check(site, labels) == []
        # a planted problem in an admin file is caught
        p = os.path.join(site, "content", "admin", f"{bid}.json.gz")
        bad = json.loads(gzip.decompress(open(p, "rb").read()))
        bad["clips"][0]["drawn_by"] = ["US09-01"]
        open(p, "wb").write(gzip.compress(json.dumps(bad).encode()))
        assert any("UNEEG subject ID" in x for x in check_upload.check(site, labels))
        if not shutil.which("node"):
            return
        open(p, "wb").write(gzip.compress(json.dumps(obj).encode(), mtime=0))
        from test_hosted import DevServer
        owner, raters = serve_review.load_token(labels, False), serve_review.load_rater_tokens(labels, False)
        srv = DevServer(site, os.path.join(tmp, "store"))
        try:
            assert json.loads(srv.req("/api/batches", owner)[1])["admin"] is True
            assert "admin" not in json.loads(srv.req("/api/batches", raters["EC"])[1])
            assert srv.req(f"/api/admin/{bid}", raters["EC"])[0] == 403
            assert srv.req(f"/api/admin/{bid}")[0] == 403
            code, body, _h = srv.req(f"/api/admin/{bid}", owner)
            assert code == 200 and json.loads(gzip.decompress(body))["batch_id"] == bid
            # labels from a viewer before kit 1.7 (no view column) are still saved
            ids = [c_["id"] for c_ in batch["clips"]]
            old = io.StringIO()
            w = csv.DictWriter(old, fieldnames=serve_review.EXPORT_COLUMNS[:-1], lineterminator="\n")
            w.writeheader()
            w.writerow({"batch_id": bid, "clip_id": ids[0], "rater": "EC", "order_index": 1, "decision": "yes"})
            vis = sj["assignment"] is None or bid in sj["assignment"]["shared"] or bid in sj["assignment"]["by_rater"]["EC"]
            if vis:
                assert srv.req(f"/api/labels/{bid}/EC", raters["EC"], "PUT", old.getvalue().encode())[0] == 200
        finally:
            srv.close()


def test_report_judges_clips_by_their_maxima_and_leaves_out_answers_that_were_not_blind():
    import report_labels
    rng = np.random.default_rng(2)
    n = 120
    key = pd.DataFrame({"clip_id": [f"c{i:07x}" for i in range(n)], "patient": [f"P{i % 4}" for i in range(n)],
                        "sampling_weight": 10.0, "patient_weight": 10.0, "bin": "x", "strata_prob": 0.3,
                        "spikenet2_prob": 0.1, "spikenet2_clipmax": 0.1, "question": "clip"})
    yes = rng.random(n) < 0.4
    key.loc[yes, "spikenet2_clipmax"] = 0.9  # the detector fires elsewhere in the clip, never on the drawn second
    merged = key.assign(rater="SJ", decision=np.where(yes, "yes", "no"), view="blind", max_electrodes="",
                        generalized=0, seconds_viewed=4)
    merged.loc[:9, "view"] = "superadmin"
    a = report_labels.analyze(merged)
    assert a["clip_level"] and a["n_not_blind"] == 10 and a["n_clips"] == n - 10
    sn2 = a["scorings"][0]
    assert sn2["at"]["sens"] == 1.0 and sn2["at"]["ppv"] == 1.0
    head, body = report_labels.render(a, "t", "s")
    assert "10 s windows with an IED" in body and "Windows flagged" in body and "not blind" not in body
    assert "10 answers given with the admin or superadmin view on" in body
    # a study that asked about the highlighted second is judged on that second's score, not the clip's maximum
    b2 = report_labels.analyze(merged.assign(question="second"))
    assert not b2["clip_level"] and b2["scorings"][0]["at"]["sens"] == 0.0
    assert not report_labels.analyze(merged.drop(columns="question"))["clip_level"]  # studies before 1.7


def test_an_advisor_link_can_be_given_the_admin_view_and_the_virtual_pairs_follow_the_side():
    import check_upload
    from test_tools import Server, live_opts
    with tempfile.TemporaryDirectory() as tmp:
        clips, admin, sq, a = admin_study(tmp)
        # --virtual sets the pairs per side; a clip with no known side shows both sides' pairs
        c = pd.read_csv(clips, dtype=str, keep_default_na=False)
        c.loc[0, "implant_side"] = ""
        c.to_csv(clips, index=False)
        admin2 = os.path.join(tmp, "admin2.csv")
        select_clips.main(["admin", clips, "--out", admin2, "--virtual", "L=F7-T3,T3-T5", "--virtual", "R=F8-T4"])
        ad = pd.read_csv(admin2, keep_default_na=False)
        assert ad.loc[0, "virtual"] == "F7-T3;T3-T5;F8-T4" and (ad.loc[1:, "virtual"] == "F8-T4").all()
        for bad in ("L=F7-X9", "Q=F7-T3", "L=F7"):
            try:
                select_clips.main(["admin", clips, "--out", admin2, "--virtual", bad])
                raise AssertionError(bad)
            except SystemExit as e:
                assert "--virtual" in str(e)
        study, labels = os.path.join(tmp, "study"), os.path.join(tmp, "labels")
        st = serve_review.LiveStore(clips, study, labels, live_opts(repeat_frac=0, batch_size=4, admin=admin))
        bid = st.plan["batches"][0]["batch_id"]
        assert set(pd.read_csv(os.path.join(study, "key.csv"))["question"]) == {"clip"}
        # Borel: ADV's link opens the admin view, EC's does not; only raters with links can be named
        raters = serve_review.load_rater_tokens(labels, False, ["EC", "ADV"])
        assert serve_review.load_admin_raters(labels, "adv", raters) == ["ADV"]
        assert serve_review.load_admin_raters(labels, None, raters) == ["ADV"]  # remembered
        try:
            serve_review.load_admin_raters(labels, "JK", raters)
            raise AssertionError("JK has no link")
        except SystemExit as e:
            assert "no rater link" in str(e)
        s = Server(tmp, raters=["EC", "ADV"], store=st, admin_raters=["ADV"])
        try:
            assert json.loads(s.req("/api/batches", who="ADV")[1])["admin"] is True
            assert "admin" not in json.loads(s.req("/api/batches", who="EC")[1])
            assert s.req(f"/api/admin/{bid}", who="EC")[0] == 403
            code, body = s.req(f"/api/admin/{bid}", who="ADV")
            assert code == 200 and json.loads(gzip.decompress(body))["batch_id"] == bid
        finally:
            s.close()
        # the site: the same rule, and the upload check refuses an admin name with no link
        site = os.path.join(tmp, "site")
        out = subprocess.run([sys.executable, os.path.join(KIT, "hosted", "make_site.py"), "--clips", clips, "--study",
                              study, "--labels", labels, "--out", site, "--raters", "EC,ADV", "--admin", admin],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stdout + out.stderr
        assert "(also opens the admin view)" in out.stdout and "and so do ADV's" in out.stdout
        sj_path = os.path.join(site, "content", "site.json")
        sj = json.load(open(sj_path))
        assert sj["admin_raters"] == ["ADV"] and check_upload.check(site, labels) == []
        p = os.path.join(site, "content", "admin", f"{bid}.json.gz")
        obj = json.loads(gzip.decompress(open(p, "rb").read()))
        assert all(item["virtual"] == [["F8", "T4"]] for item in obj["clips"])
        bad = json.loads(json.dumps(obj))
        bad["clips"][0]["virtual"] = [["F8", "US09"]]
        open(p, "wb").write(gzip.compress(json.dumps(bad).encode()))
        assert any("virtual UNEEG" in x for x in check_upload.check(site, labels))
        open(p, "wb").write(gzip.compress(json.dumps(obj).encode(), mtime=0))
        json.dump(dict(sj, admin_raters=["JK"]), open(sj_path, "w"))
        assert any("has no rater link" in x for x in check_upload.check(site, labels))
        json.dump(sj, open(sj_path, "w"), indent=1, sort_keys=True)
        if not shutil.which("node"):
            return
        from test_hosted import DevServer
        owner, toks = serve_review.load_token(labels, False), serve_review.load_rater_tokens(labels, False)
        srv = DevServer(site, os.path.join(tmp, "store"))
        try:
            assert json.loads(srv.req("/api/batches", toks["ADV"])[1])["admin"] is True
            assert "admin" not in json.loads(srv.req("/api/batches", toks["EC"])[1])
            assert srv.req(f"/api/admin/{bid}", toks["EC"])[0] == 403
            vis = sj["assignment"] is None or bid in sj["assignment"]["shared"] or bid in sj["assignment"]["by_rater"]["ADV"]
            assert srv.req(f"/api/admin/{bid}", toks["ADV"])[0] == (200 if vis else 404)
            assert srv.req(f"/api/admin/{bid}", owner)[0] == 200
        finally:
            srv.close()


# ---------------------------------------------------------------- staged study (kit 1.8)

def staged_study(tmp, **parts_kw):
    """The admin study's clips as the scalp part of a staged study, with a made-up reference rater: two training
    examples at other seconds of the file, then the scalp clips, then 4 UNEEG clips (2 reference Yes, 2 No)."""
    clips, _admin, sq, _a = admin_study(tmp)
    c = pd.read_csv(clips, dtype=str, keep_default_na=False)
    centres = c["center_s"].astype(float).to_numpy()
    free = [t for t in np.arange(6.5, 74.0, 1.0) if np.min(np.abs(centres - t)) >= 2][:3]
    ref = pd.concat([c[["source_path", "center_s"]].assign(decision=["yes" if i % 2 == 0 else "no" for i in range(len(c))]),
                     pd.DataFrame({"source_path": c["source_path"].iloc[0], "center_s": [f"{t:.1f}" for t in free],
                                   "decision": "yes"})], ignore_index=True)
    ref = pd.concat([ref.assign(rater="ADV"), ref.assign(rater="JK", decision="no")], ignore_index=True)
    refp, staged = os.path.join(tmp, "ref.csv"), os.path.join(tmp, "staged.csv")
    ref.to_csv(refp, index=False)
    args = ["parts", "--scalp", clips, "--ref", refp, "--ref-rater", "adv", "--examples-n", "2", "--gap-s", "2",
            "--uneeg-n", "4", "--uneeg-pick", "ref", "--out", staged]
    for k, v in parts_kw.items():
        args += ["--" + k.replace("_", "-"), str(v)]
    select_clips.main(args)
    return staged, sq, c


def answers_for(batch_obj, rater, decisions):
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=serve_review.EXPORT_COLUMNS, lineterminator="\n")
    w.writeheader()
    for i, (clip, d) in enumerate(zip(batch_obj["clips"], decisions)):
        w.writerow({"batch_id": batch_obj["batch_id"], "clip_id": clip["id"], "rater": rater, "order_index": i + 1,
                    "decision": d, "labeled_at": "2026-11-04T00:00:00Z"})
    return out.getvalue().encode()


def test_parts_table_holds_examples_scalp_and_a_yes_no_uneeg_mix():
    with tempfile.TemporaryDirectory() as tmp:
        staged, _sq, c = staged_study(tmp)
        t = pd.read_csv(staged, dtype=str, keep_default_na=False)
        assert list(t["part"].value_counts().sort_index().items()) == [("examples", 2), ("scalp", len(c)), ("uneeg", 4)]
        ex = t[t["part"] == "examples"]
        assert (ex["ref_answer"] == "yes").all()
        scalp_c = set(c["center_s"].astype(float).round(3))
        assert not set(ex["center_s"].astype(float).round(3)) & scalp_c  # never a scalp-part second
        u = t[t["part"] == "uneeg"]
        assert set(u["center_s"].astype(float).round(3)) <= scalp_c and (u["subq_path"] != "").all()
        # ADV's answers chose them, 2 Yes and 2 No (JK said No to everything and is not the reference)
        yes = set(c["center_s"].astype(float).round(3).iloc[::2])
        assert sum(x in yes for x in u["center_s"].astype(float).round(3)) == 2
        assert list(t.columns[-4:]) == ["part", "ref_answer", "ref_type", "ref_note"]


def test_staged_study_opens_part_by_part_on_borel():
    import make_review_batches as mrb
    from test_tools import Server, live_opts
    with tempfile.TemporaryDirectory() as tmp:
        staged, sq, c = staged_study(tmp)
        study, labels = os.path.join(tmp, "study"), os.path.join(tmp, "labels")
        st = serve_review.LiveStore(staged, study, labels, live_opts(repeat_frac=0.25, batch_size=4, question="second"))
        n_rep = int(round(0.25 * len(c)))
        n_s = -(-(len(c) + n_rep) // 4)
        sids = [f"s{i + 1:02d}" for i in range(n_s)]
        ids = sids + ["e01", "u01"]  # the 10-20 clips, then the examples, then UNEEG (kit 1.8.4)
        assert [b["batch_id"] for b in st.plan["batches"]] == ids
        assert all(b["shared"] for b in st.plan["batches"]) and st.plan["settings"]["subq_channels"] == ["D-C", "P-C", "D-P"]
        n = {b["batch_id"]: len(b["members"]) for b in st.plan["batches"]}
        assert n["e01"] == 2 and n["u01"] == 4 and sum(n[i] for i in sids) == len(c) + n_rep  # repeats: scalp only
        key = pd.read_csv(os.path.join(study, "key.csv"), dtype=str, keep_default_na=False)
        assert key["clip_id"].is_unique and set(key["part"]) == {"examples", "scalp", "uneeg"}
        b = {bid: json.loads(gzip.decompress(st.batch_bytes(bid))) for bid in n}
        assert b["e01"]["version"] == 2 and b["e01"]["training"] is True and b["e01"]["stage"] == 2
        assert [x["ref"] for x in b["e01"]["clips"]] == ["yes", "yes"] and b["e01"]["answers"] == ["seen"]
        # examples show the study's UNEEG channels under the scalp (1.8.1); these two have no SubQ file: dashed
        assert b["e01"]["view"] == "both" and b["e01"]["channels"][-3:] == ["D-C", "P-C", "D-P"]
        assert b["e01"]["groups"]["SubQ"] == ["D-C", "P-C", "D-P"]
        assert all({"D-C", "P-C", "D-P"} <= set(x["bad"]) for x in b["e01"]["clips"])
        assert b["s01"]["stage"] == 1 and "training" not in b["s01"] and "view" not in b["s01"]
        assert not any("ref" in x for x in b["s01"]["clips"])
        u = b["u01"]
        assert u["view"] == "subq" and u["stage"] == 3 and u["channels"] == ["D-C", "P-C", "D-P"]
        assert u["groups"] == {"SubQ": ["D-C", "P-C", "D-P"]}
        for item in u["clips"]:
            row = key[key["clip_id"] == item["id"]].iloc[0]
            assert row["part"] == "uneeg"
            got = mrb.decode(item["data"], 3, 2400, item["scale"])
            k = int(round((float(row["subq_center_s"]) - 6) * 200))
            want = sq[:, k:k + 2400]
            want = np.vstack([want, want[0] - want[1]])
            err = np.abs(got - (want - want.mean(axis=1, keepdims=True))).max(axis=1)
            assert err.max() < 0.25, err

        s = Server(tmp, raters=["EC"], store=st, tolerance_ms=250)
        try:
            j = json.loads(s.req("/api/batches", who="EC")[1])
            assert j["tolerance_s"] == 0.25
            got = {x["batch_id"]: x for x in j["batches"]}
            assert [x["batch_id"] for x in j["batches"]] == ids
            assert got["e01"]["stage_name"] == "UNEEG training examples" and got["e01"]["training"] is True
            assert got["s01"]["stage_name"] == "10-20 clips" and got["s01"]["stage"] == 1
            assert got["u01"]["view"] == "subq" and got["e01"]["view"] == "both"
            assert [x.get("locked", False) for x in j["batches"]] == [False] * n_s + [True, True]
            # a locked part can be neither opened nor saved to
            code, msg = s.req("/api/batch/e01", who="EC")
            assert code == 403 and b"opens once" in msg
            assert s.req("/api/labels/e01/EC", "PUT", answers_for(b["e01"], "EC", ["yes"]), who="EC")[0] == 403
            # the owner opens everything
            assert not any(x.get("locked") for x in json.loads(s.req("/api/batches")[1])["batches"])
            assert s.req("/api/batch/u01")[0] == 200
            # one 10-20 clip left: the examples stay locked; all answered: the examples open, the UNEEG part does not
            for k, bid in enumerate(sids):
                ans = ["no"] * len(b[bid]["clips"])
                if k == len(sids) - 1:
                    ans[-1] = ""
                assert s.req(f"/api/labels/{bid}/EC", "PUT", answers_for(b[bid], "EC", ans), who="EC")[0] == 200
            assert json.loads(s.req("/api/batches", who="EC")[1])["batches"][n_s].get("locked") is True
            last = sids[-1]
            body = answers_for(b[last], "EC", ["no"] * len(b[last]["clips"]))
            assert s.req(f"/api/labels/{last}/EC", "PUT", body, who="EC")[0] == 200
            j = json.loads(s.req("/api/batches", who="EC")[1])
            assert [x.get("locked", False) for x in j["batches"]] == [False] * (n_s + 1) + [True]
            assert s.req("/api/batch/e01", who="EC")[0] == 200
            # examples are looked at, not answered (kit 1.8.5): Yes or No is refused, seen is what opens the next part
            assert s.req("/api/labels/e01/EC", "PUT", answers_for(b["e01"], "EC", ["yes", "no"]), who="EC")[0] == 400
            assert s.req("/api/labels/e01/EC", "PUT", answers_for(b["e01"], "EC", ["seen", "seen"]), who="EC")[0] == 200
            j = json.loads(s.req("/api/batches", who="EC")[1])
            assert not any(x.get("locked") for x in j["batches"]) and s.req("/api/batch/u01", who="EC")[0] == 200
        finally:
            s.close()


def test_pick_by_bins_fills_each_score_bin_evenly():
    rng = np.random.default_rng(1)
    # a full 10 x 10 grid, 10 rows a cell: 250 picks give exactly 25 in every bin of both scores
    full = np.array([(i, j) for i in range(10) for j in range(10) for _ in range(10)])
    got = full[select_clips.pick_by_bins(full, 250, rng)]
    for k in range(2):
        assert list(np.bincount(got[:, k], minlength=10)) == [25] * 10
    cell = np.bincount(got[:, 0] * 10 + got[:, 1], minlength=100)
    assert cell.min() >= 2 and cell.max() <= 4  # spread over the cells too
    # a thin top bin gives what it has; the rest stay as even as they can
    thin = np.array([r for r in full if not (r[0] == 9 and r[1] < 9)] + [(9, 0)] * 2)
    got = thin[select_clips.pick_by_bins(thin, 250, rng)]
    m = np.bincount(got[:, 0], minlength=10)
    assert m[9] == 12 and m[:9].min() >= 25 and m.sum() == 250
    assert len(set(select_clips.pick_by_bins(full, 250, rng))) == 250  # no row twice
    assert select_clips._score_bin(pd.Series(["0", "0.099", "0.1", "0.95", "1", "", "1.2"])).tolist() == \
        [0, 0, 1, 9, 9, -1, -1]


def test_parts_from_hand_picked_examples_and_uneeg_bins_show_both_on_borel():
    import make_review_batches as mrb
    from test_tools import live_opts
    with tempfile.TemporaryDirectory() as tmp:
        clips, _admin, sq, _a = admin_study(tmp)
        c = pd.read_csv(clips, dtype=str, keep_default_na=False)
        centres = c["center_s"].astype(float).to_numpy()
        free = [t for t in np.arange(6.5, 74.0, 1.0) if np.min(np.abs(centres - t)) >= 2][:2]
        ex = pd.DataFrame({"source_path": c["source_path"].iloc[0], "center_s": [f"{t:.1f}" for t in free],
                           "subq_path": c["subq_path"].iloc[0], "subq_center_s": [f"{t + 100:.1f}" for t in free],
                           "ref_answer": ["", "no"], "ref_type": ["Focal, right", ""],
                           "ref_note": ["Focal, right temporal", ""]})
        exp, staged = os.path.join(tmp, "examples.csv"), os.path.join(tmp, "staged.csv")
        ex.to_csv(exp, index=False)
        # the kind of spike (kit 1.8.5) is one of three; anything else is refused
        ex.assign(ref_type=["sharpish", ""]).to_csv(exp + ".bad.csv", index=False)
        try:
            select_clips.main(["parts", "--scalp", clips, "--examples-table", exp + ".bad.csv", "--gap-s", "2",
                               "--uneeg-n", "4", "--out", staged + ".x"])
            raise AssertionError("a ref_type that is not a kind of spike")
        except SystemExit as err:
            assert "ref_type must be one of generalized, focal left, focal right" in str(err), err
        # no --ref: the examples are given and the UNEEG clips are picked by score bin
        select_clips.main(["parts", "--scalp", clips, "--examples-table", exp, "--gap-s", "2", "--uneeg-n", "4",
                           "--out", staged])
        t = pd.read_csv(staged, dtype=str, keep_default_na=False)
        e = t[t["part"] == "examples"]
        assert list(e["ref_answer"]) == ["yes", "no"] and list(e["ref_note"]) == ["Focal, right temporal", ""]
        assert list(e["ref_type"]) == ["focal right", ""]
        assert list(e["subq_center_s"]) == list(ex["subq_center_s"])
        u = t[t["part"] == "uneeg"]
        assert len(u) == 4 and set(u["center_s"].astype(float)) <= set(c["center_s"].astype(float))
        # without --ref, examples drawn from the reference cannot be made; nor can an example near a scalp clip
        for extra, why in ((["--examples-n", "2"], "--ref is needed"),
                           (["--examples-table", exp, "--gap-s", "100"], "within 100 s of a scalp clip")):
            try:
                select_clips.main(["parts", "--scalp", clips, "--out", staged + ".x"] + extra)
                raise AssertionError(why)
            except SystemExit as err:
                assert why in str(err), err
        try:
            select_clips.main(["parts", "--scalp", clips, "--examples-table", exp, "--gap-s", "2",
                               "--uneeg-bins-by", "cyclops_prob", "--out", staged + ".x"])
            raise AssertionError("no cyclops_prob column")
        except SystemExit as err:
            assert "no column cyclops_prob" in str(err) and "morgoth_prob" in str(err)

        # examples on the onsets clock with a scalp table that names no clock: refused until --edf-clock says
        # which clock the scalp times are on, then every row carries it (kit 1.8.2)
        ex.assign(edf_clock="onsets").to_csv(exp + ".clock.csv", index=False)
        noclock = clips + ".noclock.csv"
        c.drop(columns=["edf_clock"]).to_csv(noclock, index=False)
        for scalp_t, why in ((noclock, "--edf-clock"), (clips, "on the scalp table's clock")):
            try:
                select_clips.main(["parts", "--scalp", scalp_t, "--examples-table", exp + ".clock.csv", "--gap-s",
                                   "2", "--uneeg-n", "4", "--out", staged + ".x"])
                raise AssertionError("mixed clocks")
            except SystemExit as err:
                assert "different EDF clocks" in str(err) and why in str(err), err
        select_clips.main(["parts", "--scalp", noclock, "--examples-table", exp + ".clock.csv", "--gap-s", "2",
                           "--uneeg-n", "4", "--edf-clock", "onsets", "--out", staged + ".clock"])
        tc = pd.read_csv(staged + ".clock", dtype=str, keep_default_na=False)
        assert (tc["edf_clock"] == "onsets").all() and list(tc.columns[-4:]) == ["part", "ref_answer", "ref_type", "ref_note"]
        assert len(tc) == len(t)

        st = serve_review.LiveStore(staged, os.path.join(tmp, "study"), os.path.join(tmp, "labels"),
                                    live_opts(repeat_frac=0, batch_size=4, question="second"))
        e01 = json.loads(gzip.decompress(st.batch_bytes("e01")))
        n_sc = len(mrb.SCALP) + 1  # and EKG
        assert e01["view"] == "both" and e01["channels"][n_sc:] == ["D-C", "P-C", "D-P"]
        assert e01["groups"] == {"EKG": ["EKG"], "SubQ": ["D-C", "P-C", "D-P"]}
        key = pd.read_csv(os.path.join(tmp, "study", "key.csv"), dtype=str, keep_default_na=False)
        for item in e01["clips"]:
            row = key[key["clip_id"] == item["id"]].iloc[0]
            assert not {"D-C", "P-C", "D-P"} & set(item.get("bad", []))
            got = mrb.decode(item["data"], n_sc + 3, 2400, item["scale"])[n_sc:]
            k = int(round((float(row["subq_center_s"]) - 6) * 200))
            want = sq[:, k:k + 2400]
            want = np.vstack([want, want[0] - want[1]])
            assert np.abs(got - (want - want.mean(axis=1, keepdims=True))).max() < 0.25
            assert item["ref"] == ("yes" if row["ref_answer"] == "yes" else "no")
        assert [x.get("ref_note") for x in sorted(e01["clips"], key=lambda x: x["ref"], reverse=True)] == \
            ["Focal, right temporal", None]
        assert [x.get("ref_type") for x in sorted(e01["clips"], key=lambda x: x["ref"], reverse=True)] == \
            ["focal right", None]
        assert e01["answers"] == ["seen"]


def test_parts_leaves_clips_without_a_subq_time_out_of_the_uneeg_part():
    # kit 1.8.3: an empty subq_center_s (a clip in an unaligned piece) read back as NaN once became the text "nan"
    # and passed as a SubQ time, so such clips reached the UNEEG part with nothing to show
    with tempfile.TemporaryDirectory() as tmp:
        clips, _admin, _sq, _a = admin_study(tmp)
        c = pd.read_csv(clips, dtype=str, keep_default_na=False)
        assert len(c) >= 5
        c.loc[c.index[:2], "subq_center_s"] = ""
        c.loc[c.index[2], "subq_path"] = ""
        blanked = set(c["center_s"].iloc[:3].astype(float))
        holes = os.path.join(tmp, "holes.csv")
        c.to_csv(holes, index=False)
        for src in (holes, holes.replace(".csv", ".parquet")):
            if src.endswith(".parquet"):
                pd.read_csv(holes).to_parquet(src)  # NaN and None in a parquet file too
            out = src + ".staged.csv"
            select_clips.main(["parts", "--scalp", src, "--examples-n", "0", "--uneeg-n", str(len(c) * 2),
                               "--out", out])
            t = pd.read_csv(out, dtype=str, keep_default_na=False)
            u = t[t["part"] == "uneeg"]
            assert len(u) == len(c) - 3, (src, len(u))
            assert not blanked & set(u["center_s"].astype(float))
            assert not u["subq_center_s"].str.lower().isin(["", "nan", "none"]).any()
            s_ = t[t["part"] == "scalp"]
            assert len(s_) == len(c) and (s_["subq_center_s"].str.lower() != "nan").all()


def test_staged_site_matches_borel_and_passes_the_upload_check():
    import check_upload
    from test_hosted import DevServer, run, HOSTED, no_node
    with tempfile.TemporaryDirectory() as tmp:
        staged, _sq, _c = staged_study(tmp)
        study, labels, site = (os.path.join(tmp, x) for x in ("study", "labels", "site"))
        out = run(os.path.join(HOSTED, "make_site.py"), "--clips", staged, "--study", study, "--labels", labels,
                  "--out", site, "--raters", "EC", "--batch-size", "4", "--question", "second",
                  "--tolerance-ms", "250").stdout
        assert "upload check OK" in out and "UNEEG training examples" in out and "250 ms" in out
        meta = json.load(open(os.path.join(site, "content", "site.json")))
        assert meta["tolerance_s"] == 0.25
        got = [(x["batch_id"], x.get("stage"), x.get("view"), x.get("training")) for x in meta["batches"]]
        n_s = len(got) - 2
        assert got == [(f"s{i + 1:02d}", 1, None, None) for i in range(n_s)] + [("e01", 2, "both", True)] + \
            [("u01", 3, "subq", None)]
        # the tolerance is remembered; a later run without the option keeps it
        run(os.path.join(HOSTED, "make_site.py"), "--clips", staged, "--study", study, "--labels", labels, "--out", site)
        assert json.load(open(os.path.join(site, "content", "site.json")))["tolerance_s"] == 0.25
        # a planted extra field on a scalp clip, or an expert answer in a scalp batch, fails the check
        p = os.path.join(site, "content", "batches", "s01.json.gz")
        good = open(p, "rb").read()
        obj = json.loads(gzip.decompress(good))
        obj["clips"][0]["ref"] = "yes"
        open(p, "wb").write(gzip.compress(json.dumps(obj).encode(), mtime=0))
        assert any("fields ['ref']" in x for x in check_upload.check(site))
        open(p, "wb").write(good)
        p = os.path.join(site, "content", "batches", "e01.json.gz")
        good = open(p, "rb").read()
        obj = json.loads(gzip.decompress(good))
        obj["clips"][0]["ref_note"] = "from EMU1234"
        open(p, "wb").write(gzip.compress(json.dumps(obj).encode(), mtime=0))
        assert any("EMU subject" in x for x in check_upload.check(site))
        open(p, "wb").write(good)
        assert check_upload.check(site) == []
        if no_node():
            return
        owner = serve_review.load_token(labels, False)
        ec = serve_review.load_rater_tokens(labels, False)["EC"]
        bat = {x[0]: json.loads(gzip.decompress(open(os.path.join(site, "content", "batches", x[0] + ".json.gz"),
                                                        "rb").read())) for x in got}
        dev = DevServer(site, os.path.join(tmp, "db.json"))
        try:
            j = json.loads(dev.req("/api/batches", ec)[1])
            assert j["tolerance_s"] == 0.25
            assert [x.get("locked", False) for x in j["batches"]] == [False] * n_s + [True, True]
            assert j["batches"][-1]["view"] == "subq" and j["batches"][0]["stage_name"] == "10-20 clips"
            assert j["batches"][n_s]["stage_name"] == "UNEEG training examples"
            code, msg, _h = dev.req("/api/batch/e01", ec)
            assert code == 403 and b"opens once" in msg
            assert dev.req("/api/batch/e01", owner)[0] == 200
            for bid, *_ in got[:n_s]:
                body = answers_for(bat[bid], "EC", ["no"] * len(bat[bid]["clips"]))
                assert dev.req(f"/api/labels/{bid}/EC", ec, "PUT", body)[0] == 200
            j = json.loads(dev.req("/api/batches", ec)[1])
            assert [x.get("locked", False) for x in j["batches"]] == [False] * (n_s + 1) + [True]
            assert dev.req("/api/batch/e01", ec)[0] == 200
        finally:
            dev.close()


def test_admin_view_of_a_uneeg_clip_carries_its_10_20():
    # kit 1.8.5 (Svanik, November 10): on UNEEG batches the admin view shows the 10-20 for the same second, the way
    # it shows the UNEEG channels on 10-20 batches; the rater's own batch still holds only the UNEEG channels
    import make_review_batches as mrb
    from test_tools import live_opts
    with tempfile.TemporaryDirectory() as tmp:
        clips, admin, _sq, _a = admin_study(tmp)
        staged = os.path.join(tmp, "staged.csv")
        select_clips.main(["parts", "--scalp", clips, "--examples-n", "0", "--uneeg-n", "4", "--uneeg-pick", "bins",
                           "--out", staged])
        st = serve_review.LiveStore(staged, os.path.join(tmp, "study"), os.path.join(tmp, "labels"),
                                    live_opts(repeat_frac=0, batch_size=4, question="second", admin=admin))
        key = pd.read_csv(os.path.join(tmp, "study", "key.csv"), dtype=str, keep_default_na=False)
        u = json.loads(gzip.decompress(st.admin_bytes("u01")))
        assert len(u["clips"]) == 4
        reader = mrb.READERS["edf"]()
        n_sc = len(mrb.SCALP) + 1
        for item in u["clips"]:
            assert "scalp" in item and "subq" not in item and item["scalp"]["channels"] == mrb.SCALP + ["EKG"]
            row = key[key["clip_id"] == item["id"]].iloc[0]
            want, _bad, _u = mrb.read_clip(reader, row["source_path"], float(row["center_s"]), st.ns, ["EKG"])
            got = mrb.decode(item["scalp"]["data"], n_sc, 2400, item["scalp"]["scale"])
            want = np.asarray(want)
            assert np.abs(got - (want - want.mean(axis=1, keepdims=True))).max() < 0.25
        reader.close()
        s1 = json.loads(gzip.decompress(st.admin_bytes("s01")))
        assert all("scalp" not in x for x in s1["clips"]) and any("subq" in x for x in s1["clips"])
        # the upload check takes the admin 10-20 as signal only, and refuses anything more in it
        site = os.path.join(tmp, "site", "content", "admin")
        os.makedirs(site)
        path = os.path.join(site, "u01.json.gz")
        open(path, "wb").write(st.admin_bytes("u01"))
        assert check_upload.check_admin(path, "content/admin/u01.json.gz") == []
        u["clips"][0]["scalp"]["source_path"] = "x"
        open(path, "wb").write(gzip.compress(json.dumps(u).encode()))
        assert any("10-20 view" in p for p in check_upload.check_admin(path, "content/admin/u01.json.gz"))


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e))
    print(f"{'ALL PASSED' if not fails else str(fails) + ' FAILED'}")
    sys.exit(1 if fails else 0)
