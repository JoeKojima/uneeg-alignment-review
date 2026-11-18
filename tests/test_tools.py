"""Tests for select_clips.py and serve_review.py. Run: python -m pytest -q tests/test_tools.py
(or python tests/test_tools.py). Uses synthetic data only."""
import csv
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT)
import select_clips  # noqa: E402
import serve_review  # noqa: E402


# ---------------------------------------------------------------- select_clips

def fake_seconds(tmp, n_pat=6, seed=3):
    rng = np.random.default_rng(seed)
    frames = []
    for p in range(n_pat):
        n = int(rng.integers(3000, 9000))
        pr = np.clip(rng.beta(0.6, 6, n), 0, 0.999)
        hot = rng.random(n) < (0.08 if p == 0 else 0.004)  # patient 0 has most of the high seconds
        pr[hot] = rng.uniform(0.5, 0.999, hot.sum())
        frames.append(pd.DataFrame({"patient": f"P{p}", "source_path": f"/x/P{p}.edf", "start_s": np.arange(n), "prob": pr}))
    df = pd.concat(frames, ignore_index=True)
    path = os.path.join(tmp, "sec.csv")
    df.to_csv(path, index=False)
    return path, df


def test_weights_add_back_to_cell_counts_and_no_patient_dominates():
    with tempfile.TemporaryDirectory() as tmp:
        path, df = fake_seconds(tmp)
        out = os.path.join(tmp, "clips.csv")
        select_clips.main(["sample", path, "--out", out, "--n", "400", "--pilot", "20"])
        c = pd.read_csv(out)
        assert len(c) == 400
        assert set(["source_path", "center_s", "sampling_weight", "bin", "patient"]) <= set(c.columns)
        # weights per cell equal the cell's candidate count
        for (_b, _p), g in c.groupby(["bin", "patient"]):
            assert abs(g["sampling_weight"].sum() - g["cell_N"].iloc[0]) < 1e-3
        # every clip fits inside its file
        span = df.groupby("source_path")["start_s"].agg(["min", "max"])
        assert (c["center_s"] - 6 >= c["source_path"].map(span["min"])).all()
        assert (c["center_s"] + 6 <= c["source_path"].map(span["max"]) + 1).all()
        # no patient takes more than half of any bin
        share = c.groupby("bin")["patient"].value_counts(normalize=True)
        assert share.max() <= 0.5 + 1e-9
        pilot = pd.read_csv(os.path.join(tmp, "clips_pilot.csv"))
        assert len(pilot) == 20 and pilot["bin"].nunique() == 10
        assert set(pilot["center_s"].astype(str) + pilot["source_path"]) <= set(c["center_s"].astype(str) + c["source_path"])


def test_weighted_estimate_recovers_true_rate():
    """If a clip's true label is a function of its probability, the weighted mean over the sample
    should estimate the rate over all candidates."""
    with tempfile.TemporaryDirectory() as tmp:
        path, df = fake_seconds(tmp, seed=9)
        out = os.path.join(tmp, "clips.csv")
        ests = []
        for seed in range(5):
            select_clips.main(["sample", path, "--out", out, "--n", "600", "--seed", str(seed)])
            c = pd.read_csv(out)
            y = (c["morgoth_prob"] > 0.35).astype(float)
            ests.append((c["sampling_weight"] * y).sum() / c["sampling_weight"].sum())
        span = df.groupby("source_path")["start_s"].agg(["min", "max"])
        keep = (df["start_s"] + 0.5 - 6 >= df["source_path"].map(span["min"])) & (df["start_s"] + 0.5 + 6 <= df["source_path"].map(span["max"]) + 1)
        truth = (df.loc[keep, "prob"] > 0.35).mean()
        assert abs(np.mean(ests) - truth) < 0.01, (np.mean(ests), truth)



def two_scorings(tmp, seed=5):
    """Raw and SD-matched scores for the same seconds; the SD-matched one also fires on 300 seconds the raw
    one scores low."""
    path, df = fake_seconds(tmp, seed=seed)
    rng = np.random.default_rng(seed)
    sd = df.copy()
    flip = rng.choice(np.flatnonzero(df["prob"].values < 0.1), 300, replace=False)
    sd.loc[flip, "prob"] = rng.uniform(0.7, 0.99, 300)
    raw_p, sd_p = os.path.join(tmp, "raw.csv"), os.path.join(tmp, "sd.csv")
    df.assign(detector="cyclops").to_csv(raw_p, index=False)
    sd.assign(detector="cyclops").to_csv(sd_p, index=False)
    return raw_p, sd_p, df, sd, flip


def test_join_puts_scorings_side_by_side():
    with tempfile.TemporaryDirectory() as tmp:
        raw_p, sd_p, df, sd, _ = two_scorings(tmp)
        sd.drop(index=[0, 1, 2]).to_csv(sd_p, index=False)  # three seconds the SD-matched run never scored
        out = os.path.join(tmp, "joined.csv")
        buf = io.StringIO()
        from contextlib import redirect_stdout
        with redirect_stdout(buf):
            select_clips.main(["join", "--out", out, f"cyclops={raw_p}", f"cyclops_sd={sd_p}"])
        j = pd.read_csv(out)
        assert len(j) == len(df) - 3 and "3 not scored by every table" in buf.getvalue()
        assert {"cyclops_prob", "cyclops_sd_prob", "patient", "source_path", "start_s"} <= set(j.columns)
        assert (j["detector"] == "cyclops").all() and "prob" not in j.columns
        m = j.merge(df, on=["source_path", "start_s"])
        assert np.allclose(m["cyclops_prob"], m["prob"])
        # a second that belongs to two patients means the tables were made differently
        sd.assign(patient="PX").to_csv(sd_p, index=False)
        try:
            select_clips.main(["join", "--out", out, f"cyclops={raw_p}", f"cyclops_sd={sd_p}"])
            raise AssertionError("a patient mismatch should be refused")
        except SystemExit as e:
            assert "different patient" in str(e)


def test_strata_follow_the_highest_score_and_every_scoring_stays_unbiased():
    with tempfile.TemporaryDirectory() as tmp:
        raw_p, sd_p, df, sd, flip = two_scorings(tmp)
        joined, out = os.path.join(tmp, "joined.csv"), os.path.join(tmp, "clips.csv")
        select_clips.main(["join", "--out", joined, f"cyclops={raw_p}", f"cyclops_sd={sd_p}"])
        span = df.groupby("source_path")["start_s"].agg(["min", "max"])
        keep = (df["start_s"] + 0.5 - 6 >= df["source_path"].map(span["min"])) & \
               (df["start_s"] + 0.5 + 6 <= df["source_path"].map(span["max"]) + 1)
        truth = {"raw": (df.loc[keep, "prob"] >= 0.5).mean(), "sd": (sd.loc[keep, "prob"] >= 0.5).mean()}
        ests, hits_multi = {"raw": [], "sd": []}, []
        flipped = set(zip(df.loc[flip, "source_path"], df.loc[flip, "start_s"]))
        for seed in range(6):
            select_clips.main(["sample", joined, "--prob-cols", "cyclops_prob,cyclops_sd_prob", "--out", out,
                               "--n", "600", "--seed", str(seed)])
            c = pd.read_csv(out)
            assert np.allclose(c["strata_prob"], c[["cyclops_prob", "cyclops_sd_prob"]].max(axis=1), atol=1e-4)
            lo = c["bin"].str.split("-").str[0].astype(float) / 100
            hi = c["bin"].str.split("-").str[1].str.rstrip("%").astype(float) / 100
            assert ((lo <= c["strata_prob"] + 1e-4) & ((c["strata_prob"] < hi + 1e-4) | (hi == 1))).all()
            w = c["sampling_weight"]
            ests["raw"].append((w * (c["cyclops_prob"] >= 0.5)).sum() / w.sum())
            ests["sd"].append((w * (c["cyclops_sd_prob"] >= 0.5)).sum() / w.sum())
            hits_multi.append(sum((a, b) in flipped for a, b in zip(c["source_path"], c["start_s"])))
        for k in ("raw", "sd"):
            assert abs(np.mean(ests[k]) - truth[k]) < 0.01, (k, np.mean(ests[k]), truth[k])
        # stratifying on the raw score alone leaves the seconds only the SD-matched scoring flags thinly sampled
        hits_raw = []
        for seed in range(6):
            select_clips.main(["sample", joined, "--prob-col", "cyclops_prob", "--out", out, "--n", "600",
                               "--seed", str(seed)])
            c = pd.read_csv(out)
            hits_raw.append(sum((a, b) in flipped for a, b in zip(c["source_path"], c["start_s"])))
        assert np.mean(hits_multi) > 5 * max(np.mean(hits_raw), 1), (hits_multi, hits_raw)

def test_min_gap_keeps_clips_apart():
    with tempfile.TemporaryDirectory() as tmp:
        path, _ = fake_seconds(tmp)
        out = os.path.join(tmp, "clips.csv")
        select_clips.main(["sample", path, "--out", out, "--n", "400", "--min-gap-s", "12"])
        c = pd.read_csv(out)
        for _, g in c.groupby("source_path"):
            assert (np.diff(np.sort(g["center_s"].values)) >= 12).all()
        # a patient can lose every clip in a sparse bin to the gap rule; the bin's other patients then stand
        # for it, so each cell's weights are its candidates times one factor shared across the bin
        for _b, gb in c.groupby("bin"):
            ratio = gb.groupby("patient")[["sampling_weight", "cell_N"]].apply(
                lambda g: g["sampling_weight"].sum() / g["cell_N"].iloc[0])
            assert ratio.max() - ratio.min() < 1e-5 and ratio.min() > 1 - 1e-5  # weights are written to 6 decimals


def test_clips_stay_inside_unbroken_stretches():
    with tempfile.TemporaryDirectory() as tmp:
        secs = list(range(0, 60)) + list(range(100, 160))  # an EDF+D file with a gap from 60 to 100 s
        df = pd.DataFrame({"patient": "P", "source_path": "/x/d.edf", "start_s": secs, "prob": 0.95})
        path = os.path.join(tmp, "sec.csv")
        df.to_csv(path, index=False)
        out = os.path.join(tmp, "clips.csv")
        select_clips.main(["sample", path, "--out", out, "--n", "200", "--bins", "0,0.5,1", "--max-patient-frac", "1"])
        c = pd.read_csv(out)["center_s"]
        assert len(c) == 2 * (60 - 12)  # every second whose 12 s clip fits in its stretch
        assert not ((c + 6 > 60) & (c - 6 < 100)).any()


def test_share_draws_slots_at_random_when_clips_are_fewer():
    """50 clips for 436 patients: each patient has the same chance, 50/436, and it is not the first 50 by ID."""
    hits = np.zeros(436)
    firsts = set()
    for seed in range(3000):
        got, p = select_clips.share(50, np.full(436, 25), "equal", rng=np.random.default_rng(seed))
        assert got.sum() == 50 and got.max() == 1 and np.allclose(p, 50 / 436)
        hits += got
        firsts.add(tuple(np.flatnonzero(got)[:3]))
    assert len(firsts) > 100 and np.abs(hits / 3000 - 50 / 436).max() < 0.035
    got, p = select_clips.share(150, np.full(100, 75), "equal", rng=np.random.default_rng(0))
    assert got.min() == 1 and got.max() == 2 and (p == 1).all()  # every patient has a clip: nothing to draw


def test_sample_with_more_patients_than_clips_estimates_all_patients():
    """With 20 clips for 60 patients the weights still add up to every candidate, and the weighted share
    estimates the share over all patients, not over the first patients by ID."""
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(4)
        frames = []
        for i in range(60):
            n = int(rng.integers(60, 400))
            frames.append(pd.DataFrame({"patient": f"P{i:02d}", "source_path": f"/x/P{i:02d}.edf",
                                        "start_s": np.arange(n), "prob": 0.1}))
        df = pd.concat(frames, ignore_index=True)
        path, out = os.path.join(tmp, "sec.csv"), os.path.join(tmp, "clips.csv")
        df.to_csv(path, index=False)
        hot = {f"P{i:02d}" for i in range(20)}  # the first 20 by ID have IEDs, the rest none
        n_cand = int(df.groupby("patient").size().sub(12).sum())  # 12 s clips: 12 edge seconds per file
        truth = df.groupby("patient").size().sub(12)[sorted(hot)].sum() / n_cand
        ests, drawn = [], set()
        for seed in range(40):
            buf = io.StringIO()
            from contextlib import redirect_stdout
            with redirect_stdout(buf):
                select_clips.main(["sample", path, "--out", out, "--n", "20", "--bins", "0,1", "--seed", str(seed)])
            c = pd.read_csv(out)
            assert len(c) == 20 and c["patient"].nunique() == 20
            assert f"weights add to {n_cand} of {n_cand} candidates" in buf.getvalue()
            assert "drawn at random" in buf.getvalue() and "WARNING" not in buf.getvalue()
            assert np.allclose(c["cell_p"], 20 / 60)
            y = c["patient"].isin(hot).astype(float)
            ests.append((c["sampling_weight"] * y).sum() / c["sampling_weight"].sum())
            drawn |= set(c["patient"])
        assert len(drawn) == 60 and abs(np.mean(ests) - truth) < 0.05, (np.mean(ests), truth)


def test_column_names_can_be_mapped():
    with tempfile.TemporaryDirectory() as tmp:
        path, df = fake_seconds(tmp, n_pat=3)
        df.rename(columns={"patient": "pt", "source_path": "file", "start_s": "t", "prob": "p"}).to_csv(path, index=False)
        out = os.path.join(tmp, "clips.csv")
        select_clips.main(["sample", path, "--out", out, "--n", "50", "--patient-col", "pt", "--path-col", "file",
                           "--start-col", "t", "--prob-col", "p"])
        assert len(pd.read_csv(out)) == 50


def test_report_recovers_simulated_prevalence():
    import report_labels
    with tempfile.TemporaryDirectory() as tmp:
        merged, meta = report_labels.simulate(tmp, seed=5, n=600, shared=120)
        a = report_labels.analyze(merged)
        lo, hi = a["prev_ci"]
        assert lo <= meta["true_prev"] <= hi, (lo, meta["true_prev"], hi)
        # four scorings judged on one set of labels; all four raters on the shared clips, one on each other clip
        assert [x["name"] for x in a["scorings"]] == ["MORGOTH", "Cyclops", "Cyclops SD-matched", "SpikeNet1"]
        assert all(len(x["curve"]) == 19 and not math.isnan(x["at"]["ppv"]) and x["ci"]["sens"][0] <= x["at"]["sens"]
                   for x in a["scorings"])
        assert a["agree_all"]["n_clips"] == 120 and a["n_shared"] == 120 and a["n_clips"] == 600
        assert all(r["vs_n"] > 100 for r in a["rater_stats"]) and a["n_repeats"] == 0
        head, body = report_labels.render(a, "t", "s", simulated=meta)
        assert "/sim/" not in head + body  # no file paths in the page
        assert "Where the field maximum sits" not in body and "Retest" not in body and ">nan<" not in body.lower()


# ---------------------------------------------------------------- serve_review

def make_batches(tmp):
    """Two tiny batches (rater and research) in the real format, plus a key.csv that must never be served."""
    bdir = os.path.join(tmp, "batches")
    os.makedirs(bdir)
    for bid, mode in (("b01", "rater"), ("r01", "research")):
        obj = {"format": "ied-review-batch", "version": 1, "batch_id": bid, "mode": mode, "fs": 10, "clip_s": 1,
               "channels": ["Fp1"], "clips": [{"id": f"{bid}c{i}", "event_s": 0.5, "scale": [1.0], "data": "AAAA"} for i in range(3)]}
        with open(os.path.join(bdir, bid + ".json"), "w") as f:
            json.dump(obj, f)
    with open(os.path.join(bdir, "key.csv"), "w") as f:
        f.write("clip_id,patient\nb01c0,HUP999\n")
    viewer = os.path.join(tmp, "viewer.html")
    with open(viewer, "w") as f:
        f.write("<!doctype html><title>viewer</title>")
    return bdir, viewer


class Server:
    def __init__(self, tmp, allow_research=False, raters=(), store=None, admin_raters=(), tolerance_ms=0):
        bdir, viewer = make_batches(tmp)
        self.labels = os.path.join(tmp, "labels")
        os.makedirs(self.labels, mode=0o700, exist_ok=True)
        self.store = store or serve_review.Store(bdir, self.labels, allow_research)
        self.token = serve_review.load_token(self.labels, False)
        self.raters = serve_review.load_rater_tokens(self.labels, False, raters)
        tokens = {"": self.token, **self.raters}
        from http.server import ThreadingHTTPServer
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve_review.make_handler(self.store, tokens, viewer, 0))
        self.port = self.httpd.server_address[1]
        self.httpd.RequestHandlerClass = serve_review.make_handler(self.store, tokens, viewer, self.port,
                                                                   admin_raters=admin_raters, tolerance_ms=tolerance_ms)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def req(self, path, method="GET", body=None, token=True, host=None, who=None):
        h = {}
        if who:
            h["X-Review-Token"] = self.raters[who]
        elif token:
            h["X-Review-Token"] = self.token
        if host:
            h["Host"] = host
        r = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=body, method=method, headers=h)
        try:
            with urllib.request.urlopen(r) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def labels_csv(batch_id, rater, decisions):
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=serve_review.EXPORT_COLUMNS)
    w.writeheader()
    for i, d in enumerate(decisions):
        w.writerow({"batch_id": batch_id, "clip_id": f"{batch_id}c{i}", "rater": rater, "order_index": i + 1, "decision": d,
                    "generalized": 0, "artifact": 0, "seconds_viewed": 3, "labeled_at": "2026-10-19T00:00:00Z"})
    return out.getvalue().encode()


def test_export_columns_match_viewer():
    core = open(os.path.join(KIT, "src", "core.js"), encoding="utf-8").read()
    m = re.search(r"EXPORT_COLUMNS = \[(.*?)\];", core, re.S)
    assert re.findall(r"'([a-z_]+)'", m.group(1)) == serve_review.EXPORT_COLUMNS


def test_server_requires_token_and_local_host():
    with tempfile.TemporaryDirectory() as tmp:
        s = Server(tmp)
        try:
            assert s.req("/", token=False)[0] == 403
            assert s.req("/api/batches", token=False)[0] == 403
            assert s.req(f"/?token={s.token}", token=False)[0] == 200
            assert s.req("/api/batches", host="evil.example:80")[0] == 403  # DNS rebinding
            assert s.req("/?token=wrong" + "x" * 20, token=False)[0] == 403
            assert oct(os.stat(os.path.join(s.labels, ".token")).st_mode & 0o777) == "0o600"
        finally:
            s.close()


def test_server_serves_only_blinded_batches_and_never_the_key():
    with tempfile.TemporaryDirectory() as tmp:
        s = Server(tmp)
        try:
            code, body = s.req("/api/batches")
            ids = [b["batch_id"] for b in json.loads(body)["batches"]]
            assert ids == ["b01"]  # research batch hidden without --allow-research
            assert s.req("/api/batch/b01")[0] == 200
            assert s.req("/api/batch/r01")[0] == 404
            for p in ("/key.csv", "/api/batch/key.csv", "/api/batch/..%2Fkey.csv", "/api/batch/../key.csv",
                      "/batches/key.csv", "/api/labels/b01/..%2F..%2Fkey"):
                code, body = s.req(p)
                assert code == 404 and b"HUP999" not in body, p
        finally:
            s.close()
        s = Server(tempfile.mkdtemp(dir=tmp), allow_research=True)
        try:
            ids = [b["batch_id"] for b in json.loads(s.req("/api/batches")[1])["batches"]]
            assert ids == ["b01", "r01"]
        finally:
            s.close()


def test_server_saves_validates_and_returns_labels():
    with tempfile.TemporaryDirectory() as tmp:
        s = Server(tmp)
        try:
            assert s.req("/api/labels/b01/EC")[0] == 404
            code, body = s.req("/api/labels/b01/EC", "PUT", labels_csv("b01", "EC", ["yes", "", "no"]))
            assert code == 200 and json.loads(body)["answered"] == 2
            path = os.path.join(s.labels, "labels_b01_EC.csv")
            assert oct(os.stat(path).st_mode & 0o777) == "0o600"
            code, body = s.req("/api/labels/b01/ec")
            assert code == 200 and body.decode().count("\n") == 4
            assert json.loads(s.req("/api/batches")[1])["batches"][0]["progress"] == {"EC": 2}
            # second save keeps a history copy of the first
            s.req("/api/labels/b01/EC", "PUT", labels_csv("b01", "EC", ["yes", "yes", "no"]))
            assert len(os.listdir(os.path.join(s.labels, "history"))) == 1
            # rejected: wrong rater in rows, unknown clip, bad decision, other batch, bad initials, wrong columns
            bad = [
                ("/api/labels/b01/EC", labels_csv("b01", "XX", ["yes"])),
                ("/api/labels/b01/EC", labels_csv("b01", "EC", ["yes"]).replace(b"b01c0", b"zzz")),
                ("/api/labels/b01/EC", labels_csv("b01", "EC", ["maybe"])),
                ("/api/labels/r01/EC", labels_csv("r01", "EC", ["yes"])),
                ("/api/labels/b01/E.C", labels_csv("b01", "E.C", ["yes"])),
                ("/api/labels/b01/EC", b"clip_id,decision\nb01c0,yes\n"),
            ]
            for path_, body in bad:
                code, _ = s.req(path_, "PUT", body)
                assert code in (400, 404), (path_, code)
            assert sum(1 for r in csv.DictReader(open(path)) if r["decision"]) == 3  # last good save intact
        finally:
            s.close()


def test_status_prints_progress():
    with tempfile.TemporaryDirectory() as tmp:
        s = Server(tmp)
        try:
            s.req("/api/labels/b01/EC", "PUT", labels_csv("b01", "EC", ["yes", "no", ""]))
        finally:
            s.close()
        out = subprocess.run([sys.executable, os.path.join(KIT, "serve_review.py"), os.path.join(tmp, "batches"),
                              "--labels", s.labels, "--viewer", os.path.join(tmp, "viewer.html"), "--status"],
                             capture_output=True, text=True, check=True).stdout
        assert re.search(r"b01\s+3\s+everyone\s+2", out), out


def test_rater_links_sign_in_one_rater_each():
    with tempfile.TemporaryDirectory() as tmp:
        s = Server(tmp, raters=["ec", "JK"])
        try:
            assert sorted(s.raters) == ["EC", "JK"]
            assert oct(os.stat(os.path.join(s.labels, ".raters.json")).st_mode & 0o777) == "0o600"
            assert s.req("/api/labels/b01/EC", "PUT", labels_csv("b01", "EC", ["yes", "no", ""]), who="EC")[0] == 200
            assert s.req("/api/labels/b01/JK", "PUT", labels_csv("b01", "JK", ["no"]), who="JK")[0] == 200
            # a rater's link cannot read or write another rater's labels
            assert s.req("/api/labels/b01/JK", who="EC")[0] == 403
            assert s.req("/api/labels/b01/JK", "PUT", labels_csv("b01", "JK", ["yes"]), who="EC")[0] == 403
            assert s.req("/api/labels/b01/EC", who="EC")[0] == 200
            # the batch list says who the link belongs to and shows only their progress
            j = json.loads(s.req("/api/batches", who="JK")[1])
            assert j["rater"] == "JK" and j["batches"][0]["progress"] == {"JK": 1}
            j = json.loads(s.req("/api/batches")[1])
            assert j["rater"] is None and j["batches"][0]["progress"] == {"EC": 2, "JK": 1}
            assert "rater_links" not in json.loads(s.req("/api/batches", who="EC")[1])
            # 1.7.7: the owner's link is told which initials have their own link and cannot save under them
            assert j["rater_links"] == ["EC", "JK"]
            code, msg = s.req("/api/labels/b01/EC", "PUT", labels_csv("b01", "EC", ["no", "no", "no"]))
            assert code == 403 and b"own link" in msg
            assert s.req("/api/labels/b01/EC", who="EC")[1] == labels_csv("b01", "EC", ["yes", "no", ""])
            assert s.req("/api/labels/b01/SJ", "PUT", labels_csv("b01", "SJ", ["no"]))[0] == 200
            # and pairwise the other way: a rater's link cannot read or save under the owner's initials either
            assert s.req("/api/labels/b01/SJ", who="EC")[0] == 403
            assert s.req("/api/labels/b01/SJ", "PUT", labels_csv("b01", "SJ", ["yes"]), who="JK")[0] == 403
        finally:
            s.close()
        # adding keeps old links; revoking and --new-token turn links off
        r1 = serve_review.load_rater_tokens(s.labels, False, ["EC", "SJ"])
        assert r1["EC"] == s.raters["EC"] and "SJ" in r1
        r2 = serve_review.load_rater_tokens(s.labels, False, [], ["JK"])
        assert "JK" not in r2 and r2["EC"] == s.raters["EC"]
        r3 = serve_review.load_rater_tokens(s.labels, True)
        assert sorted(r3) == ["EC", "SJ"] and r3["EC"] != s.raters["EC"]


# ---------------------------------------------------------------- EDF+D reader

sys.path.insert(0, os.path.join(KIT, "tests"))
LABS19 = ["EEG FP1-REF", "EEG F3-REF", "EEG C3-REF", "EEG P3-REF", "EEG F7-REF", "EEG T7-REF", "EEG P7-REF", "EEG O1-REF",
          "EEG FZ-REF", "EEG CZ-REF", "EEG PZ-REF", "EEG FP2-REF", "EEG F4-REF", "EEG C4-REF", "EEG P4-REF",
          "EEG F8-REF", "EEG T8-REF", "EEG P8-REF", "EEG O2-REF", "ECG EKG-REF"]


def three_segment_edf(path, fs=200, spike_at=None):
    """EDF+D with segments 0-30 s, 40-60 s and 3660-3690 s; optional spike on T3 at a recording-clock time."""
    from edf_writer import write_edf
    rng = np.random.default_rng(4)
    segs = []
    for onset, dur in ((0, 30), (40, 20), (3660, 30)):
        x = rng.standard_normal((len(LABS19), fs * dur)) * 15
        if spike_at is not None and onset <= spike_at < onset + dur:
            k = int(round((spike_at - onset) * fs))
            x[5, k - 8:k + 8] -= 400 * np.hanning(16)
        segs.append((onset, x))
    write_edf(path, LABS19, fs, segs)
    return segs


def test_edfplus_finds_segments_and_refuses_gaps():
    from edfplus import EDFPlus
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "d.edf")
        segs = three_segment_edf(path)
        f = EDFPlus(path)
        assert f.discontinuous and [(s[2], s[3]) for s in f.segments] == [(0, 30), (40, 60), (3660, 3690)]
        got = dict((lab, x) for lab, _fs, x, _u in f.read(45, 50))
        want = segs[1][1][0, 5 * 200:10 * 200]
        assert len(got["EEG FP1-REF"]) == 1000 and np.max(np.abs(got["EEG FP1-REF"] - want)) < 0.2  # one 16-bit step
        for t0, t1 in ((25, 35), (55, 65), (3685, 3695), (-1, 2)):
            try:
                f.read(t0, t1)
                raise AssertionError(f"window {t0}-{t1} should be refused")
            except ValueError as e:
                assert "contiguous segment" in str(e)


def test_edfplus_reads_a_files_record_onsets_once_per_process():
    import edfplus
    from edfplus import EDFPlus
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "d.edf")
        three_segment_edf(path)
        calls = []
        orig = EDFPlus._parse_onsets

        def counted(self, a, n):
            calls.append(self.path)
            return orig(self, a, n)
        EDFPlus._parse_onsets = counted
        try:
            edfplus._ONSETS.clear()
            first = EDFPlus(path)
            second = EDFPlus(path)
            assert len(calls) == 1  # packing a study opens a file once per clip; the onsets are read once
            assert first.segments == second.segments
            second.onsets[0] = -99.0  # each reader gets its own copy
            assert EDFPlus(path).onsets[0] != -99.0 and len(calls) == 1
            # a changed file is read again
            with open(path, "r+b") as f:
                f.seek(0, 2)
                f.write(b"")
            os.utime(path, ns=(1, 1))
            EDFPlus(path)
            assert len(calls) == 2
        finally:
            EDFPlus._parse_onsets = orig
            edfplus._ONSETS.clear()


def test_edfplus_gap_tolerance_and_units():
    from edf_writer import write_edf
    from edfplus import EDFPlus
    with tempfile.TemporaryDirectory() as tmp:
        x = np.random.default_rng(1).standard_normal((2, 200 * 20)) * 20
        jitter = np.zeros(20)
        jitter[::2] = 0.0004  # clock wobble under 1 ms stays one segment
        p = write_edf(os.path.join(tmp, "a.edf"), ["EEG C3-REF", "EEG C4-REF"], 200, [(0, x)], kind="EDF+D", onset_jitter=jitter)
        assert len(EDFPlus(p).segments) == 1
        jitter[10:] += 0.005  # a 5 ms jump splits it
        p = write_edf(os.path.join(tmp, "b.edf"), ["EEG C3-REF", "EEG C4-REF"], 200, [(0, x)], kind="EDF+D", onset_jitter=jitter)
        assert len(EDFPlus(p).segments) == 2
        # volts come back as microvolts
        p = write_edf(os.path.join(tmp, "v.edf"), ["EEG C3-REF", "EEG C4-REF"], 200, [(0, x * 1e-6)], unit="V",
                      phys=(-0.005, 0.005))
        got = EDFPlus(p).read(2, 4)[0][2]
        assert np.max(np.abs(got - x[0, 400:800])) < 0.2


def test_edfplus_matches_pyedflib_on_contiguous_files():
    import pyedflib
    from edf_writer import write_edf
    from edfplus import EDFPlus
    with tempfile.TemporaryDirectory() as tmp:
        x = np.random.default_rng(2).standard_normal((3, 256 * 10)) * 30
        p = write_edf(os.path.join(tmp, "c.edf"), ["EEG F3-REF", "EEG F4-REF", "EEG CZ-REF"], 256, [(0, x)])
        mine = EDFPlus(p).read(3, 7)
        with pyedflib.EdfReader(p) as r:
            for i, (_lab, fs, y, _u) in enumerate(mine):
                assert fs == 256 and np.allclose(y, r.readSignal(i, start=3 * 256, n=4 * 256), atol=1e-9)


def test_pack_reads_edf_plus_d_on_the_recording_clock():
    import make_review_batches as mrb
    with tempfile.TemporaryDirectory() as tmp:
        edf = os.path.join(tmp, "d.edf")
        three_segment_edf(edf, spike_at=3675.0)
        clips = os.path.join(tmp, "clips.csv")
        pd.DataFrame({"source_path": [edf, edf], "center_s": [3675.0, 35.0]}).to_csv(clips, index=False)
        out = os.path.join(tmp, "b")
        res = subprocess.run([sys.executable, os.path.join(KIT, "make_review_batches.py"), "pack", clips, "--out", out],
                             capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
        assert "1 clip(s) could not be read" in res.stdout and "contiguous segment" in res.stdout
        obj = mrb.read_batch(os.path.join(out, "b01.json.gz"))
        c = obj["clips"][0]
        rows = mrb.decode(c["data"], len(obj["channels"]), int(obj["clip_s"] * obj["fs"]), c["scale"])
        t3 = rows[obj["channels"].index("T3")]
        assert abs(np.argmin(t3) / obj["fs"] - obj["event_s"]) < 0.05  # spike sits at the clip center



def test_edf_c_reads_on_the_file_clock_whatever_its_first_onset():
    """An EDF+C whose first record onset is not 0 (pyedflib allows under 1 s) is read on pyedflib's clock
    (0 = first sample) everywhere."""
    import pyedflib
    from edf_writer import write_edf
    from edfplus import EDFPlus
    with tempfile.TemporaryDirectory() as tmp:
        x = np.random.default_rng(4).standard_normal((len(LABS19), 200 * 40)) * 30
        edf = os.path.join(tmp, "edf")
        os.makedirs(edf)
        p = write_edf(os.path.join(edf, "EMU9911_iic_0.edf"), LABS19, 200, [(0, x)], start_offset=0.5)
        e = EDFPlus(p)
        assert e.segments[0][2:] == (0.0, 40.0) and len(e.segments) == 1
        mine = e.read(10, 22)
        e.close()
        with pyedflib.EdfReader(p) as r:
            assert np.allclose(mine[0][2], r.readSignal(0, start=10 * 200, n=12 * 200), atol=1e-9)
        sec, clips = os.path.join(tmp, "sec.csv"), os.path.join(tmp, "clips.csv")
        select_clips.main(["from-edf", "--clip-root", edf, "--out", sec])
        d = pd.read_csv(sec)
        assert d["start_s"].min() == 0 and d["start_s"].max() == 39
        select_clips.main(["sample", sec, "--out", clips, "--n", "2", "--min-gap-s", "12", "--max-patient-frac", "1"])
        st = serve_review.LiveStore(clips, os.path.join(tmp, "study"), os.path.join(tmp, "labels"), live_opts(repeat_frac=0))
        assert st.plan["left_out"] == [] and st.batch_bytes(st.plan["batches"][0]["batch_id"])


def test_onsets_clock_reads_edf_c_on_its_record_onsets_end_to_end():
    """The UNEEG runs time every EDF+ file by its record onsets (segio), EDF+C included. With --edf-clock onsets
    the tables say so, join refuses to mix clocks, and the clip table makes the server read on that clock."""
    from edf_writer import write_edf
    from edfplus import EDFPlus
    import make_review_batches as mrb
    with tempfile.TemporaryDirectory() as tmp:
        x = np.random.default_rng(6).standard_normal((len(LABS19), 200 * 40)) * 30
        root = os.path.join(tmp, "uneeg", "US09-02", "EEG")
        os.makedirs(root)
        p = write_edf(os.path.join(root, "US09-02_EMU_SCALP_1.edf"), LABS19, 200, [(0, x)], start_offset=0.75)
        e = EDFPlus(p, clock="onsets")
        assert len(e.segments) == 1 and e.segments[0][2:] == (0.75, 40.75)
        on = e.read(10.75, 22.75)
        e.close()
        e = EDFPlus(p)
        assert np.allclose(on[0][2], e.read(10, 22)[0][2])  # the same samples, 0.75 s later on the onset clock
        e.close()
        # a run timed by onsets: its last second only fits on the onsets clock
        run = os.path.join(tmp, "run")
        os.makedirs(run)
        pd.DataFrame({"path": [os.path.relpath(p, os.path.join(tmp, "uneeg"))] * 40, "patient": "US09-02",
                      "start_s": np.arange(40) + 0.75, "prob": np.linspace(0, 1, 40).round(3)}
                     ).to_csv(os.path.join(run, "t.csv"), index=False)
        flags = ["--time-col", "start_s", "--time-is", "start", "--score-col", "prob", "--path-col", "path",
                 "--patient-col", "patient", "--clip-root", os.path.join(tmp, "uneeg")]
        sam, ons = os.path.join(tmp, "sam.csv"), os.path.join(tmp, "ons.csv")
        select_clips.main(["from-runs", run, *flags, "--out", sam])
        select_clips.main(["from-runs", run, *flags, "--out", ons, "--edf-clock", "onsets", "--detector", "cyclops"])
        a, b = pd.read_csv(sam), pd.read_csv(ons)
        assert len(a) == 39 and len(b) == 40 and (b["edf_clock"] == "onsets").all() and (a["edf_clock"] == "samples").all()
        assert set(b["clip"]) == {"US09-02_EMU_SCALP_1"}  # named by its file when the run has no clip column
        try:
            select_clips.main(["join", "--out", os.path.join(tmp, "j.csv"), f"morgoth={sam}", f"cyclops={ons}"])
            raise AssertionError("tables on different clocks should be refused")
        except SystemExit as err:
            assert "different EDF clocks" in str(err)
        b2 = b.assign(prob=1 - b["prob"])
        b2.to_csv(os.path.join(tmp, "ons2.csv"), index=False)
        j, clips = os.path.join(tmp, "j.csv"), os.path.join(tmp, "clips.csv")
        select_clips.main(["join", "--out", j, f"cyclops={ons}", f"spikenet1={os.path.join(tmp, 'ons2.csv')}"])
        select_clips.main(["sample", j, "--out", clips, "--n", "2", "--min-gap-s", "12", "--max-patient-frac", "1",
                           "--prob-cols", "cyclops_prob,spikenet1_prob", "--bins", "0,1"])
        c = pd.read_csv(clips)
        assert (c["edf_clock"] == "onsets").all()
        # the server reads each clip on the onsets clock
        from types import SimpleNamespace
        ns = SimpleNamespace(fs=200.0, clip_s=12.0)
        reader = mrb.EDFReader()
        for _, r in c.iterrows():
            rows, _bad, _u = mrb.read_clip(reader, r["source_path"], float(r["center_s"]), ns, [], r["edf_clock"])
            k = int(round((float(r["center_s"]) - 6 - 0.75) * 200))
            assert np.allclose(rows[0], x[0, k:k + 2400], atol=0.5)  # Fp1, from the samples the onsets point at
        reader.close()
        st = serve_review.LiveStore(clips, os.path.join(tmp, "study"), os.path.join(tmp, "labels"), live_opts(repeat_frac=0))
        assert st.plan["left_out"] == [] and st.batch_bytes(st.plan["batches"][0]["batch_id"])


def test_edfplus_leaves_out_backward_and_unstamped_records():
    """A last record stamped back at 0 (seen in one recording) or a tail of records stamped 0 (seen in another) is not part of the
    recording: the reader leaves it out the way segio does, so the file still ends at its last real record."""
    from edf_writer import write_edf
    from edfplus import EDFPlus
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(8)
        segs = [(0.0, rng.standard_normal((2, 200 * 20)) * 20), (3600.0, rng.standard_normal((2, 200 * 24)) * 20)]
        n_rec = 10 + 12  # 2 s records
        for tail, name in ((1, "back.edf"), (3, "tail.edf")):
            segs_t = segs + [(3624.0, rng.standard_normal((2, 200 * 2 * tail)) * 20)]
            jitter = np.zeros(n_rec + tail)
            jitter[n_rec:] = -(3624.0 + 2.0 * np.arange(tail))  # every tail record stamped 0
            p = write_edf(os.path.join(tmp, name), ["EEG C3-REF", "EEG C4-REF"], 200, segs_t, record_s=2.0,
                          kind="EDF+D", onset_jitter=jitter)
            e = EDFPlus(p)
            assert e.dropped == tail and [s[2:] for s in e.segments] == [(0.0, 20.0), (3600.0, 3624.0)]
            assert e.end_s == 3624.0
            assert np.allclose(e.read(3610, 3622)[0][2], segs[1][1][0, 200 * 10:200 * 22], atol=0.2)
            e.close()
            assert select_clips._edf_end_s(p) == 3624.0


def test_reader_falls_back_for_a_short_last_record():
    """pyedflib refuses a contiguous file whose last record is cut short; the server reads it with edfplus."""
    from edf_writer import write_edf
    import make_review_batches as mrb
    with tempfile.TemporaryDirectory() as tmp:
        x = np.random.default_rng(3).standard_normal((len(LABS19), 200 * 30)) * 30
        p = write_edf(os.path.join(tmp, "short.edf"), LABS19, 200, [(0, x)], kind="EDF")
        with open(p, "r+b") as f:
            f.truncate(os.path.getsize(p) - 100)
        reader = mrb.EDFReader()
        sig = reader.read(p, 10, 22)
        reader.close()
        assert np.allclose(sig[0][2], x[0, 2000:4400], atol=0.2)


def test_clip_list_paths_resolve_under_the_clip_root():
    """A list of paths relative to --clip-root picks those exact files, even when another file shares the stem."""
    from edf_writer import write_edf
    with tempfile.TemporaryDirectory() as tmp:
        x = np.random.default_rng(5).standard_normal((len(LABS19), 200 * 20)) * 30
        for sub in ("a", "b"):
            os.makedirs(os.path.join(tmp, sub))
            write_edf(os.path.join(tmp, sub, "EMU9921_iic_0.edf"), LABS19, 200, [(0, x)])
        lst = os.path.join(tmp, "list.txt")
        open(lst, "w").write("a/EMU9921_iic_0.edf\n")
        from types import SimpleNamespace
        args = SimpleNamespace(clip_root=tmp, clip_list=lst, list_col=None, pattern="*")
        assert select_clips._pick_files(args) == [os.path.join(tmp, "a", "EMU9921_iic_0.edf")]
        open(lst, "w").write("EMU9921_iic_0\n")  # a bare stem matching two files stays ambiguous
        assert select_clips._pick_files(args) == []



def test_noisy_electrode_is_drawn_dashed_but_a_large_spike_is_not():
    """A loose contact (broadband noise far above the other electrodes) goes in the clip's bad list, so the viewer
    draws it dashed instead of letting it scribble over the page; a 300 µV spike on one electrode does not."""
    import make_review_batches as mrb
    fs = 200.0
    rng = np.random.default_rng(8)
    rows = mrb.synth_clip(rng, "spike", "T3", 300.0, fs, 12, 0.0)[:19]
    names = mrb.SCALP
    assert mrb.noisy_channels(rows, names, fs) == []
    loose = rows.copy()
    loose[names.index("Pz")] += 400 * rng.standard_normal(rows.shape[1]) + 600 * np.sin(2 * np.pi * 60 * np.arange(rows.shape[1]) / fs)
    assert mrb.noisy_channels(loose, names, fs) == ["Pz"]
    half_stuck = [(n, fs, y, "uV") for n, y in zip(names, rows)]  # every other sample of Pz stuck at zero
    pz = rows[names.index("Pz")].copy()
    pz[1::2] = 0.0
    half_stuck[names.index("Pz")] = ("Pz", fs, pz, "uV")
    assert mrb.assemble(half_stuck, fs, rows.shape[1], [])[2] == ["Pz"]
    mains_only = rows.copy()  # mains alone is what the notch is for, not a bad contact
    mains_only[names.index("Pz")] += 600 * np.sin(2 * np.pi * 60 * np.arange(rows.shape[1]) / fs)
    assert mrb.noisy_channels(mains_only, names, fs) == []


def test_batches_start_with_the_notch_on_unless_told_otherwise():
    import make_review_batches as mrb
    with tempfile.TemporaryDirectory() as tmp:
        for flag, want in (([], True), (["--notch", "off"], False)):
            out = os.path.join(tmp, "n" + str(want))
            mrb.main(["synthetic", "--out", out, "--n", "4", "--batch-size", "4", *flag])
            assert mrb.read_batch(os.path.join(out, "b01.json.gz"))["display"] == {"notch": want}
        st = serve_review.LiveStore(make_live(tmp), os.path.join(tmp, "study"), os.path.join(tmp, "labels"),
                                    live_opts(notch="off"))
        import gzip
        raw = st.batch_bytes(st.plan["batches"][0]["batch_id"])
        assert json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)["display"] == {"notch": False}


# ---------------------------------------------------------------- run outputs -> per-second table

def fake_runs(tmp):
    """Fake EDF clips, a Cyclops-style folder (one table per clip, two conditions), a MORGOTH-style long table
    with window centres, and an admissions table where EMU0001 and EMU0003 are one patient."""
    from edf_writer import write_edf
    rng = np.random.default_rng(9)
    edf = os.path.join(tmp, "edf", "nested")
    os.makedirs(edf)
    names = ["EMU0001_iic_0", "EMU0001_iic_1", "EMU0002_iic_0", "EMU0003_iic_0", "EMU0099_iic_0"]
    for n in names:
        write_edf(os.path.join(edf, n + ".edf"), ["EEG C3-REF", "EEG C4-REF"], 200,
                  [(0, rng.standard_normal((2, 200 * 60)) * 20)])
    cyc = os.path.join(tmp, "cyclops")
    os.makedirs(cyc)
    for n in names:
        rows = [{"second": s, "condition": c, "max_score": round(float(rng.random()), 4), "t_max": s + 0.3}
                for c in ("full19", "subq_L") for s in range(60)]
        pd.DataFrame(rows).to_csv(os.path.join(cyc, n + "_scores.csv"), index=False)
    pd.DataFrame({"patient": ["x"], "rate_per_min": [1.0]}).to_csv(os.path.join(cyc, "summary.csv"), index=False)
    mor = os.path.join(tmp, "morgoth")
    os.makedirs(mor)
    rows = [{"clip": n + ".edf", "time_s": s + 0.5, "prob": round(float(rng.random()), 4), "edge": s < 5}
            for n in names[:3] for s in range(59)]
    pd.DataFrame(rows).to_csv(os.path.join(mor, "morgoth_seconds.csv.gz"), index=False)
    adm = os.path.join(tmp, "emu_admissions.csv")
    pd.DataFrame({"emu_id": ["EMU0001", "EMU0002", "EMU0003"], "patient_anon_id": ["PA", "PB", "PA"]}).to_csv(adm, index=False)
    return os.path.join(tmp, "edf"), cyc, mor, adm


def test_from_runs_reads_cyclops_style_folders():
    with tempfile.TemporaryDirectory() as tmp:
        edf, cyc, _mor, adm = fake_runs(tmp)
        out = os.path.join(tmp, "sec.csv")
        try:  # two conditions and no --condition: refused
            select_clips.main(["from-runs", cyc, "--clip-root", edf, "--out", out])
            raise AssertionError("should ask for --condition")
        except SystemExit as e:
            assert "full19" in str(e) and "subq_L" in str(e)
        select_clips.main(["from-runs", cyc, "--clip-root", edf, "--out", out, "--condition", "full19",
                           "--detector", "cyclops", "--admissions", adm])
        d = pd.read_csv(out)
        assert len(d) == 4 * 60 and set(d["patient"]) == {"PA", "PB"}  # EMU0099 is not in the admissions table
        assert d.groupby("patient")["admission"].nunique().to_dict() == {"PA": 2, "PB": 1}
        assert (d["condition"] == "full19").all() and (d["detector"] == "cyclops").all()
        assert d["start_s"].between(0, 59).all() and (d["start_s"] % 1 == 0).all()
        assert d["source_path"].map(os.path.exists).all()
        clips = os.path.join(tmp, "clips.csv")
        select_clips.main(["sample", out, "--out", clips, "--n", "20", "--bins", "0,0.5,1"])
        c = pd.read_csv(clips)
        assert len(c) == 20 and {"clip", "admission", "detector", "cyclops_prob"} <= set(c.columns)
        assert "morgoth_prob" not in c.columns  # a Cyclops score is never filed under MORGOTH's name


def test_from_runs_moves_centres_to_starts_and_checks_times():
    with tempfile.TemporaryDirectory() as tmp:
        edf, _cyc, mor, adm = fake_runs(tmp)
        out = os.path.join(tmp, "sec.csv")
        select_clips.main(["from-runs", mor, "--clip-root", edf, "--out", out, "--admissions", adm])
        d = pd.read_csv(out)
        assert len(d) == 3 * 59 and d["start_s"].min() == 0 and (d["start_s"] % 1 == 0).all()
        assert "edge" in d.columns
        # times in samples instead of seconds run past the end of the recordings: refused
        t = pd.read_csv(os.path.join(mor, "morgoth_seconds.csv.gz"))
        t["time_s"] = t["time_s"] * 200
        t.to_csv(os.path.join(mor, "morgoth_seconds.csv.gz"), index=False)
        try:
            select_clips.main(["from-runs", mor, "--clip-root", edf, "--out", out, "--admissions", adm])
            raise AssertionError("times in samples should be refused")
        except SystemExit as e:
            assert "fall outside their recording" in str(e)
        # never reads an id_map file
        bad = os.path.join(tmp, "emu_id_map.csv")
        os.rename(adm, bad)
        try:
            select_clips.main(["from-runs", mor, "--clip-root", edf, "--out", out, "--admissions", bad])
            raise AssertionError("id_map should be refused")
        except SystemExit as e:
            assert "id_map" in str(e)


def test_from_runs_reads_relative_paths_under_clip_root():
    """Two recordings with the same file name in different folders: a relative path column keeps them apart,
    where the name search can only leave both out."""
    from edf_writer import write_edf
    rng = np.random.default_rng(4)
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "uneeg")
        rows = []
        for sub in ("visit_a", "visit_b"):
            os.makedirs(os.path.join(root, sub))
            write_edf(os.path.join(root, sub, "US09-01_SCALP_1.edf"), ["EEG C3-REF", "EEG C4-REF"], 200,
                      [(0, rng.standard_normal((2, 200 * 30)) * 20)])
            rows += [{"file": f"{sub}/US09-01_SCALP_1.edf", "patient": "US09-01", "t_in_file_s": float(t),
                      "p_full19": round(float(rng.random()), 4)} for t in range(30)]
        run = os.path.join(tmp, "run")
        os.makedirs(run)
        pd.DataFrame(rows).to_csv(os.path.join(run, "US09-01.csv.gz"), index=False)
        out = os.path.join(tmp, "sec.csv")
        flags = ["--time-col", "t_in_file_s", "--time-is", "start", "--score-col", "p_full19", "--patient-col", "patient",
                 "--clip-col", "file", "--clip-root", root, "--detector", "cyclops", "--out", out]
        try:  # by name only: both copies match two files
            select_clips.main(["from-runs", run, *flags])
            raise AssertionError("an ambiguous name should leave the rows out")
        except SystemExit as e:
            assert "no rows left" in str(e)
        select_clips.main(["from-runs", run, *flags, "--path-col", "file"])
        d = pd.read_csv(out)
        assert len(d) == 60 and d["source_path"].nunique() == 2 and d["source_path"].map(os.path.isabs).all()
        assert d["source_path"].map(os.path.exists).all() and set(d["patient"]) == {"US09-01"}


def test_inspect_prints_layout_without_values():
    with tempfile.TemporaryDirectory() as tmp:
        _edf, cyc, _mor, _adm = fake_runs(tmp)
        out = subprocess.run([sys.executable, os.path.join(KIT, "select_clips.py"), "inspect", cyc],
                             capture_output=True, text=True, check=True).stdout
        assert "time=second" in out and "score=max_score" in out and "full19 (300 rows)" in out
        assert "not usable as is" in out
        assert not re.search(r"0\.\d{3}", out)  # no score values printed


# ---------------------------------------------------------------- live mode

def live_opts(**kw):
    from types import SimpleNamespace
    o = dict(fs=200.0, clip_s=12.0, pad_s=1.0, candidate_s=1.0, batch_size=4, repeat_frac=0.25, seed=7,
             batch_prefix="b", extras=["EKG"], reader="edf")
    o.update(kw)
    return SimpleNamespace(**o)


def make_live(tmp):
    edf = os.path.join(tmp, "edf")
    subprocess.run([sys.executable, os.path.join(KIT, "tests", "make_test_edfs.py"), edf], check=True, capture_output=True)
    return os.path.join(edf, "clips.csv")


def test_live_study_matches_pack_and_leaves_out_unreadable_clips():
    import make_review_batches as mrb
    with tempfile.TemporaryDirectory() as tmp:
        clips = make_live(tmp)
        study, labels = os.path.join(tmp, "study"), os.path.join(tmp, "labels")
        st = serve_review.LiveStore(clips, study, labels, live_opts())
        plan = st.plan
        assert [f["row"] for f in plan["left_out"]] == [10]  # the clip that runs past the end of rec1
        assert sum(len(b["members"]) for b in plan["batches"]) == 10  # 8 readable clips + 2 repeats
        for name in ("plan.json", "key.csv"):
            assert oct(os.stat(os.path.join(study, name)).st_mode & 0o777) == "0o600"
        key = pd.read_csv(os.path.join(study, "key.csv"))
        assert len(key) == 10 and key["repeat_of"].notna().sum() == 2 and "truth" in key.columns
        # pack with the same settings makes the same batches
        out = os.path.join(tmp, "packed")
        subprocess.run([sys.executable, os.path.join(KIT, "make_review_batches.py"), "pack", clips, "--out", out,
                        "--batch-size", "4", "--repeat-frac", "0.25", "--seed", "7"], check=True, capture_output=True)
        for b in plan["batches"]:
            live = json.loads(__import__("gzip").decompress(st.batch_bytes(b["batch_id"])))
            packed = mrb.read_batch(os.path.join(out, b["batch_id"] + ".json.gz"))
            live.pop("created"), packed.pop("created")
            assert live == packed, b["batch_id"]
        assert pd.read_csv(os.path.join(out, "key.csv")).drop(columns=["batch_file"]).equals(key.drop(columns=["batch_file"]))
        # a restart reuses the plan; a changed clip table is refused
        again = serve_review.LiveStore(clips, study, labels, live_opts(seed=99))
        assert again.plan["created"] == plan["created"] and again.batches().keys() == st.batches().keys()
        with open(clips, "a") as f:
            f.write("\n")
        try:
            serve_review.LiveStore(clips, study, labels, live_opts())
            raise AssertionError("changed clip table should be refused")
        except SystemExit as e:
            assert "has changed" in str(e)


def test_live_server_keeps_sources_out_of_the_browser():
    with tempfile.TemporaryDirectory() as tmp:
        clips = make_live(tmp)
        st = serve_review.LiveStore(clips, os.path.join(tmp, "study"), os.path.join(tmp, "labels"), live_opts())
        s = Server(tmp, raters=["EC"], store=st)
        try:
            j = json.loads(s.req("/api/batches", who="EC")[1])
            assert [b["batch_id"] for b in j["batches"]] == ["b01", "b02", "b03"] and j["rater"] == "EC"
            code, body = s.req("/api/batch/b01", who="EC")
            assert code == 200 and body[:2] == b"\x1f\x8b"
            text = __import__("gzip").decompress(body).decode()
            for leak in ("rec1", "rec2", "TEST-", "source_path", "truth", tmp):
                assert leak not in text, leak
            for p in ("/key.csv", "/plan.json", "/api/batch/plan", "/api/batch/..%2Fplan.json", "/study/key.csv"):
                assert s.req(p, who="EC")[0] == 404, p
            # an EEG file that disappears after planning: the browser gets a plain error, not the path
            os.rename(os.path.join(tmp, "edf", "rec2.edf"), os.path.join(tmp, "edf", "moved.edf"))
            st._cache.clear()
            bad = [bid for bid in ("b01", "b02", "b03") if s.req(f"/api/batch/{bid}", who="EC")[0] == 500]
            assert bad
            code, body = s.req(f"/api/batch/{bad[0]}", who="EC")
            assert b"could not be read" in body and b"rec2" not in body and tmp.encode() not in body
        finally:
            s.close()



# ---------------------------------------------------------------- no detector output yet

def test_from_edf_draws_clips_without_scores_and_serves_them():
    import report_labels
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, os.path.join(KIT, "tests", "make_fake_study.py"), tmp], check=True, capture_output=True)
        edf, adm = os.path.join(tmp, "edf"), os.path.join(tmp, "admissions.csv")
        sec, clips = os.path.join(tmp, "sec.csv"), os.path.join(tmp, "clips.csv")
        select_clips.main(["from-edf", "--clip-root", edf, "--pattern", "*iic*", "--admissions", adm, "--out", sec])
        d = pd.read_csv(sec)
        assert len(d) == 210 * 3 and (d["prob"] == 0).all() and set(d["patient"]) == {"FAKE-A", "FAKE-B"}
        gap = d[d["clip"] == "EMU9902_iic_0"]["start_s"]
        assert not gap.between(90, 149.5).any()  # no seconds inside the EDF+D gap
        select_clips.main(["sample", sec, "--out", clips, "--n", "10", "--min-gap-s", "12"])
        c = pd.read_csv(clips)
        assert len(c) == 10 and not any(col.endswith("_prob") for col in c.columns) and set(c["bin"]) == {"0-100%"}
        st = serve_review.LiveStore(clips, os.path.join(tmp, "study"), os.path.join(tmp, "labels"), live_opts(repeat_frac=0))
        assert st.plan["left_out"] == [] and sum(len(b["members"]) for b in st.plan["batches"]) == 10
        # a report on clips with no score shows the share and skips calibration
        key = pd.read_csv(os.path.join(tmp, "study", "key.csv"))
        merged = key.assign(rater="EC", decision=["yes", "no"] * 5, max_electrodes="", generalized=0, seconds_viewed=3)
        a = report_labels.analyze(merged)
        head, body = report_labels.render(a, "t", "s")
        assert "No detector score to calibrate" in body and ">nan<" not in body.lower()


def test_check_data_says_ok_or_names_the_problem():
    from edf_writer import write_edf
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, os.path.join(KIT, "tests", "make_fake_study.py"), tmp], check=True, capture_output=True)
        run = lambda *a: subprocess.run([sys.executable, os.path.join(KIT, "select_clips.py"), "check-data", *a],
                                        capture_output=True, text=True)
        out = run("--clip-root", os.path.join(tmp, "edf"))
        assert out.returncode == 0 and "OK: the review server can read these files" in out.stdout, out.stdout
        assert "EDF+D 1" in out.stdout and "all 19 found" in out.stdout
        # clips named in a list are found under the root
        lst = os.path.join(tmp, "list.csv")
        pd.DataFrame({"clip_id": ["EMU9901_iic_0", "EMU9903_iic_0", "EMU0000_iic_9"]}).to_csv(lst, index=False)
        out = run("--clip-root", os.path.join(tmp, "edf"), "--clip-list", lst)
        assert "checked 2 of 2" in out.stdout and "1 of 3 listed clips matched no single EDF" in out.stdout
        # channel names that do not map to 10-20, and volts labelled as microvolts
        bad = os.path.join(tmp, "bad")
        os.makedirs(bad)
        x = np.random.default_rng(0).standard_normal((19, 200 * 30)) * 20
        write_edf(os.path.join(bad, "a.edf"), [f"CH{i}" for i in range(19)], 200, [(0, x)])
        out = run("--clip-root", bad)
        assert out.returncode == 1 and "missing in most files" in out.stdout
        write_edf(os.path.join(bad, "a.edf"), LABS19[:19], 200, [(0, x * 1e-6)], unit="uV", phys=(-0.005, 0.005))
        out = run("--clip-root", bad)
        assert out.returncode == 1 and "implausible amplitude" in out.stdout



# ---------------------------------------------------------------- kit 1.6: Yes/No, shared batches, agreement

def synthetic_study(tmp, n=40, size=10, shared=10, answers="yn"):
    out = os.path.join(tmp, "syn")
    subprocess.run([sys.executable, os.path.join(KIT, "make_review_batches.py"), "synthetic", "--n", str(n), "--out", out,
                    "--batch-size", str(size), "--shared-clips", str(shared), "--answers", answers, "--seed", "3"],
                   check=True, capture_output=True)
    return out


def test_shared_clips_get_batches_of_their_own():
    import make_review_batches as mrb
    plan = lambda **kw: mrb.plan_batches(30, "rater", 10, 0.0, "b", np.random.default_rng(1), **kw)[1]
    plain = plan()
    assert [b["batch_id"] for b in plain] == ["b01", "b02", "b03"] and not any(b["shared"] for b in plain)
    assert plan(shared=0) == plain  # no shared clips: the plan is what earlier kits made
    split = plan(shared=12)
    assert [(b["batch_id"], len(b["members"])) for b in split] == [("s01", 10), ("s02", 2), ("b01", 10), ("b02", 8)]
    assert [b["shared"] for b in split] == [True, True, False, False]
    ids = [m["id"] for b in split for m in b["members"]]
    assert len(ids) == len(set(ids)) == 30
    try:
        mrb.plan_batches(30, "rater", 10, 0.0, "s", np.random.default_rng(1), shared=5)
        raise AssertionError("prefix s should be refused for split batches")
    except ValueError:
        pass


def test_yes_no_batches_carry_their_answers_and_refuse_unsure():
    import make_review_batches as mrb
    with tempfile.TemporaryDirectory() as tmp:
        syn = synthetic_study(tmp)
        head = mrb.read_batch(os.path.join(syn, "s01.json.gz"))
        assert head["answers"] == ["yes", "no"]
        manifest = json.load(open(os.path.join(syn, "manifest.json")))
        assert manifest["answers"] == ["yes", "no"] and manifest["shared_clips"] == 10
        assert [b["batch_id"] for b in manifest["batches"] if b["shared"]] == ["s01"]
        st = serve_review.Store(syn, os.path.join(tmp, "labels"), False)
        assert st.batches()["s01"]["shared"] and not st.batches()["b01"]["shared"]
        ids = [c["id"] for c in head["clips"]]

        def body(decisions):
            out = io.StringIO()
            w = csv.DictWriter(out, fieldnames=serve_review.EXPORT_COLUMNS)
            w.writeheader()
            for i, (cid, d) in enumerate(zip(ids, decisions)):
                w.writerow({"batch_id": "s01", "clip_id": cid, "rater": "EC", "order_index": i + 1, "decision": d})
            return out.getvalue().encode()
        assert st.save_labels("s01", "EC", body(["yes", "no", ""])) == (2, 3)
        try:
            st.save_labels("s01", "EC", body(["yes", "unsure"]))
            raise AssertionError("unsure should be refused in a Yes/No study")
        except ValueError as e:
            assert "unsure" in str(e)
        # batches packed before 1.6 have no answers field and keep Unsure
        old = os.path.join(tmp, "old")
        bdir, _v = make_batches(old)
        assert serve_review.Store(bdir, os.path.join(old, "l"), False).batches()["b01"]["answers"] == ["yes", "no", "unsure"]


def test_assignment_splits_evenly_and_reassign_moves_only_unstarted_batches():
    with tempfile.TemporaryDirectory() as tmp:
        syn = synthetic_study(tmp, n=70, size=10, shared=20)
        labels = os.path.join(tmp, "labels")
        st = serve_review.Store(syn, labels, False)
        a = serve_review.settle_assignment(st, labels, ["EC", "JK"], False)
        assert a["shared"] == ["s01", "s02"]
        assert sorted(len(v) for v in a["by_rater"].values()) == [2, 3]
        assert sorted(b for v in a["by_rater"].values() for b in v) == ["b01", "b02", "b03", "b04", "b05"]
        batches = st.batches()
        assert serve_review.visible(a, batches, "EC", "s01") and serve_review.visible(a, batches, "", "b01")
        mine = a["by_rater"]["EC"]
        theirs = a["by_rater"]["JK"]
        assert serve_review.visible(a, batches, "EC", mine[0]) and not serve_review.visible(a, batches, "EC", theirs[0])
        # kept as is on the next start, even with a new rater
        assert serve_review.settle_assignment(st, labels, ["EC", "JK", "AB"], False) == a
        # JK starts one batch; --reassign keeps it with JK and deals the rest over three raters
        started = theirs[0]
        clip = sorted(batches[started]["clip_ids"])[0]
        out = io.StringIO()
        w = csv.DictWriter(out, fieldnames=serve_review.EXPORT_COLUMNS)
        w.writeheader()
        w.writerow({"batch_id": started, "clip_id": clip, "rater": "JK", "order_index": 1, "decision": "no"})
        st.save_labels(started, "JK", out.getvalue().encode())
        b = serve_review.settle_assignment(st, labels, ["EC", "JK", "AB"], True)
        assert started in b["by_rater"]["JK"] and set(b["by_rater"]) == {"EC", "JK", "AB"}
        assert sorted(x for v in b["by_rater"].values() for x in v) == ["b01", "b02", "b03", "b04", "b05"]
        assert all(len(v) <= 2 for v in b["by_rater"].values())
        assert serve_review.load_assignment(labels) == b


def test_server_shows_each_rater_the_shared_batches_and_their_own():
    with tempfile.TemporaryDirectory() as tmp:
        syn = synthetic_study(tmp, n=40, size=10, shared=10)
        labels = os.path.join(tmp, "labels")
        st = serve_review.Store(syn, labels, False)
        raters = serve_review.load_rater_tokens(labels, False, ["EC", "JK"])
        owner = serve_review.load_token(labels, False)
        a = serve_review.settle_assignment(st, labels, raters, False)
        viewer = os.path.join(tmp, "v.html")
        open(viewer, "w").write("<!doctype html>")
        from http.server import ThreadingHTTPServer
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve_review.make_handler(st, {}, viewer, 0))
        port = httpd.server_address[1]
        httpd.RequestHandlerClass = serve_review.make_handler(st, {"": owner, **raters}, viewer, port, a)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        def req(path, tok, method="GET", body=None):
            r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body, method=method,
                                       headers={"X-Review-Token": tok})
            try:
                with urllib.request.urlopen(r) as resp:
                    return resp.status, resp.read()
            except urllib.error.HTTPError as e:
                return e.code, e.read()
        try:
            got = json.loads(req("/api/batches", raters["EC"])[1])
            ids = [b["batch_id"] for b in got["batches"]]
            assert ids[0] == "s01" and got["batches"][0]["shared"]
            assert ids == ["s01"] + a["by_rater"]["EC"]
            other = a["by_rater"]["JK"][0]
            assert req(f"/api/batch/{other}", raters["EC"])[0] == 404
            assert req(f"/api/labels/{other}/EC", raters["EC"], "PUT", b"x")[0] == 403
            assert len(json.loads(req("/api/batches", owner)[1])["batches"]) == 4
        finally:
            httpd.shutdown()
            httpd.server_close()


def test_fleiss_and_ac1_match_published_examples():
    import merge_labels
    # Fleiss (1971) as worked on Wikipedia: 10 subjects, 14 raters, 5 categories, kappa 0.210
    t = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
         [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
    assert abs(merge_labels.fleiss_ac1(t, 5)[0] - 0.2099) < 5e-4
    # the kappa paradox: two raters agree on 118 of 125 clips, nearly all Yes; kappa near 0, AC1 0.94
    t = [[2, 0]] * 118 + [[1, 1]] * 7
    k, ac1 = merge_labels.fleiss_ac1(t, 2)
    assert abs(k - (-0.0288)) < 5e-4 and abs(ac1 - 0.9408) < 5e-4
    # from label rows: three raters, clips answered by two or three of them
    rows = []
    for c in range(30):
        for r, d in (("A", "yes" if c < 5 else "no"), ("B", "yes" if c < 6 else "no"), ("C", "no")):
            if not (r == "C" and c % 2):
                rows.append({"clip_id": f"c{c}", "rater": r, "decision": d})
    out = merge_labels.agreement_all(pd.DataFrame(rows), n_boot=200)
    assert out["n_clips"] == 30 and out["answers"] == ["yes", "no"]
    assert out["ac1"] > out["fleiss"] and out["ac1_ci"][0] <= out["ac1"] <= out["ac1_ci"][1]


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
