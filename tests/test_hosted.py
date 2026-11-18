"""Tests for the hosted review site (hosted/). Run: python -m pytest -q tests/test_hosted.py
The two server tests need node 18 or later and are skipped without it. Uses synthetic data only; nothing is sent anywhere but 127.0.0.1."""
import base64
import csv
import gzip
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOSTED = os.path.join(KIT, "hosted")
sys.path.insert(0, KIT)
sys.path.insert(0, HOSTED)
import check_upload  # noqa: E402
import serve_review  # noqa: E402

SITE_FILES = {".githooks/pre-commit", ".gitignore", "README.md", "api/review.mjs", "content/site.json", "lib/core.mjs",
              "lib/store.mjs", "package.json", "public/index.html", "tools/check_upload.py", "tools/dev_server.mjs",
              "vercel.json"}


def run(*args, ok=True):
    out = subprocess.run([sys.executable, *args], capture_output=True, text=True)
    if ok and out.returncode:
        raise AssertionError(out.stdout + out.stderr)
    return out


def make_site(tmp, *extra, raters="EC,JK"):
    syn = os.path.join(tmp, "syn")
    if not os.path.isdir(syn):
        run(os.path.join(KIT, "make_review_batches.py"), "synthetic", "--n", "40", "--out", syn, "--batch-size", "10",
            "--shared-clips", "10", "--seed", "3")
    labels, site = os.path.join(tmp, "labels"), os.path.join(tmp, "site")
    out = run(os.path.join(HOSTED, "make_site.py"), syn, "--labels", labels, "--out", site, "--raters", raters, *extra)
    return syn, labels, site, out.stdout


def files_in(site):
    return {os.path.relpath(os.path.join(t, n), site).replace(os.sep, "/")
            for t, _d, ns in os.walk(site) for n in ns if ".git" not in t.split(os.sep)}


def no_node():
    """The site's server tests need node; Borel may not have it, and make_site.py, check_upload.py and
    pull_labels.py don't."""
    if shutil.which("node"):
        return False
    print("skipped: node is not installed here")
    return True


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class DevServer:
    """tools/dev_server.mjs from the site folder, the way Vercel would serve it."""

    def __init__(self, site, store, env=None):
        self.port = free_port()
        self.proc = subprocess.Popen(["node", os.path.join(site, "tools", "dev_server.mjs"), "--port", str(self.port),
                                      "--store", store], env={**os.environ, **(env or {})},
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        line = self.proc.stdout.readline()
        assert "preview on" in line, line
        self.base = f"http://127.0.0.1:{self.port}"

    def req(self, path, token=None, method="GET", body=None, auth=None):
        h = {"X-Review-Token": token} if token else {}
        if auth:
            h["Authorization"] = "Basic " + base64.b64encode(":".join(auth).encode()).decode()
        r = urllib.request.Request(self.base + path, data=body, method=method, headers=h)
        try:
            with urllib.request.urlopen(r) as resp:
                return resp.status, resp.read(), resp.headers
        except urllib.error.HTTPError as e:
            return e.code, e.read(), e.headers

    def close(self):
        self.proc.terminate()
        self.proc.wait()


def labels_body(batch, rater, clip_ids, decisions):
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=serve_review.EXPORT_COLUMNS, lineterminator="\n")
    w.writeheader()
    for i, (cid, d) in enumerate(zip(clip_ids, decisions)):
        w.writerow({"batch_id": batch, "clip_id": cid, "rater": rater, "order_index": i + 1, "decision": d,
                    "labeled_at": "2026-10-23T00:00:00Z"})
    return out.getvalue().encode()


def tokens(labels):
    return serve_review.load_token(labels, False), serve_review.load_rater_tokens(labels, False)


# ---------------------------------------------------------------- what goes in the site

def test_site_holds_only_blinded_batches_and_token_hashes():
    with tempfile.TemporaryDirectory() as tmp:
        syn, labels, site, out = make_site(tmp)
        assert "upload check OK" in out and "s01" in out
        got = files_in(site)
        batches = {f for f in got if f.startswith("content/batches/")}
        assert got - batches == SITE_FILES
        assert batches == {f"content/batches/{b}.json.gz" for b in ("s01", "b01", "b02", "b03")}
        sj = json.load(open(os.path.join(site, "content", "site.json")))
        owner, raters = tokens(labels)
        import hashlib
        assert sj["tokens"] == {w: hashlib.sha256(t.encode()).hexdigest() for w, t in {"": owner, **raters}.items()}
        assert [b["batch_id"] for b in sj["batches"]] == ["s01", "b01", "b02", "b03"]
        assert sj["assignment"]["shared"] == ["s01"] and set(sj["assignment"]["by_rater"]) == {"EC", "JK"}
        # the batches are the study's own, with the gzip header stripped of name and time
        for b in sj["batches"]:
            raw = open(os.path.join(site, "content", "batches", b["file"]), "rb").read()
            assert raw[3] == 0 and raw[4:8] == b"\0\0\0\0"  # no FNAME flag, mtime 0
            assert json.loads(gzip.decompress(raw)) == json.loads(gzip.decompress(open(os.path.join(syn, b["file"]), "rb").read()))
        # nothing from the labels or study folders, and no raw token anywhere
        blob = b"".join(open(os.path.join(site, f), "rb").read() for f in got if not f.endswith(".gz"))
        for t in [owner, *raters.values()]:
            assert t.encode() not in blob
        assert check_upload.check(site, labels) == []


def test_make_site_is_repeatable_and_follows_link_changes():
    with tempfile.TemporaryDirectory() as tmp:
        syn, labels, site, _ = make_site(tmp)
        snap = {f: open(os.path.join(site, f), "rb").read() for f in files_in(site)}
        _, _, _, out = make_site(tmp)
        assert "0 batch file(s) new or changed" in out
        assert {f: open(os.path.join(site, f), "rb").read() for f in files_in(site)} == snap
        # the site's viewer carries the mark that makes it run like the Borel tunnel, once
        assert snap["public/index.html"].count(b'<meta name="ied-review-site"') == 1
        # a revoked link disappears from the site; the split stays as it was
        _, _, _, out = make_site(tmp, "--revoke", "JK", raters="")
        sj = json.load(open(os.path.join(site, "content", "site.json")))
        assert set(sj["tokens"]) == {"", "EC"} and set(sj["assignment"]["by_rater"]) == {"EC", "JK"}
        assert "(link revoked)" in out
        # a stray file stops the build
        open(os.path.join(site, "notes.txt"), "w").write("x")
        out = run(os.path.join(HOSTED, "make_site.py"), syn, "--labels", labels, "--out", site, ok=False)
        assert out.returncode == 1 and "notes.txt" in out.stderr
        # a folder that is not a site is refused
        other = os.path.join(tmp, "other")
        os.makedirs(other)
        open(os.path.join(other, "thesis.txt"), "w").write("x")
        out = run(os.path.join(HOSTED, "make_site.py"), syn, "--labels", labels, "--out", other, ok=False)
        assert out.returncode and "not a review site" in out.stderr
        out = run(os.path.join(HOSTED, "make_site.py"), syn, "--labels", labels, "--out", os.path.join(labels, "x"), ok=False)
        assert out.returncode and "folder of its own" in out.stderr


def test_research_batches_never_go_on_the_site():
    with tempfile.TemporaryDirectory() as tmp:
        res = os.path.join(tmp, "res")
        run(os.path.join(KIT, "make_review_batches.py"), "synthetic", "--n", "6", "--out", res, "--mode", "research")
        out = run(os.path.join(HOSTED, "make_site.py"), res, "--labels", os.path.join(tmp, "l"), "--out",
                  os.path.join(tmp, "site"), ok=False)
        assert out.returncode and "no rater batches" in out.stderr


def test_make_site_from_a_live_study_keeps_sources_out():
    with tempfile.TemporaryDirectory() as tmp:
        edf = os.path.join(tmp, "edf")
        run(os.path.join(KIT, "tests", "make_test_edfs.py"), edf)
        site = os.path.join(tmp, "site")
        out = run(os.path.join(HOSTED, "make_site.py"), "--clips", os.path.join(edf, "clips.csv"), "--study",
                  os.path.join(tmp, "study"), "--labels", os.path.join(tmp, "labels"), "--out", site, "--raters", "EC",
                  "--batch-size", "4", "--shared-clips", "4").stdout
        assert "upload check OK" in out
        assert not any(f.endswith(("key.csv", "plan.json")) for f in files_in(site))
        for f in os.listdir(os.path.join(site, "content", "batches")):
            text = gzip.decompress(open(os.path.join(site, "content", "batches", f), "rb").read()).decode()
            for leak in ("rec1", "rec2", "TEST-", "source_path", "truth", tmp):
                assert leak not in text, (f, leak)
            assert json.loads(text)["answers"] == ["yes", "no"]


def test_upload_check_catches_planted_problems():
    with tempfile.TemporaryDirectory() as tmp:
        _syn, labels, site, _ = make_site(tmp)
        owner, _r = tokens(labels)

        def batch_with(fn):
            def plant(d):
                p = os.path.join(d, "content", "batches", "b01.json.gz")
                obj = json.loads(gzip.decompress(open(p, "rb").read()))
                fn(obj)
                open(p, "wb").write(gzip.compress(json.dumps(obj).encode(), mtime=0))
            return plant

        def site_with(fn):
            def plant(d):
                p = os.path.join(d, "content", "site.json")
                obj = json.load(open(p))
                fn(obj)
                json.dump(obj, open(p, "w"))
            return plant

        def add(name, text="x"):
            def plant(d):
                os.makedirs(os.path.dirname(os.path.join(d, name)), exist_ok=True)
                open(os.path.join(d, name), "w").write(text)
            return plant

        plants = {
            "key.csv": add("content/batches/key.csv", "clip_id,source_path\n"),
            "an EEG file": add("content/rec1.edf"),
            "an id map": add("content/emu_id_map.json", "{}"),
            "a stray file": add("notes.txt"),
            "a label file": add("content/labels_b01_EC.csv"),
            "a research batch": batch_with(lambda o: o.update(mode="research")),
            "an EMU ID in notes": batch_with(lambda o: o.update(notes="from EMU1234")),
            "clip details": batch_with(lambda o: o["clips"][0].update(meta={"patient": "x"})),
            "a non-kit clip ID": batch_with(lambda o: o["clips"][0].update(id="EMU0042_s300")),
            "a path in a channel": batch_with(lambda o: o["channels"].__setitem__(0, "/users/x")),
            "a date": site_with(lambda o: o["batches"][0].update(file="b01.json.gz", n_clips=10, note="seen 2024-03-05")),
            "an unknown site field": site_with(lambda o: o.update(owner_email="x")),
            "a raw token": site_with(lambda o: o["tokens"].__setitem__("", owner)),
            "a UNEEG ID": site_with(lambda o: o.update(note="US09-13")),
            "a Borel path": add("README.md", "data in /mnt/data/eeg"),
            "an oversize batch": lambda d: open(os.path.join(d, "content", "batches", "b09.json.gz"), "wb").write(
                gzip.compress(json.dumps({"format": "ied-review-batch", "batch_id": "b09", "mode": "rater",
                                          "pad": os.urandom(4_100_000).hex()}).encode(), 0)),
        }
        for what, plant in plants.items():
            d = os.path.join(tmp, "copy")
            shutil.rmtree(d, ignore_errors=True)
            shutil.copytree(site, d)
            plant(d)
            problems = check_upload.check(d, labels)
            assert problems, f"upload check missed {what}"
        # the pre-commit hook runs the same check
        if not shutil.which("git"):
            return
        d = os.path.join(tmp, "copy")
        shutil.rmtree(d)
        shutil.copytree(site, d)
        subprocess.run(["git", "init", "-q", d], check=True)
        subprocess.run(["git", "-C", d, "config", "core.hooksPath", ".githooks"], check=True)
        subprocess.run(["git", "-C", d, "add", "-A"], check=True)
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t",
               "GIT_COMMITTER_EMAIL": "t@x"}
        assert subprocess.run(["git", "-C", d, "commit", "-qm", "site"], env=env, capture_output=True).returncode == 0
        open(os.path.join(d, "key.csv"), "w").write("clip_id\n")
        subprocess.run(["git", "-C", d, "add", "-A"], check=True)
        r = subprocess.run(["git", "-C", d, "commit", "-qm", "oops"], env=env, capture_output=True, text=True)
        assert r.returncode != 0 and "key.csv" in r.stderr


# ---------------------------------------------------------------- the site's server

def test_manuals_go_on_the_site_checked_remembered_and_behind_a_button():
    with tempfile.TemporaryDirectory() as tmp:
        man = os.path.join(tmp, "manual_scalp.md")
        with open(man, "w", encoding="utf-8") as f:
            f.write("# 10-20 manual\n\nRead **each** clip <b>alone</b>, *negative up*.\n\n"
                    "| Key | What it does |\n| --- | --- |\n| Y | Yes |\n| ← / → | Previous or next |\n\n"
                    "1. Ten examples.\n2. Scalp clips,\n   wrapped onto a second line.\n\n- one\n- two\n")
        uman = os.path.join(tmp, "manual_uneeg.md")
        with open(uman, "w", encoding="utf-8") as f:
            f.write("# UNEEG manual\n\nThree channels.\n")
        syn, labels, site, out = make_site(tmp, "--manual", "scalp=" + man, "--manual", "uneeg=" + uman)
        assert "Manual button" in out and "10-20 manual" in out and "UNEEG manual" in out
        assert {"public/manual_scalp.html", "public/manual_uneeg.html"} <= set(files_in(site))
        page = open(os.path.join(site, "public", "manual_scalp.html"), encoding="utf-8").read()
        for want in ("<title>10-20 manual</title>", "<h1>10-20 manual</h1>", "<strong>each</strong>",
                     "&lt;b&gt;alone&lt;/b&gt;", "<em>negative up</em>", "<th>What it does</th>", "<td>← / →</td>",
                     "<li>Scalp clips, wrapped onto a second line.</li>", "<ol>", "<li>two</li>"):
            assert want in page, want
        assert "<b>" not in page
        assert "<h1>UNEEG manual</h1>" in open(os.path.join(site, "public", "manual_uneeg.html"), encoding="utf-8").read()
        assert json.load(open(os.path.join(site, "content", "site.json")))["manuals"] == {
            "scalp": "manual_scalp.html", "uneeg": "manual_uneeg.html"}
        assert check_upload.check(site, labels) == []
        # remembered in the labels folder until --manual none; a bare FILE.md is the 10-20 manual
        make_site(tmp)
        assert os.path.isfile(os.path.join(site, "public", "manual_uneeg.html"))
        make_site(tmp, "--manual", "none", "--manual", man)
        assert os.path.isfile(os.path.join(site, "public", "manual_scalp.html"))
        assert not os.path.exists(os.path.join(site, "public", "manual_uneeg.html"))
        assert json.load(open(os.path.join(site, "content", "site.json")))["manuals"] == {"scalp": "manual_scalp.html"}
        make_site(tmp, "--manual", "none")
        assert not any(f.startswith("public/manual") for f in files_in(site))
        assert "manuals" not in json.load(open(os.path.join(site, "content", "site.json")))
        # a manual that names a patient, a date or a path never reaches the site; nor does a PDF
        for text, why in (("Clip from US09-02.\n", "UNEEG subject ID"), ("Updated 2026-11-10.\n", "date"),
                          ("See /users/someone/notes.\n", "Borel path")):
            with open(uman, "w", encoding="utf-8") as f:
                f.write("# Manual\n\n" + text)
            res = run(os.path.join(HOSTED, "make_site.py"), syn, "--labels", labels, "--out", site,
                      "--manual", "uneeg=" + uman, ok=False)
            assert res.returncode and why in res.stderr, (why, res.stderr)
            assert not os.path.exists(os.path.join(site, "public", "manual_uneeg.html"))
        pdf = os.path.join(tmp, "manual.pdf")
        open(pdf, "wb").write(b"%PDF-1.4")
        res = run(os.path.join(HOSTED, "make_site.py"), syn, "--labels", labels, "--out", site, "--manual", pdf, ok=False)
        assert res.returncode and "Markdown" in res.stderr
        # the upload check holds the pages to plain text, and site.json to the two manual names
        with open(man, "w", encoding="utf-8") as f:
            f.write("# Manual\n\nPlain words.\n")
        with open(uman, "w", encoding="utf-8") as f:
            f.write("# UNEEG manual\n\nThree channels.\n")
        make_site(tmp, "--manual", "scalp=" + man, "--manual", "uneeg=" + uman)
        for part in ("scalp", "uneeg"):
            mp = os.path.join(site, "public", f"manual_{part}.html")
            good = open(mp, encoding="utf-8").read()
            word = "Plain words." if part == "scalp" else "Three channels."
            for bad in ("<script>alert(1)</script>", '<a href="https://example.org">x</a>', '<img src=x onerror=alert(1)>'):
                open(mp, "w", encoding="utf-8").write(good.replace(word, bad))
                assert any(f"manual_{part}.html" in p for p in check_upload.check(site)), (part, bad)
            open(mp, "w", encoding="utf-8").write(good)
        open(os.path.join(site, "public", "manual.html"), "w").write("<p>x</p>")
        assert any("manual.html" in p and "not a file" in p for p in check_upload.check(site))
        os.remove(os.path.join(site, "public", "manual.html"))
        sp = os.path.join(site, "content", "site.json")
        good_site = open(sp).read()
        for bad in ({"scalp": "../key.csv"}, {"other": "manual_other.html"}, "manual_scalp.html", {}):
            sj = json.loads(good_site)
            sj["manuals"] = bad
            open(sp, "w").write(json.dumps(sj))
            assert any("manuals must" in p for p in check_upload.check(site)), bad
        open(sp, "w").write(good_site)
        assert check_upload.check(site, labels) == []
        if no_node():
            return
        owner, raters = tokens(labels)
        srv = DevServer(site, os.path.join(tmp, "store"))
        try:
            code, body, h = srv.req("/manual_scalp.html")  # no token needed: the manuals hold no study data
            assert code == 200 and b"<h1>Manual</h1>" in body and "frame-ancestors 'none'" in h["Content-Security-Policy"]
            assert b"<h1>UNEEG manual</h1>" in srv.req("/manual_uneeg.html")[1]
            assert json.loads(srv.req("/api/batches", raters["EC"])[1])["manuals"] == {
                "scalp": "manual_scalp.html", "uneeg": "manual_uneeg.html"}
        finally:
            srv.close()
        make_site(tmp, "--manual", "none")
        srv = DevServer(site, os.path.join(tmp, "store"))
        try:
            assert srv.req("/manual_scalp.html")[0] == 404
            assert "manuals" not in json.loads(srv.req("/api/batches", raters["EC"])[1])
        finally:
            srv.close()


def test_manual_page_reads_part_values():
    import manual_page
    with tempfile.TemporaryDirectory() as tmp:
        labels = os.path.join(tmp, "labels")
        a, b = os.path.join(tmp, "a.md"), os.path.join(tmp, "b.md")
        open(a, "w").write("# A\n")
        open(b, "w").write("# B\n")
        assert manual_page.load(labels, None) == {}
        assert manual_page.load(labels, [a]) == {"scalp": "# A\n"}
        assert manual_page.load(labels, ["UNEEG=" + b]) == {"scalp": "# A\n", "uneeg": "# B\n"}
        assert manual_page.load(labels, ["scalp=" + b]) == {"scalp": "# B\n", "uneeg": "# B\n"}
        assert manual_page.load(labels, ["none", "uneeg=" + a]) == {"uneeg": "# A\n"}
        assert manual_page.load(labels, ["none"]) == {}
        for bad in (os.path.join(tmp, "missing.md"), "uneeg=" + os.path.join(tmp, "missing.md")):
            try:
                manual_page.load(labels, [bad])
            except SystemExit as e:
                assert "no file" in str(e)
            else:
                raise AssertionError(bad)


def test_uneeg_manual_draws_the_lead_and_the_viewer_has_the_same_copy():
    """Kit 1.8.6: [[figure: uneeg-lead]] in a manual places the kit's drawing of the UNEEG lead, the kit's UNEEG
    manual uses it, the page still passes the upload check, and the viewer's own copy matches."""
    import manual_page
    _, body = manual_page.to_html("# M\n\nBefore\n\n[[figure: uneeg-lead]]\n\n[[figure: other]]\n")
    assert body.count('<figure class="lead-fig" data-fig="uneeg-lead"><svg class="lead-svg"') == 1
    for want in ("distal (SQD)", "centre (SQC)", "proximal (SQP)", ">D-C<", ">P-C<", ">D-P, twice the distance<",
                 "<figcaption>" + manual_page.UNEEG_LEAD_CAPTION + "</figcaption>"):
        assert want in body, want
    assert "<p>[[figure: other]]</p>" in body  # an unknown name stays plain, escaped text
    import re
    assert not re.search(r"<script|<iframe|<object|<embed|\bon[a-z]+\s*=|https?://", manual_page.UNEEG_LEAD_SVG, re.I)
    text = open(os.path.join(KIT, "rater_manual_uneeg.md"), encoding="utf-8").read()
    assert "[[figure: uneeg-lead]]" in text
    page = manual_page.page(text).decode("utf-8")
    assert 'data-fig="uneeg-lead"' in page and check_upload.scan_text(page, "manual_uneeg.html") == []
    with tempfile.TemporaryDirectory() as tmp:
        uman = os.path.join(tmp, "manual_uneeg.md")
        shutil.copy(os.path.join(KIT, "rater_manual_uneeg.md"), uman)
        syn, labels, site, out = make_site(tmp, "--manual", "uneeg=" + uman)
        assert 'data-fig="uneeg-lead"' in open(os.path.join(site, "public", "manual_uneeg.html"), encoding="utf-8").read()
        assert check_upload.check(site) == []
    for viewer in ("src/page.html", "ied_review.html"):
        v = open(os.path.join(KIT, viewer), encoding="utf-8").read()
        assert "const UNEEG_LEAD_SVG = " + json.dumps(manual_page.UNEEG_LEAD_SVG) + ";" in v, viewer
        assert "const UNEEG_LEAD_CAPTION = " + json.dumps(manual_page.UNEEG_LEAD_CAPTION) + ";" in v, viewer


def test_borel_server_serves_the_manuals_without_a_token():
    import threading
    from http.server import ThreadingHTTPServer
    import manual_page
    with tempfile.TemporaryDirectory() as tmp:
        syn = os.path.join(tmp, "syn")
        run(os.path.join(KIT, "make_review_batches.py"), "synthetic", "--n", "10", "--out", syn, "--batch-size", "10",
            "--seed", "3")
        store = serve_review.Store(syn, os.path.join(tmp, "labels"), False)
        both = {"scalp": manual_page.page("# 10-20 manual\n\nText.\n"), "uneeg": manual_page.page("# UNEEG manual\n")}
        for manuals in (both, {"uneeg": both["uneeg"]}, None):
            srv = ThreadingHTTPServer(("127.0.0.1", 0), serve_review.make_handler(store, {"": "t" * 32}, os.path.join(
                KIT, "ied_review.html"), 0))
            port = srv.server_address[1]
            srv.RequestHandlerClass = serve_review.make_handler(store, {"": "t" * 32}, os.path.join(KIT, "ied_review.html"),
                                                                port, manuals=manuals)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            try:
                def get(path, token=None):
                    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                               headers={"X-Review-Token": token} if token else {})
                    try:
                        with urllib.request.urlopen(r) as resp:
                            return resp.status, resp.read()
                    except urllib.error.HTTPError as e:
                        return e.code, e.read()
                for part, title in (("scalp", b"<h1>10-20 manual</h1>"), ("uneeg", b"<h1>UNEEG manual</h1>")):
                    code, body = get(f"/manual_{part}.html")
                    if manuals and part in manuals:
                        assert code == 200 and title in body, (part, code)
                    else:
                        assert code == 403, (part, code)
                assert get("/manual.html")[0] == 403
                listing = json.loads(get("/api/batches", "t" * 32)[1])
                assert listing.get("manuals") == ({k: f"manual_{k}.html" for k in manuals} if manuals else None)
            finally:
                srv.shutdown()
                srv.server_close()


def test_hosted_api_matches_the_borel_server():
    if no_node():
        return
    with tempfile.TemporaryDirectory() as tmp:
        syn, labels, site, _ = make_site(tmp)
        owner, raters = tokens(labels)
        sj = json.load(open(os.path.join(site, "content", "site.json")))
        ids = {b["batch_id"]: b["clip_ids"] for b in sj["batches"]}
        mine, theirs = sj["assignment"]["by_rater"]["EC"][0], sj["assignment"]["by_rater"]["JK"][0]
        srv = DevServer(site, os.path.join(tmp, "store"))
        try:
            code, body, h = srv.req("/?token=" + raters["EC"])
            assert code == 200 and b"IED" in body
            assert h["Referrer-Policy"] == "no-referrer" and "frame-ancestors 'none'" in h["Content-Security-Policy"]
            assert srv.req("/api/batches")[0] == 403
            assert srv.req("/api/batches", "not-a-token")[0] == 403
            got = json.loads(srv.req("/api/batches", raters["EC"])[1])
            assert got["rater"] == "EC" and [b["batch_id"] for b in got["batches"]] == ["s01", *sj["assignment"]["by_rater"]["EC"]]
            assert srv.req(f"/api/batch/{theirs}", raters["EC"])[0] == 404
            code, body, h = srv.req(f"/api/batch/{mine}", raters["EC"])
            assert code == 200 and body == open(os.path.join(site, "content", "batches", f"{mine}.json.gz"), "rb").read()
            assert h["Cache-Control"] == "no-store"
            for p in ("/content/site.json", "/content/batches/s01.json.gz", "/api/batch/..%2Fcontent%2Fsite.json", "/api/review.mjs",
                      "/lib/core.mjs", "/vercel.json"):
                assert srv.req(p, raters["EC"])[0] == 404, p
            # saving: own labels only, own batches only, answers the study offers
            body = labels_body("s01", "EC", ids["s01"], ["yes", "no", "no", ""])
            assert srv.req("/api/labels/s01/JK", raters["EC"], "PUT", body)[0] == 403
            assert srv.req(f"/api/labels/{theirs}/EC", raters["EC"], "PUT",
                           labels_body(theirs, "EC", ids[theirs], ["no"]))[0] == 403
            code, msg, _ = srv.req("/api/labels/s01/EC", raters["EC"], "PUT",
                                   labels_body("s01", "EC", ids["s01"], ["unsure"]))
            assert code == 400 and b"unsure" in msg
            code, msg, _ = srv.req("/api/labels/s01/EC", raters["EC"], "PUT", body)
            assert code == 200 and json.loads(msg)["answered"] == 3
            assert srv.req("/api/labels/s01/ec", raters["EC"])[1] == body
            assert srv.req("/api/labels/s01/JK", raters["JK"])[0] == 404
            assert srv.req("/api/labels/s01/EC", raters["JK"])[0] == 403
            # progress: a rater sees only their own, the owner sees everyone's
            jk = json.loads(srv.req("/api/batches", raters["JK"])[1])["batches"][0]
            assert jk["batch_id"] == "s01" and jk["progress"] == {"JK": 0}
            own = json.loads(srv.req("/api/batches", owner)[1])
            assert own["batches"][0]["progress"] == {"EC": 3} and len(own["batches"]) == 4
            # 1.7.7: the owner's link is told which initials have their own link and cannot save under them
            assert own["rater_links"] == ["EC", "JK"] and "rater_links" not in got
            code, msg, _ = srv.req("/api/labels/s01/EC", owner, "PUT", labels_body("s01", "EC", ids["s01"], ["no"] * 4))
            assert code == 403 and b"own link" in msg
            assert srv.req("/api/labels/s01/EC", owner)[1] == body
            # and pairwise the other way: a rater's link cannot read or save under the owner's initials either
            assert srv.req("/api/labels/s01/SJ", raters["EC"])[0] == 403
            assert srv.req("/api/labels/s01/SJ", raters["JK"], "PUT", labels_body("s01", "SJ", ids["s01"], ["yes"]))[0] == 403
            # export is the owner's only
            assert srv.req("/api/export", raters["EC"])[0] == 403
            files = json.loads(srv.req("/api/export?batch=s01", owner)[1])["files"]
            assert files == {"labels_s01_EC.csv": body.decode()}

            # pull_labels writes the same files serve_review.py saves, and merge_labels reads them
            run(os.path.join(HOSTED, "pull_labels.py"), "--labels", labels, "--site", srv.base)
            path = os.path.join(labels, "labels_s01_EC.csv")
            assert open(path, "rb").read() == body
            assert "1 already up to date" in run(os.path.join(HOSTED, "pull_labels.py"), "--labels", labels,
                                                 "--site", srv.base).stdout
            # a local copy with more answers (labelled through the tunnel) is kept
            more = labels_body("s01", "EC", ids["s01"], ["yes", "no", "no", "yes"])
            open(path, "wb").write(more)
            out = run(os.path.join(HOSTED, "pull_labels.py"), "--labels", labels, "--site", srv.base).stdout
            assert "kept labels_s01_EC.csv" in out and open(path, "rb").read() == more
            status = run(os.path.join(HOSTED, "pull_labels.py"), "--labels", labels, "--site", srv.base, "--status").stdout
            assert status.splitlines()[1].split()[:4] == ["s01", "10", "yes", "3"]
            srv.req("/api/labels/s01/JK", raters["JK"], "PUT", labels_body("s01", "JK", ids["s01"], ["yes", "no", "yes"]))
            run(os.path.join(HOSTED, "pull_labels.py"), "--labels", labels, "--site", srv.base)
            out = run(os.path.join(KIT, "merge_labels.py"), "--key", os.path.join(syn, "key.csv"),
                      "--out", os.path.join(tmp, "merged.csv"), path, os.path.join(labels, "labels_s01_JK.csv")).stdout
            assert "rater JK: 3 of" in out
            # a wrong owner link is refused with a plain message
            os.makedirs(os.path.join(tmp, "l2"))
            open(os.path.join(tmp, "l2", ".token"), "w").write("x" * 32)
            r = run(os.path.join(HOSTED, "pull_labels.py"), "--labels", os.path.join(tmp, "l2"), "--site", srv.base, ok=False)
            assert r.returncode and "does not know your link" in r.stderr
        finally:
            srv.close()


class FakeUpstash(BaseHTTPRequestHandler):
    """Enough of Upstash's REST API (POST /pipeline) to check the site's database code."""
    db = {}
    token = "fake-upstash-token"
    seen = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        if self.path != "/pipeline" or self.headers.get("Authorization") != f"Bearer {self.token}":
            self.send_response(401)
            self.end_headers()
            return
        cmds = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeUpstash.seen.append([c[0] for c in cmds])
        out = []
        for cmd, key, *args in cmds:
            if cmd == "GET":
                out.append({"result": self.db.get(key)})
            elif cmd == "SET":
                if "NX" in args and key in self.db:
                    out.append({"result": None})
                else:
                    self.db[key] = args[0]
                    out.append({"result": "OK"})
            elif cmd == "HSET":
                h = self.db.setdefault(key, {})
                h.update(dict(zip(args[::2], args[1::2])))
                out.append({"result": 1})
            elif cmd == "HGETALL":
                out.append({"result": [x for kv in self.db.get(key, {}).items() for x in kv]})
            elif cmd == "HKEYS":
                out.append({"result": list(self.db.get(key, {}))})
            else:
                out.append({"error": f"ERR unknown command '{cmd}'"})
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_labels_database_code_speaks_upstash_rest():
    if no_node():
        return
    with tempfile.TemporaryDirectory() as tmp:
        _syn, labels, site, _ = make_site(tmp)
        owner, raters = tokens(labels)
        sj = json.load(open(os.path.join(site, "content", "site.json")))
        ids = sj["batches"][0]["clip_ids"]
        up = ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstash)
        threading.Thread(target=up.serve_forever, daemon=True).start()
        env = {"KV_REST_API_URL": f"http://127.0.0.1:{up.server_address[1]}/", "KV_REST_API_TOKEN": FakeUpstash.token}
        srv = DevServer(site, os.path.join(tmp, "unused"), env)
        try:
            body = labels_body("s01", "EC", ids, ["no", "yes"])
            assert srv.req("/api/labels/s01/EC", raters["EC"], "PUT", body)[0] == 200
            assert srv.req("/api/labels/s01/EC", raters["EC"])[1] == body
            assert json.loads(srv.req("/api/batches", owner)[1])["batches"][0]["progress"] == {"EC": 2}
            assert json.loads(srv.req("/api/export", owner)[1])["files"] == {"labels_s01_EC.csv": body.decode()}
            assert ["SET", "HSET", "SET"] in FakeUpstash.seen
            keys = sorted(FakeUpstash.db)
            assert all(k.startswith(f"ied:{sj['study_id']}:") for k in keys)
            assert sum(":history:s01:EC:" in k for k in keys) == 1
            assert not os.path.exists(os.path.join(tmp, "unused"))
            # a database that refuses: the browser gets a plain message, never the database's details
            env_bad = {**env, "KV_REST_API_TOKEN": "wrong"}
            bad = DevServer(site, os.path.join(tmp, "unused"), env_bad)
            try:
                code, msg, _ = bad.req("/api/batches", owner)
                assert code == 503 and b"could not answer" in msg and b"401" not in msg
                assert bad.req("/api/batches", "nope")[0] == 403  # links are still checked first
            finally:
                bad.close()
        finally:
            srv.close()
            up.shutdown()
            up.server_close()


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


def test_results_page_fills_in_as_labels_arrive_and_needs_its_login():
    import pandas as pd
    with tempfile.TemporaryDirectory() as tmp:
        syn, labels, site, _ = make_site(tmp)
        env = {**os.environ, "IED_RESULTS_PASSWORD": "short"}
        out = subprocess.run([sys.executable, os.path.join(HOSTED, "make_site.py"), syn, "--labels", labels, "--out", site,
                              "--results-login", "sj"], capture_output=True, text=True, env=env)
        assert out.returncode != 0 and "12 characters" in out.stdout + out.stderr
        assert not os.path.exists(os.path.join(labels, ".results.json"))

        # before any labels: the page says what has come in and what it will show
        env["IED_RESULTS_PASSWORD"] = "correct horse battery"
        out = subprocess.run([sys.executable, os.path.join(HOSTED, "make_site.py"), syn, "--labels", labels, "--out", site,
                              "--results-login", "sj"], capture_output=True, text=True, env=env)
        assert out.returncode == 0, out.stderr
        assert "/results   username SJ" in out.stdout
        cfg = json.load(open(os.path.join(labels, ".results.json")))
        assert "battery" not in json.dumps(cfg) and len(cfg["hash"]) == 64
        site_json = json.load(open(os.path.join(site, "content", "site.json")))
        assert site_json["results_login"]["user"] == "SJ" and "battery" not in json.dumps(site_json)
        page = json.load(open(os.path.join(site, "content", "results.json")))
        assert "No answers yet" in page["html"] and "Not in yet" in page["html"]

        # with labels: the full page, patients as P01.., and the key to those names stays in the labels folder
        key = pd.read_csv(os.path.join(syn, "key.csv"), dtype=str, keep_default_na=False)
        for bid, g in key.groupby("batch_id"):
            for rater in (["EC", "JK"] if bid.startswith("s") else ["EC"]):
                dec = ["yes" if t == "spike" else "no" for t in g["truth"]]
                with open(os.path.join(labels, f"labels_{bid}_{rater}.csv"), "wb") as f:
                    f.write(labels_body(bid, rater, list(g["clip_id"]), dec))
        _, _, _, out = make_site(tmp)
        page = json.load(open(os.path.join(site, "content", "results.json")))
        html = page["html"]
        assert "{{updated}}" in html and "What's in so far" in html
        assert "P01" in html and not any(p in html for p in key["patient"].unique())
        assert "AC1, all raters" in html and "Cyclops" in html and "not in yet" in html
        mapping = pd.read_csv(os.path.join(labels, "results_patients.csv"))
        assert set(mapping["patient"]) == set(key["patient"]) and (os.stat(os.path.join(labels, "results_patients.csv")).st_mode & 0o077) == 0
        assert check_upload.check(site, labels) == []

        # the upload check refuses the page with a study ID or a date in it
        bad = dict(page, html=html.replace("P01", "EMU1234", 1))
        json.dump(bad, open(os.path.join(site, "content", "results.json"), "w"))
        assert any("EMU subject ID" in p for p in check_upload.check(site))
        json.dump(dict(page, html=html + "2026-10-26"), open(os.path.join(site, "content", "results.json"), "w"))
        assert any("holds a date" in p for p in check_upload.check(site))
        json.dump(page, open(os.path.join(site, "content", "results.json"), "w"))

        if no_node():
            return
        srv = DevServer(site, os.path.join(tmp, "store"))
        try:
            owner, _r = tokens(labels)
            status, body, headers = srv.req("/results")
            assert status == 401 and "Basic" in headers.get("WWW-Authenticate", "")
            assert srv.req("/results", token=owner)[0] == 401  # a link token does not open it
            assert srv.req("/results", auth=("SJ", "wrong password here"))[0] == 401
            assert srv.req("/results", auth=("XX", "correct horse battery"))[0] == 401
            status, body, headers = srv.req("/results", auth=("sj", "correct horse battery"))
            text = body.decode()
            assert status == 200 and headers["Content-Type"].startswith("text/html") and "IED Review Results" in text
            assert "{{updated}}" not in text and page["created"] in text and headers["Cache-Control"] == "no-store"
        finally:
            srv.close()
