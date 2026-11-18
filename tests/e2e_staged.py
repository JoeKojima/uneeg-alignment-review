"""End-to-end check of a staged study (kit 1.8) on synthetic data: select_clips.py parts -> make_site.py with
--tolerance-ms -> the site served the way Vercel would (tools/dev_server.mjs) -> a rater link in a browser that
meets the 10-20 part, then the training examples, then the UNEEG part (kit 1.8.4 order), each opening only when the
one before is answered, with the 10-20 or UNEEG manual opening by itself as each part starts. Usage: python tests/e2e_staged.py <work_dir> <shots_dir>   (needs node, playwright)"""
import csv
import gzip
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import urllib.request

import pandas as pd
from playwright.sync_api import sync_playwright

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT)
sys.path.insert(0, os.path.join(KIT, "tests"))
import serve_review  # noqa: E402
from test_phase5 import staged_study  # noqa: E402

work, shots = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
shutil.rmtree(work, ignore_errors=True)
os.makedirs(work)
os.makedirs(shots, exist_ok=True)
problems = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        problems.append(msg)


staged, _sq, _c = staged_study(work)
t = pd.read_csv(staged, dtype=str, keep_default_na=False)
t.loc[t.index[t["part"] == "examples"][0], "ref_note"] = "Sharp wave over the left temporal chain."
t.loc[t.index[t["part"] == "examples"][0], "ref_type"] = "focal left"
# the examples show the UNEEG channels under the scalp (kit 1.8.1): give them the SubQ file, 100 s ahead
ex = t["part"] == "examples"
t.loc[ex, "subq_path"] = t.loc[t["part"] == "uneeg", "subq_path"].iloc[0]
t.loc[ex, "subq_center_s"] = [f"{float(x) + 100:.1f}" for x in t.loc[ex, "center_s"]]
t.to_csv(staged, index=False)
study, labels, site = (os.path.join(work, x) for x in ("study", "labels", "site"))
out = subprocess.run([sys.executable, os.path.join(KIT, "hosted", "make_site.py"), "--clips", staged, "--study", study,
                      "--labels", labels, "--out", site, "--raters", "EC,JK", "--admin-raters", "EC",
                      "--admin", os.path.join(work, "admin.csv"), "--batch-size", "4", "--question", "second",
                      "--tolerance-ms", "250", "--url", "https://ied-review-test.vercel.app",
                      "--manual", "scalp=" + os.path.join(KIT, "rater_manual_scalp.md"),
                      "--manual", "uneeg=" + os.path.join(KIT, "rater_manual_uneeg.md")],
                     capture_output=True, text=True)
check(out.returncode == 0 and "upload check OK" in out.stdout, "make_site wrote the staged site")
if out.returncode:
    print(out.stdout, out.stderr)
    sys.exit(1)
meta = json.load(open(os.path.join(site, "content", "site.json")))
ids = [b["batch_id"] for b in meta["batches"]]
batches = {bid: json.loads(gzip.decompress(open(os.path.join(site, "content", "batches", bid + ".json.gz"), "rb").read()))
           for bid in ids}
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
site_base = f"http://ied-review.test:{port}"
ec = serve_review.load_rater_tokens(labels, False)["EC"]
jk = serve_review.load_rater_tokens(labels, False)["JK"]


def put_answers(bid, decisions):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=serve_review.EXPORT_COLUMNS, lineterminator="\n")
    w.writeheader()
    for i, (clip, d) in enumerate(zip(batches[bid]["clips"], decisions)):
        w.writerow({"batch_id": bid, "clip_id": clip["id"], "rater": "EC", "order_index": i + 1, "decision": d,
                    "labeled_at": "2026-11-04T00:00:00Z"})
    r = urllib.request.Request(f"http://127.0.0.1:{port}/api/labels/{bid}/EC", data=buf.getvalue().encode(),
                               method="PUT", headers={"X-Review-Token": ec})
    with urllib.request.urlopen(r) as resp:
        return resp.status


def wait_pos(page, want, tries=40):
    """The page's CSP refuses string evaluation, so poll the position from here."""
    for _ in range(tries):
        if page.evaluate("window.IEDReview.state.pos") == want:
            return True
        page.wait_for_timeout(100)
    return False


srv = subprocess.Popen(["node", os.path.join(site, "tools", "dev_server.mjs"), "--port", str(port), "--store",
                        os.path.join(work, "store")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
srv.stdout.readline()
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("IED_CHROMIUM") or None,
                                    args=["--host-resolver-rules=MAP ied-review.test 127.0.0.1", "--no-proxy-server"])
        errors = []
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        page = ctx.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{site_base}/?token={ec}")
        page.wait_for_selector("#pickModal:not([hidden]) .batches .btn", timeout=8000)
        pick = page.inner_text("#pickTable")
        check("Part 1: 10-20 clips" in pick and "Part 2: UNEEG training examples" in pick and "Part 3: UNEEG clips" in pick,
              "the batch list shows the three parts in order: 10-20, examples, UNEEG")
        sids = [b for b in ids if b.startswith("s")]
        locked = page.locator("#pickTable button:disabled").count()
        check(locked == len(ids) - len(sids), f"every batch after the 10-20 part is locked ({locked} of {len(ids) - len(sids)})")
        check(page.is_visible("#manualBtn"), "the header shows a Manual button")
        page.screenshot(path=os.path.join(shots, "staged_picker.png"))

        def manual_open():
            return page.is_visible("#manualModal")

        def wait_manual(want, tries=40):
            for _ in range(tries):
                if manual_open() == want:
                    return True
                page.wait_for_timeout(100)
            return False

        # the 10-20 part: its manual opens by itself on the first batch
        page.click("#pickTable tr:has-text('s01') button")
        page.wait_for_selector("#pickModal", state="hidden")
        check(wait_manual(True), "starting the 10-20 part opens the 10-20 manual by itself")
        check(page.inner_text("#manualTitle") == "IED Review: 10-20 Manual", f"it is the 10-20 manual ({page.inner_text('#manualTitle')!r})")
        body = page.inner_text("#manualBody")
        check(page.locator("#manualBody table td").count() >= 12 and "negative up" in body,
              "the 10-20 manual shows the key table and says the traces are negative up")
        check("Reading the UNEEG clips" not in body, "the 10-20 manual leaves the UNEEG reading to the UNEEG manual")
        check(page.get_attribute("#manualTab", "href") == "/manual_scalp.html", "its new-tab link is the 10-20 page")
        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_timeout(200)
        check(page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"),
              "the manual fits a phone screen without sideways scrolling")
        page.screenshot(path=os.path.join(shots, "staged_manual_phone.png"))
        page.set_viewport_size({"width": 1440, "height": 900})
        page.screenshot(path=os.path.join(shots, "staged_manual_scalp.png"))
        page.click("#manualClose")
        check(wait_manual(False), "Got it closes the manual")
        with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/manual_scalp.html",
                                                           headers={"Host": f"ied-review.test:{port}"})) as resp:
            check(resp.status == 200 and "frame-ancestors 'none'" in resp.headers.get("Content-Security-Policy", ""),
                  "the 10-20 page is served with the site's security headers")
        check(page.is_visible("#questionHint") and "250 ms" in page.inner_text("#questionHint"),
              "the question says to count spikes within 250 ms of the shading")
        check(page.is_visible("#montageCtl"), "10-20 clips keep the montage control")
        page.wait_for_timeout(300)
        page.locator("#chart").screenshot(path=os.path.join(shots, "staged_scalp_margin.png"))
        page.click("#manualBtn")
        check(wait_manual(True) and "10-20" in page.inner_text("#manualTitle"), "the Manual button opens the 10-20 manual again")
        page.keyboard.press("Escape")
        check(wait_manual(False), "Escape closes it")
        # the other 10-20 batches by API; then s01 is finished here in the browser, and the page says the examples are open
        for bid in sids[1:]:
            check(put_answers(bid, ["no"] * len(batches[bid]["clips"])) == 200, f"answers for {bid} saved")
        page.click("#openBtn")
        page.wait_for_selector("#pickModal:not([hidden]) .batches .btn")
        page.click("#pickTable tr:has-text('s01') button")
        page.wait_for_selector("#pickModal", state="hidden")
        page.wait_for_timeout(300)
        check(not manual_open(), "a part already started does not open its manual again")
        for i in range(len(batches["s01"]["clips"])):
            page.keyboard.press("n")
            page.wait_for_timeout(200)
        page.wait_for_timeout(300)
        toast_text = page.inner_text("#toast") if page.locator("#toast").count() else ""
        check("UNEEG training examples are open now" in toast_text, f"finishing the 10-20 part says the examples are open ({toast_text!r})")
        for _ in range(40):
            r = urllib.request.Request(f"http://127.0.0.1:{port}/api/batches", headers={"X-Review-Token": ec})
            with urllib.request.urlopen(r) as resp:
                if next(x for x in json.load(resp)["batches"] if x["batch_id"] == "s01")["progress"].get("EC") == \
                        len(batches["s01"]["clips"]):
                    break
            page.wait_for_timeout(250)
        else:
            check(False, "the browser's s01 answers reached the server")

        # the training examples open after the 10-20 part, with the UNEEG manual; the UNEEG part still waits
        page.click("#openBtn")
        page.wait_for_selector("#pickModal:not([hidden]) .batches .btn")
        page.wait_for_timeout(300)
        e_row = page.locator("#pickTable tr:has-text('e01') button")
        u_row = page.locator("#pickTable tr:has-text('u01') button")
        check(e_row.is_enabled() and not u_row.is_enabled(), "after the 10-20 part the examples open, the UNEEG part does not")
        e_row.click()
        page.wait_for_selector("#pickModal", state="hidden")
        check(wait_manual(True) and page.inner_text("#manualTitle") == "IED Review: UNEEG Manual",
              "starting the examples opens the UNEEG manual by itself")
        body = page.inner_text("#manualBody")
        check("What to look for on the UNEEG channels" in body and "D-C" in body and "heartbeat" in body.lower(),
              "the UNEEG manual says what to look for, names the channels and warns about the heartbeat")
        # kit 1.8.6: the UNEEG manual draws the lead, its three contacts and the three pairs, near the top
        fig = page.locator("#manualBody figure svg.lead-svg")
        check(fig.count() == 1 and page.locator("#manualBody figure svg .contact").count() == 3,
              "the UNEEG manual draws the lead with its three contacts")
        ftxt = page.inner_text("#manualBody figure")
        check(all(w in ftxt for w in ("distal (SQD)", "centre (SQC)", "proximal (SQP)", "D-C", "P-C",
                                      "D-P, twice the distance")), "it names the contacts and the three pairs")
        box = fig.bounding_box()
        check(box and box["width"] > 300 and box["height"] > 80, f"the drawing is a readable size ({box})")
        check(page.evaluate("getComputedStyle(document.querySelector('#manualBody .contact')).fill") not in ("", "none", "rgb(0, 0, 0)"),
              "the contacts take the accent colour")
        fig.scroll_into_view_if_needed()
        page.locator("#manualBody figure").screenshot(path=os.path.join(shots, "staged_uneeg_lead.png"))
        page.evaluate("document.getElementById('manualCard').scrollTop = 0")
        page.screenshot(path=os.path.join(shots, "staged_manual_uneeg.png"))
        page.click("#manualClose")
        check(wait_manual(False), "Got it closes the UNEEG manual")
        eid = next(b for b in ids if b.startswith("e"))
        n_ex = len(batches[eid]["clips"])
        check(not page.is_visible("#decide"), "an example has no Yes or No buttons (kit 1.8.5: shown, not asked)")
        check(page.is_visible("#reveal"), "an example shows the expert's read from the start")
        check("Training example" in page.inner_text("#questionH"), "the panel is headed Training example")
        ex_labels = page.evaluate("""(() => { const st = window.IEDReview.state, b = st.batch;
            const sig = Core.clipSignals(b, b.clips[0]);
            return Core.deriveRows(st.settings.montage, sig.scalp, sig.extras, st.hiddenGroups, b.view)
              .map(r => r.label + (r.values ? '' : ' (dashed)')); })()""")
        check(ex_labels[-3:] == ["UNEEG D-C", "UNEEG P-C", "UNEEG D-P"] and "Fp1-F7" in ex_labels,
              f"an example draws the scalp montage with D-C, P-C and D-P under it ({ex_labels[-4:]})")
        check(not any("dashed" in x for x in ex_labels[-3:]), "the examples' UNEEG lanes carry signal")
        check(page.is_visible("#montageCtl"), "examples keep the montage control")
        check("UNEEG" in page.inner_text("#decideHint") and "seen all" in page.inner_text("#decideHint"),
              "the hint says the UNEEG channels are under the scalp and the UNEEG clips open once all are seen")
        check("+ UNEEG" in page.inner_text("#chartFoot"), "the footer names the montage and UNEEG")
        reads = []
        pos0 = page.evaluate("window.IEDReview.state.pos")
        for i in range(n_ex):
            reads.append(page.inner_text("#reveal"))
            if i == 0:
                page.keyboard.press("y")
                page.wait_for_timeout(200)
                check(page.evaluate("window.IEDReview.state.pos") == pos0, "Y does nothing on an example")
                page.locator("#chart").screenshot(path=os.path.join(shots, "staged_example_both.png"))
                page.screenshot(path=os.path.join(shots, "staged_training.png"))
            if i < n_ex - 1:
                page.keyboard.press("ArrowRight")
                wait_pos(page, pos0 + i + 1)
        check(any("Focal spike, left" in r and "left temporal" in r for r in reads),
              f"the example with a type and a note shows both up front ({reads})")
        check(all(r.strip() for r in reads), "every example shows the expert's read")
        page.wait_for_timeout(400)
        toast_text = page.inner_text("#toast") if page.locator("#toast").count() else ""
        check("UNEEG clips are open now" in toast_text, f"seeing the last example says the UNEEG clips are open ({toast_text!r})")
        decisions = page.evaluate("Object.values(window.IEDReview.state.labels).map(l => l.decision)")
        check(decisions.count("seen") == n_ex, f"each example counts as seen ({decisions})")
        page.click("#manualBtn")
        check(wait_manual(True) and "UNEEG" in page.inner_text("#manualTitle"), "in the examples the Manual button opens the UNEEG manual")
        page.click("#manualClose")
        wait_manual(False)
        for _ in range(40):  # every example seen on the server before the list is asked again
            r = urllib.request.Request(f"http://127.0.0.1:{port}/api/batches", headers={"X-Review-Token": ec})
            with urllib.request.urlopen(r) as resp:
                if next(x for x in json.load(resp)["batches"] if x["batch_id"] == eid)["progress"].get("EC") == n_ex:
                    break
            page.wait_for_timeout(250)
        else:
            check(False, "the seen examples reached the server")

        # the UNEEG part: three channels, no montage
        page.click("#openBtn")
        page.wait_for_selector("#pickModal:not([hidden]) .batches .btn")
        page.wait_for_timeout(300)
        u_row = page.locator("#pickTable tr:has-text('u01') button")
        check(u_row.is_enabled(), "with the examples answered, the UNEEG part opens")
        u_row.click()
        page.wait_for_selector("#pickModal", state="hidden")
        page.wait_for_timeout(400)
        check(not manual_open(), "the UNEEG manual, already shown on this visit, does not open again for the UNEEG part")
        labels_drawn = page.evaluate("""(() => { const st = window.IEDReview.state, b = st.batch;
            const sig = Core.clipSignals(b, b.clips[0]);
            return Core.deriveRows(st.settings.montage, sig.scalp, sig.extras, st.hiddenGroups, b.view).map(r => r.label); })()""")
        check(labels_drawn == ["D-C", "P-C", "D-P"], f"a UNEEG clip draws only D-C, P-C and D-P ({labels_drawn})")
        check(not page.is_visible("#montageCtl"), "a UNEEG clip has no montage control")
        hint = page.inner_text("#decideHint")
        check("banana" not in hint, f"the hint does not offer montages on a UNEEG clip ({hint!r})")
        before = page.evaluate("window.IEDReview.state.settings.montage")
        page.keyboard.press("t")
        check(page.evaluate("window.IEDReview.state.settings.montage") == before, "T does nothing on a UNEEG clip")
        page.wait_for_timeout(300)
        page.screenshot(path=os.path.join(shots, "staged_uneeg.png"))
        # the admin view (kit 1.8.5, SJ and EC only): S draws the 10-20 of the same 12 s above the UNEEG lanes
        for _ in range(40):
            if page.evaluate("window.IEDReview.state.admin.batchId") == "u01":
                break
            page.wait_for_timeout(100)
        page.keyboard.press("s")
        page.wait_for_timeout(400)
        check(page.evaluate("window.IEDReview.state.admin.subq") is True, "S turns the admin view on for EC on a UNEEG clip")
        drawn = page.evaluate("window.IEDReview.rowsNow().map(r => r.label + (r.values ? '' : ' (dashed)'))")
        check("Fp1-F7" in drawn and drawn.index("Fp1-F7") < drawn.index("UNEEG D-C"),
              f"the admin view draws the 10-20 montage above the UNEEG lanes ({drawn[:2]} ... {drawn[-4:]})")
        check(not any(x.startswith("Fp1-F7 (dashed") for x in drawn), "the admin 10-20 lanes carry signal")
        check(page.is_visible("#montageCtl"), "with the 10-20 shown, the montage control comes back")
        page.wait_for_timeout(300)
        page.locator("#chart").screenshot(path=os.path.join(shots, "staged_uneeg_admin.png"))
        page.keyboard.press("s")
        page.wait_for_timeout(200)
        check(not page.is_visible("#montageCtl"), "S again hides the 10-20 and the montage control")
        page.keyboard.press("y")
        check(wait_pos(page, 1), "Yes on a UNEEG clip moves on")
        # another rater has no admin view
        r = urllib.request.Request(f"http://127.0.0.1:{port}/api/batches", headers={"X-Review-Token": jk})
        with urllib.request.urlopen(r) as resp:
            check(not json.load(resp).get("admin"), "a rater without the admin view is not offered it")
        try:
            urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/api/admin/s01",
                                                          headers={"X-Review-Token": jk}))
            check(False, "another rater cannot fetch the admin view")
        except urllib.error.HTTPError as e:
            check(e.code in (403, 404), f"another rater cannot fetch the admin view ({e.code})")
        page.click("#manualBtn")
        check(wait_manual(True) and "UNEEG" in page.inner_text("#manualTitle"), "in the UNEEG part the Manual button opens the UNEEG manual")
        page.click("#manualClose")
        wait_manual(False)

        # coming back later: a part already started does not open its manual again; a fresh one would
        for _ in range(40):
            r = urllib.request.Request(f"http://127.0.0.1:{port}/api/batches", headers={"X-Review-Token": ec})
            with urllib.request.urlopen(r) as resp:
                if any(x["progress"].get("EC") for x in json.load(resp)["batches"] if x["batch_id"].startswith("u")):
                    break
            page.wait_for_timeout(250)
        page.goto(f"{site_base}/?token={ec}")
        page.wait_for_selector("#pickModal:not([hidden]) .batches .btn", timeout=8000)
        page.click("#pickTable tr:has-text('u01') button")
        page.wait_for_selector("#pickModal", state="hidden")
        page.wait_for_timeout(500)
        check(not manual_open(), "after a reload, a part with answers in it does not open its manual by itself")
        check(not errors, f"no page errors ({errors[:3]})")
        browser.close()
finally:
    srv.terminate()
    srv.wait()

print("ALL PASSED" if not problems else f"{len(problems)} FAILED")
sys.exit(1 if problems else 0)
