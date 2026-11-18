"""End-to-end check: open a packaged batch in the local viewer, label it like a rater would
(using the planted truth from key.csv), export the CSV, reload to confirm autosave, and
check the hosted demo mode. Usage: python tests/e2e.py <batch_dir> <shots_dir>"""
import csv
import os
import subprocess
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from playwright.sync_api import sync_playwright

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
batch_dir, shots = sys.argv[1], sys.argv[2]
os.makedirs(shots, exist_ok=True)
key = {r["clip_id"]: r for r in csv.DictReader(open(os.path.join(batch_dir, "key.csv")))}
local_url = "file://" + os.path.join(KIT, "dist", "ied_review.html")
problems = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        problems.append(msg)


with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=os.environ.get("IED_CHROMIUM") or None)
    ctx = browser.new_context(viewport={"width": 1440, "height": 900}, accept_downloads=True)
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" and "ERR_TUNNEL" not in m.text and "Failed to load resource" not in m.text else None)
    page.goto(local_url)
    page.wait_for_timeout(400)
    check(page.is_visible("#openBtn"), "local file shows Open batch")

    # open a batch: no rater stored yet, so the initials dialog must appear
    page.set_input_files("#fileInput", os.path.join(batch_dir, "b01.json.gz"))
    page.wait_for_selector("#raterModal:not([hidden])", timeout=5000)
    page.fill("#raterInput", "e c")
    page.click("#raterForm button[type=submit]")
    check(page.inner_text("#raterError") != "", "initials with a space are rejected")
    page.fill("#raterInput", "ec")
    page.click("#raterForm button[type=submit]")
    page.wait_for_timeout(300)
    st = page.evaluate("() => ({id: IEDReview.state.batch.id, n: IEDReview.state.batch.clips.length, mode: IEDReview.state.batch.mode, rater: IEDReview.state.rater, order: IEDReview.state.order.slice(0, 5)})")
    check(st["id"] == "b01" and st["rater"] == "EC" and st["mode"] == "rater", f"batch loaded for rater EC: {st}")
    check(st["order"] != [0, 1, 2, 3, 4], "clip order is shuffled for the rater")
    check(page.is_hidden("#detailsSec"), "blinded batch shows no clip details")

    # label every clip the way a careful rater would, from the planted truth. A blinded Yes/No batch asks one
    # question: no field maximum, artifact flag or other-discharge marks
    check(all(page.is_hidden(f"#{x}") for x in ("maxSec", "artRow", "marksRow", "markCtl")),
          "a Yes/No batch shows only Yes, No and the note")
    marked_note = None
    for k in range(st["n"]):
        cid = page.evaluate("() => IEDReview.state.batch.clips[IEDReview.state.order[IEDReview.state.pos]].id")
        truth = key[cid]
        if truth["truth"] == "spike":
            if marked_note is None:
                page.click("#noteInput")
                page.keyboard.type('sharp, "classic", with aftergoing slow wave')
                page.keyboard.press("Escape")
                marked_note = cid
            page.keyboard.press("y")
        else:
            page.keyboard.press("n")
        page.wait_for_timeout(190)
    done = page.evaluate("() => Object.values(IEDReview.state.labels).filter(l => l.decision).length")
    check(done == st["n"], f"all {st['n']} clips answered ({done})")
    page.screenshot(path=os.path.join(shots, "local_batch.png"))

    # export and download
    page.click("#exportBtn")
    page.wait_for_selector("#exportModal:not([hidden])")
    check(page.is_visible("#downloadBtn") and page.is_hidden("#exportHostedNote"), "local export offers a download")
    with page.expect_download() as dl:
        page.click("#downloadBtn")
    out_csv = os.path.join(shots, dl.value.suggested_filename)
    dl.value.save_as(out_csv)
    rows = list(csv.DictReader(open(out_csv)))
    check(len(rows) == st["n"] and all(r["rater"] == "EC" for r in rows), f"exported {len(rows)} rows for EC as {dl.value.suggested_filename}")
    noted = [r for r in rows if r["clip_id"] == marked_note][0]
    check(noted["decision"] == "yes" and noted["note"].startswith("sharp"), f"answer and note exported: {noted['decision']!r} {noted['note']!r}")
    page.keyboard.press("Escape")

    # reload: labels come back from browser storage
    page.reload()
    page.wait_for_timeout(300)
    page.set_input_files("#fileInput", os.path.join(batch_dir, "b01.json.gz"))
    page.wait_for_timeout(500)
    back = page.evaluate("() => ({rater: IEDReview.state.rater, n: Object.values(IEDReview.state.labels).filter(l => l.decision).length})")
    check(back["rater"] == "EC" and back["n"] == st["n"], f"after reload, rater remembered and {back['n']} labels restored")

    # a malformed file gives a readable error instead of a crash
    bad = os.path.join(shots, "not_a_batch.json")
    open(bad, "w").write('{"hello": 1}')
    page.set_input_files("#fileInput", bad)
    page.wait_for_timeout(300)
    check("not an IED review batch" in page.inner_text("#toast"), "malformed file shows an error toast")
    check(not errors, f"no script errors in local mode {errors}")
    ctx.close()

    # hosted mode: serve over http on a non-localhost name, as the artifact viewer would
    handler = partial(SimpleHTTPRequestHandler, directory=os.path.join(KIT, "dist"))
    srv = ThreadingHTTPServer(("0.0.0.0", 8765), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    ctx = browser.new_context(viewport={"width": 1440, "height": 900}, color_scheme="dark")
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto("http://0.0.0.0:8765/ied_review.html")
    page.wait_for_timeout(500)
    check(page.is_hidden("#openBtn") and page.is_visible("#demoSeg"), "hosted mode hides Open batch and shows the demo views")
    page.click('#demoSeg button[data-view="research"]')
    page.wait_for_timeout(300)
    check(page.is_visible("#detailsSec"), "research view shows clip details")
    page.keyboard.press("y")
    page.click('#headmap .el[data-e="T3"]')
    page.click("#exportBtn")
    check(page.is_hidden("#downloadBtn") and page.is_visible("#exportHostedNote"), "hosted export offers copy, not download")
    page.keyboard.press("Escape")
    page.screenshot(path=os.path.join(shots, "hosted_research_dark.png"))
    ctx.close()

    ctx = browser.new_context(viewport={"width": 400, "height": 860}, device_scale_factor=2)
    page = ctx.new_page()
    page.goto("http://0.0.0.0:8765/ied_review.html")
    page.wait_for_timeout(500)
    sw = page.evaluate("() => [document.documentElement.scrollWidth, window.innerWidth]")
    check(sw[0] <= sw[1], f"no horizontal scroll at phone width {sw}")
    page.screenshot(path=os.path.join(shots, "phone.png"), full_page=True)
    check(not errors, f"no script errors in hosted mode {errors}")
    ctx.close()
    srv.shutdown()
    browser.close()

print("\nmerge check:")
subprocess.run([sys.executable, os.path.join(KIT, "merge_labels.py"), "--key", os.path.join(batch_dir, "key.csv"), out_csv,
                "--out", os.path.join(shots, "merged.csv")], check=True)
print("\n" + ("ALL CHECKS PASSED" if not problems else f"{len(problems)} problem(s)"))
sys.exit(1 if problems else 0)
