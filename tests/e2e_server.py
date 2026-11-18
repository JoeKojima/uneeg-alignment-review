"""End-to-end check of the Borel route on fake data: fake EDFs -> fake MORGOTH seconds -> select_clips ->
pack -> serve_review -> label in a browser -> labels on disk -> second browser picks them up -> merge.
Usage: python tests/e2e_server.py <work_dir> <shots_dir>   (needs pyedflib, scipy, pandas, playwright)"""
import csv
import os
import re
import shutil
import subprocess
import sys

from playwright.sync_api import sync_playwright

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
work, shots = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
shutil.rmtree(work, ignore_errors=True)  # a fresh study every run; old labels would change the counts
os.makedirs(work, exist_ok=True)
os.makedirs(shots, exist_ok=True)
py = sys.executable
problems = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        problems.append(msg)


def run(*args):
    out = subprocess.run([py, *args], capture_output=True, text=True)
    if out.returncode:
        print(out.stdout, out.stderr)
        sys.exit(f"step failed: {args[0]}")
    return out.stdout


edf, sec, clips, batches, labels = (os.path.join(work, x) for x in ("edf", "seconds.csv", "clips.csv", "batches", "labels"))
run(os.path.join(KIT, "tests", "make_test_edfs.py"), edf)
run(os.path.join(KIT, "select_clips.py"), "fake-morgoth", "--edf-dir", edf, "--out", sec)
print(run(os.path.join(KIT, "select_clips.py"), "sample", sec, "--out", clips, "--n", "16", "--bins", "0,0.2,0.5,1",
          "--max-patient-frac", "1").strip().splitlines()[1])
print(run(os.path.join(KIT, "make_review_batches.py"), "pack", clips, "--out", batches, "--batch-size", "8",
          "--answers", "ynu").strip().splitlines()[-3])

srv = subprocess.Popen([py, "-u", os.path.join(KIT, "serve_review.py"), batches, "--labels", labels, "--port", "0"],
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
url = None
for line in srv.stdout:
    m = re.search(r"(http://localhost:\d+/\?token=\S+)", line)
    if m:
        url = m.group(1)
        break
check(url is not None, "server printed a link")
check(not os.listdir(labels) or os.listdir(labels) == [".token", "history"] or set(os.listdir(labels)) <= {".token", "history"},
      "labels folder starts empty")

try:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("IED_CHROMIUM") or None)
        errors = []

        def page_in_new_context():
            ctx = browser.new_context(viewport={"width": 1440, "height": 900})
            pg = ctx.new_page()
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.goto(url)
            pg.wait_for_selector("#pickModal:not([hidden]) .batches .btn", timeout=8000)
            return pg

        page = page_in_new_context()
        rows = page.locator("#pickTable tr").count()
        check(rows == 3, f"picker lists two batches ({rows - 1})")
        check("key" not in page.inner_text("#pickTable").lower(), "picker shows no key file")
        page.screenshot(path=os.path.join(shots, "server_picker.png"))
        page.locator("#pickTable .btn").first.click()
        page.wait_for_selector("#raterModal:not([hidden])", timeout=5000)
        page.fill("#raterInput", "sj")
        page.click("#raterForm button[type=submit]")
        page.wait_for_timeout(500)
        st = page.evaluate("() => ({id: IEDReview.state.batch.id, src: IEDReview.state.source, n: IEDReview.state.batch.clips.length})")
        check(st["id"] == "b01" and st["src"]["kind"] == "server", f"batch b01 opened from the server: {st}")

        page.keyboard.press("y")
        page.click('#headmap .el[data-e="T3"]')
        page.keyboard.press("Enter")
        page.keyboard.press("n")
        page.wait_for_timeout(250)
        page.keyboard.press("n")
        page.wait_for_timeout(250)
        page.keyboard.press("u")
        page.wait_for_selector("#syncChip.chip-ok", timeout=6000)
        page.wait_for_timeout(1800)
        page.wait_for_selector("#syncChip.chip-ok", timeout=6000)
        check(page.inner_text("#syncChip").startswith("Saved to server"), "chip says saved to server")
        page.screenshot(path=os.path.join(shots, "server_labeling.png"))
        path = os.path.join(labels, "labels_b01_SJ.csv")
        got = list(csv.DictReader(open(path))) if os.path.exists(path) else []
        dec = sorted(r["decision"] for r in got if r["decision"])
        check(dec == ["no", "no", "unsure", "yes"], f"labels file on the server has the four answers: {dec}")
        check(any(r["max_electrodes"] == "T3" for r in got), "field maximum saved")

        # a fresh browser (no local storage) picks the labels back up from the server
        page2 = page_in_new_context()
        check("4 of 8" in page2.inner_text("#pickTable"), "picker shows progress from the server")
        page2.locator("#pickTable .btn").first.click()
        page2.wait_for_selector("#raterModal:not([hidden])", timeout=5000)
        page2.fill("#raterInput", "SJ")
        page2.click("#raterForm button[type=submit]")
        page2.wait_for_timeout(1200)
        n2 = page2.evaluate("() => Object.values(IEDReview.state.labels).filter(l => l.decision).length")
        check(n2 == 4, f"second browser restored 4 labels from the server ({n2})")
        page2.keyboard.press("Shift+ArrowRight")
        page2.keyboard.press("y")
        page2.keyboard.press("g")
        page2.wait_for_timeout(2500)
        got = list(csv.DictReader(open(path)))
        check(sum(1 for r in got if r["decision"]) == 5, "edit from the second browser saved")
        check(any(r["generalized"] == "1" for r in got), "generalized flag saved")

        # a different rater gets their own file and order
        page2.evaluate("() => document.getElementById('raterBtn').click()")
        page2.fill("#raterInput", "EC")
        page2.click("#raterForm button[type=submit]")
        page2.wait_for_timeout(600)
        page2.keyboard.press("n")
        page2.wait_for_timeout(2500)
        check(os.path.exists(os.path.join(labels, "labels_b01_EC.csv")), "second rater saved to their own file")
        check(not errors, f"no page errors {errors[:2]}")
        browser.close()
finally:
    srv.terminate()
    srv.wait()

status = run(os.path.join(KIT, "serve_review.py"), batches, "--labels", labels, "--status")
print(status.rstrip())
merged = os.path.join(work, "merged.csv")
out = run(os.path.join(KIT, "merge_labels.py"), "--key", os.path.join(batches, "key.csv"), "--out", merged,
          os.path.join(labels, "labels_b01_SJ.csv"), os.path.join(labels, "labels_b01_EC.csv"))
check("rater SJ: 5 of 8 answered" in out, "merge reads the server's label files")
m = list(csv.DictReader(open(merged)))
check(all(r.get("sampling_weight") for r in m), "merged rows carry sampling_weight from the clip table")
print("\nALL CHECKS PASSED" if not problems else f"\n{len(problems)} CHECK(S) FAILED")
sys.exit(1 if problems else 0)
