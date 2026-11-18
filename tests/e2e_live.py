"""End-to-end check of the live route on made-up data: fake EDFs (one EDF+D) and fake detector output ->
select_clips from-runs -> sample -> serve_review --clips (no pack step) with rater links -> two raters label
in a browser -> labels on disk -> merge -> results report.
Usage: python tests/e2e_live.py <work_dir> <shots_dir>   (needs pyedflib, scipy, pandas, playwright)"""
import csv
import json
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
        sys.exit(f"step failed: {os.path.basename(args[0])} {args[1] if len(args) > 1 else ''}")
    return out.stdout


fake = os.path.join(work, "fake")
sec, clips, study, labels = (os.path.join(work, x) for x in ("seconds.csv", "clips.csv", "study", "labels"))
sc = os.path.join(KIT, "select_clips.py")
run(os.path.join(KIT, "tests", "make_fake_study.py"), fake)
out = run(sc, "inspect", os.path.join(fake, "runs", "cyclops"))
check("time=second" in out and "full19" in out, "inspect finds the time, score and condition columns")
out = run(sc, "from-runs", os.path.join(fake, "runs", "cyclops"), "--condition", "full19", "--detector", "cyclops",
          "--clip-root", os.path.join(fake, "edf"), "--admissions", os.path.join(fake, "admissions.csv"), "--out", sec)
check("3 admissions, 2 patients" in out, "from-runs maps three admissions to two patients")
out = run(sc, "sample", sec, "--out", clips, "--n", "12", "--bins", "0,0.5,1", "--min-gap-s", "12")
check("weights add to 582 of 582" in out, "sampler weights add up")
with open(clips) as f:
    rows = list(csv.DictReader(f))
gap_cross = [r for r in rows if "EMU9902" in r["source_path"] and float(r["center_s"]) + 6 > 90 and float(r["center_s"]) - 6 < 150]
check(not gap_cross, "no sampled clip crosses the EDF+D gap")
check(any("EMU9902" in r["source_path"] for r in rows), "the EDF+D recording contributes clips")

def serve(clip_table, study_dir, labels_dir, port="0"):
    proc = subprocess.Popen([py, "-u", os.path.join(KIT, "serve_review.py"), "--clips", clip_table, "--study", study_dir,
                             "--labels", labels_dir, "--raters", "SJ,EC", "--batch-size", "6", "--repeat-frac", "0.17",
                             "--port", port], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    found, printed = {}, []
    for line in proc.stdout:
        printed.append(line)
        m = re.match(r"\s+(SJ|EC)\s+(http://localhost:\d+/\?token=\S+)", line)
        if m:
            found[m.group(1)] = m.group(2)
        if line.startswith("Ctrl+C"):
            break
    return proc, found, "".join(printed)


srv, links, text = serve(clips, study, labels)
check(set(links) == {"SJ", "EC"}, "server printed one link per rater")
check("planned 3 batch(es): 12 clips plus 2 repeats" in text, "study planned with repeats")
check("could not be read" not in text, "every sampled clip read from the EDFs, EDF+D included")
plan = json.load(open(os.path.join(study, "plan.json")))
check(plan["left_out"] == [], "nothing left out of the plan")

try:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("IED_CHROMIUM") or None)
        errors = []

        def open_link(url, ctx=None):
            ctx = ctx or browser.new_context(viewport={"width": 1440, "height": 900})
            pg = ctx.new_page()
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.goto(url)
            pg.wait_for_selector("#pickModal:not([hidden]) .batches .btn", timeout=8000)
            return pg

        page = open_link(links["SJ"])
        check("Your answers (SJ)" in page.inner_text("#pickTable"), "SJ's link is signed in as SJ")
        page.locator("#pickTable .btn").first.click()
        page.wait_for_function("() => IEDReview.state.batch && !IEDReview.state.isDemo", timeout=10000)
        page.wait_for_timeout(300)
        check(page.locator("#raterModal").is_hidden(), "no initials prompt with a rater link")
        check(page.inner_text("#raterName") == "SJ", "top bar shows SJ")
        st = page.evaluate("() => ({id: IEDReview.state.batch.id, n: IEDReview.state.batch.clips.length, src: IEDReview.state.source.kind})")
        check(st == {"id": "b01", "n": 6, "src": "server"}, f"b01 read live from the EDFs: {st}")
        check(page.evaluate("() => IEDReview.state.settings.notch") is True, "the study opens with the 60 Hz notch on")
        page.screenshot(path=os.path.join(shots, "live_clip.png"))
        # a new study is Yes/No: no Unsure button, and Y and N each move on
        check(page.locator('#decide [data-d="unsure"]').is_hidden(), "a Yes/No study shows no Unsure button")
        check(page.locator("#maxSec").is_hidden(), "a Yes/No study asks no field maximum")
        page.keyboard.press("y")
        page.wait_for_timeout(250)
        page.keyboard.press("n")
        page.wait_for_timeout(250)
        page.keyboard.press("u")
        page.wait_for_timeout(1800)
        page.wait_for_selector("#syncChip.chip-ok", timeout=6000)
        check(page.evaluate("() => IEDReview.state.pos") == 2, "Y and N each moved to the next clip")
        got = list(csv.DictReader(open(os.path.join(labels, "labels_b01_SJ.csv"))))
        check(sorted(r["decision"] for r in got if r["decision"]) == ["no", "yes"], "SJ's two answers saved; U did nothing")
        check(all(not r["max_electrodes"] for r in got), "no maximum in the saved file")
        page.click("#raterBtn")
        page.wait_for_timeout(300)
        check(page.locator("#raterModal").is_hidden() and "signs you in as SJ" in page.inner_text("#toast"),
              "rater button cannot switch initials on a rater link")

        page2 = open_link(links["EC"])
        pick = page2.inner_text("#pickTable")
        check("Your answers (EC)" in pick and "2 of 6" not in pick, "EC sees only EC's progress")
        page2.locator("#pickTable .btn").first.click()
        page2.wait_for_function("() => IEDReview.state.batch && !IEDReview.state.isDemo", timeout=10000)
        page2.wait_for_timeout(300)
        n_ec = page2.evaluate("() => Object.values(IEDReview.state.labels).filter(l => l.decision).length")
        check(n_ec == 0 and page2.inner_text("#raterName") == "EC", "EC starts with no answers of SJ's")
        page2.keyboard.press("n")
        page2.wait_for_timeout(2200)
        check(os.path.exists(os.path.join(labels, "labels_b01_EC.csv")), "EC's answer saved to EC's own file")
        page2.screenshot(path=os.path.join(shots, "live_picker_ec.png"))

        # a second study on the same port reuses batch ID b01; SJ's browser must not carry answers across
        srv.terminate()
        srv.wait()
        port = re.search(r"localhost:(\d+)/", links["SJ"]).group(1)
        clips2 = os.path.join(work, "clips2.csv")
        run(sc, "sample", sec, "--out", clips2, "--n", "12", "--bins", "0,0.5,1", "--min-gap-s", "12", "--seed", "5")
        srv, links2, _t = serve(clips2, os.path.join(work, "study2"), os.path.join(work, "labels2"), port)
        page3 = open_link(links2["SJ"], page.context)
        page3.locator("#pickTable .btn").first.click()
        page3.wait_for_function("() => IEDReview.state.batch && !IEDReview.state.isDemo", timeout=10000)
        page3.wait_for_timeout(600)
        n3 = page3.evaluate("() => Object.values(IEDReview.state.labels).filter(l => l.decision).length")
        check(n3 == 0 and page3.inner_text("#progLabeled") == "0", f"a new study's b01 starts empty in the same browser ({n3})")
        check(not errors, f"no page errors {errors[:2]}")
        browser.close()
finally:
    srv.terminate()
    srv.wait()

status = run(os.path.join(KIT, "serve_review.py"), "--clips", clips, "--study", study, "--labels", labels, "--status")
print(status.rstrip())
check(re.search(r"b01\s+6\s+everyone\s+1\s+2", status) is not None, "status shows EC 1 and SJ 2 on b01")
merged = os.path.join(work, "merged.csv")
out = run(os.path.join(KIT, "merge_labels.py"), "--key", os.path.join(study, "key.csv"), "--out", merged,
          os.path.join(labels, "labels_b01_SJ.csv"), os.path.join(labels, "labels_b01_EC.csv"))
check("rater SJ: 2 of 6 answered" in out, "merge reads the live study's key and labels")
m = list(csv.DictReader(open(merged)))
check(all(r.get("sampling_weight") and r.get("clip") and r.get("detector") == "cyclops" for r in m),
      "merged rows carry weight, clip and detector from the run outputs")
run(os.path.join(KIT, "report_labels.py"), merged, "--out", os.path.join(work, "report.html"), "--title", "Fake study")
page_text = open(os.path.join(work, "report.html"), encoding="utf-8").read()
check("></i>Cyclops</td>" in page_text and "MORGOTH" not in page_text, "report scores Cyclops, the one detector in the run")
print("\nALL CHECKS PASSED" if not problems else f"\n{len(problems)} CHECK(S) FAILED")
sys.exit(1 if problems else 0)
