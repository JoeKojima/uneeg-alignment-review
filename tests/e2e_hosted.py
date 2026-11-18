"""End-to-end check of the hosted route on synthetic data: synthetic batches -> make_site.py -> the site served
the way Vercel would (tools/dev_server.mjs) -> rater links in a browser -> labels database -> pull_labels.py ->
merge_labels.py. Usage: python tests/e2e_hosted.py <work_dir> <shots_dir>   (needs node, playwright)"""
import csv
import json
import os
import re
import shutil
import socket
import subprocess
import sys

from playwright.sync_api import sync_playwright

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
work, shots = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
shutil.rmtree(work, ignore_errors=True)
os.makedirs(work)
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


syn, labels, site = (os.path.join(work, x) for x in ("syn", "labels", "site"))
run(os.path.join(KIT, "make_review_batches.py"), "synthetic", "--n", "30", "--out", syn, "--batch-size", "10",
    "--shared-clips", "10", "--seed", "11")
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
base = f"http://127.0.0.1:{port}"
# The browser opens the site under a name that is not localhost, as on Vercel; the viewer treats localhost as
# the Borel tunnel, so a test on 127.0.0.1 alone would not show how the hosted page behaves.
site_base = f"http://ied-review.test:{port}"
out = run(os.path.join(KIT, "hosted", "make_site.py"), syn, "--labels", labels, "--out", site, "--raters", "EC,JK",
          "--url", "https://ied-review-test.vercel.app")
links = dict(re.findall(r"^\s+(EC|JK)\s+(https://\S+)$", out, re.M))
owner_link = re.search(r"^\s+(https://\S+)$", out, re.M).group(1)
check(set(links) == {"EC", "JK"} and "upload check OK" in out, "make_site printed a link per rater and passed the check")
links = {k: v.replace("https://ied-review-test.vercel.app", site_base) for k, v in links.items()}
assignment = json.load(open(os.path.join(site, "content", "site.json")))["assignment"]["by_rater"]

srv = subprocess.Popen(["node", os.path.join(site, "tools", "dev_server.mjs"), "--port", str(port), "--store",
                        os.path.join(work, "store")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
srv.stdout.readline()
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("IED_CHROMIUM") or None,
                                    args=["--host-resolver-rules=MAP ied-review.test 127.0.0.1", "--no-proxy-server"])
        errors, csp = [], []

        def open_link(url):
            ctx = browser.new_context(viewport={"width": 1440, "height": 900})
            pg = ctx.new_page()
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.on("console", lambda m: csp.append(m.text) if "Content Security Policy" in m.text else None)
            pg.goto(url)
            pg.wait_for_selector("#pickModal:not([hidden]) .batches .btn", timeout=8000)
            return pg

        page = open_link(links["EC"])
        pick = page.inner_text("#pickTable")
        rows = [r.split("\t")[0] for r in pick.strip().splitlines()[1:]]
        check(rows[0].startswith("s01 · shared"), f"the shared batch is listed first ({rows[0]!r})")
        check(sorted(r.split(" ")[0] for r in rows[1:]) == assignment["EC"], f"EC sees only EC's own batches ({rows})")
        check("Your answers (EC)" in pick, "EC's link is signed in as EC")
        page.screenshot(path=os.path.join(shots, "hosted_picker.png"))
        page.locator("#pickTable .btn").first.click()
        page.wait_for_function("() => IEDReview.state.batch && !IEDReview.state.isDemo", timeout=10000)
        page.wait_for_timeout(300)
        check(page.locator("#raterModal").is_hidden() and page.inner_text("#raterName") == "EC", "no initials prompt")
        check(page.locator('#decide [data-d="unsure"]').is_hidden(), "no Unsure button in a Yes/No study")
        check("each save and move to the next clip" in page.inner_text("#decideHint"), "the hint explains Yes and No move on")
        check(all(page.locator(f"#{x}").is_hidden() for x in ("maxSec", "artRow", "marksRow", "markCtl")),
              "no field maximum, artifact flag or other-discharge marks")
        montage = lambda: page.evaluate("() => IEDReview.state.settings.montage")
        seen = []
        for key in ("t", "a", "b"):
            page.keyboard.press(key)
            page.wait_for_timeout(100)
            seen.append(montage())
        check(seen == ["transverse", "average", "banana"], f"T, A and B switch the montage ({seen})")
        before = page.evaluate("() => [IEDReview.state.pos, JSON.stringify(IEDReview.state.settings)]")
        for key in ("Enter", "m", "w", "g", "d", "u", "?"):
            page.keyboard.press(key)
        page.wait_for_timeout(200)
        after = page.evaluate("() => [IEDReview.state.pos, JSON.stringify(IEDReview.state.settings)]")
        check(after == before and page.locator("#helpModal").is_hidden(), "no other key does anything")
        # 1.7.5: arrow keys in a Yes/No study. Up/Down step the sensitivity one rung of the ladder (uV/mm),
        # Left/Right move one clip without answering. Planted: 7 -> Up -> 5 -> Down -> 7; pos p -> p+1 -> p.
        sens = lambda: page.evaluate("() => Number(IEDReview.state.settings.sens)")
        pos = lambda: page.evaluate("() => IEDReview.state.pos")
        answered = lambda: page.evaluate(
            "() => Object.values(IEDReview.state.labels).filter(L => L && L.decision).length")
        page.evaluate("() => { IEDReview.state.settings.sens = 7; }")
        s0, n0 = sens(), answered()
        page.keyboard.press("ArrowUp")
        page.wait_for_timeout(100)
        s1 = sens()
        page.keyboard.press("ArrowDown")
        page.wait_for_timeout(100)
        s2 = sens()
        check((s0, s1, s2) == (7, 5, 7), f"Up and Down step the sensitivity 7 -> 5 -> 7 uV/mm ({s0}, {s1}, {s2})")
        p0 = pos()
        page.keyboard.press("ArrowRight")
        page.wait_for_timeout(100)
        p1 = pos()
        page.keyboard.press("ArrowLeft")
        page.wait_for_timeout(100)
        p2 = pos()
        check((p1, p2) == (p0 + 1, p0), f"Right and Left move one clip forward and back ({p0} -> {p1} -> {p2})")
        check(answered() == n0, "the arrow keys record no answer")
        page.keyboard.press("y")
        page.wait_for_timeout(200)
        page.keyboard.press("n")
        page.wait_for_timeout(200)
        page.keyboard.press("n")
        page.wait_for_timeout(1800)
        page.wait_for_selector("#syncChip.chip-ok", timeout=6000)
        check(page.evaluate("() => IEDReview.state.pos") == 3, "Y and N each saved and moved on")
        check(page.inner_text("#syncChip").startswith("Saved to server"), "chip says saved to server")
        page.screenshot(path=os.path.join(shots, "hosted_labeling.png"))

        # 1.7.7: links opened one after another in ONE browser stay apart, pairwise: EC and JK (two raters) and the
        # owner never pick up each other's name, answers or display settings, and the owner cannot label as EC or JK.
        # The browser also holds what an older viewer left: "EC" as the last rater, and EC's answers filed as JK's.
        page.keyboard.press("t")
        page.wait_for_timeout(150)
        page.evaluate("""() => {
            localStorage.setItem('iedreview:v1:rater', JSON.stringify('EC'));
            const k = Object.keys(localStorage).find(k => k.endsWith(':labels:s01:rater:EC'));
            localStorage.setItem('iedreview:v1:labels:s01:rater:JK', localStorage.getItem(k));
        }""")
        to_site = lambda url: url.replace("https://ied-review-test.vercel.app", site_base)
        n_ans = lambda pg: pg.evaluate("() => Object.values(IEDReview.state.labels).filter(l => l && l.decision).length")
        mont = lambda pg: pg.evaluate("() => IEDReview.state.settings.montage")

        def same_browser(url):
            pg = page.context.new_page()
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.goto(url)
            pg.wait_for_selector("#pickModal:not([hidden]) .batches .btn", timeout=8000)
            return pg

        def open_s01(pg, initials=None):
            pg.locator("#pickTable .btn").first.click()
            if initials:
                pg.wait_for_selector("#raterModal:not([hidden])", timeout=8000)
                pg.fill("#raterInput", initials)
                pg.press("#raterInput", "Enter")
            pg.wait_for_function("() => IEDReview.state.batch && !IEDReview.state.isDemo", timeout=10000)
            pg.wait_for_timeout(1200)

        def answer_one(pg):
            pg.keyboard.press("y")
            pg.wait_for_timeout(1800)
            pg.wait_for_selector("#syncChip.chip-ok", timeout=6000)

        jk = same_browser(links["JK"])
        check("Your answers (JK)" in jk.inner_text("#pickTable"), "JK's link, after EC's in the same browser, is JK")
        open_s01(jk)
        check(jk.locator("#raterModal").is_hidden() and jk.inner_text("#raterName") == "JK", "and opens as JK")
        check(n_ans(jk) == 0 and mont(jk) == "banana",
              f"JK starts with none of EC's answers or settings ({n_ans(jk)} answers, {mont(jk)})")
        answer_one(jk)

        own = same_browser(to_site(owner_link))
        check("(EC)" not in own.inner_text("#pickTable") and "(JK)" not in own.inner_text("#pickTable"),
              "the owner's link lists neither rater's answers as its own")
        own.locator("#pickTable .btn").first.click()
        own.wait_for_selector("#raterModal:not([hidden])", timeout=8000)
        refused = []
        for who in ("ec", "JK"):
            own.fill("#raterInput", who)
            own.press("#raterInput", "Enter")
            own.wait_for_timeout(200)
            refused.append(own.locator("#raterModal").is_visible() and "own link" in own.inner_text("#raterError"))
        check(refused == [True, True], f"the owner's link asks for initials and refuses EC's and JK's ({refused})")
        own.fill("#raterInput", "SJ")
        own.press("#raterInput", "Enter")
        own.wait_for_function("() => IEDReview.state.batch && !IEDReview.state.isDemo", timeout=10000)
        own.wait_for_timeout(1200)
        check(own.inner_text("#raterName") == "SJ" and n_ans(own) == 0 and mont(own) == "banana",
              f"the owner labels as SJ with nothing from EC or JK ({n_ans(own)} answers, {mont(own)})")
        answer_one(own)

        back = {}
        for name, url in (("EC", links["EC"]), ("JK", links["JK"]), ("SJ", to_site(owner_link))):
            pg = same_browser(url)
            open_s01(pg)
            back[name] = (pg.locator("#raterModal").is_hidden(), pg.inner_text("#raterName"), n_ans(pg), mont(pg))
            pg.close()
        check(back == {"EC": (True, "EC", 3, "transverse"), "JK": (True, "JK", 1, "banana"), "SJ": (True, "SJ", 1, "banana")},
              f"reopened in the same browser, each link has only its own name, answers and settings ({back})")
        jk.close()
        own.close()

        page2 = open_link(links["EC"])
        check("3 of 10" in page2.inner_text("#pickTable"), "a fresh browser shows EC's progress from the site")
        page2.locator("#pickTable .btn").first.click()
        page2.wait_for_function("() => IEDReview.state.batch && !IEDReview.state.isDemo", timeout=10000)
        page2.wait_for_timeout(1200)
        n2 = page2.evaluate("() => Object.values(IEDReview.state.labels).filter(l => l.decision).length")
        check(n2 == 3, f"the fresh browser restored 3 answers from the site ({n2})")

        page3 = open_link(links["JK"])
        pick = page3.inner_text("#pickTable")
        check("Your answers (JK)" in pick and "3 of 10" not in pick, "JK sees only JK's progress")
        check(not any(b in pick for b in assignment["EC"]), "JK does not see EC's batches")

        bad = browser.new_context().new_page()
        bad.goto(site_base + "/?token=wrong")
        bad.wait_for_selector("#pickModal:not([hidden])", timeout=8000)
        bad.wait_for_timeout(500)
        check("Forbidden" in bad.inner_text("#pickError"), "a wrong link gets a plain refusal")
        check(not errors, f"no page errors {errors[:2]}")
        check(not csp, f"the site's security policy blocks nothing the viewer needs {csp[:2]}")
        browser.close()

    out = run(os.path.join(KIT, "hosted", "pull_labels.py"), "--labels", labels, "--site", base)
    check("pulled labels_s01_EC.csv: 3 answers" in out, "pull_labels copied EC's labels to the labels folder")
    check("pulled labels_s01_SJ.csv: 1 answers" in out and "pulled labels_s01_JK.csv: 1 answers" in out,
          "the owner's answer was saved under SJ and JK's under JK, apart from EC's")
    got = list(csv.DictReader(open(os.path.join(labels, "labels_s01_EC.csv"))))
    check(sorted(r["decision"] for r in got if r["decision"]) == ["no", "no", "yes"], "the pulled file has the answers")
    check(all(not r["max_electrodes"] and not r["other_discharges_s"] for r in got), "the pulled file has no maximum or marks")
    out = run(os.path.join(KIT, "merge_labels.py"), "--key", os.path.join(syn, "key.csv"), "--out",
              os.path.join(work, "merged.csv"), os.path.join(labels, "labels_s01_EC.csv"))
    check("rater EC: 3 of 10 answered" in out, "merge_labels reads the pulled labels")
finally:
    srv.terminate()
    srv.wait()

print("\nALL CHECKS PASSED" if not problems else f"\n{len(problems)} CHECK(S) FAILED")
sys.exit(1 if problems else 0)
