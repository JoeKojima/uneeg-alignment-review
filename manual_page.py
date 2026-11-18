"""The rater manuals as pages of the review site (kit 1.8.4).

A staged study has two: the 10-20 manual (scalp) for the 10-20 clips, and the UNEEG manual (uneeg) for the
training examples and the UNEEG clips. serve_review.py and make_site.py take `--manual scalp=FILE.md` and
`--manual uneeg=FILE.md` (a bare FILE.md is the 10-20 one), keep a copy of each in the labels folder so later
runs keep them until `--manual none`, and serve them as manual_scalp.html and manual_uneeg.html. The viewer's
Manual button opens the one for the part the rater is in, and it opens by itself when a rater starts a part they
have answered nothing in. Only a small Markdown subset is read: # to ### headings, paragraphs, - and 1. lists,
| tables |, **bold**, *italic* and `code`, and (kit 1.8.6) a line `[[figure: uneeg-lead]]`, which places the
kit's own drawing of the UNEEG lead. Every character of the text is escaped, so the pages carry no markup,
links or scripts beyond what this module writes. Standard library only.
"""
import html
import os
import re

MAX_BYTES = 200_000
_HEAD = re.compile(r"^(#{1,3})\s+(.*?)\s*#*\s*$")
_UL = re.compile(r"^\s*[-*]\s+(.*)$")
_OL = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_FIG = re.compile(r"^\s*\[\[\s*figure:\s*([a-z0-9-]+)\s*\]\]\s*$", re.I)

# The kit's own figures (kit 1.8.6). The viewer draws the same figure from its own copy (src/page.html, UNEEG_LEAD);
# tests/test_hosted.py checks the two copies match. Plain shapes and text only: no links, no paths, no scripts.
UNEEG_LEAD_SVG = (
    '<svg class="lead-svg" viewBox="0 0 560 196" role="img" aria-label="The UNEEG lead: three evenly spaced '
    'contacts, D, C and P; channels D-C and P-C each span one gap and D-P spans both">'
    '<text class="end" x="44" y="38">distal end</text>'
    '<text class="end" x="516" y="38" text-anchor="end">proximal end</text>'
    '<rect class="lead" x="40" y="52" width="480" height="16" rx="8"></rect>'
    '<rect class="contact" x="108" y="47" width="28" height="26" rx="4"></rect>'
    '<rect class="contact" x="266" y="47" width="28" height="26" rx="4"></rect>'
    '<rect class="contact" x="424" y="47" width="28" height="26" rx="4"></rect>'
    '<text class="ct" x="122" y="96" text-anchor="middle">D</text>'
    '<text class="ct" x="280" y="96" text-anchor="middle">C</text>'
    '<text class="ct" x="438" y="96" text-anchor="middle">P</text>'
    '<text class="sub" x="122" y="113" text-anchor="middle">distal (SQD)</text>'
    '<text class="sub" x="280" y="113" text-anchor="middle">centre (SQC)</text>'
    '<text class="sub" x="438" y="113" text-anchor="middle">proximal (SQP)</text>'
    '<line class="br" x1="122" y1="146" x2="280" y2="146"></line>'
    '<line class="br" x1="122" y1="140" x2="122" y2="152"></line>'
    '<line class="br" x1="280" y1="140" x2="280" y2="152"></line>'
    '<line class="br" x1="280" y1="146" x2="438" y2="146"></line>'
    '<line class="br" x1="438" y1="140" x2="438" y2="152"></line>'
    '<text class="pair" x="201" y="138" text-anchor="middle">D-C</text>'
    '<text class="pair" x="359" y="138" text-anchor="middle">P-C</text>'
    '<line class="br" x1="122" y1="182" x2="438" y2="182"></line>'
    '<line class="br" x1="122" y1="176" x2="122" y2="188"></line>'
    '<line class="br" x1="438" y1="176" x2="438" y2="188"></line>'
    '<text class="pair" x="280" y="174" text-anchor="middle">D-P, twice the distance</text>'
    '</svg>')
UNEEG_LEAD_CAPTION = ("The UNEEG lead under the scalp, with its three evenly spaced contacts. Each channel is one "
                      "contact against another: D-C and P-C between neighbours, and D-P across both gaps.")
FIGURES = {
    "uneeg-lead": ('<figure class="lead-fig" data-fig="uneeg-lead">' + UNEEG_LEAD_SVG
                   + "<figcaption>" + UNEEG_LEAD_CAPTION + "</figcaption></figure>"),
}


def _inline(s):
    s = html.escape(s, quote=True)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![*\w])\*(?![\s*])(.+?)(?<![\s*])\*(?![*\w])", r"<em>\1</em>", s)
    return s


def _cells(line):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def to_html(text):
    """(title, body HTML) of a Markdown text; the title is the first top-level heading ('Rater manual' if none)."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out, title, i = [], None, 0
    para, items, kind = [], [], None

    def flush():
        nonlocal para, items, kind
        if para:
            out.append("<p>" + _inline(" ".join(para)) + "</p>")
        if items:
            out.append(f"<{kind}>" + "".join(f"<li>{_inline(x)}</li>" for x in items) + f"</{kind}>")
        para, items, kind = [], [], None

    while i < len(lines):
        line = lines[i]
        if not line.strip():
            flush()
            i += 1
            continue
        f = _FIG.match(line)
        if f and f.group(1).lower() in FIGURES:
            flush()
            out.append(FIGURES[f.group(1).lower()])
            i += 1
            continue
        m = _HEAD.match(line)
        if m:
            flush()
            level = len(m.group(1))
            if level == 1 and title is None:
                title = m.group(2)
            out.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
            i += 1
            continue
        if line.lstrip().startswith("|") and i + 1 < len(lines) and _SEP.match(lines[i + 1]):
            flush()
            head = _cells(line)
            rows = []
            i += 2
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(_cells(lines[i]))
                i += 1
            t = ["<table><thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in head) + "</tr></thead><tbody>"]
            for r in rows:
                r = (r + [""] * len(head))[:len(head)]
                t.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>")
            out.append("".join(t) + "</tbody></table>")
            continue
        for rx, k in ((_UL, "ul"), (_OL, "ol")):
            m = rx.match(line)
            if m:
                if para or (kind and kind != k):
                    flush()
                kind = k
                items.append(m.group(1))
                break
        else:
            if items:  # a wrapped list item
                items[-1] += " " + line.strip()
            else:
                para.append(line.strip())
        i += 1
    flush()
    return title or "Rater manual", "\n".join(out)


STYLE = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#5d5b55;--line:#dedbd2;--panel:#f1efe8;--accent:#2f6f5e}
@media (prefers-color-scheme:dark){:root{--bg:#141513;--fg:#e9e7e1;--muted:#a9a69d;--line:#34352f;--panel:#1d1e1b;--accent:#7cc4ad}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:46rem;margin:0 auto;padding:32px 16px 64px}
h1{font-size:1.9rem;line-height:1.2;margin:0 0 1.2rem}
h2{font-size:1.3rem;margin:2.2rem 0 .6rem;padding-top:1rem;border-top:1px solid var(--line)}
h3{font-size:1.08rem;margin:1.6rem 0 .4rem}
p,li{color:var(--fg)}
ul,ol{padding-left:1.4rem}
li{margin:.25rem 0}
strong{font-weight:650}
code{font:0.92em ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--panel);padding:.05em .3em;border-radius:4px}
table{border-collapse:collapse;width:100%;margin:.6rem 0 1rem;font-size:.95rem}
th,td{text-align:left;padding:.45rem .6rem;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600}
.note{color:var(--muted);font-size:.9rem;margin-bottom:1.6rem}
figure.lead-fig{margin:1rem 0 1.2rem;padding:12px 14px;background:var(--panel);border-radius:8px}
.lead-svg{display:block;width:100%;max-width:560px;height:auto;margin:0 auto}
.lead-svg text{fill:var(--fg);font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.lead-svg .end,.lead-svg .sub{fill:var(--muted);font-size:12px}
.lead-svg .ct{font-weight:700;font-size:16px}
.lead-svg .pair{font-weight:600}
.lead-svg .lead{fill:var(--bg);stroke:var(--muted);stroke-width:1.5}
.lead-svg .contact{fill:var(--accent)}
.lead-svg .br{stroke:var(--muted);stroke-width:1.5}
figure.lead-fig figcaption{color:var(--muted);font-size:.9rem;margin-top:6px}
"""


def page(text):
    """The whole manual.html page for a Markdown text."""
    title, body = to_html(text)
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<meta name=\"robots\" content=\"noindex, nofollow\">"
            f"<title>{html.escape(title)}</title><style>{STYLE}</style></head>\n<body><main>\n{body}\n"
            "</main></body></html>\n").encode("utf-8")


PARTS = ("scalp", "uneeg")


def file_name(part):
    return f"manual_{part}.html"


def _read(path):
    if not os.path.isfile(path):
        raise SystemExit(f"--manual: no file at {path}")
    if not path.lower().endswith((".md", ".markdown", ".txt")):
        raise SystemExit("--manual takes each manual as a Markdown file (.md), not a PDF or Word file")
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) > MAX_BYTES:
        raise SystemExit(f"--manual: {path} is over {MAX_BYTES // 1000} kB; give the text only")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        raise SystemExit(f"--manual: {path} is not UTF-8 text")


def load(labels_dir, asked):
    """{part: Markdown text} of the study's manuals. asked is the list of --manual values: none (or empty) keeps
    the copies saved in the labels folder, 'none' removes them all, 'scalp=FILE' or 'uneeg=FILE' replaces that
    part's, and a bare FILE is the 10-20 (scalp) manual."""
    for v in asked or []:
        if str(v).strip().lower() == "none":
            for part in PARTS:
                p = os.path.join(labels_dir, f".manual_{part}.md")
                if os.path.exists(p):
                    os.remove(p)
            continue
        part, sep, path = str(v).partition("=")
        if not sep or part.strip().lower() not in PARTS:
            part, path = "scalp", str(v)
        text = _read(path.strip())
        os.makedirs(labels_dir, exist_ok=True)
        with open(os.path.join(labels_dir, f".manual_{part.strip().lower()}.md"), "w", encoding="utf-8") as f:
            f.write(text)
    out = {}
    for part in PARTS:
        p = os.path.join(labels_dir, f".manual_{part}.md")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out[part] = f.read()
    return out
