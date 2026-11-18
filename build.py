#!/usr/bin/env python3
"""Builds the viewer: inlines src/core.js into src/page.html.

Writes ied_review.html (standalone document for local, offline use; also copied to dist/) and
dist/ied_review_artifact.html (page body for the hosted demo, which adds its own document skeleton).
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
page = (ROOT / "src" / "page.html").read_text(encoding="utf-8")
core = (ROOT / "src" / "core.js").read_text(encoding="utf-8")
assert "</script" not in core.lower(), "core.js must not contain a closing script tag"
assert page.count("/*__CORE__*/") == 1 and page.count("<!--HEAD-END-->") == 1
body = page.replace("/*__CORE__*/", core)
head, rest = body.split("<!--HEAD-END-->")
dist = ROOT / "dist"
dist.mkdir(exist_ok=True)
(dist / "ied_review_artifact.html").write_text(head + rest, encoding="utf-8")
standalone = (
    "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
    "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
    + head.strip() + "\n</head>\n<body>\n" + rest.strip() + "\n</body>\n</html>\n"
)
(dist / "ied_review.html").write_text(standalone, encoding="utf-8")
(ROOT / "ied_review.html").write_text(standalone, encoding="utf-8")
print("built", {p.name: p.stat().st_size for p in dist.glob("*.html")})
