"""Plain-text title, abstract and keywords for the OpenReview registration form, from main.tex, sections/abstract.tex
and the number macros in tables/numbers.tex (so the form states the same numbers as the PDF).

Usage: abstract_txt.py [--out paper/openreview.txt]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]


def macros() -> dict[str, str]:
    text = (PAPER / "tables" / "numbers.tex").read_text()
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", text))


def plain(tex: str, values: dict[str, str]) -> str:
    tex = re.sub(r"(?<!\\)%.*", "", tex)  # comments
    tex = re.sub(r"\\(\w+)\{\}", lambda m: values.get(m.group(1), m.group(0)), tex)
    tex = re.sub(r"\\(\w+)", lambda m: values.get(m.group(1), m.group(0)), tex)
    for a, b in ((r"\%", "%"), ("---", "\u2014"), ("--", "\u2013"), (r"\,", "\u2009"), ("~", " "), (r"\@", ""),
                 (r"\xspace", "")):
        tex = tex.replace(a, b)
    tex = re.sub(r"\\(?:mbox|emph|textbf|textit)\{([^{}]*)\}", r"\1", tex)
    tex = re.sub(r"\$([^$]*)\$", r"\1", tex)
    tex = tex.replace("{", "").replace("}", "")
    if "\\" in tex:
        raise ValueError(f"LaTeX left in the plain text: {tex[tex.index(chr(92)) - 20:tex.index(chr(92)) + 20]!r}")
    return re.sub(r"\s+", " ", tex).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=PAPER / "openreview.txt")
    args = ap.parse_args()
    values = macros()
    main_tex = (PAPER / "main.tex").read_text()
    title = re.search(r"\\title\[[^\]]*\]\{(.*?)\}\n\n", main_tex, re.DOTALL).group(1)
    keywords = re.search(r"\\keywords\{(.*?)\}\n", main_tex, re.DOTALL).group(1)
    abstract = plain((PAPER / "sections" / "abstract.tex").read_text(), values)
    words = len(abstract.split())
    text = (f"Title:\n{plain(title, values)}\n\nAbstract ({words} words):\n{abstract}\n\n"
            f"Keywords:\n{plain(keywords, values)}\n")
    args.out.write_text(text)
    print(text)
    if not 100 <= words <= 300:
        raise SystemExit(f"abstract has {words} words; AAMAS asks for 100-300")


if __name__ == "__main__":
    main()
