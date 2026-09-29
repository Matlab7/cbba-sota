"""Bundle docs/slides/index.html into one self-contained file, team-briefing.html.

Inlines the figures (data URIs), reveal.js and KaTeX (with its fonts), so the file works offline and can be sent
as a single attachment. The Pretendard web font stays a CDN link with system-font fallbacks.
Run: python docs/slides/build_single.py
"""
import base64
import mimetypes
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REVEAL = "https://cdn.jsdelivr.net/npm/reveal.js@5.1.0"
KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def data_uri(data, mime):
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def katex_css():
    css = fetch(f"{KATEX}/katex.min.css").decode()

    def font(m):
        name = m.group(1)
        if not name.endswith(".woff2"):
            return "url()"  # woff2 is enough for every current browser
        return f"url({data_uri(fetch(f'{KATEX}/{name}'), 'font/woff2')})"

    css = re.sub(r"url\((fonts/[^)]+)\)", font, css)
    # drop the emptied woff/ttf alternatives
    return re.sub(r',url\(\) format\("(woff|truetype)"\)', "", css)


def main():
    html = (HERE / "index.html").read_text()

    # figures
    def img(m):
        path = HERE / m.group(1)
        mime = mimetypes.guess_type(path.name)[0] or ("image/svg+xml" if path.suffix == ".svg" else "image/png")
        return f'src="{data_uri(path.read_bytes(), mime)}"'

    html = re.sub(r'src="(fig/[^"]+)"', img, html)

    # reveal.js stylesheets
    for name in ("reset.css", "reveal.css", "theme/white.css"):
        link = f'<link rel="stylesheet" href="{REVEAL}/dist/{name}">'
        assert link in html, link
        css = fetch(f"{REVEAL}/dist/{name}").decode()
        css = re.sub(r"@import url\([^)]*\);", "", css)  # theme imports web fonts we do not use
        html = html.replace(link, f"<style>{css}</style>")

    # math: bundled KaTeX + auto-render instead of reveal's math plugin (which loads KaTeX from a CDN at runtime)
    html = html.replace("</head>", f"<style>{katex_css()}</style>\n</head>")
    scripts = {
        f'<script src="{REVEAL}/dist/reveal.js"></script>': fetch(f"{REVEAL}/dist/reveal.js"),
        f'<script src="{REVEAL}/plugin/math/math.js"></script>':
            fetch(f"{KATEX}/katex.min.js") + b"\n" + fetch(f"{KATEX}/contrib/auto-render.min.js"),
        f'<script src="{REVEAL}/plugin/notes/notes.js"></script>': fetch(f"{REVEAL}/plugin/notes/notes.js"),
    }
    for tag, js in scripts.items():
        assert tag in html, tag
        html = html.replace(tag, "<script>" + js.decode().replace("</script", "<\\/script") + "</script>")

    init_old = re.search(r"  katex:\{.*?\},\n  plugins:\[RevealMath\.KaTeX, RevealNotes\]\n\}\);", html, re.S)
    assert init_old
    html = html.replace(init_old.group(0), """  plugins:[RevealNotes]
});
renderMathInElement(document.querySelector('.reveal .slides'), {
  delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],
  ignoredTags:['script','noscript','style','textarea','pre','code'], throwOnError:false
});
Reveal.on('ready', () => Reveal.layout());""")

    # Korean system-font fallbacks if the Pretendard CDN is unreachable
    html = html.replace("'Pretendard',system-ui,sans-serif",
                        "'Pretendard','Apple SD Gothic Neo','Malgun Gothic','Noto Sans KR',system-ui,sans-serif")

    out = HERE / "team-briefing.html"
    out.write_text(html)
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
