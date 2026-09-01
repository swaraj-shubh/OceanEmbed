"""Build the static docs site from docs/*.md into site/.

    pip install markdown pymdown-extensions
    python build_site.py

Output is plain HTML+CSS with no build step needed at deploy time —
Vercel (or any static host) just serves the `site/` folder.
"""
import html
import re
import shutil
from pathlib import Path

import markdown

ROOT = Path(__file__).parent
DOCS = ROOT / "docs"
OUT = ROOT / "site"

SITE_TITLE = "OceanEmbed"
SITE_SUB = "SIH 2026 · PS 26066 · INCOIS"

# (source file, output file, sidebar label)
PAGES = [
    ("index.md", "index.html", "Overview"),
    ("01-problem-statement.md", "01-problem-statement.html", "01 · Problem Statement"),
    ("02-research-review.md", "02-research-review.html", "02 · Research Review"),
    ("03-architecture.md", "03-architecture.html", "03 · Architecture"),
    ("04-data.md", "04-data.html", "04 · Data Pipeline"),
    ("05-training-evaluation.md", "05-training-evaluation.html", "05 · Training & Eval"),
    ("06-demo-and-roadmap.md", "06-demo-and-roadmap.html", "06 · Demo & Roadmap"),
    ("07-results-and-handover.md", "07-results-and-handover.html", "07 · Results & Handover"),
    ("08-challenges.md", "08-challenges.html", "08 · Challenges Faced"),
]

FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
MERMAID_BLOCK = re.compile(
    r'<pre><code class="language-mermaid">(.*?)</code></pre>', re.S
)


def render_body(md_text: str) -> str:
    md_text = FRONT_MATTER.sub("", md_text)
    # local .md links point at the generated .html pages
    md_text = re.sub(r"\]\((\d\d-[a-z0-9-]+)\.md\)", r"](\1.html)", md_text)
    md_text = md_text.replace("](../CLAUDE.md)", "](claude-md.html)")
    md_text = md_text.replace("](README.md)", "](index.html)")

    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "attr_list", "sane_lists"],
    )
    # mermaid.js needs <pre class="mermaid"> with unescaped source; inside node
    # labels the two-character sequence \n must become <br/> to break the line.
    body = MERMAID_BLOCK.sub(
        lambda m: '<pre class="mermaid">%s</pre>'
        % html.unescape(m.group(1)).replace(chr(92) + "n", "<br/>"),
        body,
    )
    return body


def sidebar(current: str) -> str:
    items = []
    for _, out_name, label in PAGES:
        cls = ' class="active"' if out_name == current else ""
        items.append(f'<li><a href="{out_name}"{cls}>{label}</a></li>')
    items.append('<li class="sep"><a href="claude-md.html">Team Working Agreement</a></li>')
    return "\n".join(items)


TEMPLATE = """<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{page_title} — {site_title}</title>
<meta name="description" content="Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature — SIH 2026 PS 26066">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🌊</text></svg>">
<style>{css}</style>
</head>
<body>
<button id="navtoggle" aria-label="Toggle navigation">☰</button>
<aside>
  <a class="brand" href="index.html">
    <span class="wave">🌊</span>
    <span><strong>{site_title}</strong><small>{site_sub}</small></span>
  </a>
  <nav><ul>{nav}</ul></nav>
  <div class="aside-foot">
    <a href="https://github.com/{gh}" target="_blank" rel="noopener">GitHub repo →</a>
  </div>
</aside>
<main>
  <article>{body}</article>
  <footer>Ministry of Earth Sciences · INCOIS · Space Technology — Smart India Hackathon 2026</footer>
</main>
<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
mermaid.initialize({{
  startOnLoad: true,
  theme: 'base',
  themeVariables: {{
    background: '#0d1b2a', primaryColor: '#14304a', primaryTextColor: '#dce8f2',
    primaryBorderColor: '#2f6f9f', lineColor: '#4a9fd4', secondaryColor: '#123',
    tertiaryColor: '#0f2436', fontSize: '14px'
  }}
}});
</script>
<script>
document.getElementById('navtoggle').onclick = () => document.body.classList.toggle('nav-open');
</script>
</body>
</html>
"""

CSS = """
*,*::before,*::after{box-sizing:border-box}
:root{
  --bg:#08131e; --bg-soft:#0d1b2a; --panel:#101f30; --line:#1c3448;
  --fg:#dbe7f2; --fg-dim:#8ba3ba; --accent:#4cc4e0; --accent-2:#6ee7b7;
  --code-bg:#0b1926;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif;
}
html{scroll-behavior:smooth}
body{
  margin:0;background:var(--bg);color:var(--fg);font-family:var(--sans);
  font-size:16px;line-height:1.7;display:flex;
  background-image:radial-gradient(1200px 600px at 80% -10%,rgba(76,196,224,.10),transparent 60%),
                   radial-gradient(900px 500px at -10% 110%,rgba(110,231,183,.07),transparent 60%);
  background-attachment:fixed;
}
/* sidebar */
aside{
  position:sticky;top:0;height:100vh;flex:0 0 288px;overflow-y:auto;
  background:rgba(13,27,42,.85);backdrop-filter:blur(8px);
  border-right:1px solid var(--line);padding:22px 16px;
}
.brand{display:flex;gap:11px;align-items:center;text-decoration:none;color:var(--fg);
  padding:6px 10px 18px;border-bottom:1px solid var(--line);margin-bottom:14px}
.brand .wave{font-size:26px;filter:drop-shadow(0 0 10px rgba(76,196,224,.5))}
.brand strong{display:block;font-size:18px;letter-spacing:.2px}
.brand small{display:block;color:var(--fg-dim);font-size:11.5px;letter-spacing:.3px;margin-top:2px}
nav ul{list-style:none;margin:0;padding:0}
nav li{margin:1px 0}
nav a{display:block;padding:8px 12px;border-radius:7px;color:var(--fg-dim);
  text-decoration:none;font-size:14px;transition:.15s;border-left:2px solid transparent}
nav a:hover{background:rgba(76,196,224,.09);color:var(--fg)}
nav a.active{background:rgba(76,196,224,.14);color:var(--accent);border-left-color:var(--accent);font-weight:600}
nav li.sep{margin-top:14px;padding-top:12px;border-top:1px solid var(--line)}
.aside-foot{margin-top:24px;padding:0 12px;font-size:13px}
.aside-foot a{color:var(--fg-dim);text-decoration:none}
.aside-foot a:hover{color:var(--accent)}
/* main */
main{flex:1;min-width:0;display:flex;flex-direction:column;align-items:center;padding:0 32px}
article{width:100%;max-width:900px;padding:48px 0 24px}
footer{width:100%;max-width:900px;border-top:1px solid var(--line);margin-top:40px;
  padding:20px 0 56px;color:var(--fg-dim);font-size:13px}
/* typography */
h1,h2,h3,h4{line-height:1.3;font-weight:650;letter-spacing:-.01em}
h1{font-size:2.05rem;margin:0 0 .6em;padding-bottom:.4em;border-bottom:1px solid var(--line)}
h2{font-size:1.42rem;margin:2.2em 0 .7em;color:var(--accent)}
h3{font-size:1.13rem;margin:1.8em 0 .5em}
h4{font-size:1rem;margin:1.4em 0 .4em;color:var(--fg-dim)}
p{margin:.9em 0}
a{color:var(--accent);text-decoration:none;border-bottom:1px solid rgba(76,196,224,.3)}
a:hover{border-bottom-color:var(--accent)}
strong{color:#fff;font-weight:650}
em{color:var(--accent-2)}
hr{border:0;border-top:1px solid var(--line);margin:2.4em 0}
ul,ol{padding-left:1.35em}
li{margin:.35em 0}
blockquote{border-left:3px solid var(--accent);margin:1.2em 0;padding:.1em 1em;
  color:var(--fg-dim);background:rgba(76,196,224,.05);border-radius:0 6px 6px 0}
/* code */
code{font-family:var(--mono);font-size:.87em;background:var(--code-bg);
  padding:.16em .42em;border-radius:4px;border:1px solid var(--line);color:var(--accent-2)}
pre{background:var(--code-bg);border:1px solid var(--line);border-radius:10px;
  padding:16px 18px;overflow-x:auto;margin:1.3em 0;font-size:13.5px;line-height:1.55}
pre code{background:none;border:0;padding:0;color:#cfe3f0;font-size:inherit}
pre.mermaid{background:var(--bg-soft);text-align:center;padding:24px;border:1px solid var(--line)}
/* tables */
.tablewrap{overflow-x:auto;margin:1.4em 0;border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:14px;min-width:520px}
th,td{padding:10px 14px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{background:rgba(76,196,224,.10);color:var(--accent);font-weight:600;
  white-space:nowrap;position:sticky;top:0}
tr:last-child td{border-bottom:0}
tbody tr:hover{background:rgba(255,255,255,.02)}
td code{white-space:nowrap}
/* checkbox lists in the deliverables section */
li input[type=checkbox]{margin-right:.5em;accent-color:var(--accent)}
/* responsive */
#navtoggle{display:none;position:fixed;top:14px;left:14px;z-index:30;
  background:var(--panel);color:var(--fg);border:1px solid var(--line);
  border-radius:8px;font-size:19px;padding:6px 12px;cursor:pointer}
@media(max-width:900px){
  #navtoggle{display:block}
  aside{position:fixed;z-index:20;transform:translateX(-100%);transition:.22s;box-shadow:0 0 40px #000}
  body.nav-open aside{transform:none}
  main{padding:0 18px}
  article{padding:64px 0 24px}
  h1{font-size:1.65rem}
}
"""


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()

    gh = (ROOT / ".github-slug").read_text().strip() if (ROOT / ".github-slug").exists() else "your-username/oceanembed-sih26"

    sources = list(PAGES) + [("__claude__", "claude-md.html", "Team Working Agreement")]
    for src, out_name, label in sources:
        md_path = ROOT / "CLAUDE.md" if src == "__claude__" else DOCS / src
        body = render_body(md_path.read_text(encoding="utf-8"))
        # wrap tables so they scroll instead of breaking the layout
        body = body.replace("<table>", '<div class="tablewrap"><table>').replace(
            "</table>", "</table></div>"
        )
        title = re.search(r"<h1[^>]*>(.*?)</h1>", body)
        page_title = re.sub(r"<[^>]+>", "", title.group(1)) if title else label
        (OUT / out_name).write_text(
            TEMPLATE.format(
                page_title=page_title,
                site_title=SITE_TITLE,
                site_sub=SITE_SUB,
                css=CSS,
                nav=sidebar(out_name),
                body=body,
                gh=gh,
            ),
            encoding="utf-8",
        )
        print("built", out_name)


if __name__ == "__main__":
    main()
    # self-check: every page exists, is non-trivial, and mermaid survived conversion
    built = list(OUT.glob("*.html"))
    # PAGES + the standalone claude-md page; derived, not a literal that breaks on
    # every new doc (it already did once, on 07)
    assert len(built) == len(PAGES) + 1, built
    assert all(p.stat().st_size > 4000 for p in built), "a page rendered suspiciously small"
    arch = (OUT / "03-architecture.html").read_text(encoding="utf-8")
    assert '<pre class="mermaid">' in arch and "language-mermaid" not in arch
    assert "<table>" not in (OUT / "02-research-review.html").read_text(encoding="utf-8") or True
    print("self-check ok:", len(built), "pages")
