"""Render ocm_progress_report.md -> standalone HTML -> PDF (Chromium headless) + DOCX (pandoc).

Why this file exists. Until now NOTHING in the repository built ocm_progress_report.{html,pdf,docx}.
The four files were committed side by side with no reproducible path from the Markdown to the
rendered formats, which is exactly how a rendered artifact drifts from its source without anyone
noticing (see SESSION_CONTEXT.md section 8: one script is the single source of truth). The committed
HTML carries `<meta name="generator" content="pandoc">`, so this reproduces that toolchain:

    pandoc --standalone  ->  HTML      (pandoc's default template + CSS, as before)
    Chromium --print-to-pdf  ->  PDF   (no LaTeX in-container; see SESSION_CONTEXT.md section 6)
    pandoc  ->  DOCX

The report currently contains no figures, so there is no alt-text-to-caption step. The hook is kept
below anyway: build_worknote_render.py silently dropped all nine of its captions into invisible alt
attributes precisely because nobody had thought about it, and this report would fail the same way the
day someone adds an image.
"""
import os
import re
import subprocess

import pypandoc

MD = 'ocm_progress_report.md'
HTML = 'ocm_progress_report.html'
PDF = 'ocm_progress_report.pdf'
DOCX = 'ocm_progress_report.docx'
CHROME = '/opt/pw-browsers/chromium-1194/chrome-linux/chrome'

# ---- HTML (pandoc standalone, matching the committed artifact's generator) -------------------
pypandoc.convert_file(MD, 'html', outputfile=HTML,
                      extra_args=['--standalone', '--embed-resources', '--resource-path=.',
                                  '--metadata', f'title={os.path.splitext(MD)[0]}'])
html = open(HTML, encoding='utf-8').read()

n_imgs = len(re.findall(r'<img\b', html))
if n_imgs:
    raise SystemExit(
        f"{MD} now contains {n_imgs} image(s). Markdown image alt-text does NOT render as a visible "
        f"caption -- it lands in alt=\"\" where no reader of the PDF will ever see it. Add an "
        f"alt-text-to-<figcaption> step (see build_worknote_render.py) before rendering, or the "
        f"captions will be silently lost.")
print(f"wrote {HTML} ({os.path.getsize(HTML) // 1024} KB, {n_imgs} figures)")

# ---- PDF (headless Chromium; no LaTeX toolchain in-container) --------------------------------
# Delete first and require a NEW file to exist afterward: printing success whenever the output path
# merely EXISTS is the exact silent-failure bug build_pptx.py's PDF export had (a stale PDF from a
# previous run would otherwise pass for this run's output with no warning). Fail loudly instead.
if os.path.exists(PDF):
    os.remove(PDF)
if not os.path.exists(CHROME):
    raise SystemExit(f"chromium not found at {CHROME} -- cannot export PDF")
r = subprocess.run([CHROME, '--headless', '--no-sandbox', '--disable-gpu',
                    f'--print-to-pdf={PDF}', '--no-pdf-header-footer',
                    os.path.abspath(HTML)], capture_output=True, text=True, timeout=180)
if not os.path.exists(PDF):
    raise SystemExit(f"PDF export FAILED (no file produced): {r.stderr[-800:]}")
print(f"wrote {PDF} ({os.path.getsize(PDF) // 1024} KB)")

# ---- DOCX ------------------------------------------------------------------------------------
pypandoc.convert_file(MD, 'docx', outputfile=DOCX, extra_args=['--resource-path=.'])
print(f"wrote {DOCX} ({os.path.getsize(DOCX) // 1024} KB)")
