"""Render ocm_worknote_taniike.md -> self-contained HTML (base64 figures) -> PDF (Chromium headless)."""
import base64, os, re, subprocess, markdown

MD = 'ocm_worknote_taniike.md'
HTML = 'ocm_worknote_taniike.html'
PDF = 'ocm_worknote_taniike.pdf'
CHROME = '/opt/pw-browsers/chromium-1194/chrome-linux/chrome'

text = open(MD).read()
body = markdown.markdown(text, extensions=['tables', 'fenced_code', 'sane_lists'])


# Promote image alt-text to a VISIBLE <figcaption>. The work note carries its figure captions as
# markdown image alt-text (`![**Figure 1 - ...**](fig.png)`), which python-markdown emits only into
# the alt="" attribute, where a reader of the PDF never sees it. Without this step all nine captions
# silently vanish from the rendered output.
def caption(m):
    alt, src = m.group(1), m.group(2)
    if not alt.strip():
        return m.group(0)
    inner = markdown.markdown(alt).removeprefix('<p>').removesuffix('</p>')
    return f'<figure><img alt="{alt}" src="{src}"><figcaption>{inner}</figcaption></figure>'


n_imgs_before = len(re.findall(r'<img\b', body))
body, n_caps = re.subn(r'<img alt="([^"]*)" src="([^"]+)"(?:\s+title="[^"]*")?\s*/?>', caption, body)
if n_caps != n_imgs_before:
    raise SystemExit(
        f"caption promotion matched {n_caps} of {n_imgs_before} <img> tag(s) in the rendered "
        f"markdown -- {n_imgs_before - n_caps} image(s) would silently keep an invisible alt="" "
        f"caption. Markdown likely emitted an <img> attribute/ordering this regex doesn't expect; "
        f"inspect it and extend the regex rather than widening this check.")
print(f"promoted {n_caps} figure captions to visible <figcaption>")

# inline <img src="fig.png"> as base64 data URIs
def embed(m):
    src = m.group(1)
    if src.startswith('data:') or not os.path.exists(src):
        return m.group(0)
    b64 = base64.b64encode(open(src, 'rb').read()).decode('ascii')
    return m.group(0).replace(src, f'data:image/png;base64,{b64}')
body = re.sub(r'<img[^>]*\bsrc="([^"]+)"', embed, body)

CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1a1a1a;
 max-width:820px;margin:0 auto;padding:32px 40px;line-height:1.55;font-size:14px}
h1{font-size:22px;border-bottom:3px solid #1f4e79;padding-bottom:8px;color:#12325a}
h2{font-size:17px;color:#12325a;border-bottom:1px solid #ccc;padding-bottom:4px;margin-top:28px}
h3{font-size:15px;color:#1f4e79;margin-top:20px}
img{max-width:88%;display:block;margin:14px auto;border:1px solid #e2e2e2;border-radius:4px}
figure{margin:18px 0}
figcaption{font-size:12.5px;color:#444;text-align:center;max-width:88%;margin:6px auto 0;line-height:1.45}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:13px}
th,td{border:1px solid #ccc;padding:6px 9px;text-align:left}
th{background:#1f4e79;color:#fff}
tr:nth-child(even){background:#f6f8fa}
code{background:#eef1f4;padding:1px 4px;border-radius:3px;font-size:12.5px}
blockquote{border-left:4px solid #e0b25f;background:#fff8ec;margin:12px 0;padding:8px 14px;color:#4a3c1a}
strong{color:#12325a}
@page{margin:14mm}
"""
html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
open(HTML, 'w').write(html)
print(f"wrote {HTML} ({os.path.getsize(HTML)//1024} KB)")

if os.path.exists(PDF):
    os.remove(PDF)
if not os.path.exists(CHROME):
    raise SystemExit(f"chromium not found at {CHROME} -- cannot export PDF")
r = subprocess.run([CHROME, '--headless', '--no-sandbox', '--disable-gpu',
                    f'--print-to-pdf={PDF}', '--no-pdf-header-footer',
                    os.path.abspath(HTML)], capture_output=True, text=True, timeout=120)
if not os.path.exists(PDF):
    raise SystemExit(f"PDF export FAILED (no file produced): {r.stderr[-800:]}")
print(f"wrote {PDF} ({os.path.getsize(PDF)//1024} KB)")

# ---- editable DOCX via pandoc (pypandoc) ----
import pypandoc
pypandoc.convert_file(MD, 'docx', outputfile='ocm_worknote_taniike.docx',
                      extra_args=['--resource-path=.', '--toc', '--toc-depth=2'])
print(f"wrote ocm_worknote_taniike.docx ({os.path.getsize('ocm_worknote_taniike.docx')//1024} KB)")
