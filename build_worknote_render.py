"""Render a work note (default ocm_worknote_taniike.md) -> self-contained HTML (base64 figures) -> PDF (Chromium headless)."""
import base64, os, re, subprocess, sys, markdown

# usage: python3 build_worknote_render.py [note.md]   (default: the v2 note)
# add --compact for the tighter layout used to keep update notes within 3 pages
ARGS = [a for a in sys.argv[1:] if not a.startswith('--')]
COMPACT = '--compact' in sys.argv
MD = ARGS[0] if ARGS else 'ocm_worknote_taniike.md'
STEM = MD[:-3]
HTML = STEM + '.html'
PDF = STEM + '.pdf'
DOCX = STEM + '.docx'
CHROME = '/opt/pw-browsers/chromium-1194/chrome-linux/chrome'

text = open(MD).read()
body = markdown.markdown(text, extensions=['tables', 'fenced_code', 'sane_lists'])

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
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:13px}
th,td{border:1px solid #ccc;padding:6px 9px;text-align:left}
th{background:#1f4e79;color:#fff}
tr:nth-child(even){background:#f6f8fa}
code{background:#eef1f4;padding:1px 4px;border-radius:3px;font-size:12.5px}
blockquote{border-left:4px solid #e0b25f;background:#fff8ec;margin:12px 0;padding:8px 14px;color:#4a3c1a}
strong{color:#12325a}
@page{margin:14mm}
"""
if COMPACT:
    CSS += ("body{font-size:12.5px;line-height:1.4;padding:10px 22px;max-width:none}"
            "h1{font-size:19px;margin:4px 0 6px}h2{font-size:15px;margin-top:14px;margin-bottom:6px}"
            "p,ul{margin:5px 0}li{margin:1px 0}table{margin:8px 0;font-size:12px}th,td{padding:3px 7px}"
            "hr{margin:8px 0}@page{margin:11mm}")
html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
open(HTML, 'w').write(html)
print(f"wrote {HTML} ({os.path.getsize(HTML)//1024} KB)")

if os.path.exists(CHROME):
    r = subprocess.run([CHROME, '--headless', '--no-sandbox', '--disable-gpu',
                        f'--print-to-pdf={PDF}', '--no-pdf-header-footer',
                        os.path.abspath(HTML)], capture_output=True, text=True, timeout=120)
    if os.path.exists(PDF):
        print(f"wrote {PDF} ({os.path.getsize(PDF)//1024} KB)")
    else:
        print("PDF FAILED:", r.stderr[-500:])
else:
    print("chromium not found at", CHROME)

# ---- editable DOCX via pandoc (pypandoc) ----
try:
    import pypandoc
    pypandoc.convert_file(MD, 'docx', outputfile=DOCX,
                          extra_args=['--resource-path=.', '--toc', '--toc-depth=2'])
    print(f"wrote {DOCX} ({os.path.getsize(DOCX)//1024} KB)")
except Exception as e:
    print("docx skipped:", e)
