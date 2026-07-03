# Embeds appendix figures into ocm_walkthrough.ipynb as base64 cell attachments,
# so markdown ![](fig.png) references render on GitHub/nbviewer/Jupyter/VS Code
# without the external PNG files. Idempotent: skips cells already using attachment:.
import json, base64, re, os

NB = 'ocm_walkthrough.ipynb'
nb = json.load(open(NB))
assert nb['nbformat_minor'] >= 5, "attachments need nbformat 4.5+"

ref = re.compile(r'!\[([^\]]*)\]\((fig_[A-Za-z0-9_]+\.png)\)')
changed, embedded = 0, 0
for c in nb['cells']:
    if c['cell_type'] != 'markdown':
        continue
    src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
    figs = ref.findall(src)
    if not figs:
        continue
    if 'attachment:' in src:            # idempotency guard
        continue
    attachments = c.get('attachments', {})
    for _alt, fname in figs:
        if not os.path.exists(fname):
            raise FileNotFoundError(fname)
        b64 = base64.b64encode(open(fname, 'rb').read()).decode('ascii')
        attachments[fname] = {'image/png': b64}
        embedded += 1
    # rewrite ](fig.png) -> ](attachment:fig.png)
    new_src = ref.sub(lambda m: f'![{m.group(1)}](attachment:{m.group(2)})', src)
    c['source'] = new_src
    c['attachments'] = attachments
    changed += 1

json.dump(nb, open(NB, 'w'), indent=1)
print(f"cells updated: {changed}  |  images embedded: {embedded}")
