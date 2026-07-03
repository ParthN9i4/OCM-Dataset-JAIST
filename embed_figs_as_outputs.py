# Replace markdown-attachment figures (which Colab/GitHub don't render) with stored
# code-cell image outputs (which render everywhere). Self-contained: figures are held as
# inline base64 in a FIG dict, shown via show_fig(key), and baked as display_data outputs
# so they appear on open without running. Idempotent-ish (safe to re-run).
import json, base64, re

NB = 'ocm_walkthrough.ipynb'
FIGS = {
    'drst_sweep':     'fig_drst_threshold_sweep.png',
    'pft_tau1':       'fig_pft_tau1_sweep.png',
    'kmm_overlap':    'fig_kmm_drst_overlap.png',
    'repeated_runs':  'fig_repeated_runs.png',
    'qn_tradeoff':    'fig_qn_tradeoff.png',
    'shap_stability': 'fig_shap_stability.png',
}
B64 = {k: base64.b64encode(open(v, 'rb').read()).decode('ascii') for k, v in FIGS.items()}

def img_output(key):
    return {"output_type": "display_data",
            "data": {"image/png": B64[key]},
            "metadata": {}}

nb = json.load(open(NB))
cells = nb['cells']

def src_of(c):
    return ''.join(c['source']) if isinstance(c['source'], list) else c['source']

# 1) Build the FIG-data + helper cell (base64 kept out of the readable cells).
fig_dict_lines = "FIG = {\n" + "".join(
    f"    {k!r}: {B64[k]!r},\n" for k in FIGS) + "}\n"
figdata_src = (
    "# ── Embedded appendix figures (base64) — renders in Colab / GitHub / Jupyter, no external files.\n"
    "# show_fig(key) decodes and displays; outputs are also stored so they show without running.\n"
    "import base64\n"
    "from IPython.display import Image, display\n\n"
    + fig_dict_lines +
    "\ndef show_fig(*keys):\n"
    "    for k in keys:\n"
    "        display(Image(data=base64.b64decode(FIG[k])))\n"
    "\nprint('Embedded figures ready:', ', '.join(FIG))\n"
)

# Locate the setup cell (has RUN_HEAVY) to (a) drop its old os-based show_fig and (b) insert FIG cell after it.
setup_idx = next(i for i, c in enumerate(cells)
                 if c['cell_type'] == 'code' and 'RUN_HEAVY =' in src_of(c))

# remove any previously-injected show_fig/os helper from the setup cell
s = src_of(cells[setup_idx])
s = re.sub(r"\nimport os\ndef show_fig\(name\):.*?above'\] % name\)\n", "\n", s, flags=re.S)
s = re.sub(r"\ndef show_fig\([^\n]*\):.*?(?=\nprint\('Appendix ready)", "\n", s, flags=re.S)
cells[setup_idx]['source'] = s

# insert / refresh the FIG-data cell right after setup
if setup_idx + 1 < len(cells) and 'Embedded appendix figures (base64)' in src_of(cells[setup_idx + 1]):
    cells[setup_idx + 1]['source'] = figdata_src
    cells[setup_idx + 1]['outputs'] = []
    cells[setup_idx + 1]['execution_count'] = None
else:
    cells.insert(setup_idx + 1, {
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": figdata_src, "id": "figdata0"})

# 2) Strip markdown attachment images, keep captions; drop attachments.
att_pat = re.compile(r'\n?!\[[^\]]*\]\(attachment:[^)]+\)\n?')
for c in cells:
    if c['cell_type'] == 'markdown':
        s = src_of(c)
        if 'attachment:' in s:
            c['source'] = att_pat.sub('\n', s).rstrip() + '\n'
        if 'attachments' in c:
            del c['attachments']

# 3) Point display cells at embedded data + bake outputs.
def set_show(cell, keys):
    s = src_of(cell)
    call = "; ".join(f"show_fig({k!r})" for k in keys) if False else \
           "show_fig(" + ", ".join(repr(k) for k in keys) + ")"
    # replace any existing show_fig(...)/display(Image(...)) line(s) in the else/tail with the new call
    s = re.sub(r"(?m)^\s*(show_fig\([^\n]*\)|display\(Image\([^\n]*\))\s*$", "    " + call, s)
    cell['source'] = s
    cell['outputs'] = [img_output(k) for k in keys]
    cell['execution_count'] = None

by_marker = {
    'Q5/Q6': ['drst_sweep', 'pft_tau1'],
    'Q8 —': ['kmm_overlap'],
    'Q11/Q14/Q15': ['repeated_runs'],
    'Q13 —': ['shap_stability'],
}
for c in cells:
    if c['cell_type'] != 'code':
        continue
    s = src_of(c)
    for marker, keys in by_marker.items():
        if marker in s:
            set_show(c, keys)

# 4) qn_tradeoff has no code cell — insert one right after its markdown (Q16).
if not any(c['cell_type'] == 'code' and "show_fig('qn_tradeoff')" in src_of(c) for c in cells):
    q16_idx = next(i for i, c in enumerate(cells)
                   if c['cell_type'] == 'markdown' and 'Q16.' in src_of(c))
    cells.insert(q16_idx + 1, {
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [img_output('qn_tradeoff')],
        "source": "# Q16 — quantile-normalisation trade-off (in-distribution vs leak-free OOD)\nshow_fig('qn_tradeoff')\n",
        "id": "qnfig0"})

json.dump(nb, open(NB, 'w'), indent=1)
print("done. total cells:", len(cells))
