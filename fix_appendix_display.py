# Makes appendix code-cell figure displays crash-proof in environments without the PNGs
# (e.g. Colab). Adds a guarded show_fig() helper to the setup cell and replaces
# display(Image('fig.png')) with show_fig('fig.png'). Idempotent.
import json, re

NB = 'ocm_walkthrough.ipynb'
nb = json.load(open(NB))

HELPER = (
    "import os\n"
    "def show_fig(name):\n"
    "    # Figures are also embedded in the markdown cells above; show from disk if present,\n"
    "    # otherwise don't crash (e.g. Colab without the PNG files).\n"
    "    if os.path.exists(name):\n"
    "        display(Image(name))\n"
    "    else:\n"
    "        print('[%s - shown embedded in the markdown above]' % name)\n"
)

pat = re.compile(r"display\(Image\((\'[^\']+\'|\"[^\"]+\")\)\)")
added_helper = 0
replaced = 0

for c in nb['cells']:
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']

    if 'RUN_HEAVY =' in src:
        # setup cell: inject helper once, right before the readiness print
        if 'def show_fig' not in src:
            src = src.replace(
                "print('Appendix ready. RUN_HEAVY =', RUN_HEAVY)",
                HELPER + "\nprint('Appendix ready. RUN_HEAVY =', RUN_HEAVY)")
            added_helper += 1
        c['source'] = src
        continue

    # all other code cells: display(Image('x')) -> show_fig('x')
    new, n = pat.subn(r"show_fig(\1)", src)
    if n:
        c['source'] = new
        replaced += n

json.dump(nb, open(NB, 'w'), indent=1)
print(f"helper added to setup: {added_helper} | display(Image()) -> show_fig(): {replaced}")
