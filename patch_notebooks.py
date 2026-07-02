# Patches the two notebooks for the corrected, leak-free OOD story.
import nbformat

def replace_in_cell(nb, needle, old, new, expect=1):
    n = 0
    for c in nb.cells:
        src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
        if needle in src and old in src:
            c['source'] = src.replace(old, new)
            n += 1
    assert n == expect, f"expected {expect} replacements, got {n} for: {old[:50]!r}"
    return n

# ── ocm_analysis.ipynb — fix cell-41 leakage + add disclosure markdown ──────────
an = nbformat.read('ocm_analysis.ipynb', as_version=4)
replace_in_cell(
    an, "pre_full",
    "pre_full.fit(np.vstack([X_lit_sc, X_ours_sc]),\n             np.concatenate([y_lit, y_ours]))",
    "# LEAK-FREE: prior trained on impregnation literature only, which EXCLUDES the\n"
    "# non-impregnation OOD test rows. (The earlier version trained on all X_lit_sc,\n"
    "# including the OOD samples and their labels -> leakage -> optimistic 3.60.)\n"
    "pre_full.fit(np.vstack([X_lit_impreg, X_ours_sc]),\n"
    "             np.concatenate([y_lit_impreg, y_ours]))")

# insert a markdown disclosure right after cell 41 (the OOD cell)
ood_idx = None
for i, c in enumerate(an.cells):
    s = ''.join(c['source'])
    if c.cell_type == 'code' and 'ood_mask' in s and 'pre_full' in s:
        ood_idx = i; break
assert ood_idx is not None, "could not find OOD cell"
disclosure = nbformat.v4.new_markdown_cell(
    "### Correction — the OOD number was leakage-inflated\n\n"
    "The earlier version of this cell trained the Stage-1 prior on **all** literature, which "
    "**includes the 2,139 non-impregnation OOD test rows and their true labels**. The prior "
    "therefore handed the final model the answers through the feature — data leakage — which is "
    "why it scored a spectacular **3.60 (−45%)**. That number is not a real generalisation result.\n\n"
    "The cell above is now **leak-free**: the prior is trained on impregnation literature only, "
    "which excludes the OOD set. A clean quantile-normalisation × filtering ablation "
    "(`qn_tradeoff.py`, `fig_qn_tradeoff.png`) gives the honest picture:\n\n"
    "| Config (leak-free) | In-dist CV RMSE | OOD RMSE |\n"
    "|---|---|---|\n"
    "| Baseline | 2.133 | 6.53 |\n"
    "| DRST-filtered + raw | 1.915 | 6.11 |\n"
    "| **DRST-filtered + QN (= PFT)** | **1.910** | 6.77 |\n"
    "| Full impreg-lit + raw | 1.924 | 6.05 |\n"
    "| Full impreg-lit + QN | 1.913 | 6.32 |\n"
    "| Old 3.60 (leaky repro) | 1.932 | 3.62 |\n\n"
    "**Takeaways:** (1) leak-free OOD gains are modest (~6.0–6.8, near baseline); (2) quantile "
    "normalisation **improves in-distribution CV and slightly worsens OOD** — a deliberate dial "
    "trading extrapolation for local accuracy; (3) the strong, honest result is in-distribution "
    "(−10.6%, 10/10 seeds, p<10⁻¹⁴, held-out 2.097→1.892).")
an.cells.insert(ood_idx + 1, disclosure)
nbformat.write(an, 'ocm_analysis.ipynb')
print("ocm_analysis.ipynb patched (leak-free cell 41 + disclosure).")

# ── ocm_walkthrough.ipynb — fix Ch9 claim + update appendix Q16 ─────────────────
wt = nbformat.read('ocm_walkthrough.ipynb', as_version=4)
replace_in_cell(
    wt, "Prior Feature Transfer",
    "Best result; decouples the label offset from the loss; large OOD gain",
    "Best result (−10.6%, 10/10 seeds, p<10⁻¹⁴, held-out); decouples the label offset from the loss")

# appendix Q16: replace the earlier 7.60 estimate with the clean leak-free number + figure
replace_in_cell(
    wt, "Q16",
    "the faithful run gives baseline 6.53 → **PFT 7.60 — worse**, and does **not** reproduce the deck's \"6.53 → 3.60\". This is *by design*: quantile-normalisation calibrates the model to the **lab yield scale** (~5.25%), so it under-predicts the higher-yield literature. PFT optimises *our* regime — that is the goal — it is **not** a literature-extrapolation model. **That deck number should be corrected.**",
    "a **leak-free** run (prior never sees the OOD rows) gives baseline 6.53 → **PFT 6.77 — roughly level with baseline**, and does **not** reproduce the old \"6.53 → 3.60\". That 3.60 was **data leakage**: the prior had been trained on the OOD test rows and their labels (it reproduces as 3.62 only when the leak is put back). Quantile normalisation calibrates to the **lab yield scale** (~5.25%), which *improves in-distribution accuracy but slightly worsens OOD* — a deliberate trade-off (see figure). PFT optimises *our* regime, which is the goal; it is **not** a literature-extrapolation model.\n\n![QN trade-off: in-distribution vs leak-free OOD](fig_qn_tradeoff.png)")
nbformat.write(wt, 'ocm_walkthrough.ipynb')
print("ocm_walkthrough.ipynb patched (Ch9 claim + appendix Q16 + fig_qn_tradeoff).")
