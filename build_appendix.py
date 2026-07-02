# Builds the "Chapter 10: Reviewer Q&A" appendix and appends it to ocm_walkthrough.ipynb.
# Cells reuse variables already defined in the notebook. No backslashes / no triple-double-quotes
# inside cell sources (so triple-single-quote delimiters below stay safe).
import json, nbformat

NB = 'ocm_walkthrough.ipynb'
nb = nbformat.read(NB, as_version=4)

# Idempotency guard: do not append twice.
if any('Chapter 10 — Reviewer Q&A' in ''.join(c.get('source', '')) for c in nb.cells):
    print('Appendix already present — nothing to do.')
    raise SystemExit(0)

cells = []  # (type, source_text)

cells.append(('md', r'''# Chapter 10 — Reviewer Q&A: How Each Question Was Addressed

The 17 review questions collapse into **6 themes**. Answer the theme and you answer the cluster.

| Theme (what is really being probed) | Questions | One-line answer |
|---|---|---|
| **1. Did you prepare the data right?** | Q1–Q4 | Standard choices; PCA is only a diagnostic; all 3,852 literature rows used. |
| **2. Did you pick methods/thresholds honestly?** | Q5, Q6, Q8 | Swept every threshold 0→1; the winner is robust, not tuned. KMM & DRST agree (r=0.79). |
| **3. Is the result real or luck?** | Q7, Q11, Q14–16 | 10 seeds, PFT wins 10/10, p=3.9e-15, holds on an untouched 20% test set. |
| **4. Did it learn real chemistry?** | Q12, Q13 | Exact TreeSHAP, stable over 10 runs; known OCM promoters + prior. LIME adds nothing. |
| **5. Where does it sit in the literature?** | Q9, Q10 | Pieces exist; the combination + OCM application is new. It is a drift-correction pipeline. |
| **6. Data honesty & next steps** | Q17 (+OOD) | No synthetic data yet (options below); I corrected an over-optimistic OOD number. |

Full detail lives in `ocm_feedback_responses.md`; every figure below is produced by `feedback_experiments.py`.

> **Tip for the defense:** finding and *correcting* the over-optimistic OOD number (Q16) is a **strength**, not a weakness — it shows the pipeline is reported honestly. Lead with the 6 themes, not 17 separate points.'''))

cells.append(('code', r'''# ── Appendix setup — reuses variables already defined earlier in this notebook ──────────────
# Heavy experiments are GATED so the notebook reads instantly. Set RUN_HEAVY = True to recompute.
RUN_HEAVY = False

import numpy as np
from IPython.display import Image, display
from scipy.stats import ttest_rel, wilcoxon
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb, lightgbm as lgb

def baseline_cv(seed=42, cv=5):
    # Single LightGBM on lab only; validation fold is lab (matches the paper protocol).
    rmses = []
    for tr, va in KFold(cv, shuffle=True, random_state=seed).split(X_ours_sc):
        m = lgb.LGBMRegressor(**lgb_params()).fit(X_ours_sc[tr], y_ours[tr])
        rmses.append(np.sqrt(mean_squared_error(y_ours[va], m.predict(X_ours_sc[va]))))
    return float(np.mean(rmses)), rmses

def two_stage_cv(X_lit_pre, y_lit_pre, seed=42, cv=5):
    # PFT: Stage-1 XGB on (quantile-normalised filtered lit + train fold); Stage-2 LGB with the
    # Stage-1 prediction appended as one extra feature; trained on lab labels only.
    rmses = []
    for tr, va in KFold(cv, shuffle=True, random_state=seed).split(X_ours_sc):
        y_pre = quantile_normalize_y(y_lit_pre, y_ours[tr])
        pre = xgb.XGBRegressor(**xgb_params()).fit(
            np.vstack([X_lit_pre, X_ours_sc[tr]]), np.concatenate([y_pre, y_ours[tr]]))
        fin = lgb.LGBMRegressor(**lgb_params()).fit(
            np.hstack([X_ours_sc[tr], pre.predict(X_ours_sc[tr]).reshape(-1, 1)]), y_ours[tr])
        p = fin.predict(np.hstack([X_ours_sc[va], pre.predict(X_ours_sc[va]).reshape(-1, 1)]))
        rmses.append(np.sqrt(mean_squared_error(y_ours[va], p)))
    return float(np.mean(rmses)), rmses

print('Appendix ready. RUN_HEAVY =', RUN_HEAVY)'''))

# ---- Theme 1 ----
cells.append(('md', r'''### Theme 1 — "Did you prepare the data correctly?"

**Q1. Why StandardScaler, not MinMaxScaler? Is linearity lost?**

Both are **linear (affine)** per-feature maps — `z=(x−μ)/σ` and `x'=(x−min)/(max−min)`. So **no, linearity is not lost**: z-scoring cannot bend a straight relationship (correlations and linear fits are preserved). The only *nonlinear* rescaling we use is **quantile normalisation on the labels** (Q5) — likely what the question pictured.

Why StandardScaler: (a) scaling only affects the *distance/kernel/linear* steps — PCA, the DRST logistic classifier, the KMM RBF kernel — and is **irrelevant to the tree models** that predict yield (trees split on thresholds); (b) it is the textbook default for PCA/RBF and is robust to the extreme literature compositions, whereas MinMax lets one outlier squash the informative bulk into [0,1].'''))

cells.append(('md', r'''**Q2. What are the "PCA values", how are they found, why only 2?**

PCA finds orthogonal directions of maximum variance (eigen-decomposition / SVD of the centred features). The "values" are each sample's **scores** (coordinates on PC1/PC2) plus the **explained-variance ratio** per component. We keep 2 because PCA here is a **2-D diagnostic picture** of the domain gap — *not* part of the model. Two PCs capture only part of the variance, so the visible gap is a *lower bound*; the rigorous measure is the DRST classifier on all 67 features (**78.5% OOD**).'''))

cells.append(('md', r'''**Q3. PCA on literature only or the whole set? Ideal way to think?**

Fit on the **combined** set (lab + literature) so both live in **one shared projection** — the only way a scatter can show whether they overlap. Fitting on one domain and reading the other's spread as meaningful is the classic trap. General rule: shared basis for *comparison*, and for a real covariate-shift number use a domain classifier, not a 2-PC picture.'''))

cells.append(('md', r'''**Q4. For PCA you used 3,000 — why not 3,852?**

A misread of *which* number was thinned. **All 3,852 literature rows are used.** The 3,000 is a subsample of the **89,074 lab** points, taken *only* so the scatter is legible (89k blue dots would bury the orange literature). Nothing is dropped from the literature.'''))

# ---- Theme 2 ----
cells.append(('md', r'''### Theme 2 — "Did you pick methods & thresholds honestly?"

**Q5 & Q6. Why τ = 0.30 for DRST? Did you try others?**

I swept **every** threshold 0→1. Two findings:

1. **Single-stage** DRST (just adding filtered literature to training) barely helps at *any* τ — best point τ=0.85 → RMSE 2.127 vs baseline 2.133; at τ=0.30 it is actually *worse* (2.181). Even lab-like literature still carries the +3.42 pp label shift that poisons the loss.
2. **Two-stage PFT** (filtered literature only trains the Stage-1 prior) is **flat and robust** across τ₁ — every threshold lands in 1.906–1.913 (~10% below baseline). τ=0.30 is statistically tied with the global best, so it is a **safe, non-cherry-picked** choice: **the win is architectural, not a tuned threshold.**

![DRST single-stage threshold sweep](fig_drst_threshold_sweep.png)

![Two-stage PFT tau1 sweep](fig_pft_tau1_sweep.png)'''))

cells.append(('code', r'''# Q5/Q6 — regenerate both sweeps (gated; ~4-6 min). Otherwise the saved figures are shown above.
if RUN_HEAVY:
    import matplotlib.pyplot as plt
    base, _ = baseline_cv()
    taus = np.round(np.arange(0.05, 0.96, 0.05), 2)
    ss, ts = [], []
    for t in taus:
        m = p_ours_lit >= t
        rr = []
        for tr, va in KFold(5, shuffle=True, random_state=42).split(X_ours_sc):
            X = np.vstack([X_ours_sc[tr], X_lit_sc[m]]); Y = np.concatenate([y_ours[tr], y_lit[m]])
            mm = lgb.LGBMRegressor(**lgb_params()).fit(X, Y)
            rr.append(np.sqrt(mean_squared_error(y_ours[va], mm.predict(X_ours_sc[va]))))
        ss.append(np.mean(rr))
        ts.append(two_stage_cv(X_lit_sc[m], y_lit[m])[0])
        print('tau=%.2f  n=%4d  single=%.4f  two-stage=%.4f' % (t, int(m.sum()), ss[-1], ts[-1]))
    plt.figure(figsize=(7, 4))
    plt.axhline(base, ls='--', c='gray', label='baseline %.3f' % base)
    plt.plot(taus, ss, 'o-', label='single-stage DRST'); plt.plot(taus, ts, 's-', label='two-stage PFT')
    plt.xlabel('threshold tau'); plt.ylabel('5-fold CV RMSE'); plt.legend(); plt.show()
else:
    display(Image('fig_drst_threshold_sweep.png')); display(Image('fig_pft_tau1_sweep.png'))'''))

cells.append(('md', r'''**Q8. Does KMM select the same 782 samples as DRST? Are they compatible?**

Not identical by construction — DRST *hard-keeps* 782 rows at τ=0.30; KMM assigns *soft weights* [0,5] to all 3,852. But they read the **same signal**: Pearson **r = 0.79**, and DRST's 782 overlap **634 (81%)** with KMM's top-782 (Jaccard 0.68). They **corroborate** rather than compete — the agreement is evidence the covariate-shift signal is real and method-independent.

![KMM weight vs DRST score, and selection overlap](fig_kmm_drst_overlap.png)'''))

cells.append(('code', r'''# Q8 — light, always runs: quantify DRST/KMM agreement from variables already in memory.
r = float(np.corrcoef(p_ours_lit, w_kmm)[0, 1])
N = int((p_ours_lit >= 0.30).sum())
drst_set = set(np.where(p_ours_lit >= 0.30)[0])
kmm_top  = set(np.argsort(-w_kmm)[:N])
inter = len(drst_set & kmm_top); jacc = inter / len(drst_set | kmm_top)
print('Pearson r (DRST score vs KMM weight) = %.3f' % r)
print('DRST keeps %d rows; KMM top-%d overlaps %d (%.0f%%), Jaccard %.2f' % (N, N, inter, 100*inter/N, jacc))
display(Image('fig_kmm_drst_overlap.png'))'''))

# ---- Theme 3 ----
cells.append(('md', r'''### Theme 3 — "Is the result real, or luck?"

**Q7. Did you use the whole dataset for training? How?**

Yes — all 89,074 lab rows, via **asymmetric 5-fold CV**: every lab row trains in 4/5 folds and is scored once. The key: the **validation fold is always lab-only**; literature only ever enters the *training* side. So the RMSE measures accuracy on *our* experiments, not a lab+literature average.

**Q11. How many times did you run the transfer method?**

Originally **once** (seed 42) — a fair criticism. Now **10 independent seeds** (fresh split each run). Results below.'''))

cells.append(('code', r'''# Q11/Q14/Q15 — 10-seed baseline vs PFT (gated; ~7 min). Otherwise the saved figure is shown.
if RUN_HEAVY:
    SEEDS = [0, 1, 2, 7, 13, 21, 42, 77, 123, 2025]
    m30 = p_ours_lit >= 0.30
    b, pf = [], []
    for s in SEEDS:
        bs = baseline_cv(s)[0]; ps = two_stage_cv(X_lit_sc[m30], y_lit[m30], seed=s)[0]
        b.append(bs); pf.append(ps)
        print('seed=%5d  baseline=%.4f  PFT=%.4f  delta=%+.4f' % (s, bs, ps, ps-bs))
    b = np.array(b); pf = np.array(pf)
    print('')
    print('baseline %.4f +/- %.4f   |   PFT %.4f +/- %.4f' % (b.mean(), b.std(ddof=1), pf.mean(), pf.std(ddof=1)))
    print('PFT wins %d/%d   paired t p=%.2e   Wilcoxon p=%.2e' % ((pf < b).sum(), len(SEEDS),
          ttest_rel(b, pf)[1], wilcoxon(b, pf)[1]))
else:
    display(Image('fig_repeated_runs.png'))'''))

cells.append(('md', r'''**Q14 & Q15. Mean/std of RMSE across runs, and is it significant?**

- Baseline **2.121 ± 0.006**  vs  PFT **1.909 ± 0.002**
- PFT beats baseline in **10 / 10** runs — the curves never cross (worst PFT 1.912 still beats best baseline 2.113).
- Paired t-test **p = 3.9×10⁻¹⁵**, Wilcoxon signed-rank **p = 2.0×10⁻³**.

Identical hyper-parameters every seed, no per-seed tuning — nothing is force-fit. A consistent win across 10 independent splits with p this small ⇒ **not a lucky split.** PFT is also *tighter* (std 0.002 vs 0.006) — better **and** more stable.

![10-seed repeated runs](fig_repeated_runs.png)'''))

cells.append(('md', r'''**Q16. Did you validate on untouched data?**

Yes — a **true held-out test**: a random 20% of lab rows set aside and **never used in any training** (not DRST, not Stage 1, not Stage 2): baseline 2.097 → **PFT 1.892** (R² 0.763), ~10% better — matching the CV story on data the model never saw.

**Honest caveat (and a correction).** On the *OOD-literature* stress test (non-impregnation methods), the faithful run gives baseline 6.53 → **PFT 7.60 — worse**, and does **not** reproduce the deck's "6.53 → 3.60". This is *by design*: quantile-normalisation calibrates the model to the **lab yield scale** (~5.25%), so it under-predicts the higher-yield literature. PFT optimises *our* regime — that is the goal — it is **not** a literature-extrapolation model. **That deck number should be corrected.**'''))

cells.append(('code', r'''# Q16 — true held-out 20% test (gated; ~30 s). Otherwise prints the saved result.
if RUN_HEAVY:
    rng = np.random.default_rng(2025); perm = rng.permutation(len(X_ours_sc))
    nt = int(0.20 * len(perm)); te, trn = perm[:nt], perm[nt:]
    m30 = p_ours_lit >= 0.30
    mb = lgb.LGBMRegressor(**lgb_params()).fit(X_ours_sc[trn], y_ours[trn])
    rb = np.sqrt(mean_squared_error(y_ours[te], mb.predict(X_ours_sc[te])))
    yq = quantile_normalize_y(y_lit[m30], y_ours[trn])
    pre = xgb.XGBRegressor(**xgb_params()).fit(
        np.vstack([X_lit_sc[m30], X_ours_sc[trn]]), np.concatenate([yq, y_ours[trn]]))
    fin = lgb.LGBMRegressor(**lgb_params()).fit(
        np.hstack([X_ours_sc[trn], pre.predict(X_ours_sc[trn]).reshape(-1, 1)]), y_ours[trn])
    rp = np.sqrt(mean_squared_error(y_ours[te],
        fin.predict(np.hstack([X_ours_sc[te], pre.predict(X_ours_sc[te]).reshape(-1, 1)]))))
    print('Held-out 20%% (never trained on): baseline %.4f | PFT %.4f  (%.1f%% better)' % (rb, rp, 100*(rb-rp)/rb))
else:
    print('Held-out 20% (never trained on): baseline 2.097 -> PFT 1.892  (9.8% better, R2 0.763)')'''))

# ---- Theme 4 ----
cells.append(('md', r'''### Theme 4 — "Did it learn real chemistry?"

**Q12. Is SHAP on lab data or everything? How should it be?**

On a 3,000-row **lab** subsample (with the prior feature), explaining the final Stage-2 LightGBM with exact **TreeExplainer**. That is correct: the model predicts *lab* yields and operates on lab chemistry, so we explain it there. Explaining on literature would instead probe OOD extrapolation — a different question.'''))

cells.append(('md', r'''**Q13. How do I read the beeswarm (blue/red, lines), and is it stable over 10 runs?**

- **Each dot = one lab catalyst.** **x = SHAP value** (signed push on that prediction, in yield-% units — right raises yield). **Colour = the feature's value** (red high, blue low).
- **Blue near the zero line** = low/absent feature value with little effect (e.g. an element absent → no push). **Red spread to one side** = high value's directional effect (temperature red→right = hotter → higher yield; Li/K red→left = suppressive).
- All dots are **lab**; the literature's influence enters through the single `lit_prior_prediction` feature — which ranks **#1**.
- **Stability:** recomputed on **10 independent subsamples** — `lit_prior_prediction` is #1 in **10/10 runs**, and 9 features are top-10 in every run. Not an artefact of one draw.
- **Why not LIME?** For a tree model TreeSHAP is *exact* and consistent; a local linear LIME surrogate would be noisier and add nothing here.

![SHAP feature-importance stability across 10 runs](fig_shap_stability.png)'''))

cells.append(('code', r'''# Q13 — SHAP top-feature stability across 10 subsamples (gated; ~1 min). Otherwise shows saved figure.
if RUN_HEAVY:
    from collections import Counter
    aug_names = FEATURES + ['lit_prior_prediction']
    model = lgb.LGBMRegressor(**lgb_params()).fit(X_aug, y_ours)
    expl = shap.TreeExplainer(model); tops = []
    for s in range(10):
        idx = np.random.default_rng(s).choice(len(X_aug), 2000, replace=False)
        imp = np.abs(expl.shap_values(X_aug[idx])).mean(0)
        tops.append([aug_names[j] for j in np.argsort(-imp)[:10]])
    c1 = Counter(t[0] for t in tops)
    allc = Counter(f for t in tops for f in t)
    print('SHAP #1 feature:', c1.most_common(1)[0][0], '- ranked #1 in', c1.most_common(1)[0][1], '/10 runs')
    print('In top-10 in ALL 10 runs:', [f for f, n in allc.items() if n == 10])
else:
    display(Image('fig_shap_stability.png'))'''))

# ---- Theme 5 ----
cells.append(('md', r'''### Theme 5 — "Where does this sit in the literature?"

**Q9. Is this transfer method novel?**

The building blocks exist — *stacked generalisation* (Wolpert 1992), *Δ-ML / multi-fidelity* learning in materials chemistry (source-prediction-as-feature), and *importance weighting* / *label-shift correction*. **But** Δ-ML *trusts* the source label (`final = source_pred + Δ`); we deliberately pass the prior only as a **feature**, so Stage-2 uses its *ranking* while ignoring the biased absolute value. **Novel here:** the combination *DRST filter → quantile-normalised Stage-1 prior → prediction-as-feature Stage-2* applied to **OCM literature→lab transfer** under simultaneous covariate *and* label shift.

**Q10. What is model drift — is it relevant? How is it corrected?**

Drift = the statistics a model relies on change: **covariate shift** (P(x)), **label/prior shift** (P(y)), **concept drift** (P(y|x)). We do not have *temporal* drift but a **static two-domain shift** — mathematically the same family: our +3.42 pp label gap is label shift, the 78.5% OOD is covariate shift. Correction menu — importance weighting (**KMM**), instance selection (**DRST**), label-shift correction (**quantile normalisation**), recalibration/fine-tuning (**PFT Stage 2**) — all already used.'''))

# ---- Theme 6 ----
cells.append(('md', r'''### Theme 6 — "Data honesty & next steps"

**Q17. How could we generate synthetic data with a controlled proportion?**

We currently generate **none** (the 15% "oversampling" ablation was duplication, and it *did not* help — informative in itself). Real options: **SMOGN/SMOTER** (regression SMOTE — interpolate neighbours), **jitter** (Gaussian noise on real compositions), **CTGAN/TVAE** (tabular generative models), **conditional generation** (condition on yield bins to fight label shift), or **physics/simulation surrogates**. For a synthetic fraction *p*: draw `n_synth = p/(1−p)·n_real`. **Critical:** keep synthetic rows out of validation/test (validate only on real held-out lab), and enforce chemical validity (feasible elements, loading constraints) — naive GANs can produce unphysical catalysts.

---

**How to present this deck of answers:** lead with the **6 themes**, not 17 points. The strongest single exhibit is the **10-seed plot** (Q14/15) — "PFT wins every time, p < 10⁻¹⁴." The most credibility-building sentence is the **OOD correction** (Q16) — it shows the pipeline is reported honestly.'''))

# ---- assemble ----
def mk(t, src):
    if t == 'md':
        return nbformat.v4.new_markdown_cell(src)
    return nbformat.v4.new_code_cell(src)

for t, src in cells:
    nb.cells.append(mk(t, src))

nbformat.write(nb, NB)
print('Appended %d cells. Notebook now has %d cells.' % (len(cells), len(nb.cells)))
