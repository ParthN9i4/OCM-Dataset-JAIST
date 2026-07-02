# OCM Project — Responses to Presentation Feedback

This document answers the 17 feedback points raised on the Chapter 1–8 presentation.
Each answer states **(a)** the concept, **(b)** exactly what we did and where it lives in the
code, and **(c)** any new analysis run to back the claim. New experiments were produced by
`feedback_experiments.py` (results in `feedback_results.json`, figures `fig_*.png`).

Pipeline facts used throughout:
- **Lab data** = 89,074 rows (`year==2025`, all Impregnation, mean Y(C2)=5.25 %).
- **Literature** = 3,852 rows (`year≤2019`, 15+ prep methods, mean Y(C2)=8.67 %).
- **Features** = 67: `Temperature_C`, `prep_enc` (label-encoded Preparation), + 65 element columns.
- **Target** = `Y(C2), %`.
- Label shift = +3.42 pp; covariate shift = 78.5 % of literature out-of-distribution (OOD).

---

## 1. Why StandardScaler instead of MinMaxScaler? Is linearity lost?

**The two transforms.**

| | StandardScaler (z-score) | MinMaxScaler |
|---|---|---|
| Formula | `z = (x − μ) / σ` | `x' = (x − min) / (max − min)` |
| Output | mean 0, unit variance, **unbounded** | bounded to **[0, 1]** |
| Outlier behaviour | bulk of data keeps spread; outliers stay far but don't squash the rest | a single extreme min/max **compresses all other points** into a narrow band |
| Per-feature map | affine: `z = a·x + b` | affine: `x' = a·x + b` |

**Both are strictly linear (affine) per-feature maps** — multiply by a constant, add a constant.

**Does StandardScaler lose linearity? No — this is a misconception.**
Because z-scoring is `z = (x − μ)/σ`, it cannot bend a straight relationship: Pearson
correlations are unchanged, linear-model fits are identical up to a rescaling of the
coefficients, and linear separability is preserved. What it changes is only the *scale and
origin* of each axis. The transform that *is* nonlinear in our project is **quantile
normalisation of the labels** (rank-based, see Q5/Q15) — it is possible the feedback conflated
the two. StandardScaler on the *features* is linear and rank-preserving.

**Why we chose it.**
1. Scaling only matters for our **distance/kernel/linear** steps — PCA (Q2), the DRST logistic
   classifier (Q5), and the KMM RBF kernel (Q8). For the tree models (XGBoost, LightGBM) that
   actually predict yield, scaling is **irrelevant** (trees split on thresholds and are
   monotonic-invariant), so this choice does not touch the headline RMSE.
2. `Temperature_C` lives on ~500–900 while element loadings are ~0–15. Unscaled, temperature
   would dominate every Euclidean distance and the first principal component. Both scalers fix
   this; the question is which is safer.
3. **StandardScaler is the textbook default for PCA and RBF kernels**, and it is robust to the
   extreme catalyst compositions in the literature set. MinMaxScaler would let one outlier
   composition define the [0,1] range and crush the informative bulk into a sliver — bad for
   distance-based covariate-shift estimation.

Code: `scaler = StandardScaler(); X_ours_sc = scaler.fit_transform(X_ours); X_lit_sc = scaler.transform(X_lit)`
(fit on lab only, then applied to literature so both sit on the *same* ruler).

---

## 2. What are the PCA values, how are they found, and why only the first 2?

**What PCA is.** PCA finds orthogonal directions ("principal components") along which the data
varies most. It is computed by eigendecomposition of the feature covariance matrix (equivalently
the SVD of the mean-centred data). Component 1 is the direction of maximum variance; component 2
is the next orthogonal direction; and so on.

**"PCA values" = two things we report:**
- **Scores** — each sample's coordinates after projecting the 67-D feature vector onto PC1/PC2
  (the `x,y` of each dot in the scatter).
- **Explained-variance ratio** — the fraction of total variance each component captures (printed
  on the axis labels, e.g. `PC1 (xx.x% var)`).

**Why only 2.** Purely for a **2-D picture of the domain gap**. PCA is *not* used anywhere in the
predictive pipeline — it is a diagnostic. Two components give a plottable plane on which we can
see that literature (orange) and lab (blue) occupy different regions. Two PCs capture only part
of the total variance, so the visual gap is a *lower bound*: in the full 67-D space the
separation is larger. We therefore **confirm the gap quantitatively** with the DRST domain
classifier on all 67 features (78.5 % OOD), not with the picture alone.

Code: `PCA(n_components=2, random_state=42)`.

---

## 3. Where did we check PCA — literature only or the whole set? What is the ideal way to think?

**What we did.** We fit PCA on the **combined** set (a 3,000-point lab subsample + all 3,852
literature points), then plotted both. This is the correct choice for *comparing* two groups: to
judge whether they overlap you must embed them in **one shared projection**. Fitting PCA on the
literature alone would reveal literature's internal structure but give no common axes on which to
compare lab against it.

**Ideal general principle.**
- Define the projection on a **common basis** (fit on the union, or fit on one domain and
  *project* the other) so the two domains are directly comparable.
- Never fit PCA on one domain and then read the other domain's spread as if it were meaningful
  internal variance — that is the classic trap.
- Always report explained variance and remember it is a projection.
- For a *rigorous* covariate-shift measurement, don't rely on a 2-PC picture at all — use a
  **domain classifier** (our DRST), which is exactly what we did (Q5).

---

## 4. For PCA we used 3,000 — why not 3,852?

This is a misread of which number was subsampled. **All 3,852 literature samples were used.**
The `3000` is a subsample of the **89,074 lab** points, taken *only* so the scatter plot is
legible — 89 k blue dots would form an opaque blob that hides the 3,852 orange literature dots.

- Literature: **3,852 / 3,852 used** (100 %).
- Lab: subsampled to 3,000 *for the visualization only*; the modelling uses all 89,074.

So nothing was thrown away from the literature; PCA scores are unaffected in expectation by
thinning the lab cloud for plotting.

---

## 5. Why τ = 0.30 for DRST? Did you evaluate other thresholds? (full analysis)

**What the threshold does.** DRST trains a logistic classifier to output `P(lab | x)` for each
literature sample, then keeps literature samples with `P(lab|x) ≥ τ` to add to training. τ trades
off two errors:
- **τ too low** → keep many literature rows, but they carry covariate + label shift → they
  *contaminate* training → RMSE rises.
- **τ too high** → keep only a handful of very lab-like rows → too little extra signal →
  benefit shrinks and RMSE drifts back toward baseline.

So RMSE-vs-τ is **U-shaped** with a sweet spot.

**Two sweeps were run — and they tell an important story.**

**(i) Single-stage DRST augmentation** (add the kept literature rows directly to training), τ from
0.05→0.95, same 5-fold CV. Baseline (no transfer) = **2.1331**.

| τ | kept | RMSE | | τ | kept | RMSE |
|---|---|---|---|---|---|---|
| 0.05 | 1361 | 2.1963 | | 0.50 | 521 | 2.1675 |
| 0.10 | 1193 | 2.1880 | | 0.55 | 470 | 2.1629 |
| 0.15 | 1091 | 2.1829 | | 0.60 | 407 | 2.1436 |
| 0.20 | 994 | 2.1709 | | 0.65 | 375 | 2.1466 |
| 0.25 | 894 | 2.1791 | | 0.70 | 330 | 2.1375 |
| **0.30** | **782** | **2.1809** | | 0.75 | 290 | 2.1399 |
| 0.35 | 711 | 2.1747 | | 0.80 | 262 | 2.1321 |
| 0.40 | 603 | 2.1628 | | **0.85** | **209** | **2.1270** ← best |
| 0.45 | 547 | 2.1531 | | 0.90 | 143 | 2.1335 |
|  |  |  | | 0.95 | 83 | 2.1320 |

**Honest finding:** *single-stage* DRST augmentation barely helps at any threshold — the best
point (τ=0.85, RMSE 2.1270) is only 0.006 below baseline, and at τ=0.30 it is actually **worse**
(2.1809). Simply dumping filtered literature into the training pool does not work, because even
"lab-like" literature still carries the +3.42 pp label shift that poisons the loss.

**(ii) Two-stage PFT** (use the filtered literature only to train the Stage-1 prior, then feed its
prediction as a feature) — τ₁ sweep, same CV:

| τ₁ | kept | PFT RMSE | | τ₁ | kept | PFT RMSE |
|---|---|---|---|---|---|---|
| 0.05 | 1361 | 1.9125 | | 0.30 | 782 | **1.9066** |
| 0.10 | 1193 | 1.9100 | | 0.40 | 603 | 1.9097 |
| 0.15 | 1091 | 1.9107 | | 0.50 | 521 | 1.9087 |
| 0.20 | 994 | 1.9086 | | 0.65 | 375 | 1.9087 |
| 0.25 | 894 | 1.9092 | | 0.80 | 262 | 1.9065 |

**This is the key result.** The two-stage PFT is **flat and robust across τ₁** (every threshold
lands in 1.906–1.913, all ≈10 % below baseline). τ₁ = 0.30 (1.9066) is statistically tied with
the global best (τ₁ = 0.80, 1.9065 — a 0.0001 difference, i.e. noise). So **τ = 0.30 is a safe,
near-optimal, *non-cherry-picked* choice**, and the improvement does **not** depend on tuning the
threshold — it comes from the *architecture* (label-shift quarantined into a feature), not from a
lucky τ. See `fig_drst_threshold_sweep.png` (single-stage) and `fig_pft_tau1_sweep.png`
(two-stage).

---

## 6. Plot RMSE vs threshold from 0 to 1 (DRST)

See **`fig_drst_threshold_sweep.png`** (single-stage DRST, Exp A) and **`fig_pft_tau1_sweep.png`**
(two-stage PFT). The first shows that single-stage augmentation hugs the baseline at every τ; the
second shows the two-stage PFT sitting ~10 % below baseline and essentially flat across τ₁. Both
plots carry the baseline line, the τ=0.30 marker, the empirical optimum, and (right axis on the
single-stage plot) the number of literature samples surviving each threshold.

---

## 7. Did we use the entire dataset for training? How was it implemented?

**Yes — every lab row is used.** Validation is **5-fold cross-validation on the lab data**, so
each of the 89,074 lab rows is in the training set for 4 of 5 folds and is held out for scoring
exactly once. Implementation is an **asymmetric CV**:

- The **validation fold is always lab-only** — never literature.
- **Literature only ever enters the training side** (filtered/weighted/quantile-normalised
  depending on method).

This makes the reported RMSE measure "accuracy on *our* experiments," not a diluted lab+lit
average. Code: `KFold(n_splits=5, shuffle=True, random_state=seed)` inside `evaluate_cv_ours`
and `two_stage_cv`. The final reported model is trained on **all** lab rows + the filtered
literature.

---

## 8. Does KMM select the same 782 samples as DRST? Are the methods compatible?

**They are not identical by construction, but they strongly agree.** DRST *hard-thresholds*
(keep 782 rows at τ=0.30); KMM assigns every one of the 3,852 literature rows a *continuous*
importance weight in [0, 5] (RBF kernel mean matching) — it never hard-selects.

**New quantification (Exp E):**
- Pearson correlation between KMM weight and DRST score `P(lab|x)`: **r = 0.79**.
- Comparing DRST's 782 kept rows to KMM's top-782 by weight:
  overlap = **634** rows (81 % of the DRST set), Jaccard = **0.68**.

See **`fig_kmm_drst_overlap.png`**. Interpretation: both methods read the *same* covariate-shift
signal — "this literature row looks like our chemistry." KMM keeps the full gradient (softer,
retains partial-credit samples); DRST gives a sharp, interpretable cut. They are **corroborating,
not competing**; the high correlation is evidence the covariate-shift signal is real and
method-independent.

---

## 9. Has this transfer-learning approach been done before in the literature?

**The building blocks exist; the specific combination and the OCM application do not.** (Full
cited review researched separately.)

- **"Source-model prediction as an extra feature"** is known — *stacked generalisation*
  (Wolpert, 1992) and especially **Δ-ML / multi-fidelity learning** in materials/quantum
  chemistry (Ramakrishnan 2015; npj Comp. Mater. 2022; Vinod 2025), where a cheap model's output
  feeds a better model. **But** Δ-ML *trusts* the source label (`final = source_pred + Δ`); ours
  deliberately does **not** — it passes the prior only as a *feature*, so Stage 2 can use its
  ranking while ignoring the biased absolute value.
- **Covariate-shift filtering / importance weighting** (Sugiyama 2006; KMM, Huang 2007) — our
  DRST and KMM are instances; not novel alone.
- **Label-shift correction** (BBSE/RLLS; Lipton 2018, Alexandari 2020) reweights examples;
  *rescaling labels by quantile matching* is a less common variant, and a near-parallel academic
  formalisation ("Transfer Learning Through Conditional Quantile Matching") only appeared in 2026.

**Novel here:** the combination *DRST covariate filter → quantile-normalised Stage-1 prior →
prediction-as-feature Stage-2 calibrated on target labels only*, applied to **OCM catalyst-yield
transfer from public literature to a private lab** under simultaneous covariate **and** label
shift. No prior OCM/catalysis paper does literature→lab transfer this way.

---

## 10. What is model drift? Is it relevant here? How is it corrected?

**Model / concept drift** = the statistics the model relies on change, degrading it. Three kinds:
- **Covariate shift** — `P(x)` changes (inputs move).
- **Label / prior shift** — `P(y)` changes (target distribution moves).
- **Concept drift** — `P(y|x)` changes (the input→output rule itself changes).

**Relevance.** We don't have an *online temporal* deployment-drift problem; we have a **static
two-domain shift** (literature 1982–2019 vs lab 2025). Mathematically it is the same family as
drift:
- our **+3.42 pp label gap** *is* a label/prior-shift instance;
- our **78.5 % OOD** *is* a covariate-shift instance.

So drift concepts underpin the entire project — the whole method is a drift-correction pipeline.

**Correction methods (and which we use):**
- Importance weighting → **KMM** ✓
- Instance selection / filtering → **DRST** ✓
- Label-shift correction → **quantile normalisation** ✓
- Recalibration / fine-tuning around the offset → **PFT Stage 2** ✓
- Feature alignment (CORAL, domain-adversarial) — not used (tree models, not deep).
- Monitoring (KS-test, PSI drift detectors), periodic retraining — relevant for future
  deployment.

---

## 11. How many times did we run the transfer-learning method to confirm it is better?

**Originally once** (a single 5-fold CV at `random_state=42`). That is a fair criticism, so we
re-ran it. **New (Exp B): 10 independent seeds** — `[0,1,2,7,13,21,42,77,123,2025]` — each a fresh
5-fold split and fresh model seed, for both baseline and PFT. Results in Q14/Q15.

---

## 12. Is the SHAP beeswarm on lab data or the entire data? How should it be done? What did we do?

**What we did.** SHAP (`TreeExplainer`, exact) on a 3,000-sample subsample of the **lab** data,
with the `lit_prior_prediction` feature appended, explaining the final **Stage-2 LightGBM** model.

**Why that is correct.** SHAP explains *a specific model on a specific dataset*. Our final model
predicts **lab** yields and is meant to operate on lab-like chemistry, so explaining it on lab
data answers the right question: "what drives predictions in our operating regime?" Explaining on
literature would instead probe how the model *extrapolates OOD* — a different (also legitimate)
question we can add separately. Subsampling 3,000 is fine for a beeswarm: mean|SHAP| rankings
converge quickly, and we now verify that explicitly (Q13/Exp D).

---

## 13. How to read the beeswarm (blue/red, lines, lab vs lit), and is it stable over ≥10 runs?

**Reading a SHAP beeswarm:**
- Each **dot = one sample** (here, one lab catalyst).
- **x-position = SHAP value** = that feature's signed push on that prediction, in **yield-% units**
  (right = pushes predicted yield up, left = down).
- **Colour = the feature's *value*** (red = high, blue = low).
- Vertical thickness = density of points (jitter), and features are ordered top-to-bottom by
  mean|SHAP| (overall importance).

**Your specific questions:**
- **"Blue bubbles near the grey/zero line"** — blue = *low* feature value; sitting near SHAP≈0
  means those samples' predictions are barely affected. For an element column, blue≈"element
  absent (loading≈0)" clustered at 0 means *absence of that element doesn't move the prediction*,
  while the coloured tail shows what *presence* does.
- **"Red points / colour code"** — red = *high* feature value. If red sits at **positive** SHAP,
  a high value of that feature **raises** predicted yield (e.g., temperature red→right = hotter
  → higher predicted yield, correct physics up to the optimum); red at negative SHAP means high
  value **lowers** yield (e.g., Li/K loadings, which we see as suppressive).
- **"Lab near 0 / literature data"** — the beeswarm is computed *only on lab samples*, so every
  dot is lab. The literature's influence appears through the single `lit_prior_prediction`
  feature: its colour shows whether the literature-expert's optimism (high prior) systematically
  pushes the lab prediction up — and it ranks as the #1 feature.

**Stability over 10 runs (new, Exp D).** We recomputed SHAP on **10 independent 2,000-sample
subsamples** and compared the top-feature rankings. See **`fig_shap_stability.png`** (mean|SHAP|
± std). Findings: the **`lit_prior_prediction` feature was ranked #1 in all 10/10 runs**, and **9 features appeared in the top-10 in every single run** (`lit_prior_prediction`, `Temperature_C`, `Ba`, `Zr`, `Mn`, `La`, `Ce`, `Cu`, `Al`). The mean|SHAP| error bars are small relative to the gaps between features. This shows the importance ranking is **not an artefact of one
random draw**.

---

## 14. Mean & std of RMSE; plot all 10 runs; should be below baseline every time; no force-fitting

**New (Exp B), 10 seeds, identical hyper-parameters, nothing tuned per seed:**

- **Baseline:** RMSE = **2.1207 ± 0.0055**
- **PFT:** RMSE = **1.9090 ± 0.0017**
- **Mean improvement:** 0.2117 RMSE (9.98 % relative)
- **PFT beat baseline in 10 / 10 runs.**

See **`fig_repeated_runs.png`** (left: both curves across seeds; right: per-seed improvement).
The two curves never cross: the worst PFT seed (1.9118) still beats the best baseline seed (2.1132) by a wide margin. No per-seed tuning, no early stopping on the validation fold, identical
`xgb_params`/`lgb_params` throughout — nothing is force-fit.

---

## 15. Demonstrate the results are not accidental

From the same 10-seed paired experiment (Exp B), testing baseline-vs-PFT RMSE **per seed**:

- **Paired t-test:** t = 103.0, p = **3.9×10⁻¹⁵**
- **Wilcoxon signed-rank:** p = **2.0×10⁻³**
- PFT wins **10 / 10**; the std across seeds (0.0017) is far smaller than the gap
  to baseline (0.2117).

Note also how tight PFT is across seeds (std 0.0017) versus the baseline (std 0.0055) — PFT is both better *and* more stable. A consistent win across 10 independent random splits with p-values this small
means the improvement is **statistically significant, not a lucky split**.

---

## 16. Did we validate the model on untested data?

**Originally: cross-validation only** (every lab row scored once, but no fully untouched
hold-out). We have now added a **true held-out test set (Exp C)**: a random **20 % of lab rows
(seed 2025) set aside and never used in any training** (not in DRST, not in Stage 1, not in
Stage 2).

- **Held-out baseline:** RMSE = 2.0970 (R² = 0.708)
- **Held-out PFT:** RMSE = 1.8917 (R² = 0.763) — 9.8 % better.

This held-out result (independent of the CV folds) confirms the ~10 % gain is real on data the
model has genuinely never touched.

**An honest caveat on the OOD-literature stress test — with a leakage correction.** We also
evaluated on the 2,139 literature rows made by *non-impregnation* methods (chemistry the lab never
uses). The old deck's "6.53 → 3.60 (−45 %)" was **data leakage**: that prior had been trained on the
OOD test rows and their labels, so it reproduces (3.62) *only* when the leak is present. A clean,
**leak-free** ablation (`qn_tradeoff.py`, `fig_qn_tradeoff.png`) — prior never sees the OOD rows —
gives the honest picture:

| Config (leak-free) | In-dist CV RMSE | OOD RMSE |
|---|---|---|
| Baseline | 2.133 | 6.53 |
| DRST-filtered + raw | 1.915 | 6.11 |
| **DRST-filtered + QN (= PFT)** | **1.910** | 6.77 |
| Full impreg-lit + raw | 1.924 | 6.05 |
| Full impreg-lit + QN | 1.913 | 6.32 |

So leak-free OOD is ~6.0–6.8 (**near baseline**, not −45 %). Quantile normalisation calibrates the
prior onto the **lab yield scale** (mean ≈ 5.25 %) while non-impregnation literature has higher true
yields (≈ 8–9 %); this is why **QN improves in-distribution CV and slightly worsens OOD** — a
deliberate dial trading extrapolation for local accuracy. The defensible claim is the
**in-distribution** result (−10.6 %, 10/10 seeds, p < 10⁻¹⁴, held-out 2.097 → 1.892), not OOD
literature superiority.

---

## 17. How could we generate synthetic data with a controlled proportion?

**What we do now:** *no synthetic data.* We only *resampled with replacement* to a target
fraction (15 %) in two ablations (Methods E/F) — that is duplication, not synthesis, and it did
**not** help (RMSE rose), which is itself informative.

**Options to actually synthesise, with proportion control:**
1. **SMOGN / SMOTER** — the regression variants of SMOTE: interpolate between near-neighbours
   (with Gaussian noise in sparse target regions). Good for rebalancing rare high-yield cases.
2. **Jittering / noise augmentation** — add small Gaussian perturbations to real compositions.
3. **Tabular generative models** — **CTGAN / TVAE** (or a diffusion-tabular model): train on real
   data, then sample as many synthetic rows as needed.
4. **Conditional generation** — generate conditioned on target-yield bins to balance the label
   distribution (directly attacks our label shift).
5. **Physics / simulation surrogates** — a kinetic or DFT-based model to emit plausible
   `(composition, conditions) → yield` tuples.

**Setting the proportion:** for a synthetic fraction *p*, draw `n_synth = p/(1−p) · n_real`.

**Critical caveats for catalysis:**
- Keep synthetic rows **out of validation/test** — validate only on *real* held-out lab data,
  or metrics are meaningless.
- Enforce **chemical validity** (loading/sum constraints, feasible element sets); naive GAN draws
  can be unphysical. Prefer physics-constrained generation or conservative in-distribution
  interpolation (SMOGN) over free-form GANs for a first pass.

---

### Artefacts produced for this response
- `feedback_experiments.py` — reproducible script for all new analyses.
- `feedback_results.json` — every number quoted above.
- `fig_drst_threshold_sweep.png` (Q6), `fig_repeated_runs.png` (Q14/Q15),
  `fig_shap_stability.png` (Q13), `fig_kmm_drst_overlap.png` (Q8).
