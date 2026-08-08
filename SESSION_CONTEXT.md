# OCM Project — Session Context / Handoff

Use this file to resume the discussion in a new session. It captures the project state, the verified
numbers, the terminology/novelty findings, the file map, and the open (deferred) work.

**Repo:** `ParthN9i4/OCM-Dataset-JAIST` · **Branch:** `claude/add-catalyst-dataset-orerR`

---

## 1. The project in one paragraph

We predict OCM (Oxidative Coupling of Methane) C₂ yield from catalyst composition. We have **lab data**
(89,074 in-house experiments, year 2025, impregnation only, mean yield 5.25 %) and **literature data**
(3,852 published experiments, 1982–2019, diverse methods, mean yield 8.67 %). Goal: use the literature
data to improve the lab-data model **without hurting** in-lab accuracy. Two shifts make this hard:
**label shift** (+3.42 pp publication bias) and **covariate shift** (different chemistry, ~78.5 % of
literature is out-of-distribution vs our lab). Data file: `OCM_lab_data_and_literature_datal.csv`
(split by `year`: 2025 = lab, ≤2019 = literature). 67 features = Temperature_C + prep_enc + 65 element
loadings. Target column = `Y(C2), %`.

## 2. The five methods + VERIFIED numbers (asymmetric 5-fold CV, validation = lab data only; 3 decimals)

| # | Method | Model(s) | CV RMSE | vs baseline |
|---|---|---|---|---|
| 1 | Baseline (lab only) | LightGBM | **2.133** | — |
| 2 | Naive merge (all lit as labels) | LightGBM | **2.241** | worse (+5.1 %) |
| 3 | DRST filter (LogReg selector → LGBM) | best **2.127** (τ=0.85); at τ=0.30 = **2.181** | ≈ no gain |
| 4 | KMM (kernel weights → LGBM) | **2.261** | worse (+6.0 %) |
| 5 | **Two-stage PFT** (XGB expert → LGBM w/ prior feature) | **1.907** (seed 42) / **1.909 ± 0.002** (10-seed) | **−10.6 %** |

- **Single-stage filtering (DRST/KMM) does NOT beat baseline** — this is the honest, verified finding
  (the older "DRST 2.019 / KMM 2.035 improvements" were **stale/non-reproducible**; see `verify_drst_kmm.py`).
- **PFT rigour:** wins **10/10** seeds, paired t-test p = 3.9×10⁻¹⁵; true held-out 20 % lab test:
  baseline **2.097** → PFT **1.892** (R² 0.763).
- DRST at τ=0.30 keeps **782 / 3,852 = 20.3 %** of the literature.
- **OOD (leak-free)** — predicting non-impregnation literature: baseline **6.53** → PFT **6.77** (slightly
  worse; PFT is calibrated to the lab yield scale). The old "6.53 → 3.60 (−45 %)" was **data leakage**
  (the prior had seen the OOD rows' labels; reproduces 3.62 only with the leak). Numbers in
  `qn_tradeoff.json`.

## 3. PFT — naming & real novelty (research conclusion; report verbatim in discussions)

- **"Prior Feature Transfer" is our coinage** — not an existing term. BUT **"feature transfer" already
  means representation transfer** in the literature, so the name can mislead. Cleaner alternative name:
  **"Literature-Prior Stacking."**
- **The mechanism is NOT new.** Using a model's prediction as an input feature = **stacking / stacked
  generalisation** (Wolpert 1992); doing it across distributions = **stacked transfer learning** (a
  documented family). Because lab & literature share the **same task**, this setting is **domain
  adaptation**, not "transfer to a new domain."
- **What is genuinely ours (applied/methodological contribution — frame honestly):**
  (a) the specific recipe — covariate filter → literature expert with **rank-rescaled labels** → its
  prediction as the *only* channel into a lab model **trained on lab labels only**;
  (b) the **OCM literature→lab** application under *simultaneous* label + covariate shift;
  (c) the empirical result that this is the **only** method (of five) that beats baseline.
- Related work (for the novelty section): Wolpert 1992 (stacking); Ramakrishnan 2015 (Δ-ML — adds the
  source estimate as an *additive baseline* so its scale enters the output — soften any "trusts the
  label" wording to this); Huang 2006 / Sugiyama 2007 (KMM / importance weighting = our DRST/KMM
  baselines); Lipton 2018 (BBSE label-shift). Hypothesis Transfer Learning (HTL) uses source hypotheses
  as a *regularizer* — related but different (we use a feature, not a regularizer).

## 4. Using the ~80 % of literature DRST discards / OOD insight

- Naive merge fails on dissimilar literature because it puts their biased **labels in the loss**. PFT
  never does — it converts literature into a **feature** the lab model can down-weight per sample. So we
  can safely feed **all** literature into **Stage 1**.
- **RESULT — all-literature vs DRST-filtered Stage 1** (`pft_stage1_all_vs_filtered.py`, 10 seeds):
  baseline RMSE **2.121**; **PFT-filtered (782 rows) 1.909**; **PFT-all (3,852 rows) 1.917**. Both beat
  baseline 10/10 seeds. The filter gives a **tiny** edge (mean diff 0.008 RMSE, ~0.4 %, paired-t p≈0) —
  statistically real but practically negligible. **Takeaway:** the win comes from the two-stage
  architecture, not the filter; PFT is **robust to including the ~80 % dissimilar literature** in Stage 1
  (Stage 2 down-weights the prior as needed). MAE: filtered 1.386 / all 1.395; R²: 0.765 / 0.763.
- To make PFT help **OOD/extrapolation** (unfamiliar catalysts): train Stage 1 on all literature and
  **do not rank-rescale to the lab scale** (or add a second, un-rescaled prior). In-distribution accuracy
  and OOD extrapolation pull in opposite directions; the rescaling is the knob.

## 5. File map (which file is which)

- **`ocm_methodology.ipynb`** — THE clean, executed, external-sharing notebook (5 methods only; no OOD/QN/Q&A).
- **`ocm_worknote_taniike.md` / `.pdf` / `.docx`** — the report for Prof. Taniike (accessible, no code).
- `ocm_walkthrough.ipynb` — internal PRESENTATION notebook (Chapters + reviewer Q&A appendix, OOD, QN). Not for external sharing.
- `ocm_analysis.ipynb` — full experimental notebook.
- Experiment scripts (authoritative numbers): `feedback_experiments.py`+`feedback_results.json`,
  `qn_tradeoff.py`+`qn_tradeoff.json`, `tau1_sweep.py`+`tau1_sweep.json`, `verify_drst_kmm.py`,
  `pft_stage1_all_vs_filtered.py`.
- Builders: `build_methodology_nb.py`, `build_worknote_render.py` (md→pdf+docx), `build_pptx.py`, etc.
- Deck/reports: `ocm_presentation.tex/.pptx/.pdf`, `ocm_initial_report.html`, `ocm_presentation.html`,
  `ocm_project_summary.xlsx`.

## 6. Pending / deferred worknote-v2 feedback (NOT yet applied to `ocm_worknote_taniike.md`)

1. Reframe intro: "we tried several approaches; the last (PFT) is our contribution."
2. Add **MAE + R²** to the results table (deferred by user as future work).
3. Add a **description under each figure** (captions drafted in chat).
4. **Group DRST + KMM as one "selective merge"** method (hard filter / soft weights variants).
5. Brief **evaluation overview before the results**; fold EDA into the baseline/problem framing.
6. **Neutralise language** — remove "poison / contaminate / corrupt / attack"; don't disparage literature.
7. Fix "covariate shift is **addressed** by …"; reword the Method-2 conclusion; add a repeatability sentence.
8. Describe **PFT as 3 stages** (Stage 0 = DRST filter *or* keep-all; Stage 1 = expert; Stage 2 = final);
   state τ=0.30 and 20.3 % kept; include the all-literature vs filtered comparison.
9. **Consistent 3-decimal numbers** everywhere.
10. Rewrite the **novelty section** per §3 (stacked transfer / domain adaptation; clarify PFT is our label;
    soften Δ-ML wording).
11. **Variable-name uniformity:** use `lit_prior_prediction` everywhere (rename `prior_prediction` in the
    methodology notebook to match the SHAP figure; re-execute — numbers unchanged).
12. "Bring in domain knowledge" → one line in limitations/future work.
13. After edits, re-render `.pdf` + `.docx` via `build_worknote_render.py` (needs `pypandoc-binary`;
    Chromium at `/opt/pw-browsers/chromium-1194/chrome-linux/chrome` for PDF).

## 7. Catalyst-grouped validation round (Prof. Taniike's feedback) — CURRENT TRUE STATE

**Script/source of truth:** `taniike_validation.py` → `taniike_validation.json`. Every number below is
labeled with its protocol; row-level and grouped numbers must never be shown unlabeled side-by-side.

- **The lab data has only 917 unique catalysts** (~97 rows each: 5 temperatures × ~27 unrecorded
  condition settings ≈ Taniike's "135 conditions"). Row-level CV leaks catalyst identity across folds.
- **Headline (5 seeds, per-fold-mean RMSE; per-fold train-only DRST classifier + scaler — stricter
  than the published setup, hence 1.912 here vs published 1.909):**
  - Row-level: baseline **2.118 ± 0.004**, PFT-filtered **1.912 (−9.7%)** — reproduces the worknote.
  - **Catalyst-grouped: baseline 2.943 ± 0.031, PFT-filtered 2.995 (+1.8%), PFT-all-lit 2.982 (+1.3%)**
    — the published PFT gain does NOT survive; PFT ≈ marginally worse than baseline for unseen catalysts.
- **Mechanism (identified via ablation):** Stage 1 trains jointly on literature + lab training rows;
  under a row split those lab rows include the test catalysts (even same-temp replicates), so
  `lit_prior_prediction` partly memorized test-catalyst yields (identity leakage). Evidence: lit-only
  Stage 1 keeps only **−2.6%** row-level (2.062–2.067); under grouped CV joint variants (2.982–3.005)
  are WORSE than lit-only (2.938–2.939 ≈ baseline 2.943).
- **QN ablation (Taniike's follow-up email) — his hypothesis CONFIRMED:** QN ≈ raw ≈ pure ranks
  (differences 0.001–0.005 RMSE in both protocols). Stage-2 trees consume only the prior's ordering.
- **Catalyst-level metrics (grouped, unseen catalysts):** baseline Spearman(max-yield) **0.747**,
  enrichment@top-10% **4.0×**, precision@20 **0.47**. PFT ≈ same. Screening is viable with baseline
  alone; the literature prior currently adds ≈ nothing for unseen catalysts.
- **Family holdout (seed 42):** ρ 0.60–0.75 / enrichment 4–6× for La/Ti/Zr/Ce; **Ba (largest family,
  291 catalysts) is hard for all models** (ρ 0.43, enrichment ~2×; PFT worse). Ce with element-containing
  literature included actively hurt RMSE (3.72 vs 3.03) — trust-gating the prior is a real need.
- **Status vs external communication:** these results are NOT yet in the worknote/README/notebook and
  NOT yet sent to Taniike. The honest message: his validation caught a real flaw; his QN simplification
  is confirmed; baseline screening viability (ρ 0.75 / 4×) is the constructive result; next work =
  making the literature prior genuinely help unseen catalysts (catalyst-level targets, similarity-gated
  prior, family-aware tuning), then an uncertainty-ranked candidate list for prospective validation.

## 8. Phase 1 (protocol reset) — DONE

- **`ocm_eval.py`** is now the single shared implementation of the strict protocol (grouped folds,
  per-fold train-only scaler + DRST, Stage-1 label treatments, row + catalyst metrics) — verified to
  reproduce `taniike_validation.json` exactly. All new experiments must import it.
- **Grouped-CV baseline re-tuned** (`grouped_tuning.py` → `grouped_tuning.json`; 30-config random
  search, 2-seed search + 5-seed confirmation, folds shared between search and confirm → winner
  carries mild selection optimism):
  - Default params: **2.943 ± 0.031**, Spearman(max) 0.747, enrichment 4.02×.
  - **Tuned params: 2.896 ± 0.030 (−1.6%), Spearman(max) 0.766, enrichment 4.13×** — overrides:
    n_estimators=300, learning_rate=0.03, num_leaves=127, max_depth=-1, min_child_samples=20,
    subsample=0.6, colsample_bytree=0.8, reg_alpha=0, reg_lambda=1.
  - **The number Phase-3 literature-prior variants must beat: 2.896 (grouped protocol, tuned
    params in Stage 2 for both arms).**

## 9. Phase 2 (catalyst-level reformulation) — DONE

`catalyst_level.py` → `catalyst_level.json`. Grouped protocol, 5 seeds, identical catalyst-fold
assignments across formulations, tuned LGBM params:

| Formulation | Spearman(max) | Enrich@10% | Prec@20 |
|---|---|---|---|
| A row-level model → aggregate max | 0.766 ± 0.007 | 4.13× | 0.40 |
| **B direct per-catalyst max model (917 rows, no temp)** | 0.760 ± 0.002 | **4.28×** | **0.47** |
| B2 per (catalyst,temp) max → max over temps | 0.766 ± 0.007 | 4.28× | 0.43 |

**Gate decision: adopt B as the primary screening formulation** — statistically equivalent ranking
(all within noise), slightly better enrichment/precision, ~100× cheaper to train (917 vs 89k rows),
and it plugs directly into Phase-3b (catalyst-level literature prior) and Phase-5 candidate
screening (no need to predict 135 conditions per candidate). A remains the reference for row-level
RMSE reporting.

## 10. Phase 3 (earn back the literature prior) — DONE, honest NULL result

`phase3_lit_prior.py` → `phase3_lit_prior.json`. Formulation B, grouped protocol, identical folds,
5 seeds, tuned params. Pre-registered rule (Δspearman ≥ 0.01 AND Δenrich > 0 AND ≥4/5 seed wins):

| Variant | Spearman | Δ vs V0 | Enrich | Verdict |
|---|---|---|---|---|
| V0 control (formulation B) | 0.761 ± 0.002 | — | 4.28× | reproduces Phase 2 (prec@20 0.44 vs 0.47 = scaling-induced numerical jitter, ~½ catalyst) |
| V1 literature rank prior (lit-only expert) | 0.757 | −0.004 | 4.22× | null |
| V2 similarity/distance features | 0.740 | −0.021 | 3.85× | hurts |
| V3 gated prior (V1+V2) | 0.744 | −0.017 | 3.93× | hurts |
| V4 catalyst-level direct merge (control) | 0.758 | −0.003 | 4.33× | null (as expected) |

- **NO variant wins.** With the leakage channel closed, none of the literature-prior designs improves
  unseen-catalyst screening. Phase-3d fallback applies: contribution = validation methodology +
  baseline screening viability (Spearman 0.76 / 4.3× enrichment); framing to be decided with Taniike.
- Instructive: V3's trees USE the prior (16% of gain) and distance features (17%) heavily yet
  generalize worse — feature importance ≠ generalization value.
- One anecdote worth a targeted follow-up: Ce family holdout (lit-excluded) V0 0.671 → V3 0.713
  (single split, no seeds — anecdotal only). Ba holdout: V3 slightly worse (0.504 vs 0.522).
- Phases 4 (Ba diagnosis) and 5 (ranked candidate list on V0) remain valuable regardless of the null.

## 11. Phase 4 (family-holdout diagnosis) — DONE

`phase4_family_diagnosis.py` → `phase4_family_diagnosis.json`. Formulation B, family holdouts,
lit-excluded for prior variants, 5 model seeds.

- **Why Ba fails — label-coverage, mechanistically proven.** Ba catalysts average max yield 13.76%
  vs 8.95% for the rest; **78% of the global top-decile is Ba-containing**. Hold Ba out and training
  has almost no high performers left. Drop-the-Ba-column check: predictions bit-identical (max diff
  0.0000) — the model provably ignores Ba loading (constant 0 in training) and prices Ba catalysts
  as if Ba were absent → best Ba catalysts underpredicted by **−9.8 yield points** on average
  (predictions cap at 14.5 vs observed up to 21.2). Random-291 pseudo-family control scores
  0.752 ± 0.028 (≈ grouped-CV 0.761) → failure is chemical/label-coverage, NOT size.
- **Sr is not a learnable analogue**: only 67 catalysts, 4% of top decile, and 28% of Sr catalysts
  also contain Ba — no independent group-2 promotion signal to transfer from.
- **Ce anecdote KILLED by seeding**: V3 delta +0.007 ± noise (Phase-3 single-split +0.042 was luck).
  Pre-registered skepticism worked exactly as intended.
- **New hypothesis (one family only, do not over-claim):** Zr holdout, V1 lit-rank-prior:
  0.653 → 0.678 (+0.025, std 0.004, consistent across seeds); V3 +0.023. Zr also has the lowest
  element-specific literature coverage (57 comps). A targeted "prior for under-covered families"
  test could follow, but n=1 family.
- **Implication for Phase 5:** the candidate list must carry uncertainty/coverage flags — the model
  cannot price promoter chemistry absent from training (the Ba lesson), so extrapolative candidates
  need to be labeled as such.

## 12. P1 Target integrity audit — DONE, conclusions STAND

`phase5_target_audit.py` → `phase5_target_audit.json`. Triggered by
`Spearman(n_measurements, observed_max) = 0.293` and the 47 catalysts with <20 measurements
(mean observed max 3.28 vs 10.86; none in the global top decile).

- **Sampling bias is real and quantified** (chemistry held fixed — subsample well-measured
  catalysts): n=1 → observed max **−6.22** points, n=5 → −3.18, n=10 → −2.22, n=20 → −1.45,
  n=50 → −0.62. max-of-n genuinely understates poorly-measured catalysts.
- **Cause is mixed**: AUC(composition → low-count) = **0.772 ± 0.136** — the lab preferentially
  under-measured chemically distinguishable catalysts (selection) *and* the statistical artifact is
  present. AUC is noisy (only 47 positives); do not over-claim either cause.
- **Headline conclusions unaffected.** Excluding the 47 low-count catalysts moves Spearman by
  **0.002** (0.761 → 0.759). Phases 2–4 stand as reported.
- **The "better target" was an illusion — the control that matters.** Scored against their *own*
  targets, p90 (0.793) and top5mean (0.785) beat max (0.761). Scored against the **same ground truth
  (true max = Taniike's stated objective)**, differences collapse to noise: max **0.761 / 4.28× /
  0.44**, p90 **0.768 / 4.30× / 0.47**, top5mean **0.765 / 4.24× / 0.44**. The apparent gain was
  target *learnability*, not screening skill. **Decision: keep max.**
- **Enrichment is far less precise than previously quoted.** Bootstrap 95% CIs (n=917): Spearman
  **0.725–0.785**, enrichment **3.04–4.89×**. Quote enrichment as a range, never as "4.28×".
- **No meaningful tuning optimism**: true 20% catalyst holdout gives Spearman 0.791 (tuned) vs 0.783
  (default); holdout is not below the CV estimate.

## 13. How to run things

- Env: `pip install xgboost lightgbm shap matplotlib nbformat pypandoc-binary openpyxl python-pptx`.
- Experiments print authoritative numbers to stdout / write `*.json`.
- All work goes on branch `claude/add-catalyst-dataset-orerR`; commit + push there.
