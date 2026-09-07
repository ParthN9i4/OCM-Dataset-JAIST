# OCM Project — Session Handoff

**Purpose.** Everything a new session needs to continue this research accurately. Every number here
was re-read from a stored JSON during the handoff pass — none is quoted from memory. Where a number
could be confused with a similar one, both are given and labelled.

**Branch:** `claude/ocm-data-verification-qn9829`. **Tree:** clean.
The phase 8/9 work was stranded on unmerged `claude/add-catalyst-dataset-orerR` (102 commits — an
earlier version of this file said 108, which was wrong) and has now been merged here. A verification
pass re-derived every claim below from the data; see `ocm_verification_report.md`.

**Status in one line.** The original claim has been withdrawn and the corrected study is complete and
internally consistent. Nothing has been sent to Prof. Taniike since the withdrawal — the corrected
work note and the reply are **drafted, not sent**.

---

## 1. The problem

Predict C₂ yield for Oxidative Coupling of Methane (OCM) catalysts from catalyst composition.

One CSV, `OCM_lab_data_and_literature_datal.csv` — 92,926 rows × 69 columns, split by the `year`
column with nothing unaccounted:

| subset | filter | rows | preparation | mean Y(C2) |
|---|---|---|---|---|
| **lab** | `year == 2025` | 89,074 | 100% Impregnation | 5.245% |
| **literature** | `year <= 2019` | 3,852 | 20 distinct `Preparation` values (19 named + `n.a.`), 1982–2019 | 8.670% |

Columns: `Preparation`, `Temperature_C`, 65 element weight-% columns, `Y(C2), %`, `year`. **No
reaction-condition columns exist beyond temperature.** This absence is the single most consequential
fact in the project (see §5).

Two distribution shifts make it a domain-adaptation problem: **label shift** (+3.425 pp, publication
practice) and **covariate shift** (different chemistry emphasised).

**Original question:** can literature data improve the lab model?
**Reformulated question, after review:** can we rank *unseen* catalysts well enough to guide synthesis?
The second is the one now being answered.

---

## 2. How PFT arose — the five-method ladder

Each method answered the failure of the one before it. This history matters because it explains why
PFT looked compelling, and why its failure was not obvious.

| # | Method | Mechanism | Why it failed |
|---|---|---|---|
| 1 | Baseline | LightGBM on lab data only | — the number to beat |
| 2 | Direct merge | Pool literature + lab, train once | Label shift enters the training target |
| 3 | Selective merge (DRST) | Logistic domain classifier scores `P(lab \| features)`; keep τ ≥ 0.30 (20.3% of literature), then merge | Still puts literature *labels* in the objective |
| 4 | Selective merge (KMM) | Continuous importance weights instead of a hard filter | Same flaw as 3 |
| 5 | **PFT** | Stage 1: literature expert trained on quantile-normalised labels. Stage 2: that model's *prediction* becomes one extra input feature to a lab model trained on **lab labels only** | Beat baseline — **under the row-level split only** |

**What PFT is, honestly.** The mechanism is **stacked generalisation** (Wolpert, *Neural Networks* 5,
1992, 241–259) applied across distributions. "Prior Feature Transfer" is our coinage and is arguably
misleading, since "feature transfer" normally denotes *representation* transfer, which this is not.
A cleaner name would be *Literature-Prior Stacking*.

**Why it cannot help in general — the argument to keep.** The prior `p = f(x)` is a deterministic
function of features the Stage-2 model already holds, so `I(y; x, f(x)) = I(y; x)`. **It adds no
information.** It can only supply an inductive bias, or exploit source regions the target data does
not cover. Neither applies here at the scale that matters: ~79.7% of the literature is
out-of-distribution relative to the lab; quantile normalisation aligns marginal but not conditional
distributions; and literature yields are measured under each paper's own conditions while our target
is a maximum over a standardised battery — semantically different quantities.

**Scope statement (use verbatim):** this family of methods is a **local-coverage tool, not a
global-information tool**. §4 shows the one place local coverage actually pays.

Distinct related work: Δ-ML (Ramakrishnan et al., *JCTC* 11, 2015, 2087) adds the source estimate as
an additive baseline, inheriting its scale; importance weighting (Huang 2006; Sugiyama 2007) keeps
source labels in the objective — our DRST/KMM baselines are instances; BBSE label-shift correction
(Lipton 2018) reweights the label distribution; Hypothesis Transfer Learning uses source hypotheses as
a *regulariser*, not a feature.

---

## 3. Prof. Taniike's remarks, elaborated, with status

He raised six points. All six are addressed. The table states what each meant and where the evidence
lives.

| # | His remark | What it actually meant | Status |
|---|---|---|---|
| 1 | The dataset holds many measurements of the same catalyst under different reaction conditions | 89,074 rows are not 89,074 independent facts — they are 917 catalysts | **Addressed** |
| 2 | With a random split the same catalyst appears in train and test, so the test does not show prediction of genuinely unseen catalysts | Our headline measured recall, not discovery | **Addressed** — grouped CV is the default in `ocm_eval.py` and cannot be skipped by accident |
| 3 | A stronger test: hold out entire catalyst families (an element or support) | Tests genuinely new chemistry, not interpolation | **Addressed** — 28 families |
| 4 | Since each catalyst is run under the same 135 conditions, the objective is not the optimum condition but whether a catalyst reaches a high maximum somewhere | Screening, not condition prediction | **Addressed** — formulation B |
| 5 | Catalyst-level metrics (correlation of predicted vs observed max; enrichment of high performers) are more relevant than point-wise RMSE | RMSE is the wrong target for this decision | **Addressed**, and RMSE demoted with reasons |
| 6 | Does the quantile-normalisation step actually do anything? | Trees read feature *order*, not scale | **Addressed — he was right** |

### The evidence for points 1–2: the reversal

From `taniike_validation.json` → `A_grouped_vs_row`, 5 seeds, `rmse_foldmean`:

| protocol | baseline | PFT-filtered | PFT-all-lit | Spearman (baseline) |
|---|---|---|---|---|
| row-level *(superseded)* | 2.1184 | **1.9120 (−9.7%)** | 1.9194 | 0.9152 |
| **catalyst-grouped** | 2.9425 | **2.9955 (+1.8% worse)** | 2.9817 | 0.7472 |

The published claim and its inversion are the same models under two protocols. **The difference is
leakage, not modelling.**

**Mechanism, identified by ablation.** Stage 1 was trained on literature *together with the lab
training rows*. Under a row split those rows included the test catalysts, so the prior feature
partly carried each test catalyst's own measured yields. `ocm_eval.stage1_data()` now carries a
warning naming the `*_joint` kinds as the leakage channel.

**Point 3.** Family holdouts for Ba/La/Ti/Zr/Ce in `taniike_validation.json` → `B_family_holdout`,
extended to 28 families in `phase6_our_experiments.json`.

**Point 4.** Formulation B: one row per catalyst, composition only, temperature dropped, target =
observed maximum yield. 917 training examples instead of 89,074. Ranks as well as the row-level model
(`catalyst_level.json`).

**Point 5.** Primary metrics are now Spearman on max yield, enrichment@10%, precision@20. See §5B for
why RMSE was demoted rather than merely deprioritised.

**Point 6.** `taniike_validation.json` → `D_qn_ablation`: QN vs raw vs rank moves RMSE by 0.001–0.023,
inside run-to-run noise at 3 seeds. Quantile normalisation can be dropped with no measurable penalty.

---

## 4. What we found after the correction

### Literature integration is null in-domain

`phase3_lit_prior.json`. Pre-registered rule, fixed before running: *Δρ ≥ 0.01 **and** Δenrichment > 0
**and** ≥4/5 seed wins.*

| variant | Spearman | enrichment | Δ vs control | seed wins |
|---|---|---|---|---|
| **V0 composition-only (control)** | **0.7606** | 4.28× | — | — |
| V1 literature rank prior | 0.7569 | 4.22× | −0.0038 | 2/5 |
| V2 similarity features | 0.7396 | 3.85× | −0.0210 | 0/5 |
| V3 gated prior | 0.7439 | 3.93× | −0.0168 | 0/5 |
| V4 catalyst-level direct merge | 0.7577 | 4.33× | −0.0030 | 1/5 |

**None passed. All at or below the control.**

### Our own follow-up also died

`phase6_our_experiments.json` → `E1`: the "prior helps where literature coverage is thin" hypothesis,
tested across 28 families with a pre-registered threshold of |ρ| ≥ 0.5. Strongest observed |ρ| =
**0.276** (on `n_lit_compositions`). **NOT SUPPORTED.** Positive delta in only 14 of 28 families.

### A novel family is structurally unpriceable

`phase4_family_diagnosis.json`: with zero family members in training, the column is constant, trees
never split on it, and deleting the column gives **bit-identical** predictions (max |difference| =
0.0000000000). This is structural, not a modelling deficiency.

**Data budget — report both thresholds.** Ba learning curve
(`phase6_our_experiments.json` → `E2_learning_curve`):

| members seen | 0 | 5 | 10 | 25 | 50 | 100 | 200 | 204 (all) |
|---|---|---|---|---|---|---|---|---|
| Spearman | 0.509 | 0.543 | 0.565 | 0.616 | 0.651 | 0.679 | **0.687** | 0.683 |

The n=200 point is usually omitted, but it is the curve's maximum — the tail is **non-monotone**, so
"0.683 at n=204" is the end of the curve, not its peak. "204 (all)" means all Ba members available in
the training folds; the family holds **291** catalysts in total.

The stored `unlock_threshold_80pct = 10` is **80% of the final level** (0.8 × 0.683 = 0.547, first
cleared at n=10). **80% of the *gain*** is 0.509 + 0.8 × 0.174 = 0.648, first cleared near **n ≈ 50**.
Since Ba already scores 0.509 having seen *no* Ba at all, the level-based bar is nearly met by seeing
nothing — so quote ≈50, or quote both. The same caveat applies to the stored "0 needed" for La, Ti,
Zr and Ce.

### The one positive result

`phase7_prep_ood.json` — cross-preparation transfer, the setting where the lab has **no** coverage:

| configuration | Spearman | enrichment | precision@20 |
|---|---|---|---|
| C1 lab only → non-impregnation literature | 0.2385 | **0.42×** (worse than random) | 0.04 |
| **C2 plain merge** | **0.3883** | **1.34×** | **0.26** |
| C3 prior-feature construction | 0.3181 | 0.42× | 0.02 |

Reference: predicting *impregnation* literature scores 0.3980 / 1.97×, so changing preparation costs a
further ~0.16 Spearman.

**Two things to carry forward.** Literature data helps only where the lab has no coverage — exactly
what §2's scope statement predicts. And **plain merging beats the two-stage machinery** (0.3883 vs
0.3181): the value is the *data*, not PFT.

---

## 5. This session's two findings

### A. The within-cell rows are a designed condition grid, not repeats

`phase8_target_robustness.json` → `condition_grid_evidence`, recomputed live by the notebook:

| quantity | value |
|---|---|
| (catalyst, temperature) cells | 4,399 |
| mean rows per cell | 20.25 |
| **modal cell size** | **27** |
| **maximum cell size** | **54** (= 2 × 27) — but on **exactly 1 cell**, so weak evidence |
| cells larger than 54 | **0** |
| catalysts with exactly 135 rows | 15 |
| **…of which split as exactly (27,27,27,27,27)** | **15 of 15** |

5 × 27 = **135**, matching the number of conditions Prof. Taniike states each catalyst is run under.

The rival reading — that 27 was a top-27-by-yield export cut-off — is **rejected**, but by a simpler
argument than the one previously given: **104 cells hold more than 27 rows**, which a top-27 cut-off
cannot emit. An order-statistic test (bottom-of-distribution gap vs interior gap) agrees in direction
under all four definitions of interior spacing tried. Both now live in
`phase11_condition_grid_forensics.json` — **quote them from there**.

*Provenance correction.* The figures previously quoted here, **2.03× vs 0.90×**, were computed by no
script and stored in no JSON, and were not reproduced by any of the four definitions. The direction
they assert holds; **the numbers must not be requoted.**

*Caveat that was missing.* Every cell is stored **sorted descending by yield** — 0 violations across
84,675 within-cell adjacent comparisons in all 4,399 cells. Pre-sorting by yield is exactly what a
"take the top N" export produces, so the truncation reading is mechanically *more* plausible than the
order statistic alone implies. The counting argument, not the order statistic, is what settles it.

**Corrections this forced, already applied:**

| | was | is |
|---|---|---|
| within-cell share of yield variance | 18.2% | **19.9%** |
| best attainable row-level RMSE | 1.680 | **1.757** |

The old numbers came from `within = g.var().mean()` — an *unweighted* mean of per-cell variances,
which weights a 2-row cell like a 27-row cell and drops 83 singleton cells as `NaN`. The corrected
form uses the pooled within-cell sum of squares. Consequently "Version 1 reported 1.907, only 0.23
above the floor" became **0.15 above**, which strengthens the original suspicion.

**Vocabulary now banned for this data:** *repeats, replicates, denoising, noise floor, irreducible,
identical inputs.* The variance is unreachable **with these features**, not irreducible in principle,
and the floor presumes knowing each catalyst's own cell means — so it is not attainable for an unseen
catalyst and is **not** a headroom target.

**Target choice re-tested and settled.** Spearman is flat at 0.766–0.768 for every within-cell
quantile from 0.50 to 0.95 and falls to 0.761 only at q=1.00; enrichment shows no trend (4.15–4.43×);
the q=0.50 label does not clear zero under a catalyst-level bootstrap; and **seed-averaging alone,
with no target change, gains +0.0065 — the size of the whole effect.** **Decision: keep the observed
maximum.** Report the sweep as a robustness result. Do not re-litigate.

### B. The headline was coverage-inflated

Grid coverage is coupled to performance — cells run further contain better yields — so a score over
all 917 catalysts is partly a record of which experiments were finished.
`phase9_equal_effort_eval.json`:

| | n | Spearman | enrichment |
|---|---|---|---|
| real model, all catalysts | 917 | 0.7672 | 4.35× |
| **real model, equal-effort set** | **771** | **0.7235** | **3.77×** |
| effort-only control, all | 917 | **0.3996** | **0.87×** |
| effort-only control, equal-effort | 771 | 0.2244 | 1.04× |

*Equal-effort set = catalysts with ≥20 rows in at least one temperature cell.* There the confound is
gone **by measurement**: Spearman(measurement count, observed max) falls **+0.2932 → +0.0029**.

**The control that makes this a finding.** 300 *random* 771-catalyst subsets of the *same* predictions
give 0.7671 [0.7556, 0.7800]. The equal-effort value 0.7235 lies **below** that interval, so the drop
is a real coverage effect, not an artifact of scoring fewer catalysts.

**The effort-only negative control** is the sharpest diagnostic in the project: a model trained only
to predict *how many measurements a catalyst received* — it never sees a yield — reaches Spearman
**0.3996**, but enrichment **0.87×**, no better than random. **Rank correlation is partly purchasable
from experimental effort; enrichment is not.** That is the concrete justification for enrichment as
the primary metric.

**Campaign limit.** Inside the model's own top-ranked region — the only regime a synthesis campaign
occupies:

| restricted to | internal Spearman | mean observed max |
|---|---|---|
| top 20 | **−0.066** | 17.33% |
| top 50 | +0.053 | 17.34% |
| top 150 | **+0.179** | 16.71% |
| top 300 | +0.492 | 15.18% |

**The model selects a good set but cannot order within it** (library mean ≈ 10.5%). A shortlist is a
set to test, not a league table — and a 17-catalyst campaign drawn entirely from that region cannot
confirm or refute the model.

---

## 6. File map and protocol

### Experiment scripts (13) — each writes the JSON that every document quotes

| script | JSON | what it settles |
|---|---|---|
| `taniike_validation.py` | `taniike_validation.json` | the reversal, family holdouts, QN ablation |
| `grouped_tuning.py` | `grouped_tuning.json` | tuned LGBM params under grouped CV *(shared fold family — mild selection optimism, stated in its docstring)* |
| `catalyst_level.py` | `catalyst_level.json` | formulation A vs B vs B2 |
| `phase3_lit_prior.py` | `phase3_lit_prior.json` | four literature designs — null |
| `phase4_family_diagnosis.py` | `phase4_family_diagnosis.json` | Ba drop-column proof |
| `phase5_target_audit.py` | `phase5_target_audit.json` | max-of-n confound audit |
| `phase6_our_experiments.py` | `phase6_our_experiments.json` | coverage hypothesis (null) + family learning curves |
| `phase6_candidates.py` | `phase6_candidates.json` + 2 CSVs | 26,414 candidates, coverage-gated |
| `phase7_prep_ood.py` | `phase7_prep_ood.json` | cross-preparation — the one positive |
| `phase8_target_robustness.py` | `phase8_target_robustness.json` | condition grid + target robustness |
| `phase9_equal_effort_eval.py` | `phase9_equal_effort_eval.json` | coverage correction + effort control |
| `phase10_ground_truth_invariance.py` | `phase10_ground_truth_invariance.json` | **does the conditions-vs-time-on-stream answer change our recommendation?** |
| `phase11_condition_grid_forensics.py` | `phase11_condition_grid_forensics.json` | order statistic, counting disproof, measurement-budget replay, within-cell ordering |
| `feedback_experiments.py` | `feedback_results.json` | pre-review presentation feedback |
| `qn_tradeoff.py` | `qn_tradeoff.json` | **SUPERSEDED — do not quote.** Row-level protocol; retained only as a record |

Plus `ocm_eval.py` (shared module) and 16 builder/patcher scripts. **No orphan JSONs; every declared
output exists.**

### Supersessions — check before quoting anything

- `phase8_target_robustness.py` **supersedes** `phase8_denoised_target.py` (renamed; both old script
  and its JSON removed, git history retains them). The old version framed the question as "denoising
  repeats" and concluded two summarised targets beat the observed max — **both framing and conclusion
  were wrong**.
- `phase7_prep_ood.json` **supersedes** `qn_tradeoff.json` (never requote 6.531 / 6.772 / 6.047 / 3.615).
- `fig_protocol_comparison.png` retires `fig_repeated_runs.png`.

### Protocol rules (verbatim from `ocm_eval.py`)

1. "Catalyst-grouped CV is the DEFAULT protocol. Row-level CV exists only for comparison with
   historical numbers and must always be labeled as such."
2. "Anything fit on data (scaler, DRST classifier, Stage 1, Stage 2) sees training-fold data only."
3. "Catalyst-level metrics … are primary for the screening objective; row RMSE is secondary and must
   be labeled as such."

**Catalyst identity** = `Preparation` + the full 65-element loading vector, pipe-joined and factorised
→ 917 groups. **Temperature is deliberately excluded — it is a condition, not identity** (it remains a
feature in row-level models).

**Caveat worth knowing:** six pre-refactor scripts do **not** import `ocm_eval` and carry their own
pipeline copies — `taniike_validation.py` (the file it was extracted *from*),
`feedback_experiments.py`, `qn_tradeoff.py`, `tau1_sweep.py`, `verify_drst_kmm.py`,
`pft_stage1_all_vs_filtered.py`. Changes to the shared protocol will not propagate to them.

### Documents

- `ocm_worknote_taniike.md` (+ docx/pdf/html) — **the document for Prof. Taniike. Drafted, not sent.**
- `ocm_progress_report.md` (+ docx/pdf/html) — for Dr. M S Srinath. Simple sentences, active voice.
  Rendered by `build_progress_report_render.py` (pandoc → HTML → Chromium PDF, + pandoc DOCX). Until
  September 2026 **nothing in the repo built it**, which is how a rendered document drifts unnoticed.
- `ocm_results_walkthrough.ipynb` — **the presentable notebook.** 16 code cells, all executed, zero
  errors, every number computed live. Built by `build_presentation_nb.py`. Bundle with companions:
  `ocm_eval.py`, `grouped_tuning.json`, `phase5_target_audit.json`, and the CSV.
- `ocm_analysis.ipynb`, `ocm_methodology.ipynb`, `ocm_walkthrough.ipynb` — historical, row-level.
- `feedback.md` — working-style corrections, loaded via `CLAUDE.md`. **Read it.**
- `ocm_verification_report.md` — findings of the data-verification pass, with confidence and severity.

### Presentation set — registered here so it cannot drift again

These files were absent from this map, and that is precisely how they came to assert a **withdrawn**
claim in four formats long after it was retracted. All have now been rewritten to the corrected
results.

| file | built by | narrated by |
|---|---|---|
| `ocm_presentation.pptx` + `.pdf` | `build_pptx.py` (PDF via LibreOffice — **requires `libreoffice-impress`**; `libreoffice-core` alone silently fails) | `ocm_speaking_notes.md` Part 1 (16 slides) |
| `ocm_presentation.html` | `build_presentation.py` (8 slides) | `presentation_script.md` |
| `ocm_walkthrough.ipynb` ch. 9–10 | historical notebook | `ocm_ch9_ch10_script.md` |

- **`ocm_presentation.tex` — SUPERSEDED, do not edit or quote.** Produced by neither builder, no LaTeX
  toolchain in-container, and it still carries the withdrawn `1.907` / `−10.6%` claim. The PDF now
  comes from `build_pptx.py` via LibreOffice.
- `ocm_speaking_notes.md` **Part 2** and `ocm_ch9_ch10_script.md` narrate the *historical* notebook and
  keep its row-level numbers on purpose; both carry a status banner saying so.
- **Caveat:** `build_corrected_figures.py` still regenerates `fig_repeated_runs.png`, which
  `fig_protocol_comparison.png` retired. No builder references it any more.

### Running things

```bash
pip install numpy pandas scikit-learn lightgbm xgboost shap scipy matplotlib seaborn pypandoc-binary
python3 phase9_equal_effort_eval.py     # ~30 s
python3 phase8_target_robustness.py     # ~100 s
```
PDF rendering: no LaTeX in-container. Use pandoc → HTML (`--embed-resources`) → headless Chromium at
`/opt/pw-browsers/chromium-*/chrome-linux/chrome --headless --print-to-pdf`. Verify PDF text with
`pypdf`, never `strings`.

---

## 7. How to continue — ranked by value per unit effort

1. **Ask JAIST for the condition metadata.** Highest value, lowest cost, one email. It converts 19.9%
   of currently unreachable variance into modellable signal, makes row-level RMSE a well-posed target
   again, turns 917 training examples back into 89,074, and would let us predict *which condition* to
   run rather than only a catalyst ceiling. The work note already asks.
2. **Settle distinct-conditions vs time-on-stream — by asking, because the file cannot.** We now know
   *why* two prior analyses found nothing: **every one of the 4,399 cells is stored sorted descending
   by yield** (0 violations in 84,675 within-cell adjacent comparisons;
   `phase11_condition_grid_forensics.json`). Row order encodes **rank, not acquisition sequence**, and
   there is no time, run-index or condition column. Every ordering-based test — decay profile,
   periodicity, autocorrelation — is therefore impossible **in principle**, not merely inconclusive.
   A third attempt would fail too. Only JAIST can answer it.

   **The risk is now bounded, though the question is still open** (`phase10_ground_truth_invariance.json`).
   At the data level the two readings genuinely disagree where it hurts: ranking catalysts by their
   *floor* (q0.05) instead of their *ceiling* (the observed max) shares only **7 of the top 20**
   (Spearman 0.859 overall). But a model trained on the ceiling still scores **0.7576** against the
   floor ground truth, versus **0.7745** for the best floor-suited training target — a regret of
   **−0.0169 Spearman**, smaller than the +0.0065 that seed-averaging alone buys — and its
   enrichment@10% never falls below **4.02×** against any ground truth in q ∈ [0.05, 1.00].
   **The answer changes the labels but not the decision.** Still ask; but the shortlist does not
   hinge on the reply.
3. **Ask why coverage is incomplete** — 186 catalyst-temperature cells absent, sizes 1–54, only 811 of
   917 catalysts have all five temperatures. Decides whether the bias is correctable or itself
   informative.
4. **Re-scope the campaign before reactor time is spent** (gated on #2). Retrospective replay on the
   lab's own archive (`phase11_condition_grid_forensics.json`, 200 draws): 20 runs per catalyst
   (4 temperatures × 5 rows) reproduces the full-135 ranking at ρ **0.9445** [5–95 pct 0.9396–0.9495],
   buying ~72 catalysts plus a randomised control arm for the same reactor budget as 17 exhaustive
   ones. Under the time-on-stream reading the saving is analytical rather than thermal and the
   arithmetic changes.

   *Provenance correction.* This was previously quoted as ρ **0.955**, with no script or JSON behind
   it. That value is not what a 20-run budget gives; **0.9563** is what **25** runs (5 temperatures ×
   5 rows) gives, so the figure appears to have been carried over from a different configuration than
   the one printed beside it. Quote 0.9445 for 20 runs.
5. **Send the corrected work note.** It is complete and verified.

### Do not re-litigate — already piloted, all null

- In-domain literature integration (4 designs + 28-family follow-up).
- Target relabelling for coverage: rarefaction and deficit-extrapolation correlate **0.977** and
  **0.9989** with the plain observed max; every induced shortlist change sits inside seed noise.
- Learning-to-rank and top-decile classification — decisively worse.
- Closed-loop active learning — buys ≈0.6 catalysts over a single ranked batch; every acquisition
  function ties with plain greedy.
- Coverage-weighted training — significant under seed resampling, fails a catalyst-clustered bootstrap.

---

## 8. Number hygiene — the rules that were learned the hard way

- **0.761** (per-seed mean, all 917), **0.767** (seed-averaged, all 917), **0.724** (equal-effort, 771)
  are **three different quantities**. Never print unlabelled.
- Row-level and grouped numbers never appear side by side unlabelled.
- Enrichment is quoted as a range (**3.04–4.89×**), never as a bare "4.28×".
- Precision@20 is 0.44 with CI **0.15–0.65** — too wide to decide anything on its own.
- One script is the single source of truth for any reused number. Regenerate documents; never
  hand-edit a number into prose.
- Recompute derived percentages from their stated base numbers before reuse.
- Verify PDF/DOCX content by structured extraction (`pypdf`, `zipfile`+XML), never `strings`.
- When an internal copy and an externally-sent copy of a document both exist, **find the sent one**.
- Local absence of a file is not evidence it never existed — check the remote before concluding.
- **A number with no script behind it is not a result.** Three load-bearing figures were being quoted
  from prose alone (2.03×/0.90×, ρ 0.955); two of the three did not survive recomputation. Before
  quoting any number, confirm a committed script writes it to a JSON.
- **Check that a figure belongs to the configuration printed beside it.** ρ 0.955 sat next to a 20-run
  budget while matching the 25-run one.
- Do not compare a file against itself when verifying a reproduction — check the paths resolve to two
  different files before believing a "byte-identical" result.

### Honest posture

Every self-reported number moved **down** this session: Spearman 0.761 → 0.724, enrichment 4.28× →
3.77×, Ba's data budget 10 → ≈50, and the campaign demoted from validation to exploration. That is the
right position to write to a collaborator from, and it should be presented as *we found it before a
reviewer did*.
