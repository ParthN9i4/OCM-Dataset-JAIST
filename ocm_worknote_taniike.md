# Work Note (v2) — Incorporating Published Literature Data into the Lab OCM Yield Model

**To:** Prof. Taniike
**Topic:** A domain-adaptation study for C₂-yield prediction in Oxidative Coupling of Methane (OCM)
**Supersedes:** version 1 of this note, which describes the problem setup, the two datasets and the
five methods in full; that material is not repeated here.
**Companion code:** `ocm_eval.py`, `taniike_validation.py`, `phase3_lit_prior.py`,
`phase4_family_diagnosis.py`, `phase5_target_audit.py`, `phase6_our_experiments.py`,
`phase6_candidates.py`, `phase7_prep_ood.py`, `phase8_target_robustness.py`,
`phase9_equal_effort_eval.py`, `phase10_condition_subsampling.py`

---

## What changed since version 1

Version 1 reported that a two-stage prior-feature method ("PFT") improved C₂-yield prediction by
10.6 % over a lab-only baseline. **Following the stricter validation you proposed, that improvement
does not survive, and we withdraw the claim.**

| Claim in v1 | Status in v2 |
|---|---|
| PFT improves CV RMSE by 10.6 % (1.907 vs baseline 2.133) | **Withdrawn.** The gain was catalyst-identity leakage; under catalyst-grouped CV, PFT is 1.8 % *worse* than baseline |
| Literature data measurably helps the lab model | **Not demonstrated** in-domain. Four designs, all ≤ the composition-only control. It does help across preparation methods (§8) |
| Quantile normalisation is a necessary component | **Not supported.** Label treatment moves RMSE by 0.001–0.023, inside run-to-run noise at 3 seeds |
| — | **New:** the composition-only model screens *unseen* catalysts usefully (ρ = 0.761 over all 917 catalysts, 0.724 on the 771 with comparable measurement effort) |
| — | **New:** the Ba-family failure is mechanistically explained, and yields a data budget for new chemistry |
| — | **New:** one measurement per catalyst–temperature ranks nearly as well as all ~27 (§5) |

The cause was specific: our Stage-1 expert was trained on literature data *together with the lab
training rows*, so under a random row split it had seen the very catalysts it was later asked to help
predict.

![**Figure 1 — The central finding.** Identical models under two evaluation protocols. When every measurement of a catalyst is confined to one fold, the reported improvement inverts.](fig_protocol_comparison.png)

## 1. Evaluation methodology

**Catalyst-grouped cross-validation is now our default.** All measurements of a catalyst are assigned
to exactly one fold, so every number answers: *how well do we predict a catalyst nobody has made yet?*
Anything fitted on data — scaler, domain classifier, both model stages — sees training-fold data only.

**Why the previous protocol was inadequate.** The 89,074 measurements comprise only **917 distinct
catalysts** at 5 temperatures, giving 4,399 (catalyst, temperature) cells of 20.2 rows on average.
Your description of these rows as measurements *under different reaction conditions* is borne out by
the row counts, and we should have taken it more literally. Cell sizes have a hard ceiling at exactly
**27**, a second at exactly **54 = 2 × 27**, and nothing above; **15 catalysts hold exactly 135 rows,
and all 15 decompose as exactly (27, 27, 27, 27, 27)** — that is **5 temperatures × 27 condition
settings = 135**, matching the figure you gave us. We tested the competing reading that 27 was an
export cut-off and rejected it: the spacing between the two lowest values in a 27-row cell is 2.03×
the interior spacing, whereas truncating a larger cell to 27 rows gives 0.90× — a complete sample,
not a truncated one.

These rows are therefore **not replicates**. Because the 27 settings are absent from the feature
table, **19.9 % of total yield variance lies within cells** and cannot be reached from composition and
temperature, flooring row-level RMSE at **1.757**. That is a property of the missing columns, not a
physical limit — recovering the condition data would make most of it learnable.

Version 1 reported 1.907, only 0.15 above that floor. In hindsight that should itself have prompted
suspicion: a model cannot approach a floor that presumes knowledge of each catalyst's own cell means
unless it has effectively memorised those catalysts. Point-wise RMSE is close to uninformative here,
and it is precisely the metric a catalyst-identity leak flatters most. **Primary metrics are therefore
catalyst-level, as you proposed:** Spearman correlation on maximum yield, and enrichment of true high
performers among top-ranked predictions.

## 2. What the stricter validation showed

**The improvement was leakage.** Under catalyst-grouped CV: baseline **2.943**, PFT **2.995**
(+1.8 %). The same two models under the row-level split give **2.118** and **1.912** (−9.7 %) — the
difference between the two lines is leakage, not a change of model.
Three readings of one 3-seed ablation grid point to the mechanism (not independent experiments):
Stage 1 on literature *alone* reduces the row-level gain from −9.7 % to **−2.4 %** (QN prior) or
**−2.7 %** (rank prior); under grouped CV the joint variant (**2.982**) is worse than literature-only
(**2.938**); and literature-only is indistinguishable from baseline (**2.928**). *(The headline 2.995
is a 5-seed mean; 2.982 is the same configuration over the first 3 seeds.)*

**Your quantile-normalisation hypothesis is supported**, though not as strongly as we first stated.
Gaps are 0.006 row-level, and catalyst-grouped 0.023 (QN vs raw) and 0.001 (QN vs rank) — the largest
sits under our primary protocol and is close to the run-to-run spread there (per-configuration SD
0.022 and 0.035 over 3 seeds). The honest statement is that **no label treatment is distinguishable
from another at this seed count**. The normalisation step can be dropped without measurable penalty.

**With the leak closed we retested literature integration properly** — a literature-only rank prior,
similarity features, a gated prior, and a catalyst-level direct merge. Criteria were fixed before
running. **None improved on composition alone.**

![**Figure 2 — No literature variant beats composition alone.** Catalyst-grouped protocol; dashed line is the composition-only control.](fig_grouped_results.png)

We also tested a hypothesis of our own — that the prior might help where our coverage is thin,
suggested by one family (Zr). Across **all 28 element families with ≥50 catalysts** the mean effect was
**−0.0025**, 14/28 positive, strongest correlation with any coverage measure |ρ| = 0.276 against a
pre-registered threshold of 0.5. The Zr result was selection from noise, and we discarded it.

## 3. Screening unseen catalysts

Rebuilt at catalyst level — composition → maximum yield, 917 training examples, no temperature — the
model matches the full 89,074-row model on ranking while training on ~100× fewer rows.

| Metric | All 917 catalysts | 95 % CI | Equal-effort set (771) |
|---|---|---|---|
| Spearman ρ (predicted vs. observed max yield) | 0.761 | 0.725 – 0.785 | **0.724** |
| Enrichment of true top-decile among top-decile predicted | 4.28× | **3.04 – 4.89×** | **3.77×** |
| Precision@20 | 0.44 | **0.15 – 0.65** | 0.35 |

With 92 catalysts in the top decile of 917, these are less precise than a point estimate suggests.

**Why the second column exists, and why we consider it the honest one.** Grid coverage in your data is
coupled to performance — cells run further contain better yields — so a score over all 917 catalysts
is partly a record of which experiments were completed. The *equal-effort set* is the 771 catalysts
with ≥20 measurements in at least one cell; there the coupling is gone by measurement, with
Spearman(measurement count, observed maximum) falling from **+0.293** to **+0.003**. Moving to it costs
0.037 Spearman and 0.51× enrichment.

That drop is not an artifact of scoring fewer catalysts: 300 **random** 771-catalyst subsets of the
*same* predictions give 0.767 with a 95 % range of 0.756 – 0.780, and 0.724 lies below it.

**A negative control worth more than either number.** We refitted the identical model with the *number
of measurements* as its target — it never sees a yield. That ranking reaches **Spearman 0.400** against
observed maximum yield, but enrichment **0.87×**, no better than chance. Rank correlation is partly
purchasable from experimental effort; enrichment is not. That is why we treat enrichment as primary.

**One consequence for campaign design.** Inside the model's own top-ranked region — the only regime a
campaign occupies — internal ordering carries little information: ρ = **0.179** within the top 150 and
**−0.066** within the top 20. The model *selects* well (its top 20 average 17.3 % observed maximum
against ~10.5 % library-wide) but does not *order* within its selection. A shortlist is a set to test,
not a league table.

![**Figure 3 — What drives achievable maximum yield.** Composition-only model. *(Regenerated: the corresponding v1 figure ranked the literature prior first, because that model was the leaked pipeline.)*](fig_shap_bar.png)

## 4. How much of the condition grid does a catalyst need?

Keep only k of the ~27 measurements per catalyst–temperature cell (drawn at random, 5 independent
draws per k, always scored against the true maximum from the full data):

| Measurements kept | Share of full grid | Spearman | Enrichment |
|---|---|---|---|
| 1 per cell (~5 runs/catalyst) | 5 % | 0.759 ± 0.002 | 3.99× |
| 2 per cell (~10 runs/catalyst) | 10 % | 0.760 ± 0.002 | 4.11× |
| 3 per cell (~15 runs/catalyst) | 15 % | 0.765 ± 0.002 | 4.24× |
| all ~27 per cell | 100 % | 0.761 | 4.28× |

One measurement per cell ranks almost as well as all 27, stable across five independent draws. Labels
are biased low (−2.8 yield points at k = 1, below 0.4 by k = 13), but the bias is roughly uniform
across catalysts, so ranking survives even though absolute yield estimates would not.

Temperature is different. 700 °C alone gives ρ = 0.346, enrichment 1.07× — no better than chance —
because 70 % of catalysts reach their maximum at 800 °C or above. **Redundancy is within a
temperature, not across temperatures.** Mechanism: between-catalyst spread in true maximum yield
(SD 5.32) is 3.8× the typical spread within one cell (SD 1.41).

**This depends on the 27 slots being individually selectable.** If they are successive samples from one
continuous run, "keep 1 of 27" means stopping a run early, not choosing a condition, and the result
would not transfer to a prospective design. See question 2 in §8.

## 5. Why Ba fails, and how much data a new family needs

Family holdouts behave reasonably for La, Ti, Zr and Ce (ρ 0.62–0.68) but poorly for Ba (0.526). Ba
catalysts average **13.76 %** maximum yield against **8.95 %** for the rest, and **78 % of the lab's
top decile contains Ba** — removing them removes the high-yield regime.

The mechanism is provable. With no Ba in training the Ba column is constant, so no tree splits on it,
and retraining with that column **deleted entirely** gives **bit-identical predictions**. The model
prices Ba catalysts as though Ba were absent, underpredicting the best by **9.8 yield points**. Ten
random pseudo-families of equal size score 0.752, so this is label coverage, not sample size.

| Family | ρ, none seen | ρ, fully seen | % of ceiling at zero |
|---|---|---|---|
| **Ba** | 0.509 | 0.683 | **74.4 %** |
| La | 0.678 | 0.731 | 92.7 % |
| Ti | 0.618 | 0.664 | 93.0 % |
| Zr | 0.646 | 0.724 | 89.2 % |
| Ce | 0.643 | 0.722 | 89.1 % |

![**Figure 4 — A data budget for new chemistry.** Left: performance against family members already measured. Right: fraction of achievable performance reached having seen none.](fig_learning_curve.png)

Ba is the most *consequential* family to lose, not the hardest to predict — five of the 28 score below
it (Pd 0.217, Cu 0.286, Al 0.329, Ni 0.329, Co 0.493). What sets Ba apart is weight: it holds 78 % of
the top decile and gains far more than any other family from seeing its own members (+0.175, against
+0.047 to +0.079 elsewhere).

**On the data budget, the threshold depends on the definition, and the two readings differ by five
times.** The Ba curve runs 0.509 (none seen) → 0.565 (10) → 0.616 (25) → 0.651 (50) → 0.683 (all 204).
Taking 80 % of the **final level** (0.547) gives **10 members**; taking 80 % of the **gain** (0.648)
gives roughly **50**. We now think the second is more meaningful: Ba already reaches 0.509 having seen
no Ba at all, so the level-based bar is nearly cleared by seeing nothing. The same caution applies to
the "0 needed" for La, Ti, Zr and Ce. Reaching 95 % takes about 50 for Ba and 25–50 for the others —
order-of-magnitude guidance only, since for Ti and Zr the per-point seed spread (0.03–0.12) is
comparable to the whole gain.

## 6. A candidate list for prospective validation

We enumerated **26,414 unseen candidates** in your own design grammar — impregnation, one support at
~90 % with 2–3 promoters at ~3.33 %, drawn from supports and promoters already in use — and scored
them with a 10-seed ensemble. No literature prior is used. Every candidate carries a coverage flag,
verified rather than assumed: for an element absent from our data (Ag), predictions with and without
its column differ by exactly zero, confirming such candidates are unpriceable and must be flagged.

The highest-ranked candidate is **Ba(90) + Mo(3.33) + Zn(3.33) + Fe(3.33)**, predicted 18.79 %. We
deliberately attach no error bar: the ± 0.09 our ensemble reports is only seed spread, while the
model's catalyst-level error on held-out catalysts is about **2.7 yield points** (MAE). The number
ranks candidates; it does not forecast a yield. Absolute predictions also compress at the extreme —
our maximum is 18.79 % against observed training yields reaching 21.50 %.

The top 20 are chemically monotonous (all Ba, mostly Mo), and a diversity constraint did not change
this, because the model's Ba preference is genuine. Best candidate per support:

| Support | Ba | Ti | La | Ca | Mg | Si | Al | Zr | Ce |
|---|---|---|---|---|---|---|---|---|---|
| Best predicted max yield (%) | 18.79 | 16.49 | 15.82 | 15.54 | 14.66 | 13.83 | 13.63 | 13.02 | 12.93 |

**Our suggested campaign (17 catalysts, `campaign_shortlist.csv`)** splits the budget: **Tier A**, 12
catalysts at the model's optimum (18.33–18.79 %), where roughly **2–8 of the 12** would be genuine
top-decile performers on the retrospective precision@20 CI of 0.15–0.65; and **Tier B**, 5 catalysts
one per alternative support (Ti 16.49, La 15.82, Ca 15.54, Mg 14.66, Si 13.83). Tier B costs predicted
yield and we do not expect it to win — it is an information purchase, since 78 % of the existing top
decile already contains Ba, so the model's preference may partly reflect coverage rather than
chemistry. We defer to your judgement on synthesis feasibility.

**A limit we should state plainly before you spend reactor time.** Seventeen catalysts from the
top-ranked region with no control arm cannot confirm or refute the model — per §3, internal ordering
there is close to uninformative. The list is a reasonable set to *try*; it is not a test.

If a test is wanted, a different allocation of the same reactor budget: measuring 5 conditions at each
of 750, 800, 850 and 900 °C — 20 runs rather than 135 — reproduces the ranking of your 811
fully-measured catalysts at ρ = **0.955** (low by 1.31 yield points, a bias that can be pre-declared;
§4 has the stability-checked version). That buys roughly **72 catalysts screened instead of 17**
measured exhaustively, the best few then confirmed at full coverage, with part of the batch drawn at
random as a control arm. The cost is more syntheses for the same reactor hours.

## 7. Relation to prior work

The mechanism of PFT — a model's prediction used as an input feature — is **stacked generalisation**
(Wolpert, *Neural Networks* 5, 1992, 241–259) applied across distributions. "Prior Feature Transfer"
was our internal shorthand, not a standard term.

**Our corrected understanding of when it can help.** Because the prior `p = f(x)` is a deterministic
function of features the final model already has, `I(y; x, f(x)) = I(y; x)` — it cannot add
information. It can only supply an inductive bias, or exploit source data covering regions the target
does not. Neither applies here at scale: ~79.7 % of the literature is out-of-distribution relative to
the lab; quantile normalisation aligns marginal but not conditional distributions; and literature
yields come from each paper's own conditions, whereas our target is a maximum over a standardised
battery. **The honest scope statement is that this family of methods is a local-coverage tool, not a
global-information tool** — which is exactly what §8 then confirms.

Related approaches remain distinct: Δ-machine-learning (Ramakrishnan et al., *JCTC* 11, 2015, 2087)
adds a source estimate as an additive baseline; importance weighting (Huang et al., NIPS 2006;
Sugiyama et al., *JMLR* 8, 2007) keeps source labels in the objective — our DRST and KMM baselines are
instances; label-shift correction (Lipton et al., ICML 2018) reweights the label distribution.

## 8. Limitations, open questions, and what we are doing next

- **The literature contribution is not demonstrated in-domain.** Four designs plus a 28-family
  follow-up all returned null. We report this rather than keep searching for a variant that scores.
- **Cross-preparation transfer is where literature data finally helps.** Predicting *impregnation*
  literature gives ρ = 0.398; *non-impregnation* gives 0.238, where the model cannot select at all
  (top-decile picks average 11.50 % against a population mean of 10.34 %, while the true top decile
  averages 21.92 % — enrichment 0.42×, worse than random). **Adding impregnation literature to
  training raises ρ from 0.238 to 0.388 (+0.150, 5/5 seeds).** Two caveats: absolute performance
  remains poor, and plain merging outperforms the prior-feature construction (0.388 vs 0.318) — the
  value is the data, not the two-stage machinery.
- **Novel promoter families cannot be priced.** Structural, not a modelling deficiency; §5 quantifies
  the data needed to remove it.
- **The target carries a measurement-effort confound.** 47 catalysts have fewer than 20 measurements
  and systematically low maxima. More generally, how much of the grid was run tracks how well the
  catalyst performed: Spearman(cell size, cell maximum yield) = **+0.441**, mean cell yield rising from
  2.22 % in cells of 1–5 rows to 6.05 % in cells of 27. Only 811 of 917 catalysts have all five
  temperatures, and 186 cells are absent entirely. Excluding low-count catalysts changes our headline
  by 0.002, so nothing here hinges on it.

**Three questions, in order of value to us:**

1. **Do the ~27 condition settings per catalyst–temperature exist in retrievable form?** This is the
   largest single opportunity: the 19.9 % of variance now unreachable becomes largely learnable,
   condition-level modelling becomes meaningful, and row-level RMSE becomes a well-posed target again.
2. **Are those ~27 measurements distinct reaction conditions, or successive time-on-stream samples at
   a single condition?** We could not settle this from the file. If the latter, a catalyst's maximum
   is a fresh-catalyst transient rather than an achievable optimum — which changes both what our
   target means and whether §4's result transfers to a prospective design.
3. **Were incomplete runs stopped deliberately when results looked poor?** Coverage correlates with
   performance, and the answer decides whether that bias is correctable or is itself informative.

**What we are doing meanwhile.** Turning §4's subsampling result into a pre-registered screening
protocol, and preparing the wider screen-then-confirm design in §6 — both pending your answer to
question 2. We consider further in-domain literature-integration variants closed and are not pursuing
more. Any prospective campaign would have its expected hit rate (precision@20, CI 0.15–0.65)
pre-registered before results arrive.

*All numbers here are stored outputs of the scripts named at the head of this note; each figure is
generated from the corresponding experiment's JSON, so figures cannot drift from the experiments that
produced them.*
