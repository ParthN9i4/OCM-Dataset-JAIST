# Work Note (v2) — Incorporating Published Literature Data into the Lab OCM Yield Model

**To:** Prof. Taniike
**Topic:** A domain-adaptation study for C₂-yield prediction in Oxidative Coupling of Methane (OCM)
**Supersedes:** version 1 of this note. **Companion code:** `ocm_methodology.ipynb`,
`taniike_validation.py`, `phase3_lit_prior.py`, `phase4_family_diagnosis.py`,
`phase5_target_audit.py`, `phase6_our_experiments.py`, `phase6_candidates.py`

---

## What changed since version 1

Version 1 reported that a two-stage prior-feature method ("PFT") improved C₂-yield prediction by
10.6 % over a lab-only baseline. **Following the stricter validation Prof. Taniike proposed, that
improvement does not survive, and we withdraw the claim.**

| Claim in v1 | Status in v2 |
|---|---|
| PFT improves CV RMSE by 10.6 % (v1: 1.907 vs baseline 2.133) | **Withdrawn.** The gain was catalyst-identity leakage; under catalyst-grouped CV, PFT is 1.8 % *worse* than baseline |
| Literature data measurably helps the lab model | **Not demonstrated.** Four honest designs, all ≤ the composition-only control |
| Quantile normalisation is a necessary component | **Not supported.** Label treatment moves RMSE by 0.001–0.023, which at 3 seeds is inside run-to-run noise |
| — | **New:** the composition-only model screens *unseen* catalysts usefully (ρ = 0.761 over all 917 catalysts, 0.724 on the 771 with comparable measurement effort; enrichment 3.04–4.89×) |
| — | **New:** the Ba-family failure is mechanistically explained, and yields a data budget for new chemistry |

The cause was specific and is now understood: our Stage-1 expert was trained on literature data
*together with the lab training rows*, so under a random row split it had seen the very catalysts it
was later asked to help predict. What follows is the corrected study.

![**Figure 1 — The central finding.** Identical models under two evaluation protocols. When every measurement of a catalyst is confined to one fold, the reported improvement inverts.](fig_protocol_comparison.png)

## 1. Objective

We predict C₂ yield from catalyst composition using the lab's own data (89,074 measurements, 2025,
impregnation route). A body of published literature data (3,852 measurements, 1982–2019) is also
available. The question is whether the literature can improve the lab model without degrading it.

Two distributional differences make this a **domain-adaptation** problem: a **label shift** (literature
mean 8.670 % vs. lab 5.245 %, a +3.425 pp gap reflecting publication practice) and a **covariate
shift** (the two collections emphasise different chemistry).

## 2. Evaluation methodology

**Catalyst-grouped cross-validation is now our default protocol.** All measurements of a given
catalyst are assigned to exactly one fold, so every reported number answers: *how well do we predict
a catalyst nobody has made yet?* Anything fitted on data — scaler, domain classifier, both model
stages — sees training-fold data only.

**Why the previous protocol was inadequate — a quantitative argument.** The 89,074 measurements
comprise only **917 distinct catalysts** at 5 temperatures, giving 4,399 (catalyst, temperature)
cells holding 20.2 rows on average. Your description of these rows as measurements *under different
reaction conditions* is borne out by the row counts themselves, and we should have taken it more
literally than we first did. Cell sizes have a hard ceiling at exactly **27**, a second at exactly
**54 = 2 × 27**, and nothing above; **15 catalysts hold exactly 135 rows, and all 15 decompose as
exactly (27, 27, 27, 27, 27)**. That is **5 temperatures × 27 condition settings = 135**, matching
the 135 conditions you describe. We also tested the competing reading that 27 was an export cut-off
rather than a design size, and rejected it: **104 cells hold more than 27 rows**, which a top-27
cut-off could not have produced. (An order-statistic comparison of the gap structure agrees, but the
count is the argument that settles it.) One caveat we should state plainly: the rows within each cell
arrive already sorted by yield, which is what an export cut-off would also look like — so we lean on
the count rather than on the shape of the distribution.

The consequence is that these rows are **not replicates**, and the variation across them is not
measurement noise. Because those 27 condition settings are absent from the feature table, **19.9 %
of total yield variance lies within cells** and cannot be reached from composition and temperature
alone, which floors row-level RMSE at **1.757**. This is a property of the missing columns, not an
irreducible physical limit — recovering the condition data would make most of it learnable.

Version 1 reported 1.907 — only 0.15 above that floor. In hindsight this should itself have prompted
suspicion: a model cannot approach a floor that assumes knowledge of each catalyst's own cell means
unless it has, in effect, memorised those catalysts. Point-wise RMSE is therefore not merely less
relevant than catalyst-level metrics here; it is close to uninformative — and it is precisely the
metric that a catalyst-identity leak flatters most.

**Primary metrics** are consequently catalyst-level, as proposed: Spearman correlation between
predicted and observed *maximum* yield, and enrichment of true high performers among top-ranked
predictions.

## 3. Baseline and the two datasets

A LightGBM regressor trained on lab data only. Under the grouped protocol its CV RMSE is **2.943**
(row-level: 2.118 — the difference is the leakage, not a change of model).

![**Figure 2 — Label shift.** The literature (orange) is centred about 3.4 points above the lab data (blue). Systematic, not noise.](fig_mean_shift.png)

![**Figure 3 — Covariate shift.** The most-used elements differ between the two collections.](fig_element_usage.png)

## 4. Approaches evaluated

**Direct merge.** Pool all literature records with the lab data. The label shift enters the training
target and accuracy is not improved.

**Selective merge.** Make the literature more lab-like before merging — either a hard filter (DRST: a
logistic-regression domain classifier scoring each record by `P(lab | features)`, retaining 20.3 % at
τ = 0.30) or continuous importance weights (KMM). Both still place literature yields in the training
objective, so the label shift is unresolved.

![**Figure 4 — DRST scores.** Each literature record scored by how lab-like its chemistry is.](fig_drst_scores.png)

![**Figure 5 — KMM weights.** Continuous weights agree closely with the DRST filter (r = 0.792).](fig_kmm_weights.png)

**Prior-feature method (PFT).** Let the literature influence the final model only through a *predicted
value used as an input feature*, never as a training label: Stage 1 trains an expert on literature
data whose yields are rank-rescaled onto the lab scale; Stage 2 trains on lab data with that
prediction as one extra feature, using lab labels only.

![**Figure 6 — Stage-1 rescaling.** Literature yields (orange) mapped onto the lab range (green). *Note: the ablation in §5 shows this rescaling is not actually necessary — the final model uses only the prior's ordering.*](fig_bias_correction.png)

## 5. What the stricter validation showed

**The improvement was leakage.** Under catalyst-grouped CV: baseline **2.943**, PFT **2.995** (+1.8 %).
Three readings of the same ablation grid point to the mechanism (they share one 3-seed run, so we do
not present them as independent experiments):

1. Training Stage 1 on literature *alone* reduces the row-level gain from −9.7 % to **−2.4 %** (QN prior) or **−2.7 %** (rank prior)
2. Under grouped CV the joint variant (**2.982**, 3 seeds) is worse than literature-only (**2.938**, rank prior, 3 seeds)
3. On those same three seeds, literature-only (2.938) and the baseline (2.928) are indistinguishable

*(Seed counts differ between these checks and the headline table. The headline PFT figure of 2.995 is a
5-seed mean; 2.982 is the same configuration over the first 3 seeds only. The ablation grid ran at 3
seeds throughout.)*

**Your quantile-normalisation hypothesis is supported, though we will not state it as strongly as we
first did.** We compared the three label treatments you named under both protocols. The gaps are:
row-level 0.006 (QN vs raw) and 0.006 (QN vs rank); catalyst-grouped 0.023 (QN vs raw) and 0.001
(QN vs rank). The largest gap sits under our primary protocol, and it is close to the run-to-run
spread there (per-configuration standard deviations of 0.022 and 0.035 over 3 seeds). The honest
statement is therefore that **no label treatment is distinguishable from another at this seed
count**, not that the effect is exact. On that evidence the normalisation step can be dropped without
a measurable penalty, which is the practical conclusion you anticipated.

**With the leakage channel closed, we retested literature integration properly** — a literature-only
rank prior, similarity-to-literature features, a gated prior combining both, and a catalyst-level
direct merge. Success criteria were fixed before running. **None improved on composition alone.**

![**Figure 7 — No literature variant beats composition alone.** Catalyst-grouped protocol; the dashed line is the composition-only control. Deltas in parentheses.](fig_grouped_results.png)

We also tested a hypothesis of our own — that the prior might help specifically where our own
coverage is thin, suggested by one family (Zr) showing a consistent gain. Across **all 28 element
families with ≥50 catalysts** the mean effect was **−0.0025**, 14/28 positive, and the strongest
correlation with any coverage measure was |ρ| = 0.276 against a pre-registered threshold of 0.5. The
Zr result was selection from noise, and we discarded it.

## 6. What the model can do: screening unseen catalysts

Rebuilt at catalyst level — composition → maximum yield, 917 training examples, no temperature — the
model matches the full 89,074-row model on ranking (Spearman 0.760 vs 0.766, within seed noise) while
training on ~100× fewer rows. On genuinely unseen
catalysts:

| Metric | All 917 catalysts | 95 % CI | Equal-effort set (771) |
|---|---|---|---|
| Spearman ρ (predicted vs. observed max yield) | 0.761 | 0.725 – 0.785 | **0.724** |
| Enrichment of true top-decile among top-decile predicted | 4.28× | **3.04 – 4.89×** | **3.77×** |
| Precision@20 (of 20 nominated, fraction truly top-decile) | 0.44 | **0.15 – 0.65** | 0.35 |

We quote intervals rather than point estimates: with 92 catalysts in the top decile of 917, these
quantities are considerably less precise than a single number suggests.

**Why the second column exists, and why we consider it the honest one.** Grid coverage in your data
is coupled to performance — cells that were run further contain better yields — so a score computed
over all 917 catalysts is partly a record of which experiments were carried to completion. The
*equal-effort set* is the 771 catalysts with at least 20 measurements in at least one temperature
cell; on that set the coupling is gone by measurement rather than by assumption, with
Spearman(measurement count, observed maximum) falling from **+0.293** to **+0.003**. Moving to it
costs us 0.037 Spearman and 0.51× enrichment.

That drop is not an artifact of scoring fewer catalysts. Drawing 300 **random** 771-catalyst subsets
from the *same* predictions gives Spearman 0.767 with a 95 % range of 0.756 – 0.780; the equal-effort
value of 0.724 lies below that range.

**A negative control we think is worth more than either number.** We refitted the identical model
with the *number of measurements* as its target — it never sees a yield — and ranked catalysts by
predicted measurement effort. That ranking reaches **Spearman 0.400** against observed maximum yield,
but its enrichment is **0.87×**, i.e. no better than choosing at random. So rank correlation is
partly purchasable from experimental effort, whereas enrichment is not. This is the concrete reason
we treat enrichment, not ρ, as the primary screening metric — and it is the sharpest test we could
devise of whether the model is merely recognising which catalysts you chose to finish.

One consequence for campaign design. Inside the model's own top-ranked catalysts — the only regime a
synthesis campaign ever occupies — the internal ranking carries little information: ρ within the top
150 predictions is **0.179**, and within the top 20 it is **−0.066**. The model *selects* well (its
top 20 average 17.3 % observed maximum against ~10.5 % for the library as a whole) but does not
*order* within its own selection. A shortlist should therefore be treated as a set to be tested, not
as a league table, and a campaign small enough to fit inside that regime cannot by itself confirm or
refute the model.

![**Figure 8 — What drives achievable maximum yield.** Composition-only model. Ba dominates, followed by chemically sensible contributors. *(The corresponding figure in v1 ranked the literature prior first; that model was the leaked pipeline, so the figure illustrated the leak rather than the chemistry.)*](fig_shap_bar.png)

## 7. Why the Ba family fails, and how much data a new family needs

Family holdouts behave reasonably for La, Ti, Zr and Ce (ρ 0.62–0.68) but poorly for Ba (0.526). The
cause is quantifiable: Ba catalysts average **13.76 %** maximum yield against **8.95 %** for the rest,
and **78 % of the lab's top decile contains Ba**. Removing them removes the high-yield regime itself.

The mechanism is provable rather than inferred. With no Ba catalysts in training, the Ba column is
constant, so no tree splits on it — and retraining with that column **deleted entirely** produces
**bit-identical predictions**. The model prices Ba catalysts as though Ba were absent, underpredicting
the best of them by **9.8 yield points**. Ten random pseudo-families of equal size score 0.752, so
this is chemistry and label coverage, not sample size.

That raised a question we found more useful than *whether* it fails: **how many labelled members of a
family are needed before the model can price it?**

| Family | ρ, none seen | ρ, fully seen | % of ceiling at zero |
|---|---|---|---|
| **Ba** | 0.509 | 0.683 | **74.4 %** |
| La | 0.678 | 0.731 | 92.7 % |
| Ti | 0.618 | 0.664 | 93.0 % |
| Zr | 0.646 | 0.724 | 89.2 % |
| Ce | 0.643 | 0.722 | 89.1 % |

![**Figure 9 — A data budget for new chemistry.** Left: performance against the number of family members already measured. Right: fraction of achievable performance reached having seen none.](fig_learning_curve.png)

Ba is the most *consequential* family to lose, not the hardest to predict. Across all 28 families we
tested, five score below Ba's 0.526: Pd 0.217, Cu 0.286, Al 0.329, Ni 0.329 and Co 0.493. What sets Ba
apart is its weight. It holds 78 % of the top decile, so losing it removes the high-yield regime
itself, and it gains far more than any other family from seeing its own members (+0.175, against
+0.047 to +0.079 elsewhere).

**On the data budget**, the threshold depends entirely on how "80 % of achievable performance" is
defined, and the two natural readings differ by a factor of five. We give both rather than quote one.

For Ba the learning curve runs 0.509 (no Ba in training) → 0.565 (10 members) → 0.616 (25) → 0.651
(50) → 0.683 (all 204). Taking 80 % of the **final level** (0.8 × 0.683 = 0.547) puts the threshold at
**10 members**; taking 80 % of the **gain from seeing the family** (0.509 + 0.8 × 0.174 = 0.648) puts
it at roughly **50**. Our stored figure uses the first definition. We now think the second is the more
meaningful one, because Ba already reaches 0.509 having seen no Ba catalysts at all — so 80 % of the
level is a bar that is nearly cleared by seeing nothing, and the "10" understates what is actually
required. The same caution applies to the **0 for La, Ti, Zr and Ce**: those four exceed 80 % of their
final level having seen none, which says more about the leniency of that bar than about the families.

Reaching 95 % takes about 50 for Ba and 25–50 for the others. We would treat all of this as
order-of-magnitude guidance only: for Ti and Zr the per-point seed spread (0.03–0.12) is comparable to
the whole learning-curve gain, so those two curves are not resolved.

## 8. A candidate list for prospective validation

We enumerated **26,414 unseen candidates** in the laboratory's own design grammar — impregnation, one
support at ~90 % with 2–3 promoters at ~3.33 %, drawn from the supports and promoters already in use —
and scored them with a 10-seed ensemble. No literature prior is used.

Every candidate carries a coverage flag, and the safeguard is verified rather than assumed: for an
element absent from our data (Ag), predictions computed with and without its column differ by exactly
zero — confirming such candidates are unpriceable and must be flagged rather than ranked.

The highest-ranked candidate is **Ba(90) + Mo(3.33) + Zn(3.33) + Fe(3.33)**, predicted 18.79 %. We
deliberately do not attach an error bar to that figure. The ± 0.09 our ensemble reports is only the
spread across 10 seeds; the model's actual catalyst-level error on held-out catalysts is about
2.7 yield points (MAE). The number ranks candidates. It does not forecast a yield.

Two caveats. Absolute predictions compress at the extreme — our ensemble maximum is 18.79 % while
observed training yields reach 21.50 % — so the **ranking** is the deliverable, not the predicted
value. And the top 20 are chemically monotonous: all contain Ba, most contain Mo. A diversity
constraint did not meaningfully change this, because the model's preference for Ba is genuine. We
therefore also provide the best candidate per support, with its cost in predicted yield:

| Support | Ba | Ti | La | Ca | Mg | Si | Al | Zr | Ce |
|---|---|---|---|---|---|---|---|---|---|
| Best predicted max yield (%) | 18.79 | 16.49 | 15.82 | 15.54 | 14.66 | 13.83 | 13.63 | 13.02 | 12.93 |

**Our recommended campaign (17 catalysts, `campaign_shortlist.csv`).** Rather than submit twenty
near-identical catalysts, we suggest splitting the budget:

- **Tier A — 12 catalysts, the model's optimum** (predicted 18.33–18.79 %). This is where the expected
  hits are. From the retrospective grouped-CV precision@20, we would expect roughly **2–8 of the 12**
  to be genuine top-decile performers (95 % CI 0.15–0.65 — deliberately a wide interval).
- **Tier B — 5 catalysts, one per alternative support** (Ti 16.49, La 15.82, Ca 15.54, Mg 14.66,
  Si 13.83). These cost predicted yield and we do not expect them to win. They are an *information*
  purchase: 78 % of the lab's existing top decile already contains Ba, so the model's strong Ba
  preference may partly reflect that coverage rather than chemistry. If several Tier-B catalysts
  outperform their predictions, that tells us something the Tier-A catalysts cannot.

We would of course defer to your judgement on synthesis feasibility, and are happy to reweight the
split.

**A limit we should state plainly before you spend reactor time.** Seventeen catalysts, all drawn from
the model's top-ranked region and with no control arm, cannot confirm or refute the model. As noted in
§6, the model's internal ordering inside that region is close to uninformative (ρ = 0.179 within its
top 150, −0.066 within its top 20), so a campaign drawn entirely from it has very wide error bars on
any correlation it measures. The list is a reasonable set of catalysts to *try*; it is not a test.

If a test is wanted, our retrospective replay on your own archive suggests a different allocation of
the same reactor budget. Measuring 5 conditions at each of 750, 800, 850 and 900 °C — 20 runs rather
than 135 — reproduces the full ranking at ρ = **0.949** (200 resamples, spread ±0.003; systematically
low by 1.38 yield points, a bias that can be pre-declared). This is measured on the **759** catalysts
that carry at least 5 rows in each of those four cells, out of the 811 measured at all five
temperatures. That buys roughly 72 catalysts screened
instead of 17 measured exhaustively, with the best few then confirmed at full coverage, and with part
of the batch drawn at random from the same candidate grammar as a control arm. The cost is more
syntheses for the same number of reactor runs, which may or may not suit your constraints.

This depends on one thing we could not determine from the data file, and it is the question we would
most like answered: **are the ~27 measurements per catalyst–temperature distinct reaction conditions,
or successive time-on-stream samples at a single condition?** If the latter, then selecting 5 of 27
saves analysis rather than reactor hours, the arithmetic above changes, and — more importantly — a
catalyst's observed maximum is a fresh-catalyst transient rather than an achievable optimum, which
would change what our target quantity means.

## 9. Relation to prior work

The mechanism of PFT — using a model's prediction as an input feature — is **stacked generalisation**
(Wolpert, *Neural Networks* 5, 1992, 241–259) applied across distributions. "Prior Feature Transfer"
was our internal shorthand, not a standard term, and we note that "feature transfer" usually denotes
*representation* transfer, which this is not.

**Our corrected understanding of when such a method can help.** Because the prior `p = f(x)` is a
deterministic function of features the final model already has, `I(y; x, f(x)) = I(y; x)` — it cannot
add information. It can only supply an inductive bias, or exploit source data covering regions the
target data does not. Our results indicate neither applies here at the scale that matters, for three
reasons: ~79.7 % of the literature is out-of-distribution relative to the lab; quantile normalisation
aligns marginal but not conditional distributions; and literature yields are measured under each
paper's own conditions, whereas our target is the maximum over a standardised battery — semantically
different quantities. **The honest scope statement is that this family of methods is a local-coverage
tool, not a global-information tool.**

Related approaches remain distinct: Δ-machine-learning (Ramakrishnan et al., *JCTC* 11, 2015, 2087)
adds a source estimate as an additive baseline, inheriting its scale; importance weighting (Huang et
al., NIPS 2006; Sugiyama et al., *JMLR* 8, 2007) corrects which records are used but keeps source
labels in the objective — our DRST and KMM baselines are instances; label-shift correction (Lipton et
al., ICML 2018) reweights to correct the label distribution.

## 10. Limitations and next steps

- **The literature contribution is not demonstrated.** Four designs plus a 28-family follow-up all
  returned null. We report this rather than continue searching for a variant that scores well.
- **Novel promoter families cannot be priced.** This is structural, not a modelling deficiency — but
  §7 quantifies the data required to remove the limitation.
- **The target carries a measurement-effort confound, and grid coverage is not random.** 47 catalysts
  have fewer than 20 measurements and systematically low maxima. More generally, how much of the
  135-condition grid was actually run tracks how well the catalyst performed: Spearman(cell size,
  cell maximum yield) = **+0.441**, and mean cell yield rises monotonically from 2.22 % in cells with
  1–5 rows to 6.05 % in cells with 27. Only 811 of 917 catalysts have all five temperatures present,
  and 186 catalyst–temperature cells are absent entirely. This reads as unpromising combinations
  being abandoned partway through the grid. Excluding the low-count catalysts changes our headline
  metric by 0.002, so no conclusion here hinges on it — but we would welcome confirmation of whether
  the incomplete cells were stopped deliberately, since that decides whether the bias is correctable
  or is itself informative.
- **The unrecorded reaction conditions are the largest single opportunity.** If those ~27 condition
  settings per catalyst–temperature exist in a retrievable form, the 19.9 % of variance now
  unreachable from composition and temperature becomes largely learnable, condition-level modelling
  becomes meaningful, and row-level RMSE becomes a well-posed target again rather than a floor
  artefact. One further question would settle a reading we could not resolve from the file alone:
  are the ~27 measurements per catalyst–temperature **distinct reaction conditions**, or successive
  **time-on-stream samples** at a single condition? If the latter, a catalyst's maximum is a
  fresh-catalyst transient rather than an achievable optimum, which would change the target we
  should be predicting.
- **Cross-preparation transfer is now measured, and it is where literature data finally helps.**
  Predicting *impregnation* literature (different source, same preparation) gives ρ = 0.398; predicting
  *non-impregnation* literature gives ρ = 0.238 — so changing preparation costs a further 0.160. At that
  range the model is not usable for selection out of preparation: its top-decile picks average 11.50 %
  true yield against a population mean of 10.34 %, while the genuine top decile averages 21.92 %
  (enrichment 0.42×, i.e. worse than picking at random). **But adding impregnation literature to training
  raises ρ from 0.238 to 0.388 (+0.150, 5/5 seeds)** — the first setting in this study where literature
  data measurably helps, and consistent with the scope statement in §9: the lab has *no* coverage of
  non-impregnation chemistry, so the literature supplies genuinely new information there. Two caveats:
  absolute performance remains poor, and plain merging outperforms the prior-feature construction
  (0.388 vs 0.318). Details in `phase7_prep_ood.py`; this supersedes the earlier row-level OOD numbers.
- **Prospective validation** is the natural next step, and we would pre-register the expected hit rate
  (precision@20, CI 0.15–0.65) before any results arrive.

*All numbers in this note are stored outputs of the scripts named at the head of the document; each
figure is generated from the corresponding experiment's JSON so that figures cannot drift from the
experiments that produced them.*
