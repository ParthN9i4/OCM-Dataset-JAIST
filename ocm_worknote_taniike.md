# Work Note v2 — Update

**To:** Prof. Taniike
**Topic:** C₂-yield prediction for OCM — results under the stricter validation
**Supersedes:** version 1. Problem setup, datasets and the five candidate methods are described there
and are not repeated. This note reports only what has changed.
**Code:** `ocm_eval.py`, `taniike_validation.py`, `phase3_lit_prior.py`, `phase4_family_diagnosis.py`,
`phase5_target_audit.py`, `phase6_our_experiments.py`, `phase6_candidates.py`, `phase7_prep_ood.py`,
`phase8_target_robustness.py`, `phase9_equal_effort_eval.py`, `phase10_condition_subsampling.py`

---

## 1. The improvement does not survive, and we withdraw it

| Claim in v1 | Status |
|---|---|
| PFT improves CV RMSE 10.6 % (1.907 vs 2.133) | **Withdrawn** — the gain was catalyst-identity leakage |
| Literature data helps the lab model | **Not demonstrated in-domain**; it does help across preparation methods (§3) |
| Quantile normalisation is necessary | **Not supported** — label treatment moves RMSE inside run-to-run noise |

| Protocol | Baseline | PFT |
|---|---|---|
| Row-level split (v1) | 2.118 | **1.912 (−9.7 %)** |
| **Catalyst-grouped split** | 2.943 | **2.995 (+1.8 %, worse)** |

Same models, same data; the difference between the two lines is leakage. **Mechanism:** our Stage-1
expert was trained on literature data *together with the lab training rows*, so under a random row
split it had already seen the catalysts it was later asked to help predict. Catalyst-grouped
splitting is now the default in our shared evaluation module and cannot be bypassed by accident.

## 2. What the data actually contains

The 89,074 measurements are **917 distinct catalysts** across 4,399 (catalyst, temperature) cells
averaging 20.2 rows. Your description of these rows as measurements *under different reaction
conditions* is borne out by the row counts, and we should have taken it more literally.

Cell sizes have a hard ceiling at exactly **27**, a second at exactly **54 = 2 × 27**, nothing above;
**15 catalysts hold exactly 135 rows, and all 15 decompose as (27, 27, 27, 27, 27)** — that is
5 temperatures × 27 condition settings = 135, matching your figure. We tested the competing reading
that 27 was an export cut-off and rejected it: spacing between the two lowest values in a 27-row cell
is 2.03× the interior spacing, against 0.90× when a larger cell is truncated to 27.

Because those 27 settings are absent from the feature table, **19.9 % of total yield variance lies
within cells**, unreachable from composition and temperature, flooring row-level RMSE at **1.757** —
a property of missing columns, not a physical limit. Version 1 reported 1.907, only 0.15 above that
floor, which should itself have prompted suspicion. **Primary metrics are therefore catalyst-level,
as you proposed:** rank correlation on maximum yield, and enrichment among top-ranked predictions.

## 3. Literature integration: null in-domain, useful across preparations

With the leak closed we retested four designs — a literature-only rank prior, similarity features, a
gated prior, and a catalyst-level direct merge — with criteria fixed before running. **None improved
on composition alone** (control ρ = 0.761; best variant 0.758). A follow-up across **all 28 element
families with ≥50 catalysts** was also null: mean effect −0.0025, strongest correlation with any
coverage measure |ρ| = 0.276 against a pre-registered threshold of 0.5.

**The exception is cross-preparation transfer.** Predicting *non-impregnation* literature from lab
data alone gives ρ = 0.238 with enrichment 0.42× — worse than random selection. Adding impregnation
literature to training raises this to **ρ = 0.388, enrichment 1.34×**. This is the one setting where
literature data measurably helps, and it is where the lab has no coverage at all. Two caveats:
absolute performance remains poor, and a **plain merge outperforms our two-stage construction (0.388
vs 0.318)** — the value is the data, not the machinery.

## 4. Screening performance, and a correction to our own number

| Metric | All 917 catalysts | Equal-effort set (771) |
|---|---|---|
| Rank correlation (predicted vs observed max) | 0.767 | **0.724** |
| Enrichment of true top-decile | 4.35× | **3.77×** |
| Precision@20 | 0.45 | 0.35 |

**Why the second column exists.** Grid coverage is coupled to performance — cells run further contain
better yields — so a score over all 917 is partly a record of which experiments were completed. The
equal-effort set is the 771 catalysts with ≥20 measurements in at least one cell; there the coupling
is gone by measurement, with Spearman(measurement count, observed maximum) falling from **+0.293 to
+0.003**. We regard 0.724 as the honest figure. The drop is not an artifact of scoring fewer
catalysts: 300 **random** 771-catalyst subsets of the same predictions give 0.767 with a 95 % range
of 0.756–0.780, and 0.724 lies below it.

**A negative control we value more than either number.** Refitting the identical model with the
*number of measurements* as its target — it never sees a yield — still reaches **ρ = 0.400** against
observed maximum yield, but enrichment **0.87×**, no better than chance. Rank correlation is partly
purchasable from experimental effort; enrichment is not. That is why enrichment is our primary metric.

**One limit for campaign design.** Inside the model's own top-ranked region the internal ordering
carries little information: ρ = 0.179 within the top 150, **−0.066 within the top 20**. It *selects*
well (top 20 average 17.3 % observed maximum against ~10.5 % library-wide) but does not *order*
within its selection — a shortlist is a set to test, not a ranking.

## 5. Training label: maximum versus median

Each catalyst's label must be built from its ~135 readings. We compared the single highest reading
anywhere against the median *within* each temperature group — across the cluster of near-identical
readings — then the best of those per-group medians. Both scored against the same quantity, the true
observed maximum; only the training label differs.

| Training label | ρ, all 917 | ρ, equal-effort | Enrichment, all 917 | Enrichment, equal-effort |
|---|---|---|---|---|
| Single highest reading | 0.767 | 0.724 | 4.35× | 3.77× |
| Per-group median | **0.772** | **0.733** | 4.13× | **3.90×** |

The median leads on rank correlation in both populations and on enrichment among comparably-tested
catalysts; it trails on enrichment across all 917. Differences are small and a catalyst-level
bootstrap does not separate them. **We are proceeding with the per-group median**, because it cannot
be set by a single high reading — a choice rather than a result, since the evidence permits but does
not compel it.

## 6. How much of the condition grid is actually needed

Keeping only k of the ~27 measurements per catalyst–temperature cell (random draws, 5 independent
repeats per k, always scored against the true maximum from the full data):

| Measurements kept | Share of grid | ρ | Enrichment |
|---|---|---|---|
| 1 per cell (~5 runs/catalyst) | 5 % | 0.759 ± 0.002 | 3.99× |
| 3 per cell (~15 runs/catalyst) | 15 % | 0.765 ± 0.002 | 4.24× |
| all ~27 per cell | 100 % | 0.761 | 4.28× |

One measurement per cell ranks almost as well as all 27, stable across draws. Labels are biased low
(−2.8 yield points at k = 1) but roughly uniformly, so ranking survives; absolute estimates would not.

**Temperature is different.** 700 °C alone gives ρ = 0.346, enrichment 1.07× — no better than chance
— because 70 % of catalysts peak at 800 °C or above. Redundancy is *within* a temperature, not across
them: between-catalyst spread in true maximum (SD 5.32) is 3.8× the spread within one cell (SD 1.41).

This assumes the 27 slots are individually selectable — see question 2.

## 7. Novel promoter families, and a candidate list

Family holdouts behave reasonably for La, Ti, Zr and Ce (ρ 0.62–0.68) but poorly for Ba (ρ = 0.526,
`phase4_family_diagnosis.py`). Ba catalysts average 13.76 % maximum yield against 8.95 % for the rest,
and **78 % of the lab's top decile contains Ba** — removing them removes the high-yield regime. The
mechanism is provable: with no Ba in training the Ba column is constant, no tree splits on it, and
retraining with that column **deleted entirely gives bit-identical predictions**. The model
underprices the best Ba catalysts by **9.8 yield points**.

A learning curve (`phase6_our_experiments.py`) gives a data budget: Ba runs 0.509 (none seen) → 0.565
(10) → 0.616 (25) → 0.651 (50) → 0.683 (all 204). Taking 80 % of the *final level* gives ~10 members;
80 % of the *gain* gives ~50. The second is the meaningful reading, since Ba reaches 0.509 having seen
none.

We also ranked **26,414 unseen candidates** in your design grammar; `campaign_shortlist.csv` holds a
suggested 17. The ranking is the deliverable, not the predicted value (ensemble maximum 18.79 %
against observed yields reaching 21.50 %; catalyst-level error ~2.7 points).

## 8. Two questions, and what we are doing meanwhile

Both concern information absent from the exported data file that further analysis on our side cannot
recover.

1. **Do the ~27 condition settings per catalyst–temperature exist in retrievable form?** They are
   currently invisible to the model, leaving the 19.9 % above unreachable. With them we could model
   conditions directly rather than averaging across them.
2. **Are those ~27 measurements distinct reaction conditions, or successive time-on-stream samples at
   a single condition?** We could not settle this from the file, and it changes what our target
   represents: under the second reading a catalyst's maximum reflects early-run behaviour rather than
   a sustained operating point, and §6 would not transfer to a prospective design.

A related observation rather than a question: coverage correlates with performance
(Spearman(cell size, cell maximum) = +0.441) and 186 cells are absent entirely. If incomplete runs
were stopped deliberately, that bias is correctable; if not, it may itself be informative. The
correction in §4 holds either way.

**Meanwhile**, our focus is accuracy rather than scope: refining the composition-based model, since §6
suggests the data may support a simpler and more stable formulation than we use now; and revisiting
the two-stage construction for cross-preparation specifically, where it still loses to a plain merge
(0.318 vs 0.388). That gap looks closable.

*All numbers are stored outputs of the scripts named above.*
