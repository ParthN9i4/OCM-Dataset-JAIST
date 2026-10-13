# Work Note v3 — Analysis of the Gas-Flow Condition Data

**To:** Prof. Taniike  
**Follows:** Work Note v2 and your reply with `OCM_data.csv`. This note reports only what is new.  
**Code:** `phase11_condition_baseline.py`, `phase12_runs_and_coverage.py`, `phase8b_label_sweep.py`,
`phase13_condition_value.py`, `phase14_crossprep_pft.py`. Every number below is stored in the JSON file
of the same name.

**Protocol.** As in v2:
- All measurements of a catalyst stay together, in either training or testing.
- Five repeats use different random splits.
- A difference between two models is reported with a 95% confidence interval (CI), from resampling
  catalysts.
- Yields and errors are in percentage points of C₂ yield ("points").

---

## 1. What the condition data shows

- The design has 5 temperatures (700–900 °C) and 27 gas-flow conditions per temperature.
- The 27 conditions form a full 3 × 3 × 3 design:
    - total flow of 10, 15 or 20 (the file gives no unit)
    - Ar fraction of 0.70, 0.40 or 0.15
    - CH₄:O₂ ratio of 2, 4 or 6
- The condition file answers both questions in v2 (§8). Your mass-balance explanation accounts for the
  incomplete groups.
- The readings at one temperature are therefore 27 different conditions. v2 (§5) called them
  near-identical readings, which was wrong. The training label is unchanged. It is now understood as a
  median over up to 27 different conditions, not over repeated readings.

## 2. Prediction of individual measurements

- Adding the three gas-flow columns lowers the RMSE for unseen catalysts from 2.90 to 2.66 points.
- The reduction is 0.24 points (8.1%), with a 95% CI of 0.22 to 0.26.
- Most of the gain comes from predicting how yield changes between conditions at the same temperature.
- The model settings were tuned after v2, which is why the baseline here differs from v2's 2.94.

## 3. Ranking of catalysts

The conditions do not improve the ranking of catalysts by best yield.

- **The comparison.** Each catalyst is scored on the same 18 conditions: CH₄:O₂ = 2 at 800 and 850 °C.
  465 of the 917 catalysts were measured at all 18. We chose this set by measurement coverage alone,
  before comparing models.
- **The metric.** The rank correlation between predicted and observed best yield over those 18
  conditions: 0.732 with the conditions and 0.727 without.
- **The result.** The difference is +0.005 (95% CI −0.007 to +0.015), which is no detectable change.
- **Two other scorings.**
    - Scoring each catalyst only over the conditions that passed the mass-balance check shows an apparent
      gain. It disappears on the 771 catalysts with comparable measurement effort (the equal-effort set
      of v2), so we do not count it as a gain.
    - Taking the best prediction over all 135 conditions ranks slightly worse (−0.014). That comparison
      is biased, because excluded runs are missing from the observed best yield.

## 4. Where the conditions add value

**Choosing conditions for a new catalyst.** The condition-aware model of §2 ranks a new catalyst's 135
conditions (5 temperatures × 27).

| Rule for choosing three conditions to run | Mean shortfall from the catalyst's best yield |
|---|---|
| Model ranking | 1.75 points |
| Best three conditions for the same support oxide, taken from other catalysts | 1.90 points |

- The model's advantage is 0.15 points (95% CI 0.07 to 0.23), and it holds in all five repeats.
- The mean best yield is 10.4 points, so the advantage is 1.4% of it. The effect is consistent but small.
- About 16% of chosen conditions were never measured under either rule; these count as failed runs.
  Skipping them instead gives the same advantage.

**A reduced measurement set.** From training catalysts only, we chose the 10 of the 135 conditions that
most often give a catalyst its best yield.

- **What was chosen.** All 10 have CH₄:O₂ = 2, and most have total flow 20. The set varies slightly
  between splits: 7 conditions were chosen in all 25 splits.
- **Result.** On held-out catalysts, the best yield over these 10 conditions ranks catalysts with a rank
  correlation of 0.969 against the full data. This is the mean over five repeats; 92% of catalysts had
  at least one of the 10 measured.
- **Comparison.** Random sets of 10 conditions give 0.917 on average (97.5th percentile 0.925). A second
  selection rule, which adds the conditions that most improve the ranking, gave 0.939.
- **Caution.** The result is optimistic. 84.5% of catalysts have their best yield at CH₄:O₂ = 2, and the
  full-data best yield comes from the same runs. A single fixed set is needed before any prospective test.
- **Training on the reduced labels.** A composition model trained on the best yield over the second rule's
  10 conditions ranked 0.009 lower (95% CI −0.018 to +0.001). A small loss is possible.

**Per-catalyst response models.** Two models of each catalyst's response to the conditions were less
accurate than the model of §2, so we have stopped this work.

## 5. Training label

We keep the per-temperature median, then the highest of these medians, as in v2.

- **Stability.** Removing each catalyst's single highest reading moves the median label by 0.01 points.
  It moves the maximum by 0.38 points. Both are medians over catalysts.
- **New comparison.** The upper quartile ranks slightly better than the maximum (+0.006, 95% CI +0.001 to
  +0.012), but not better than the median (+0.001, 95% CI −0.002 to +0.005). We therefore keep the
  median.
- **Caution.** 84.5% of catalysts have their best yield at CH₄:O₂ = 2, the lowest ratio tested. The true
  optimum for these catalysts may lie outside the tested range.

## 6. Literature data across preparation methods

In v2, one model trained on laboratory and impregnation literature data together (a "merge") ranked
non-impregnation literature better than our two-stage model: rank correlation 0.388 versus 0.318.

- **Two-stage variants.**
    - We tested three further variants. In one, every stage-1 prediction comes from a model that did not
      see that row.
    - None of these is the original two-stage model: none uses the DRST filter, and stage 1 sees
      literature data only.
    - None ranked better than the merge. Two ranked lower, and one showed no detectable difference.
- **Rescaling.**
    - One element of our two-stage method helps on its own: rescaling the literature yields onto the
      laboratory yield distribution (quantile normalisation) before merging.
    - Rank correlation rises from 0.388 to 0.422. The difference is +0.034 (95% CI +0.018 to +0.050).
      It holds in all five repeats and under the median label.
    - Enrichment of top performers shows no detectable change.
- **Why this differs from v2.** In v2 (§1), the same rescaling made no measurable difference for
  laboratory catalysts. It helps here, where the literature yields differ in scale from the laboratory yields.
- **Status.** This comparison was added during the analysis, so we treat it as exploratory.
- **Scope.** The literature file has no condition columns, so the gas-flow data cannot enter this part.

## 7. Two observations on the data

**Catalysts with two sets of runs.**
- For 28 catalysts, 1,906 conditions appear twice.
- The two readings of a condition differ by a median factor of 2.3.
- The higher readings form a smooth pattern across conditions, and so do the lower readings. A random
  mix of the two does not. This suggests two runs at different yield levels.
- 10 of the 92 catalysts in the top 10% by best yield are among these 28; chance would give 2.8. If the
  two readings are averaged, 4 remain.

**Excluded runs.**
- The rank correlation between a catalyst's number of missing conditions (out of 135) and its best yield
  is −0.29. Exclusions are therefore more frequent for weaker catalysts.
- Within a catalyst, excluded conditions rank slightly lower than measured ones, by 2–3 percentile
  points. The difference is detectable, but it is below the 5-point threshold we set in advance.
- Only 46% of excluded conditions could be tested this way, because the test needs the same condition
  measured at a neighbouring temperature.

## 8. Questions and next steps

Both questions concern records that are not in the data file.

1. **For the 28 catalysts with two sets of runs, is there a record of the run date or catalyst batch?**
   About 6 of the 92 top-10% places depend on how the two sets are treated.
2. **Are the records excluded for poor mass balance still available?**
   - Our indirect test reached only 46% of them.
   - They would show whether each catalyst's best observed yield is complete.
   - They would also show how a shorter protocol should treat a failed run.

**Next steps.**
- Build on the rescaling result (§6).
    - We will add the DRST filter and an out-of-fold stage-1 feature, one at a time, with the rule for
      success fixed in advance.
    - Stage 1 will not be trained on laboratory rows, the source of the leakage reported in v2.
- Fix a single 10-condition set (§4) and test how it ranks catalysts not used to choose it, before any
  prospective use.
