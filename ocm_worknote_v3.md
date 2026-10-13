# Work Note v3 — Analysis of the Gas-Flow Condition Data

**To:** Prof. Taniike  
**Follows:** Work Note v2 and your reply with `OCM_data.csv`. This note reports only what is new.  
**Code:** `phase11_condition_baseline.py`, `phase12_runs_and_coverage.py`, `phase8b_label_sweep.py`,
`phase13_condition_value.py`, `phase14_crossprep_pft.py`. Every number below is stored in the JSON file
of the same name.

**Protocol.** Unchanged from v2. All measurements of a catalyst stay in one fold. Five repeats with
different fold assignments. A difference between two models is reported with a 95% confidence interval
from resampling catalysts.

---

## 1. What the condition data settled

- Each catalyst was tested at 5 temperatures (700–900 °C) and 27 gas-flow conditions per temperature.
- The 27 conditions form a full 3 × 3 × 3 design:
    - total flow of 10, 15 or 20 (in the units recorded in the file)
    - Ar fraction of 0.70, 0.40 or 0.15
    - CH₄:O₂ ratio of 2, 4 or 6
- The readings within one temperature are therefore 27 distinct conditions. Version 2 (§5) called them
  near-identical readings. That description was wrong. The training label is unchanged (§5 below). Only
  the reason for it changes.
- Incomplete groups come from runs excluded for poor mass balance, as you explained. This answers both
  questions in version 2 (§8).

## 2. Prediction of individual measurements

Adding the three gas-flow columns lowers the prediction error for unseen catalysts:

- RMSE falls from 2.90 to 2.66, a reduction of 8.1%.
- The 95% confidence interval of the reduction is 0.22 to 0.26.

Most of the gain comes from predicting how yield changes between conditions at the same temperature.

## 3. Ranking of catalysts

The conditions do not improve the ranking of catalysts by best yield.

- A fair comparison needs the same conditions for every catalyst. We used the 18 conditions with
  CH₄:O₂ = 2 at 800 and 850 °C. 465 catalysts were measured at all of them. We chose this set by
  measurement coverage alone, before comparing any models.
- The change in rank correlation is +0.005 (95% CI −0.007 to +0.015). We find no detectable change.
- Scoring each catalyst only over the conditions that passed the mass-balance check shows an apparent
  gain. That gain disappears on catalysts with comparable measurement effort, so we do not report it.

## 4. Where the conditions add value

**Choosing conditions for a new catalyst.** The model ranks a new catalyst's 135 conditions (5 temperatures
× 27).

| Rule for choosing three conditions to run | Mean shortfall from the catalyst's best yield |
|---|---|
| Model ranking | 1.75 yield points |
| Best three conditions for the same support oxide, taken from other catalysts | 1.90 yield points |

- The model's advantage is 0.15 yield points (95% CI 0.07 to 0.23). It holds in all five repeats.
- The effect is consistent but small: about 1.4% of the mean best yield (10.4%).
- A chosen condition that was never measured counts as a failed run.

**A reduced measurement set.** We chose 10 of the 135 conditions, using only training catalysts: the 10
that most often give a training catalyst its best yield.

- On held-out catalysts, the best yield over these 10 conditions ranks catalysts with a rank correlation
  of 0.969 against the full data. This is the mean over five repeats; 92% of catalysts had at least one
  of the 10 measured.
- Ten random conditions reach at most 0.925 (97.5th percentile).
- This estimate is optimistic. The full-data best yield comes from the same runs.
- A ranking model trained on 10-condition labels shows no detectable loss (−0.009, 95% CI −0.018 to
  +0.001).

**Per-catalyst response models.** A model that predicts each catalyst's response to the conditions
directly did not improve on the current model. The current model already captures most of the learnable
pattern, so we have stopped this line.

## 5. Training label

We keep the per-temperature median, then the highest of these medians, as stated in v2.

- **Reason.** The label rests on many readings, not one. Removing each catalyst's single highest reading
  moves the median label by 0.01 yield points. It moves the maximum by 0.38 points. Both figures are
  medians over catalysts.
- **Ranking.** The median versus the maximum: +0.005 (95% CI −0.002 to +0.012), no detectable
  difference. The upper quartile: +0.006 (95% CI +0.001 to +0.012).
- **One caution.** 84.5% of catalysts reach their highest yield at CH₄:O₂ = 2, the lowest ratio tested.
  For most catalysts the best yield therefore lies at the edge of the tested conditions.

## 6. Literature data across preparation methods

In v2, a plain merge of laboratory and impregnation literature data ranked non-impregnation literature
better than our two-stage model (rank correlation 0.388 versus 0.318).

- **Two-stage designs.** We tested three further designs, including one in which every stage-1 prediction
  comes from a model that did not see that row. None ranked better than the plain merge. Two ranked
  lower, and one showed no detectable difference.
- **Label-shift correction.** One element of our two-stage method does help. Before merging, the literature
  yields are rescaled onto the laboratory yield distribution (quantile normalisation). Rank correlation
  rises from 0.388 to 0.422. The difference is +0.034 (95% CI +0.018 to +0.050), and it holds in all
  five repeats and under the median label.
- **Enrichment** of top performers shows no detectable change.
- **Status.** This comparison was added during the analysis, so we treat it as exploratory.
- **Scope.** The literature file has no condition columns, so the gas-flow data cannot enter this part.

## 7. Two observations on the data

**Catalysts with a second set of runs.**
- For 28 catalysts, 1,906 conditions appear twice.
- At a given temperature, one set is higher at every condition, by a median factor of 2.3. Both sets
  follow the same pattern across conditions.
- 10 of the 92 top-decile catalysts are among these 28; chance would give 2.8. If the paired readings
  are averaged, 4 remain.

**Excluded runs.**
- Exclusions are more frequent for weaker catalysts. The rank correlation between a catalyst's number
  of missing conditions (out of 135) and its best yield is −0.29.
- Within a catalyst, the excluded conditions do not appear to be its low-yield ones. The difference fell
  below our pre-set threshold for all three CH₄:O₂ ratios.

## 8. Questions and next steps

Both questions concern records that are not in the data file.

1. **For the 28 catalysts with two sets of runs, is there a record of the run date or catalyst batch?**
   About 6 of the 92 top-decile places depend on how the two sets are treated.
2. **Are the records excluded for poor mass balance still available?** They decide whether each
   catalyst's best observed yield is complete. They also decide how a shorter protocol should treat a
   failed run.

**Next steps.**
- Build on the label-shift correction (§6). We will add the remaining elements of the two-stage method
  one at a time and test each.
- Prepare the 10-condition set (§4) for prospective testing, if useful to your group.
- Continue refining accuracy for individual measurements and condition choice.
