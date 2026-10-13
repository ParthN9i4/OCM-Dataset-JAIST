Subject: Condition Data Analysis, Model Results, and Next Steps

Dear Prof. Taniike,

Thank you for sending the dataset with the reaction conditions, and for explaining the mass-balance
exclusions. Your reply answered both of our open questions. Below is a summary of what we did with the
new data. The attached work note gives the details.

Validation Updates

Data structure. The 27 conditions at each temperature form a full 3 × 3 × 3 design: total flow, Ar
fraction, and CH₄:O₂ ratio, each at three levels. Our previous note described the readings within a
temperature as near-identical. That was incorrect, and we have corrected it. We keep the median-based
training label. The reason is now that it rests on many readings rather than on a single high reading.

Model Results

Prediction of individual measurements. Adding the gas-flow conditions lowers the RMSE for unseen
catalysts from 2.90 to 2.66, a reduction of 8.1%.

Ranking of catalysts. Under a fair comparison, in which every catalyst is scored on the same 18
conditions, the conditions do not improve the ranking of catalysts by best yield.

Choice of conditions. For a new catalyst, the model can suggest which conditions to run. Its top three
conditions come 0.15 yield points closer to the catalyst's best yield than the best three conditions for
the same support oxide. The gain is consistent but small.

Reduced measurement set. In a retrospective test, 10 of the 135 conditions reproduce the full ranking of
catalysts closely (rank correlation 0.969). Ten random conditions reach at most 0.925.

Literature across preparation methods. Our further two-stage designs did not improve on a plain merge.
However, one element of the two-stage method helps on its own. Rescaling the literature yields onto the
laboratory yield distribution before merging raises the rank correlation from 0.388 to 0.422. We treat
this result as exploratory.

Questions

We ask the following because the answers are not in the data file and affect how we treat parts of it.

1. For 28 catalysts, many conditions were measured twice, and one set of runs is consistently higher
   than the other. Is there a record of the run date or catalyst batch for these catalysts? About 6 of
   the 92 catalysts with the highest observed yields depend on how these runs are treated.

2. Are the records excluded for poor mass balance still available? They would show whether each
   catalyst's best observed yield is complete. They would also show how a shorter protocol should treat
   a failed run.

Next Steps

First, we will build on the rescaling result. We will add the remaining elements of the two-stage
method one at a time and test each. Second, we will prepare the 10-condition set for prospective
testing, if this would be useful to your group. Third, we will continue to refine prediction accuracy
for individual measurements and for the choice of conditions.

The work note is attached for your review.

Sincerely,

Parth Nagar
Doctoral Research Scholar, Department of Mathematics & Computer Science
Sri Sathya Sai Institute of Higher Learning
nparth@sssihl.edu.in
