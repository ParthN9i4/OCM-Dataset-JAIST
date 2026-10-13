Subject: Condition Data Analysis, Model Results, and Next Steps

Dear Prof. Taniike,

Thank you for sending the dataset with the reaction conditions, and for explaining the mass-balance
exclusions. The data answered both of our open questions. Below is a summary of what we did with it.
The attached work note gives the details.

Data and Label Updates

Data structure. The 27 conditions at each temperature form a full 3 × 3 × 3 design: total flow, Ar
fraction, and CH₄:O₂ ratio, each at three levels. Our previous note described the readings within a
temperature as near-identical. That was incorrect, and we have corrected it.

Training label. We keep the median-based label. It is now understood as a median over up to 27 different
conditions, not over repeated readings. It ranks catalysts as well as the upper quartile and the maximum.

Model Results

Prediction of individual measurements. Adding the gas-flow conditions lowers the RMSE for unseen
catalysts from 2.90 to 2.66, a reduction of 8.1%.

Ranking of catalysts. We scored every catalyst on the same 18 conditions. On that basis, the conditions
do not improve the ranking of catalysts by best yield.

Choice of conditions. For a new catalyst, the model can suggest which conditions to run. Its top three
conditions come 0.15 yield points closer to the catalyst's best yield than the best three conditions for
the same support oxide. The gain is consistent but small.

Reduced measurement set. In a retrospective test, 10 of the 135 conditions reproduce the ranking of
catalysts closely (rank correlation 0.969). Random sets of 10 give 0.917 on average. This estimate is
optimistic, because most catalysts reach their best yield at the conditions chosen.

Literature across preparation methods. Our further two-stage variants did not improve on a plain merge
of laboratory and literature data. One element of the two-stage method helps on its own: rescaling the
literature yields onto the laboratory yield distribution before merging. This raises the rank
correlation from 0.388 to 0.422. Enrichment of top performers does not change detectably. We treat
this result as exploratory.

Questions

We ask the following because the answers are not in the data file and affect how we treat parts of it.

1. For 28 catalysts, many conditions were measured twice, and the two readings differ by a median
   factor of 2.3. Is there a record of the run date or catalyst batch for these catalysts? About 6 of
   the 92 places in the top 10% depend on how these runs are treated.

2. Are the records excluded for poor mass balance still available? Our indirect test could reach only
   about half of them. The records would show whether each catalyst's best observed yield is complete,
   and how a shorter protocol should treat a failed run.

Next Steps

First, we will build on the rescaling result. We will add the remaining elements of the two-stage
method one at a time, with the test for success fixed in advance. Second, we will fix a single
10-condition set and test it on catalysts not used to choose it. If it holds, it could be offered for
prospective testing, should this be useful to your group.

The work note is attached for your review.

Sincerely,

Parth Nagar
Doctoral Research Scholar, Department of Mathematics & Computer Science
Sri Sathya Sai Institute of Higher Learning
nparth@sssihl.edu.in
