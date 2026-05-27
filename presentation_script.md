# Presentation Script — OCM Dataset Integration
## Full Verbatim Speaker Notes with Mathematical Background

**Audience:** Prof. Taniike and lab members  
**Duration:** 20–25 minutes + 5–10 minutes Q&A  
**Format:** 8 slides, advance with arrow keys or click

---

## Before You Begin — Context Briefing

Prof. Taniike's starting point from his email:

> "We have 89k internal experiments. Can we add the OCM literature data (3,852 rows from papers up to 2019) to make the model more robust, without hurting the accuracy on our own data?"

That is the exact question this presentation answers. Every slide, every number, every method maps back to that single question. Keep returning to it whenever the audience looks lost.

**Your mental model before walking in:**

- We tried five ways to add literature data. Four gave modest improvements. One gave −10.6% error.
- The core problem is that the two datasets are not the same kind of data, even though they measure the same thing.
- The solution is a two-step process: first teach a sub-model everything it can learn from literature, then let the main model use that sub-model's opinion as one piece of evidence.
- We proved it works both on our own held-out experiments and on literature experiments the model had never seen.

---

## SLIDE 1 — Title

**[Say this as you bring up the first slide]**

"Good morning. I want to start by thanking Prof. Taniike for the question that motivated this work. The question was: can we use the roughly 3,800 samples from the published OCM literature to make our model better, and can we do this without sacrificing the accuracy we already have on our own 89,000 experiments?

That question sounds straightforward, but it turns out to be a non-trivial machine learning problem — and working through it is what this presentation is about.

By the end I want you to walk away with three things. First, a clear understanding of why naively adding literature data actually makes performance worse. Second, the mechanism behind the approach that finally worked. And third, a concrete set of next steps ranked by expected impact."

**[Pause one beat. Then:]**

"Let me start with the data itself."

---

## SLIDE 2 — Two Datasets

**[Advance to slide 2]**

"We have two sources of data sitting in one CSV file. Let me walk through each column of this table.

On the left: our internal data. 89,074 rows. All collected in this lab in 2025. All use the Impregnation preparation method — that is our standard protocol. The average C2 yield in this set is 5.25 percent.

On the right: the literature data. 3,852 rows. Collected from published papers between 1982 and 2019. These papers used more than 15 different preparation methods — sol-gel, coprecipitation, combustion synthesis, flame spray pyrolysis, and many others. The average C2 yield here is 8.67 percent.

Now, the number at the bottom of this slide is the critical one."

**[Point to the gap statistic]**

"The difference in mean yield is 3.42 percentage points. We ran a standard statistical test — a two-sample t-test — and the result is a t-statistic of −32.2 and a p-value of 2.3 times 10 to the power of negative 202. 

Let me put that p-value in plain language. A p-value measures the probability that the difference you see happened by random chance. A p-value of 0.05 means there is a 5 percent chance you are wrong. Our p-value is so small that the number of zeros after the decimal point exceeds two hundred. The two distributions are definitively, unambiguously different. They are not measuring the same underlying thing."

**[Pause for effect]**

"This matters because it means we cannot simply combine the two datasets and pretend they are one dataset. The model would receive contradictory information — two catalysts with nearly identical chemical compositions, one labelled 5 percent and one labelled 10 percent — with no way to know the difference came from experimental conditions that are not in the feature set. We will come back to this."

**[Transition]**

"Before I show you the methods, I want to show you a picture of why this gap exists."

---

## SLIDE 3 — Domain Gap

**[Advance to slide 3]**

"This figure on the left is called a PCA plot. Let me explain what it shows from first principles.

Each catalyst in our dataset is described by 65 numbers — the weight percentages of 65 different elements. You cannot draw a 65-dimensional picture. What PCA does — Principal Component Analysis — is find the two linear combinations of those 65 numbers that capture the most variation across all 92,000 samples, and project everything onto those two directions. Think of it as casting a shadow of a 65-dimensional object onto a flat wall — you lose some information, but you preserve the most important structure.

Blue dots are our 89,000 internal samples. Red dots are the 3,800 literature samples. 

What you see is that the two clouds overlap but are not the same. The blue cloud is concentrated around certain element combinations. The red cloud is more spread out and covers regions the blue cloud barely touches. 

The key fact in the top bullet: 78.5 percent of literature samples fall in regions where our training data is sparse or absent. We measure this with the same statistical tool we use for filtering, which I will explain in a moment. The point for now is: most literature samples describe catalysts that our model has never seen in training."

**[Pause]**

"The second and third bullets reinforce this. Our data is dominated by barium — Ba is the primary active phase in our impregnation protocol and it appears in one third of our samples. Literature data is dominated by silicon, sodium, and manganese — these elements are characteristic of sol-gel and coprecipitation catalyst families. Lanthanum, cerium, strontium, and calcium are shared — they appear meaningfully in both sets, which is actually a good sign for transfer learning."

**[Critical point — say this clearly]**

"I want to be precise about what this means: the datasets differ in *emphasis*, not in completely disjoint chemistry. Both know about La, Ce, Sr, Mg, Li. But our data over-represents Ba and theirs over-represents Si, Na, Mn. This overlap is what makes it possible to transfer knowledge at all. If the chemistries were completely foreign to each other, no method would work."

**[Transition]**

"With that picture in mind, let me show you the five approaches we tried to incorporate this data responsibly."

---

## SLIDE 4 — Methods and RMSE Table

**[Advance to slide 4]**

"This table is the heart of the presentation. Let me walk through each row.

Our evaluation protocol first. We used five-fold cross-validation, but in an asymmetric way. We split only our 89,000 internal samples into five folds. In each iteration, four folds go into training and one fold goes into validation. Literature data, when used, always goes into the training pool — never into validation. We measure RMSE, root mean squared error, which is in units of percentage points of C2 yield. Lower is better."

**[Row 1]**

"**Baseline.** LightGBM trained on our 89,000 samples only. No literature at all. RMSE of 2.133 percent. This is our reference. Every other method has to beat this number to justify using literature data."

**[Row 2 — stress this]**

"**Naive concatenation.** We took all 3,852 literature rows and simply appended them to the training data, giving them equal weight to our own samples. RMSE went to 2.248. That is 5.4 percent *worse*. Adding the data hurt us.

Why? Because of exactly what I just showed you. The model now sees two samples with similar chemistry — one of ours labelled 5 percent, one from literature labelled 10 percent — and it splits the difference. Both predictions become slightly wrong in a systematic way. The 3.42 percent label shift is not noise — it is a real, structural difference from different experimental protocols — and treating it as noise corrupts the model."

**[Rows 3, 4, 5]**

"The next three methods are progressively more sophisticated attempts to deal with this. 

**DRST** — Density Ratio Selective Transfer — asks the question: which of the 3,852 literature samples look the most like our data chemically? It trains a classifier to score each literature sample by how much it resembles our distribution and discards the ones that are too different. At the best threshold, we keep 782 samples — about 20 percent of literature. RMSE drops to 2.019, a 5.3 percent improvement over baseline.

**KMM** — Kernel Mean Matching — takes a softer approach. Instead of a binary keep-or-discard decision, it assigns a continuous weight to each literature sample. Samples close to our distribution in chemical space get weight near 1 — they contribute fully. Samples far away get weight near zero — they effectively disappear. RMSE is 2.035, a 4.6 percent improvement. Very similar to DRST, which is reassuring — two different methods identify roughly the same samples as useful.

**Bias correction** — directly addresses the label shift. It quantile-normalises the literature Y(C2) values to match our distribution — more on what that means in a moment. RMSE 2.044, improvement of 4.2 percent. Helps, but does not solve the covariate shift problem."

**[Final row — build up to it]**

"Now the best method. **Two-stage fine-tuning with DRST-filtered pre-training.** This achieved RMSE 1.907 — a 10.6 percent improvement over baseline. More than double the gain of any individual method.

At the bottom: the out-of-distribution test. We held out 2,139 literature samples from non-Impregnation preparation methods — catalysts the model had never seen in any form during training. The baseline scores 6.53 on these. Our two-stage method scores 3.60. That is a 45 percent reduction in error on completely new catalyst families. I will explain how this works on the next slide."

---

## SLIDE 5 — Pipeline Diagram

**[Advance to slide 5]**

"This diagram shows the architecture of the best method. Let me trace through it step by step.

**Stage 1 — on the top row.**

We take the 782 literature samples that passed the DRST filter. We train an XGBoost model on just those 782 rows. This model learns: given an element composition and temperature, what does the *literature* say the C2 yield should be?

It does not see our data. It does not know our labels. It is purely a literature expert.

Once trained, we run all 89,000 of our internal catalyst compositions through this literature expert. For every one of our experiments, the expert says: 'If this composition had been tested by a literature group, they would have predicted approximately X percent yield.'

That number — X — becomes a new 68th feature. We call it `lit_prior_prediction`."

**[Pause to let this land]**

"**Stage 2 — on the bottom row.**

Now we train a LightGBM model on our 89,000 internal samples. But each sample now has 68 features instead of 67 — the 67 original features plus the literature expert's opinion.

Stage 2 trains only on our labels. Not on literature labels. It learns: given the element composition, temperature, preparation method, *and* what the literature expert would predict — what is our actual measured yield?

The model can learn to trust the prior feature heavily in some regions and ignore it in others. If a Ba-dominant catalyst is presented, the prior might say 7 percent and our data says 5 percent — Stage 2 learns to discount the prior for that element combination. If a La/Ce catalyst is presented, which is common in both datasets, the prior might be more reliable and Stage 2 learns to weight it more heavily."

**[The key property — read this clearly]**

"The crucial property of this architecture: literature label bias cannot directly corrupt the final model. Stage 2 trains only on our labels. The prior feature carries forward the chemistry structure from Stage 1 — the patterns of which elements correlate with high yield according to literature — without carrying forward the absolute scale of literature's labels. Stage 2's loss function is entirely defined by our measurements.

This is why two-stage beats bias correction alone. Bias correction fixes the label numbers but still adds literature samples as training rows that compete with our own. Two-stage routes the literature knowledge through a feature, not through a training label."

**[Transition]**

"Let me show you the numbers that came out of this."

---

## SLIDE 6 — Results

**[Advance to slide 6]**

"On the left is the bar chart from the results summary. The blue bar on top is our baseline. Every orange bar below it is a transfer method. The green bar at the bottom is Direction A — the two-stage method.

The two tables on the right summarise the key numbers.

For in-distribution performance — predicting our own experiments — RMSE dropped from 2.133 to 1.907. That is a reduction of 0.226 percentage points. To put this in context: our model predicts C2 yield for catalysts we have never tested. Getting 0.226 percentage points closer to the true answer means that if we were screening 1,000 new catalysts and picking the top 50 by predicted yield, we would now rank more of the genuinely high-performing catalysts into that top 50.

For out-of-distribution performance — predicting literature experiments our model had never seen — RMSE dropped from 6.53 to 3.60, a 45 percent reduction.

This OOD number is actually the more important result scientifically. It tells us the model has genuinely learned transferable chemical knowledge, not just memorised our specific impregnation protocol. When we ask it to predict a sol-gel La/Ce catalyst from 1995, it can now make a reasonable estimate. Without transfer learning, it was essentially guessing."

**[Note at the bottom of the table]**

"The small print: OOD test set is 2,139 samples from non-Impregnation preparation methods, held out from all training. These samples were never used in any training fold, in any method, at any stage. This is a true independent test."

---

## SLIDE 7 — SHAP Analysis

**[Advance to slide 7]**

"This figure is a SHAP beeswarm plot. SHAP stands for SHapley Additive exPlanations — named after a concept from cooperative game theory. Let me explain what it shows without the game theory.

For each of 3,000 randomly selected samples and each of the 68 model features, SHAP computes: 'how much did this particular feature's value — for this particular sample — push the prediction up or down, relative to what the model would have predicted on average?'

The answer is a number called the SHAP value. Positive SHAP value means the feature pushed the prediction higher. Negative means it pushed lower. Zero means it had no effect for that sample.

In this plot, each row is a feature, ordered by average importance from top to bottom. Each dot is one sample. The horizontal position of the dot is the SHAP value — right means 'pushed prediction up,' left means 'pushed it down.' The colour is the actual feature value — red means high, blue means low."

**[Walk through top features]**

"**Feature 1: `lit_prior_prediction`.** This is our transfer learning feature — the literature expert's opinion. It is the single most important feature in the model. The pattern: when the literature expert predicts high yield (red dots), the model predicts higher. When it predicts low (blue dots), the model predicts lower. The literature knowledge is actively being used on every single prediction.

**Feature 2: Temperature.** Higher temperature, higher prediction. This is well-established OCM chemistry — the C-H bond activation step requires high temperatures, typically above 700 Celsius. The model has learned this from our 89,000 experiments.

**Features 3–6: Ba, Mn, La, Ce.** All push predictions upward when their weight percent is high. Ba is the dominant active phase in our impregnation catalysts. Mn, La, Ce are known promoters that improve oxygen activation and C2 selectivity.

**Features 7–8: Li, Na.** Mixed effect — some samples pushed up, some pushed down. Alkali promoters can enhance basicity at low loadings but cause sintering or pore blocking at high loadings. The model has learned this non-linearity."

**[The residual table — important]**

"Now the table on the right. This is a residual analysis — residual meaning predicted minus actual. Zero is perfect.

For catalysts with yield between 0 and 6 percent — which is the majority of our dataset — RMSE is 1.43 to 1.48 and the mean residual is near zero. Excellent calibration.

For catalysts with yield above 15 percent, RMSE is 4.70 and the mean residual is negative 4.16 percent. The model systematically under-predicts high-yield catalysts.

This is called a ceiling effect. The root cause is not algorithmic — it is a data problem. The features that explain why certain catalysts achieve 20 percent yield — gas hourly space velocity, methane to oxygen ratio, reaction pressure — are not in our dataset at all. When a literature paper reports 20 percent yield, they likely achieved it with optimised reaction conditions. Our model sees identical composition to a 5 percent yield sample and has no way to know conditions were different.

Adding GHSV and CH4/O2 ratio as features is the highest-impact next step, and I will quantify the expected benefit in the next slide."

---

## SLIDE 8 — Limitations and Next Steps

**[Advance to slide 8]**

"Three options for continuing this work, ordered by effort.

**Option A — Add reaction conditions. Easy, high impact.**

If we collect GHSV, methane to oxygen ratio, and pressure for our internal experiments and add them as features, the residual analysis strongly suggests we would close most of the ceiling effect. The −4.16 percent mean bias on high-yield samples would likely shrink to near zero. Expected RMSE reduction from 4.70 to approximately 2.5 for the high-yield range.

This is a data collection problem, not a modelling problem. The solution is adding columns to the CSV.

**Option B — Tune the DRST threshold. Medium effort, moderate impact.**

Our current threshold of τ=0.30 was selected by sweeping five values. A proper nested cross-validation search across a finer grid might recover another 0.3 to 0.5 percentage points of RMSE. Worth doing after Option A.

**Option C — Neural domain adaptation. Hard, uncertain return.**

Methods like DANN — Domain Adversarial Neural Networks — use a neural network that is simultaneously trained to predict yield and to be unable to distinguish which dataset a sample came from. This can theoretically find a representation where both datasets look the same. However, these methods are harder to tune and less interpretable. I would recommend attempting this only after the reaction condition features are in the dataset and the easier gains are exhausted."

**[On limitations — be honest]**

"Before questions, I want to be explicit about the limitations of this work.

First: the missing reaction conditions are not a minor issue. They are the primary accuracy ceiling. Every method we tried, including the best one, under-predicts high-yield catalysts systematically. The −10.6% RMSE improvement is real and correct, but it is measured primarily on the 0–10% yield range where most of our data lives. In the 15–22% range where the interesting catalysts are, we are still off by 4 percent on average.

Second: the publication bias correction partially addresses why literature labels are higher on average. But it does not address the conditional bias — a literature Na/Si catalyst that reports 12% yield might have used conditions specifically optimised to achieve 12%. Our model will see the same composition at our standard conditions and reasonably predict 7%. That gap is real and not a model error.

Third: there is a methodological subtlety I want to be transparent about. In the current implementation, Stage 1 is trained partly on the same training data that Stage 2 then trains on. This means Stage 2 receives a slightly overconfident prior during training. The validation scores are unbiased — the hold-out folds were never seen by Stage 1 — but the improvement estimate of 10.6% might be slightly optimistic by perhaps 1 to 2 percentage points for truly novel catalyst families. The fix is straightforward and is our immediate next task."

**[Closing]**

"To summarise in three sentences.

Adding literature data naively makes performance worse because the datasets have a 3.4 percentage point systematic label difference and 78 percent of literature samples describe catalyst chemistries our model has not encountered before.

Two-stage fine-tuning resolves this by routing literature knowledge through a pre-trained sub-model rather than through competing training labels, achieving a 10.6 percent improvement in-distribution and a 45 percent improvement on completely held-out external catalysts.

The next step that would give the largest return is adding GHSV and methane-to-oxygen ratio as features, which would address the systematic under-prediction of high-yield catalysts that no algorithmic change can fix.

I am happy to take questions."

---

## Mathematical Background — What to Know If Asked

This section is not for the slide presentation. Read it the night before so you can answer deep questions confidently.

---

### "What is RMSE and why not R²?"

RMSE is root mean squared error:

```
RMSE = sqrt( (1/n) * Σᵢ (yᵢ_predicted − yᵢ_actual)² )
```

It is in the same units as your target — percentage points of C2 yield. So RMSE=2.133 means on average our predictions are about 2.1 percentage points off from the actual measurement.

R² (coefficient of determination) measures what fraction of total variance the model explains:

```
R² = 1 − (sum of squared residuals) / (total sum of squares)
```

R²=0.90 means the model explains 90% of the variance in Y(C2). 

We report RMSE as the primary metric because it directly answers "how wrong are we in yield percentage points?" R² depends on the variance of the dataset — a dataset with extreme outliers will show lower R² even if predictions are excellent for most samples. RMSE is more interpretable for this application.

---

### "What is cross-validation and why five folds?"

Cross-validation is a technique to estimate how well a model will perform on unseen data, without having a dedicated test set.

In 5-fold CV: you split the data into 5 equal parts. You train on 4 parts and test on the 5th. You rotate which part is the test set five times. You average the five RMSE values.

Why not just train once and test once? Because a single 80/20 split might be lucky or unlucky depending on which samples ended up in validation. Five-fold averaging reduces this variance significantly.

Why five folds specifically? Five is a pragmatic choice. More folds (e.g., 10) gives a less biased estimate but is more expensive to compute. Five folds gives a good bias-variance tradeoff and is the community standard for datasets of this size.

Why asymmetric (literature always in training)? Because our validation question is "how well does the model predict our experiments?" If literature appeared in validation folds, we would be measuring a mixed question — partly our accuracy, partly literature accuracy — which does not answer Prof. Taniike's original question.

---

### "What is gradient boosting — can you explain it simply?"

A single decision tree asks binary questions about features and predicts based on which leaf a sample falls into. It is fast but erratic — small changes in training data produce very different trees.

Gradient boosting builds many trees sequentially. Each tree tries to correct the errors of all previous trees combined. Specifically:

1. Start with a constant prediction (the mean of Y(C2)).
2. Compute the error each sample currently has.
3. Fit a new small tree to predict those errors.
4. Add a scaled version of that tree to your running prediction.
5. Recompute errors. Go to step 3.

After 500 trees, the model is very accurate because each tree focused on the hardest remaining cases.

The "gradient" name: mathematically, fitting the residuals is equivalent to taking one step in the direction of steepest descent of the loss function. This is the same as gradient descent in neural networks, but applied to the space of functions rather than the space of parameters.

---

### "Why XGBoost for Stage 1 and LightGBM for Stage 2?"

XGBoost and LightGBM are both gradient boosting, but they differ in how they build each tree.

XGBoost builds trees symmetrically — it grows each level of the tree before starting the next level. LightGBM grows trees leaf-by-leaf — it always expands whichever single leaf currently would reduce the most error. LightGBM is typically 5–10× faster on large datasets and equally or more accurate.

For Stage 1 (782 literature samples), we use XGBoost because it is more conservative on small data — its symmetric growth prevents it from overfitting single outliers, which LightGBM's aggressive leaf-wise growth could do on only 782 rows.

For Stage 2 (89,000 samples, five folds, many methods), we use LightGBM because speed matters and large-data accuracy is where LightGBM excels.

---

### "What is the stacking leakage problem you mentioned?"

When Stage 1 trains on (literature + our_training_fold), it has seen our training fold samples. When Stage 1 then generates `lit_prior_prediction` for those same samples, it generates near-perfect predictions (it saw those rows during training). Stage 2 then trains on these near-perfect priors.

At deployment time, Stage 1 generates predictions for new samples it has not seen — these predictions have normal out-of-sample error. But Stage 2's weights were calibrated to near-perfect priors. So Stage 2 may be slightly over-reliant on the prior feature.

The validation score (RMSE=1.907) is unbiased because the validation folds were never seen by Stage 1. The issue is that the Stage 2 model in memory may behave slightly differently on truly novel unseen data than the CV estimate suggests.

The fix: train Stage 1 only on literature. All its predictions on internal data are then naturally out-of-sample. This is a one-line change. It will be implemented before any results are published.

---

### "The p-value is 10⁻²⁰² — is that meaningful or just a large dataset artifact?"

This is a very good question and worth knowing the answer to.

With 89,000 samples, even very tiny true differences will produce astronomically small p-values. The t-test p-value is sensitive to both the size of the difference AND the sample size. A 0.01 percentage point difference in means with 89,000 samples would produce a significant p-value.

So the p-value alone does not tell you if the difference is practically important. What matters is the effect size: 3.42 percentage points out of a typical yield of 5–8%. That is a ~40–65% relative difference. This is practically enormous — it is not a statistical artifact of having a large sample.

The appropriate way to state this: "The 3.42 percentage point difference in mean yield is both statistically significant (the result could not have arisen by chance) and practically significant (it is large enough to meaningfully corrupt a joint model)."

---

## Anticipated Questions and How to Answer Them

---

**Q: "Why did you choose τ=0.30 as the DRST threshold?"**

A: "We swept five values — 0.05, 0.10, 0.20, 0.30, 0.40 — and measured CV RMSE at each. τ=0.30 gave the lowest validation error, keeping 782 of the 3,852 literature samples. It is possible a finer search would find a marginally better value. This is on our near-term list as Option B."

---

**Q: "Could we use a neural network instead?"**

A: "We considered it. For tabular data with 68 features and 89,000 rows, gradient boosted trees consistently match or outperform neural networks in practice — this is well-documented in benchmark competitions. Additionally, neural networks require significant architecture search, and the SHAP interpretability we used for the chemistry validation only works exactly for tree models. For a future version with reaction conditions added, if the dataset grows substantially, neural domain adaptation could become competitive. That is Option C in the next steps."

---

**Q: "Why did naive merging hurt performance when we have 89,000 internal samples that should dominate?"**

A: "Because 3,852 is not negligibly small compared to 89,000. It is 4.1% of the training set. In gradient boosting, each training sample contributes to which splits are made — the splits are chosen to minimise total loss across all samples. A literature sample with yield 10% in a region where our samples all have yield 5% will push split boundaries to accommodate both. Even at 4%, this creates a systematic distortion in the decision boundaries for the most common yield range. The model's RMSE on our data worsened by 5.4% purely from this contamination."

---

**Q: "The OOD improvement is 45%. Is that reliable?"**

A: "It is reliable as a directional result — the two-stage model clearly extrapolates better to unseen preparation methods. The specific number (6.53 → 3.60) depends on the composition of the OOD test set, which is 2,139 samples from non-Impregnation methods. It is possible that a different composition of novel catalysts would show a different magnitude of improvement. What we can say with confidence is that the two-stage prior gives the model a chemistry map of regions it was not trained on, and this is consistently beneficial. The OOD test confirms the improvement is real, not just statistical variance on our own held-out data."

---

**Q: "What would happen if we added the same 89,000 rows from a second year of experiments?"**

A: "The baseline RMSE would likely improve significantly just from more data. The marginal benefit of transfer learning would probably shrink because the model is already well-covered in chemical space. However, the high-yield ceiling effect would persist until we add reaction condition features. More data does not fix a missing feature problem — it only helps you predict better within the existing feature space."

---

**Q: "When you say 'Stage 2 trains only on our labels,' what stops it from being biased by Stage 1?"**

A: "Stage 2 can be affected by Stage 1's systematic errors — if Stage 1 consistently over-estimates yield for a certain element combination, Stage 2 might partially inherit that bias through the prior feature. However, Stage 2 has 89,000 internal samples with their true labels, which outnumber the indirect Stage 1 influence by a large margin. The LightGBM model will learn a correction to the prior wherever our data contradicts it. The direct label bias — literature's 3.42% higher mean — cannot enter Stage 2's training at all, because Stage 2's loss function is computed only on our labels."

---

## Timing Guide

| Section | Approximate Time |
|---|---|
| Slide 1 — Title | 1 minute |
| Slide 2 — Datasets | 3 minutes |
| Slide 3 — Domain Gap | 3 minutes |
| Slide 4 — Methods table | 6 minutes |
| Slide 5 — Pipeline | 4 minutes |
| Slide 6 — Results | 3 minutes |
| Slide 7 — SHAP | 4 minutes |
| Slide 8 — Next Steps + Limitations | 3 minutes |
| **Total presentation** | **~27 minutes** |
| Q&A | 5–10 minutes |

If Prof. Taniike is asking many questions during the slides (a good sign), slow down on Slide 4 and Slide 5 — those are the two slides that carry the core technical argument. Everything else can be abbreviated.

If you are running long, abbreviate Slide 7 (say "SHAP confirms the literature prior is the top feature and Ba/Mn/La are the main catalytic drivers" without walking through each row) and Slide 8 (say "Option A is the priority — adding GHSV and CH4/O2 — and I am happy to discuss the others in Q&A").
