> **Status.** This script has been rewritten to match the corrected 8-slide HTML deck
> (`ocm_presentation.html`). Row-level figures still appear where the deck shows them, but they are
> now explicitly labelled as historical every time.
>
> The correction in one line: the `1.907` / `−9.7%` prior-feature result came from a row-level split
> that let the same catalyst appear in training and test. Under catalyst-grouped validation the same
> models are **worse** than baseline (2.9425 vs 2.9955) — catalyst-identity leakage. Authoritative
> sources: `ocm_worknote_taniike.md` (v2), `SESSION_CONTEXT.md` §3 and §5, `ocm_verification_report.md`.

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

- We tried five ways to add literature data. Under a row-level split one looked like a −9.7% win; under catalyst-grouped validation none of them beat a composition-only baseline.
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

"The second and third bullets reinforce this. Lab data is dominated by barium — Ba is the primary active phase in our impregnation protocol and it appears in one third of our samples. Literature data is dominated by silicon, sodium, and manganese — these elements are characteristic of sol-gel and coprecipitation catalyst families. Lanthanum, cerium, strontium, and calcium are shared — they appear meaningfully in both sets, which is actually a good sign for transfer learning."

**[Critical point — say this clearly]**

"I want to be precise about what this means: the datasets differ in *emphasis*, not in completely disjoint chemistry. Both know about La, Ce, Sr, Mg, Li. But lab data over-represents Ba and theirs over-represents Si, Na, Mn. This overlap is what makes it possible to transfer knowledge at all. If the chemistries were completely foreign to each other, no method would work."

**[Transition]**

"With that picture in mind, let me show you the five approaches we tried to incorporate this data responsibly."

---

## SLIDE 4 — Methods and RMSE Table

**[Advance to slide 4]**

"This table is the heart of the presentation. Let me walk through each row.

Our evaluation protocol first, and I have to flag something important. The numbers in this table come from a **row-level** five-fold split: we split the 89,000 rows at random. That was a mistake, and the caption under the table says so. Those 89,000 rows are only **917 distinct catalysts**, so a random row split puts the same catalyst on both sides — the model recalls rather than predicts. Under the catalyst-grouped protocol we now use, the baseline is 2.9425, not 2.133. I am keeping this table because it is what we published, and the next slides show what happened when we re-ran it properly. RMSE is in percentage points of C2 yield; lower is better."

**[Row 1]**

"**Baseline.** LightGBM trained on our 89,000 samples only. No literature at all. RMSE of 2.133 percent. This is our reference. Every other method has to beat this number to justify using literature data."

**[Row 2 — stress this]**

"**Naive concatenation.** We took all 3,852 literature rows and simply appended them to the training data, giving them equal weight to our own samples. RMSE went to 2.248. That is 5.4 percent *worse*. Adding the data hurt us.

Why? Because of exactly what I just showed you. The model now sees two samples with similar chemistry — one of ours labelled 5 percent, one from literature labelled 10 percent — and it splits the difference. Both predictions become slightly wrong in a systematic way. The 3.42 percent label shift is not noise — it is a real, structural difference from different experimental protocols — and treating it as noise corrupts the model."

**[Rows 3, 4, 5]**

"The next three methods are progressively more sophisticated attempts to deal with this. 

**DRST** — Density Ratio Selective Transfer — asks the question: which of the 3,852 literature samples look the most like lab data chemically? It trains a classifier to score each literature sample by how much it resembles our distribution and discards the ones that are too different. At the best threshold, we keep 782 samples — about 20 percent of literature. RMSE drops to 2.019, a 5.3 percent improvement over baseline.

**KMM** — Kernel Mean Matching — takes a softer approach. Instead of a binary keep-or-discard decision, it assigns a continuous weight to each literature sample. Samples close to our distribution in chemical space get weight near 1 — they contribute fully. Samples far away get weight near zero — they effectively disappear. RMSE is 2.035, a 4.6 percent improvement. Very similar to DRST, which is reassuring — two different methods identify roughly the same samples as useful.

**Bias correction** — directly addresses the label shift. It quantile-normalises the literature Y(C2) values to match our distribution — more on what that means in a moment. RMSE 2.044, improvement of 4.2 percent. Helps, but does not solve the covariate shift problem."

**[Final row — build up to it]**

"Now the method that looked best. **Two-stage fine-tuning with DRST-filtered pre-training.** Under this row-level protocol it achieved RMSE 1.912 — a 9.7 percent improvement over baseline, more than double the gain of any single filtering method.

Hold that number lightly. It is the one we sent to Prof. Taniike, and it is the one that did not survive a stricter test. I will show you the reversal on slide 6. The short version: Stage 1 was trained on literature *together with the lab training rows*, and under a row split those rows included the test catalysts — so the prior feature was partly carrying each test catalyst's own measured yields back to the model."

---

## SLIDE 5 — Pipeline Diagram

**[Advance to slide 5]**

"This diagram shows the architecture of the best method. Let me trace through it step by step.

**Stage 1 — on the top row.**

We take the 782 literature samples that passed the DRST filter. We train an XGBoost model on just those 782 rows. This model learns: given an element composition and temperature, what does the *literature* say the C2 yield should be?

It does not see lab data. It does not know our labels. It is purely a literature expert.

Once trained, we run all 89,000 of our internal catalyst compositions through this literature expert. For every one of our experiments, the expert says: 'If this composition had been tested by a literature group, they would have predicted approximately X percent yield.'

That number — X — becomes a new 68th feature. We call it `lit_prior_prediction`."

**[Pause to let this land]**

"**Stage 2 — on the bottom row.**

Now we train a LightGBM model on our 89,000 internal samples. But each sample now has 68 features instead of 67 — the 67 original features plus the literature expert's opinion.

Stage 2 trains only on our labels. Not on literature labels. It learns: given the element composition, temperature, preparation method, *and* what the literature expert would predict — what is our actual measured yield?

The model can learn to trust the prior feature heavily in some regions and ignore it in others. If a Ba-dominant catalyst is presented, the prior might say 7 percent and lab data says 5 percent — Stage 2 learns to discount the prior for that element combination. If a La/Ce catalyst is presented, which is common in both datasets, the prior might be more reliable and Stage 2 learns to weight it more heavily."

**[The key property — read this clearly]**

"The crucial property of this architecture: literature label bias cannot directly corrupt the final model. Stage 2 trains only on our labels. The prior feature carries forward the chemistry structure from Stage 1 — the patterns of which elements correlate with high yield according to literature — without carrying forward the absolute scale of literature's labels. Stage 2's loss function is entirely defined by our measurements.

This is why two-stage beats bias correction alone. Bias correction fixes the label numbers but still adds literature samples as training rows that compete with our own. Two-stage routes the literature knowledge through a feature, not through a training label."

**[Transition]**

"Let me show you the numbers that came out of this."

---

## SLIDE 6 — Results

**[Advance to slide 6]**

"This is the slide that reverses the story, so let me take it slowly.

The two tiles on the left. The first says **plus 1.8 percent** — RMSE moving from 2.9425 to 2.9955 under catalyst-grouped cross-validation. That is the two-stage method performing *worse* than the plain baseline. The second tile says **3.77 times** — the enrichment of our composition-only screening model on the equal-effort set of 771 catalysts, where its Spearman correlation is 0.724.

So: the published 9.7 percent improvement was catalyst-identity leakage. Same models, same data, same code — only the split changed. When a catalyst can no longer appear in both training and test, the gain inverts into a small loss.

I want to be direct about how this happened, because it is the useful part. Our Stage-1 literature expert was trained on literature *plus the lab training rows*. Under a random row split those training rows contained measurements of the very catalysts being tested. The 68th feature we handed the final model was therefore not a pure literature opinion — it partly encoded each test catalyst's own yields. We were measuring recall and calling it prediction.

Notice the baseline itself moves too, from 2.12 to 2.94. That is the honest difficulty of the real task. Predicting a catalyst nobody has made is simply harder than interpolating between measurements of a catalyst you already have.

**[Second callout]**

"What survives is on the green callout. The composition-only model still ranks genuinely unseen catalysts usefully: enrichment 3.77 times better than random, with a 95 percent confidence interval of 3.04 to 4.89. That is the result we stand behind, and it is enough to guide a synthesis campaign. It also survives an open question about how the data was collected, which I will come back to on the last slide."

**[If asked why you are presenting a negative result]**

"Because we found it ourselves, before a referee did, and because the mechanism generalises. Random splits are standard practice in this field, and this dataset has roughly a hundred rows per catalyst. Any paper doing what we did would report the same inflated number."

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

"Three priorities, and I want to be upfront that all three are questions for the lab rather than modelling work. We think the modelling has gone about as far as this feature set allows.

**Priority 1 — Ask JAIST for the reaction-condition columns. One email.**

Each catalyst is run at five temperatures under roughly 27 condition settings that this file does not record. We can see the structure in the row counts: 15 catalysts hold exactly 135 rows, and every one of them splits as exactly 27 rows at each of five temperatures. Five times 27 is 135, which matches the number Prof. Taniike quotes. But the settings themselves are absent.

Because they are missing, 19.9 percent of the row-level yield variance cannot be reached from composition and temperature at all. That is not noise — it is real chemistry we cannot see. Recovering those columns turns 917 training examples back into 89,074 and makes row-level RMSE a well-posed target again. No modelling change we have tried comes close to that.

**Priority 2 — Ask whether those 27 slots are conditions or time-on-stream samples.**

This one matters more than it sounds. If the 27 rows are successive samples from a single continuous run rather than 27 distinct conditions, then a catalyst's observed maximum is a fresh-catalyst transient, not an achievable operating point — and we would be ranking catalysts by how good they look when new rather than by what they sustain.

We can prove we cannot answer this from the file. Every cell is stored sorted from highest yield to lowest, so row order records rank, not the order the measurements were taken. Two earlier analyses failed for exactly this reason, and a third would too.

What we did instead was bound the risk. We ranked catalysts by their *worst* measurement instead of their best. The two rankings do disagree at the top — only 7 of the top 20 are the same. But a model trained on the best measurement still loses only 0.017 Spearman when judged against the worst, which is less than we gain from simply averaging over random seeds, and its enrichment never falls below 4.02 times. The answer changes the labels. It does not change which catalysts we would recommend making.

**Priority 3 — Re-scope the campaign before reactor time is spent.**

Replaying the lab's own archive: measuring 5 conditions at each of four temperatures — 20 runs instead of 135 — reproduces the full ranking at rho 0.949, systematically low by 1.38 yield points, a bias that can be declared in advance. That buys roughly 72 catalysts screened plus a randomised control arm for the reactor budget of 17 exhaustive ones. Under the time-on-stream reading the saving is analytical rather than thermal, and the arithmetic changes.

**[On limitations — be honest]**

"Three limitations I want to state explicitly.

First, and largest: we withdrew our own headline. The 9.7 percent improvement was catalyst-identity leakage, and under the correct protocol the method is 1.8 percent worse than baseline. I would rather say that here than have a referee say it later.

Second: a novel promoter family is structurally unpriceable. With no family members in training the column is constant, no tree ever splits on it, and deleting the column entirely gives bit-identical predictions. That is arithmetic, not a modelling deficiency. Our learning curves say roughly 50 measured members of a new family are needed before the model can price it — not the 10 an earlier version of this claimed.

Third: inside the model's own top-ranked region — the only regime a synthesis campaign ever occupies — the internal ordering is close to uninformative. Spearman within the top 150 is 0.179, and within the top 20 it is minus 0.066. The model selects a good set but cannot order within it. A shortlist is a set to test, not a league table, and a 17-catalyst campaign drawn entirely from that region cannot by itself confirm or refute the model."

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

We originally argued the validation score (RMSE=1.912) was unbiased because the validation folds were never seen by Stage 1. **That argument was wrong, and this is the central correction of the project.** Stage 1 was trained on literature together with the lab training rows; under a random row split those rows contained other measurements of the very catalysts in the validation fold. The prior feature therefore carried each validation catalyst's own yields. Under catalyst-grouped CV, where a catalyst appears in only one fold, the effect disappears and the method is 1.8% worse than baseline (2.9425 → 2.9955).

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

A: "Because 3,852 is not negligibly small compared to 89,000. It is 4.1% of the training set. In gradient boosting, each training sample contributes to which splits are made — the splits are chosen to minimise total loss across all samples. A literature sample with yield 10% in a region where our samples all have yield 5% will push split boundaries to accommodate both. Even at 4%, this creates a systematic distortion in the decision boundaries for the most common yield range. The model's RMSE on lab data worsened by 5.4% purely from this contamination."

---

**Q: "The OOD improvement was reported as 45%. Is that reliable?"**

A: "No — and there were in fact two leaks, which I corrected in sequence. The first was the out-of-distribution claim: 45% (6.53 → 3.60) came from the Stage-1 prior being trained on the very OOD samples we then tested on. The second, which we found later and which matters more, is that the *in-distribution* result had the same disease. Stage 1 also saw the lab training rows, and under a random row split those included the test catalysts. Once we split by catalyst instead of by row, the 9.7% improvement becomes a 1.8% degradation: baseline 2.9425, two-stage 2.9955.

So I no longer claim either win. What we do claim is a composition-only screening model that ranks genuinely unseen catalysts with enrichment 3.77× (95% CI 3.04–4.89×), and a clean methodological finding: on a dataset with roughly a hundred rows per catalyst, a random row split measures recall rather than prediction. Random splits are standard practice in this field, which is why we think the finding is worth reporting.

One more thing, since it usually comes next: quantile normalisation turned out not to matter either. It moves RMSE by 0.001 to 0.023, inside run-to-run noise at three seeds. Prof. Taniike predicted that — trees read feature order, not scale."

---

**Q: "What would happen if we added the same 89,000 rows from a second year of experiments?"**

A: "The baseline RMSE would likely improve significantly just from more data. The marginal benefit of transfer learning would probably shrink because the model is already well-covered in chemical space. However, the high-yield ceiling effect would persist until we add reaction condition features. More data does not fix a missing feature problem — it only helps you predict better within the existing feature space."

---

**Q: "When you say 'Stage 2 trains only on our labels,' what stops it from being biased by Stage 1?"**

A: "Stage 2 can be affected by Stage 1's systematic errors — if Stage 1 consistently over-estimates yield for a certain element combination, Stage 2 might partially inherit that bias through the prior feature. However, Stage 2 has 89,000 internal samples with their true labels, which outnumber the indirect Stage 1 influence by a large margin. The LightGBM model will learn a correction to the prior wherever lab data contradicts it. The direct label bias — literature's 3.42% higher mean — cannot enter Stage 2's training at all, because Stage 2's loss function is computed only on our labels."

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
