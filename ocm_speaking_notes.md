# OCM Walkthrough — Personal Speaking Notes

Personal reference only. Not included in `ocm_walkthrough.ipynb`.

---

## Chapter 1 — The Problem

"So the starting point for this whole project was a question Prof. Taniike asked — we have this big public dataset from OCM literature, can we use it to make our model more robust? On the surface it seems straightforward, just add more data. But when I started looking at the numbers, there are actually three separate problems. The first is that the average yield in literature is 8.67%, and ours is 5.25%. That gap is not random — it's systematic, and it's because literature only publishes the experiments that worked, and because conditions like flow rate and gas ratio are set optimally. The second problem is that when you look at the chemistry — the actual elements being used — 78% of literature samples describe catalysts that look nothing like what we test in our lab. And the third is the label shift from the measurement conventions being different. So the rest of this notebook is basically our attempt to solve these three problems one at a time."

---

## Chapter 2 — Setup

"This cell is mostly housekeeping, but there are two lines I want to highlight. First, the LabelEncoder — the preparation method is a text column, things like 'Impregnation' or 'Sol-gel'. Tree models can't process text, so we convert each method name to an integer. Second — and this is the one that tripped me up initially — we call fit_transform on our data, but only transform on literature. Why? Because the scaler learns the mean and standard deviation from our data. That becomes our reference ruler. If I re-fit the scaler on literature separately, then the same physical measurement — say, Ba weight of 2% — would get a different number depending on which dataset it came from. That would break every distance-based calculation downstream. So: fit on ours, apply to both."

---

## Chapter 3 — How We Measure Success

"So before we run any method, I want to explain how we measure success, because the choice here matters a lot. We use 5-fold cross-validation, but in an asymmetric way. We split only our 89,000 samples into five groups. Four groups go into training, one goes into validation. Literature, when we use it, always goes into training — it never appears in the test set. The reason is simple: Prof. Taniike asked 'can we improve accuracy on our experiments', so the validation set has to be our experiments. If I put literature samples in validation, I'd be measuring something different and the numbers wouldn't answer the actual question. RMSE is in units of percentage points of yield — so an RMSE of 2.1 means we're typically off by about 2 percentage points."

---

## Chapter 4 — Baseline + Why Naive Merging Fails

"So the obvious first thing I tried was just: append all 3,852 literature rows to the training set and retrain. More data should help, right? It didn't. RMSE went from 2.133 to 2.248 — that's 5% worse. And when I thought about why, it makes sense. The model now sees two catalysts with nearly identical element compositions. One of them is labelled 5.2% from our data, one is labelled 8.7% from literature. The model has no feature to explain that difference — it doesn't know that one was done at a specific flow rate and the other wasn't. So it splits the difference and predicts both badly. The 3.42 percentage point gap is not noise — it's a real systematic offset, and treating it as noise corrupts the model. This told me we can't just mix the data — we need to be much more careful."

---

## Chapter 5 — DRST: Filtering by Chemical Similarity

"So after naive merging failed, the obvious question was: what if we only add the literature samples that actually look like our data? DRST answers this by training a classifier. We give it two groups of samples — ours labelled 1, literature labelled 0 — and ask it to learn the boundary between them in the 65-element feature space. Once trained, we score every literature sample: what's the probability this classifier would call it 'ours'? High score means it looks like our chemistry. Low score means it's foreign.

We set a cutoff at 0.30 — everything above is kept, everything below is discarded. That cutoff was chosen by trying five values and seeing which gave the lowest validation error. With tau=0.30, we keep 782 of the 3852 samples — about 20%.

RMSE dropped to 2.019, so 5.3% better than baseline. But here's what bothered me: the cutoff is hard. A sample scoring 0.29 is completely thrown away. A sample scoring 0.31 gets full weight. That felt arbitrary. Which led me to ask — can we do something softer?"

---

## Chapter 6 — KMM: Soft Weights Instead of Hard Filter

"KMM takes the same idea as DRST but makes it continuous. Instead of keep or discard, every literature sample gets a weight between 0 and 10. The weight is found by solving a small optimisation problem — essentially, find weights such that the weighted average of literature feature vectors matches the average of our feature vectors in a kernel space. Samples that look like our data get pulled toward high weights. Samples that are chemically foreign end up with weights near zero.

The result is RMSE 2.035, so about 4.6% better than baseline. Very similar to DRST. And here's what I found reassuring: the KMM weights and the DRST scores correlate at r=0.79. Two completely different methods — one a logistic classifier, one a quadratic optimisation — both independently decided the same 78.5% of samples are useless. That gives me some confidence that it's a real signal, not an artifact.

But here's what both DRST and KMM still have in common: they're adding literature labels directly into training. Those labels have a 3.42 percentage point systematic offset. We're reducing the damage by filtering, but we're not eliminating the root cause. That's what the next method does."

---

## Chapter 7 — Two-Stage Fine-Tuning: The Winner

"This is the method that actually worked well. The core idea is: instead of adding literature labels to training, use literature to train a separate model first — call it the literature expert. Then take that expert's prediction for each of our 89,000 catalyst compositions and add it as a new feature. So now our main model has 68 features instead of 67, and that 68th feature is 'what would the literature expert predict for this composition?'

Stage 2 — the main LightGBM — trains only on our labels. It never sees literature labels directly. It sees the expert's opinion as one input, and it can learn to weight that opinion more in regions where the expert is reliable and less where the expert is off.

The RMSE dropped to 1.907 — that's 10.6% better than baseline, more than double the improvement from either DRST or KMM alone. And on the OOD test — the literature samples the model had never seen in any form — RMSE went from 6.53 to 3.60, which is a 45% reduction.

Why is OOD so much better? Because Stage 1 was trained on literature, which covers all sorts of sol-gel and coprecipitation chemistries. When the final model encounters one of those unfamiliar compositions, the prior feature gives it a reasonable starting estimate from the literature domain. Without the prior, it was essentially guessing."

---

## Chapter 8 — Results + SHAP: Did It Learn Real Chemistry?

"The last thing I want to show is the SHAP analysis, because I think it answers the most important question: did the model actually learn real OCM chemistry, or is it just fitting statistical patterns that happen to work in cross-validation?

SHAP tells you, for each individual prediction, how much each feature pushed the prediction up or down. The beeswarm shows all 3,000 samples at once. Each row is a feature, each dot is a sample, the horizontal position is the effect on the prediction, and the colour is the feature value — red means high, blue means low.

The most reassuring thing is that lit_prior_prediction comes out as the number one feature. The transfer learning is genuinely being used, not just sitting there as dead weight. Then temperature is second — higher temperature pushes yield up, which is consistent with OCM chemistry. Then Ba, Mn, La, Ce all have positive effects, which again matches what we know about these elements as OCM active phases and promoters.

The one thing that worries me is the high-yield range. For catalysts with yield above 15%, the model systematically under-predicts by about 4 percentage points. That's a ceiling effect. But it's not a modelling failure — it's a data problem. The conditions that explain why a catalyst achieves 20% yield — gas flow rate, methane-to-oxygen ratio — are simply not in our feature set. No algorithm can compensate for a missing column."
