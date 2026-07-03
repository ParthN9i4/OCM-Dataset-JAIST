# OCM Walkthrough — Personal Speaking Notes

Personal reference only. Not included in `ocm_walkthrough.ipynb`.

---

# Part 1 — PPT Speaking Notes (15 slides)

One block per slide. Verbatim first-person speech — not a script, more like thinking out
loud while pointing at the screen. Read alongside the corresponding slide.

---

## Slide 1 — Title

"Good afternoon, everyone. Today I'm going to walk you through some work we've been doing
on using published OCM literature to improve our catalyst screening model. The central
question is simple: we have 89,000 of our own experiments, and there are 3,852 published
experiments from 40 years of OCM research around the world — can that published data make
our model more accurate? The short answer is yes, but it turns out to require some care
about *how* you use it. I'll show you five approaches we tried, what each one taught us,
and what we'd focus on next. I'll also end with an honest critical look at what these
methods still can't do."

---

## Slide 2 — The Problem (Chapter 1)

"This slide sets up why this isn't just 'download some data and append it'. The two cards
show our lab data and the published literature side by side.

Our data: 89,074 experiments, all done with the impregnation preparation method, all in
2025, with a mean yield of 5.25%. The literature: 3,852 papers from 1982 to 2019, using
15 different preparation methods, with a mean yield of 8.67%. That 3.42 percentage-point
gap is not noise — the t-statistic is −32.2 with a p-value below 10 to the power of
negative 200. It's as systematic as it gets.

Three sources of that gap. First, **label shift**: publications report their best results,
not their average ones — that's publication bias. And published labs often optimise their
gas flow rates and methane-to-oxygen ratios to get the best yield, while we run fixed
conditions. Second, **covariate shift**: 78.5% of the literature describes catalysts we've
never tested — completely different elements, different preparation methods. Third,
**publication bias as distributional skew**: the literature distribution isn't just shifted
upward, it has a heavier right tail, because failed experiments don't get published. A
shape mismatch can't be fixed by subtracting a constant.

The bottom callout explains why we use ML at all: the search space — 65 elements, varying
loadings, temperature, preparation — is too large to screen physically. A surrogate model
guides which experiments to actually run in the lab."

---

## Slide 3 — Setup — Label Shift & PCA (Chapter 2)

"This slide shows the label shift and covariate shift in data form. The figure has two
panels side by side.

Left panel — yield distributions. Blue is our 89,074 experiments, peaking around 3 to 4
percent. Orange is the 3,852 published literature experiments, peaking around 8 to 9
percent. The smooth curves are kernel density estimates. The red shaded band between the
two vertical dashed lines is the 3.42 percentage-point mean gap drawn as a physical
distance on the plot. That gap is not noise — the t-statistic is −32.2 and the p-value is
below 10 to the power of negative 200. And notice the orange distribution isn't just
shifted — it has a heavier right tail. That asymmetric skew is publication bias: labs
don't publish experiments that failed, so the literature over-represents high-yield
outcomes.

Right panel — PCA of the chemistry. Principal component analysis takes the 67 element and
condition features and compresses them to 2 dimensions. Each dot is a catalyst. Blue dots
are ours, orange are literature. The blue cloud is relatively compact. The orange cloud
extends well beyond the right boundary of the blue cluster — those are the 78.5% of
literature samples describing chemistry our lab has never synthesised. If we added those
to training, we'd teach the model to predict compositions it will never be asked about.

Both the yield gap and the chemistry gap warn against naive concatenation. We confirm that
warning on slide 6."

---

## Slide 4 — Setup — Element Usage (Chapter 2)

"The previous slide's PCA showed that chemistry differs in an abstract two-dimensional
projection. This figure makes it concrete. Each panel is a horizontal bar chart of the top
15 elements by usage frequency — meaning what fraction of experiments in that dataset
include a non-zero loading of that element.

Our lab data is on the left, literature is on the right. The palettes are clearly
different. Our data in 2025 focused on a specific set of active phases and promoters.
Literature, compiled over 40 years from dozens of independent research groups around the
world, spreads across a broader and different mix — different rare-earth promoters,
different support materials, different alkali metal choices.

This is what '78.5% out-of-distribution' looks like in plain terms. It is not abstract
statistics — it is literally different elements and preparation methods. A model trained on
a mixture of both would have to simultaneously predict yield for our chemistry and for
chemistries we will never synthesise. The selective filtering methods that follow use
exactly this observation: keep only the literature that, at the element level, resembles
our experimental programme."

---

## Slide 5 — How We Measure Success (Chapter 3)

"Before results, I want to explain how we measure success, because the design choice here
determines whether the numbers are meaningful.

We use 5-fold cross-validation, but in an asymmetric way. Split our 89,074 samples into 5
groups. In each round, 4 groups train the model and 1 group tests it. We rotate which
group is the test 5 times and average the RMSE.

The asymmetry: literature, when we use it, only ever goes into the *training* side — never
the test fold. The diagram shows fold 3 in red as the held-out test, folds 1, 2, 4, 5 in
blue as training, and the literature always in the training row below.

Why this design? The question is 'how well does the model predict *our* experiments?' If
literature appeared in the test fold we'd be measuring a diluted answer to a different
question — and we'd be unable to compare the 5 methods fairly. Every method goes through
exactly the same CV function, so the comparison is apples-to-apples.

RMSE is in units of percentage points of yield. An RMSE of 2.1 means we're typically off
by about 2 percentage points. Smaller is better. The baseline number — training on our
data only — is 2.133. That is the target every subsequent method must beat."

---

## Slide 6 — Baseline + Why Naive Merging Fails (Chapter 4)

"Step one: establish the baseline. Train on our data alone — no literature — and get RMSE
2.133. That's the number to beat. If a method returns anything higher, it has actively
harmed the model.

Step two: try the obvious thing. Append all 3,852 literature rows and retrain. 'More data
is always better' is the intuition. RMSE goes to 2.248 — 5.4% worse. It harmed the model.

The explanation box on the right shows why. The model now sees two catalyst compositions
that are nearly identical in element space. One is labelled 5.2% yield — from our lab. The
other is labelled 8.7% — from literature. There is no feature in our dataset that explains
the difference: gas flow rate, methane-to-oxygen ratio, pressure are missing columns. The
model does the only rational thing it can: it hedges and predicts something in between,
which makes it wrong for both.

The 3.42 pp offset is a systematic signal the model is trying to fit, but cannot fit
correctly because the explanatory features aren't there. So it corrupts everything around
it. Conclusion: we cannot just add literature labels directly. We need to be much more
selective about *which* samples we add and *how* we represent their information."

---

## Slide 7 — DRST — Filtering by Chemical Similarity (Chapter 5)

"The first serious attempt: instead of adding all literature, only add the part that
chemically resembles our data. DRST — Density-Ratio Selective Transfer — scores each
literature sample on how much it looks like our chemistry.

The scoring method: train a logistic classifier to separate our samples from literature,
then read off the classifier's probability that a literature sample belongs to our class.
High probability means 'this looks like our chemistry.' Low probability means 'chemically
foreign.'

The histogram on the left shows the score distribution for all 3,852 literature samples.
There's a big spike near zero — most of literature is confidently foreign. A thin tail
stretches toward 1 — those are the samples that resemble our lab's work. The dashed lines
mark the candidate thresholds.

The table on the right shows the threshold sweep. At τ=0.30, 782 samples survive and CV
RMSE drops to 2.019 — 5.3% better than baseline. Both higher and lower thresholds perform
worse: lower keeps too many foreign samples, higher discards too many useful ones.

The weakness I'll come back to in the critical analysis: the cutoff is a hard cliff. A
sample scoring 0.289 is completely discarded; one scoring 0.31 gets full weight. That
binary decision throws away gradual similarity information."

---

## Slide 8 — KMM — Soft Weights Instead of a Hard Filter (Chapter 6)

"KMM — Kernel Mean Matching — is the natural continuation of DRST's idea but made
continuous. Instead of keep-or-discard, every literature sample gets a weight between 0
and 10. We find the entire weight vector at once by solving an optimisation problem: choose
weights so the weighted average of literature samples, in a kernel similarity space,
matches the average of our data as closely as possible.

The RBF kernel measures similarity by distance in the 67-feature space. Catalysts with
similar element profiles score high; chemically distant ones score near zero. The bandwidth
parameter σ is set by the median heuristic — the median pairwise distance in a subsample
— so there's no manual tuning.

The weight histogram on the left: a huge spike at zero. The same ~78.5% of literature that
DRST threw away gets weight near zero here. The scatter plot shows the reassuring result:
KMM weights and DRST scores correlate at r=0.79. Two completely different mathematical
approaches — one a logistic classifier, one a quadratic optimisation — independently
concluded the same 78.5% of literature is chemically irrelevant. That agreement is
reassuring evidence that the signal is real and not an artifact.

CV RMSE is 2.035 — 4.6% better than baseline, essentially tied with DRST. Why do they
tie? Because both still feed literature labels directly into training. The 3.42 pp offset
is still bleeding into the model, just from a smaller number of samples. That is the shared
ceiling the next method breaks."

---

## Slide 9 — Prior Feature Transfer — The Winning Pipeline (Chapter 7)

"This is the method that worked well, and its logic is a fundamental reframing of the
problem.

DRST and KMM both had literature labels in the training loss. The offset corrupts that
loss no matter how carefully you filter. The reframe: what if literature never enters as a
*label* at all — only as a *feature*?

Stage 1, top row: the 782 DRST-filtered literature samples train an XGBoost model — the
'literature expert'. Before training, we quantile-normalise the labels onto our yield
scale: the top-10% literature catalyst maps to our top-10% value, and so on. This strips
the absolute offset while preserving the ranking of which chemistry is good versus bad.

Stage 2, bottom row: for each of our 89,074 catalysts, we ask the expert 'what would you
predict for this composition?' and add that as a 68th feature. We then train LightGBM on
those 68 features with our own labels — never literature labels.

The key property highlighted at the bottom: Stage 2 never sees a literature label. The
3.42 pp offset lives in the *value* of the 68th feature. Stage 2 can learn 'when the
expert says 8, my lab gets about 4.5' — a calibration it discovers automatically from our
own data. A corrupted label cannot be un-corrupted; a feature value is just another number
to learn around.

Why XGBoost for Stage 1? It's more conservative on small data — 782 samples — and less
prone to overfitting. Why LightGBM for Stage 2? It's much faster on large datasets like
our 89,000 samples, using histogram-based tree building."

---

## Slide 10 — Results — All Five Methods at a Glance (Chapter 8)

"Here's the full picture. The bar chart shows all five methods with the baseline as a
dashed reference line. Naive merge is the one bar to the right of that line — worse.
DRST and KMM dip modestly below — about 5% improvement each. Prior Feature Transfer makes
a distinctly larger step: 1.907, a 10.6% improvement.

The OOD table on the right measures generalisation to chemistry we've never trained on:
2,139 non-Impregnation literature samples. Baseline RMSE is 6.53. I need to be honest and
correct an earlier version of this slide here: it claimed Prior Feature Transfer dropped
that to 3.60, a 45% reduction. That number was leakage. In the original experiment the
Stage-1 prior was trained on *all* literature, which includes those very OOD test samples
and their true yields — so the prior effectively handed the final model the answers. When
I retrain the prior with the OOD rows removed — a genuinely blind test — the leak-free OOD
RMSE is about 6.0 to 6.8 depending on configuration, i.e. roughly level with baseline, not
a dramatic win.

This doesn't touch the headline: on our own chemistry the 10.6% gain is real, repeatable
across ten random seeds at p below ten-to-the-minus-fourteen, and confirmed on a held-out
test set. What it changes is the extrapolation claim: this is a model that is very good in
our operating regime, not one that dramatically re-derives the literature.

It also ties to quantile normalisation, which the committee asked about. QN rescales the
literature labels onto our yield scale. Turning it on and off cleanly shows a consistent
trade-off: with QN, in-distribution accuracy is better because the prior speaks in our
units, but OOD is slightly worse because the prior can no longer reach the higher yields of
unfamiliar literature. QN is a deliberate dial — we chose local accuracy because our goal
is predicting our next experiment. I'll show this on the Rigour slide."

---

## Slide 11 — SHAP — Feature Importance Beeswarm (Chapter 8)

"RMSE tells us how accurate the model is — SHAP tells us *why* it makes each prediction.
For each of the 3,000 sample points, SHAP decomposes the prediction into individual
feature contributions: how much did each of the 68 features push that particular
prediction above or below the mean?

The beeswarm visualises all 3,000 samples at once. Each row is one feature, ranked
top-to-bottom by its mean absolute SHAP value — the most influential feature is at the
top. Each dot is one catalyst sample. Horizontal position is the SHAP value: dots to the
right pushed the prediction up; dots to the left pushed it down. Dot colour is the feature
value — red is high, blue is low. So a red dot far to the right means 'high value of this
feature strongly increases the predicted yield.'

Three things stand out immediately. First: the very top row is `lit_prior_prediction`.
The literature prior is the single most influential feature across all 3,000 samples. The
transfer learning is not dead weight — it is actively driving every prediction. That
validates the entire Prior Feature Transfer approach.

Second: temperature is ranked second. Red dots — high temperature — cluster on the right
side, meaning high temperature systematically pushes yield predictions up. That is correct
OCM physics: higher temperatures promote the gas-phase radical chain reaction that
produces ethylene.

Third: Ba, Mn, La, and Ce all show net-positive SHAP values — they are well-known OCM
active phases and promoters, consistent with decades of experimental literature. Li and K
show mixed or slightly negative contributions — both can over-reduce the catalyst surface,
which suppresses C₂ selectivity. The model has learned real chemistry from the data it
has access to."

---

## Slide 12 — SHAP — Ceiling Effect & Model Limits (Chapter 8)

"This slide shows the one important limitation the residual analysis reveals: a ceiling
effect at high yields.

The table breaks down prediction error by yield range. Below 6% yield — where the vast
majority of our data lives — RMSE is 1.43 to 1.48, quite good. Between 6 and 10%, it
rises to 1.95. Between 10 and 15%, 2.58 — still manageable. But above 15%, RMSE jumps to
4.70, and there is a systematic bias of −4.2 percentage points. The model is not just
imprecise at high yields — it is consistently and substantially under-predicting.

This is the ceiling effect. It looks like a modelling failure but it is a data failure.
The conditions that explain why a specific catalyst achieves 20% yield — gas hourly space
velocity, the methane-to-oxygen ratio, reactor pressure — are absent from our feature set.
They were never recorded in either our experiments or the literature database. No machine
learning algorithm can predict a phenomenon driven by features that don't exist in the
dataset.

The right column summarises what SHAP confirmed works correctly: the literature prior,
temperature, and the known promoter elements are the primary drivers of what the model
predicts accurately. The ceiling is bounded — data below 15% yield is predicted well. Fix
the data by adding the missing operating conditions; the ceiling lifts automatically."

---

## Slide 13 — Critical Analysis — Per-Method Review (Chapter 9)

"Let me be honest about what each method can and cannot do.

The table: naive merge is a negative control. It proves the shift is real and harmful, but
it provides no solution. DRST is simple, fast, and interpretable, but it uses a hard
cliff, discards about 80% of the literature, and the threshold was tuned on the same
cross-validation we then reported — so that 5.3% improvement is slightly optimistic. KMM
is more principled — no cliff, continuous weights — but the n-by-n kernel matrix is slow
to compute, it's sensitive to the bandwidth choice, it still feeds literature labels into
the loss, and the fact that it just *ties* DRST tells us filtering has hit an inherent
ceiling. Prior Feature Transfer is the winner, but it requires two models to maintain, and
Stage 1 trained on only 782 samples has relatively high variance.

On the right: three limits that cut across all methods. First, all methods were scored on
our own data cross-validation. Out-of-distribution performance was only checked for Prior
FT — we don't know how the filtering methods generalise. Second, point predictions only —
no uncertainty estimates — which limits our ability to identify the most informative
experiments to run next. Third, and most critically: the Y>15% ceiling is a data limit,
not a model limit. The missing condition features — GHSV, methane-to-oxygen ratio,
pressure — explain the ceiling. The key insight at the bottom sums it up: the ceiling
effect and much of the label shift share the same root cause. Fix the data; the models
follow."

---

## Slide 14 — What's Next — Priority-Ordered (Chapter 9)

"Six next steps in priority order.

Item 1 — the highest impact by far: add GHSV, methane-to-oxygen ratio, and pressure to
the feature set. These are the operating conditions that explain the bulk of the label
shift and all of the ceiling effect. This is a data collection problem, not a modelling
problem, and data improvements always dominate. If those features existed, a much simpler
model would likely already beat what we've built.

Item 2: bag the Stage-1 expert. Right now we train one XGBoost on 782 samples.
Bootstrapping several instances and averaging their predictions reduces the variance of the
prior feature. Low cost, likely meaningful gain.

Item 3: add uncertainty. Conformal prediction or quantile regression would let us produce
prediction intervals. In an active-learning setting, we want to prefer candidates that are
both likely to be high-yield *and* uncertain — those are the most informative experiments.

Item 4: stack DRST and KMM inside the Prior FT pipeline. Instead of three separate
methods, integrate filtering and weighting into a single pipeline. Item 5: neural domain
adaptation — DANN or CORAL — once the feature set is richer. With 67 features and limited
diversity, these deep methods don't have enough signal to outperform gradient boosting.
Item 6: close the active-learning loop. Model proposes candidates, lab measures them,
model retrains. Over time the label shift itself shrinks as we accumulate data from
chemistry we previously hadn't explored."

---

## Slide 15 — Summary

"Let me bring this together in four points.

We asked: can published OCM literature improve a lab-data model without harming it? The
answer is yes, but the right method matters a lot.

Naive merging *harms* accuracy — a 5.4% RMSE penalty — because the systematic offset
corrupts training when there are no features to explain it. Selective filtering with DRST
and KMM recovers about 5% improvement, and two independent methods agreeing at r=0.79
confirms the signal is real, not an artifact. Using literature as a *prior feature* — never
as a label — gives the biggest gain: 10.6% better CV accuracy, and this is repeatable
across ten random seeds at p below ten-to-the-minus-fourteen and holds on a held-out test
set. On out-of-distribution literature the honest, leak-free gain is modest — roughly
baseline — and I corrected an earlier slide that overstated it. SHAP confirms the model
uses real OCM chemistry — temperature, barium,
manganese, lanthanum, cerium, and the literature prior are the primary drivers.

The next step that matters most is a data step: add GHSV and the methane-to-oxygen ratio
to the feature set. The modelling work here is solid. The bottleneck is now the data.

Thank you — happy to take questions."

---

---

# Part 2 — Notebook Walkthrough Speaking Notes

This section follows the notebook **cell by cell, in order**. For every code cell it walks
through the code **line by line** and explains what each line does and *why*, so you can
narrate each line as it appears on screen.

---

## Figures in each chapter (quick reference)

| Chapter | Figure | What to point at |
|---|---|---|
| 2 | Label shift KDE + PCA | The 3.42 pp red band; the orange cloud outside the blue cluster |
| 2 | Top 15 element usage | Different elements dominating each panel |
| 5 | DRST score histogram | Where the τ=0.30 dashed line sits in the score distribution |
| 6 | KMM weights + DRST scatter | The diagonal trend showing r=0.79 |
| 8 | 5-method bar chart | Red bar (naive merge) above baseline; orange bars below |
| 8 | SHAP beeswarm | `lit_prior_prediction` at the top of the y-axis |

---

# Chapter 1 — The Problem

*(Markdown cell — no code.)*

"Let me set up the whole project. OCM — Oxidative Coupling of Methane — is a reaction
that turns natural gas, methane, into ethylene. Ethylene is the building block for most
plastics, so an efficient OCM process would be a cheaper, cleaner route to a very
high-value chemical. The thing that controls the reaction is the catalyst — a surface
coated with metal oxides — and the hard part is finding the right one.

The reason this is a machine learning problem is scale. There are about 65 candidate
elements, each at different loadings, plus temperature and preparation method. You can't
physically test all the combinations. So instead we train a model on the experiments we
have already run, and use it to predict the yield of untested compositions — then we only
run the most promising ones in the lab.

Our lab has done 89,074 of these experiments. There's also published OCM literature going
back to the 1980s — 3,852 experiments from labs around the world. The question Prof.
Taniike asked is simple to state: can we use that literature to make our model more
accurate on our own work?

And the answer turns out to be: not by just adding it. There are three problems. First,
**label shift** — literature averages 8.67% yield, ours averages 5.25%, a 3.42 point gap.
That gap is systematic: papers publish their successes, and they report numbers at
optimised flow rates and gas ratios, while we run fixed conditions. Second, **covariate
shift** — 78.5% of literature describes chemistry we've never tested, so adding it teaches
the model about catalysts it will never be asked about. Third, **publication bias as
skew** — the literature distribution isn't just shifted, it's a different *shape*, with a
heavy tail toward high yields, because failures don't get published. A shape mismatch
can't be fixed by subtracting a constant.

The rest of the notebook is five attempts to use literature properly, each one fixing a
weakness the previous one exposed."

---

# Chapter 2 — Setup

*(Markdown cell.)*

"This chapter is data loading, but three small decisions matter a lot later, so I'll
flag them before the code runs.

One — we split the data by the `year` column: 2025 is ours, 2019-and-earlier is
literature. Two — the preparation method is text, and tree models need numbers, so we
encode it; and we fit that encoder on both datasets combined so the integer codes line
up. Three, and this is the one that's easy to get wrong — we scale the features using
*our* data as the reference, then apply that same scaling to literature. If we scaled
each dataset on its own, the same Ba=2% sample would land at different coordinates in
each, and every similarity calculation downstream would be corrupted.

I won't belabour why we skip linear regression and random forests — briefly, linear can't
capture that two elements together behave differently than each alone, and random forests
build trees independently whereas gradient boosting builds them sequentially, correcting
mistakes as it goes, which works much better here."

## Code cell — setup (`449ca273`)

"Let me walk through this line by line.

- The imports: numpy, pandas, matplotlib, then the three model libraries — LightGBM,
  XGBoost, and SHAP — and a handful of scikit-learn and scipy utilities. `np.random.seed(42)`
  fixes randomness so every run is reproducible.
- `df = pd.read_csv(DATA_PATH)` loads the single combined CSV.
- `df_ours = df[df['year'] == 2025]` and `df_lit = df[df['year'] <= 2019]` — this is the
  split. The year column literally *is* the dataset label. `.reset_index(drop=True)` just
  renumbers the rows cleanly.
- The three print lines report the sizes and mean yields, and compute the label shift —
  that's where the 3.42 pp number comes from, live.
- `ELEM_COLS` is built by taking every column that isn't preparation, temperature, target,
  or year — those are the 65 element columns.
- `le = LabelEncoder()` then `le.fit(df['Preparation'])` — note we fit on the full `df`,
  the combined vocabulary, *then* apply it to each subset with `le.transform`. That's the
  shared-mapping point I made earlier.
- `FEATURES` lists temperature, the encoded prep method, and the 65 elements — 67 total.
  We pull those into `X_ours` and `X_lit`, and the yields into `y_ours` and `y_lit`.
- The last three lines are the critical ones: `scaler.fit_transform(X_ours)` learns the
  mean and standard deviation *from our data* and scales it; `scaler.transform(X_lit)`
  applies that *same* ruler to literature. Fit on ours, transform both.

The output confirms 89,074 × 67 for us and 3,852 × 67 for literature."

---

# Chapter 2 (continued) — Visual evidence

*(Markdown cell.)*

"Before any modelling, let me show the two shifts in a picture. The left panel will show
the label shift — the yield distributions — and the right panel the covariate shift — the
chemistry, projected down to two dimensions."

## Code cell — KDE + PCA figure (`241a1f1c`)

"Two panels in this cell.

Left panel, the yield distributions:
- We set two colours, and grab the two means.
- The two `ax.hist(...)` calls use `density=True`. That's important — it normalises each
  histogram so the *area* under it is 1, regardless of sample count. Without it, our 89,000
  samples would dwarf the 3,852 literature ones and you couldn't compare shapes. We also
  use slightly different bin counts, 60 versus 40, because literature has far fewer points
  and too many bins would make it look spiky.
- The loop with `gaussian_kde` overlays a smooth density curve on each histogram — a
  cleaner version of the same shape.
- The two `axvline` calls drop dashed lines at each mean, and `axvspan` shades the band
  between them red — that red band *is* the 3.42 pp gap, drawn as a physical distance.

**[Point at the left panel.]** Blue peaks around 3–4%, orange around 8–9%, and notice the
orange isn't just shifted — it has a heavier right tail. That's the publication-bias skew.

Right panel, the chemistry:
- `idx_sub = np.random.choice(..., size=3000)` subsamples our data to 3,000 points, because
  plotting all 89,000 would be a solid blob.
- `np.vstack([X_ours_sc[idx_sub], X_lit_sc])` stacks our subsample *on top of* literature
  into one array — ours first, literature second. That stacking order is why, two lines
  later, `X_pca[:3000]` is ours and `X_pca[3000:]` is literature.
- `PCA(n_components=2).fit_transform(...)` finds the two directions of greatest variance in
  the 67-dimensional space and projects every point onto them — a 2D shadow of the full
  feature space.
- The two scatter calls plot our 3,000 points in blue and all literature in orange.

**[Point at the right panel.]** The blue cloud is tight; lots of orange dots sit outside
it. Those outside dots are the 78.5% of literature describing chemistry we've never tested."

---

# Chapter 2 (continued) — Which chemistry differs?

*(Markdown cell.)*

"The PCA shows *that* the chemistry differs but not *which* elements. This next figure
makes it concrete — the top 15 elements by usage in each dataset, side by side."

## Code cell — element usage figure (`07f04316`)

"Line by line:
- `(df_ours[ELEM_COLS] > 0).mean()` — for each element column, this asks 'what fraction of
  samples have a non-zero loading of this element?' `> 0` turns it into True/False, and
  `.mean()` of booleans is just the fraction that are True. `.sort_values` ranks them.
  Same for literature.
- `top_n = 15`, then two horizontal bar charts — `barh` — one per dataset. The `[::-1]`
  reverses the order so the most-used element sits at the top of the chart.

**[Point at the two panels.]** They're clearly different palettes. Our lab leans on one
set of promoters and active phases; literature spreads across a broader, different mix.
This is what '78.5% foreign chemistry' looks like in plain terms."

---

# Chapter 3 — How We Measure Success

*(Markdown cell.)*

"Quick but important: how do we score success? We use RMSE in percentage points of yield —
an RMSE of 2.1 means we're typically off by about 2.1 points. We square errors so big
misses hurt more than small ones, which matters when a bad prediction can waste a real
experiment.

The key design choice is **asymmetric** cross-validation. We split our data into five
folds, rotate which one is held out, and average. But literature, when we use it, only
ever goes into the *training* side — never the held-out test fold. The reason is that the
question is 'how well do we predict *our* experiments?' If literature leaked into the test
fold we'd be answering a different, diluted question. Every method runs through the same
function, so the comparison is always apples-to-apples."

## Code cell — `evaluate_cv_ours` (`98985fde`)

"This function is the engine — every method's score comes out of here. Line by line:

- `lgb_params()` just returns the LightGBM settings as a dict, so every model uses
  identical hyperparameters — only the *data* changes between methods. This is important
  for a fair comparison: if each method had its own hyperparameters we'd be comparing
  (method + tuning luck), not just method.
- `evaluate_cv_ours` takes optional extra training data: `X_train_extra`, `y_train_extra`,
  and optional `sample_weight_extra`. For the baseline we pass nothing.
- `KFold(n_splits=5, shuffle=True, random_state=42)` — three design decisions here.
  Why 5 folds? More folds (e.g. 10) give a lower-variance estimate but take twice as long
  and are rarely necessary when n=89,074. Fewer folds (e.g. 3) increase variance and leave
  less data for training. 5 is the standard trade-off. Why `shuffle=True`? Our 89,074
  samples were collected in a specific order (a designed experimental sequence), not
  randomly. Without shuffling, each fold would contain a contiguous block of experiments
  that might share systematic structure (similar catalyst families explored together). With
  shuffling, each fold is a random 20% of all samples, so folds are interchangeable.
  Why `random_state=42`? Reproducibility — we want the exact same folds every run so the
  comparison between methods is deterministic.
- Inside the loop, the first two lines pull out the validation fold: `X_ours_sc[val_idx]`
  and `y_ours[val_idx]`. This is *always our data* — that line never changes. That's the
  asymmetry, right there.
- The `if X_train_extra is not None` branch: `np.vstack` stacks our training fold on top of
  the extra literature, and `np.concatenate` does the same for the labels. The sample
  weights are assembled too — ours get weight 1, literature gets whatever the method passed
  (used by KMM).
- The `else` branch is the baseline: just our training fold, no extras, no weights.
- `model = lgb.LGBMRegressor(**lgb_params())` then `model.fit(...)` trains on whatever we
  assembled, and `model.predict(X_val_f)` predicts the held-out fold.
- The last lines compute RMSE and R² for the fold and store them; after the loop we average
  across the five folds, print, and return.

So to test any method, you just hand this function a different `X_train_extra`."

---

# Chapter 4 — Baseline + Why Naive Merging Fails

*(Two markdown cells.)*

"First, the baseline: train on our data only, no literature. That gives RMSE 2.133 — the
number every method has to beat. Anything above it has *harmed* the model.

Then the obvious experiment: just append all 3,852 literature rows and retrain. More data
should help. It doesn't — RMSE goes *up* to 2.248, about 5% worse. The reason is that the
model now sees two nearly identical compositions with different labels — 5.2% from us, 8.7%
from literature — and nothing in the features explains the gap, because the conditions
behind it were never recorded. So it splits the difference and predicts both badly. That
3.42 pp offset is a systematic signal it can't account for, and trying to fit it corrupts
everything. This is the result that motivates the rest of the notebook."

## Code cell — baseline + naive merge (`4b0d32ad`)

"Just two calls to the function:
- The first `evaluate_cv_ours(label='Baseline...')` passes no extra data — that's the
  baseline, and we save its RMSE as `baseline_rmse`.
- The second passes `X_train_extra=X_lit_sc, y_train_extra=y_lit` — all literature, no
  filtering, full weight. That's the naive merge.
- The print at the end shows the two numbers and the delta.

**[Point at the output.]** Baseline 2.133, naive merge 2.248, delta +0.115 — worse. That's
the evidence."

---

# Chapter 5 — DRST: Filtering by Chemical Similarity

*(Markdown cell.)*

"After naive merging fails, the question is: what if we only add the literature that looks
like our chemistry? The damage was worst for literature *similar* to ours, because that's
what competes with our labels. So DRST scores each literature sample by similarity and
keeps only the high scorers.

The way it scores is clever: train a classifier to tell our samples from literature, then
read off its probability that a given literature sample is 'ours'. High probability means
it looks like our chemistry. We keep everything above a threshold τ, which we pick by
trying five values and taking the best — τ=0.30, which keeps 782 of 3,852.

Its weakness: the cutoff is a hard cliff. A sample at 0.29 is thrown away, one at 0.31 gets
full weight, even though they're almost identical. That's what KMM fixes."

## Code cell — DRST classifier (`156b2acf`)

"Line by line:
- `sub_idx = rng.choice(..., size=10_000)` subsamples our data to 10,000. Why subsample?
  Our 89,074 samples are already heavily dominant — using all of them would make the
  classifier's decision boundary overly biased toward the majority class and slow to train.
  10,000 is enough to characterise our distribution's shape without dominating.
- `X_dom = np.vstack([X_ours_sc[sub_idx], X_lit_sc])` stacks our subsample on top of all
  literature into one matrix; `y_dom` labels ours as 1 and literature as 0.
- `LogisticRegression(C=0.5).fit(X_dom, y_dom)` trains the classifier. Why logistic
  regression and not something like XGBoost or SVM? Because logistic regression estimates
  a *linear* decision boundary in the 67-dimensional feature space, which is exactly what
  we need for density ratio estimation — we want a smooth, general sense of which region
  of feature space is 'ours', not a highly nonlinear boundary that memorises the training
  set. A complex boundary would over-certify some literature samples as 'ours' and
  incorrectly exclude others. Why C=0.5 specifically? C is the regularisation parameter —
  smaller C means stronger regularisation, a smoother, more generalised boundary. C=0.5
  was swept alongside the threshold τ; it gives the best overall CV RMSE. If C were too
  large, the boundary would fit noise in the subsample; too small and the boundary becomes
  too crude to distinguish anything useful.
- `clf_dom.predict_proba(X_lit_sc)[:, 1]` — this is the key line. `predict_proba` returns
  a two-column matrix: column 0 is P(literature|x), column 1 is P(ours|x). We want column
  1 — the probability that this sample belongs to our class. That vector `p_ours_lit` is
  the continuous similarity score for every literature sample.
- The histogram block plots those scores with dashed lines at the candidate thresholds.
- The final loop prints, for each τ, how many samples survive the cutoff.

**[Point at the histogram.]** Big spike near zero — most literature is confidently 'not
ours'. A thin tail stretches toward 1 — those are the ones that look like our chemistry.
The τ=0.30 line sits in that tail."

## Code cell — τ sweep (`9b876dc0`)

"This picks the threshold:
- We loop over five τ values. For each, `mask = p_ours_lit >= tau` selects the surviving
  literature, and we feed `X_lit_sc[mask]` into `evaluate_cv_ours` as the extra training
  data.
- We track the lowest RMSE in `best_drst` and store the winning τ and its mask.

**[Point at the output.]** τ=0.30 gives the lowest RMSE, 2.019 — a 5.3% improvement.

Decision-making note: why five candidate values and not a finer grid? With 5-fold CV, each
evaluation takes several seconds. Sweeping 5 values is quick enough to do every run. A
finer grid (say 10 values) risks overfitting to the CV folds — we'd be optimising τ on the
very folds we report. That's already a mild concern here: the τ was tuned on the same 5
folds that produce the 2.019 number. A held-out validation set would remove this bias but
requires even more data. The 2.019 is therefore a slight overestimate — a fully honest
number would come from a separate test split not used in τ selection."

---

# Chapter 6 — KMM: Soft Weights Instead of a Hard Filter

*(Markdown cell.)*

"KMM keeps DRST's goal but removes the cliff. Instead of keep-or-discard, every literature
sample gets a continuous weight between 0 and 10. The clever part is that it doesn't score
samples one at a time — it solves for the *whole set* of weights at once, choosing them so
the weighted literature cloud looks, on average, like our data cloud. Samples sitting
inside our cloud earn weight; samples outside it have nothing to match and fade to zero.

The result, 2.035, is basically the same as DRST. The interesting bit is the agreement:
the KMM weights and DRST scores correlate at r=0.79. Two completely different methods flag
the same ~78.5% of literature as useless — that tells us the signal is real. But both still
feed literature labels into training, so the offset still leaks. That's the ceiling the
next method breaks."

## Code cell — KMM weights (`5d670d20`)

"There's infrastructure here, but the heart is a few lines:
- `rbf_kernel(X, Y, sigma)` uses `cdist` to get squared distances between all pairs, then
  `np.exp(-dist / 2σ²)` turns distance into similarity — identical samples give 1, far ones
  give near 0. Why RBF? It's the standard choice for density matching because it gives
  smooth, bounded similarity values in any feature space. No assumption about the shape of
  the distribution.
- Inside `kmm_weights`, `sigma` is set by the **median heuristic**: take the median of all
  pairwise distances in a subsample. Why median? The median distance is a natural 'typical
  separation' scale for the data — it's large enough to avoid treating every pair as
  identical (which σ too small would do) and small enough to avoid treating all pairs as
  equally distant (σ too large). The key advantage: no manual tuning, σ adapts to the
  actual spread of the feature space.
- `K_ss` is literature-to-literature similarity — how similar each pair of literature
  samples is to each other. `K_st` is literature-to-ours — how similar each literature
  sample is to our data as a whole.
- `kappa = (n_s / n_t) * K_st.sum(axis=1)` — for each literature sample, this aggregates
  its total similarity to all our data points, scaled by the size ratio. High kappa means
  this literature sample is broadly similar to many of ours → it should get weight.
- The `obj` and `grad` functions define the optimisation — minimise ½ wᵀK_ss·w − κ·w.
  Intuitively: minimising wᵀK_ss·w penalises assigning high weight to samples that are
  far from each other (it's a smoothness term), while maximising κ·w rewards putting weight
  on samples that overlap with our distribution.
- `minimize(..., method='L-BFGS-B', bounds=[(0, B)]...)` solves it with weights bounded
  between 0 and B=10. Why cap at 10? Without a cap, outlier literature samples that happen
  to be very close to one of our points could get extremely large weights, dominating the
  training set. B=10 is a common default that prevents extreme upweighting while still
  allowing meaningful differentiation.
- Back outside, we compute the correlation with the DRST scores and print how many weights
  are near zero.
- The two-panel figure: left is the weight histogram, right is the scatter of DRST score
  versus KMM weight.

**[Point at the left panel.]** Huge spike at zero — 78.5% of literature effectively
ignored. **[Point at the right panel.]** The diagonal trend is the r=0.79 agreement: two
different methods, same conclusion. This is the key sanity check: if the two methods
disagreed strongly, at least one of them would be wrong. Their agreement means the signal
is structural, not an artifact of method choice."

## Code cell — KMM evaluate (`da4a3980`)

"One call. Same `evaluate_cv_ours`, but now we pass *all* literature as training data with
`sample_weight_extra=w_kmm` — the weights we just computed. The function hands those to
LightGBM's `sample_weight`. RMSE 2.035, a 4.6% gain — and, as I said, essentially tied with
DRST."

---

# Chapter 7 — Prior Feature Transfer: The Winner

*(Markdown cell.)*

"Here's the reframe that breaks the ceiling. DRST and KMM both still put literature
*labels* into training, so the offset can't be fully escaped. What if literature never
enters as a label at all — only as a feature?

Two stages. Stage 1: train a small 'literature expert' on the 782 filtered samples — but
first quantile-normalise its labels onto our scale, so a top-10% literature catalyst maps
to our top-10% value. That strips the absolute offset but keeps the *ranking* of good
versus bad chemistry. Stage 2: for each of our catalysts, ask the expert its opinion and
add that as a 68th feature, then train on *our* labels only.

Why this is fundamentally better: the offset now lives in the value of one input column,
not in the labels. Stage 2 can learn 'when the expert says 8, my lab actually gets about
4.5' — a calibration it discovers from our own data. You can't un-corrupt a label, but you
can learn around a feature.

The result is 1.907 — a 10.6% improvement, double the filtering methods, and repeatable
across ten seeds and on a held-out test set. On out-of-distribution literature the honest
leak-free gain is modest (near baseline); an earlier version of this slide overstated it
due to leakage, which I've corrected."

## Code cell — Prior Feature Transfer (`560d7b02`)

"Walk through it carefully — the order inside the fold matters:

- `quantile_normalize_y` ranks the source labels (literature's yields) and maps each rank
  to the matching quantile of the target (our training fold's yields). Why quantile
  normalisation and not a simpler mean-shift — just subtracting the 3.42 pp gap? Two
  reasons. First, the gap is not constant across the yield range — literature's
  distribution has a heavier right tail, so the shift is larger at high yields than low.
  Subtracting a constant would under-correct at high yields and over-correct at low yields.
  Quantile normalisation fixes the *shape* mismatch, not just the mean. Second, quantile
  normalisation preserves *rank ordering* — if catalyst A gives higher yield than catalyst
  B in literature, it still gives higher yield after normalisation. That's the signal we
  want Stage 1 to learn.
- `xgb_params()` returns the XGBoost settings for the small Stage-1 model. Why XGBoost
  for Stage 1 and not LightGBM? XGBoost is more conservative by default — it uses exact
  greedy tree building with stronger regularisation, which is preferable on small datasets
  (782 samples). LightGBM's histogram-based approach trades some accuracy for speed, and
  speed isn't the bottleneck on 782 samples. On our 89,074 samples in Stage 2, the
  speed advantage of LightGBM matters.
- `mask = p_ours_lit >= best_drst[3]` reuses DRST's filter — Stage 1 trains only on the
  782 chemically relevant samples, not the full 3,852.
- Now the CV loop. For each fold:
  - `y_lit_qn = quantile_normalize_y(y_lit[mask], y_ours[train_idx])` — critical: we
    normalise *inside the fold*, using only the *training fold's* yields as the target
    distribution. If we normalised once outside the loop using all our data, the
    normalisation would 'see' the validation fold's yields — that's data leakage. Doing it
    inside the loop ensures the validation fold's yield distribution never influenced the
    normalisation.
  - `pre = xgb.XGBRegressor(...)` then `pre.fit(X_lit_sc[mask], y_lit_qn)` — train the
    literature expert on the 782 filtered, quantile-normalised literature samples only.
  - `prior_tr = pre.predict(X_ours_sc[train_idx])` and `prior_val = pre.predict(...val_idx)`
    — the expert's opinion on each of our training and validation catalysts. `.reshape(-1,1)`
    makes it a column we can append.
  - `final = lgb.LGBMRegressor(...)` then `final.fit(np.hstack([X_ours_sc[train_idx],
    prior_tr]), y_ours[train_idx])` — here's the crux: we glue the prior column onto our 67
    features to make 68, and train on `y_ours` — *our* labels, never literature's.
  - `final.predict(np.hstack([X_ours_sc[val_idx], prior_val]))` predicts the held-out fold
    with its prior column attached, and we score it.
- After the loop we average and print.

**[Point at the output.]** 1.907, down 10.6%. And the summary block lists all five methods
in order — baseline, the failed naive merge above it, DRST and KMM below, and Prior Feature
Transfer at the bottom, clearly the best."

---

# Chapter 8 — Results + SHAP: Did It Learn Real Chemistry?

*(Markdown cell.)*

"Two things here: a chart that pulls the whole story together, and then SHAP, which checks
*why* the model works.

The bar chart: naive merge is the one bar above the baseline line — worse. DRST and KMM dip
slightly below. Prior Feature Transfer is the big jump, because it fixed the root cause
rather than the symptom.

Then SHAP. RMSE tells us how accurate the model is, but not whether it's right for the
right reasons. SHAP attributes each prediction to its features — for every sample it says
which features pushed the prediction up or down. If the model learned real chemistry, the
physics should show up at the top. And it does: the literature prior is the most important
feature, temperature is second and pushes yield up as it should, and known promoter
elements — Ba, Mn, La, Ce — all contribute positively. The one weakness is a ceiling above
15% yield, which is a missing-data problem, not a model problem — we'll come back to it in
Chapter 9."

## Code cell — results bar chart (`96646662`)

"Line by line:
- `df_res = pd.DataFrame(...)` builds a table from the `results` dictionary we've been
  filling in all along, and `.sort_values('CV RMSE')` orders the methods best-to-worst.
- The `palette` list-comprehension colours each bar — blue for baseline, red for naive
  merge, orange for the rest — so the eye immediately separates the failure from the wins.
- `ax.barh(...)` draws the horizontal bars with `xerr` error bars from the CV standard
  deviation.
- `ax.axvline(baseline_rmse, ...)` drops the dashed reference line — anything right of it is
  worse than baseline, anything left is better. `invert_yaxis` puts the best at the top."

## Code cell — SHAP (`76bea5fc`)

"Line by line:
- We retrain the whole pipeline once on *all* our data, outside the CV loop. Why retrain?
  In cross-validation, each fold produces a different model — there's no single model to
  explain. Here we want one stable model to interrogate. Quantile-normalise the full
  literature, train Stage-1 `pre_shap` on all filtered literature, generate `prior_all`
  for all 89,074 of our samples, and `np.hstack` it on to make the 68-feature matrix
  `X_aug`. `aug_feat_names` adds the name `lit_prior_prediction` for that 68th column.
- `final_shap = lgb.LGBMRegressor(...).fit(X_aug, y_ours)` is the model we'll interrogate.
- `idx_sh = rng_sh.choice(..., size=3000)` subsamples 3,000 points. Why 3,000 and not all
  89,074? SHAP for tree models is O(n × depth × leaves) per sample — it's fast per sample
  but 89,074 samples would take minutes. 3,000 captures the full population structure;
  beyond ~1,000 the beeswarm pattern stabilises.
- `explainer = shap.TreeExplainer(final_shap)` — for tree models this computes *exact* SHAP
  values, not an approximation. TreeExplainer exploits the tree structure to compute the
  conditional expectations analytically, rather than the KernelExplainer approach which
  would sample random subsets. The result is exact Shapley values, guaranteed to satisfy
  efficiency, symmetry, and dummy properties.
- `shap_values = explainer.shap_values(X_sh)` produces one SHAP value per sample per
  feature. A positive SHAP value for a feature means 'this feature pushed this prediction
  above the average prediction.' A negative value means it pushed it below.
- `shap.summary_plot(...)` draws the beeswarm. In the beeswarm: each row is a feature
  sorted by mean |SHAP value|. Each dot is one sample. Horizontal position is the SHAP
  value. Color is the *feature value* — red means high, blue means low. So a red dot far
  right for temperature means 'high temperature → big positive contribution to yield' —
  which is physically correct. A blue dot far left means 'low feature value → depresses
  yield.' The width of the dot cloud shows variability in how much that feature matters
  across different catalysts.

**[Point at the beeswarm.]** Top row is `lit_prior_prediction` — the prior is the single
most influential feature, so the transfer learning is genuinely doing work. Temperature is
next, with high values (red) pushing yield up — correct OCM physics. Then Ba, Mn, La, Ce —
the known active phases and promoters — all positive. Li and K are mixed or net-negative,
also physically reasonable as these can over-reduce the catalyst surface. That's the model
using real chemistry, which is exactly what we wanted to confirm."

---

# Chapter 9 — Critical Analysis & Next Steps

*(Markdown cell — no code.)*

"Let me close with an honest assessment, because the headline number isn't the whole story.

On the methods: **naive merge** is really just a negative control — it proves the shift is
real and harmful, but it's useless on its own. **DRST** is simple, fast, and gives an
interpretable score, but it throws away about 80% of the literature at a hard cutoff, and —
being honest — we tuned that cutoff on the same cross-validation we then reported, so the
5.3% is slightly flattering. **KMM** is more principled, matching whole distributions with
no cliff, but it builds a big kernel matrix so it's slow, it's sensitive to the bandwidth,
and it still uses literature labels — and the fact that it just *ties* DRST tells us that
filtering has hit its ceiling. **Prior Feature Transfer** is the clear winner because it
decouples the offset from the loss, but it's not free: the Stage-1 expert is trained on
only 782 samples so it's high-variance, it leans on the assumption that the quantile map
preserves ranking across the two domains, and it's two models to maintain instead of one.

Cross-cutting: everything was scored on our-data CV, and only the winner was checked
out-of-distribution; the model gives point predictions with no uncertainty, which the
experiment-selection loop really wants; and the ceiling above 15% yield is a *data* limit —
GHSV, the gas ratio, and pressure simply aren't columns we have.

So what's next, in priority order. First and biggest: **add those reaction-condition
features** — GHSV, CH₄:O₂ ratio, pressure. That likely fixes both the ceiling and a chunk
of the label shift itself. Second, **bag the Stage-1 expert** — average several experts
trained on bootstrapped literature to cut its variance. Third, **add uncertainty
estimates**, so we can prefer high-value, low-confidence candidates when choosing
experiments. Then there's stacking the filtering and the prior together, trying neural
domain adaptation once the feature set is richer, and finally closing the active-learning
loop — using the model to propose experiments, running them, and retraining. That loop was
the point of the whole project."
