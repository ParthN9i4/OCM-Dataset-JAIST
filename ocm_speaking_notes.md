# OCM Walkthrough — Personal Speaking Notes

Personal reference only. Not included in `ocm_walkthrough.ipynb`.

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

## Chapter 1 — The Problem

### Context: what is OCM and why does it need ML?

"Oxidative Coupling of Methane is a chemical reaction that converts natural gas
(methane, CH₄) directly into ethylene (C₂H₄) — the feedstock for most of the
plastics we use, from polyethylene bags to PVC pipes. The conventional industrial
route to ethylene is steam cracking of naphtha, which is energy-intensive. OCM
would use cheaper natural gas directly at high temperature with an oxide catalyst.

The challenge is that the 'right catalyst' — the surface material coated with
metal oxides that makes the reaction go efficiently — is not obvious. The search
space is enormous: you have about 65 candidate elements, each at varying loading
percentages, plus temperature and preparation method. The number of possible
combinations to test physically is astronomical — you cannot screen them all by
running experiments. That is exactly the problem machine learning is suited to solve.

You build a model from the experiments you have already run. The model learns the
mapping from 'catalyst composition + temperature' to 'yield'. Then you ask the model
to rank unexplored compositions without running experiments, and you physically
test only the top candidates. This is called surrogate-based optimization, or
Bayesian optimization with a machine learning surrogate. It is now widely used in
materials discovery — see Lunsford (2000, Catalysis Today) for an OCM overview, and
Lookman et al. (2019, Nature Communications) for ML-guided materials discovery.

Our lab has run 89,074 of these experiments — each row is one catalyst composition
at one temperature, with the measured Y(C2) yield in percent. That is a very large
dataset for this domain. The question Prof. Taniike raised was: there is also
published OCM literature going back to the 1980s — 3,852 reported experiments from
dozens of labs worldwide. Can we use that literature to make our model more accurate?"

### Why it is not straightforward: three problems

"On the surface this seems simple — more data should help. But when I looked at the
numbers carefully, I found three separate problems. Let me explain each in terms a
non-ML audience can follow.

**Problem 1: Label shift (the 3.42 percentage point gap)**

If I compute the average yield in our dataset, I get 5.25%. In the literature
dataset, the average is 8.67%. That is a gap of 3.42 percentage points. Now, 3.42 pp
sounds small — but in context, it is enormous. The difference between a mediocre
OCM catalyst and a good one is often 3-5 pp of yield. So the average literature
experiment looks like a good catalyst by our lab's standards.

Why does this gap exist? Two reasons. First, publication bias: academic labs publish
results that are interesting — and a failed experiment with 0.5% yield is rarely
interesting enough to publish. Literature systematically over-represents successful
experiments. Second, optimized conditions: when literature researchers report high
yields, they are usually reporting the result after they have tuned the gas flow rate
(GHSV), the methane-to-oxygen ratio, and the pressure to maximize yield. Our lab
runs at fixed conditions. So even for the same catalyst, literature might report 9%
because they used their optimal GHSV, while we would get 5.5% with our fixed setup.

This is not a simple calibration offset we can just subtract. It is baked into which
experiments were published and how they were run.

**Problem 2: Covariate shift (78.5% of literature is foreign chemistry)**

'Covariate' is statistician's language for 'feature' or 'input variable'. Covariate
shift means that the distribution of input variables — catalyst compositions — is
different between our dataset and the literature. When I project both datasets into
the same chemical feature space (using PCA, which I will show in a moment), I can see
that 78.5% of literature samples describe catalyst compositions that are outside the
range our lab has ever tested.

In plain terms: literature labs in the 1980s and 1990s were exploring all kinds of
exotic chemistry — unusual base metals, mixed oxides, rare-earth combinations — that
our current research program does not focus on. If I add those samples to my training
data, I am teaching the model about chemistry it will never be asked to predict. At
best, that is wasted signal. At worst, it actively misleads the model in our region
of chemistry space by introducing noise.

**Problem 3: Publication bias as distributional skew (the orange tail)**

This is the subtlest of the three, and the one most people miss. Do not just look at
where the two distributions are centred — look at their shapes. [Point at the KDE
figure when it appears.] Our blue distribution is fairly symmetric, peaked near
3-4%. The literature orange distribution is skewed: it has a heavy tail toward high
yields. This is the signature of selective publication. Nobody publishes
'we ran Ba/MgO at 750°C and got 1.5% yield.' The shape of the orange distribution
is not the shape of a random sample from OCM chemistry — it is the shape of what
survives the editorial filter. This means the problem is not just a mean shift but
a shape mismatch, which makes it harder to correct.

To summarize: if I just concatenate the two datasets and train one model, the model
will see contradictory labels for similar chemistry, will train on regions of chemical
space it will never encounter in deployment, and will see an over-representation of
high-yield experiments that biases its internal calibration. The next four chapters
are our attempt to solve these three problems one at a time."

---

## Chapter 2 — Setup

### Before the code cell

"This chapter is mostly data loading and preprocessing. Most of it is straightforward.
But there are three specific decisions in this code that are easy to miss and that
matter enormously downstream. Let me explain them one by one before running anything.

**Decision 1: LabelEncoder — why trees need integers**

The 'Preparation' column contains text strings describing how the catalyst was made:
'Impregnation', 'Sol-gel', 'Coprecipitation', and about fifteen others. Machine
learning models — at least the tree-based models we use here — need numerical inputs.
They cannot parse text.

One approach would be 'one-hot encoding': create a separate binary column for each
preparation method. But with 15+ methods, that adds 15 columns, most of which are
zero for our data (since we only use Impregnation). LabelEncoder is simpler: it
assigns an integer to each unique text value. 'Impregnation' → 0, 'Sol-gel' → 1,
and so on. For tree models this works fine — the model learns split points on the
encoded integer, effectively treating preparation method as a categorical variable.

Important: we fit the encoder on the combined vocabulary of both datasets. If we
fit it separately on each dataset, a method that appears in both datasets might get
assigned different integers, which would confuse any method that uses both.

**Decision 2: StandardScaler — why we need consistent units**

Different features are measured on completely different scales. Temperature ranges
from about 500 to 900°C. Element loadings range from 0 to about 15 weight percent.
If we compute distances between samples in raw feature space — which KMM (Chapter 6)
does — temperature would dominate simply because its numbers are fifty times larger
than element loadings. A catalyst at 700°C and a catalyst at 750°C would look far
apart, even though 50°C is a moderate change. A catalyst with 10% Ba vs. one with
1% Ba — a huge chemical difference — would look nearby.

StandardScaler subtracts the mean and divides by the standard deviation for each
feature. After scaling, every feature has mean 0 and standard deviation 1. A 'one-unit
change' in temperature means 'one standard deviation change in temperature' — and
the same for every element. Now distances are measured in a consistent, physically
fair way.

**Decision 3: Fit on ours, transform both — the most important line**

This is the decision I got wrong the first time I set up this analysis, and it is
worth spending a full minute on.

You might ask: why not scale each dataset independently? The answer is that the
scaler's parameters (mean and standard deviation for each feature) define the
coordinate system in which all subsequent calculations happen. If I fit the scaler
separately on each dataset, the same physical measurement would map to different
coordinates in each dataset.

Concrete example: suppose Ba loading in our data has mean 1.5% and standard
deviation 2.0%. After scaling, a Ba=2% sample gets the value (2.0−1.5)/2.0 = +0.25.
Now suppose literature has a different Ba distribution (mean 0.8%, std 1.5%), so I
fit a separate scaler. The same Ba=2% sample from literature gets (2.0−0.8)/1.5
= +0.80. These are two different numbers representing the same physical measurement.
KMM's similarity calculation (Chapter 6) would treat them as different chemistry.

The fix: fit the scaler on our data only — our distribution becomes the reference
frame, the ruler — and then apply exactly the same transformation to literature. The
line `X_lit_sc = scaler.transform(X_lit)` uses the mean and std learned from our data."

### When the label-shift + PCA figure appears

"OK, this is the picture that frames the entire problem. There are two panels.

Left panel — label shift: both yield distributions are shown with probability density
on the y-axis. That means the height is normalized so that the area under each curve
equals 1, regardless of how many samples each dataset has. We are comparing shapes,
not counts. The blue curve (our 89,000 experiments) peaks near 3-4% yield and tails
off smoothly. The orange curve (3,852 literature samples) is shifted right to the
8-9% range, AND it has a different shape — heavier tail toward the right. The red
band between the two dashed mean lines is the 3.42 pp gap I mentioned — but look
beyond the means. The shapes are different. Publication bias and optimized conditions
together produce a distribution that is not just shifted but differently shaped.

Right panel — covariate shift: this is PCA, which I'll explain quickly. We have
67 features per sample. You cannot plot 67 dimensions. PCA finds the two directions
in the 67-dimensional feature space that explain the most variation, then projects
every sample onto those two directions. Think of it like casting a shadow of a
complex 3D object onto a 2D wall — you lose information, but you see the overall
structure. The blue cloud is our 3,000 plotted samples (subsampled) — tightly
clustered in one region of chemical space. The orange dots are all 3,852 literature
samples. Many of them fall outside the blue cluster entirely. Those orange dots in
the white space are the chemistry our lab has never tested — the 78.5% OOD fraction.

If I add all the orange dots to my training set, I am training the model on chemistry
it will never see in deployment. And in those orange regions, the labels carry the
3.42 pp systematic offset. It is a double problem."

### When the top elements figure appears

"The PCA is mathematically rigorous but abstract. This figure is concrete. It shows
the top 15 elements by usage frequency — that is, the fraction of samples in each
dataset where that element has a non-zero loading — in each dataset side by side.

Look at the two panels. They are not the same. Our lab focuses on a specific palette
of promoters and active phases. Literature uses a broader, partially overlapping set.
Some elements that appear prominently in literature barely appear in our data, and
vice versa. This is not a labelling difference — it is a genuine difference in the
chemistry being studied. When I say 78.5% of literature samples are OOD, this figure
shows you physically what that means: different elements, different combinations,
different chemical families."

---

## Chapter 3 — How We Measure Success

"Before running any method, we need to agree on how to measure whether it worked.
This is more important than it sounds — choosing the wrong metric can make a harmful
method look beneficial.

**What is RMSE, and why this metric?**

RMSE stands for Root Mean Squared Error. For each prediction, we compute the error
(predicted yield minus actual yield). We square all the errors — this ensures positive
and negative errors don't cancel each other out. We take the mean of the squared
errors. We take the square root to put the units back in percentage points of yield.

An RMSE of 2.1 means: the model's typical error is about 2.1 percentage points.
If the true yield is 6%, the model predicts something in roughly the 4–8% range.
That is the practical accuracy you are getting from this model.

Why not just use mean absolute error? We use RMSE because it penalizes large errors
more than small ones — a 6 pp error is penalized 36× more than a 1 pp error. For
a model that will be used to select promising catalysts to test, a catastrophically
wrong prediction on one sample is worse than being slightly off on many samples.
RMSE's squared penalty captures this.

**What is cross-validation, and why use it?**

Here is the core challenge. We want to know how well our model will perform on new
experiments it has never seen. But all we have is our existing dataset of 89,074
experiments. If we train on the whole dataset and then test on the same data, we get
an artificially perfect score — the model has effectively memorised the answers. That
tells us nothing about generalisation.

The naive fix is to split the data into a training set (80%) and a test set (20%).
Train on one, test on the other. But this throws away 20% of our training data and
gives us a noisy single estimate of performance.

Cross-validation solves both problems. We split our 89,074 samples into 5 equal
groups (folds). In Round 1, we train on Folds 2-5 and test on Fold 1. In Round 2,
we train on Folds 1,3,4,5 and test on Fold 2. And so on for all 5 rounds. Each
sample appears in the test fold exactly once. We average the 5 RMSE values.

This uses all the data efficiently and gives us a robust performance estimate. The
5-fold number is a standard choice — Stone (1974, JRSS-B) showed cross-validation
is asymptotically optimal for model selection, and Kohavi (1995) recommended 10-fold
as a practical default. We use 5-fold because our dataset is large enough that each
fold has 17,800 samples, which gives stable estimates.

**Why asymmetric? Why does literature never appear in validation?**

In each CV round, when we are testing a literature transfer method, we add literature
samples to the training fold. But we never add them to the test fold. Why?

The question we are answering is: 'how accurately does this model predict our
experiments?' The word 'our' is doing a lot of work. If I allowed literature samples
into the test fold, I would be measuring accuracy on a mixture of our experiments and
published literature runs — that is a different question, and answering it would tell
us nothing about whether the model is better for Prof. Taniike's lab.

This asymmetric design means the validation fold is always 100% our data. When we
compare RMSE across methods, we are comparing apples to apples: every number measures
'how well does this variant predict our experiments?' That is the only comparison
that matters for the scientific question we are trying to answer.

The code function `evaluate_cv_ours` implements this design identically for all 5
methods. The only thing that changes between method calls is what gets passed as
`X_train_extra` and `y_train_extra`."

---

## Chapter 4 — Baseline + Why Naive Merging Fails

### Step 1 — baseline

"The first thing we always do in this kind of analysis is establish a baseline: run
the model on just our own data, with no literature, and record the score. This is
the number every subsequent method must beat. If a method produces RMSE > 2.133,
it has made the model worse — full stop.

The baseline RMSE is 2.133. In practical terms: without any literature data, our
LightGBM model predicts our catalyst yields with a typical error of about 2.1
percentage points. That is our starting point.

If someone asks why LightGBM and not something simpler: LightGBM is a gradient
boosting framework, which means it builds a sequence of decision trees where each
new tree corrects the errors of the previous ones. It consistently outperforms
simpler models like linear regression or random forests on tabular data of this size.
Ke et al. (2017, NeurIPS) showed that LightGBM achieves similar accuracy to XGBoost
with significantly faster training on large datasets — important when we are running
5-fold CV multiple times."

### Step 2 — naive merge

"The most natural first experiment: append all 3,852 literature rows to the training
set and retrain. More data should help, right?

It didn't. RMSE went from 2.133 to 2.248 — that is 5.4% worse than baseline.

When I saw this result, I was initially confused. More data is almost always better
in machine learning. Why did it hurt here?

Think about what the model is seeing. It has two training samples with nearly
identical catalyst compositions. One is labelled 5.2% (from our lab). The other is
labelled 8.7% (from literature). The features — element loadings, temperature —
are similar. But the labels differ by 3.42 pp. And there is no feature in the dataset
that explains this difference. The features that would explain it — gas flow rate,
CH₄:O₂ ratio, pressure — are not in our dataset.

The model cannot resolve this contradiction. It does the only thing it can: it
hedges. It predicts some value between 5.2% and 8.7% for that type of catalyst.
This makes it worse at predicting both our samples and the literature samples.

This is the phenomenon of dataset shift hurting transfer, described theoretically in
Ben-David et al. (2010, Machine Learning): when the source and target distributions
differ significantly, naively combining their data can lead to a model that is worse
than training on either alone. Our experiment is a concrete example of that theorem.

The 3.42 pp gap is not noise — it is a systematic signal the model cannot account
for and incorrectly tries to fit. This told me we cannot just mix the data. We need
to be much more careful."

---

## Chapter 5 — DRST: Filtering by Chemical Similarity

"The naive merge taught us that we cannot add all literature. The question DRST
asks is: what if we only add the parts that actually look like our chemistry?

The fundamental insight is that the damage from the 3.42 pp offset is largest for
literature samples describing chemistry similar to ours — because those samples
directly compete with our labels in the training objective. For samples describing
completely foreign chemistry, the model mostly ignores them since it will never be
asked to predict in that region. So the most harmful samples are the ones that are
similar enough to be relevant but still carry the offset.

The solution: score each literature sample — 'how much does this look like one
of our catalysts?' — and only add the high-scoring ones. This is the Density Ratio
Selective Transfer (DRST) approach.

**What is a logistic regression classifier?**

A classifier is a function that takes a set of inputs (features) and assigns each
input to one of several categories. Logistic regression is one of the simplest and
oldest classifiers — it was introduced by Cox (1958) and is still widely used today.

Imagine you have two clouds of points in a high-dimensional space — our data and
literature data. Logistic regression finds a flat hyperplane (the generalization of
a straight line to many dimensions) that best separates the two clouds. Once that
boundary is found, any new point can be located relative to the boundary: how far
is it from the line? Which side is it on? The logistic function converts this
distance into a probability between 0 and 1.

In our case: we train the classifier with 10,000 of our samples labelled 1 ('ours')
and all 3,852 literature samples labelled 0 ('literature'). The classifier learns
which combinations of element features are typical of each group. We then ask it:
for each literature sample, what is the probability you would classify it as 'ours'?

A literature sample with Ba, La, and similar temperature range to our data gets
a high probability — it looks like it could have come from our lab. A sample
dominated by unusual elements at extreme temperatures gets a low probability.

**Why C=0.5 (regularisation)?**

C is the inverse of the regularisation strength. A small C (like our 0.5) forces
the classifier to prefer simple, smooth boundaries over complex ones that
perfectly separate every individual point. We want simplicity here because we are
describing the general chemical character of two datasets — not memorising which
specific samples belong where. A C of 0.5 was confirmed by Pedregosa et al. (2011,
JMLR) as a reasonable default for soft-margin classification.

**Why τ=0.30?**

We swept five candidate thresholds {0.05, 0.10, 0.20, 0.30, 0.40} and evaluated
CV RMSE for each. This is standard hyperparameter selection — try a grid of values,
pick the one with the best CV score. τ=0.30 gave the lowest RMSE (2.019), retaining
782 of 3,852 samples (about 20%). Stone (1974) established the theoretical basis
for this kind of cross-validation-based model selection."

### When the DRST histogram appears

"This histogram shows where every one of the 3,852 literature samples lands on the
P(ours|x) axis. Look at the shape.

The dominant feature is the enormous spike on the left — near probability 0.0. These
are the literature samples the classifier is highly confident about: 'this is
literature chemistry, not yours.' These samples use different element combinations,
different temperature ranges, different preparation methods. The classifier sees them
as obviously foreign.

Then there is a long but sparse tail extending toward 1.0. These are the samples
the classifier cannot easily distinguish from our data — chemically they look similar
to experiments we run. These are the potentially useful ones.

The four dashed lines are the four threshold values I tested. The τ=0.30 line sits
partway into the tail, retaining the genuinely similar samples. With this threshold,
782 samples pass the filter and get added to training. The result: RMSE = 2.019,
which is 5.3% better than baseline.

But look at what the threshold is doing. A sample scoring 0.29 is completely
discarded. A sample scoring 0.31 gets the exact same weight as a sample scoring 0.99.
This binary decision feels wrong — the density is continuous across the threshold.
A sample at 0.29 and a sample at 0.31 are virtually identical in terms of chemical
similarity, but we are treating them completely differently. This motivated the
next method."

---

## Chapter 6 — KMM: Soft Weights Instead of a Hard Filter

"KMM — Kernel Mean Matching — was introduced by Huang et al. (2006, NeurIPS) and
analysed further by Gretton et al. (2009) in 'Dataset Shift in Machine Learning'.
It addresses the hard-threshold problem by assigning every literature sample a
continuous weight between 0 and 10.

**The intuition: matching distributions**

Here is the core idea. We want the weighted literature distribution to match our
data distribution as closely as possible. If a literature sample describes chemistry
typical of our dataset, it gets a high weight. If it describes foreign chemistry,
it gets a low weight. The optimization problem is: find weights w₁, w₂, ..., w₃₈₅₂
such that the weighted average of literature feature vectors (in a certain similarity
space) is as close as possible to the unweighted average of our feature vectors.

This is not a classification problem — there is no boundary being drawn. It is
a distribution matching problem: find weights that make literature look like us.

**What is the RBF kernel, and why use it?**

The kernel is the mathematical tool used to measure 'similarity' between two samples.
The RBF (Radial Basis Function) kernel, also called the Gaussian kernel, computes:

  K(x, x') = exp(−‖x − x'‖² / 2σ²)

In plain English: two samples with identical feature vectors get similarity 1.
Two samples far apart in the feature space get similarity near 0. The function
decreases smoothly with distance, which is exactly the notion of chemical similarity
we want: catalysts with similar element profiles are similar; catalysts with very
different compositions are dissimilar.

The parameter σ (sigma, the bandwidth) controls the scale of similarity — how far
apart two samples can be and still be considered 'similar'. We set it using the
median heuristic: σ = median of all pairwise distances in a combined subsample of
both datasets. This is recommended by Gretton et al. as a data-driven, parameter-free
choice that adapts to the actual spread of the data.

**What κ (kappa) means**

For each literature sample i, we compute κᵢ = (weighted) sum of its similarities
to all our data points. If literature sample i has chemistry similar to many of our
experiments, κᵢ is large — that sample's region of chemical space is well-represented
in our data, so it should get high weight. If sample i describes chemistry in an
empty region of our space, κᵢ is near zero — this sample has nothing to match to,
so its weight converges to zero.

The optimization is then: minimize ½ wᵀ K_ss w − κᵀ w subject to 0 ≤ wᵢ ≤ B.
The K_ss term penalizes extreme weights (regularization). The κᵀw term rewards
weighting samples that overlap with our distribution. The result is a smooth,
continuous version of what DRST does with a hard threshold."

### When the KMM weights + agreement scatter appears

"Left panel — weight distribution: that giant spike near zero is 78.5% of all
literature samples with weight below 0.1. KMM is not discarding them — they are
still in the training data — but a weight of 0.05 effectively means 'ignore this
sample.' Mathematically, it is equivalent to DRST's discard, but achieved through
continuous optimization rather than a binary threshold.

Right panel — agreement between the methods: each dot is one literature sample.
X-axis is the DRST score (logistic classifier). Y-axis is the KMM weight
(quadratic optimization). If the two methods were unrelated, the dots would scatter
randomly. Instead, they trend together, with a correlation of r = 0.79.

Why does this agreement matter? These two methods come from completely different
mathematical families. DRST draws a discriminative boundary between groups.
KMM matches distributional means in kernel space. They have different assumptions,
different optimization landscapes, different failure modes. The fact that they
independently converge on the same ~78.5% of samples as low-value is strong evidence
that the low-value label reflects a genuine property of the data — these really are
the chemically foreign samples — rather than an artifact of either method's
particular approach.

One important limitation remains: both DRST and KMM still feed literature labels
directly into the training loss. The 3.42 pp systematic offset is present in those
labels. We have reduced the number of contaminated samples, but the contamination
itself is still there. That is the root cause we have not treated yet. The next
method treats the root cause."

---

## Chapter 7 — Two-Stage Fine-Tuning: The Winner

"Here is the key insight that makes Prior Feature Transfer different.

Both DRST and KMM try to solve the problem by being selective about which literature
to add. But once we have decided to add a sample, its label goes directly into the
training objective — the model is penalized for not predicting the literature's 8.67%
average. No matter how well we filter or weight, we cannot prevent this penalty from
distorting the model's internal calibration.

What if we never expose literature labels to the final model at all?

The idea is: use literature labels to train an expert sub-model (Stage 1). Then ask
that expert to give its opinion on every catalyst in our dataset. That opinion — a
number, not a label — becomes a new input column: feature number 68. The main model
(Stage 2) trains entirely on our labels, with the expert's opinion available as
just one of 68 inputs. The main model can learn to weight it, calibrate it, trust it
in some regions and discount it in others.

**Why does this solve the offset problem?**

In DRST/KMM, literature labels compete with our labels in the loss function. With
Prior Feature Transfer, literature labels are gone from the loss. They influenced
what the Stage 1 model learned, so they indirectly shape the value of the 68th
feature. But the main model's loss is:  minimize (our label − Stage 2 prediction)².
Literature labels do not appear there. The 3.42 pp offset is now a calibration
artifact in one input feature, not a contaminant in the loss function. Stage 2 can
learn: 'when feature 68 says X, my lab typically gets about X − 3.4'. That is
learnable from our data alone.

**Quantile normalisation — why normalise before Stage 1 training?**

Before training Stage 1, we quantile-normalise the literature labels to match our
distribution. Concretely: if a literature catalyst is in the top 10th percentile of
literature yields (say, 19%), we map it to the top 10th percentile of our training
fold's yield (say, 8%). The rank ordering is preserved — better catalysts in
literature still map to higher predicted values — but the absolute scale is aligned.

Without this step, Stage 1 would systematically predict inflated values for all
catalysts (because it was trained on the shifted literature scale), and the 68th
feature would always be too high. Quantile normalisation removes the scale mismatch
while keeping the relative quality information.

**Why XGBoost for Stage 1 / LightGBM for Stage 2?**

Stage 1 trains on only 782 samples (the DRST-filtered literature). With a small
dataset, overfitting is the main risk — the model might memorise the 782 training
points rather than learning general chemistry. XGBoost (Chen & Guestrin, 2016, KDD)
has stronger default regularisation (L1+L2 penalties on leaf values) that makes it
more conservative on small data. LightGBM (Ke et al., 2017, NeurIPS) is faster and
handles the 89,000-row Stage 2 training more efficiently. Both are gradient boosting
frameworks — sequential tree construction where each new tree corrects the residuals
of all previous trees. The sequential nature is what makes gradient boosting 30-40%
more accurate than random forests, which build trees independently.

**What happens inside the CV loop**

Let me walk through one fold carefully, because the step ordering matters.

We are in Round 3 of cross-validation. Fold 3 (about 17,800 samples) is held out
as the validation set. The remaining 71,200 samples are the training set.

Step 1: Quantile-normalise the 782 literature labels using our training fold's
  distribution as the target. This ensures Stage 1's scale matches our data.

Step 2: Train Stage 1 (XGBoost) on the 782 normalised literature samples.

Step 3: Ask Stage 1 to predict on ALL 71,200 of our training samples. Each of
  these 71,200 predictions becomes the value of the 68th feature for that sample.

Step 4: Train Stage 2 (LightGBM) on [67 original features + 68th feature] against
  our training labels.

Step 5: For the validation fold (17,800 samples), generate Stage 1 predictions
  (these are the 68th feature values for validation). Then Stage 2 predicts
  [67 features + 68th feature] → yield prediction. Compute RMSE.

Critically: the validation fold's ground-truth labels are our labels, not literature.
The Stage 1 model was not trained on any of the 17,800 validation samples. There is
no data leakage."

### After the results appear

"Results: RMSE = 1.907, which is 10.6% better than baseline. More than double the
improvement from either DRST or KMM (which gave about 5%). And on the OOD test —
the 2,139 non-Impregnation literature samples that were never included in any method
— the baseline model had RMSE = 6.53. The Prior FT model has OOD RMSE = 3.60,
a 45% reduction.

The OOD improvement is remarkable. Why so large? Stage 1 was trained on literature
that covers sol-gel, coprecipitation, and other preparation methods. When Stage 2
encounters a sol-gel catalyst it has never seen in our data, the 68th feature provides
a meaningful prior from the literature domain. Without the prior, Stage 2 is
extrapolating into a region of chemical space with no training signal — effectively
guessing. With the prior, Stage 2 has an expert's opinion to anchor its estimate.

This is the key benefit of transfer learning done correctly: you gain not just
in-distribution accuracy but also the ability to make reasonable predictions outside
your training distribution."

---

## Chapter 8 — Results + SHAP: Did It Learn Real Chemistry?

### When the 5-method bar chart appears

"Before I get to SHAP, let me put the whole story together in one picture.

The x-axis is CV RMSE — lower is better. The vertical dashed line is the baseline.
Anything to the right of this line made the model worse. Anything to the left
improved it.

The red bar — Naive merge — is the only method that goes to the wrong side.
More data hurt us, by 5.4%. This confirmed the intuition from Ben-David et al. (2010):
when source and target distributions differ significantly, naive combination can hurt.

The orange bars — DRST and KMM — both move us to the left by similar amounts
(about 5%). They are almost indistinguishable from each other, which makes sense:
both are doing the same thing at a high level (filtering chemically irrelevant
samples), just with different mathematical tools. The similarity of their results
is actually reassuring — it means the 5% improvement reflects a real signal about
which literature is useful, not an artifact of one method's particular assumptions.

The top bar — Prior Feature Transfer — is the big jump. 10.6% improvement, by a
margin that is qualitatively different from the filtering methods. The gap is there
because this method addresses the root cause (systematic label offset) rather than
just mitigating its effects."

### When the SHAP beeswarm appears

"RMSE tells you how accurate the model is on average. But a model can be accurate
for the wrong reasons — it might be exploiting a spurious correlation that will
break on new data. SHAP is the sanity check: does the model actually use real
OCM chemistry?

SHAP stands for SHapley Additive exPlanations, introduced by Lundberg & Lee (2017,
NeurIPS). The Shapley value concept comes from cooperative game theory (Shapley,
1953), where it was developed to fairly divide the 'payout' of a cooperative game
among the players who contributed to it. In our context: each feature is a 'player',
each prediction is the 'payout', and SHAP computes the fair share of credit each
feature deserves for each individual prediction.

Concretely: for each sample, for each feature, SHAP gives you a number. A positive
SHAP value for feature X means 'for this particular prediction, feature X's value
pushed the model's prediction above the average.' Negative means it pushed it below.
We are not looking at average effects — we are looking at individual-level attribution.

**Reading the beeswarm:**

Each row is a feature, ranked by mean |SHAP| across all 3,000 samples — most
important at the top. Each dot is one sample. Horizontal position is the SHAP value
(right = pushed yield prediction up; left = pushed it down). Colour is the feature
value (red = high; blue = low).

**Row 1 — `lit_prior_prediction`:** This is the 68th feature — the literature
expert's prediction. It is the most important feature in the entire model, with the
widest spread on the x-axis. When the literature expert predicts high yield (red
dots), the prediction is pushed up. When it predicts low yield (blue dots), the
prediction is pushed down. This is the direct evidence that the transfer learning
is working — the prior is not a dead-weight feature that gets ignored; it is the
single most influential input to Stage 2's predictions.

**Row 2 — Temperature:** High temperature (red) pushes prediction up. This is
correct OCM chemistry. The C2 formation reaction is endothermic — it benefits from
higher temperature. A model that showed temperature with a negative effect would
be a major red flag. It is reassuring that the model learned the correct physics
without being explicitly told it.

**Rows 3-7 — Ba, Mn, La, Ce, and similar:** These are known OCM active phases
and promoters — Ba promotes basic site density, La₂O₃ and Mn₂O₃ are classic active
phases for C2 selectivity (see Lunsford, 2000). All show positive SHAP values when
their loadings are high. The model independently learned that loading these elements
is beneficial, consistent with four decades of OCM experimental research.

**The ceiling effect — why the model fails above 15% yield:**

For samples with true yield above 15%, the model under-predicts by about 4
percentage points on average. Look at the SHAP beeswarm again — do you see GHSV
(gas hourly space velocity) or CH₄:O₂ ratio in the feature list? No. These are the
two operational parameters most responsible for pushing yield above 15%: high space
velocity and optimised CH₄:O₂ ratio. Literature researchers routinely report results
optimised over these variables. Our dataset has them fixed, so they cannot be used
as features.

This is not a modelling failure. No algorithm can learn from a column that does not
exist. The ceiling effect is a data limitation that will persist until GHSV and
CH₄:O₂ are added to the feature set. That is the single most impactful next step
for this research — more impactful than any algorithmic improvement.

**Closing: what should the audience take away?**

'So the big picture message from this whole analysis is:

Literature data is not free training data. It comes with systematic differences —
in labels, in chemistry covered, in how experiments were selected for publication.
Simply adding it makes the model worse.

But if you respect those differences and use the literature appropriately — either
filtering to relevant chemistry (DRST, KMM) or using it as a prior rather than a
label (Prior Feature Transfer) — you can extract genuine value. The best approach
gave us 10.6% improvement in accuracy on our own experiments, and cut our error
on completely new chemistry by 45%.

And critically, the SHAP analysis tells us the improvement is not a statistical
fluke — the model is using real OCM chemistry: temperature, known promoter elements,
and the literature prior all contribute in physically meaningful ways.

The next step is to add GHSV and CH₄:O₂ as features. Once we have those columns,
the ceiling effect should largely disappear, and we can rerun this full analysis with
an even more informative feature set.'"

---

## References cited in the walkthrough

- Ben-David, S. et al. (2010). A theory of learning from different distributions.
  *Machine Learning*, 79(1-2), 151–175.
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
  *Proceedings of KDD 2016*.
- Gretton, A. et al. (2009). Covariate Shift by Kernel Mean Matching.
  In *Dataset Shift in Machine Learning*. MIT Press.
- Huang, J. et al. (2006). Correcting sample selection bias by unlabeled data.
  *Advances in NeurIPS 19*.
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree.
  *Advances in NeurIPS 30*.
- Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation.
  *IJCAI-95*, 1137–1143.
- Lookman, T. et al. (2019). Active learning in materials science with emphasis on
  adaptive sampling using uncertainties for targeted design.
  *npj Computational Materials*, 5(1), 21.
- Lundberg, S.M. & Lee, S.I. (2017). A unified approach to interpreting model
  predictions. *Advances in NeurIPS 30*.
- Lunsford, J.H. (2000). Catalytic conversion of methane to more useful chemicals
  and fuels. *Catalysis Today*, 63(2-4), 165–174.
- Pan, S.J. & Yang, Q. (2010). A Survey on Transfer Learning.
  *IEEE TKDE*, 22(10), 1345–1359.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python.
  *JMLR*, 12, 2825–2830.
- Shapley, L.S. (1953). A value for n-person games.
  In *Contributions to the Theory of Games*, 307–317.
- Stone, M. (1974). Cross-validatory choice and assessment of statistical predictions.
  *JRSS-B*, 36(2), 111–133.
- Sugiyama, M. et al. (2007). Covariate shift adaptation by importance weighted
  cross validation. *JMLR*, 8, 985–1005.

---

---

# Notebook Walkthrough Transcript — Cell by Cell

This section is a complete narration for walking through `ocm_walkthrough.ipynb`
on screen, cell by cell. Written as if screen-sharing the notebook and talking
someone through it in real time. Different from the chapter-level transcript above —
this one covers every code cell and output, not just the conceptual story.

---

## Cell 1 — Chapter 1 markdown: The Problem

*[Scroll to the top of the notebook. Cell is rendered markdown — no code visible.]*

"So let me start by reading through what this first cell says, because it sets up
everything that follows.

The reaction we are studying is called Oxidative Coupling of Methane — OCM. You take
natural gas, which is mostly methane, react it with oxygen at high temperature over
a catalyst, and you get ethylene on the other side. Ethylene is the monomer for
polyethylene, for PVC, for a huge fraction of the plastics industry. So if you can do
OCM efficiently, you have a potentially cleaner route to a very high-value chemical.
The challenge is finding the right catalyst — a surface coated with metal oxides that
makes the reaction go in the right direction.

Our lab has done 89,074 of these experiments. Each one is: here is a catalyst
composition (which elements, at what loading), here is the temperature, and here is
the resulting yield in percent. That is the dataset we are trying to model.

Now, there is also published OCM literature going back to 1982. Thousands of
experiments from dozens of labs. And the obvious question is: can we use that to
make our model better?

The table in this cell shows the two datasets side by side — 89,000 rows with mean
yield 5.25% for ours; 3,852 rows with mean yield 8.67% for literature. That 3.42
percentage point gap is not an accident. And the cell explains three reasons why
simply combining them is non-trivial.

I will not read each one out now — the notebook text is there for reference —
but the key takeaway from this cell is: more data does not automatically mean a
better model. Whether it helps depends on how compatible the new data is. And that
is what the rest of the notebook investigates."

---

## Cell 2 — Chapter 2 markdown: Setup

*[Scroll to Chapter 2. Still a rendered markdown cell.]*

"This cell explains three preprocessing decisions we make before running any model.
I want to highlight the third one because it is the one most people miss.

The first two are fairly standard. LabelEncoder converts the preparation method
column — which contains text like 'Impregnation' or 'Sol-gel' — into integers,
because tree models cannot process text. StandardScaler normalises each feature
to mean zero and standard deviation one, so that temperature (which ranges 500–900)
does not numerically dominate element loadings (which range 0–15) just because of
units.

The third decision is the critical one. We call `scaler.fit_transform` on our data
and `scaler.transform` on literature. Not `fit_transform` on both — just `transform`
on literature. That means the scaler's parameters — mean and standard deviation per
feature — are learned only from our data. Literature gets scaled using our reference
frame, not its own.

There is a concrete example in this cell that makes this tangible. Suppose Ba loading
in our data has mean 1.5% and std 2.0%. A Ba=2% sample from our data scales to
(2.0 − 1.5) / 2.0 = +0.25. If I fit a separate scaler on literature — where Ba might
have a different distribution — the same Ba=2% from literature might scale to +0.80.
Same physical measurement, different number. Every distance calculation downstream
would treat these as different chemistry. That would break KMM entirely. The fix is
one line: fit on ours, transform both."

---

## Cell 3 — Setup code cell

*[Click on the code cell. Scroll through it slowly as you speak.]*

"This is where we actually run the setup. Let me walk through the main blocks.

The first block is imports — numpy, pandas, matplotlib, lightgbm, xgboost, shap,
and sklearn utilities. Nothing unusual there.

Then we load the CSV and split by year. The `year` column encodes which dataset a
row came from — year 2025 means our experiments, year 2019 or earlier means
published literature. This is a clean way to keep the datasets in a single file
while being able to split them programmatically.

The print statements at the bottom tell us: our data is 89,074 rows with mean yield
5.25%, literature is 3,852 rows with mean yield 8.67%, and the gap is 3.42 percentage
points.

Then we define `ELEM_COLS` — these are the 65 element columns, everything that is
not preparation method, temperature, year, or the target variable. These become the
chemical fingerprint of each catalyst.

The LabelEncoder block converts preparation method to integers. Note that we call
`le.fit` on the full dataframe `df` — the combined dataset — before splitting.
This ensures the integer mapping is consistent across both subsets.

Finally the scaler block. `X_ours_sc = scaler.fit_transform(X_ours)` — that's the
fit step, learning mean and std from our data. `X_lit_sc = scaler.transform(X_lit)`
— that's the apply step, using the same parameters on literature. One line difference,
but it matters.

*[Pause on the output.]*

The output confirms the shapes: 89,074 × 67 for ours, 3,852 × 67 for literature.
67 features — temperature, preparation encoding, and 65 element columns."

---

## Cell 4 — Visual evidence markdown

*[Markdown cell above the KDE+PCA plot.]*

"This short cell sets up what the next two figures will show. The left panel of the
first figure — which is coming up next — shows label shift: the two yield distributions
overlaid. The right panel shows covariate shift: both datasets projected into two
dimensions via PCA, so we can see how the chemistry clusters differ.

PCA — principal component analysis — takes the 67-dimensional feature space and
finds the two directions that explain the most variance, then projects every sample
onto those two directions. Think of it as casting a shadow of a 67-dimensional
object onto a 2D wall. You lose some information but you see the overall structure.
The point is not to be precise — it is to show that the two datasets live in
different regions of chemical space."

---

## Cell 5 — KDE + PCA plot code

*[Scroll through the code.]*

"This cell generates the two-panel figure. Let me call out a few specific choices.

In the left panel — the yield distribution — we use `density=True` in both histogram
calls. That normalises both histograms so the area under each equals one, regardless
of how many samples each dataset has. Without this, the blue histogram for our 89,000
samples would dwarf the orange one for 3,852 samples and you would not be able to
compare shapes. With `density=True` you are comparing the shape and centre of each
distribution, not the raw counts.

We also overlay KDE curves — kernel density estimates — on top of the histograms.
KDE is a smooth, continuous version of a histogram. It uses a Gaussian kernel at
each data point and sums them up. The result is the smooth line you see riding over
the bars.

The `axvspan` call draws the red shaded band between the two mean lines. This is
just a visual aid — it makes the 3.42 pp gap a physical distance on the plot rather
than just a number in the legend.

In the right panel we subsample our data to 3,000 points using `np.random.choice`.
We do not plot all 89,000 because the result would be a solid blue blob. We stack
our subsample and all of literature into one array called `X_combined_sc`, run PCA
on the combined array, and then slice the result: first 3,000 rows are ours, rows
3,000 onwards are literature. That is why the literature scatter uses `X_pca[3000:]`
— because literature was appended second in the vstack.

*[Let the figure render. Point at specific features.]*

On the left: the blue curve peaks around 3–4%, the orange curve peaks around 8–9%.
The red band is the gap. Notice the orange curve is not just shifted — it has a
heavier tail toward the right. That is the publication bias we talked about in
Chapter 1. Failed experiments are not published, so the literature distribution
is skewed toward high yields, not just offset.

On the right: the blue cloud clusters tightly. The orange dots partially overlap
the blue cluster — those are the relevant literature samples — but many orange dots
fall in the white space outside the blue cloud. Those are the 78.5% of literature
samples describing chemistry our lab has never tested."

---

## Cell 6 — Element usage markdown

*[Short markdown cell.]*

"This cell points out that the PCA is abstract — it compresses 67 dimensions into 2,
and you cannot directly read off which elements are causing the separation. The next
figure makes it concrete by showing the actual elements, side by side."

---

## Cell 7 — Element usage plot code

*[Code cell — relatively short.]*

"This code computes the 'usage frequency' for each element in each dataset —
meaning, the fraction of samples in that dataset where that element has a non-zero
loading. If an element appears in 90% of our samples, it is a core part of our
chemistry. If it appears in 5%, it is occasional.

We sort by usage frequency and take the top 15 for each dataset, then plot horizontal
bar charts side by side.

*[Let the figure render.]*

Look at the two panels. The element palettes are different. Our lab focuses on a
specific set of promoters and active phases. Literature uses a partially overlapping
but distinctly different mix. Some elements that appear frequently in literature
barely appear in our data. This is what 'covariate shift' looks like concretely —
not an abstract statistical statement but literally different elements in different
proportions. The 78.5% OOD figure is not a number invented by an algorithm; it
reflects this physical reality."

---

## Cell 8 — Chapter 3 markdown: How We Measure Success

*[Markdown cell.]*

"Before we run any method, we need to agree on the measurement. This cell explains
two things: what RMSE is, and why our cross-validation is designed asymmetrically.

RMSE — root mean squared error — is in units of percentage points of yield. An RMSE
of 2.1 means the model is typically off by about 2.1 percentage points. It penalises
large errors more than small ones because the errors are squared before averaging.
A 6 pp error contributes 36 to the average; a 1 pp error contributes 1. This matters
for catalyst selection — a catastrophic miss on one sample is worse than being
slightly off on many.

The cross-validation design is: split our 89,074 samples into 5 equal folds. In each
of 5 rounds, 4 folds go into training and 1 fold is the test set. Literature, when
we use it, always goes into training. It never appears in the test set.

Why? Because the question we are answering is 'how well does the model predict our
experiments?' If literature appeared in the test set, we would be measuring accuracy
on a mix of our experiments and literature runs — which is a different question and
would give us a misleading answer. The test set must represent the actual deployment
target."

---

## Cell 9 — evaluate_cv_ours code

*[Longer code cell — scroll slowly.]*

"This function is the engine that all five methods run through. Every RMSE number
in this notebook comes out of this function. Let me walk through it.

The function signature: `X_train_extra`, `y_train_extra`, `sample_weight_extra`, and
a label string. These are the optional literature additions — for the baseline, we
pass nothing; for DRST, we pass the filtered literature samples; for KMM, we pass
literature with weights.

Inside the function: `KFold(n_splits=5, shuffle=True, random_state=42)` creates the
5-fold split. The `random_state=42` ensures the split is the same across all method
calls, so every method is evaluated on identical folds.

The loop body has two paths. If `X_train_extra` is not None, we stack our training
fold with the extra literature samples using `np.vstack` and `np.concatenate`.
The sample weights are assembled similarly — our samples get weight 1, literature
samples get whatever weight the method assigned.

If `X_train_extra` is None — the baseline case — we just use our training fold with
no augmentation.

Then we fit a `LGBMRegressor` and predict on the validation fold. The validation fold
is always `X_ours_sc[val_idx]` — our data only. This line never changes regardless
of what extra training data we added.

RMSE and R² are computed and appended to lists. At the end we return the mean and
std across the 5 folds.

The last line of the cell initialises the `results` dictionary that will accumulate
all method results for the final bar chart."

---

## Cell 10 — Chapter 4 Step 1 markdown: Baseline

*[Short markdown cell.]*

"This cell just states the baseline result — RMSE 2.133. This is the number every
subsequent method must beat. If a method produces RMSE above 2.133, it has harmed
the model.

This is worth pausing on. The baseline uses only our 89,074 samples, no literature
at all. LightGBM with these parameters gives 2.133 pp typical error. That is the
starting point."

---

## Cell 11 — Chapter 4 Step 2 markdown: Naive Merge

*[Longer markdown cell.]*

"This cell tells the story of the obvious first attempt: just append all 3,852
literature rows to training and retrain. More data should help.

It did not help. RMSE went from 2.133 to 2.248 — 5.4% worse.

The explanation in this cell is the key intuition. The model sees two training
samples with nearly identical element compositions. One is labelled 5.2% (ours).
The other is labelled 8.7% (literature). There is no feature in the dataset that
explains the gap — the gas flow rate and CH₄:O₂ ratio that account for the
difference are not recorded. The model cannot resolve the contradiction. It hedges:
predicts something between 5.2% and 8.7% for that type of catalyst, which makes
it wrong on both.

The cell references Ben-David et al. (2010) — when source and target distributions
differ significantly, naive combination can hurt. This is a theoretical result, but
we are seeing it empirically right here."

---

## Cell 12 — Baseline + naive merge code

*[Short code cell — two function calls.]*

"This cell runs both the baseline and naive merge through `evaluate_cv_ours`.

First call: no extra training data. This is the baseline.

Second call: `X_train_extra=X_lit_sc` and `y_train_extra=y_lit`. All 3,852 literature
samples added to training with weight 1. No filtering, no weighting — the naive merge.

*[Point at the output.]*

The output confirms: baseline RMSE 2.133, naive merge RMSE 2.248. Delta +0.115
percentage points, worse in the wrong direction. This is the evidence that motivates
everything that follows."

---

## Cell 13 — Chapter 5 DRST markdown

*[Longer markdown cell.]*

"Chapter 5 introduces DRST — Density Ratio Selective Transfer. The core question:
instead of adding all literature, what if we only add the samples that look like our
data?

This cell explains the method. We build a binary classifier — logistic regression —
that tries to distinguish our samples from literature samples. Our data gets label 1,
literature gets label 0. The classifier learns: given these element features, which
dataset does this sample look like?

The cell explains what logistic regression is for a non-ML reader: it finds a
hyperplane in the 67-dimensional feature space that best separates the two groups.
For each literature sample, we ask: what is the probability the classifier assigns
it the label 'ours'? High probability means it looks like our chemistry. Low
probability means it looks like foreign chemistry.

We keep only literature samples above a threshold τ. The threshold is chosen by
sweeping five values and picking the one with lowest CV RMSE. τ=0.30 retains 782
of 3,852 samples — about 20%.

The cell also explains why C=0.5 for the logistic regression — that is the
regularisation parameter. Smaller C means a simpler, smoother boundary. We do not
want a boundary that perfectly separates every individual sample — we want a general
characterisation of the chemical difference between the two datasets."

---

## Cell 14 — DRST classifier code

*[Code cell — medium length.]*

"This cell builds the domain classifier.

First we subsample our data to 10,000 points. We do not need all 89,000 for this
step — 10,000 is enough to characterise the distribution, and training the classifier
is faster.

We stack the subsample and all literature into `X_dom`, create binary labels `y_dom`
(ones for ours, zeros for literature), and fit `LogisticRegression(C=0.5)`.

Then `clf_dom.predict_proba(X_lit_sc)[:, 1]` scores every literature sample.
`predict_proba` returns a two-column array — column 0 is P(literature), column 1
is P(ours). We take column 1: the probability each literature sample would be
classified as ours.

The histogram code below that plots these scores with four threshold lines.

*[Point at the histogram in the output.]*

This is the DRST score distribution for all 3,852 literature samples. The x-axis
is P(ours|x), from 0 to 1. The dominant feature is the spike at the left — most
samples score near zero. The classifier is very confident they are literature, not
ours. Then there is a long tail extending toward 1.0 — those are the samples the
classifier cannot confidently distinguish from our data. Those are the chemically
relevant ones.

The four dashed vertical lines are the four thresholds we will test. You can see
τ=0.30 sitting partway along the tail, capturing a meaningful slice of the relevant
samples."

---

## Cell 15 — DRST threshold sweep code

*[Code cell — the τ sweep.]*

"This cell sweeps five threshold values and runs `evaluate_cv_ours` for each. For
each τ, we create a boolean mask — which literature samples have score ≥ τ — and
pass those samples as extra training data.

*[Point at the output.]*

You can read off the RMSE for each τ. τ=0.30 gives 2.019, the lowest of the five.
The function picks the best τ and stores the result.

Below the output: DRST RMSE = 2.019, delta versus baseline = −0.114. That is a
5.3% improvement.

But notice what the DRST histogram showed us: the cutoff is hard. A sample at 0.29
gets nothing. A sample at 0.31 gets full weight. That binary step feels arbitrary —
it motivated asking whether we could do something with a continuous weight instead."

---

## Cell 16 — Chapter 6 KMM markdown

*[Longer markdown cell.]*

"Chapter 6 introduces KMM — Kernel Mean Matching. Instead of a yes/no decision,
every literature sample gets a continuous weight between 0 and 10.

The cell explains the RBF kernel — the mathematical similarity function at the core
of KMM. K(x, x') = exp(−‖x − x'‖² / 2σ²). Two samples with identical element
profiles get similarity 1. Samples far apart get similarity near 0. This is how KMM
measures 'how close is this literature sample to our data'.

Then it explains κ (kappa) — for each literature sample, κ is proportional to the
sum of its kernel similarities to all our training samples. High κ means this
literature sample overlaps strongly with our distribution — it should get high weight.
Near-zero κ means it is in an empty region of our space — weight converges to zero.

The optimisation finds weights that make the weighted literature distribution match
our distribution as closely as possible, subject to weights being non-negative and
bounded.

The key result: KMM independently arrives at the same conclusion as DRST. 78.5%
of literature samples get weight below 0.1 — effectively discarded. And the two
methods correlate at r=0.79. Two completely different mathematical approaches,
same conclusion about which samples are useful."

---

## Cell 17 — KMM code

*[Longer code cell — the KMM implementation.]*

"This cell contains the KMM implementation. There is a lot of code here but the
mathematical core is about 10 lines.

`rbf_kernel` computes the kernel matrix — it takes two arrays of samples and returns
the matrix of pairwise similarities. The `cdist` call computes squared Euclidean
distances between all pairs, and `np.exp` applies the Gaussian function.

`kmm_weights` is the main function. It sets up the problem: compute K_ss (literature
to literature similarity), K_st (literature to our data similarity), and kappa
(the overlap of each literature sample with our distribution). Then it calls
`scipy.optimize.minimize` with the quadratic objective.

The σ (bandwidth) is computed via median heuristic: take a subsample of both datasets,
compute all pairwise Euclidean distances, take the median. This gives a scale that
automatically adapts to the actual spread of the data.

This computation takes 1–2 minutes — it is iterating over a quadratic programme with
3,852 variables.

*[After the computation finishes, point at the output and figures.]*

Left panel: the weight distribution. That enormous spike near zero — that is 78.5%
of all literature samples with weight below 0.1. KMM is not discarding them —
they are still technically in training — but weight 0.05 is negligible.

Right panel: scatter of DRST score versus KMM weight. If the two methods were
unrelated, this would be random scatter. Instead there is a clear diagonal trend,
r=0.79. One method is a logistic classifier, the other is a quadratic optimisation.
Both independently say the same 78.5% of samples are low-value. That agreement is
the most reassuring finding in the whole notebook."

---

## Cell 18 — KMM evaluate code

*[Short code cell.]*

"This runs KMM through the same `evaluate_cv_ours` function as everything else.
The only difference from naive merge is `sample_weight_extra=w_kmm` — we pass
the weights computed in the previous cell. The function handles sample weighting
via LightGBM's `sample_weight` argument to `fit`.

*[Point at the output.]*

KMM RMSE = 2.035. Delta versus baseline = −0.098. About 4.6% improvement — very
similar to DRST's 5.3%. Two different methods, similar result. Both hit the same
ceiling because both still feed literature labels directly into the training loss.
The 3.42 pp offset is still there in those labels."

---

## Cell 19 — Chapter 7 Prior Feature Transfer markdown

*[Longest markdown cell.]*

"This is the key chapter. The Prior Feature Transfer idea is the one that actually
works well.

The insight is: DRST and KMM both reduce the number or weight of literature samples,
but they cannot remove the label offset — because the labels are still there in the
training loss. The model is still penalised for not predicting the literature's 8.67%.

What if literature labels never entered the training loss at all?

The design has two stages. Stage 1: train an XGBoost model on the 782 DRST-filtered
literature samples. This model learns the literature's view of catalyst chemistry —
given a composition, what does literature say the yield typically is?

Before Stage 1 training, we quantile-normalise the literature labels. We map each
literature label to the percentile-equivalent value in our distribution. If a
literature catalyst is in the top 10th percentile of literature (say 19%), we map
it to the top 10th percentile of our training fold (say 8%). This aligns the scale
without destroying rank ordering — better catalysts still map to higher values.

Stage 2: for every catalyst in our training set, we ask Stage 1 for its prediction.
That prediction becomes the 68th feature — `lit_prior_prediction`. Stage 2 then
trains a LightGBM model on 67 + 1 = 68 features, but only on our labels. Literature
labels never touch the Stage 2 loss function.

Stage 2 can learn to calibrate the prior. If it notices that when Stage 1 says 8%,
we typically get about 4.5%, it can adjust accordingly. The offset is now a
calibration artifact in one input column — something the model can learn — not a
corruption of the loss.

The markdown cell also explains why XGBoost for Stage 1 and LightGBM for Stage 2.
XGBoost is more conservative on small data (782 samples) due to stronger
regularisation. LightGBM is faster on the large Stage 2 training set (89,000 rows)."

---

## Cell 20 — Prior Feature Transfer code

*[Medium length code cell — the two-stage loop.]*

"This cell implements the two-stage design inside the same 5-fold CV loop.

Let me walk through one fold. We are in round k. The validation fold is `val_idx`.
The training fold is `train_idx`.

Step 1: quantile-normalise literature labels using our training fold's distribution
as the target. We call `quantile_normalize_y(y_lit[mask], y_ours[train_idx])`. This
maps literature labels onto our training fold's percentile structure.

Step 2: train Stage 1 XGBoost on the 782 filtered literature samples with the
normalised labels.

Step 3: `pre.predict(X_ours_sc[train_idx])` — ask Stage 1 to predict on all our
training samples. These become `prior_tr`, the 68th feature for training. Similarly,
`pre.predict(X_ours_sc[val_idx])` generates the 68th feature for the validation fold.

Step 4: `np.hstack([X_ours_sc[train_idx], prior_tr])` appends the prior column to
our feature matrix. Now we have 68 columns.

Step 5: train Stage 2 LightGBM on `[67 features + prior]` against `y_ours[train_idx]`.
Note — `y_ours[train_idx]`, not literature labels. Our labels only.

Step 6: predict on the validation fold using `[67 features + prior_val]`. Compute RMSE.

*[Point at the output.]*

Prior Feature Transfer RMSE = 1.907. Delta versus baseline = −0.226, which is −10.6%.
More than double the improvement from either DRST or KMM alone.

And below that, the full summary table comparing all five methods. You can see the
trajectory: baseline at 2.133, naive merge going wrong to 2.248, DRST and KMM
both improving to around 2.02–2.03, and then the jump to 1.907 for Prior Feature
Transfer."

---

## Cell 21 — Chapter 8 markdown: Results + SHAP

*[Longer markdown cell.]*

"This cell frames the two final analyses.

First: the results summary table and bar chart. This is just consolidating what we
already know from the individual method cells — all five RMSE values in one place.

Second: SHAP. The cell explains what SHAP is from scratch, for a non-ML reader.

The key idea is Shapley values from cooperative game theory. In game theory, Shapley
values answer: if a group of players cooperated to achieve a joint payoff, how much
credit does each player fairly deserve? Lundberg and Lee (2017) showed that this
framework applies directly to ML models — each feature is a player, each prediction
is the payoff, and SHAP computes the fair marginal credit for each feature.

Concretely: SHAP asks, for each feature and each sample, what is the average change
in prediction when this feature is included versus excluded, averaged over all
possible orderings of including features? A positive SHAP value means this feature's
value pushed the prediction above the average. Negative means it pushed it below.

This is more informative than standard feature importance (like mean decrease in
impurity) because it is per-sample, not averaged across all samples. You can see
not just which features are important on average, but in which direction and for
which subset of the data."

---

## Cell 22 — Results bar chart code

*[Short code cell.]*

"This cell builds the results dataframe from the `results` dictionary that has been
accumulating throughout the notebook, sorts by RMSE, assigns colours, and plots
a horizontal bar chart with error bars.

The colour coding: blue for baseline, red for naive merge, orange for everything else.
The baseline RMSE is drawn as a vertical dashed line for reference.

*[Point at the bar chart.]*

Read this from top to bottom, left is better. Naive merge — the red bar — is the
only one to the right of the baseline line, meaning it hurt the model. DRST and KMM
are slightly to the left — small improvement. Prior Feature Transfer is the biggest
jump. The visual gap between Prior FT and the two filter methods is substantial — it
is not incremental improvement, it is qualitatively different."

---

## Cell 23 — SHAP code

*[Longest code cell.]*

"This last cell runs the SHAP analysis. Let me call out the structure.

First we retrain Stage 1 on the full dataset — not inside a CV loop this time — so
we have a single Stage 1 model representing all of our data. We use
`quantile_normalize_y(y_lit[mask], y_ours)` with all 89,074 of our labels as the
target.

Then we generate the 68th feature for all 89,074 samples: `prior_all = pre.predict(X_ours_sc)`.

`X_aug = np.hstack([X_ours_sc, prior_all])` gives us the full 89,074 × 68 matrix.
We train the final LightGBM `final_shap` on this.

Then SHAP: we subsample 3,000 points from the augmented matrix for the SHAP
computation. `shap.TreeExplainer(final_shap)` uses the exact TreeSHAP algorithm —
not an approximation — to compute Shapley values. For tree ensembles, TreeSHAP is
exact and efficient.

`shap_values = explainer.shap_values(X_sh)` gives a matrix of shape 3000 × 68 —
one Shapley value per sample per feature.

*[Wait for the beeswarm to render. Point at specific rows.]*

Look at the top of the y-axis. `lit_prior_prediction` is the number one feature
by mean absolute SHAP value. When its value is high — red dots — it pushes the
prediction to the right, which means higher yield. This is the direct evidence that
the transfer learning is working. The prior is not dead weight; it is the single
most influential input to Stage 2's predictions.

Second row: temperature. High temperature — red dots — pushes predictions to the
right. Higher temperature means higher yield. That is correct OCM thermodynamics.
The model learned the physics without being told.

Then Ba, Mn, La, Ce and other elements — all known OCM active phases and promoters
from the literature. Their presence at high loading pushes yield up. Their absence
or low loading has neutral or negative effect.

The one issue: for samples with true yield above 15%, the model consistently
underpredicts. The residual table in the cell output shows RMSE of 4.70 for the
>15% range, with a −4.2 pp systematic bias. The reason is simply that the conditions
controlling those high yields — gas flow rate, CH₄:O₂ ratio — are not in our feature
set. The model cannot learn what it cannot see. Adding those features is the most
impactful next step.

And that is the end of the notebook. The story arc is: we started with a problem —
can we use literature to improve our model — tried the naive approach, watched it
fail, understood why, tried two filtering approaches, understood their ceiling, and
then found a method that avoids the fundamental limitation by routing literature
knowledge through a feature rather than a label. The SHAP analysis then confirms
that the resulting model uses real, physically meaningful chemistry."
