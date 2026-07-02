# Verbatim Presenter Script — Chapters 9 & 10

First-person spoken script. Every claim is tied to a specific number or reason — nothing vague.
Numbers reflect the **corrected, leak-free** analysis (see `qn_tradeoff.json`, `feedback_results.json`).

Key numbers to keep straight:
- Baseline (no transfer), asymmetric 5-fold CV: **RMSE 2.133**.
- Prior Feature Transfer (PFT = DRST-filtered + quantile-normalised prior): **RMSE 1.907** (−10.6%).
- 10-seed repeat: baseline **2.121 ± 0.006** vs PFT **1.909 ± 0.002**, PFT wins **10/10**, paired-t **p = 3.9×10⁻¹⁵**.
- True held-out 20% lab (never trained on): baseline **2.097** → PFT **1.892** (R² 0.763).
- **OOD (corrected, leak-free)**: baseline **6.53**; honest transfer configs **6.05–6.77**. The old
  "3.60 (−45%)" was **leakage** — it reproduces (3.62) *only* when the prior is trained on the OOD
  test rows themselves.
- QN trade-off: quantile normalisation **improves in-distribution CV** and **slightly worsens OOD**.

---

## CHAPTER 9 — Critical Analysis & What's Next

### 9.0 — Why this chapter exists

"Before I show what's next, I want to be openly critical of what we built. A result you can't
attack is a result you don't understand. So in this chapter I'll do three things: grade each method
honestly — including the ones that failed; state the limitations that bound every number I've shown
you; and then lay out the next steps in priority order. I'll also correct one number from the
earlier version of this deck, because I found a subtle evaluation flaw and I'd rather flag it myself
than have it found for me."

### 9.1 — Honest review of each method

**Naive merge.**
"The naive merge — just pooling all 3,852 literature rows with our 89,000 — is the one method that
made things *worse*: RMSE went from 2.133 up to 2.248, about 5% worse than baseline. I keep it in
the deck on purpose, as a negative control. It proves two things: first, the domain shift between
literature and our lab is real and harmful, not cosmetic; and second, that any benefit we later get
is *because* of the correction machinery, not just because we added more rows. More data is not
automatically better data."

**DRST — density-ratio filtering.**
"DRST trains a logistic classifier to score each literature sample by how much it looks like our
chemistry, and keeps the ones above a threshold. Its strength is that it's simple, fast, and gives
one interpretable number per sample. Its honest weaknesses are three. One: it's a hard cliff — a
sample at 0.31 is kept and one at 0.29 is discarded, even though they're almost identical. Two: at
our threshold it throws away roughly 80% of the literature. Three — and this is the one a referee
will catch — the threshold was chosen on the *same* cross-validation we then report, so any
single-stage DRST gain is mildly optimistic. When I swept the threshold cleanly from 0 to 1, the
best single-stage filtering barely moved RMSE below baseline. Filtering alone is not where the win
is."

**KMM — kernel mean matching.**
"KMM is the more principled cousin: instead of a hard cut, it assigns every literature sample a
continuous weight so that the *weighted* literature distribution matches ours. No cliff. But it has
its own costs. It builds an n-by-n kernel matrix, so it's slow — a minute or two — and it's
sensitive to the kernel bandwidth. More importantly, it still feeds the literature *labels* into
the loss, so it can't escape the label shift. And empirically it lands almost exactly where DRST
lands — the two agree at correlation 0.79, with 81% of the kept samples overlapping. When two
different methods tie, that tells you filtering has hit its ceiling; the remaining error isn't a
filtering problem."

**Prior Feature Transfer — the method that worked.**
"PFT is the one that actually breaks through, from 2.133 to 1.907 — a 10.6% reduction. The reason
it works is structural: the literature never enters the final model as a *label*, only as a
*predicted feature*. A Stage-1 model learns the literature's view of chemistry; then Stage-2 — our
model — trains only on our labels, with the Stage-1 prediction added as one extra feature. That
means the 3.4-point label offset lives inside a feature value, where Stage-2 can calibrate around
it, instead of inside the loss, where it would poison every gradient. Its honest limitations: the
Stage-1 expert is trained on only 782 samples, so it's a bit high-variance; it assumes the quantile
map preserves the *ranking* of good and bad chemistry across domains; and it's two models to
maintain instead of one."

### 9.2 — The correction I owe you: the OOD number

"Now the correction. The earlier version of this deck claimed a 45% improvement out-of-distribution
— RMSE 6.53 down to 3.60 — on literature made by preparation methods our lab never uses. That
number was too good, and here's exactly why. In that experiment, the Stage-1 prior was trained on
*all* literature, which *includes* the very out-of-distribution samples we then tested on, together
with their true yields. So the prior had effectively memorised the answers and handed them to the
final model through the feature. That's leakage. When I retrain the prior with those OOD rows
*removed* — a genuinely blind test — the honest OOD RMSE is about 6.0 to 6.8 depending on the
configuration: a small improvement over baseline at best, and for our best in-distribution model,
essentially level with baseline. I reproduced the old 3.60 exactly (3.62) *only* by putting the
leak back, which confirms that leakage — not skill — produced it."

"I want to be clear about what this does and doesn't change. It does **not** touch the headline: on
our own chemistry, transfer learning gives a solid, repeatable 10% gain, confirmed across ten random
seeds at p below ten-to-the-minus-fourteen and on a fully held-out test set. What it changes is the
*extrapolation* claim: we should not sell this as a model that generalises dramatically to
unfamiliar catalyst families. It's a model that's very good in our operating regime."

### 9.3 — The quantile-normalisation trade-off (the deeper lesson)

"This ties directly to the quantile-normalisation question from earlier. Quantile normalisation
rescales the literature labels onto our yield scale. When I turn it on and off cleanly, holding
everything else fixed, the pattern is consistent: with quantile normalisation, in-distribution RMSE
is *better* — because the prior speaks in our units — but OOD RMSE is *worse*, because the prior can
no longer reach the genuinely higher yields of unfamiliar literature. Without it, the reverse. So
quantile normalisation is not a free win; it's a *dial* that trades extrapolation for local
accuracy. We chose local accuracy on purpose, because our goal is to predict *our* next experiment,
not to re-derive the whole literature."

### 9.4 — Cross-cutting limitations

"Three limitations sit above any single method. First, evaluation lens: every method is scored on
our-data cross-validation, and only PFT was stress-tested OOD — so I can't claim the filtering
methods generalise, I simply didn't test them there. Second, no uncertainty: the model gives point
predictions, but for deciding which experiment to run next, confidence matters as much as the number
itself. Third, and biggest, the ceiling above 15% yield: the model under-predicts the very best
catalysts by about 4 points, and that's a *data* limit, not an algorithm limit. The variables that
push yield past 15% — gas hourly space velocity, the methane-to-oxygen ratio, pressure — are simply
not columns in our dataset. No model can learn from a column that isn't there."

### 9.5 — What's next, in priority order

"So the roadmap, ordered by expected impact. Number one, and far ahead of the rest: add the
reaction-condition features — space velocity, methane-to-oxygen ratio, pressure. That directly
attacks the ceiling, and I suspect it also shrinks the label shift itself, because a lot of the
literature's higher yields are really just better-optimised conditions we currently can't see. Two:
bag the Stage-1 expert — train several on bootstrapped literature and average them, to cut the
variance from that small 782-sample fit. Three: add uncertainty, via quantile or conformal
prediction, so the experiment-selection loop can target high-value, low-confidence candidates. Four:
stack the methods — feed DRST-filtered, KMM-weighted literature into the PFT pipeline instead of
picking one. Five: once the features are richer, try neural domain adaptation like DANN or CORAL.
And six, the endgame that motivated the whole project: close the active-learning loop — let the
model propose the next experiments, run them, and retrain."

---

## CHAPTER 10 — Reviewer Q&A: Answering the Committee, Logically

### 10.0 — Framing

"The committee raised seventeen questions. Rather than answer them as seventeen disconnected points,
I've grouped them into six themes, because each theme is really one concern in disguise. If I satisfy
the theme, I've satisfied the cluster. I'll take them in order, and every answer is backed by either
a figure or a number, both of which are reproducible in the notebook."

### 10.1 — Theme 1: "Did you prepare the data correctly?" (Q1–Q4)

"First, scaling. I used StandardScaler, not MinMax, and no, standardisation does not destroy
linearity — both are straight-line, affine maps; z-scoring can't bend a relationship, it only
recentres and rescales. The only nonlinear rescaling in the whole pipeline is the quantile map on
the *labels*, which is probably what the question was picturing. I chose StandardScaler because it
only matters for the distance-based steps — PCA, the DRST classifier, the KMM kernel — and it's
irrelevant to the tree models that actually predict yield; and because it's robust to the extreme
literature compositions, whereas MinMax would let one outlier crush everything else into a sliver."

"Second, PCA. The PCA values are just each sample's coordinates on the first two principal
components — the two directions of maximum variance — plus the percentage of variance each explains.
I use only two because this is a *picture*, a diagnostic to show the domain gap, not a modelling
step. Two components under-state the gap, if anything, because they can't capture all 67 dimensions;
the rigorous gap measure is the DRST classifier, which says 78.5% of the literature is
out-of-distribution. And to be precise about a common misunderstanding: I fit the PCA on the
*combined* set, because you can only judge whether two clouds overlap if they're drawn on the same
axes. Finally, the '3,000' in that plot is a subsample of *our 89,000 lab points*, taken purely so
the scatter is legible — all 3,852 literature rows are used; nothing is thrown away."

### 10.2 — Theme 2: "Did you pick methods and thresholds honestly?" (Q5, Q6, Q8)

"On the threshold: I didn't pick 0.30 by eye, I swept every value from 0 to 1. Two findings. Single-
stage filtering barely helps at any threshold — the best point is only a hair below baseline, and at
0.30 it's actually slightly worse. But the two-stage PFT is *flat* across the threshold — every
value from 0.05 to 0.8 lands within a hundredth of an RMSE point of the best, all about 10% below
baseline. That's the important part: 0.30 is a safe, near-optimal choice, and the win comes from the
*architecture*, not from tuning a knob. If I'd cherry-picked the threshold, the curve would be
sharp; it's flat, which means I didn't."

"On KMM versus DRST: they're not the same 782 samples by construction — one hard-cuts, one soft-
weights all 3,852 — but they read the same signal. Their correlation is 0.79, and DRST's kept set
overlaps KMM's top-782 by 81%. They corroborate each other, which is exactly what you want: it means
the covariate-shift signal is real and not an artefact of one method."

### 10.3 — Theme 3: "Is the result real, or luck?" (Q7, Q11, Q14–Q16)

"Did I use all the data — yes: all 89,000 lab rows, through asymmetric five-fold cross-validation.
Every lab row trains in four folds and is scored once, and the validation fold is *always* lab-only;
the literature only ever enters the training side. So the number measures accuracy on *our*
experiments, not a blurred average."

"How many times did I run it — originally once, which was a fair criticism, so I now run it ten
times with ten different random splits. Baseline is 2.121 plus or minus 0.006; PFT is 1.909 plus or
minus 0.002. PFT wins all ten out of ten — the curves never cross, the worst PFT run still beats the
best baseline run. A paired t-test gives p equal to 3.9 times ten-to-the-minus-fifteen, and the
Wilcoxon test agrees. Nothing is tuned per seed, so this isn't force-fitting; it's a genuine,
repeatable effect. And it's not only better, it's *tighter* — a third of the baseline's variance."

"Did I validate on untouched data — yes, I added a true hold-out: a random 20% of the lab, set aside
and never used in any stage of any method. Baseline 2.097, PFT 1.892 — the same 10% gain on data the
model has genuinely never seen. That, not the OOD number, is the honest generalisation evidence, for
exactly the leakage reason I explained in Chapter 9."

### 10.4 — Theme 4: "Did it learn real chemistry?" (Q12, Q13)

"The SHAP analysis is computed on our lab data, which is correct: the model predicts *lab* yields and
operates on lab chemistry, so we explain it in its operating regime. Reading the beeswarm: each dot
is one catalyst; its horizontal position is that feature's push on the prediction, in yield-percent
units; and the colour is the feature's value — red high, blue low. So a blue dot near the centre is
a low or absent feature value with little effect; a red tail to the right means a high value raises
yield — which is exactly what temperature does, correct physics. The literature prior ranks number
one, temperature two, and known OCM promoters — barium, manganese, lanthanum, cerium — follow, so
the model learned chemistry, not a shortcut. And I checked stability: across ten independent
subsamples, the literature prior is number one in all ten, and nine features are top-ten every time.
I didn't use LIME, deliberately — for a tree model, TreeSHAP is *exact*, so a LIME approximation
would be strictly noisier and add nothing."

### 10.5 — Theme 5: "Where does this sit in the literature?" (Q9, Q10)

"On novelty: the ingredients exist — stacked generalisation, delta and multi-fidelity learning where
a cheap model's prediction feeds a better one, and importance weighting for domain shift. But those
methods *trust* the source label; we deliberately don't — we pass the prior only as a feature, so
the bias is calibrated away rather than inherited. What's new is the specific combination — filter,
then quantile-normalise the prior, then use it as a feature — applied to OCM literature-to-lab
transfer under *both* covariate and label shift at once. On model drift: we don't have temporal
drift, but we have a static two-domain shift, which is the same mathematics — our 3.4-point label
gap is label shift, our 78.5% OOD is covariate shift — and the standard corrections for it,
importance weighting, instance selection, label rescaling, and recalibration, are exactly the four
tools this pipeline uses."

### 10.6 — Theme 6: "Data honesty and next steps" (Q17)

"On synthetic data: we don't generate any today — the one resampling ablation we tried was pure
duplication, and it didn't help, which is itself informative. If we did, the honest options are
SMOGN or SMOTER for regression, jittering real compositions, or tabular generative models like CTGAN
— and to hit a target proportion p you draw p-over-one-minus-p times the real count. But two rules
are non-negotiable: keep every synthetic row out of validation and test, so we validate only on real
held-out lab data; and enforce chemical validity, because a naive generator will happily invent
impossible catalysts. And the meta-point for the committee: the strongest evidence in this whole
project is the ten-seed plot, and the most important thing I did this round was to *find and correct*
an over-optimistic OOD number rather than defend it. Reporting the pipeline honestly is the result."

---

### One-line close

"To summarise the two chapters: on our own chemistry the method delivers a real, repeatable,
statistically significant 10% gain, validated on genuinely held-out data; the earlier out-of-
distribution claim was inflated by leakage and I've corrected it; and quantile normalisation is best
understood as a deliberate dial that trades extrapolation for the local accuracy we actually need."
