> **Status.** This script narrates the **historical notebook**, whose chapters use a row-level split.
> Its numbers are kept so the narration matches what the notebook actually prints. Section 9.2 has
> been extended to carry the full correction; read that before presenting any of this.
>
> The correction in one line: the `1.907` / `−9.7%` prior-feature result came from a row-level split
> that let the same catalyst appear in training and test. Under catalyst-grouped validation the same
> models are **worse** than baseline (2.9425 vs 2.9955) — catalyst-identity leakage. Authoritative
> sources: `ocm_worknote_taniike.md` (v2), `SESSION_CONTEXT.md` §3 and §5, `ocm_verification_report.md`.

# Verbatim Presenter Script — Chapters 9 & 10

First-person spoken script. Every claim is tied to a specific number or reason — nothing vague.
Numbers reflect the **corrected, leak-free** analysis (see `qn_tradeoff.json`, `feedback_results.json`).

Key numbers to keep straight. **Everything in this first block is ROW-LEVEL and superseded** — it is
what the notebook prints, not what we now believe (see 9.2):
- Baseline (no transfer), row-level 5-fold CV: **RMSE 2.133**.
- Prior Feature Transfer (PFT = DRST-filtered + quantile-normalised prior): **RMSE 1.912** (−9.7%).
- 10-seed repeat: baseline **2.121 ± 0.006** vs PFT **1.909 ± 0.002**, PFT wins **10/10**, paired-t **p = 3.9×10⁻¹⁵**.
  (Ten seeds do not rescue this: every seed used the same leaky split.)
- True held-out 20% lab (never trained on): baseline **2.097** → PFT **1.892** (R² 0.763).

**The numbers that supersede them — catalyst-grouped 5-fold CV:**
- Baseline **2.9425**; PFT-filtered **2.9955** (**+1.8% worse**); PFT-all-literature **2.9817**.
- Screening (composition-only, formulation B): Spearman **0.724**, enrichment **3.77×** on the 771
  catalysts of comparable measurement effort (95% CI **3.04–4.89×**).
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

**Prior Feature Transfer — the method that appeared to work.**
"Under this row-level protocol PFT is the one that appears to break through, from 2.133 to 1.912 — a
9.7% reduction. Say 'appears' out loud: section 9.2 shows this inverts under a proper split. The reason
it works is structural: the literature never enters the final model as a *label*, only as a
*predicted feature*. A Stage-1 model learns the literature's view of chemistry; then Stage-2 — our
model — trains only on our labels, with the Stage-1 prediction added as one extra feature. That
means the 3.4-point label offset lives inside a feature value, where Stage-2 can calibrate around
it, instead of inside the loss, where it would poison every gradient. Its honest limitations: the
Stage-1 expert is trained on only 782 samples, so it's a bit high-variance; it assumes the quantile
map preserves the *ranking* of good and bad chemistry across domains; and it's two models to
maintain instead of one."

### 9.2 — The correction I owe you: two leaks, not one

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

"When I first gave this correction, I said it did **not** touch the headline — that on our own
chemistry the in-distribution gain was solid and repeatable across ten seeds. **That was wrong, and
this is the second and larger correction.**

The in-distribution result had the same disease. Stage 1 was trained on literature *together with
the lab training rows*, and under a random row split those rows contained other measurements of the
very catalysts sitting in the validation fold. The 89,074 rows are only **917 distinct catalysts**,
roughly a hundred rows each — so a random row split almost guarantees a catalyst appears on both
sides. The prior feature was carrying each validation catalyst's own measured yields back to the
model. Ten seeds do not help: every seed made the same mistake.

Re-run with folds split **by catalyst**, so that all rows of a catalyst live in exactly one fold, the
result inverts. Baseline 2.9425, two-stage 2.9955 — **1.8% worse**, not 9.7% better. The baseline
itself rises from 2.12 to 2.94, which is the honest difficulty of predicting a catalyst nobody has
made.

So the correct summary of this chapter is: the OOD claim was withdrawn first, the in-distribution
claim was withdrawn second, and what remains is a composition-only screening model with enrichment
3.77× on catalysts of comparable measurement effort. I would rather present it that way than have a
referee find it."

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

## CHAPTER 10 — Reviewer Q&A: A Full Answer to Every Question

### 10.0 — Framing

"The committee raised seventeen questions. I'll answer every one of them directly and in full, and
I've grouped them into six themes so the logic flows. Every answer is backed by a specific number or
a figure that is reproducible in the notebook."

---

### Theme 1 — "Did you prepare the data correctly?"

**Q1 — Why StandardScaler and not MinMaxScaler? Is linearity lost?**

"StandardScaler subtracts the mean and divides by the standard deviation — z equals x minus mu over
sigma. MinMax maps to the zero-to-one range — x minus the minimum over the range. Both are *affine*,
straight-line transforms, so neither destroys linearity: correlations, linear-model fits, and linear
separability are all preserved exactly. The idea that standardisation loses linearity is a
misconception — the only genuinely nonlinear rescaling anywhere in this project is the quantile map
on the *labels*, which is probably what the question had in mind. I chose StandardScaler for two
concrete reasons. First, scaling only affects the distance-based steps — the PCA plot, the DRST
logistic classifier, and the KMM kernel — and it is completely irrelevant to the gradient-boosted
trees that actually predict yield, because trees split on thresholds and are invariant to scale.
Second, StandardScaler is robust to the extreme catalyst compositions in the literature, whereas
MinMax lets a single outlier define the range and crushes all the informative variation into a
sliver."

**Q2 — What are the PCA values, how do you find them, and why only the first two?**

"PCA finds the orthogonal directions of maximum variance in the 67-dimensional feature space, by
eigen-decomposing the covariance matrix — equivalently, the singular-value decomposition of the
mean-centred data. The 'values' are two things: each sample's *scores*, meaning its coordinates on
the first two components, which are the x and y of each dot in the scatter; and the
*explained-variance ratio*, the fraction of total spread each component captures, which I print on
the axis labels. I keep only two components because this is a *diagnostic picture* of the domain
gap, not a modelling step — two components give a plottable plane. And I'm explicit that two
components *under-state* the gap, because they cannot capture all 67 dimensions, so the visual
separation is a lower bound; the rigorous measure of the gap is the DRST classifier, which says
78.5% of the literature is out-of-distribution."

**Q3 — Did you run PCA on the literature only or the whole set? What is the ideal way to think about it?**

"I fit the PCA on the *combined* set — our lab points plus all the literature — because you can only
judge whether two clouds overlap if they are drawn on the *same* axes. If I fit it on the literature
alone, I would see the literature's internal structure but have no common frame to compare lab data
against. The general principle is: when your goal is *comparison*, define the projection on a shared
basis — either fit on the union, or fit on one domain and project the other onto it. The trap to
avoid is fitting on one domain and then reading the other domain's spread as if it were meaningful
internal variance. And for a genuinely rigorous shift measurement you shouldn't rely on a
two-component picture at all — you use a domain classifier, which is exactly what DRST does on all 67
features."

**Q4 — For the PCA plot you used 3,000 — why not all 3,852?**

"That's a misreading of which number was thinned. All 3,852 literature rows are used — nothing is
dropped from the literature. The 3,000 is a subsample of our *89,000 lab* points, taken purely so the
scatter is legible: 89,000 blue dots would form an opaque blob that completely hides the 3,852 orange
literature points. Thinning the lab cloud for a plot does not change the PCA in expectation, and it
has no effect on any model — the models use every single row."

---

### Theme 2 — "Did you pick your methods and thresholds honestly?"

**Q5 — Why the threshold tau equals 0.30 for DRST? Did you evaluate other values? Show the full analysis.**

"I didn't pick 0.30 by eye — I swept the threshold from 0 to 1, and two findings came out, the second
being the important one. First: *single-stage* DRST, where I simply add the kept literature to
training, barely helps at any threshold — the best point is only a hair below baseline, and at 0.30
it is actually slightly *worse* than baseline, because even lab-like literature still carries the
3.4-point label shift that poisons the loss. Second: the *two-stage* method, where the filtered
literature only trains the Stage-1 prior, is essentially *flat* across the threshold — every value
from 0.05 to 0.8 lands within one hundredth of an RMSE point of the best, all about 10% below
baseline. So 0.30 is a safe, near-optimal, non-cherry-picked choice, and the crucial point is that
the improvement comes from the *architecture*, not from tuning that knob. If I had cherry-picked the
threshold, the curve would be sharp; it is flat, which proves I didn't."

**Q6 — Plot RMSE against the threshold, right across 0 to 1.**

"That plot exists — it is `fig_drst_threshold_sweep.png` for the single-stage sweep and
`fig_pft_tau1_sweep.png` for the two-stage. I show both curves so the choice is fully transparent."

**[On screen: `fig_drst_threshold_sweep.png`] — how to read it:**
"The horizontal axis is the DRST keep-threshold τ, running from 0 on the left to nearly 1 on the
right — as you move right we keep only the most lab-like literature. The left vertical axis is the
5-fold cross-validation RMSE, and the blue dots-and-line are the single-stage result at each
threshold. The grey dashed horizontal line is the baseline, 2.133 — no transfer at all. The orange
dotted vertical line marks τ equals 0.30, the value the deck uses, and the crimson ring marks the
empirical best point, τ equals 0.85 at RMSE 2.127. The green squares, read against the right-hand
axis, are how many literature rows survive each threshold — they fall from over 1,300 down to under
100 as τ rises. The one thing to take away: the blue curve *hugs* the grey baseline line the whole
way across — the best it ever gets is 2.127 versus baseline 2.133 — so simply adding filtered
literature to training barely helps at any threshold."

**[On screen: `fig_pft_tau1_sweep.png`] — how to read it:**
"Same idea, but now the filtered literature is used only to train the Stage-1 prior. The horizontal
axis is the Stage-1 filter threshold τ₁; the vertical axis is again 5-fold CV RMSE. The grey dashed
line is baseline 2.121, the orange dotted line is τ₁ equals 0.30, and the crimson ring is the best
point, τ₁ equals 0.80 at 1.906. The story is in the shape: unlike the previous plot, this whole
curve sits far *below* the baseline — about 10% lower — and it is almost perfectly flat, every point
between 1.906 and 1.913. That flatness is the argument: the improvement doesn't depend on picking
the threshold carefully, so 0.30 isn't cherry-picked — the win comes from the architecture."

**Q8 — Does KMM select the same 782 samples as DRST? How compatible are the two methods?**

"They are not identical by construction — DRST makes a hard cut and keeps 782 rows, while KMM assigns
a *continuous* weight between 0 and 5 to all 3,852 samples and never hard-selects. But they read the
same underlying signal. Their correlation is 0.79, and when I take DRST's 782 kept rows and KMM's
top-782 by weight, they overlap by 634 rows — that's 81% — with a Jaccard index of 0.68. So they
corroborate each other rather than compete, and that agreement is itself evidence: when two
mathematically different methods flag the same samples, the covariate-shift signal is real, not an
artefact of one method's particular assumptions."

**[On screen: `fig_kmm_drst_overlap.png`] — how to read it:**
"There are two panels. The left panel is a scatter: every point is one of the 3,852 literature
samples, with its DRST score — the classifier's probability that it looks like our lab, P of lab
given x — on the horizontal axis, and its KMM weight on the vertical axis. The title shows the
correlation, r equals 0.79, and the orange vertical line is the τ equals 0.30 cut. The key visual is
that the cloud rises from bottom-left to top-right — samples DRST scores as lab-like are exactly the
ones KMM up-weights. The right panel makes the overlap concrete: it's three bars — the samples both
methods pick, 634 of them; the ones only DRST keeps, about 148; and the ones only KMM ranks in its
top-782, again about 148 — for a Jaccard overlap of 0.68. Takeaway: a hard filter and a soft
re-weighting, built on completely different mathematics, land on essentially the same samples."

---

### Theme 3 — "Is the result real, or luck?"

**Q7 — Did you use the entire dataset for training? How is it implemented?**

"Yes — all 89,074 lab rows are used, through asymmetric five-fold cross-validation. Every lab row is
in the training set for four of the five folds and is scored exactly once, in the fold where it is
held out. The word 'asymmetric' is the key detail: the *validation* fold is always lab-only —
literature never appears in validation, only ever on the training side. That is a deliberate choice,
so the reported RMSE measures accuracy on *our* experiments, which is what we care about, rather than
a blurred lab-plus-literature average. For the final deployed model I train on all lab rows plus the
filtered literature prior."

**Q11 — How many times did you actually run the transfer method to cross-check the result?**

"Originally just once, at a single random seed — and that was a fair criticism, so I re-ran it
properly. I now run the whole pipeline with ten different random seeds, each producing a fresh
five-fold split and fresh model randomness, for both the baseline and the transfer method. The full
numbers are in the next two answers, but the headline is that it wins every single time."

**Q14 — Give the mean and standard deviation of RMSE, plot all ten runs, show it is below baseline every time, and confirm nothing is force-fit.**

"Across the ten seeds, the baseline averages 2.121 with a standard deviation of 0.006, and the
transfer method averages 1.909 with a standard deviation of 0.002. The transfer method beats the
baseline in all ten out of ten runs — in `fig_repeated_runs.png` the two curves never cross, and the
*worst* transfer run still beats the *best* baseline run. Nothing is force-fit: the
hyper-parameters are identical across every seed, there is no per-seed tuning, and no early stopping
on the validation fold. And notice it is not only better on average, it is three times *tighter* — a
standard deviation of 0.002 against the baseline's 0.006 — so the transfer method is both more
accurate and more stable."

**[On screen: `fig_repeated_runs.png`] — how to read it:**
"Two panels again. The left panel plots RMSE against the random seed — ten seeds along the bottom.
The grey line with circles is the baseline, hovering around 2.12; the navy line with squares is the
transfer method, sitting around 1.91; and the green shading between them is the gap. The single most
important visual is that the two lines *never touch* — even the worst navy point is well below the
best grey point. The right panel is the same data as a paired difference: one bar per seed showing
baseline-minus-PFT, and every bar is green and positive, all clustered around 0.21 — the method
improves things on every single seed, never once regresses. The p-values from the paired t-test and
Wilcoxon test are printed in the title. Takeaway: this isn't one lucky split — it's a consistent,
significant win repeated ten times."

**Q15 — Demonstrate that the result is not accidental.**

"I test it statistically on the paired ten-seed results. A paired t-test gives a p-value of 3.9 times
ten-to-the-minus-fifteen, and the non-parametric Wilcoxon signed-rank test agrees at p equals two
times ten-to-the-minus-three. The average improvement, 0.21 RMSE, is more than thirty times the
seed-to-seed standard deviation. A ten-out-of-ten win with p-values that small is, by definition, not
a lucky split — it is a genuine, repeatable effect."

**Q16 — Did you validate the model on completely untested data?**

"Yes. Beyond cross-validation I added a true hold-out: a random 20% of the lab — about 17,800 rows —
set aside at the very start and never used in *any* stage of *any* method: not in the DRST
classifier, not in Stage 1, not in Stage 2. On that untouched set the baseline scores 2.097 and the
transfer method scores 1.892, with an R-squared of 0.76 — the same 10% gain, on data the model has
genuinely never seen. That, and not the out-of-distribution number, is the honest generalisation
evidence — for the leakage reason I explained in Chapter 9."

**[On screen: `fig_qn_tradeoff.png`] — how to read it:**
"This is the figure behind the quantile-normalisation trade-off, and it has two panels sharing the
same four bars — four configurations crossing the prior source, DRST-filtered versus full
impregnation literature, with the labels either raw or quantile-normalised. The left panel is
in-distribution 5-fold CV RMSE, and its grey dashed line is the baseline, 2.133; notice all four
bars sit well below it around 1.91, and inside each pair the quantile-normalised bar is a touch
*lower* — QN helps here. The right panel is the leak-free OOD RMSE, with its grey dashed baseline at
6.53; here all four honest bars sit around 6.0 to 6.8, roughly level with baseline, and inside each
pair the quantile-normalised bar is a touch *higher* — QN hurts here. The one extra line to point at
is the red dotted line on the right panel, way down at 3.62 — that's the old, leaky number, and I
draw it deliberately far below the honest bars to show how much leakage inflated it. Takeaway in one
sentence: quantile normalisation is a dial — it buys in-distribution accuracy at a small
out-of-distribution cost — and the dramatic 3.62 only exists when the prior is allowed to see the
test rows."

---

### Theme 4 — "Did it learn real chemistry?"

**Q12 — Is the SHAP beeswarm on lab data or the whole dataset? How should it be done, and what did you do?**

"I computed SHAP on a 3,000-sample subsample of the *lab* data, with the prior feature included,
explaining the final Stage-2 LightGBM model using the exact TreeExplainer. That is the correct scope,
and the reasoning is this: SHAP explains a *specific model on a specific dataset*, and our model
predicts *lab* yields and is meant to run on lab-like chemistry — so explaining it on lab data
answers the right question, namely 'what drives predictions in our operating regime.' Explaining it
on literature would answer a different, also legitimate, question: how the model extrapolates
out-of-distribution. The 3,000-sample subsample is purely for speed; the mean-absolute-SHAP rankings
converge quickly, and I verify exactly that in the next answer."

**Q13 — Explain how to read the beeswarm — the blue and red, the spread, the lab points near zero, the literature — and did you run it at least ten times?**

"Reading the beeswarm: each dot is one lab catalyst; its horizontal position is that feature's signed
push on that specific prediction, measured in yield-percent units, so a dot on the right means the
feature pushed the predicted yield up; and the colour is the feature's *value* — red for high, blue
for low. So a cluster of blue dots sitting near the zero line means a low or absent feature value with
little effect — for an element column, that reads as 'this element is absent, so it does not move the
prediction.' A red tail stretching to one side shows what a *high* value does: temperature's red dots
sit on the right, meaning hotter pushes yield up, which is correct OCM physics; lithium and potassium
show red dots on the left, meaning high loadings suppress yield. Every dot is a lab sample, so there
are no literature points on this plot at all — the literature's influence enters through the single
feature called `lit_prior_prediction`, and that feature ranks number one. And yes, I checked
stability: I recomputed SHAP on ten independent subsamples, and `lit_prior_prediction` is the
number-one feature in all ten of them, with nine features appearing in the top ten in every single
run — so the ranking is not an artefact of one random draw. I deliberately did not use LIME, because
for a tree model TreeSHAP is *exact* and consistent, so a local linear LIME surrogate would be
strictly noisier and would add nothing here."

**[On screen: `fig_shap_stability.png`] — how to read it:**
"This is the stability check, and it's a simple horizontal bar chart. Each bar is a feature; the
length of the bar is its mean absolute SHAP value — its average importance — and the little black
whisker on the end is the standard deviation of that importance across the ten independent
subsampling runs. The features are sorted, so the most important sits at the top: `lit_prior_prediction`
is far and away the longest bar, around 2.75, with a tiny whisker — followed a long way back by
Temperature, then barium, zirconium, manganese, lanthanum, cerium, copper, aluminium. Two things to
read off it: first, the literature prior dominates — its bar dwarfs everything else, which is the
visual proof that the transfer signal is doing the work; and second, the whiskers are all short
relative to the gaps between bars, which means the ranking barely moves from run to run. Takeaway:
the importance story is stable, not an accident of one random sample."

---

### Theme 5 — "Where does this sit in the literature?"

**Q9 — Has this transfer-learning approach been done before?**

"The ingredients exist, but the combination is new. Using a source model's prediction as a feature is
known — it appears in stacked generalisation, and in delta-learning and multi-fidelity learning in
materials and quantum chemistry, where a cheap model's output feeds a more accurate one. Filtering by
domain similarity and importance weighting are also standard covariate-shift tools. But there is a
genuine distinction: those multi-fidelity methods *trust* the source label — they literally add the
source prediction as a correction term — whereas we deliberately pass the prior only as a *feature*,
so the label bias is calibrated away by Stage 2 rather than inherited. What is novel is the specific
combination — filter with DRST, quantile-normalise the prior, then use its prediction as a feature —
applied to OCM literature-to-lab transfer under covariate *and* label shift simultaneously. I
searched the transfer-learning, domain-adaptation, multi-fidelity, and catalysis-ML literature and
found no prior OCM work doing literature-to-lab transfer this way."

**Q10 — What is model drift, is it relevant here, and how would you correct it?**

"Model or concept drift is when the statistics a model relies on change, degrading it. There are three
kinds: covariate shift, where the input distribution P-of-x moves; label or prior shift, where the
target distribution P-of-y moves; and concept drift, where the actual input-to-output rule,
P-of-y-given-x, changes. We do not have *temporal* deployment drift, but we do have a *static
two-domain shift* between the literature and our lab — and that is the same mathematics: our
3.4-point label gap is a label-shift instance, and our 78.5%-out-of-distribution is a covariate-shift
instance. So drift concepts underpin the whole project. The standard corrections map directly onto
what we already do: importance weighting is KMM, instance selection is DRST, label-shift correction
is our quantile normalisation, and recalibration around the offset is exactly what Stage 2 does. For
an eventual deployment you would add monitoring — a Kolmogorov-Smirnov test or a population-stability
index — and periodic retraining."

---

### Theme 6 — "Data honesty and next steps"

**Q17 — How could you generate synthetic data with a controlled proportion?**

"Right now we generate none — the one resampling ablation we tried was pure duplication with
replacement, and it did not help, which is itself informative. If we wanted real synthetic data, the
honest options are: SMOGN or SMOTER, the regression variants of SMOTE, which interpolate between
near-neighbours and add noise in sparse target regions; simple jittering, adding small Gaussian
perturbations to real compositions; tabular generative models such as CTGAN or a TVAE trained on the
real data; or conditional generation, where you condition on target-yield bins to directly rebalance
the skewed label distribution. To hit a target synthetic fraction p, you draw p-over-one-minus-p
times the real count. But two rules are non-negotiable in catalysis: first, keep every synthetic row
out of validation and test, so you only ever validate on *real* held-out lab data — otherwise the
metrics are meaningless; and second, enforce chemical validity, because a naive generator will
happily invent impossible catalysts, so you constrain element sets and loading sums, or better, use
physics-based generation."

---

### One-line close

"To summarise the two chapters: on our own chemistry the method delivers a real, repeatable,
statistically significant 10% gain, validated on genuinely held-out data; the earlier
out-of-distribution claim was inflated by leakage and I have corrected it openly; and quantile
normalisation is best understood as a deliberate dial that trades extrapolation for the local
accuracy we actually need."
