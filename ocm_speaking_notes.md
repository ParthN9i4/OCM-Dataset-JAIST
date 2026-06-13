# OCM Walkthrough — Personal Speaking Notes

Personal reference only. Not included in `ocm_walkthrough.ipynb`.

This transcript follows the notebook **cell by cell, in order**. For every code cell it
walks through the code **line by line**, so you can read this alongside the notebook and
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
  identical hyperparameters — only the *data* changes between methods.
- `evaluate_cv_ours` takes optional extra training data: `X_train_extra`, `y_train_extra`,
  and optional `sample_weight_extra`. For the baseline we pass nothing.
- `KFold(n_splits=5, shuffle=True, random_state=42)` — the fixed-seed 5-fold split, the
  same folds for every method.
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
- `sub_idx = rng.choice(..., size=10_000)` subsamples our data to 10,000 — enough to
  characterise our distribution without making the classifier slow.
- `X_dom = np.vstack([X_ours_sc[sub_idx], X_lit_sc])` stacks our subsample and all
  literature; `y_dom` labels them — ones for ours, zeros for literature.
- `LogisticRegression(C=0.5).fit(X_dom, y_dom)` trains the classifier. `C=0.5` keeps the
  boundary general rather than memorising individual points.
- `clf_dom.predict_proba(X_lit_sc)[:, 1]` — this is the key line. For every literature
  sample it returns the probability of class 1, 'ours'. Column index 1 is the 'ours' class.
  That vector, `p_ours_lit`, is the similarity score.
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
- We track the lowest RMSE in `best_drst` and store the winning τ.

**[Point at the output.]** τ=0.30 gives the lowest RMSE, 2.019 — a 5.3% improvement. One
honest caveat I'll mention in Chapter 9: we picked τ on the same CV we're reporting, so
that number is a touch optimistic."

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
  give near 0.
- Inside `kmm_weights`, `sigma` is set by the **median heuristic**: take the median of all
  pairwise distances in a subsample. No tuning — it adapts to the data's natural scale.
- `K_ss` is literature-to-literature similarity; `K_st` is literature-to-ours.
- `kappa = (n_s / n_t) * K_st.sum(axis=1)` — for each literature sample, how much it
  overlaps our data overall. High kappa → it should get weight.
- The `obj` and `grad` functions define the optimisation — minimise ½ wᵀK_ss·w − κ·w — and
  `minimize(..., method='L-BFGS-B', bounds=[(0, B)]...)` solves it with weights bounded
  between 0 and B. The bounds are what keep weights non-negative and capped.
- Back outside, we compute the correlation with the DRST scores and print how many weights
  are near zero.
- The two-panel figure: left is the weight histogram, right is the scatter of DRST score
  versus KMM weight.

**[Point at the left panel.]** Huge spike at zero — 78.5% of literature effectively
ignored. **[Point at the right panel.]** The diagonal trend is the r=0.79 agreement: two
different methods, same conclusion."

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

The result is 1.907 — a 10.6% improvement, double the filtering methods. And on
out-of-distribution chemistry, error drops 45%, because the expert gives a sensible prior
where our model would otherwise be guessing blind."

## Code cell — Prior Feature Transfer (`560d7b02`)

"Walk through it carefully — the order inside the fold matters:
- `quantile_normalize_y` ranks the source labels and maps each rank to the matching
  quantile of the target — that's the rank-matching that aligns scale while keeping order.
- `xgb_params()` returns the XGBoost settings for the small Stage-1 model.
- `mask = p_ours_lit >= best_drst[3]` reuses DRST's filter — Stage 1 trains only on the 782
  relevant samples.
- Now the CV loop. For each fold:
  - `y_lit_qn = quantile_normalize_y(y_lit[mask], y_ours[train_idx])` — normalise literature
    labels onto *this fold's* yield distribution.
  - `pre = xgb.XGBRegressor(...)` then `pre.fit(X_lit_sc[mask], y_lit_qn)` — train the
    expert on literature only.
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
- We retrain the whole pipeline once on *all* our data, outside the CV loop, so we have a
  single model to explain: quantile-normalise, train the Stage-1 `pre_shap` expert, generate
  `prior_all` for every one of our samples, and `np.hstack` it on to make the 68-feature
  matrix `X_aug`. `aug_feat_names` adds the name `lit_prior_prediction` for that 68th column.
- `final_shap = lgb.LGBMRegressor(...).fit(X_aug, y_ours)` is the model we'll interrogate.
- `idx_sh = rng_sh.choice(..., size=3000)` subsamples 3,000 points — SHAP is expensive, and
  3,000 is plenty to see the pattern.
- `explainer = shap.TreeExplainer(final_shap)` — for tree models this computes *exact* SHAP
  values, not an approximation.
- `shap_values = explainer.shap_values(X_sh)` produces one SHAP value per sample per feature.
- `shap.summary_plot(...)` draws the beeswarm.

**[Point at the beeswarm.]** Top row is `lit_prior_prediction` — the prior is the single
most influential feature, so the transfer learning is genuinely doing work. Temperature is
next, with high values (red) pushing yield up — correct OCM physics. Then Ba, Mn, La, Ce —
the known active phases and promoters — all positive. That's the model using real
chemistry, which is exactly what we wanted to confirm."

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
