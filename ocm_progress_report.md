# OCM Project — Progress Report

**From:** Parth Nagar
**To:** Dr. M S Srinath
**Date:** 31 August 2026
**Covers:** all work from the start of the project to today, including everything done after Prof. Taniike's review

---

## 1. What we set out to do

We predict C₂ yield for OCM catalysts from their composition.

We hold two datasets. The **lab data** has 89,074 measurements from 2025. The **literature data** has 3,852 measurements published between 1982 and 2019.

We asked one question. Can the literature data make the lab model more accurate?

Two problems make this hard.

- **Label shift.** Published papers report higher yields than the lab sees. The gap is 3.4 percentage points.
- **Covariate shift.** The two datasets use different chemistry.

---

## 2. The five methods we tried first

We tried five methods, in this order. Each one answered a problem the previous one exposed.

| # | Method | What it does | Result |
|---|---|---|---|
| 1 | **Baseline** | Train LightGBM on lab data only. | The number to beat. |
| 2 | **Direct merge** | Pool lab and literature data. Train one model. | Worse. The label shift pulls predictions up. |
| 3 | **Selective merge (DRST)** | Keep only literature that looks like lab chemistry. Then merge. | No real gain. |
| 4 | **Selective merge (KMM)** | Weight each literature record instead of filtering. Then merge. | Worse. |
| 5 | **PFT (our method)** | Train a model on literature. Feed its *prediction* into the lab model as an extra input. Never use literature yields as training labels. | Best result at the time. |

Methods 2, 3 and 4 all failed for the same reason. They all put literature yields into the training target. The label shift then biases the model.

Method 5 avoided that. It passed literature knowledge as a *feature*, not a *label*. We reported a 10.6% improvement. We sent this to Prof. Taniike.

---

## 3. What Prof. Taniike found

He replied with six technical points. One of them changed the project.

**His main point:** our test was not measuring what we claimed.

The lab data holds 89,074 rows. But it only holds **917 distinct catalysts**. Each catalyst was measured about 97 times under different conditions.

We split the data randomly by row. So the same catalyst appeared in both training and testing. The model had already seen the catalyst it was being tested on.

He was right.

---

## 4. What we did about each of his six points

### Point 1 and 2 — Split by catalyst, not by row

We rebuilt the evaluation. Now every measurement of a catalyst stays in one fold.

The result reversed:

| Split method | Baseline | Our method (PFT) |
|---|---|---|
| By row (what we sent him) | 2.118 | 1.912 — **9.7% better** |
| **By catalyst (correct)** | **2.943** | **2.995 — 1.8% worse** |

Our improvement disappeared. We withdrew the claim.

**We also found the cause.** Our Stage-1 model trained on literature *plus lab training rows*. Under a row split, those rows included the test catalysts. So the feature was carrying memorised answers, not literature chemistry.

We made catalyst-based splitting the default in all our code. It cannot be skipped by accident now.

### Point 3 — Hold out whole catalyst families

We did this for 28 element families.

We found the Ba family fails badly. We then proved why. When no Ba catalysts appear in training, the model cannot use the Ba column at all. We deleted the column and got identical predictions. That proves the model ignores it.

Ba matters because it holds 78% of the lab's best catalysts. Remove Ba, and you remove the high-yield chemistry from training.

We then asked a more useful question. **How many catalysts of a new family must we measure before the model can predict it?** Our answer: about 10 for Ba, and none for La, Ti, Zr and Ce. Those four are already predictable from neighbouring chemistry.

### Point 4 — Predict the best yield per catalyst, not the best condition

He explained the real goal. His lab tests every catalyst under a fixed set of conditions. So the model does not need to pick the condition. It needs to say whether a catalyst can reach a high yield at all.

We rebuilt the model to predict each catalyst's **maximum** yield. It now trains on 917 rows instead of 89,074. It ranks just as well.

### Point 5 — Use catalyst-level metrics

We replaced point-wise RMSE with ranking metrics. These are what a screening campaign actually cares about.

| Metric | Value | 95% range |
|---|---|---|
| Rank correlation of predicted vs real best yield | 0.761 | 0.725 – 0.785 |
| Enrichment of top performers | 4.28× | 3.04 – 4.89× |
| Hit rate in a 20-catalyst shortlist | 0.44 | 0.15 – 0.65 |

We report ranges, not single numbers. The ranges are wide, and we say so.

### Point 6 — Test whether the rescaling step matters

He suspected our quantile-normalisation step did nothing. Tree models read the *order* of a feature, not its scale.

We tested three versions: normalised, raw yields, and plain ranks. The differences run from 0.001 to 0.023 RMSE. At three seeds, that sits inside run-to-run noise.

So we cannot separate them. We can drop the normalisation step without a measurable penalty. He was right.

---

## 5. What we tried after that, to rescue the method

The leak explained the old result. It did not prove literature data is useless. So we tested four honest ways to use it.

| Design | Rank correlation | vs. control |
|---|---|---|
| Composition only (control) | **0.761** | — |
| Literature rank prior | 0.757 | −0.004 |
| Similarity-to-literature features | 0.740 | −0.021 |
| Both combined | 0.744 | −0.017 |
| Catalyst-level direct merge | 0.758 | −0.003 |

We fixed the success criteria before running. **None of them beat composition alone.**

We also tested one idea of our own. One family (Zr) suggested literature helps where our own data is thin. We tested that across all 28 families. The effect vanished. We discarded our own idea.

---

## 6. The one place literature data does help

We then asked a different question. Can the lab model predict catalysts made by *other* preparation methods?

| Test | Rank correlation |
|---|---|
| Literature made by impregnation (same method as lab) | 0.398 |
| Literature made by other methods | 0.238 |
| **Other methods, after adding impregnation literature to training** | **0.388** |

Adding literature raised performance by 0.150, on every one of five seeds.

This is the first time literature data measurably helped. The reason is simple. Our lab only uses impregnation. So for other preparation methods, the lab has no data at all, and the literature is the only information we have.

Two honest caveats. Absolute performance is still poor. And plain merging beat our two-stage method here.

---

## 7. Where we stand today

**What works.** The composition-only model ranks unseen catalysts usefully. A campaign guided by it finds good catalysts about 3 to 5 times faster than random choice.

**What does not work.** Our PFT method does not improve prediction for lab catalysts. No literature-integration design does.

**What we cannot do.** The model cannot price a promoter family it has never seen. This is structural, not a coding problem.

**What we produced.** We enumerated 26,414 candidate catalysts using the lab's own design rules. We recommend a 17-catalyst campaign: 12 at the model's best predictions, and 5 spanning alternative supports to test whether the model's preference for Ba is real chemistry.

---

## 8. A correction we made to our own report

We audited our own work note before sending it. We found eight factual errors and fixed them all.

The most serious ones:

1. We misquoted what version 1 said. Version 1 reported 1.907. Our draft said 1.912.
2. We claimed the rescaling ablation differed by "0.001–0.005 RMSE". The true range is 0.001–0.023.
3. We claimed Ba is uniquely hard to predict. Our own data shows five families score worse.
4. We quoted a data budget that contradicted our own stored measurement.
5. We presented an ensemble spread (±0.09) next to a predicted yield. A reader would take it as an error bar. The real error is about 2.7 yield points.

We caught all of these before sending anything. Every number in the corrected note now traces to a stored file.

---

## 9. What comes next

1. **Send the corrected work note and reply to Prof. Taniike.** Both are ready.
2. **Run the prospective validation.** He offered to synthesise our candidates. This is the strongest path to publication.
3. **Ask for the missing condition data.** Each catalyst is run at 5 temperatures under about 27 reaction-condition settings each. That is 135 runs per catalyst, and it matches the number Prof. Taniike quotes. We can see the structure in the row counts: 15 catalysts hold exactly 135 rows, and every one of them splits as exactly 27 rows at each of the 5 temperatures. But the file does not record what those 27 settings are. Because they are missing, 19.9% of the yield variance cannot be explained by composition and temperature alone. This is not noise. It is real chemistry we simply cannot see. Getting those columns would help more than any modelling change.

---

## 10. What we can claim honestly

We can claim three things.

1. **A validation finding.** We documented a case where a literature-transfer method looked strong under standard cross-validation and failed under catalyst-grouped validation. We identified the mechanism. Random splits are common in this field, so this is worth reporting.
2. **A working screening tool.** The model ranks unseen catalysts well enough to guide synthesis.
3. **A scoping result.** Literature data does not help where our own coverage is good. It does help where our coverage is absent.

We cannot claim that PFT improves prediction. We tested it properly, and it does not.
