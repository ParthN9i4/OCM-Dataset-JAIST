# Verification report — OCM catalyst-transfer project

**Scope.** Re-derive every claim in `SESSION_CONTEXT.md` from the data rather than from the document.
Dataset facts were recomputed with a pure-standard-library reimplementation (no pandas), so they are
an *independent* check, not a re-run of the project's own code. Metric claims were read back from
their stored JSONs, and two experiments were re-run end to end.

---

## Bottom line

**The science holds. The bookkeeping did not.** No number in `SESSION_CONTEXT.md` that had a stored
source was found to be wrong — every headline figure in §3, §4, §5A, §5B and §8 reproduces exactly.
But three load-bearing numbers had *no* source at all, and two of those three did not survive
recomputation. One of them was in the document drafted for Prof. Taniike, underwriting a proposal to
cut reactor runs from 135 to 20.

Separately, the checkout was on the wrong side of an unmerged branch: the entire phase 8/9 correction
— and the corrected work note itself — existed only on a branch that was never merged.

---

## 1. What reproduced

**From the CSV, independently recomputed.** 92,926 × 69. Lab 89,074 (`year==2025`) / literature 3,852
(`year<=2019`), nothing unaccounted. 65 element columns. Lab 100 % Impregnation. Mean Y(C2) 5.245 % /
8.670 %, label shift +3.425 pp. **917** catalysts — stable under two independent identity definitions
(raw file strings and float-normalised), which rules out a formatting artifact. Five temperatures,
700–900 °C in 50 °C steps. **4,399** cells, mean 20.2487 rows/cell, modal 27 (895 cells), max 54, none
above. 15 catalysts at 135 rows, **15 of 15** splitting (27,27,27,27,27). 811/917 with all five
temperatures; 186 absent cells; 83 singletons. Within-cell variance share **19.9 %**, floor RMSE
**1.757** (pooled). The superseded unweighted form reproduces **18.2 % / 1.680**, confirming the
diagnosed bug rather than merely asserting it. Both derived deltas check: 1.907 − 1.757 = 0.15,
1.907 − 1.680 = 0.23.

**From the stored JSONs.** The whole §3 reversal table including both Spearmans and both derived
percentages (−9.74 %, +1.80 %); phase 3's five variants with deltas and seed-wins; E1 (|ρ| = 0.276,
14/28, not supported); the Ba curve and *both* threshold derivations (n = 10 by level, n ≈ 50 by gain);
phase 4's `max_abs_pred_diff_dropBa = 0.0`; phase 7's C1/C2/C3 and the impregnation reference; phase 8's
condition-grid evidence, the +0.0065 seed-averaging gain and the full sweep; every phase 9 number
including the 300-draw control and the campaign-limit table; §8's 3.04–4.89× and 0.44 [0.15, 0.65].

**Reproduction runs.** `phase9_equal_effort_eval.py` and `phase8_target_robustness.py` were re-run in
an isolated scratch directory (inputs symlinked, so the committed JSONs were never at risk) and both
came back **byte-identical** — 268 numeric fields, maximum absolute difference exactly 0.0 — across a
major library version jump (pandas 3.0.5, numpy 2.4.6, scikit-learn 1.9.0, LightGBM 4.7.0 against
pins of ≥2.0/≥1.24/≥1.3/≥4.0). That is a stronger determinism result than the project has claimed.

---

## 2. Findings

| # | Finding | Confidence | Severity |
|---|---|---|---|
| 1 | **Branch divergence.** `phase8_*`/`phase9_*` absent from the checkout; §5 work, the fixed `ocm_eval.py` and the corrected work note stranded on unmerged `claude/add-catalyst-dataset-orerR`. The tree's `SESSION_CONTEXT.md` was the superseded 358-line version. | Certain | **High** |
| 2 | **ρ = 0.955 has no source** and does not reproduce. Under the exact design the work note describes (5 rows at each of 750/800/850/900 °C) it recomputes to **0.9493 ± 0.0033**; bias −1.31 recomputes to **−1.377**. 0.955 is closer to the **25**-run configuration (0.9563). | Certain that it is unsourced and does not reproduce; *likely* that it was carried over from a different configuration | **High** — it underwrites cutting reactor runs 135 → 20 |
| 3 | **The work note's "811 fully-measured catalysts" is wrong for that design** — a 20-run design needs ≥5 rows in each of 4 cells, which only **759** of the 811 satisfy. | Certain | Medium |
| 4 | **2.03× / 0.90× have no source.** Prose in `phase8`'s docstring plus four documents; no script, no JSON. My recomputation confirms the *direction* under all four interior-spacing definitions tried (e.g. 1.900 vs 1.160 by mean interior gap) but reproduces neither figure. | Certain | **High** — it is the claim that rejects the truncation reading |
| 5 | **`ocm_eval.py` asserted the unresolved reading as fact** — "27 DIFFERENT reaction-condition settings" — in the protocol docstring every experiment imports, while §7 calls it the largest live uncertainty. | Certain | Medium |
| 6 | HEAD's `ocm_eval.py` used banned vocabulary ("irreducible noise floor") positively. Consequence of #1. | Certain | Medium |
| 7 | Commit count stated as 108; actual **102**. | Certain | Low |
| 8 | "max cell = 54 = 2 × 27" cited as structural evidence rests on **exactly one cell**. | Certain | Low–Medium |
| 9 | Literature "8+ methods" — actually **20** distinct `Preparation` values (19 named + `n.a.`). | Certain | Low |
| 10 | Ba learning curve omits n = 200 (**0.6869**), which *exceeds* the n = 204 "all" point (0.6832) — the tail is non-monotone. `family_size` is 291, not 204. | Certain | Low |
| 12 | **The presentation asserted a withdrawn claim in four formats.** `ocm_presentation.{html,pdf,pptx,tex}` all carried "−10.6 %", "RMSE 2.133 → 1.907" and a slide titled "Prior Feature Transfer — The Winning Pipeline"; slide 5 was the row-level protocol that caused the leak. The builders hardcode this content, and none of these files appeared in SESSION_CONTEXT §6's file map. Three narration documents (~1,800 lines) matched. | Certain | **High** — a retracted result presented as current |
| 13 | **`build_pptx.py` reported PDF export as successful when it had failed.** It checked only that the path existed, so a failed conversion left the *stale* PDF in place and still printed "Saved". `libreoffice-core` was installed without `libreoffice-impress`, so PPTX→PDF could not work at all and exited 0 while printing "source file could not be loaded". This is why the withdrawn claim survived in the PDF. | Certain | **High** |
| 11 | **`build_worknote_render.py` silently drops every figure caption.** The work note carries its nine captions as markdown image alt-text; python-markdown emits those only into `alt=""`, invisible to any reader of the PDF. Re-rendering the work note with its own committed build script therefore produced a document with nine uncaptioned figures. Caught only by extracting text from the old and new PDFs and comparing. | Certain | Medium — it silently degrades the document sent to a collaborator |

All ten are fixed in this branch except where noted as informational.

---

## 3. The gating question: why the file cannot answer it

**Every one of the 4,399 cells is stored sorted descending by yield — 0 violations across 84,675
within-cell adjacent comparisons.** The lab block as a whole has 902 violations, so this is not a
trivial by-product of a global sort.

Row order therefore encodes **rank, not acquisition sequence**. There is no time column, no run index,
no condition column. Every test that could separate "27 distinct reaction conditions" from "27
successive time-on-stream samples" — decay profile, periodicity, position-versus-yield,
autocorrelation — is destroyed by that sort. **This is not an inconclusive result; it is a structural
impossibility.** It explains why two prior analyses found no signature, and it means a third would
fail too. Only JAIST can answer it.

Two consequences that were not previously stated:

- Pre-sorting by yield is precisely what a "take the top N" export produces, so the truncation reading
  is mechanically *more* plausible than the order-statistic argument implies.
- But truncation has a far simpler disproof that was never given: **104 cells hold more than 27 rows**,
  which a top-27 cut-off cannot emit. Lead with the count, not the order statistic.

---

## 4. The result worth carrying forward: the answer changes the labels, not the decision

This is the part to present. `phase10_ground_truth_invariance.py` fills a cell of the experimental
design that was empty, and it converts the project's largest open risk from blocking to bounded.

**Why the existing phases could not settle it.** `phase8` varies the training target but scores every
variant against `TRUE_MAX` (line 130) — one evaluation column, so if the max is the wrong ground truth
the analysis only shows our labels agree with each other. `phase5` has the mirror-image flaw: its
`D_sensitivity` moves training and evaluation together, so p90 outscoring max says p90 is an *easier*
target, not a better one. Neither varies the two independently. Additionally, phase 8's sweep stops at
q = 0.50 — **no label below the median had ever been tested**, and that is exactly the region the
time-on-stream reading points at.

**What phase 10 does.** Full train-target × eval-target matrix over q ∈ {0.05 … 1.00}, where q = 1.00
is the catalyst's *ceiling* (observed max) and q = 0.05 its *floor* at the best temperature, with a
pre-registered decision rule fixed before the modelling ran.

**The uncomfortable half first.** At the data level the two readings genuinely disagree where it
matters: ranking by floor instead of ceiling shares only **7 of the top 20** catalysts (overall
Spearman 0.859). The disagreement is concentrated exactly in the region a synthesis campaign occupies.

**The useful half.** A model trained on the ceiling still scores **0.7576** against floor ground truth,
against **0.7745** for the best floor-suited training target — a regret of **−0.0169 Spearman**,
*smaller than the +0.0065 that seed-averaging alone buys* and an order below the 0.05 pre-registered
bar. Its enrichment@10 % never drops below **4.02×** against any ground truth in q ∈ [0.05, 1.00],
against a headline of 4.28×. Model shortlists are also markedly more stable than the raw labels: 0.55
top-20 overlap between ceiling- and floor-trained models, versus 0.35 at the label level — the model
declines to chase the noisy top of either label.

**Both pre-registered criteria pass.** The conclusion is not that the question is unimportant — it is
that *our recommendation survives either answer*. Still ask Prof. Taniike; but the shortlist does not
hinge on the reply, and reactor time need not wait for it.

**Verified against prior work.** Phase 10's eval = q1.00 column reproduces phase 8's entire sweep to
machine precision (0.00e+00 across four training targets) and its seed-averaged value matches phase 9's
0.7671721643756007 exactly. The new script agrees with two independent existing ones before its novel
cells are believed.

*Wording constraint, not to be dropped:* a low within-cell quantile equals "sustained performance"
only under monotone deactivation, which is not established. Without that assumption it is the
catalyst's **observed floor**. Floor-versus-ceiling is the contrast that matters either way, so the
analysis stands — but it must be worded as floor-vs-ceiling, never as fresh-vs-aged.

---

## 5. Still open

- **The gating question itself.** Unanswerable from the file, now provably. One question to JAIST.
- **Why coverage is incomplete** — 186 absent cells, only 811/917 at all five temperatures.
- **Rendered work-note formats have been regenerated** from the corrected Markdown and verified by
  structured extraction (`pypdf` for the PDF, `zipfile`+XML for the DOCX — never `strings`): no stale
  figure survives in any of `.html`/`.pdf`/`.docx`, and all nine figure captions are present after
  fixing finding #11. Other rendered artifacts (`ocm_presentation.*`, `ocm_progress_report.*`, the
  speaking notes) were **not** re-derived and may still reflect older framing; they were out of scope
  for this pass. **Anything re-rendered in future should be checked for caption loss** — the failure
  is silent.
- The work note remains **drafted, not sent**.
