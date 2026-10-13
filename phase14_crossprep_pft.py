"""
phase14_crossprep_pft.py -- can a two-stage (PFT-style) design close the gap to a plain merge on the
cross-preparation test set?

BACKGROUND. In phase7_prep_ood.json, on T2 (non-impregnation literature), plain merging (C2) had a
higher mean Spearman than a prior-feature arm (C3): 0.388 vs 0.318, per-seed means. phase7 compared
seed means only and ran no paired test; the paired bootstrap here tests C3 vs C2. C3 used a rank-label
stage 1. This script tests other two-stage designs (raw and quantile-normalised stage-1 labels; an
out-of-fold prior inside the merge), plus three merge variants, on the same test set.

RELATION TO THE PROJECT'S PFT. No arm here is the project's PFT. The project's PFT filtered the
literature with DRST, quantile-normalised the literature labels onto the lab scale, and trained
stage 1 jointly with the lab training rows (taniike_validation.py: s1_kind='qn_joint',
use_filter=True). That joint training is the leakage channel named in the work note, section 1,
where the PFT gain was withdrawn. Here every stage 1 sees literature only, and no DRST filter is
used. "pft" in the arm names C4_pft_raw and C4q_pft_qn means "PFT-style two-stage", not the
project's PFT. A failure of these arms is a result about these arms, not about the project's PFT.

DATA. Copied from phase7_prep_ood.py without change. Data.load(); formulation B (one row per
composition, label = max yield over its rows); 65 element columns, prep_enc dropped (constant in the
lab). Literature compositions are grouped by preparation + element vector and labelled by max.
Leakage guard: literature compositions identical to a lab catalyst are dropped from every literature
set (9 dropped). T2 = non-impregnation literature (1,003 compositions), the only test set. TRAIN_LIT =
impregnation literature (742). Lab = 917 catalysts. StandardScaler fit on the 917 lab rows (as in
phase7). T2 labels never enter training: they are used only for scoring.
Limits of the inherited guard (kept unchanged so phase7 reproduces exactly; counts asserted in code):
  - The guard matches element vectors rounded to 4 decimals. 1 TRAIN_LIT composition matches a lab
    catalyst at 2 decimals but not at 4, so it stays in TRAIN_LIT. No T2 composition matches a lab
    catalyst at 2 decimals.
  - Literature grouping uses the raw strings. 3 pairs of TRAIN_LIT compositions (6 rows) are
    identical at 4 decimals but stay separate. In C6, a pair can fall into different KFold folds, so
    a near-twin's label can enter an out-of-fold prior. This touches at most 6 of 742 rows, not T2.
  - T2 holds 176 compositions in 72 groups that share one element vector (the same composition under
    different non-impregnation preparations; 899 distinct element vectors in all). Preparation is not
    a feature, so every arm predicts each group identically. The paired bootstrap treats them as
    independent units, so its CIs may be slightly too narrow. A clustered bootstrap is not run here.

GAS-FLOW CONDITIONS. They cannot enter this analysis. The literature file has no condition columns,
so a literature composition has no gas-flow setting to join. This is not possible from the current
CSV. The script checks that the literature columns hold no condition column.

ARMS. Stage 2 is the TUNED LightGBM (grouped_tuning.json, 300 trees, n_jobs=2). Stage-1 models are
XGBoost (xgb_params(seed, n_jobs=2)), as in phase7. Seeds 0, 1, 2, 7, 13.
  C1_lab_only           lab rows only.                                   (phase7 C1, reproduced)
  C2_merge              lab + TRAIN_LIT rows, raw labels.                (phase7 C2, reproduced)
  C3_prior              stage 1 on TRAIN_LIT rank labels; stage 2 = lab only + prior feature.
                                                                         (phase7 C3, reproduced)
  C4_pft_raw            as C3, stage-1 labels = raw TRAIN_LIT max yields.
  C4q_pft_qn            as C3, stage-1 labels = TRAIN_LIT labels quantile-normalised onto the lab
                        label distribution (ocm_eval.quantile_normalize_y). This is the QN label
                        treatment of the project's PFT, with a literature-only stage 1 and no DRST
                        filter. The project's PFT also filtered the literature with DRST and trained
                        stage 1 jointly with lab rows; that joint training is the leakage channel named
                        in the work note, section 1. Neither is tested here.
  C5_merge_source_flag  C2 + binary column is_literature (lab 0, TRAIN_LIT 1, T2 1).
  C6_merge_oof_prior    C2 + a prior column from a stage-1 model on TRAIN_LIT raw labels. TRAIN_LIT
                        rows get an out-of-fold prior (KFold(5, shuffle, random_state=seed)); lab rows
                        and T2 get the prior from a stage-1 model fit on all of TRAIN_LIT. The code
                        asserts that no TRAIN_LIT row's prior came from a model that saw that row.
  C7_merge_qn_labels    C2 with TRAIN_LIT labels quantile-normalised onto the lab label distribution.
In C3, C4 and C4q the lab rows never trained stage 1 (the leakage guard removes lab compositions from
TRAIN_LIT; asserted), so their prior is out-of-sample, like the prior of a T2 row.

METRICS (population T2_1003). Truth = each T2 composition's literature max yield. Spearman,
enrichment@10% (top-10% overlap / 0.10) and precision@20 (share of the top-20 predicted that are in
the true top 10%), from ocm_eval.cat_metrics. Each is reported two ways and labelled:
  per_seed_mean      mean over the 5 seeds of the metric of each seed's predictions;
  seed_averaged      the metric of the 5-seed mean prediction.
RMSE and mean bias are stored as *_CONFOUNDED, as in phase7: lab labels are max-of-~97 and literature
labels max-of-~2.2, and C4q/C7 change the label scale. They are not skill measures.
precision@20 between arms differs by a few picks out of 20 and has no bootstrap. It is reported, not
used to rank arms.
Differences between arms: ocm_eval.paired_catalyst_bootstrap over the T2 compositions, seed-averaged
predictions, n_boot=3000, seed=7, metrics spearman and enrichment. Every arm vs C2_merge and vs
C1_lab_only. Delta = arm minus reference. A CI that includes zero is "no detectable change".
Seed-to-seed spread is never used as a test.

PRE-REGISTERED RULES (fixed before the results; evaluated in code exactly as written here).
  R-help (phase7's rule). An arm helps out-of-preparation prediction if its T2 Spearman minus C1's
    T2 Spearman, same seed, is >= 0.02 in >= 4 of the 5 seeds.
    phase7 coded this rule as: mean over seeds of that delta >= 0.02 AND the arm beats C1 (delta > 0)
    in >= 4 of 5 seeds. That form is also reported; if the two forms disagree, the script says so.
    R-help is a seed-consistency rule, not a significance test. The bootstrap vs C1 is reported next
    to it.
    If an arm passes on exactly 4 of 5 seeds, the output says so and gives the delta of the seed
    that missed 0.02.
  R-close. A two-stage (PFT-style) arm (C4_pft_raw, C4q_pft_qn, C6_merge_oof_prior) closes the gap
    to the merge only if BOTH hold:
      (a) its T2 Spearman minus C2's T2 Spearman, same seed, is >= 0.02 in >= 4 of the 5 seeds, and
      (b) the paired bootstrap 95% CI of Spearman(arm) - Spearman(C2) (seed-averaged predictions)
          has its lower end above zero.
    Otherwise it does not close the gap. R-close is a Spearman rule; the enrichment bootstrap vs C2
    is printed next to each verdict.

ADDED RULE (not in the spec; added by the implementer; not pre-registered).
  R-beat-merge. The same two-part test as R-close, with the same thresholds, applied to C3_prior,
    C5_merge_source_flag and C7_merge_qn_labels. It answers "does this arm improve on the plain merge
    on T2 Spearman"; it is not the R-close verdict. It cannot be shown that this rule was fixed
    before any result was seen (a --dev run prints seed-0 Spearman for every arm). Because the
    criterion copies R-close, no threshold was chosen. A pass is one of 6 arm-vs-C2 Spearman
    comparisons, with CIs not adjusted for multiplicity, on one test set (T2) only. It must be
    reviewed before it is quoted, and quoted with the enrichment result beside it.

ADDED CHECKS (not pre-registered; reported as sensitivity or description, no verdicts).
  S1  Lab label = B_median (per-(catalyst, temperature) median, then max over temperatures; the
      project's adopted training label). All 8 arms rerun; literature labels unchanged (max); the
      quantile normalisation in C4q/C7 targets the B_median lab distribution. Truth on T2 unchanged.
      The same rules are applied and labelled as sensitivity results. For small deltas the per-seed
      mean and the seed-averaged estimate can differ in sign; quote S1 numbers with their estimator.
  S2  T2 subsets, same seed-averaged predictions as the main run (no retraining):
      T2_no_twin = T2 without the compositions whose element vector also appears in TRAIN_LIT
      (same composition, impregnation vs another preparation); T2_known_prep = T2 without
      preparation "n.a." (preparation unknown). Bootstrap vs C2 on each subset (mask).
  D1  Stage-1 prior alone on T2: Spearman of the stage-1 prediction against T2 truth. Paired
      bootstrap (Spearman, seed-averaged, 3000, seed 7) of prior-alone minus C2_merge and of the
      arm's stage-2 output minus its prior-alone. Descriptive; no verdict on the question.
  D2  Prior coverage: share of T2 compositions whose prior lies outside the range of the prior over
      the stage-2 training rows. Trees cannot extrapolate past that range.
  D3  Share of the merge gain recovered = (arm - C1) / (C2 - C1), seed-averaged Spearman, for the
      arms whose stage 2 is lab-only (C3, C4, C4q). No test. Not computed for C6: C6 already holds
      the whole merge training set, so the ratio would not measure what the prior adds.

Output:  phase14_crossprep_pft.json
Usage:   python3 phase14_crossprep_pft.py [--dev]
         --dev = seed 0 only, 300 resamples, no S1, no file written; rules are not evaluated.
"""
import sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import rankdata, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from ocm_eval import (Data, lgb_params, xgb_params, cat_metrics, TARGET, COND_COLS,
                      quantile_normalize_y, paired_catalyst_bootstrap)

t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)
DEV = '--dev' in sys.argv
SEEDS = [0] if DEV else [0, 1, 2, 7, 13]
N_BOOT = 300 if DEV else 3000
NJ = 2
ARMS = ['C1_lab_only', 'C2_merge', 'C3_prior', 'C4_pft_raw', 'C4q_pft_qn',
        'C5_merge_source_flag', 'C6_merge_oof_prior', 'C7_merge_qn_labels']
TWO_STAGE_ARMS = ['C4_pft_raw', 'C4q_pft_qn', 'C6_merge_oof_prior']
OTHER_VS_MERGE = ['C3_prior', 'C5_merge_source_flag', 'C7_merge_qn_labels']
PRIOR_ARMS = ['C3_prior', 'C4_pft_raw', 'C4q_pft_qn', 'C6_merge_oof_prior']
PHASE7 = json.load(open('phase7_prep_ood.json'))

# ================================================================ data: phase7, unchanged
d = Data.load()
TUNED = json.load(open('grouped_tuning.json'))['confirmation']['tuned']['overrides']
el_all = [c for c in d.features if c != 'Temperature_C']
elements = [c for c in el_all if c != 'prep_enc']            # drop constant prep column (see phase7)

grp = d.dl_lab.groupby(d.groups)
lab_df = grp.agg(**{c: (c, 'first') for c in elements}, y_max=(TARGET, 'max'))
X_lab = lab_df[elements].values.astype(float); y_lab = lab_df['y_max'].values
assert d.dl_lab.Preparation.nunique() == 1, "lab is expected to be single-preparation"

lit_id = d.dl_lit[['Preparation'] + elements].astype(str).agg('|'.join, axis=1)
lit_groups, _ = pd.factorize(lit_id)
lg = d.dl_lit.groupby(lit_groups)
lit_df = lg.agg(Preparation=('Preparation', 'first'),
                **{c: (c, 'first') for c in elements}, y_max=(TARGET, 'max'))
is_imp = (lit_df['Preparation'] == 'Impregnation').values
X_lit = lit_df[elements].values.astype(float); y_lit = lit_df['y_max'].values

lab_keys = set(map(tuple, np.round(X_lab, 4)))
dup = np.array([tuple(r) in lab_keys for r in np.round(X_lit, 4)])
T1 = np.where(is_imp & ~dup)[0]
T2 = np.where(~is_imp & ~dup)[0]
TRAIN_LIT = np.where(is_imp & ~dup)[0]
assert len(set(T1) & set(T2)) == 0
# ================================================================ end of phase7 data block

assert len(T2) == PHASE7['meta']['test_sets']['T2_non_impregnation_lit'] == 1003
assert len(y_lab) == 917 and (lab_df.index.values == np.arange(d.n_cat)).all()
assert len(set(T2) & set(TRAIN_LIT)) == 0, "a T2 composition is in the training literature"
tl_keys = set(map(tuple, np.round(X_lit[TRAIN_LIT], 4)))
assert not (tl_keys & lab_keys), "a lab composition would train stage 1"
# gas-flow conditions: the literature columns must hold none
lit_nonel = sorted(c for c in d.dl_lit.columns if c not in elements)
assert lit_nonel == sorted(['Preparation', 'Temperature_C', TARGET, 'year', 'prep_enc']), lit_nonel
assert not any(c in d.dl_lit.columns for c in COND_COLS)

assert len(elements) == 65 and int(dup.sum()) == 9 and len(TRAIN_LIT) == 742

# limits of the inherited guard (docstring DATA section); counts asserted so the docstring cannot drift
def _keys(X, dp):
    return [tuple(r) for r in np.round(X, dp)]
_tl4 = pd.Series(_keys(X_lit[TRAIN_LIT], 4)).value_counts()
_lab2 = set(_keys(X_lab, 2))
_t2c = pd.Series(_keys(X_lit[T2], 4)).value_counts()
GUARD = {
    'TRAIN_LIT_groups_identical_at_4dp_kept_separate': int((_tl4 > 1).sum()),
    'TRAIN_LIT_rows_in_those_groups': int(_tl4[_tl4 > 1].sum()),
    'TRAIN_LIT_compositions_matching_a_lab_catalyst_at_2dp': int(sum(k in _lab2 for k in _keys(X_lit[TRAIN_LIT], 2))),
    'T2_compositions_matching_a_lab_catalyst_at_2dp': int(sum(k in _lab2 for k in _keys(X_lit[T2], 2))),
    'T2_distinct_element_vectors_4dp': int(len(_t2c)),
    'T2_groups_sharing_one_element_vector': int((_t2c > 1).sum()),
    'T2_compositions_in_those_groups': int(_t2c[_t2c > 1].sum()),
    'note': ('inherited from phase7 and kept for exact reproduction. Shared-element-vector groups in T2 '
             'get identical predictions from every arm; the paired bootstrap treats them as independent '
             'units, so CIs may be slightly too narrow (no clustered bootstrap run here).')}
assert (GUARD['TRAIN_LIT_groups_identical_at_4dp_kept_separate'], GUARD['TRAIN_LIT_rows_in_those_groups'],
        GUARD['TRAIN_LIT_compositions_matching_a_lab_catalyst_at_2dp'],
        GUARD['T2_compositions_matching_a_lab_catalyst_at_2dp'], GUARD['T2_distinct_element_vectors_4dp'],
        GUARD['T2_groups_sharing_one_element_vector'], GUARD['T2_compositions_in_those_groups']) \
    == (3, 6, 1, 0, 899, 72, 176), GUARD

y_T2 = y_lit[T2]
twin = np.array([tuple(r) in tl_keys for r in np.round(X_lit[T2], 4)])
known_prep = (lit_df['Preparation'].values[T2] != 'n.a.')
y_lab_median = (pd.DataFrame({'c': d.groups, 't': d.dl_lab.Temperature_C.values,
                              'y': d.dl_lab[TARGET].values})
                .groupby(['c', 't']).y.median().groupby(level=0).max()
                .reindex(range(d.n_cat)).values)
log(f"lab {len(y_lab)} | TRAIN_LIT {len(TRAIN_LIT)} | T2 {len(T2)} "
    f"(with a TRAIN_LIT twin {int(twin.sum())}; preparation n.a. {int((~known_prep).sum())}) | "
    f"lit compositions dropped as lab duplicates {int(dup.sum())}")


# ================================================================ phase7 metric function, unchanged
def metrics(yt, yp, ids):
    m = cat_metrics(ids, yt, yp)
    m['rmse_CONFOUNDED'] = float(np.sqrt(mean_squared_error(yt, yp)))
    m['mean_signed_bias_CONFOUNDED'] = float(np.mean(yp - yt))
    K = max(1, int(round(0.10 * len(yt))))
    top_pred = np.argsort(-yp)[:K]
    m['topdecile_picks_mean_true_yield'] = float(yt[top_pred].mean())
    m['population_mean_true_yield'] = float(yt.mean())
    m['true_topdecile_mean_yield'] = float(np.sort(yt)[-K:].mean())
    m['n_in_decile'] = int(K)
    return m


def stage1(X, y, seed):
    return xgb.XGBRegressor(**xgb_params(seed, n_jobs=NJ)).fit(X, y)


def fit_predict(arm, seed, yl, test_idx=T2):
    """Train one arm with lab labels yl; return (prediction on test_idx, diagnostics)."""
    sc = StandardScaler().fit(X_lab)
    Xlab = sc.transform(X_lab); Xtl = sc.transform(X_lit[TRAIN_LIT]); Xte = sc.transform(X_lit[test_idx])
    ytl = y_lit[TRAIN_LIT]
    Xtr, ytr, diag = Xlab, yl, {}
    if arm in ('C2_merge', 'C5_merge_source_flag', 'C6_merge_oof_prior', 'C7_merge_qn_labels'):
        ytl_use = quantile_normalize_y(ytl, yl) if arm == 'C7_merge_qn_labels' else ytl
        Xtr = np.vstack([Xlab, Xtl]); ytr = np.concatenate([yl, ytl_use])
        if arm == 'C5_merge_source_flag':
            Xtr = np.hstack([Xtr, np.r_[np.zeros(len(Xlab)), np.ones(len(Xtl))].reshape(-1, 1)])
            Xte = np.hstack([Xte, np.ones((len(Xte), 1))])
        if arm == 'C6_merge_oof_prior':
            full = stage1(Xtl, ytl, seed)
            oof = np.full(len(Xtl), np.nan); src = np.full(len(Xtl), -1); seen = {}
            for k, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=seed).split(Xtl)):
                mk = stage1(Xtl[tr], ytl[tr], seed)
                oof[va] = mk.predict(Xtl[va]); src[va] = k; seen[k] = tr
            # no TRAIN_LIT row's prior came from a model that saw that row
            assert not np.isnan(oof).any() and (src >= 0).all()
            for k, tr in seen.items():
                assert np.intersect1d(np.where(src == k)[0], tr).size == 0, "OOF prior leaked"
            p_lab, p_te = full.predict(Xlab), full.predict(Xte)
            prior_tr = np.r_[p_lab, oof]
            Xtr = np.hstack([Xtr, prior_tr.reshape(-1, 1)]); Xte = np.hstack([Xte, p_te.reshape(-1, 1)])
            diag = {'prior_test': p_te, 'prior_train': prior_tr}
    elif arm in ('C3_prior', 'C4_pft_raw', 'C4q_pft_qn'):
        ys1 = {'C3_prior': rankdata(ytl, method='average') / (len(TRAIN_LIT) + 1),
               'C4_pft_raw': ytl,
               'C4q_pft_qn': quantile_normalize_y(ytl, yl)}[arm]
        pre = stage1(Xtl, ys1, seed)
        p_lab, p_te = pre.predict(Xlab), pre.predict(Xte)
        Xtr = np.hstack([Xlab, p_lab.reshape(-1, 1)]); Xte = np.hstack([Xte, p_te.reshape(-1, 1)])
        diag = {'prior_test': p_te, 'prior_train': p_lab}
    elif arm != 'C1_lab_only':
        raise ValueError(arm)
    m = lgb.LGBMRegressor(**lgb_params(seed, **TUNED, n_jobs=NJ)).fit(Xtr, ytr)
    return m.predict(Xte), diag


def run_all(yl, tag):
    """All arms x seeds on T2. Returns per-seed predictions and diagnostics."""
    P, D = {}, {}
    for arm in ARMS:
        P[arm], D[arm] = [], []
        for s in SEEDS:
            p, dg = fit_predict(arm, s, yl)
            P[arm].append(p); D[arm].append(dg)
        sp = [spearmanr(y_T2, p)[0] for p in P[arm]]
        log(f"{tag} T2_1003 {arm:21s} Spearman per seed {np.round(sp, 4).tolist()}")
    return P, D


def summarise(P):
    """Per-seed, per-seed-mean and seed-averaged-prediction metrics on T2_1003."""
    out = {}
    for arm, preds in P.items():
        runs = [metrics(y_T2, p, np.arange(len(T2))) for p in preds]
        out[arm] = {
            'population': 'T2_1003',
            'per_seed': {'seeds': SEEDS,
                         'spearman': [r['spearman_max'] for r in runs],
                         'enrichment_top10pct': [r['enrichment_top10pct'] for r in runs],
                         'precision_at20_vs_top10pct': [r['precision_at20_vs_top10pct'] for r in runs]},
            'per_seed_mean': {k: float(np.mean([r[k] for r in runs])) for k in runs[0]},
            'seed_averaged_predictions': metrics(y_T2, np.mean(preds, axis=0), np.arange(len(T2)))}
    return out


def boot_block(P, refs, mask=None, metrics_=('spearman', 'enrichment')):
    out = {}
    avg = {a: np.mean(p, axis=0) for a, p in P.items()}
    for ref in refs:
        out[f'vs_{ref}'] = {}
        for arm in ARMS:
            if arm == ref:
                continue
            out[f'vs_{ref}'][arm] = {}
            for met in metrics_:
                b = paired_catalyst_bootstrap(y_T2, avg[arm], avg[ref], mask=mask,
                                              n_boot=N_BOOT, seed=7, metric=met)
                b['verdict'] = ('arm higher (CI above zero)' if b['ci95_low'] > 0 else
                                'arm lower (CI below zero)' if b['ci95_high'] < 0 else
                                'no detectable change')
                out[f'vs_{ref}'][arm][met] = b
    return out


def evaluate_rules(S, B):
    """R-help, R-close, R-beat-merge, exactly as stated in the docstring."""
    if len(SEEDS) != 5:
        return {'note': 'rules need 5 seeds; not evaluated in a dev run'}
    sp = {a: np.array(S[a]['per_seed']['spearman']) for a in ARMS}
    R = {'R_help': {}, 'R_close': {}, 'R_beat_merge': {}}
    for arm in ARMS[1:]:
        dl = sp[arm] - sp['C1_lab_only']
        n_ge = int((dl >= 0.02).sum())
        literal = bool(n_ge >= 4)
        coded = bool(dl.mean() >= 0.02 and int((dl > 0).sum()) >= 4)
        R['R_help'][arm] = {
            'per_seed_delta_vs_C1': dl.tolist(), 'n_seeds_delta_ge_0.02': n_ge,
            'HELPS': literal,
            'passes_on_exactly_4_of_5_seeds': bool(literal and n_ge == 4),
            'smallest_per_seed_delta_vs_C1': float(dl.min()),
            'phase7_coded_form': {'mean_over_seeds_of_per_seed_delta': float(dl.mean()),
                                  'n_seeds_delta_gt_0': int((dl > 0).sum()), 'HELPS': coded},
            'forms_agree': literal == coded,
            'bootstrap_spearman_vs_C1_verdict': B['vs_C1_lab_only'][arm]['spearman']['verdict']}
    for arm in TWO_STAGE_ARMS + OTHER_VS_MERGE:
        dm = sp[arm] - sp['C2_merge']
        n_ge = int((dm >= 0.02).sum())
        b = B['vs_C2_merge'][arm]['spearman']
        a_ok, b_ok = bool(n_ge >= 4), bool(b['ci95_low'] > 0)
        key = 'R_close' if arm in TWO_STAGE_ARMS else 'R_beat_merge'
        be = B['vs_C2_merge'][arm]['enrichment']
        R[key][arm] = {'per_seed_delta_vs_C2': dm.tolist(), 'n_seeds_delta_ge_0.02': n_ge,
                       'part_a_seed_rule': a_ok,
                       'bootstrap_spearman_delta_point_seed_averaged': b['delta_point'],
                       'bootstrap_spearman_ci95': [b['ci95_low'], b['ci95_high']],
                       'part_b_ci_above_zero': b_ok, 'PASSES': bool(a_ok and b_ok),
                       'not_part_of_rule_enrichment_vs_C2_seed_averaged': {
                           'delta_point': be['delta_point'], 'ci95': [be['ci95_low'], be['ci95_high']],
                           'verdict': be['verdict']}}
    R['R_beat_merge_status'] = ('added by the implementer, not in the spec, not pre-registered; same '
                                'criterion and thresholds as R_close; Spearman only; one of 6 arm-vs-C2 '
                                'Spearman comparisons, CIs not adjusted for multiplicity; T2 only')
    return R


# ================================================================ main run (lab label = max, phase7)
P_main, D_main = run_all(y_lab, 'main')
S_main = summarise(P_main)

# ---- reproduction of phase7 (exact, per seed) ----
REPRO = {}
B7 = PHASE7['B_can_literature_help']
for arm in ['C1_lab_only', 'C2_merge', 'C3_prior']:
    ref = dict(zip(PHASE7['meta']['seeds'], B7[arm]['per_seed_spearman']))
    here = S_main[arm]['per_seed']['spearman']
    diff = max(abs(h - ref[s]) for s, h in zip(SEEDS, here))
    assert diff < 1e-9, f"{arm} does not reproduce phase7 (max |diff| {diff})"
    if not DEV:
        for k in ['spearman_max', 'enrichment_top10pct', 'precision_at20_vs_top10pct']:
            assert abs(S_main[arm]['per_seed_mean'][k] - B7[arm][k]) < 1e-9, (arm, k)
    REPRO[arm] = {'phase7_per_seed_spearman': [ref[s] for s in SEEDS], 'here_per_seed_spearman': here,
                  'max_abs_diff': float(diff), 'reproduced': True}
# T1 lab-only reference (phase7 section A)
t1 = [metrics(y_lit[T1], fit_predict('C1_lab_only', s, y_lab, test_idx=T1)[0], np.arange(len(T1)))
      ['spearman_max'] for s in SEEDS]
ref_t1 = dict(zip(PHASE7['meta']['seeds'], PHASE7['A_shift_decomposition']['T1_impregnation_lit']['per_seed_spearman']))
diff_t1 = max(abs(a - ref_t1[s]) for s, a in zip(SEEDS, t1))
assert diff_t1 < 1e-9, "T1 lab-only does not reproduce phase7"
REPRO['T1_742_C1_lab_only'] = {'phase7_per_seed_spearman': [ref_t1[s] for s in SEEDS],
                               'here_per_seed_spearman': t1, 'max_abs_diff': float(diff_t1),
                               'per_seed_mean': float(np.mean(t1)), 'reproduced': True}
log("phase7 reproduction: C1, C2, C3 on T2 and C1 on T1 match per seed (|diff| < 1e-9)")

B_main = boot_block(P_main, ['C2_merge', 'C1_lab_only'])
R_main = evaluate_rules(S_main, B_main)
log("main bootstrap and rules done")

# ---- D1 / D2: the stage-1 prior on its own, and its coverage ----
DIAG = {}
for arm in PRIOR_ARMS:
    sp1 = [spearmanr(y_T2, dg['prior_test'])[0] for dg in D_main[arm]]
    sp1_avg = spearmanr(y_T2, np.mean([dg['prior_test'] for dg in D_main[arm]], axis=0))[0]
    out = [float(np.mean((dg['prior_test'] < dg['prior_train'].min()) |
                         (dg['prior_test'] > dg['prior_train'].max()))) for dg in D_main[arm]]
    prior_avg = np.mean([dg['prior_test'] for dg in D_main[arm]], axis=0)
    arm_avg, c2_avg = np.mean(P_main[arm], axis=0), np.mean(P_main['C2_merge'], axis=0)
    bt = {}
    for lbl, a_, b_ in [('prior_alone_minus_C2_merge', prior_avg, c2_avg),
                        ('stage2_arm_minus_prior_alone', arm_avg, prior_avg)]:
        b = paired_catalyst_bootstrap(y_T2, a_, b_, n_boot=N_BOOT, seed=7, metric='spearman')
        b['verdict'] = ('first higher (CI above zero)' if b['ci95_low'] > 0 else
                        'first lower (CI below zero)' if b['ci95_high'] < 0 else 'no detectable change')
        bt[lbl] = b
    DIAG[arm] = {'population': 'T2_1003',
                 'stage1_prior_alone_spearman_per_seed_mean': float(np.mean(sp1)),
                 'stage1_prior_alone_spearman_seed_averaged': float(sp1_avg),
                 'share_T2_prior_outside_training_prior_range_per_seed_mean': float(np.mean(out)),
                 'bootstrap_spearman_seed_averaged_descriptive': bt}
# C6's full stage-1 model is the same model as C4's stage 1 (same labels, data, seed)
for i in range(len(SEEDS)):
    assert np.array_equal(D_main['C6_merge_oof_prior'][i]['prior_test'], D_main['C4_pft_raw'][i]['prior_test'])
DIAG['note'] = ("C6's T2 prior comes from the same stage-1 model as C4's (asserted identical), so their "
                "prior-alone Spearman matches. Its training-row prior range differs: C6 trains on lab + "
                "TRAIN_LIT OOF priors, C4 on lab priors only.")

# ---- S2: T2 subsets, same seed-averaged predictions ----
S2 = {}
for name, mask in [('T2_no_twin', ~twin), ('T2_known_prep', known_prep)]:
    avg = {a: np.mean(P_main[a], axis=0) for a in ARMS}
    S2[name] = {'n_compositions': int(mask.sum()),
                'seed_averaged_spearman': {a: float(spearmanr(y_T2[mask], avg[a][mask])[0]) for a in ARMS},
                'bootstrap': boot_block(P_main, ['C2_merge'], mask=mask, metrics_=('spearman',))['vs_C2_merge']}
log("S2 subsets done")

# ================================================================ S1: lab label = B_median
S1 = None
if not DEV:
    P_med, _ = run_all(y_lab_median, 'S1_B_median')
    S_med = summarise(P_med)
    B_med = boot_block(P_med, ['C2_merge', 'C1_lab_only'])
    S1 = {'label': 'lab label = B_median (per-(catalyst, temperature) median, max over temperatures); '
                   'literature labels and T2 truth unchanged. Sensitivity, not the pre-registered result.',
          'arms': S_med, 'bootstrap': B_med, 'rules_applied_as_sensitivity': evaluate_rules(S_med, B_med)}
    log("S1 done")

# ================================================================ key numbers (flat, labelled)
KEY = {}
for arm in ARMS:
    KEY[f'T2_1003.{arm}.spearman.per_seed_mean'] = S_main[arm]['per_seed_mean']['spearman_max']
    KEY[f'T2_1003.{arm}.spearman.seed_averaged'] = S_main[arm]['seed_averaged_predictions']['spearman_max']
    KEY[f'T2_1003.{arm}.enrichment.seed_averaged'] = S_main[arm]['seed_averaged_predictions']['enrichment_top10pct']
    KEY[f'T2_1003.{arm}.precision_at20.seed_averaged'] = S_main[arm]['seed_averaged_predictions']['precision_at20_vs_top10pct']
for arm in ARMS:
    for ref in ['C2_merge', 'C1_lab_only']:
        if arm == ref:
            continue
        b = B_main[f'vs_{ref}'][arm]['spearman']
        KEY[f'T2_1003.{arm}_minus_{ref}.spearman.seed_averaged.delta'] = b['delta_point']
        KEY[f'T2_1003.{arm}_minus_{ref}.spearman.seed_averaged.ci95'] = [b['ci95_low'], b['ci95_high']]
sa = {a: S_main[a]['seed_averaged_predictions']['spearman_max'] for a in ARMS}
gap = sa['C2_merge'] - sa['C1_lab_only']
for arm in ['C3_prior', 'C4_pft_raw', 'C4q_pft_qn']:      # lab-only stage 2 only (see D3)
    KEY[f'T2_1003.{arm}.share_of_merge_gain_recovered.seed_averaged_spearman'] = (sa[arm] - sa['C1_lab_only']) / gap
for arm in PRIOR_ARMS:
    KEY[f'T2_1003.{arm}.stage1_prior_alone.spearman.seed_averaged'] = DIAG[arm]['stage1_prior_alone_spearman_seed_averaged']
if 'note' not in R_main:
    for arm, r in R_main['R_help'].items():
        KEY[f'rule.R_help.{arm}.HELPS'] = r['HELPS']
    for arm, r in R_main['R_close'].items():
        KEY[f'rule.R_close.{arm}.PASSES_on_T2_spearman'] = r['PASSES']
    for arm, r in R_main['R_beat_merge'].items():
        KEY[f'rule.R_beat_merge_ADDED_not_in_spec.{arm}.PASSES_on_T2_spearman'] = r['PASSES']
for name, v in S2.items():
    for arm in ['C2_merge', 'C4q_pft_qn', 'C6_merge_oof_prior', 'C7_merge_qn_labels']:
        KEY[f'S2.{name}_{v["n_compositions"]}.{arm}.spearman.seed_averaged'] = v['seed_averaged_spearman'][arm]
if S1 is not None:
    for arm in ARMS:
        KEY[f'S1_B_median_label.T2_1003.{arm}.spearman.seed_averaged'] = S1['arms'][arm]['seed_averaged_predictions']['spearman_max']
    for arm in ARMS[1:]:
        if arm != 'C2_merge':
            b = S1['bootstrap']['vs_C2_merge'][arm]['spearman']
            KEY[f'S1_B_median_label.T2_1003.{arm}_minus_C2_merge.spearman.seed_averaged.ci95'] = [b['ci95_low'], b['ci95_high']]

# ================================================================ caveats (computed from the results)
def sign_disagreements(S_):
    """Arms whose delta vs C2 has a different sign as a per-seed mean than on seed-averaged predictions."""
    out = {}
    for arm in ARMS:
        if arm == 'C2_merge':
            continue
        dps = S_[arm]['per_seed_mean']['spearman_max'] - S_['C2_merge']['per_seed_mean']['spearman_max']
        dsa = (S_[arm]['seed_averaged_predictions']['spearman_max']
               - S_['C2_merge']['seed_averaged_predictions']['spearman_max'])
        if np.sign(dps) != np.sign(dsa):
            out[arm] = {'spearman_minus_C2_per_seed_mean': float(dps),
                        'spearman_minus_C2_seed_averaged': float(dsa)}
    return out


CAVEATS = {}
if 'note' not in R_main:
    for arm, r in R_main['R_beat_merge'].items():
        if r['PASSES']:
            e = r['not_part_of_rule_enrichment_vs_C2_seed_averaged']
            lo, hi = r['bootstrap_spearman_ci95']
            CAVEATS[f'{arm}.passes_R_beat_merge'] = (
                f"new positive result: T2_1003 Spearman minus C2_merge (seed-averaged predictions) "
                f"{r['bootstrap_spearman_delta_point_seed_averaged']:+.4f} [{lo:+.4f}, {hi:+.4f}]. "
                f"Enrichment minus C2_merge {e['delta_point']:+.2f} [{e['ci95'][0]:+.2f}, {e['ci95'][1]:+.2f}]: "
                f"{e['verdict']}. precision@20 not tested. R_beat_merge was added by the implementer and is "
                f"not pre-registered; this is one of 6 arm-vs-C2 Spearman comparisons (CIs not adjusted for "
                f"multiplicity), on T2 only. Review before quoting, and quote both metrics.")
    for arm, r in R_main['R_help'].items():
        if r['passes_on_exactly_4_of_5_seeds']:
            CAVEATS[f'{arm}.R_help_on_exactly_4_of_5_seeds'] = (
                f"HELPS rests on exactly 4 of 5 seeds; the fifth seed's delta vs C1 is "
                f"{r['smallest_per_seed_delta_vs_C1']:+.4f}. One more seed below 0.02 would flip the verdict.")
CAVEATS['estimator_sign_disagreements_vs_C2.main_T2_1003'] = sign_disagreements(S_main)
if S1 is not None:
    CAVEATS['estimator_sign_disagreements_vs_C2.S1_B_median'] = sign_disagreements(S1['arms'])
CAVEATS['precision_at20'] = 'few picks out of 20 separate arms; no bootstrap; not used to rank arms'

RES = {
    'meta': {
        'question': ('can a two-stage design (C4_pft_raw, C4q_pft_qn, C6_merge_oof_prior) close the gap to a '
                     'plain merge on the cross-preparation test set T2?'),
        'relation_to_project_PFT': ('no arm here is the project\'s PFT. The project\'s PFT filtered the literature '
                                    'with DRST, quantile-normalised literature labels, and trained stage 1 jointly '
                                    'with lab training rows (taniike_validation.py s1_kind=qn_joint, '
                                    'use_filter=True); that joint training is the leakage channel named in the work '
                                    'note section 1, where the PFT gain was withdrawn. "pft" in arm names means '
                                    'PFT-style two-stage.'),
        'C4q_relation_to_project_PFT': ('the QN label treatment of the project\'s PFT, with a literature-only '
                                        'stage 1 and no DRST filter. The project\'s PFT also filtered the '
                                        'literature with DRST and trained stage 1 jointly with lab rows; that '
                                        'joint training is the leakage channel named in the work note section 1. '
                                        'Neither is tested here.'),
        'data': 'phase7_prep_ood.py data block, unchanged (formulation B, labels = max, prep_enc dropped, lab-duplicate guard)',
        'populations': {'T2_1003': 'non-impregnation literature compositions (test only)',
                        'TRAIN_LIT_742': 'impregnation literature compositions (training only)',
                        'lab_917': 'lab catalysts (training only)',
                        'T2_twin_in_TRAIN_LIT': int(twin.sum()), 'T2_preparation_na': int((~known_prep).sum()),
                        'n_T2': int(len(T2)), 'n_TRAIN_LIT': int(len(TRAIN_LIT)), 'n_lab': int(len(y_lab)),
                        'n_elements': int(len(elements)),
                        'n_lit_dropped_as_lab_duplicates': int(dup.sum())},
        'guard_limits_and_T2_structure': GUARD,
        'scoring': 'truth = literature max yield per T2 composition; spearman, enrichment@10%, precision@20 from cat_metrics',
        'per_seed_mean_vs_seed_averaged': 'per_seed_mean = mean of per-seed metrics; seed_averaged = metric of the 5-seed mean prediction',
        'seeds': SEEDS, 'n_boot': N_BOOT, 'bootstrap_seed': 7, 'n_jobs': NJ, 'dev_run': DEV,
        'stage2': 'TUNED LightGBM (grouped_tuning.json)', 'stage1': 'XGBoost xgb_params(seed, n_jobs=2)',
        'C6_stage1_labels': 'raw TRAIN_LIT max yields',
        'gas_flow_conditions': ('cannot enter this analysis: the literature file has no condition columns '
                                '(not possible from the current CSV); checked in code'),
        'rules': {'R_help': 'per-seed (arm - C1) Spearman >= 0.02 in >= 4/5 seeds; phase7 coded form also reported',
                  'R_close': 'two-stage (PFT-style) arms C4, C4q, C6: per-seed (arm - C2) Spearman >= 0.02 in '
                             '>= 4/5 seeds AND paired bootstrap CI of Spearman(arm) - Spearman(C2) lower end > 0',
                  'R_beat_merge': ('ADDED by the implementer, not in the spec, not pre-registered; same criterion '
                                   'as R_close, applied to C3, C5, C7; Spearman only; not the R_close verdict')},
        'precision_at20_note': ('differences between arms are a few picks out of 20 and have no bootstrap; '
                                'reported, not used to rank arms'),
        'rmse_caveat': PHASE7['meta']['rmse_caveat'] + '; C4q and C7 also change the label scale'},
    'reproduction_of_phase7': REPRO,
    'arms_T2_1003': S_main,
    'bootstrap_T2_1003_seed_averaged': B_main,
    'rules': R_main,
    'diagnostics_stage1_prior': DIAG,
    'sensitivity_S2_T2_subsets': S2,
    'sensitivity_S1_lab_label_B_median': S1,
    'caveats': CAVEATS,
    'key_numbers': KEY,
    'runtime_minutes': (time.time() - t0) / 60}

if not DEV:
    json.dump(RES, open('phase14_crossprep_pft.json', 'w'), indent=1)
    log("wrote phase14_crossprep_pft.json")
else:
    log("dev run: nothing written")

# ================================================================ summary
print("\n================ SUMMARY (population T2_1003, truth = literature max) ================")
print(f"Seeds {SEEDS}. 'per-seed mean' = mean of per-seed metrics; 'seed-avg' = metric of the mean prediction.")
print(f"{'arm':22s} {'Spearman':>9s} {'Spearman':>9s} {'enrich':>7s} {'prec@20':>8s}")
print(f"{'':22s} {'per-seed':>9s} {'seed-avg':>9s} {'seed-avg':>7s} {'seed-avg':>8s}")
for arm in ARMS:
    a = S_main[arm]
    print(f"{arm:22s} {a['per_seed_mean']['spearman_max']:9.4f} {a['seed_averaged_predictions']['spearman_max']:9.4f} "
          f"{a['seed_averaged_predictions']['enrichment_top10pct']:6.2f}x "
          f"{a['seed_averaged_predictions']['precision_at20_vs_top10pct']:8.2f}")
print("\nPaired catalyst bootstrap over T2_1003 compositions, seed-averaged predictions, "
      f"{N_BOOT} resamples, seed 7. Delta = arm - reference.")
for ref in ['C2_merge', 'C1_lab_only']:
    print(f" vs {ref}")
    for arm in ARMS:
        if arm == ref:
            continue
        s_, e_ = B_main[f'vs_{ref}'][arm]['spearman'], B_main[f'vs_{ref}'][arm]['enrichment']
        print(f"   {arm:22s} Spearman {s_['delta_point']:+.4f} [{s_['ci95_low']:+.4f}, {s_['ci95_high']:+.4f}] "
              f"{s_['verdict']:27s} | enrichment {e_['delta_point']:+.2f} [{e_['ci95_low']:+.2f}, "
              f"{e_['ci95_high']:+.2f}] {e_['verdict']}")
if 'note' in R_main:
    print("\nRules:", R_main['note'])
else:
    print("\nR-help (per-seed arm - C1 Spearman >= 0.02 in >= 4/5 seeds):")
    for arm, r in R_main['R_help'].items():
        flag = '' if r['forms_agree'] else '  <-- phase7 coded form DISAGREES: ' + str(r['phase7_coded_form']['HELPS'])
        if r['passes_on_exactly_4_of_5_seeds']:
            flag += f"  [exactly 4/5; fifth seed's delta {r['smallest_per_seed_delta_vs_C1']:+.4f}]"
        print(f"   {arm:22s} {r['n_seeds_delta_ge_0.02']}/5 seeds -> {'HELPS' if r['HELPS'] else 'does not help'}"
              f"   (bootstrap vs C1: {r['bootstrap_spearman_vs_C1_verdict']}){flag}")
    for key, title in [('R_close', 'R-close (two-stage arms C4/C4q/C6 vs C2_merge; Spearman rule)'),
                       ('R_beat_merge', 'R-beat-merge (ADDED by the implementer, not in the spec; '
                                        'C3/C5/C7 vs C2_merge; Spearman rule; not R-close)')]:
        print(f"{title}: per-seed delta >= 0.02 in >= 4/5 seeds AND bootstrap CI above zero")
        for arm, r in R_main[key].items():
            verdict = ('closes the gap' if r['PASSES'] else 'does not close the gap') if key == 'R_close' \
                else ('beats the merge on T2 Spearman' if r['PASSES'] else 'does not beat the merge on T2 Spearman')
            e = r['not_part_of_rule_enrichment_vs_C2_seed_averaged']
            print(f"   {arm:22s} seeds {r['n_seeds_delta_ge_0.02']}/5, Spearman CI [{r['bootstrap_spearman_ci95'][0]:+.4f}, "
                  f"{r['bootstrap_spearman_ci95'][1]:+.4f}] -> {verdict}; enrichment vs C2 (not part of the rule): "
                  f"{e['delta_point']:+.2f} [{e['ci95'][0]:+.2f}, {e['ci95'][1]:+.2f}] {e['verdict']}")
print("\nD1/D2 stage-1 prior alone on T2_1003 (descriptive):")
for arm in PRIOR_ARMS:
    v = DIAG[arm]
    print(f"   {arm:22s} prior-alone Spearman per-seed mean {v['stage1_prior_alone_spearman_per_seed_mean']:.4f}, "
          f"seed-avg {v['stage1_prior_alone_spearman_seed_averaged']:.4f}; T2 priors outside training range "
          f"{v['share_T2_prior_outside_training_prior_range_per_seed_mean']:.1%} (per-seed mean)")
    for lbl, b in v['bootstrap_spearman_seed_averaged_descriptive'].items():
        print(f"      {lbl:30s} Spearman {b['delta_point']:+.4f} [{b['ci95_low']:+.4f}, {b['ci95_high']:+.4f}] "
              f"{b['verdict']}")
print("\nS2 subsets (sensitivity; same seed-averaged predictions; bootstrap Spearman vs C2_merge):")
for name, v in S2.items():
    print(f"   {name} (n={v['n_compositions']})")
    for arm in ARMS:
        extra = '' if arm == 'C2_merge' else (
            f"  vs C2 {v['bootstrap'][arm]['spearman']['delta_point']:+.4f} "
            f"[{v['bootstrap'][arm]['spearman']['ci95_low']:+.4f}, {v['bootstrap'][arm]['spearman']['ci95_high']:+.4f}] "
            f"{v['bootstrap'][arm]['spearman']['verdict']}")
        print(f"      {arm:22s} Spearman seed-avg {v['seed_averaged_spearman'][arm]:.4f}{extra}")
if S1 is not None:
    print("\nS1 sensitivity: lab label = B_median (not the pre-registered result). Population T2_1003.")
    for arm in ARMS:
        a = S1['arms'][arm]
        extra = '' if arm == 'C2_merge' else (
            f"  vs C2 {S1['bootstrap']['vs_C2_merge'][arm]['spearman']['delta_point']:+.4f} "
            f"[{S1['bootstrap']['vs_C2_merge'][arm]['spearman']['ci95_low']:+.4f}, "
            f"{S1['bootstrap']['vs_C2_merge'][arm]['spearman']['ci95_high']:+.4f}] "
            f"{S1['bootstrap']['vs_C2_merge'][arm]['spearman']['verdict']}")
        print(f"   {arm:22s} Spearman per-seed mean {a['per_seed_mean']['spearman_max']:.4f}, "
              f"seed-avg {a['seed_averaged_predictions']['spearman_max']:.4f}{extra}")
    rr = S1['rules_applied_as_sensitivity']
    print("   R-close applied as sensitivity: " +
          ', '.join(f"{a} {'passes' if r['PASSES'] else 'fails'}" for a, r in rr['R_close'].items()))
print("\nCaveats:")
for k, v in CAVEATS.items():
    print(f"   {k}: {v}")
print("\nNo arm here is the project's PFT (DRST filter + QN labels + stage 1 trained jointly with lab rows). "
      "C4q uses the QN labels only, with a literature-only stage 1.")
print("\nGas-flow conditions: not used. The literature file has no condition columns, so this is not "
      "possible from the current CSV.")
log("done")
