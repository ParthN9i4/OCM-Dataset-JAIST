"""
phase6_our_experiments.py — two experiments of our own design, not prompted by the review.

EXPERIMENT 1 — Coverage-moderated prior hypothesis.
Phase 4 found one seed-stable positive: the literature rank prior helped the Zr family holdout
(+0.025 over composition-only, 5 seeds), and Zr had the thinnest element-specific literature
coverage. That suggested "the prior helps where coverage is thin" — but n=5 families has no power
and the ordering was not clean (Ce broke it). Here we test it across every element with >=50 lab
catalysts and correlate the per-family (V1 - V0) delta against coverage descriptors.

PRE-REGISTERED RULE (fixed before running): the hypothesis is supported only if
|Spearman(delta, some coverage descriptor)| >= 0.5 with a consistent sign across >=12 families.
Otherwise Zr is declared a chance finding and the Phase-3 null stands unqualified.

EXPERIMENT 2 — Family learning curve (the actionable one).
The Phase-4 drop-column proof showed an unseen promoter family is structurally unpriceable: with
zero family members in training the column is constant, trees never split on it, and predictions are
bit-identical to deleting the column. The decision-relevant follow-up is therefore NOT "does it
fail" but "how many labeled examples of a family are needed before the model can price it?"
For each family we hold out a fixed test subset, then vary how many OTHER family members appear in
training (0, 5, 10, 25, ...), and record where performance saturates. This converts a limitation
into an experimental-design recommendation: synthesize ~N of a new family to make it predictable.

Protocol throughout: formulation B (per-catalyst max yield), tuned LGBM, per-split train-only
scaler, element-containing literature excluded from the prior so the family claim stays clean.
Output: phase6_our_experiments.json
"""
import warnings, time, json
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import rankdata, spearmanr
from sklearn.preprocessing import StandardScaler
from ocm_eval import Data, lgb_params, xgb_params, cat_metrics, TARGET

t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

d = Data.load()
TUNED = json.load(open('grouped_tuning.json'))['confirmation']['tuned']['overrides']
el_cols = [c for c in d.features if c != 'Temperature_C']
elements = [c for c in el_cols if c != 'prep_enc']

grp = d.dl_lab.groupby(d.groups)
cat_df = grp.agg(**{c: (c, 'first') for c in el_cols}, y_max=(TARGET, 'max'))
Xc = cat_df[el_cols].values.astype(float); yc = cat_df['y_max'].values
n_cat = len(yc)
TOP10 = yc >= np.quantile(yc, 0.90)

lit_id = d.dl_lit[['Preparation'] + elements].astype(str).agg('|'.join, axis=1)
lit_groups, _ = pd.factorize(lit_id)
litc_df = d.dl_lit.groupby(lit_groups).agg(**{c: (c, 'first') for c in el_cols}, y_max=(TARGET, 'max'))
Xlc = litc_df[el_cols].values.astype(float); ylc = litc_df['y_max'].values
ylc_rank = rankdata(ylc, method='average') / (len(ylc) + 1)

FAMILIES = sorted([e for e in elements if (cat_df[e].values > 0).sum() >= 50],
                  key=lambda e: -(cat_df[e].values > 0).sum())
log(f"{n_cat} catalysts, {len(ylc)} literature compositions, {len(FAMILIES)} families with >=50 catalysts")
log(f"families: {FAMILIES}")

SEEDS = [0, 1, 2, 7, 13]
RES = {'meta': {'n_catalysts': int(n_cat), 'n_families': len(FAMILIES), 'families': FAMILIES,
                'seeds': SEEDS, 'protocol': 'formulation B family holdout, tuned LGBM, lit-excluded prior',
                'exp1_preregistered_rule': '|Spearman(delta, coverage descriptor)| >= 0.5, consistent sign, >=12 families'}}


# ============================================================ EXPERIMENT 1
def family_holdout(el, use_prior, seed):
    has = cat_df[el].values > 0
    te, tr = np.where(has)[0], np.where(~has)[0]
    sc = StandardScaler().fit(Xc[tr])
    Xc_sc = sc.transform(Xc)
    X = Xc_sc
    if use_prior:
        keep = litc_df[el].values <= 0          # element-containing literature removed: clean claim
        pre = xgb.XGBRegressor(**xgb_params(seed)).fit(sc.transform(Xlc[keep]), ylc_rank[keep])
        X = np.hstack([Xc_sc, pre.predict(Xc_sc).reshape(-1, 1)])
    m = lgb.LGBMRegressor(**lgb_params(seed, **TUNED)).fit(X[tr], yc[tr])
    return cat_metrics(np.arange(n_cat)[te], yc[te], m.predict(X[te]))


E1 = {}
for el in FAMILIES:
    has = cat_df[el].values > 0
    v0 = [family_holdout(el, False, s)['spearman_max'] for s in SEEDS]
    v1 = [family_holdout(el, True, s)['spearman_max'] for s in SEEDS]
    E1[el] = {
        'n_lab_catalysts': int(has.sum()),
        'n_lit_compositions': int((litc_df[el].values > 0).sum()),
        'lab_lit_ratio': float(has.sum() / max(1, (litc_df[el].values > 0).sum())),
        'top_decile_share': float(TOP10[has].sum() / TOP10.sum()),
        'family_ymax_mean': float(yc[has].mean()),
        'v0_spearman': float(np.mean(v0)), 'v1_spearman': float(np.mean(v1)),
        'delta': float(np.mean(v1) - np.mean(v0)),
        'delta_std': float(np.std(np.array(v1) - np.array(v0), ddof=1)),
        'seed_wins': int(sum(a > b for a, b in zip(v1, v0)))}
    log(f"E1 {el:3s} n={E1[el]['n_lab_catalysts']:3d} lit={E1[el]['n_lit_compositions']:3d}  "
        f"V0={E1[el]['v0_spearman']:.3f} V1={E1[el]['v1_spearman']:.3f}  "
        f"delta={E1[el]['delta']:+.3f} wins={E1[el]['seed_wins']}/5")

deltas = np.array([E1[e]['delta'] for e in FAMILIES])
CORR = {}
for desc in ['n_lab_catalysts', 'n_lit_compositions', 'lab_lit_ratio', 'top_decile_share', 'family_ymax_mean']:
    vals = np.array([E1[e][desc] for e in FAMILIES])
    rho, p = spearmanr(vals, deltas)
    CORR[desc] = {'spearman': float(rho), 'p': float(p)}
best = max(CORR.items(), key=lambda kv: abs(kv[1]['spearman']))
supported = bool(abs(best[1]['spearman']) >= 0.5 and len(FAMILIES) >= 12)
RES['E1_coverage_hypothesis'] = {'per_family': E1, 'correlations': CORR,
                                 'strongest_descriptor': best[0],
                                 'strongest_spearman': best[1]['spearman'],
                                 'n_families': len(FAMILIES),
                                 'HYPOTHESIS_SUPPORTED': supported,
                                 'mean_delta': float(deltas.mean()),
                                 'families_with_positive_delta': int((deltas > 0).sum())}
log(f"E1 verdict: strongest descriptor={best[0]} rho={best[1]['spearman']:+.3f} -> "
    f"{'SUPPORTED' if supported else 'NOT SUPPORTED'}")


# ============================================================ EXPERIMENT 2
def learning_curve(el, n_seen, seed, test_frac=0.30):
    has = np.where(cat_df[el].values > 0)[0]
    rng = np.random.default_rng(1000 + seed)
    perm = rng.permutation(has)
    n_te = int(round(test_frac * len(has)))
    te, pool = perm[:n_te], perm[n_te:]
    seen = pool[:min(n_seen, len(pool))]
    tr = np.concatenate([np.where(cat_df[el].values <= 0)[0], seen]).astype(int)
    sc = StandardScaler().fit(Xc[tr])
    m = lgb.LGBMRegressor(**lgb_params(seed, **TUNED)).fit(sc.transform(Xc[tr]), yc[tr])
    r = cat_metrics(np.arange(n_cat)[te], yc[te], m.predict(sc.transform(Xc[te])))
    r['n_seen_actual'] = int(len(seen))
    return r


LC_FAMS = ['Ba', 'La', 'Ti', 'Zr', 'Ce']
E2 = {}
for el in LC_FAMS:
    fam_n = int((cat_df[el].values > 0).sum())
    max_seen = fam_n - int(round(0.30 * fam_n))
    grid = [g for g in [0, 5, 10, 25, 50, 100, 200] if g < max_seen] + [max_seen]
    curve = {}
    for g in grid:
        runs = [learning_curve(el, g, s) for s in SEEDS]
        curve[str(g)] = {
            'n_seen': int(np.mean([r['n_seen_actual'] for r in runs])),
            'spearman_mean': float(np.mean([r['spearman_max'] for r in runs])),
            'spearman_std': float(np.std([r['spearman_max'] for r in runs], ddof=1)),
            'enrichment_mean': float(np.mean([r['enrichment_top10pct'] for r in runs]))}
        log(f"E2 {el:3s} n_seen={g:3d}: spearman={curve[str(g)]['spearman_mean']:.3f} "
            f"+/- {curve[str(g)]['spearman_std']:.3f}  enrich={curve[str(g)]['enrichment_mean']:.2f}x")
    full = curve[str(grid[-1])]['spearman_mean']
    zero = curve['0']['spearman_mean']
    thr = None
    for g in grid:
        if curve[str(g)]['spearman_mean'] >= 0.8 * full:
            thr = g; break
    E2[el] = {'family_size': fam_n, 'curve': curve, 'spearman_at_zero': zero,
              'spearman_at_full': full, 'gain_from_seeing_family': float(full - zero),
              'unlock_threshold_80pct': thr}
    log(f"E2 {el}: 0 seen -> {zero:.3f}, full({grid[-1]}) -> {full:.3f}, "
        f"80%-unlock at n={thr}")
RES['E2_learning_curve'] = E2

json.dump(RES, open('phase6_our_experiments.json', 'w'), indent=1)
log("wrote phase6_our_experiments.json")

print("\n============ EXPERIMENT 1: coverage-moderated prior ============")
print(f"{'fam':4s} {'n_lab':>5s} {'n_lit':>5s} {'ratio':>6s} {'top10%':>7s} {'V0':>6s} {'V1':>6s} {'delta':>7s} {'wins':>5s}")
for el in FAMILIES:
    e = E1[el]
    print(f"{el:4s} {e['n_lab_catalysts']:5d} {e['n_lit_compositions']:5d} {e['lab_lit_ratio']:6.2f}"
          f" {e['top_decile_share']:7.2f} {e['v0_spearman']:6.3f} {e['v1_spearman']:6.3f}"
          f" {e['delta']:+7.3f} {e['seed_wins']:4d}/5")
print(f"\nmean delta = {deltas.mean():+.4f}; families with positive delta = {(deltas>0).sum()}/{len(FAMILIES)}")
for k, v in CORR.items():
    print(f"  Spearman(delta, {k:20s}) = {v['spearman']:+.3f}  (p={v['p']:.3f})")
print(f"\nPRE-REGISTERED VERDICT: {'SUPPORTED' if supported else 'NOT SUPPORTED'} "
      f"(strongest |rho| = {abs(best[1]['spearman']):.3f} on {best[0]}, needed >= 0.5 with >=12 families)")

print("\n============ EXPERIMENT 2: family learning curve ============")
print("how many labeled family members must the model see before it can price that family?")
for el, e in E2.items():
    pts = ", ".join(f"{k}->{v['spearman_mean']:.3f}" for k, v in e['curve'].items())
    print(f"\n{el} (family size {e['family_size']}): {pts}")
    print(f"   never-seen {e['spearman_at_zero']:.3f} -> fully-seen {e['spearman_at_full']:.3f} "
          f"(gain {e['gain_from_seeing_family']:+.3f}); 80% unlock at n={e['unlock_threshold_80pct']}")
log("done")
