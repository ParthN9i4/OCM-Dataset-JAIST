"""
phase11_condition_baseline.py — what do the recovered gas-flow conditions actually buy?

BACKGROUND. Prof. Taniike supplied the original lab dataset (Sept 2026) with the reaction conditions
restored. Every catalyst was run at 5 temperatures x 27 gas-flow settings = 135 conditions; rows with
poor mass balance were excluded, so cells hold at most 27. The export we had been working from
dropped those columns to make the lab data compatible with the literature set.

Two questions, which are NOT the same and can have different answers:

  A. Does row-level prediction improve? Previously, rows sharing a (composition, temperature) were
     indistinguishable to the model, so 19.9% of yield variance sat within those groups and row-level
     RMSE could not go below 1.757. This part measures what a real model gains from the conditions.

     A WARNING ABOUT THE TWO "FLOORS", because they are not the same kind of quantity and must not
     be compared as though they were. The 1.757 figure groups by (catalyst, temperature) -- 4,399
     groups averaging 20.2 rows -- so it means "perfectly predict each cell's mean", a real and
     learnable quantity. Adding the flow columns splits the data into 87,168 groups of which 85,262
     (97.8%) hold exactly ONE row. A singleton's within-variance is zero by construction, not
     because a model could predict it, so the resulting 0.595 is the error of a model that has
     memorised every distinct condition combination. It is an interpolation bound, not a
     generalisation target, and no model should be expected to approach it. Measured: even with the
     catalyst present in training (row-level CV) the conditions arm reaches 1.706, not 0.595.

  B. Does CATALYST SELECTION improve? This is the screening objective and the one that matters for a
     synthesis campaign. A condition-aware model predicts every condition and takes the max of its
     PREDICTIONS as the catalyst's estimate, instead of training on a collapsed label. Better row
     accuracy does not automatically mean better ranking -- the ranking depends on between-catalyst
     differences, which were never the thing the conditions explained.

PROTOCOL. Catalyst-grouped CV throughout (all rows of a catalyst in one fold), 5 seeds, tuned LGBM.
Ground truth for screening stays the catalyst's TRUE observed maximum, as everywhere else in this
project. Only the feature set changes between arms, so the comparison isolates the conditions.

Output: phase11_condition_baseline.json
"""
import warnings, json, time
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from ocm_eval import load_with_conditions, lgb_params, cat_metrics, TARGET, COND_COLS

t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

df, feat_all, groups, n_cat = load_with_conditions()
TUNED = json.load(open('grouped_tuning.json'))['confirmation']['tuned']['overrides']
y = df[TARGET].values
SEEDS = [0, 1, 2, 7, 13]

feat_nocond = [c for c in feat_all if c not in COND_COLS]   # composition + temperature (as before)
log(f"{len(df):,} rows, {n_cat} catalysts, {len(feat_all)} features "
    f"({len(feat_nocond)} without conditions)")

# ---------------------------------------------------------------- variance decomposition
def within_share(keys):
    g = pd.Series(y).groupby(keys)
    n, v = g.size(), g.var()
    wss = ((n - 1) * v).fillna(0).sum()
    tss = ((y - y.mean()) ** 2).sum()
    return 100 * wss / tss, float(np.sqrt(wss / len(y)))

k_comp = pd.Series(list(zip(groups, df.Temperature_C)))
k_cond = pd.Series(list(zip(groups, df.Temperature_C, df.Ar_flow, df.CH4_flow, df.O2_flow)))
w_old, floor_old = within_share(k_comp)
w_new, floor_new = within_share(k_cond)
sz_old, sz_new = k_comp.value_counts(), k_cond.value_counts()
DECOMP = {'within_pct_composition_temperature': w_old, 'rmse_floor_composition_temperature': floor_old,
          'within_pct_with_conditions': w_new, 'rmse_floor_with_conditions': floor_new,
          'variance_attributable_to_conditions_pct_points': w_old - w_new,
          'share_of_within_cell_variance_attributable': (w_old - w_new) / w_old,
          'n_groups_composition_temperature': int(len(sz_old)),
          'mean_rows_per_group_composition_temperature': float(sz_old.mean()),
          'n_groups_with_conditions': int(len(sz_new)),
          'singleton_groups_with_conditions': int((sz_new == 1).sum()),
          'singleton_share_with_conditions': float((sz_new == 1).mean()),
          'n_replicate_rows': int(sz_new[sz_new >= 2].sum()),
          'CAVEAT': ('the two floors are not comparable. 1.757 = predict each cell mean (4,399 groups, '
                     'mean 20.2 rows) and is a learnable target. 0.595 = predict each condition '
                     'exactly, but 97.8% of those groups are singletons whose within-variance is zero '
                     'by construction, so it is an interpolation bound and unreachable in practice. '
                     'Report the measured arm-vs-arm improvement, not the distance to 0.595.')}
log(f"floor without conditions {floor_old:.3f} ({w_old:.1f}% within) -> "
    f"with conditions {floor_new:.3f} ({w_new:.1f}% within)")

# ---------------------------------------------------------------- grouped CV
def folds(seed, k=5):
    r = np.random.default_rng(seed); perm = r.permutation(n_cat); f = np.empty(n_cat, int)
    for i, ch in enumerate(np.array_split(perm, k)): f[ch] = i
    return f[groups]


TRUE_MAX = pd.Series(y).groupby(groups).max().reindex(range(n_cat)).values


def run_arm(feats, seed):
    """Row-level model. Returns (row predictions, per-catalyst max-of-predictions)."""
    X = df[feats].values.astype(float)
    f = folds(seed); yp = np.empty(len(df))
    for k in range(5):
        tr, va = np.where(f != k)[0], np.where(f == k)[0]
        sc = StandardScaler().fit(X[tr])
        m = lgb.LGBMRegressor(**lgb_params(seed, **TUNED)).fit(sc.transform(X[tr]), y[tr])
        yp[va] = m.predict(sc.transform(X[va]))
    cat_pred = pd.Series(yp).groupby(groups).max().reindex(range(n_cat)).values
    return yp, cat_pred


RES = {'meta': {'protocol': 'catalyst-grouped 5-fold CV, tuned LGBM, 5 seeds',
                'seeds': SEEDS, 'n_rows': int(len(df)), 'n_catalysts': int(n_cat),
                'screening_ground_truth': 'catalyst true observed maximum'},
       'variance_decomposition': DECOMP, 'arms': {}}

for name, feats in [('composition_temperature', feat_nocond), ('plus_conditions', feat_all)]:
    rmses, sps, ens, p20s = [], [], [], []
    for s in SEEDS:
        yp, cp = run_arm(feats, s)
        rmses.append(float(np.sqrt(np.mean((y - yp) ** 2))))
        m = cat_metrics(np.arange(n_cat), TRUE_MAX, cp)
        sps.append(m['spearman_max']); ens.append(m['enrichment_top10pct'])
        p20s.append(m['precision_at20_vs_top10pct'])
    RES['arms'][name] = {
        'n_features': len(feats),
        'row_rmse_mean': float(np.mean(rmses)), 'row_rmse_std': float(np.std(rmses, ddof=1)),
        'screening_spearman_mean': float(np.mean(sps)),
        'screening_spearman_std': float(np.std(sps, ddof=1)),
        'screening_enrichment_mean': float(np.mean(ens)),
        'screening_precision_at20_mean': float(np.mean(p20s))}
    r = RES['arms'][name]
    log(f"{name:26s} row RMSE {r['row_rmse_mean']:.3f}  |  screening rho "
        f"{r['screening_spearman_mean']:.4f}  enrich {r['screening_enrichment_mean']:.2f}x")

a, b = RES['arms']['composition_temperature'], RES['arms']['plus_conditions']
RES['comparison'] = {
    'row_rmse_improvement': a['row_rmse_mean'] - b['row_rmse_mean'],
    'row_rmse_pct_improvement': 100 * (a['row_rmse_mean'] - b['row_rmse_mean']) / a['row_rmse_mean'],
    'headroom_without_conditions': a['row_rmse_mean'] - floor_old,
    'headroom_with_conditions': b['row_rmse_mean'] - floor_new,
    'screening_spearman_delta': b['screening_spearman_mean'] - a['screening_spearman_mean'],
    'screening_enrichment_delta': b['screening_enrichment_mean'] - a['screening_enrichment_mean']}

json.dump(RES, open('phase11_condition_baseline.json', 'w'), indent=1)
log("wrote phase11_condition_baseline.json")

print("\n==================== WHAT THE CONDITIONS BUY ====================")
print(f"{'':28s} {'row RMSE':>10s} {'vs floor':>10s} {'screen rho':>11s} {'enrich':>8s}")
print(f"{'composition + temperature':28s} {a['row_rmse_mean']:10.3f} {a['row_rmse_mean']-floor_old:10.3f} "
      f"{a['screening_spearman_mean']:11.4f} {a['screening_enrichment_mean']:7.2f}x")
print(f"{'+ gas-flow conditions':28s} {b['row_rmse_mean']:10.3f} {b['row_rmse_mean']-floor_new:10.3f} "
      f"{b['screening_spearman_mean']:11.4f} {b['screening_enrichment_mean']:7.2f}x")
print(f"\nrow-level RMSE improves by {RES['comparison']['row_rmse_pct_improvement']:.1f}% "
      f"({a['row_rmse_mean']:.3f} -> {b['row_rmse_mean']:.3f}) -- this is the real gain.")
print(f"NOT comparable: the {floor_old:.3f} floor means 'predict each cell mean' (groups average "
      f"{DECOMP['mean_rows_per_group_composition_temperature']:.1f} rows);")
print(f"the {floor_new:.3f} figure means 'predict each condition exactly', and "
      f"{100*DECOMP['singleton_share_with_conditions']:.1f}% of those groups hold one row, so it is")
print("an interpolation bound. Do not quote the model as being 'far from' it.")
print(f"catalyst SCREENING rho {a['screening_spearman_mean']:.4f} -> {b['screening_spearman_mean']:.4f} "
      f"({RES['comparison']['screening_spearman_delta']:+.4f}), "
      f"enrichment {a['screening_enrichment_mean']:.2f}x -> {b['screening_enrichment_mean']:.2f}x "
      f"({RES['comparison']['screening_enrichment_delta']:+.2f})")
log("done")
