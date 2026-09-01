"""
phase3_lit_prior.py — Phase 3: can the literature improve the adopted catalyst-level screening
model (formulation B) for genuinely unseen catalysts, with the leakage channel closed?

Lessons baked in from the Taniike validation round: the literature expert contains NO lab rows
(identity-leakage lesson), uses RANK labels (QN-ablation lesson: trees only consume ordering), and
one variant lets the model learn WHEN to trust the prior (Ce family holdout showed literature can
actively hurt).

Variants (identical catalyst-fold assignments per seed, tuned LGBM params, per-fold train-only
scaler; catalyst-grouped 5-fold CV, seeds 0,1,2,7,13):

  V0  control — formulation B (prep + 65 elements -> per-catalyst max yield). Must reproduce
      Phase-2 B_direct_tuned (0.760 / 4.28x / 0.47) on identical folds.
  V1  literature rank prior — XGB expert on ~unique literature COMPOSITIONS only (rank of
      per-composition max yield); its prediction appended as one feature.
  V2  similarity features only — nearest-literature distance + mean 5-NN distance (scaled element
      space), no prior. Does "position relative to literature coverage" itself carry signal?
  V3  gated prior — V1 + V2 features together (trees can interact prior x proximity).
  V4  catalyst-level direct-merge control — literature compositions as extra TRAINING rows, labels
      rank-mapped onto the lab-train max-yield scale. Expected not to help; kept as a control.

Pre-registered decision rule: a variant wins only if it beats V0 on Spearman by >= 0.01 AND on
enrichment, consistently across the 5 seeds. If a winner exists, Ba and Ce family holdouts are run
for winner vs V0. Output: phase3_lit_prior.json
"""
import warnings, time, json
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import rankdata
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from ocm_eval import Data, lgb_params, xgb_params, quantile_normalize_y, cat_metrics, TARGET

t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

d = Data.load()
TUNED = json.load(open('grouped_tuning.json'))['confirmation']['tuned']['overrides']

# ---- catalyst-level tables (identical construction to catalyst_level.py) ----
el_cols = [c for c in d.features if c != 'Temperature_C']                     # prep_enc + 65 elements
cat_df = d.dl_lab.groupby(d.groups).agg(**{c: (c, 'first') for c in el_cols}, y_max=(TARGET, 'max'))
Xc = cat_df[el_cols].values.astype(float); yc = cat_df['y_max'].values        # 917 lab catalysts

elements = [c for c in el_cols if c != 'prep_enc']
lit_id = d.dl_lit[['Preparation'] + elements].astype(str).agg('|'.join, axis=1)
lit_groups, lit_uniq = pd.factorize(lit_id)
litc_df = d.dl_lit.groupby(lit_groups).agg(**{c: (c, 'first') for c in el_cols}, y_max=(TARGET, 'max'))
Xlc = litc_df[el_cols].values.astype(float); ylc = litc_df['y_max'].values    # unique lit compositions
ylc_rank = rankdata(ylc, method='average') / (len(ylc) + 1)
log(f"lab catalysts={len(yc)}  unique literature compositions={len(ylc)}")

FEATNAMES = {'V0': el_cols,
             'V1': el_cols + ['lit_rank_prior'],
             'V2': el_cols + ['nn1_lit_dist', 'knn5_lit_dist'],
             'V3': el_cols + ['lit_rank_prior', 'nn1_lit_dist', 'knn5_lit_dist'],
             'V4': el_cols}


def fold_assignment(seed, n_folds=5):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(d.n_cat)
    fog = np.empty(d.n_cat, dtype=int)
    for k, chunk in enumerate(np.array_split(perm, n_folds)):
        fog[chunk] = k
    return fog


def eval_variant(variant, seed, collect_imp=None):
    fog = fold_assignment(seed)
    yp = np.empty(d.n_cat)
    for k in range(5):
        tr, va = np.where(fog != k)[0], np.where(fog == k)[0]
        sc = StandardScaler().fit(Xc[tr])
        Xc_sc = sc.transform(Xc); Xlc_sc = sc.transform(Xlc)
        cols = [Xc_sc]
        if variant in ('V1', 'V3'):    # literature-only rank expert (no lab rows: leakage lesson)
            pre = xgb.XGBRegressor(**xgb_params(seed)).fit(Xlc_sc, ylc_rank)
            cols.append(pre.predict(Xc_sc).reshape(-1, 1))
        if variant in ('V2', 'V3'):    # proximity to literature coverage
            D = cdist(Xc_sc, Xlc_sc)
            nn = np.sort(D, axis=1)
            cols.append(nn[:, :1])                                   # nearest lit composition
            cols.append(nn[:, :5].mean(axis=1, keepdims=True))       # mean 5-NN distance
        Xv = np.hstack(cols) if len(cols) > 1 else Xc_sc
        if variant == 'V4':            # catalyst-level direct-merge control
            Xtr = np.vstack([Xc_sc[tr], Xlc_sc])
            ytr = np.concatenate([yc[tr], quantile_normalize_y(ylc, yc[tr])])
        else:
            Xtr, ytr = Xv[tr], yc[tr]
        m = lgb.LGBMRegressor(**lgb_params(seed, **TUNED)).fit(Xtr, ytr)
        yp[va] = m.predict(Xc_sc[va] if variant == 'V4' else Xv[va])
        if collect_imp is not None and variant == 'V3':
            gain = m.booster_.feature_importance(importance_type='gain')
            collect_imp.append(gain / gain.sum())
    return cat_metrics(np.arange(d.n_cat), yc, yp)


SEEDS = [0, 1, 2, 7, 13]
RES, imp_acc = {}, []
for variant in ['V0', 'V1', 'V2', 'V3', 'V4']:
    runs = []
    for s in SEEDS:
        r = eval_variant(variant, s, collect_imp=imp_acc if variant == 'V3' else None)
        runs.append(r)
        log(f"{variant} seed={s:2d}  spearman={r['spearman_max']:.3f}  "
            f"enrich={r['enrichment_top10pct']:.2f}x  prec@20={r['precision_at20_vs_top10pct']:.2f}")
    RES[variant] = {'per_seed': runs,
                    'spearman_mean': float(np.mean([r['spearman_max'] for r in runs])),
                    'spearman_std': float(np.std([r['spearman_max'] for r in runs], ddof=1)),
                    'enrichment_mean': float(np.mean([r['enrichment_top10pct'] for r in runs])),
                    'precision_at20_mean': float(np.mean([r['precision_at20_vs_top10pct'] for r in runs]))}

# V3 feature-importance readout (normalized gain, averaged over folds x seeds)
imp = np.mean(imp_acc, axis=0)
order = np.argsort(-imp)
V3_TOP = [(FEATNAMES['V3'][j], float(imp[j])) for j in order[:10]]
prior_share = float(imp[FEATNAMES['V3'].index('lit_rank_prior')])
dist_share = float(imp[FEATNAMES['V3'].index('nn1_lit_dist')] + imp[FEATNAMES['V3'].index('knn5_lit_dist')])

# ---- pre-registered decision rule ----
v0 = RES['V0']
winners = []
for v in ['V1', 'V2', 'V3', 'V4']:
    dsp = RES[v]['spearman_mean'] - v0['spearman_mean']
    den = RES[v]['enrichment_mean'] - v0['enrichment_mean']
    per_seed_wins = sum(RES[v]['per_seed'][i]['spearman_max'] > v0['per_seed'][i]['spearman_max']
                        for i in range(len(SEEDS)))
    RES[v]['delta_spearman_vs_V0'] = float(dsp); RES[v]['delta_enrich_vs_V0'] = float(den)
    RES[v]['per_seed_spearman_wins_vs_V0'] = int(per_seed_wins)
    if dsp >= 0.01 and den > 0 and per_seed_wins >= 4:
        winners.append(v)

# ---- family holdout for winner (or for V3 as the most instructive variant if no winner) ----
def family_holdout(variant, element):
    has = cat_df[element].values > 0
    te, tr = np.where(has)[0], np.where(~has)[0]
    lit_keep = litc_df[element].values <= 0            # exclude element-containing literature: clean claim
    sc = StandardScaler().fit(Xc[tr])
    Xc_sc = sc.transform(Xc); Xlc_sc = sc.transform(Xlc[lit_keep])
    cols = [Xc_sc]
    if variant in ('V1', 'V3'):
        pre = xgb.XGBRegressor(**xgb_params(42)).fit(Xlc_sc, ylc_rank[lit_keep])
        cols.append(pre.predict(Xc_sc).reshape(-1, 1))
    if variant in ('V2', 'V3'):
        D = cdist(Xc_sc, Xlc_sc); nn = np.sort(D, axis=1)
        cols.append(nn[:, :1]); cols.append(nn[:, :5].mean(axis=1, keepdims=True))
    Xv = np.hstack(cols) if len(cols) > 1 else Xc_sc
    m = lgb.LGBMRegressor(**lgb_params(42, **TUNED)).fit(Xv[tr], yc[tr])
    return cat_metrics(np.arange(d.n_cat)[te], yc[te], m.predict(Xv[te]))

probe = winners[0] if winners else 'V3'
HOLD = {}
for el in ['Ba', 'Ce']:
    HOLD[el] = {'V0': family_holdout('V0', el), probe: family_holdout(probe, el)}
    log(f"holdout {el}: V0 spearman={HOLD[el]['V0']['spearman_max']:.3f} "
        f"{probe} spearman={HOLD[el][probe]['spearman_max']:.3f}")

json.dump({'meta': {'protocol': 'catalyst-grouped 5-fold CV, formulation B, identical folds, tuned LGBM',
                    'seeds': SEEDS, 'decision_rule': 'delta spearman >= 0.01 AND delta enrich > 0 AND >=4/5 seed wins',
                    'winners': winners, 'holdout_probe': probe},
           'results': RES,
           'v3_importances_top10': V3_TOP,
           'v3_prior_gain_share': prior_share, 'v3_distance_gain_share': dist_share,
           'family_holdout_lit_excl': HOLD},
          open('phase3_lit_prior.json', 'w'), indent=1)
log("wrote phase3_lit_prior.json")

print("\n================= SUMMARY (grouped, formulation B, 5 seeds) =================")
for v in ['V0', 'V1', 'V2', 'V3', 'V4']:
    r = RES[v]
    extra = (f"  d_spear={r.get('delta_spearman_vs_V0', 0):+.3f} d_enr={r.get('delta_enrich_vs_V0', 0):+.2f}"
             f" wins={r.get('per_seed_spearman_wins_vs_V0', '-')}/5" if v != 'V0' else '')
    print(f"{v}: spearman {r['spearman_mean']:.3f} +/- {r['spearman_std']:.3f}"
          f"  enrich {r['enrichment_mean']:.2f}x  prec@20 {r['precision_at20_mean']:.2f}{extra}")
print(f"\nwinners per decision rule: {winners if winners else 'NONE'}")
print(f"V3 gain shares: prior {prior_share:.3f}, distance feats {dist_share:.3f}")
print("V3 top-10 features by gain:", [f'{n}:{v:.3f}' for n, v in V3_TOP])
log("done")
