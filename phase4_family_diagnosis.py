"""
phase4_family_diagnosis.py — Phase 4: why does the Ba family fail under holdout, and is the
Phase-3 Ce anecdote (gated prior helping) a real, seed-stable effect?

  A. Family anatomy: size, max-yield distribution, share of the GLOBAL top-10% (92 catalysts),
     top co-occurring elements, literature coverage — for {Ba, La, Ti, Zr, Ce, Sr, Mn}.
  B. Ba error anatomy: signed errors + calibration slope on the Ba holdout; mechanism check —
     retrain with the Ba column dropped; identical predictions prove the model ignores Ba loading
     (constant 0 in training) and prices Ba catalysts as if Ba were absent.
  C. Size control: 10 random pseudo-families of 291 catalysts. If they score ~grouped-CV level,
     Ba's failure is chemical/label-coverage, not size.
  D. Ce effect, seeded: {Ba, La, Ti, Zr, Ce} x {V0, V1 lit-rank-prior, V3 gated} x 5 model seeds
     (family split fixed; element-containing literature excluded for V1/V3, as in Phase 3).
  E. Synthesis table.

Protocol: formulation B (per-catalyst max yield), tuned LGBM, per-holdout train-only scaler.
Output: phase4_family_diagnosis.json
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
from ocm_eval import Data, lgb_params, xgb_params, cat_metrics, TARGET

t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

d = Data.load()
TUNED = json.load(open('grouped_tuning.json'))['confirmation']['tuned']['overrides']

el_cols = [c for c in d.features if c != 'Temperature_C']
cat_df = d.dl_lab.groupby(d.groups).agg(**{c: (c, 'first') for c in el_cols}, y_max=(TARGET, 'max'))
Xc = cat_df[el_cols].values.astype(float); yc = cat_df['y_max'].values
elements = [c for c in el_cols if c != 'prep_enc']
lit_id = d.dl_lit[['Preparation'] + elements].astype(str).agg('|'.join, axis=1)
lit_groups, _ = pd.factorize(lit_id)
litc_df = d.dl_lit.groupby(lit_groups).agg(**{c: (c, 'first') for c in el_cols}, y_max=(TARGET, 'max'))
Xlc = litc_df[el_cols].values.astype(float); ylc = litc_df['y_max'].values
ylc_rank = rankdata(ylc, method='average') / (len(ylc) + 1)
n_cat = len(yc)
TOP92 = set(np.argsort(-yc)[:int(round(0.10 * n_cat))])
log(f"lab catalysts={n_cat}, lit compositions={len(ylc)}, global top-10% = {len(TOP92)}")

FAMS = ['Ba', 'La', 'Ti', 'Zr', 'Ce', 'Sr', 'Mn']

# ---------------------------------------------------------------- A. family anatomy
A = {}
for el in FAMS:
    m = cat_df[el].values > 0
    fam_idx = set(np.where(m)[0])
    co = {e: float((cat_df.loc[m, e].values > 0).mean()) for e in elements if e != el}
    top_co = sorted(co.items(), key=lambda kv: -kv[1])[:8]
    lit_m = litc_df[el].values > 0
    A[el] = {'n_lab_catalysts': int(m.sum()),
             'family_ymax_mean': float(yc[m].mean()), 'family_ymax_median': float(np.median(yc[m])),
             'rest_ymax_mean': float(yc[~m].mean()),
             'n_family_in_global_top10pct': int(len(fam_idx & TOP92)),
             'share_of_global_top10pct': float(len(fam_idx & TOP92) / len(TOP92)),
             'top_cooccurring_elements': top_co,
             'n_lit_compositions_with_element': int(lit_m.sum()),
             'lit_family_ymax_mean': float(ylc[lit_m].mean()) if lit_m.any() else None}
    log(f"A {el:2s}: n={m.sum():3d}  fam_mean={yc[m].mean():5.2f}  rest_mean={yc[~m].mean():5.2f}  "
        f"share_of_top10%={A[el]['share_of_global_top10pct']:.2f}  lit_n={lit_m.sum()}")

# ---------------------------------------------------------------- B. Ba error anatomy + mechanism
mBa = cat_df['Ba'].values > 0
te, tr = np.where(mBa)[0], np.where(~mBa)[0]
sc = StandardScaler().fit(Xc[tr])
mdl = lgb.LGBMRegressor(**lgb_params(42, **TUNED)).fit(sc.transform(Xc[tr]), yc[tr])
pred = mdl.predict(sc.transform(Xc[te]))
err = pred - yc[te]                                             # signed: + = over-prediction
fam_top = yc[te] >= np.quantile(yc[te], 0.9)
slope = float(np.polyfit(pred, yc[te], 1)[0])
# mechanism check: drop the Ba column entirely
keep = [i for i, c in enumerate(el_cols) if c != 'Ba']
sc2 = StandardScaler().fit(Xc[tr][:, keep])
mdl2 = lgb.LGBMRegressor(**lgb_params(42, **TUNED)).fit(sc2.transform(Xc[tr][:, keep]), yc[tr])
pred2 = mdl2.predict(sc2.transform(Xc[te][:, keep]))
B = {'mean_signed_error_all': float(err.mean()),
     'mean_signed_error_family_top_decile': float(err[fam_top].mean()),
     'mean_signed_error_family_rest': float(err[~fam_top].mean()),
     'calibration_slope_obs_on_pred': slope,
     'pred_range': [float(pred.min()), float(pred.max())],
     'obs_range': [float(yc[te].min()), float(yc[te].max())],
     'mechanism_check_max_abs_pred_diff_dropBa': float(np.abs(pred - pred2).max()),
     'mechanism_check_corr': float(np.corrcoef(pred, pred2)[0, 1])}
log(f"B Ba: signed err all={err.mean():+.2f}, top-decile={err[fam_top].mean():+.2f}, "
    f"slope={slope:.2f}, drop-Ba max pred diff={B['mechanism_check_max_abs_pred_diff_dropBa']:.4f}")

# ---------------------------------------------------------------- C. random pseudo-family control
C_runs = []
for s in range(10):
    rng = np.random.default_rng(100 + s)
    fake = rng.choice(n_cat, size=int(mBa.sum()), replace=False)
    mask = np.zeros(n_cat, bool); mask[fake] = True
    trf, tef = np.where(~mask)[0], np.where(mask)[0]
    scf = StandardScaler().fit(Xc[trf])
    m = lgb.LGBMRegressor(**lgb_params(42, **TUNED)).fit(scf.transform(Xc[trf]), yc[trf])
    r = cat_metrics(np.arange(n_cat)[tef], yc[tef], m.predict(scf.transform(Xc[tef])))
    C_runs.append(r)
C = {'spearman_mean': float(np.mean([r['spearman_max'] for r in C_runs])),
     'spearman_std': float(np.std([r['spearman_max'] for r in C_runs], ddof=1)),
     'enrichment_mean': float(np.mean([r['enrichment_top10pct'] for r in C_runs])),
     'per_run': C_runs}
log(f"C random-291 holdouts: spearman={C['spearman_mean']:.3f} +/- {C['spearman_std']:.3f} "
    f"enrich={C['enrichment_mean']:.2f}x")

# ---------------------------------------------------------------- D. seeded family x variant
def run_family(el, variant, seed):
    has = cat_df[el].values > 0
    tef, trf = np.where(has)[0], np.where(~has)[0]
    lit_keep = litc_df[el].values <= 0
    scf = StandardScaler().fit(Xc[trf])
    Xc_sc = scf.transform(Xc); Xlc_sc = scf.transform(Xlc[lit_keep])
    cols = [Xc_sc]
    if variant in ('V1', 'V3'):
        pre = xgb.XGBRegressor(**xgb_params(seed)).fit(Xlc_sc, ylc_rank[lit_keep])
        cols.append(pre.predict(Xc_sc).reshape(-1, 1))
    if variant == 'V3':
        D_ = cdist(Xc_sc, Xlc_sc); nn = np.sort(D_, axis=1)
        cols.append(nn[:, :1]); cols.append(nn[:, :5].mean(axis=1, keepdims=True))
    Xv = np.hstack(cols) if len(cols) > 1 else Xc_sc
    m = lgb.LGBMRegressor(**lgb_params(seed, **TUNED)).fit(Xv[trf], yc[trf])
    return cat_metrics(np.arange(n_cat)[tef], yc[tef], m.predict(Xv[tef]))

SEEDS = [0, 1, 2, 7, 13]
D_res = {}
for el in ['Ba', 'La', 'Ti', 'Zr', 'Ce']:
    for variant in ['V0', 'V1', 'V3']:
        runs = [run_family(el, variant, s) for s in SEEDS]
        D_res[f'{el}/{variant}'] = {
            'spearman_mean': float(np.mean([r['spearman_max'] for r in runs])),
            'spearman_std': float(np.std([r['spearman_max'] for r in runs], ddof=1)),
            'enrichment_mean': float(np.mean([r['enrichment_top10pct'] for r in runs])),
            'per_seed_spearman': [float(r['spearman_max']) for r in runs]}
        log(f"D {el:2s}/{variant}: spearman={D_res[f'{el}/{variant}']['spearman_mean']:.3f}"
            f" +/- {D_res[f'{el}/{variant}']['spearman_std']:.3f}"
            f"  enrich={D_res[f'{el}/{variant}']['enrichment_mean']:.2f}x")

json.dump({'meta': {'protocol': 'formulation B family holdout, tuned LGBM, lit-excluded for V1/V3',
                    'seeds': SEEDS},
           'A_family_anatomy': A, 'B_ba_error_anatomy': B, 'C_random_family_control': C,
           'D_seeded_family_variant': D_res}, open('phase4_family_diagnosis.json', 'w'), indent=1)
log("wrote phase4_family_diagnosis.json")

print("\n================= SYNTHESIS =================")
print(f"{'family':7s} {'n':>4s} {'top10%share':>11s} {'lit_n':>6s} {'V0':>14s} {'V1 d':>8s} {'V3 d':>8s}")
for el in ['Ba', 'La', 'Ti', 'Zr', 'Ce']:
    v0 = D_res[f'{el}/V0']; v1 = D_res[f'{el}/V1']; v3 = D_res[f'{el}/V3']
    print(f"{el:7s} {A[el]['n_lab_catalysts']:4d} {A[el]['share_of_global_top10pct']:11.2f}"
          f" {A[el]['n_lit_compositions_with_element']:6d}"
          f" {v0['spearman_mean']:.3f}+/-{v0['spearman_std']:.3f}"
          f" {v1['spearman_mean']-v0['spearman_mean']:+8.3f} {v3['spearman_mean']-v0['spearman_mean']:+8.3f}")
print(f"\nrandom-291 control: {C['spearman_mean']:.3f} +/- {C['spearman_std']:.3f} (vs Ba {D_res['Ba/V0']['spearman_mean']:.3f})")
print(f"Ba mechanism: drop-Ba-column max |pred diff| = {B['mechanism_check_max_abs_pred_diff_dropBa']:.4f} "
      f"(0 = model provably ignores Ba loading)")
print(f"Ba signed error, family top decile: {B['mean_signed_error_family_top_decile']:+.2f} "
      f"(negative = underprediction of the best Ba catalysts)")
log("done")
