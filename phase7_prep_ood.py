"""
phase7_prep_ood.py — P3: does the model transfer ACROSS PREPARATION METHOD, honestly measured?

Replaces the row-level OOD numbers in qn_tradeoff.json (baseline 6.531 / PFT 6.772 / best raw 6.047;
the old "-45% / 3.615" was leakage). Those inherit the protocol flaw and must not be requoted.

TWO CONFOUNDS THE ORIGINAL DESIGN DID NOT CONTROL, and how this script handles them:

 1. TARGET-ESTIMATOR MISMATCH. Lab labels are max-over-~97 measurements; literature labels are
    max-over-~2.2. Per the P1 sampling simulation, max-of-2 sits several yield points below
    max-of-97 for identical chemistry, so ABSOLUTE RMSE ACROSS THAT BOUNDARY COMPARES TWO DIFFERENT
    ESTIMATORS, not two levels of skill. -> ranking metrics are primary; RMSE/bias are reported but
    explicitly flagged; a matched-n sensitivity check (lab labels recomputed as max-of-3) tests
    whether any conclusion depends on the mismatch.

 2. PREPARATION SHIFT ENTANGLED WITH SOURCE SHIFT. The lab is 100% impregnation, so any literature
    test set differs in both preparation AND provenance. -> two test sets:
        T1 = impregnation literature      (same preparation, different source) => source shift only
        T2 = non-impregnation literature  (both)                              => source + preparation
    The T2 - T1 difference is the honest cost of changing preparation.

NOTE ON prep_enc: the lab is entirely impregnation, so the preparation column is CONSTANT in
training. Trees never split on a constant, so the model is structurally blind to preparation --
the same mechanism proven for Ba in phase4. It is dropped here (numerically identical, and keeping
it would imply a capability the model does not have).

PRE-REGISTERED DECISION RULE (fixed before running): a training config "helps out-of-preparation
prediction" only if it improves Spearman on T2 by >= 0.02 with >= 4/5 seeds.

Output: phase7_prep_ood.json
"""
import warnings, json, time
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from ocm_eval import Data, lgb_params, xgb_params, cat_metrics, TARGET

t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

d = Data.load()
TUNED = json.load(open('grouped_tuning.json'))['confirmation']['tuned']['overrides']
el_all = [c for c in d.features if c != 'Temperature_C']
elements = [c for c in el_all if c != 'prep_enc']            # drop constant prep column (see header)

# ---- lab catalysts (all impregnation) ----
grp = d.dl_lab.groupby(d.groups)
lab_df = grp.agg(**{c: (c, 'first') for c in elements}, y_max=(TARGET, 'max'))
X_lab = lab_df[elements].values.astype(float); y_lab = lab_df['y_max'].values
lab_n_meas = grp.size().values
assert d.dl_lab.Preparation.nunique() == 1, "lab is expected to be single-preparation"

# ---- literature compositions, split by preparation ----
lit_id = d.dl_lit[['Preparation'] + elements].astype(str).agg('|'.join, axis=1)
lit_groups, _ = pd.factorize(lit_id)
lg = d.dl_lit.groupby(lit_groups)
lit_df = lg.agg(Preparation=('Preparation', 'first'),
                **{c: (c, 'first') for c in elements}, y_max=(TARGET, 'max'))
is_imp = (lit_df['Preparation'] == 'Impregnation').values
X_lit = lit_df[elements].values.astype(float); y_lit = lit_df['y_max'].values

# ---- leakage guard: a literature composition identical to a lab catalyst must not be a test row ----
lab_keys = set(map(tuple, np.round(X_lab, 4)))
dup = np.array([tuple(r) in lab_keys for r in np.round(X_lit, 4)])
log(f"literature compositions identical to a lab catalyst: {int(dup.sum())} (excluded from test sets)")

T1 = np.where(is_imp & ~dup)[0]          # impregnation literature  -> source shift only
T2 = np.where(~is_imp & ~dup)[0]         # non-impregnation         -> source + preparation shift
TRAIN_LIT = np.where(is_imp & ~dup)[0]   # impregnation literature usable as extra training for T2
log(f"T1 impregnation-lit = {len(T1)} | T2 non-impregnation-lit = {len(T2)} | lab = {len(y_lab)}")
assert len(set(T1) & set(T2)) == 0

def metrics(yt, yp, ids):
    m = cat_metrics(ids, yt, yp)
    m['rmse_CONFOUNDED'] = float(np.sqrt(mean_squared_error(yt, yp)))
    m['mean_signed_bias_CONFOUNDED'] = float(np.mean(yp - yt))
    return m

def run(test_idx, config, seed, y_lab_use=None):
    """Train on lab (+optional impregnation literature) and predict a literature test set."""
    yl = y_lab if y_lab_use is None else y_lab_use
    sc = StandardScaler().fit(X_lab)
    Xtr, ytr = sc.transform(X_lab), yl
    Xte = sc.transform(X_lit[test_idx])
    if config == 'C2_merge':                       # impregnation literature as extra training rows
        assert len(set(test_idx) & set(TRAIN_LIT)) == 0, "C2 would leak into this test set"
        Xtr = np.vstack([Xtr, sc.transform(X_lit[TRAIN_LIT])])
        ytr = np.concatenate([ytr, y_lit[TRAIN_LIT]])
    elif config == 'C3_prior':                     # impregnation-literature rank prior as a feature
        assert len(set(test_idx) & set(TRAIN_LIT)) == 0, "C3 would leak into this test set"
        r = rankdata(y_lit[TRAIN_LIT], method='average') / (len(TRAIN_LIT) + 1)
        pre = xgb.XGBRegressor(**xgb_params(seed)).fit(sc.transform(X_lit[TRAIN_LIT]), r)
        Xtr = np.hstack([Xtr, pre.predict(Xtr).reshape(-1, 1)])
        Xte = np.hstack([Xte, pre.predict(Xte).reshape(-1, 1)])
    m = lgb.LGBMRegressor(**lgb_params(seed, **TUNED)).fit(Xtr, ytr)
    return metrics(y_lit[test_idx], m.predict(Xte), np.arange(len(test_idx)))

SEEDS = [0, 1, 2, 7, 13]
def agg(runs):
    return {k: float(np.mean([r[k] for r in runs])) for k in runs[0]} | \
           {'spearman_std': float(np.std([r['spearman_max'] for r in runs], ddof=1)),
            'per_seed_spearman': [float(r['spearman_max']) for r in runs]}

RES = {'meta': {
    'supersedes': 'qn_tradeoff.json (row-level protocol; do not requote 6.531/6.772/6.047/3.615)',
    'protocol': 'formulation B (composition -> per-catalyst max yield), tuned LGBM, train-only scaler',
    'test_sets': {'T1_impregnation_lit': int(len(T1)), 'T2_non_impregnation_lit': int(len(T2))},
    'n_lab_train': int(len(y_lab)), 'seeds': SEEDS,
    'prep_column': 'dropped - constant in lab training data, so the model is preparation-blind by construction',
    'rmse_caveat': 'lab labels are max-of-~97 measurements, literature labels max-of-~2.2; absolute '
                   'RMSE across that boundary compares two estimators, not two skill levels',
    'decision_rule': 'a config helps if Spearman on T2 improves >= 0.02 with >= 4/5 seeds'}}

# ---- A. source shift alone (T1) vs source+preparation (T2), lab-only model ----
A = {}
for name, idx in [('T1_impregnation_lit', T1), ('T2_non_impregnation_lit', T2)]:
    A[name] = agg([run(idx, 'C1_lab_only', s) for s in SEEDS])
    log(f"A {name:26s} spearman={A[name]['spearman_max']:.3f}  enrich={A[name]['enrichment_top10pct']:.2f}x "
        f"prec@20={A[name]['precision_at20_vs_top10pct']:.2f}  [rmse {A[name]['rmse_CONFOUNDED']:.2f}, "
        f"bias {A[name]['mean_signed_bias_CONFOUNDED']:+.2f}]")
RES['A_shift_decomposition'] = A

# ---- B. can literature help out-of-preparation prediction? (T2 only) ----
B = {}
for cfg in ['C1_lab_only', 'C2_merge', 'C3_prior']:
    B[cfg] = agg([run(T2, cfg, s) for s in SEEDS])
    log(f"B T2 {cfg:12s} spearman={B[cfg]['spearman_max']:.3f} +/- {B[cfg]['spearman_std']:.3f}  "
        f"enrich={B[cfg]['enrichment_top10pct']:.2f}x")
base = B['C1_lab_only']
for cfg in ['C2_merge', 'C3_prior']:
    wins = sum(a > b for a, b in zip(B[cfg]['per_seed_spearman'], base['per_seed_spearman']))
    B[cfg]['delta_spearman'] = float(B[cfg]['spearman_max'] - base['spearman_max'])
    B[cfg]['seed_wins'] = int(wins)
    B[cfg]['HELPS'] = bool(B[cfg]['delta_spearman'] >= 0.02 and wins >= 4)
RES['B_can_literature_help'] = B

# ---- C. sensitivity: match the label estimator (lab labels = max of ~3 measurements) ----
C = {}
raw = {g: v[TARGET].values for g, v in d.dl_lab.groupby(d.groups)}
for s in SEEDS:
    rng = np.random.default_rng(500 + s)
    y_matched = np.array([rng.choice(raw[g], size=min(3, len(raw[g])), replace=False).max()
                          for g in range(len(y_lab))])
    C.setdefault('runs', []).append(run(T2, 'C1_lab_only', s, y_lab_use=y_matched))
C = agg(C['runs'])
RES['C_matched_estimator_sensitivity'] = C
log(f"C matched-n (lab labels = max of 3): spearman={C['spearman_max']:.3f} "
    f"(vs {base['spearman_max']:.3f} with max-of-97 labels)")

json.dump(RES, open('phase7_prep_ood.json', 'w'), indent=1)
log("wrote phase7_prep_ood.json")

print("\n================ SUMMARY ================")
print("A. Decomposing the shift (lab-only model, ranking metrics are primary)")
for k, v in A.items():
    print(f"   {k:26s} spearman {v['spearman_max']:.3f}   enrich {v['enrichment_top10pct']:.2f}x   "
          f"prec@20 {v['precision_at20_vs_top10pct']:.2f}")
dt = A['T2_non_impregnation_lit']['spearman_max'] - A['T1_impregnation_lit']['spearman_max']
print(f"   -> cost of ALSO changing preparation: {dt:+.3f} Spearman")
print("\nB. Can impregnation literature help predict OTHER preparations? (T2)")
for cfg, v in B.items():
    extra = "" if cfg == 'C1_lab_only' else f"  delta {v['delta_spearman']:+.3f}  wins {v['seed_wins']}/5  HELPS={v['HELPS']}"
    print(f"   {cfg:12s} spearman {v['spearman_max']:.3f} +/- {v['spearman_std']:.3f}{extra}")
helpers = [c for c in ('C2_merge', 'C3_prior') if B[c]['HELPS']]
print(f"   PRE-REGISTERED VERDICT: {helpers if helpers else 'NO config helps'}")
print(f"\nC. Matched-estimator sensitivity: spearman {C['spearman_max']:.3f} vs {base['spearman_max']:.3f} "
      f"({C['spearman_max']-base['spearman_max']:+.3f}) -> conclusions "
      f"{'UNCHANGED' if abs(C['spearman_max']-base['spearman_max']) < 0.05 else 'SHIFTED - report prominently'}")
print("\nConfounded-metric disclosure (do not quote as skill):")
for k, v in A.items():
    print(f"   {k:26s} RMSE {v['rmse_CONFOUNDED']:.2f}  mean bias {v['mean_signed_bias_CONFOUNDED']:+.2f}")
log("done")
