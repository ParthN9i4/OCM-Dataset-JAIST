"""
phase6_candidates.py — ranked list of unseen candidate catalysts, with uncertainty and
coverage gating, for prospective experimental validation.

DESIGN GRAMMAR (derived from the lab's own 917 catalysts, not invented):
  - every lab catalyst is Impregnation
  - structure = 1 support at ~90% + 2-3 promoters at ~3.33% each (loadings sum to 100)
  - supports actually used: Ba, Ti, La, Zr, Ca, Mg, Al, Si, Ce
  - promoters: elements appearing as a non-major component (W, Cs, Mo, Li, Zn, Pd, Fe, K, ...)
We enumerate the same grammar and remove compositions the lab has already made.

MODEL: formulation B (composition -> per-catalyst max yield), tuned LGBM, 10-seed ensemble trained
on all 917 catalysts. No literature prior: Phase 3's null and the 28-family test in
phase6_our_experiments.py both closed that option.

COVERAGE GATING (the Phase-4 Ba lesson): a model cannot price an element it has never seen with
non-zero loading -- the column is constant, trees never split on it, and predictions are identical
to deleting the column. Every candidate therefore carries:
  - min_element_support : the rarest of its elements, counted over the 917 training catalysts
  - nn_distance        : distance to the nearest training catalyst in scaled element space
  - ensemble_std       : seed-to-seed disagreement
  - tier               : IN_SUPPORT (ranks trustworthy) or EXTRAPOLATIVE (flagged, rank unreliable)
The unpriceable flag is verified by an explicit drop-column test.

HONEST EXPECTATION: retrospective grouped-CV precision@20 with bootstrap CI, NOT a point estimate.
The model also regresses to the mean at the extremes (training max 21.2%, predictions cap ~14.5%),
so the RANKING is the product here, not the absolute predicted yield.

Output: phase6_candidates.json + phase6_candidates.csv
"""
import warnings, time, json, itertools
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from ocm_eval import Data, lgb_params, TARGET

t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

d = Data.load()
TUNED = json.load(open('grouped_tuning.json'))['confirmation']['tuned']['overrides']
AUDIT = json.load(open('phase5_target_audit.json'))
el_cols = [c for c in d.features if c != 'Temperature_C']          # prep_enc + 65 elements
elements = [c for c in el_cols if c != 'prep_enc']
ei = {e: el_cols.index(e) for e in elements}

grp = d.dl_lab.groupby(d.groups)
cat_df = grp.agg(**{c: (c, 'first') for c in el_cols}, y_max=(TARGET, 'max'))
Xc = cat_df[el_cols].values.astype(float); yc = cat_df['y_max'].values
prep_val = float(cat_df['prep_enc'].mode()[0])                      # Impregnation code
n_cat = len(yc)

# ---- vocabulary from the lab's own designs ----
E = cat_df[elements].values.astype(float)
major_idx = E.argmax(1)
SUPPORTS = [e for e, c in pd.Series([elements[i] for i in major_idx]).value_counts().items() if c >= 20]
prom_counts = pd.Series([elements[j] for r, m in zip(E, major_idx)
                         for j in np.where(r > 0)[0] if j != m]).value_counts()
PROMOTERS = [e for e, c in prom_counts.items() if c >= 20]
elem_support = {e: int((cat_df[e].values > 0).sum()) for e in elements}
log(f"{len(SUPPORTS)} supports {SUPPORTS}")
log(f"{len(PROMOTERS)} promoters {PROMOTERS}")

# ---- enumerate candidates in-grammar ----
existing = set(map(tuple, np.round(Xc, 3)))
rows, labels = [], []
for support in SUPPORTS:
    for k, (s_load, p_load) in [(3, (90.0, 3.3333)), (2, (93.3333, 3.3333))]:
        for combo in itertools.combinations([p for p in PROMOTERS if p != support], k):
            v = np.zeros(len(el_cols)); v[el_cols.index('prep_enc')] = prep_val
            v[ei[support]] = s_load
            for p in combo:
                v[ei[p]] = p_load
            if tuple(np.round(v, 3)) in existing:
                continue
            rows.append(v)
            labels.append({'support': support, 'promoters': '+'.join(combo),
                           'formula': f"{support}({s_load:.1f}) " + " ".join(f"{p}({p_load:.2f})" for p in combo)})
Xcand = np.array(rows)
log(f"enumerated {len(Xcand):,} unseen candidates in the lab's own grammar")

# ---- 10-seed ensemble trained on all 917 ----
sc = StandardScaler().fit(Xc)
Xc_sc, Xcand_sc = sc.transform(Xc), sc.transform(Xcand)
preds = np.array([lgb.LGBMRegressor(**lgb_params(s, **TUNED)).fit(Xc_sc, yc).predict(Xcand_sc)
                  for s in range(10)])
pred_mean, pred_std = preds.mean(0), preds.std(0)
log(f"ensemble done; predicted max yield range {pred_mean.min():.2f}-{pred_mean.max():.2f}% "
    f"(training observed max reaches {yc.max():.2f}%)")

# ---- coverage descriptors ----
nn_dist = cdist(Xcand_sc, Xc_sc).min(1)
train_nn = np.sort(cdist(Xc_sc, Xc_sc), axis=1)[:, 1]      # each training catalyst's nearest neighbour
NN_CUT = float(np.quantile(train_nn, 0.95))                # "as far as the sparsest training region"
min_elem_support = np.array([min(elem_support[e] for e in ([lab['support']] + lab['promoters'].split('+')))
                             for lab in labels])
MIN_SUPPORT_CUT = 20
tier = np.where((nn_dist <= NN_CUT) & (min_elem_support >= MIN_SUPPORT_CUT), 'IN_SUPPORT', 'EXTRAPOLATIVE')
log(f"tiers: IN_SUPPORT={np.sum(tier=='IN_SUPPORT'):,}  EXTRAPOLATIVE={np.sum(tier=='EXTRAPOLATIVE'):,} "
    f"(nn cutoff {NN_CUT:.2f} = 95th pct of training NN distance)")

# ---- verify the unpriceable-element safeguard with a drop-column test ----
absent = [e for e in elements if elem_support[e] == 0]
if absent:
    probe = absent[0]
    keep = [i for i, c in enumerate(el_cols) if c != probe]
    s2 = StandardScaler().fit(Xc[:, keep])
    m_full = lgb.LGBMRegressor(**lgb_params(0, **TUNED)).fit(Xc_sc, yc)
    m_drop = lgb.LGBMRegressor(**lgb_params(0, **TUNED)).fit(s2.transform(Xc[:, keep]), yc)
    probe_v = Xcand[:50].copy(); probe_v[:, ei[probe]] = 3.3333
    diff = float(np.abs(m_full.predict(sc.transform(probe_v)) -
                        m_drop.predict(s2.transform(probe_v[:, keep]))).max())
    SAFEGUARD = {'probe_element': probe, 'max_abs_pred_diff': diff,
                 'interpretation': 'zero difference confirms an absent element is provably unpriceable'}
else:
    SAFEGUARD = {'probe_element': None,
                 'note': 'every element in the feature set appears in >=1 training catalyst; '
                         'the enumeration draws only on well-supported elements, so no candidate is unpriceable'}
log(f"safeguard check: {SAFEGUARD}")

# ---- rank & assemble ----
df = pd.DataFrame(labels)
df['predicted_max_yield'] = np.round(pred_mean, 3)
df['ensemble_std'] = np.round(pred_std, 3)
df['nn_distance'] = np.round(nn_dist, 3)
df['min_element_support'] = min_elem_support
df['tier'] = tier
df = df.sort_values('predicted_max_yield', ascending=False).reset_index(drop=True)
shortlist = df[df.tier == 'IN_SUPPORT'].head(20).copy()
explor = df[df.tier == 'EXTRAPOLATIVE'].head(5).copy()

# ---- diversity-aware shortlist -------------------------------------------------
# Ranking by predicted yield alone returns 20 near-identical Ba/Mo catalysts: that campaign tests
# one hypothesis twenty times. Greedy selection with a minimum-separation constraint spends the same
# synthesis budget on genuinely different chemistry.
insup = df[df.tier == 'IN_SUPPORT'].reset_index(drop=True)
insup_vec = sc.transform(np.array([rows[i] for i in
                                   [labels.index(l) for l in insup[['support', 'promoters', 'formula']]
                                    .to_dict('records')]])) if False else None   # (index map unused)
# recompute vectors for the in-support subset directly
lab_key = {l['formula']: i for i, l in enumerate(labels)}
insup_idx = np.array([lab_key[f] for f in insup['formula']])
V = Xcand_sc[insup_idx]
SEP = float(np.quantile(train_nn, 0.50))          # separation = typical spacing of the lab's own designs
chosen = [0]
for i in range(1, len(insup)):
    if len(chosen) >= 20:
        break
    if cdist(V[i:i + 1], V[chosen]).min() >= SEP:
        chosen.append(i)
diverse = insup.iloc[chosen].copy()

# ---- best candidate per support (spread across chemistries) ---------------------
per_support = (df[df.tier == 'IN_SUPPORT'].sort_values('predicted_max_yield', ascending=False)
               .groupby('support', as_index=False).first()
               .sort_values('predicted_max_yield', ascending=False))

df.head(500).to_csv('phase6_candidates.csv', index=False)
diverse.to_csv('phase6_candidates_diverse.csv', index=False)

ci = AUDIT['E_bootstrap_ci']
OUT = {'meta': {
        'model': 'formulation B (composition -> per-catalyst max yield), tuned LGBM, 10-seed ensemble on all 917',
        'literature_prior_used': False,
        'reason_no_prior': 'Phase 3 null + 28-family coverage test (phase6_our_experiments.py) both negative',
        'n_candidates_enumerated': int(len(Xcand)),
        'grammar': 'Impregnation; support ~90% + 2-3 promoters ~3.33%; supports/promoters taken from lab vocabulary',
        'supports': SUPPORTS, 'promoters': PROMOTERS,
        'tier_rule': f'IN_SUPPORT if nn_distance <= {NN_CUT:.3f} (95th pct of training NN) and every element '
                     f'appears in >= {MIN_SUPPORT_CUT} training catalysts',
        'honest_expected_hit_rate': {
            'metric': 'retrospective grouped-CV precision@20 vs true top-decile',
            'point': 0.44, 'ci95': ci['precision_at20_ci95'],
            'enrichment_ci95': ci['enrichment_ci95'],
            'caveat': 'ranking is the deliverable; absolute predicted yields regress to the mean '
                      f'(training observed max {float(yc.max()):.2f}%, ensemble max {float(pred_mean.max()):.2f}%)'}},
       'unpriceable_safeguard': SAFEGUARD,
       'diversity_rule': f'greedy: highest predicted first, each subsequent candidate >= {SEP:.3f} '
                         f'(median training NN distance) from all already chosen',
       'shortlist_in_support': shortlist.to_dict('records'),
       'shortlist_diverse': diverse.to_dict('records'),
       'best_per_support': per_support.to_dict('records'),
       'examples_extrapolative_flagged': explor.to_dict('records')}
json.dump(OUT, open('phase6_candidates.json', 'w'), indent=1)
log("wrote phase6_candidates.json + phase6_candidates.csv")

print("\n===== TOP 20 CANDIDATES (IN_SUPPORT tier — ranks trustworthy) =====")
print(f"{'#':>2s} {'formula':38s} {'pred':>6s} {'+/-':>5s} {'nnd':>5s} {'minsup':>6s}")
for i, r in shortlist.iterrows():
    print(f"{i+1:2d} {r['formula']:38s} {r['predicted_max_yield']:6.2f} {r['ensemble_std']:5.2f} "
          f"{r['nn_distance']:5.2f} {r['min_element_support']:6d}")
print(f"\nExpected hit rate (true top-decile among these 20): {ci['precision_at20_ci95'][0]:.2f}-{ci['precision_at20_ci95'][1]:.2f} "
      f"(95% CI); enrichment {ci['enrichment_ci95'][0]:.2f}-{ci['enrichment_ci95'][1]:.2f}x over random")
print(f"\n===== DIVERSITY-AWARE SHORTLIST ({len(diverse)} candidates, min separation {SEP:.2f}) =====")
print("same budget, spread across genuinely different chemistry")
for i, (_, r) in enumerate(diverse.iterrows(), 1):
    print(f"{i:2d} {r['formula']:38s} {r['predicted_max_yield']:6.2f} +/-{r['ensemble_std']:.2f}  nnd {r['nn_distance']:5.2f}")
print("\n===== BEST PER SUPPORT =====")
for _, r in per_support.iterrows():
    print(f"   {r['support']:3s} {r['formula']:38s} {r['predicted_max_yield']:6.2f}")
print("\n===== EXTRAPOLATIVE examples (FLAGGED — rank not reliable) =====")
for i, r in explor.iterrows():
    print(f"   {r['formula']:38s} pred {r['predicted_max_yield']:5.2f} nnd {r['nn_distance']:5.2f} "
          f"minsup {r['min_element_support']:3d}")
log("done")
