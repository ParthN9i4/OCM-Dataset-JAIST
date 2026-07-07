import json
nb=json.load(open('ocm_methodology.ipynb'))
def setsrc(pred, new):
    for c in nb['cells']:
        s="".join(c['source'])
        if pred(s): c['source']=new; return True
    raise SystemExit("NOT FOUND: "+new[:40])

# 1) PFT def: add use_filter
setsrc(lambda s: s.startswith('def two_stage_pft'),
"""def two_stage_pft(seed=42, tau1=0.30, cv=5, use_filter=True):
    # Stage 0 (optional): DRST filter, else use ALL literature for the Stage-1 expert
    if use_filter:
        m = p_lab_lit >= tau1; Xlp, ylp = X_lit_sc[m], y_lit[m]
    else:
        Xlp, ylp = X_lit_sc, y_lit                                        # all 3,852 literature rows
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed); rmses = []
    for tr, va in kf.split(X_lab_sc):
        y_pre = quantile_normalize_y(ylp, y_lab[tr])                       # rescale onto THIS fold's lab labels
        pre = xgb.XGBRegressor(**xgb_params(seed))
        pre.fit(np.vstack([Xlp, X_lab_sc[tr]]), np.concatenate([y_pre, y_lab[tr]]))   # Stage 1
        ptr = pre.predict(X_lab_sc[tr]).reshape(-1, 1); pva = pre.predict(X_lab_sc[va]).reshape(-1, 1)
        fin = lgb.LGBMRegressor(**lgb_params(seed))
        fin.fit(np.hstack([X_lab_sc[tr], ptr]), y_lab[tr])                 # Stage 2 (lab labels only)
        rmses.append(np.sqrt(mean_squared_error(y_lab[va], fin.predict(np.hstack([X_lab_sc[va], pva])))))
    return float(np.mean(rmses))
RESULTS['5. Two-stage PFT (ours)'] = two_stage_pft(seed=42)
b = RESULTS['1. Baseline (lab data only)']; p = RESULTS['5. Two-stage PFT (ours)']
print(f"PFT RMSE = {p:.3f}   ({100*(b-p)/b:+.1f}% vs baseline {b:.3f})")""")

# 2) seed-loop cell -> baseline vs PFT (filtered) + PFT (all lit) with MAE/R2
setsrc(lambda s: 'SEEDS =' in s and 'ttest_rel' in s,
"""from sklearn.metrics import mean_absolute_error, r2_score
def pft_eval(seed, use_filter=True, cv=5):
    if use_filter: m = p_lab_lit >= 0.30; Xlp, ylp = X_lit_sc[m], y_lit[m]
    else:          Xlp, ylp = X_lit_sc, y_lit
    kf = KFold(cv, shuffle=True, random_state=seed); rm, ma, rs = [], [], []
    for tr, va in kf.split(X_lab_sc):
        ypre = quantile_normalize_y(ylp, y_lab[tr])
        pre = xgb.XGBRegressor(**xgb_params(seed)).fit(np.vstack([Xlp, X_lab_sc[tr]]), np.concatenate([ypre, y_lab[tr]]))
        fin = lgb.LGBMRegressor(**lgb_params(seed)).fit(np.hstack([X_lab_sc[tr], pre.predict(X_lab_sc[tr]).reshape(-1,1)]), y_lab[tr])
        yp = fin.predict(np.hstack([X_lab_sc[va], pre.predict(X_lab_sc[va]).reshape(-1,1)]))
        rm.append(np.sqrt(mean_squared_error(y_lab[va], yp))); ma.append(mean_absolute_error(y_lab[va], yp)); rs.append(r2_score(y_lab[va], yp))
    return np.mean(rm), np.mean(ma), np.mean(rs)

SEEDS = [0, 1, 2, 7, 13, 21, 42, 77, 123, 2025]
b_arr, pf, pa = [], [], []
for s in SEEDS:
    b_arr.append(evaluate_cv_ours(seed=s)); pf.append(pft_eval(s, True)); pa.append(pft_eval(s, False))
b_arr = np.array(b_arr); pf = np.array(pf); pa = np.array(pa)
print("config                RMSE                MAE      R2")
print(f"Baseline            {b_arr.mean():.3f} +/- {b_arr.std(ddof=1):.3f}     (RMSE only)")
print(f"PFT (DRST-filtered) {pf[:,0].mean():.3f} +/- {pf[:,0].std(ddof=1):.3f}   {pf[:,1].mean():.3f}    {pf[:,2].mean():.3f}")
print(f"PFT (all literature){pa[:,0].mean():.3f} +/- {pa[:,0].std(ddof=1):.3f}   {pa[:,1].mean():.3f}    {pa[:,2].mean():.3f}")
print(f"\\nPFT-filtered wins {(pf[:,0]<b_arr).sum()}/{len(SEEDS)}; PFT-all wins {(pa[:,0]<b_arr).sum()}/{len(SEEDS)} seeds")
print(f"baseline vs PFT-filtered: paired t p = {ttest_rel(b_arr, pf[:,0])[1]:.1e}")
print(f"Stage-0 filter edge (all - filtered) = {pa[:,0].mean()-pf[:,0].mean():+.4f} RMSE (~{100*(pa[:,0].mean()-pf[:,0].mean())/pf[:,0].mean():.1f}%) -> optional")
plt.figure(figsize=(8, 3.4)); x = np.arange(len(SEEDS))
plt.plot(x, b_arr, 'o-', color='gray', label=f'baseline {b_arr.mean():.3f}')
plt.plot(x, pf[:,0], 's-', color='#1f4e79', label=f'PFT filtered {pf[:,0].mean():.3f}')
plt.plot(x, pa[:,0], '^--', color='#e07a5f', label=f'PFT all-lit {pa[:,0].mean():.3f}')
plt.xticks(x, SEEDS, rotation=45); plt.ylabel('CV RMSE')
plt.title('PFT beats baseline on every seed; Stage-0 filter is optional'); plt.legend()
plt.tight_layout(); plt.show()""")

# 3) held-out cell: train-only DRST classifier (airtight untouched claim)
setsrc(lambda s: 'Held-out 20%' in s,
"""rng_h = np.random.default_rng(2025); perm = rng_h.permutation(len(X_lab_sc)); nt = int(0.2*len(perm))
te, trn = perm[:nt], perm[nt:]
# Train the DRST domain classifier on the TRAIN split only -> the held-out 20% touches NOTHING.
rc = np.random.default_rng(42); ssub = rc.choice(len(trn), min(10000, len(trn)), replace=False)
clf_h = LogisticRegression(C=0.5, max_iter=1000, random_state=42, n_jobs=-1).fit(
    np.vstack([X_lab_sc[trn[ssub]], X_lit_sc]), np.r_[np.ones(len(ssub)), np.zeros(len(X_lit_sc))])
mask = clf_h.predict_proba(X_lit_sc)[:, 1] >= 0.30
mb = lgb.LGBMRegressor(**lgb_params(42)).fit(X_lab_sc[trn], y_lab[trn])
rb = np.sqrt(mean_squared_error(y_lab[te], mb.predict(X_lab_sc[te])))
yq = quantile_normalize_y(y_lit[mask], y_lab[trn])
pre = xgb.XGBRegressor(**xgb_params(42)).fit(np.vstack([X_lit_sc[mask], X_lab_sc[trn]]), np.concatenate([yq, y_lab[trn]]))
fin = lgb.LGBMRegressor(**lgb_params(42)).fit(np.hstack([X_lab_sc[trn], pre.predict(X_lab_sc[trn]).reshape(-1,1)]), y_lab[trn])
rp = np.sqrt(mean_squared_error(y_lab[te], fin.predict(np.hstack([X_lab_sc[te], pre.predict(X_lab_sc[te]).reshape(-1,1)]))))
print(f"Held-out 20% (untouched; DRST classifier trained on TRAIN split only): baseline {rb:.3f} -> PFT {rp:.3f} ({100*(rb-rp)/rb:+.1f}%)")""")

# 4) global rename prior_prediction -> lit_prior_prediction, and update §9 markdown
n=0
for c in nb['cells']:
    s="".join(c['source'])
    if 'prior_prediction' in s and 'lit_prior_prediction' not in s:
        c['source']=s.replace('prior_prediction','lit_prior_prediction'); n+=1
    if s.startswith('## 9. Is the PFT gain real'):
        c['source']=s.replace(
            "held-out 20% of the lab data**, set aside from the start and never used in any training.",
            "held-out 20% of the lab data**, set aside from the start and never used in any training (the DRST classifier is retrained on the train split only). We also check whether the **Stage-0 filter matters** by training Stage 1 on all literature vs the DRST-filtered subset.")
print("prior_prediction renamed in", n, "cells")

json.dump(nb, open('ocm_methodology.ipynb','w'), indent=1)
import nbformat, ast
v=nbformat.read('ocm_methodology.ipynb',4); nbformat.validate(v)
for c in nb['cells']:
    if c['cell_type']=='code': ast.parse("".join(c['source']))
print("patched + valid + all code parses; cells:", len(nb['cells']))
