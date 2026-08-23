"""
build_corrected_figures.py — figures for worknote v2, generated from the committed experiment JSONs.

Reading numbers from the JSONs (rather than retyping them) keeps every figure locked to the
experiment that produced it, so a figure can never drift from its source. Each figure names the
script that produced its data.

Emits:
  fig_protocol_comparison.png  <- taniike_validation.json      (the central finding)
  fig_grouped_results.png      <- phase3_lit_prior.json        (replaces fig_worknote_results.png)
  fig_learning_curve.png       <- phase6_our_experiments.json  (the data-budget contribution)
  fig_shap_bar.png             <- REGENERATED on the composition-only grouped model
                                  (the previous version ranked lit_prior_prediction #1, which came
                                   from the leaked pipeline and now actively misleads)

Retained unchanged (protocol-independent): fig_mean_shift, fig_element_usage, fig_drst_scores,
fig_kmm_weights, fig_bias_correction.
Retired: fig_repeated_runs.png (10-seed "PFT wins" plot) — superseded by fig_protocol_comparison.
"""
import warnings, json
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VAL = json.load(open('taniike_validation.json'))
P3 = json.load(open('phase3_lit_prior.json'))
P6 = json.load(open('phase6_our_experiments.json'))
AUD = json.load(open('phase5_target_audit.json'))

NAVY, GREY, RED, GREEN = '#1f4e79', '#8aa0b8', '#c0392b', '#2e7d32'

# ---------------------------------------------------------------- 1. protocol comparison
A = VAL['A_grouped_vs_row']
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
groups = ['Baseline\n(lab only)', 'Prior-feature\nmethod (PFT)']
row = [A['row/baseline']['rmse_foldmean_mean'], A['row/pft_filtered']['rmse_foldmean_mean']]
grp = [A['grouped/baseline']['rmse_foldmean_mean'], A['grouped/pft_filtered']['rmse_foldmean_mean']]
row_e = [A['row/baseline']['rmse_foldmean_std'], A['row/pft_filtered']['rmse_foldmean_std']]
grp_e = [A['grouped/baseline']['rmse_foldmean_std'], A['grouped/pft_filtered']['rmse_foldmean_std']]
x = np.arange(2); w = 0.36
ax[0].bar(x - w/2, row, w, yerr=row_e, capsize=4, color=GREY, edgecolor='k', label='Row-level split')
ax[0].bar(x + w/2, grp, w, yerr=grp_e, capsize=4, color=NAVY, edgecolor='k', label='Catalyst-grouped split')
for xi, (r, g) in enumerate(zip(row, grp)):
    ax[0].text(xi - w/2, r + .05, f'{r:.3f}', ha='center', fontsize=9)
    ax[0].text(xi + w/2, g + .05, f'{g:.3f}', ha='center', fontsize=9)
ax[0].set_xticks(x); ax[0].set_xticklabels(groups)
ax[0].set_ylabel('CV RMSE  (lower = better)'); ax[0].set_ylim(0, 3.5); ax[0].legend(fontsize=9)
ax[0].set_title('Same models, two evaluation protocols', fontsize=11)

d_row = 100*(row[1]-row[0])/row[0]; d_grp = 100*(grp[1]-grp[0])/grp[0]
ax[1].bar(['Row-level', 'Catalyst-grouped'], [d_row, d_grp],
          color=[GREEN if d_row < 0 else RED, GREEN if d_grp < 0 else RED], edgecolor='k')
ax[1].axhline(0, color='k', lw=1)
for xi, v in enumerate([d_row, d_grp]):
    ax[1].text(xi, v + (0.4 if v > 0 else -0.9), f'{v:+.1f}%', ha='center', fontweight='bold')
ax[1].set_ylabel('PFT change vs. baseline (%)'); ax[1].set_ylim(-13, 5)
ax[1].set_title('The reported improvement does not survive\nwhen catalysts are kept whole', fontsize=11)
plt.tight_layout(); plt.savefig('fig_protocol_comparison.png', dpi=130, bbox_inches='tight'); plt.close()
print('fig_protocol_comparison.png   <- taniike_validation.json')

# ---------------------------------------------------------------- 2. grouped results
R = P3['results']
names = ['Composition only\n(control)', 'Literature\nrank prior', 'Similarity\nfeatures',
         'Gated prior', 'Catalyst-level\ndirect merge']
keys = ['V0', 'V1', 'V2', 'V3', 'V4']
sp = [R[k]['spearman_mean'] for k in keys]; se = [R[k]['spearman_std'] for k in keys]
fig, ax = plt.subplots(figsize=(9.5, 4.2))
cols = [NAVY] + [GREY]*4
b = ax.bar(range(5), sp, yerr=se, capsize=4, color=cols, edgecolor='k')
ax.axhline(sp[0], color=NAVY, ls='--', lw=1, alpha=.6)
for i, (v, k) in enumerate(zip(sp, keys)):
    delta = R[k].get('delta_spearman_vs_V0', 0.0)
    lbl = f'{v:.3f}' + ('' if k == 'V0' else f'\n({delta:+.3f})')
    ax.text(i, v + .012, lbl, ha='center', fontsize=9)
ax.set_xticks(range(5)); ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("Spearman ρ, predicted vs. observed\nmaximum yield (unseen catalysts)")
ax.set_ylim(0.70, 0.80)
ax.set_title('Catalyst-grouped protocol: no literature-integration variant\nimproves on composition alone',
             fontsize=11)
plt.tight_layout(); plt.savefig('fig_grouped_results.png', dpi=130, bbox_inches='tight'); plt.close()
print('fig_grouped_results.png       <- phase3_lit_prior.json')

# ---------------------------------------------------------------- 3. learning curve
E2 = P6['E2_learning_curve']
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for el, style in zip(['Ba', 'La', 'Ti', 'Zr', 'Ce'],
                     [dict(color=RED, lw=2.6, marker='o', zorder=5)] + [dict(color=GREY, lw=1.5, marker='.')]*4):
    c = E2[el]['curve']
    ns = sorted(int(k) for k in c)
    ax[0].plot(ns, [c[str(n)]['spearman_mean'] for n in ns], label=el, **style)
ax[0].set_xlabel('number of that family already measured (training)')
ax[0].set_ylabel('Spearman ρ on held-out family members')
ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)
ax[0].set_title('How much data does a new catalyst family need?', fontsize=11)

fams = ['Ba', 'La', 'Ti', 'Zr', 'Ce']
frac = [100*E2[f]['spearman_at_zero']/E2[f]['spearman_at_full'] for f in fams]
ax[1].bar(fams, frac, color=[RED] + [GREY]*4, edgecolor='k')
ax[1].axhline(90, color='k', ls=':', lw=1)
for i, v in enumerate(frac):
    ax[1].text(i, v + .8, f'{v:.1f}%', ha='center', fontsize=9)
ax[1].set_ylabel('% of achievable ρ reached with\nZERO family members seen'); ax[1].set_ylim(0, 105)
ax[1].set_title('Ba is uniquely non-transferable;\nother families are inferable from neighbouring chemistry',
                fontsize=11)
plt.tight_layout(); plt.savefig('fig_learning_curve.png', dpi=130, bbox_inches='tight'); plt.close()
print('fig_learning_curve.png        <- phase6_our_experiments.json')

# ---------------------------------------------------------------- 4. SHAP, regenerated honestly
# The previous fig_shap_bar.png ranked `lit_prior_prediction` first by a wide margin. That model was
# the leaked pipeline, so the figure effectively illustrated the leak. Regenerated here on the
# composition-only model under the adopted formulation: it now answers a chemistry question —
# which composition features drive achievable maximum yield.
import shap
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from ocm_eval import Data, lgb_params, TARGET

d = Data.load()
TUNED = json.load(open('grouped_tuning.json'))['confirmation']['tuned']['overrides']
el_cols = [c for c in d.features if c != 'Temperature_C']
grp = d.dl_lab.groupby(d.groups)
cat_df = grp.agg(**{c: (c, 'first') for c in el_cols}, y_max=(TARGET, 'max'))
Xc = cat_df[el_cols].values.astype(float); yc = cat_df['y_max'].values
sc = StandardScaler().fit(Xc)
model = lgb.LGBMRegressor(**lgb_params(42, **TUNED)).fit(sc.transform(Xc), yc)
sv = shap.TreeExplainer(model).shap_values(sc.transform(Xc))
imp = np.abs(sv).mean(0)
order = np.argsort(-imp)[:15]
fig, ax = plt.subplots(figsize=(8.5, 5.2))
labels = [el_cols[j] for j in order][::-1]
vals = imp[order][::-1]
ax.barh(labels, vals, color=[NAVY if l == 'Ba' else GREY for l in labels], edgecolor='k', linewidth=.5)
ax.set_xlabel('mean |SHAP|  —  average effect on predicted maximum yield (%)')
ax.set_title('What drives achievable maximum yield (composition-only model,\n'
             'catalyst-level, 917 catalysts)', fontsize=11)
plt.tight_layout(); plt.savefig('fig_shap_bar.png', dpi=130, bbox_inches='tight'); plt.close()
print('fig_shap_bar.png              <- REGENERATED on composition-only model')
print('   top features:', [el_cols[j] for j in order[:6]])
print('\nRetired: fig_repeated_runs.png (superseded by fig_protocol_comparison.png)')
