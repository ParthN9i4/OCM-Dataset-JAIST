"""
patch_methodology_figs.py
Make ocm_methodology.ipynb the SINGLE source for all 8 report figures.
Adds seaborn import + inserts/extends cells so every report figure is saved to its exact filename:
  fig_mean_shift, fig_element_usage, fig_drst_scores, fig_kmm_weights,
  fig_bias_correction, fig_repeated_runs, fig_shap_bar, fig_worknote_results
Faithful to the styles in ocm_analysis.ipynb / feedback_experiments.py, with lab-data variable names.
"""
import json, ast, nbformat

NB = 'ocm_methodology.ipynb'
nb = json.load(open(NB))
cells = nb['cells']

def by_id(cid):
    for i, c in enumerate(cells):
        if c.get('id') == cid:
            return i, c
    raise SystemExit('cell id not found: ' + cid)

def src(c):
    return ''.join(c['source']) if isinstance(c['source'], list) else c['source']

def setsrc(c, s):
    c['source'] = s
    c['outputs'] = []           # clear stale outputs; re-execution rebakes them
    c['execution_count'] = None

def newcode(cid, s):
    return {'cell_type': 'code', 'id': cid, 'metadata': {}, 'source': s,
            'outputs': [], 'execution_count': None}

def newmd(cid, s):
    return {'cell_type': 'markdown', 'id': cid, 'metadata': {}, 'source': s}

# ---------------------------------------------------------------------------
# 1) imports cell: add seaborn
# ---------------------------------------------------------------------------
i, c = by_id('d98f1a18')
s = src(c)
if 'import seaborn' not in s:
    s = s.replace('import matplotlib.pyplot as plt',
                  'import matplotlib.pyplot as plt\nimport seaborn as sns')
    setsrc(c, s)

# ---------------------------------------------------------------------------
# 2) NEW cells after §1 data cell (776a11c2): fig_mean_shift + fig_element_usage
# ---------------------------------------------------------------------------
FIG_MEAN_SHIFT = r'''# ── Report Figure 1 — label shift (three complementary views) ────────────────
from scipy.stats import gaussian_kde
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Y(C2) distribution — Lab data vs Literature data", fontsize=14, fontweight='bold', y=1.02)
BLUE = '#2196F3'; ORANGE = '#FF9800'
mean_lab = y_lab.mean(); mean_lit = y_lit.mean(); shift = mean_lit - mean_lab

# Panel 1: normalised density (hist + KDE)
ax = axes[0]
ax.hist(y_lab, bins=60, density=True, alpha=0.35, color=BLUE,   label='Lab data')
ax.hist(y_lit, bins=40, density=True, alpha=0.45, color=ORANGE, label='Literature data')
for y, color in [(y_lab, BLUE), (y_lit, ORANGE)]:
    kde = gaussian_kde(y, bw_method=0.15); xs = np.linspace(0, y.max(), 400)
    ax.plot(xs, kde(xs), color=color, lw=2.5)
ax.axvline(mean_lab, color=BLUE,   lw=2, ls='--', label=f'Mean lab = {mean_lab:.2f}%')
ax.axvline(mean_lit, color=ORANGE, lw=2, ls='--', label=f'Mean lit = {mean_lit:.2f}%')
ax.axvspan(mean_lab, mean_lit, alpha=0.12, color='red', label=f'Shift = {shift:.2f} pp')
ax.set_xlabel('Y(C2) [%]', fontsize=12); ax.set_ylabel('Probability density', fontsize=12)
ax.set_title('Normalised density\n(both curves sum to 1)', fontsize=11)
ax.set_xlim(0, 30); ax.legend(fontsize=9)

# Panel 2: empirical CDF
ax = axes[1]
for y, color, label in [(y_lab, BLUE, 'Lab data'), (y_lit, ORANGE, 'Literature data')]:
    sy = np.sort(y); cdf = np.arange(1, len(y) + 1) / len(y)
    ax.plot(sy, cdf, color=color, lw=2.5, label=label)
p50_lab = np.percentile(y_lab, 50); p50_lit = np.percentile(y_lit, 50)
ax.annotate('', xy=(p50_lit, 0.50), xytext=(p50_lab, 0.50), arrowprops=dict(arrowstyle='<->', color='red', lw=1.8))
ax.text((p50_lab + p50_lit) / 2, 0.53, f'delta median\n= {p50_lit - p50_lab:.2f} pp', ha='center', fontsize=9, color='red')
frac = (y_lab < mean_lit).mean()
ax.plot([mean_lab, mean_lit], [frac, frac], 'r--', lw=1.5, alpha=0.6)
ax.text(mean_lit + 0.3, frac, f'<- mean shift\n   {shift:.2f} pp', fontsize=8.5, color='red', va='center')
ax.axvline(mean_lab, color=BLUE, lw=1.5, ls=':'); ax.axvline(mean_lit, color=ORANGE, lw=1.5, ls=':')
ax.set_xlabel('Y(C2) [%]', fontsize=12); ax.set_ylabel('Cumulative fraction of samples', fontsize=12)
ax.set_title('Empirical CDF\n(rightward = higher yields in literature)', fontsize=11)
ax.set_xlim(0, 30); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

# Panel 3: violin + box
ax = axes[2]
plot_df = pd.DataFrame({'Y(C2) [%]': np.concatenate([y_lab, y_lit]),
    'Dataset': ['Lab data'] * len(y_lab) + ['Literature data'] * len(y_lit)})
sns.violinplot(data=plot_df, x='Dataset', y='Y(C2) [%]',
               palette={'Lab data': BLUE, 'Literature data': ORANGE}, inner='box', cut=0, ax=ax)
ax.scatter(['Lab data', 'Literature data'], [mean_lab, mean_lit], color='white', edgecolor='black', s=80, zorder=5, label='Mean')
ax.annotate('', xy=(1, mean_lit), xytext=(1, mean_lab), arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(1.08, (mean_lab + mean_lit) / 2, f'{shift:.2f} pp\nshift', color='red', fontsize=10, va='center')
ax.set_title('Violin + box\n(white dot = mean, box = IQR)', fontsize=11)
ax.legend(fontsize=9); ax.set_ylim(0, 35); ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout(); plt.savefig('fig_mean_shift.png', bbox_inches='tight', dpi=130); plt.show()
print(f"Lab mean {mean_lab:.3f}%  |  Literature mean {mean_lit:.3f}%  |  shift {shift:.3f} pp  -> fig_mean_shift.png")'''

FIG_ELEMENT_USAGE = r'''# ── Report Figure 2 — covariate shift (element usage differs) ────────────────
nonzero_lab = (df_lab[ELEM_COLS] > 0).mean().sort_values(ascending=False)
nonzero_lit = (df_lit[ELEM_COLS] > 0).mean().sort_values(ascending=False)
top_n = 15
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
top_lab = nonzero_lab.head(top_n)
axes[0].barh(top_lab.index[::-1], top_lab.values[::-1], color='steelblue', edgecolor='k')
axes[0].set_xlabel('Fraction of samples with element > 0'); axes[0].set_title(f'Top {top_n} elements - Lab data')
top_lit = nonzero_lit.head(top_n)
axes[1].barh(top_lit.index[::-1], top_lit.values[::-1], color='darkorange', edgecolor='k')
axes[1].set_xlabel('Fraction of samples with element > 0'); axes[1].set_title(f'Top {top_n} elements - Literature data')
plt.tight_layout(); plt.savefig('fig_element_usage.png', bbox_inches='tight'); plt.show()
print('-> fig_element_usage.png')'''

EDA_MD = ('### The two distribution shifts (report Figures 1-2)\n\n'
          'Before evaluating any method, we visualise the two differences between the datasets that make '
          'this a domain-adaptation problem: the **label shift** (literature yields sit ~3.4 pp above the '
          'lab) and the **covariate shift** (the two datasets emphasise different elements). '
          'These two cells produce report figures `fig_mean_shift.png` and `fig_element_usage.png`.')

di, _ = by_id('776a11c2')                                   # after the §1 data cell
block = [newmd('edafig_md', EDA_MD),
         newcode('fig_meanshift', FIG_MEAN_SHIFT),
         newcode('fig_elemusage', FIG_ELEMENT_USAGE)]
cells[di + 1:di + 1] = block

# ---------------------------------------------------------------------------
# 3) §5 DRST cell (f229ebd1): append fig_drst_scores
# ---------------------------------------------------------------------------
i, c = by_id('f229ebd1')
if 'fig_drst_scores' not in src(c):
    setsrc(c, src(c) + '\n\n' + r'''# ── Report Figure 3 — DRST density-ratio scores ─────────────────────────────
fig, ax = plt.subplots(figsize=(8, 3))
ax.hist(p_lab_lit, bins=40, color='teal', edgecolor='k', alpha=0.8)
for t in [0.1, 0.2, 0.3, 0.4]:
    ax.axvline(t, ls='--', label=f'tau={t}')
ax.set_xlabel('P(lab data | x)  - density-ratio proxy'); ax.set_ylabel('Count')
ax.set_title('DRST: how lab-like each literature record is')
ax.legend(); plt.tight_layout(); plt.savefig('fig_drst_scores.png', bbox_inches='tight'); plt.show()''')

# ---------------------------------------------------------------------------
# 4) §6 KMM cell (daa28bb3): append fig_kmm_weights
# ---------------------------------------------------------------------------
i, c = by_id('daa28bb3')
if 'fig_kmm_weights' not in src(c):
    setsrc(c, src(c) + '\n\n' + r'''# ── Report Figure 4 — KMM weights and their agreement with DRST ─────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 3))
axes[0].hist(w_kmm, bins=40, color='mediumpurple', edgecolor='k', alpha=0.8)
axes[0].set_xlabel('KMM weight'); axes[0].set_ylabel('Count'); axes[0].set_title('Distribution of KMM weights')
axes[1].scatter(p_lab_lit, w_kmm, alpha=0.3, s=8, color='purple')
axes[1].set_xlabel('DRST score P(lab data | x)'); axes[1].set_ylabel('KMM weight')
r_corr = np.corrcoef(p_lab_lit, w_kmm)[0, 1]
axes[1].set_title(f'KMM weight vs DRST score (r={r_corr:.3f})')
plt.tight_layout(); plt.savefig('fig_kmm_weights.png', bbox_inches='tight'); plt.show()''')

# ---------------------------------------------------------------------------
# 5) NEW cell in §7 (after the §7 markdown 764c382e): fig_bias_correction
# ---------------------------------------------------------------------------
FIG_BIAS = r'''# ── Report Figure 5 — Stage-1 label rescaling (quantile normalisation) ──────
# Show what the rank-preserving rescaling does BEFORE it is used inside PFT Stage 1 below.
y_lit_qnorm = quantile_normalize_y(y_lit, y_lab)
print("Literature Y(C2) before vs after rescaling onto the lab-data scale:")
print(f"  Raw literature : mean={y_lit.mean():.3f}  median={np.median(y_lit):.3f}")
print(f"  Rescaled       : mean={y_lit_qnorm.mean():.3f}  median={np.median(y_lit_qnorm):.3f}")
print(f"  Lab data       : mean={y_lab.mean():.3f}  median={np.median(y_lab):.3f}")
fig, ax = plt.subplots(figsize=(9, 3))
ax.hist(y_lab,       bins=40, alpha=0.55, color='steelblue',  density=True, label='Lab data')
ax.hist(y_lit,       bins=40, alpha=0.45, color='darkorange', density=True, label='Literature (raw)')
ax.hist(y_lit_qnorm, bins=40, alpha=0.45, color='green',      density=True, label='Literature (rescaled to lab scale)')
ax.set_xlabel('Y(C2) [%]'); ax.set_ylabel('Density')
ax.set_title('Stage-1 rescaling: literature yields mapped onto the lab yield range')
ax.legend(); plt.tight_layout(); plt.savefig('fig_bias_correction.png', bbox_inches='tight'); plt.show()'''

bi, _ = by_id('764c382e')                                   # after §7 markdown, before the PFT code cell
cells[bi + 1:bi + 1] = [newcode('fig_biascorr', FIG_BIAS)]

# ---------------------------------------------------------------------------
# 6) §8 results cell (83522f3d): add savefig fig_worknote_results
# ---------------------------------------------------------------------------
i, c = by_id('83522f3d')
if 'fig_worknote_results' not in src(c):
    setsrc(c, src(c).replace(
        "plt.tight_layout(); plt.show()",
        "plt.tight_layout(); plt.savefig('fig_worknote_results.png', bbox_inches='tight', dpi=130); plt.show()"))

# ---------------------------------------------------------------------------
# 7) §9 repeated-runs cell (fc045793): replace single-panel plot with 2-panel + savefig
# ---------------------------------------------------------------------------
i, c = by_id('fc045793')
s = src(c)
old_plot = s[s.index("plt.figure(figsize=(8, 3.4))"):]
NEW_PLOT = r'''from scipy.stats import wilcoxon
n_wins = int((pf[:,0] < b_arr).sum())
t_p = ttest_rel(b_arr, pf[:,0])[1]; w_p = wilcoxon(b_arr, pf[:,0])[1]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6)); x = np.arange(len(SEEDS))
ax[0].plot(x, b_arr, 'o-', color='gray', lw=2, label=f'Baseline  {b_arr.mean():.3f}+/-{b_arr.std(ddof=1):.3f}')
ax[0].plot(x, pf[:,0], 's-', color='#1f4e79', lw=2, label=f'PFT  {pf[:,0].mean():.3f}+/-{pf[:,0].std(ddof=1):.3f}')
ax[0].fill_between(x, b_arr, pf[:,0], where=pf[:,0] < b_arr, color='green', alpha=0.12)
ax[0].set_xticks(x); ax[0].set_xticklabels([str(sd) for sd in SEEDS], rotation=45, fontsize=8)
ax[0].set_xlabel('random seed'); ax[0].set_ylabel('5-fold CV RMSE')
ax[0].set_title(f'PFT beats baseline in {n_wins}/{len(SEEDS)} runs'); ax[0].legend(fontsize=8)
ax[1].bar(x, b_arr - pf[:,0], color=['green' if d > 0 else 'red' for d in (b_arr - pf[:,0])])
ax[1].axhline(0, color='k', lw=0.8)
ax[1].set_xticks(x); ax[1].set_xticklabels([str(sd) for sd in SEEDS], rotation=45, fontsize=8)
ax[1].set_xlabel('random seed'); ax[1].set_ylabel('RMSE improvement (baseline - PFT)')
ax[1].set_title(f'Improvement every run\npaired t p={t_p:.1e}, Wilcoxon p={w_p:.1e}')
plt.tight_layout(); plt.savefig('fig_repeated_runs.png', dpi=130); plt.show()'''
setsrc(c, s.replace(old_plot, NEW_PLOT))

# ---------------------------------------------------------------------------
# 8) §10 SHAP cell (8d0929dc): top-20 orange-highlighted bar + savefig fig_shap_bar
# ---------------------------------------------------------------------------
i, c = by_id('8d0929dc')
s = src(c)
old_plot = s[s.index("imp = np.abs(sv).mean(0)"):]
NEW_SHAP = r'''imp = np.abs(sv).mean(0)
shap_imp = pd.Series(imp, index=names_aug).sort_values(ascending=False).head(20)
fig_bar, ax_bar = plt.subplots(figsize=(9, 6))
colors = ['#E65100' if f == 'lit_prior_prediction' else '#1565C0' for f in shap_imp.index[::-1]]
ax_bar.barh(shap_imp.index[::-1], shap_imp.values[::-1], color=colors, edgecolor='k', linewidth=0.5)
ax_bar.set_xlabel('Mean |SHAP value| - average impact on Y(C2) prediction [%]')
ax_bar.set_title('Top 20 features by SHAP importance\nOrange = literature prior feature; Blue = catalyst features', fontsize=11)
from matplotlib.patches import Patch
ax_bar.legend(handles=[Patch(color='#1565C0', label='Catalyst / condition feature'),
                       Patch(color='#E65100', label='Literature prior prediction')], loc='lower right')
plt.tight_layout(); plt.savefig('fig_shap_bar.png', bbox_inches='tight', dpi=130); plt.show()
print('Top features:', list(shap_imp.index[:6]))'''
setsrc(c, s.replace(old_plot, NEW_SHAP))

# ---------------------------------------------------------------------------
# save + validate
# ---------------------------------------------------------------------------
json.dump(nb, open(NB, 'w'), indent=1)
v = nbformat.read(NB, 4); nbformat.validate(v)
n_save = 0
for c in cells:
    if c['cell_type'] == 'code':
        cs = src(c)
        ast.parse(cs)
        n_save += cs.count('plt.savefig(')
print(f'OK: patched + valid + all code parses. total cells={len(cells)}, savefig calls={n_save}')
figs = ['fig_mean_shift','fig_element_usage','fig_drst_scores','fig_kmm_weights',
        'fig_bias_correction','fig_repeated_runs','fig_shap_bar','fig_worknote_results']
full = ''.join(src(c) for c in cells if c['cell_type'] == 'code')
missing = [f for f in figs if f + ".png'" not in full]
print('missing figure savefigs:', missing if missing else 'NONE (all 8 present)')
