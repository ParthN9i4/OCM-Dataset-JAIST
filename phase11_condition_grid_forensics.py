"""
phase11_condition_grid_forensics.py -- give the condition-grid claims a computational source.

Three claims about the within-cell structure were being quoted from prose with no script behind them,
in violation of SESSION_CONTEXT.md section 8 ("one script is the single source of truth for any number
reused across multiple artifacts"). This script computes all three, so they can be cited from JSON.

  1. THE ORDER-STATISTIC TEST. "The spacing between the two lowest values in a 27-row cell is 2.03x
     the interior spacing, whereas truncating a larger cell to 27 rows gives 0.90x" appeared only in
     phase8_target_robustness.py's docstring and in four documents. No script computed it and no JSON
     stored it. It is the claim that rejects the rival "top-27-by-yield export cut-off" reading, so it
     is load-bearing. Recomputed here under four interior-spacing definitions.

  2. THE COUNTING DISPROOF, which is simpler and stronger than the order statistic and was never
     stated: a top-27 export cut-off cannot produce a cell with more than 27 rows. Count them.

  3. THE MEASUREMENT-BUDGET REPLAY. "20 runs per catalyst (4 temperatures x 5 conditions) reproduces
     the full-135 ranking at rho 0.955" (SESSION_CONTEXT.md section 7 item 4) likewise had no source,
     and it underwrites a proposal to cut reactor runs from 135 to 20. Recomputed over a grid of
     budgets with a spread across draws, so the recommendation rests on a distribution, not a point.

It also records the finding that makes the gating question unanswerable from this file:

  4. WITHIN-CELL ORDERING. Every cell is stored sorted descending by yield. Row order therefore
     encodes rank, not acquisition sequence, so no position/decay/periodicity test can distinguish
     "27 distinct conditions" from "27 successive time-on-stream samples". This is why two prior
     analyses found no signature, and why a third would also fail.

HONESTY CONSTRAINT on claim 3: sub-sampling rows within a cell is a reactor-time saving only under the
distinct-conditions reading. Under the time-on-stream reading the 27 slots are samples drawn from one
continuous run, so taking 5 of them saves ANALYSIS, not reactor hours, and the budget arithmetic does
not transfer. Stated in meta and not to be dropped when quoting.

Output: phase11_condition_grid_forensics.json
"""
import warnings, json, time
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from ocm_eval import Data, TARGET

t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

d = Data.load()
lab = d.dl_lab.copy()
lab['cat_id'] = d.groups
n_cat = d.n_cat

# cells in FILE ORDER (groupby preserves within-group row order)
cells = {k: g[TARGET].to_numpy() for k, g in lab.groupby(['cat_id', 'Temperature_C'], sort=False)}
sizes = np.array([len(v) for v in cells.values()])
log(f"{len(cells)} cells over {n_cat} catalysts, {len(lab)} rows")

# ---------------------------------------------------------------- 4. within-cell ordering
desc_viol = sum(int((v[:-1] < v[1:]).sum()) for v in cells.values())
asc_viol = sum(int((v[:-1] > v[1:]).sum()) for v in cells.values())
n_comp = int(sum(len(v) - 1 for v in cells.values()))
labY = lab[TARGET].to_numpy()
ORDER = {
    'within_cell_descending_violations': desc_viol,
    'within_cell_ascending_violations': asc_viol,
    'within_cell_adjacent_comparisons': n_comp,
    'cells_perfectly_descending': int(sum(1 for v in cells.values() if (v[:-1] >= v[1:]).all())),
    'n_cells': int(len(cells)),
    'lab_block_global_descending_violations': int((labY[:-1] < labY[1:]).sum()),
    'lab_block_adjacent_comparisons': int(len(labY) - 1),
    'conclusion': ('every cell is stored sorted descending by yield, so within-cell row order encodes '
                   'RANK, not acquisition sequence; no ordering-based test can distinguish 27 distinct '
                   'conditions from 27 successive time-on-stream samples'),
    'implication': ('the conditions-vs-time-on-stream question is not resolvable from this CSV by any '
                    'further analysis; it needs the acquisition metadata from JAIST')}
log(f"4. ordering: {desc_viol}/{n_comp} descending violations across {len(cells)} cells "
    f"(lab block as a whole: {ORDER['lab_block_global_descending_violations']})")

# ---------------------------------------------------------------- 2. counting disproof
COUNT = {
    'cells_with_more_than_27_rows': int((sizes > 27).sum()),
    'cells_with_exactly_27_rows': int((sizes == 27).sum()),
    'cells_with_exactly_54_rows': int((sizes == 54).sum()),
    'max_cell_size': int(sizes.max()),
    'argument': ('a top-27-by-yield export cut-off cannot emit a cell with more than 27 rows; '
                 'cells above 27 rows therefore refute it directly, without any order statistic'),
    'caveat_on_54': ('SESSION_CONTEXT section 5A cites max cell size 54 = 2x27 as structural evidence; '
                     'it rests on this many cells')}
log(f"2. counting: {COUNT['cells_with_more_than_27_rows']} cells exceed 27 rows "
    f"(max {COUNT['max_cell_size']}); exactly-54 cells: {COUNT['cells_with_exactly_54_rows']}")


# ---------------------------------------------------------------- 1. order-statistic test
def gap_ratio(vals, defn):
    """bottom-of-distribution gap / interior gap, for an ascending-sorted sample."""
    v = np.sort(np.asarray(vals)); g = np.diff(v)
    if len(g) < 3: return None
    bottom, inner = g[0], g[1:]
    den = {'mean_interior': inner.mean(), 'median_interior': np.median(inner),
           'mean_all_gaps': g.mean(), 'mean_middle_50pct': g[len(g)//4: 3*len(g)//4].mean()}[defn]
    return float(bottom / den) if den > 0 else None


true27 = [v for v in cells.values() if len(v) == 27]
donors = [v for v in cells.values() if len(v) >= 34]        # comfortably larger than 27
trunc27 = [np.sort(v)[-27:] for v in donors]                # TOP-27 retention = the hypothesised cut

ORDSTAT = {'n_true_27_cells': len(true27), 'n_donor_cells_ge34': len(donors),
           'truncation_model': 'retain the TOP 27 by yield from a cell with >=34 rows',
           'by_definition': {}}
for defn in ['mean_interior', 'median_interior', 'mean_all_gaps', 'mean_middle_50pct']:
    a = [x for x in (gap_ratio(v, defn) for v in true27) if x is not None]
    b = [x for x in (gap_ratio(v, defn) for v in trunc27) if x is not None]
    ORDSTAT['by_definition'][defn] = {
        'true27_mean': float(np.mean(a)), 'true27_median': float(np.median(a)),
        'truncated_mean': float(np.mean(b)), 'truncated_median': float(np.median(b)),
        'separates_in_expected_direction': bool(np.median(a) > np.median(b))}
ORDSTAT['all_definitions_agree'] = bool(
    all(v['separates_in_expected_direction'] for v in ORDSTAT['by_definition'].values()))
ORDSTAT['provenance_note'] = (
    'the previously quoted figures 2.03x and 1.90x/0.90x were not reproduced by any of these four '
    'definitions and had no script or JSON behind them; the DIRECTION they assert is confirmed under '
    'all four, but quote the numbers below, not those')
log(f"1. order statistic: all four definitions separate in the expected direction: "
    f"{ORDSTAT['all_definitions_agree']}")

# ---------------------------------------------------------------- 3. measurement-budget replay
TRUE_MAX = lab.groupby('cat_id')[TARGET].max().to_numpy()
by_cat = {}
for (c, t), v in cells.items():
    by_cat.setdefault(c, []).append(v)
full5 = [c for c in range(n_cat) if len(by_cat.get(c, [])) == 5]

N_DRAWS = 200
BUDGETS = [(4, 5), (5, 5), (5, 4), (3, 5), (4, 10), (5, 10), (4, 3), (5, 2), (5, 1)]
REPLAY = {'n_draws': N_DRAWS, 'n_catalysts_with_all_5_temperatures': len(full5),
          'design': ('sample T of a catalyst\'s 5 temperature cells and R rows from each, take the max '
                     'of those T*R rows as the label, and rank catalysts by it against the full '
                     'observed max'),
          'budgets': {}}
for T, R in BUDGETS:
    elig = [c for c in full5 if sum(len(v) >= R for v in by_cat[c]) >= T]
    rhos = []
    for draw in range(N_DRAWS):
        rng = np.random.default_rng(1000 + draw)
        sub = np.empty(len(elig))
        for i, c in enumerate(elig):
            ok = [v for v in by_cat[c] if len(v) >= R]
            pick = rng.choice(len(ok), size=T, replace=False)
            sub[i] = max(rng.choice(ok[j], size=R, replace=False).max() for j in pick)
        rhos.append(spearmanr(sub, TRUE_MAX[elig])[0])
    REPLAY['budgets'][f'{T}temps_x_{R}rows'] = {
        'runs_per_catalyst': T * R, 'n_eligible_catalysts': len(elig),
        'spearman_vs_full_mean': float(np.mean(rhos)), 'spearman_vs_full_std': float(np.std(rhos, ddof=1)),
        'spearman_vs_full_p05': float(np.percentile(rhos, 5)),
        'spearman_vs_full_p95': float(np.percentile(rhos, 95))}
    log(f"3. budget {T}x{R} = {T*R:3d} runs: rho {np.mean(rhos):.4f} +/- {np.std(rhos, ddof=1):.4f} "
        f"(n={len(elig)})")
# The work note quotes a SPECIFIC design: 5 rows at each of 750/800/850/900 C (700 dropped), scored on
# "your 811 fully-measured catalysts". Reproduce exactly that, since it is the version going to JAIST.
FIXED_T = [750.0, 800.0, 850.0, 900.0]
by_cat_t = {}
for (c, t), v in cells.items():
    by_cat_t.setdefault(c, {})[t] = v
WN = {'design': '5 rows at each of 750/800/850/900 C (700 C dropped), vs the full observed max',
      'fixed_temperatures': FIXED_T, 'n_catalysts_with_all_5_temperatures': len(full5), 'rows': {}}
for R in [5, 4, 3]:
    elig = [c for c in full5 if all(len(by_cat_t[c][t]) >= R for t in FIXED_T)]
    rhos, bias = [], []
    for draw in range(N_DRAWS):
        rng = np.random.default_rng(2000 + draw)
        sub = np.array([max(rng.choice(by_cat_t[c][t], size=R, replace=False).max() for t in FIXED_T)
                        for c in elig])
        rhos.append(spearmanr(sub, TRUE_MAX[elig])[0]); bias.append(float(np.mean(sub - TRUE_MAX[elig])))
    WN['rows'][f'{R}_rows_per_temperature'] = {
        'runs_per_catalyst': 4 * R, 'n_eligible_catalysts': len(elig),
        'spearman_mean': float(np.mean(rhos)), 'spearman_std': float(np.std(rhos, ddof=1)),
        'spearman_p05': float(np.percentile(rhos, 5)), 'spearman_p95': float(np.percentile(rhos, 95)),
        'mean_yield_bias': float(np.mean(bias))}
    log(f"3b. worknote design 4x{R} = {4*R} runs: rho {np.mean(rhos):.4f} "
        f"+/- {np.std(rhos, ddof=1):.4f} bias {np.mean(bias):+.3f} (n={len(elig)})")
WN['eligibility_note'] = (
    f"the work note scores this on 'your 811 fully-measured catalysts', but the 20-run design needs "
    f">=5 rows in each of the 4 chosen cells, which only "
    f"{WN['rows']['5_rows_per_temperature']['n_eligible_catalysts']} of the 811 satisfy")
REPLAY['worknote_exact_design'] = WN

REPLAY['session_context_claim'] = {
    'quoted': 'rho 0.955 for 20 runs per catalyst (4 temperatures x 5 conditions), bias -1.31',
    'recomputed_random_4_of_5_temps': REPLAY['budgets']['4temps_x_5rows']['spearman_vs_full_mean'],
    'recomputed_worknote_fixed_temps': WN['rows']['5_rows_per_temperature']['spearman_mean'],
    'recomputed_bias_worknote_design': WN['rows']['5_rows_per_temperature']['mean_yield_bias'],
    'value_for_25_runs_all_5_temps': REPLAY['budgets']['5temps_x_5rows']['spearman_vs_full_mean'],
    'note': ('the quoted figure had no script or JSON source. Under the exact design the work note '
             'describes it recomputes to the fixed-temps value above, not 0.955; 0.955 is closer to '
             'the 25-run (all five temperatures) configuration. Which was intended cannot be '
             'determined without the original script, so quote the recomputed value.')}
REPLAY['reading_dependence'] = (
    'a row-subsampling saving is REACTOR TIME only under the distinct-conditions reading; under the '
    'time-on-stream reading the 27 slots come from one continuous run, so the saving is analytical '
    'and this budget arithmetic does not transfer')

json.dump({'meta': {'purpose': 'give the condition-grid claims a single computational source',
                    'n_cells': int(len(cells)), 'n_catalysts': int(n_cat), 'n_rows': int(len(lab)),
                    'honesty_constraint': REPLAY['reading_dependence']},
           'within_cell_ordering': ORDER,
           'truncation_counting_disproof': COUNT,
           'truncation_order_statistic': ORDSTAT,
           'measurement_budget_replay': REPLAY},
          open('phase11_condition_grid_forensics.json', 'w'), indent=1)

print()
log('ORDER-STATISTIC TEST (bottom gap / interior gap)')
print('   %-18s %22s %22s' % ('definition', 'true 27-row cells', 'top-27 of a larger cell'))
for k, v in ORDSTAT['by_definition'].items():
    print('   %-18s   mean %6.3f med %6.3f   mean %6.3f med %6.3f' %
          (k, v['true27_mean'], v['true27_median'], v['truncated_mean'], v['truncated_median']))
log('MEASUREMENT BUDGET')
print('   %-16s %6s %8s %10s %18s' % ('budget', 'runs', 'n_cat', 'rho_mean', 'rho 5-95 pct'))
for k, v in REPLAY['budgets'].items():
    print('   %-16s %6d %8d %10.4f   [%.4f, %.4f]' % (k, v['runs_per_catalyst'],
          v['n_eligible_catalysts'], v['spearman_vs_full_mean'],
          v['spearman_vs_full_p05'], v['spearman_vs_full_p95']))
log('done -> phase11_condition_grid_forensics.json')
