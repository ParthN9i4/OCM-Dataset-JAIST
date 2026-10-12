"""
phase11_condition_baseline.py -- what do the gas-flow conditions buy, measured honestly?

SUPERSEDES the committed version of this script and its JSON (commit a9fffc7). That version made
three claims that are withdrawn here:
  1. The 0.595 "RMSE floor with conditions" (grouping by catalyst, temperature and flows). It came
     entirely from the 1,906 duplicated slots of the 28 doubled catalysts, which are two runs with a
     run-level offset, not repeats. No floor for that grouping is computed here.
  2. The "+0.0215 screening gain". It was the max of predictions over each catalyst's MEASURED rows,
     which an unseen catalyst does not have. It is kept below only as a labelled diagnostic, (i).
  3. "Not seed noise", argued from the seed-to-seed SD. Seed SD is fold-shuffle spread on the same
     917 catalysts. It says nothing about other catalysts. Differences are now judged only by a
     paired catalyst bootstrap.

QUESTION. What do the gas-flow conditions buy, (a) for row-level prediction and (b) for ranking
catalysts?

ARMS. Catalyst-grouped 5-fold CV (catalyst_folds), seeds 0, 1, 2, 7, 13, TUNED LightGBM from
grouped_tuning.json (300 trees), n_jobs=2, StandardScaler fit on the training fold only.
  A         row-level; 65 element cols + temperature (66 cols).
  C         row-level; A + raw flows Ar, CH4, O2 (69 cols).
  Cp        row-level; A + design axes total flow, Ar fraction, CH4:O2 (69 cols).
  B_max     one row per catalyst; 65 element cols; label = observed max.
  B_median  as B_max; label = max over temperatures of the per-(catalyst, temperature) median.
            This is the project's adopted label. B_max and B_median reproduce phase9.

ROW RMSE. Pooled RMSE over all rows, per seed. It splits exactly into a cell part and a within part,
with a cell = (catalyst, temperature): per row, cell part = (mean y - mean p)^2 and within part =
((y - mean y) - (p - mean p))^2. A predicts one value per cell, so its within part equals the
within-cell SD of yield (the 1.757 figure).

SCREENING. Row-level arms need a rule that turns row predictions into one score per catalyst.
Truth is TRUE_MAX, the max over all of a catalyst's rows, except in (iii).
  (i)   measured_rows_max. Max over the catalyst's measured rows. NON-DEPLOYABLE DIAGNOSTIC: an unseen
        catalyst has no measured set, and the measured set reflects which runs passed mass balance.
  (ii)  full_grid_max. The fold model predicts all 135 slots for each validation catalyst; max over
        135. Deployable. But TRUE_MAX is truncated for catalysts with excluded slots and the
        prediction is not, so this estimator is biased against the condition-aware arms.
  (iii) fixed_slot_set_max. HEADLINE. A fixed slot set S, chosen by coverage only, before any model
        output was seen. Prediction = max of grid predictions over S. truth_S = max of measured
        yields over the rows in S; a doubled slot contributes both rows (max over rows, the same
        convention as TRUE_MAX). Scored only on catalysts with every slot of S measured, so the
        truth is not truncated. Deployable and fair to truth. B arms have no slot resolution, so
        their catalyst prediction is scored against truth_S directly. B was trained on a label over
        all slots, not on truth_S, so C - B and A - B under (iii) lean towards the row-level arms.
  Choice rule for S, in this order; the first with >= 300 complete catalysts is used:
    S1 = the 45 slots with CH4:O2 = 2; S2 = the 18 of those at 800 and 850 C; S3 = the 9 at 850 C.

POPULATIONS. all_917; EE_771 = equal effort (>= 20 rows in some (catalyst, temperature) cell);
ND_889 = without the 28 doubled catalysts. For (iii) each is intersected with the complete-S set.

DECISION RULES (fixed before the results).
  - An arm difference is "detected" only if its 95% paired catalyst bootstrap CI (3000 resamples,
    seed 7, seed-averaged out-of-fold predictions) excludes zero. Otherwise: "no detectable change".
  - The row RMSE difference uses the same catalyst resampling, with per-catalyst SSE and row counts.
  - Seed-to-seed spread is never used as a test. Per-seed values are stored for checking only.
  - The capacity check is one seed and descriptive. It has no test.

CAPACITY CHECK (seed 0, arm C). TUNED vs 1500 trees / 255 leaves vs 5000 / 1023 (dropped if it would
run past 25 min); other params as TUNED. Also TUNED with subsample_freq=1: LightGBM 4.x ignores
subsample unless subsample_freq > 0, so TUNED's subsample has had no effect. A one-fold run with
subsample=1.0 checks that. TUNED is not changed anywhere else.

Every stored number carries its population, its estimator, and whether it is a per-seed mean or was
computed on seed-averaged predictions.

Outputs: phase11_condition_baseline.json; phase11_predictions.npz (float32, reused by phase13).
Usage:   python3 phase11_condition_baseline.py [--dev]
         --dev = seed 0 only, 300 resamples, no capacity check, no files written.
"""
import sys, warnings, json, time
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from ocm_eval import (load_with_conditions, lgb_params, cat_metrics, TARGET, COND_COLS,
                      catalyst_folds, equal_effort_mask, design_axes, slot_keys, condition_grid,
                      doubled_catalyst_mask, paired_catalyst_bootstrap, MIN_CELL)

DEV = '--dev' in sys.argv
SEEDS = [0] if DEV else [0, 1, 2, 7, 13]
N_BOOT = 300 if DEV else 3000
BOOT_SEED = 7
N_JOBS = 2
CAP_LIMIT_MIN = 25.0
OUT_JSON, OUT_NPZ = 'phase11_condition_baseline.json', 'phase11_predictions.npz'

t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

# ------------------------------------------------------------------------------- data
df, feat_all, groups, n_cat = load_with_conditions()
TUNED = json.load(open('grouped_tuning.json'))['confirmation']['tuned']['overrides']
y = df[TARGET].values.astype(float)
N = len(y)
EL = [c for c in feat_all if c not in COND_COLS + ['Temperature_C']]
AX = ['total_flow', 'ar_fraction', 'ch4_o2']
ax = design_axes(df)
for c in AX:
    df[c] = ax[c].values
FEATS = {'A': EL + ['Temperature_C'], 'C': list(feat_all), 'Cp': EL + ['Temperature_C'] + AX}
assert len(EL) == 65 and len(FEATS['A']) == 66 and len(FEATS['C']) == 69 and len(FEATS['Cp']) == 69
ROW_ARMS, B_ARMS = ['A', 'C', 'Cp'], ['B_max', 'B_median']

POPS = {'all_917': np.ones(n_cat, bool),
        'EE_771': equal_effort_mask(groups, df.Temperature_C.values, n_cat, MIN_CELL),
        'ND_889': ~doubled_catalyst_mask(df, groups, n_cat)}
DBL = ~POPS['ND_889']
assert POPS['EE_771'].sum() == 771 and POPS['ND_889'].sum() == 889

COMP = df.groupby(groups)[EL].first().reindex(range(n_cat)).values.astype(float)
TRUE_MAX = pd.Series(y).groupby(groups).max().reindex(range(n_cat)).values
Y_MEDIAN = (pd.DataFrame({'c': groups, 't': df.Temperature_C.values, 'y': y})
            .groupby(['c', 't']).y.median().groupby(level=0).max().reindex(range(n_cat)).values)
B_LABEL = {'B_max': TRUE_MAX, 'B_median': Y_MEDIAN}

# (catalyst, temperature) cells for the row-RMSE split
CELL = pd.factorize(pd.MultiIndex.from_arrays([groups, df.Temperature_C.values]))[0]

# ------------------------------------------------------------------------------- slots
GRID = condition_grid(df)
N_SLOT = len(GRID)
key2slot = {k: int(s) for k, s in zip(zip(GRID.Temperature_C.astype(float), GRID.total_flow,
                                          GRID.ar_fraction, GRID.ch4_o2), GRID.slot)}
ROW_SLOT = np.array([key2slot[k] for k in slot_keys(df)])
MEAS = np.zeros((n_cat, N_SLOT), bool); MEAS[groups, ROW_SLOT] = True
slot_n = pd.Series(groups * N_SLOT + ROW_SLOT).value_counts()
dup_slots = slot_n[slot_n >= 2]
# The only repeated (catalyst, temperature, flows) keys are the 1,906 slots measured twice, all in
# the 28 doubled catalysts. This is why the old 0.595 "floor" for that grouping is withdrawn.
assert N_SLOT == 135 and len(dup_slots) == 1906 and dup_slots.max() == 2
assert DBL[np.unique(dup_slots.index.values // N_SLOT)].all() and DBL.sum() == 28

# Fixed slot set S: chosen by coverage only, before any model is fit.
CH, TT = GRID.ch4_o2.values, GRID.Temperature_C.values.astype(float)
S_CANDIDATES = [('S1', 'all 45 slots with CH4:O2 = 2', CH == 2),
                ('S2', 'the 18 CH4:O2 = 2 slots at 800 and 850 C', (CH == 2) & np.isin(TT, [800, 850])),
                ('S3', 'the 9 CH4:O2 = 2 slots at 850 C', (CH == 2) & (TT == 850))]
S_COVERAGE, S_NAME = [], None
for name, desc, m in S_CANDIDATES:
    sl = np.where(m)[0]
    comp_ = MEAS[:, sl].all(1)
    S_COVERAGE.append({'name': name, 'description': desc, 'n_slots': int(len(sl)),
                       'n_complete_catalysts_all_917': int(comp_.sum()),
                       'n_complete_catalysts_EE_771': int((comp_ & POPS['EE_771']).sum()),
                       'n_complete_catalysts_ND_889': int((comp_ & POPS['ND_889']).sum())})
    if S_NAME is None and comp_.sum() >= 300:
        S_NAME, S_SLOTS, S_COMPLETE = name, sl, comp_
assert [c['n_slots'] for c in S_COVERAGE] == [45, 18, 9] and S_NAME is not None
in_S = np.isin(ROW_SLOT, S_SLOTS)
TRUTH_S = np.full(n_cat, -np.inf); np.maximum.at(TRUTH_S, groups[in_S], y[in_S])
TRUTH_S[~S_COMPLETE] = np.nan
log(f"{N:,} rows, {n_cat} catalysts; EE {POPS['EE_771'].sum()}, ND {POPS['ND_889'].sum()}; "
    f"slot set {S_NAME} ({len(S_SLOTS)} slots, {S_COMPLETE.sum()} complete catalysts); "
    f"coverage {[(c['name'], c['n_complete_catalysts_all_917']) for c in S_COVERAGE]}")

# ------------------------------------------------------------------------------- models
class _Deadline(Exception):
    pass


def _deadline_cb(deadline):
    def cb(env):
        if time.time() > deadline:
            raise _Deadline()
    cb.order = 99
    return cb


def grid_X(arm, cats):
    """Feature rows for all 135 slots of each catalyst in cats (catalyst-major, slot order)."""
    cond = GRID[{'A': ['Temperature_C'], 'C': ['Temperature_C'] + COND_COLS,
                 'Cp': ['Temperature_C'] + AX}[arm]].values.astype(float)
    return np.hstack([np.repeat(COMP[cats], N_SLOT, axis=0), np.tile(cond, (len(cats), 1))])


def run_row_arm(arm, seed, over=None, deadline=None, only_fold=None):
    """Row-level model, catalyst-grouped CV. Returns OOF row predictions, OOF grid predictions
    (n_cat x 135), and seconds per fold."""
    X = df[FEATS[arm]].values.astype(float)
    fc = catalyst_folds(seed, n_cat); f = fc[groups]
    params = lgb_params(seed, **{**TUNED, **(over or {}), 'n_jobs': N_JOBS})
    cbs = [_deadline_cb(deadline)] if deadline else None
    yp = np.full(N, np.nan); gp = np.full((n_cat, N_SLOT), np.nan); secs = []
    for k in (range(5) if only_fold is None else [only_fold]):
        ts = time.time()
        tr, va, cats = np.where(f != k)[0], np.where(f == k)[0], np.where(fc == k)[0]
        sc = StandardScaler().fit(X[tr])
        m = lgb.LGBMRegressor(**params).fit(sc.transform(X[tr]), y[tr], callbacks=cbs)
        yp[va] = m.predict(sc.transform(X[va]))
        gp[cats] = m.predict(sc.transform(grid_X(arm, cats))).reshape(len(cats), N_SLOT)
        secs.append(time.time() - ts)
    return yp, gp, secs


def run_B(label, seed):
    """Formulation B: one row per catalyst, element cols only. OOF catalyst predictions."""
    fc = catalyst_folds(seed, n_cat); yp = np.empty(n_cat)
    for k in range(5):
        tr, va = np.where(fc != k)[0], np.where(fc == k)[0]
        sc = StandardScaler().fit(COMP[tr])
        m = lgb.LGBMRegressor(**lgb_params(seed, **TUNED, n_jobs=N_JOBS)).fit(
            sc.transform(COMP[tr]), label[tr])
        yp[va] = m.predict(sc.transform(COMP[va]))
    return yp


# ------------------------------------------------------------------------------- metrics
def cat_max_rows(yp):
    out = np.full(n_cat, -np.inf); np.maximum.at(out, groups, yp); return out


def rmse_split(yp, rows=None):
    """Pooled row RMSE and its exact (catalyst, temperature) cell / within split."""
    r = y - yp
    s, n = np.bincount(CELL, r), np.bincount(CELL)
    mr = (s / n)[CELL]
    cellp, within = mr ** 2, (r - mr) ** 2
    sel = np.ones(N, bool) if rows is None else rows
    tot = float(np.sqrt(np.mean(r[sel] ** 2)))
    c, w = float(np.sqrt(np.mean(cellp[sel]))), float(np.sqrt(np.mean(within[sel])))
    assert abs(tot ** 2 - c ** 2 - w ** 2) < 1e-8 * max(1.0, tot ** 2)
    return tot, c, w


ROW_POPS = {'all_917': np.ones(N, bool), 'EE_771': POPS['EE_771'][groups],
            'ND_889': POPS['ND_889'][groups], 'doubled_28': DBL[groups]}
ESTIMATORS = ['measured_rows_max', 'full_grid_max', 'fixed_slot_set_max']
TRUTH = {'measured_rows_max': TRUE_MAX, 'full_grid_max': TRUE_MAX, 'fixed_slot_set_max': TRUTH_S}
EST_POPS = {'measured_rows_max': POPS, 'full_grid_max': POPS,
            'fixed_slot_set_max': {f'{k}_S_complete': v & S_COMPLETE for k, v in POPS.items()}}
METRICS = ['spearman', 'enrichment', 'precision_at20']


def cat_score(arm, est, yp=None, gp=None, bp=None):
    if arm in B_ARMS:
        return bp
    if est == 'measured_rows_max':
        return cat_max_rows(yp)
    return gp.max(1) if est == 'full_grid_max' else gp[:, S_SLOTS].max(1)


def scr(truth, pred, mask):
    m = cat_metrics(np.arange(int(mask.sum())), truth[mask], pred[mask])
    return {'spearman': float(m['spearman_max']), 'enrichment': float(m['enrichment_top10pct']),
            'precision_at20': float(m['precision_at20_vs_top10pct'])}


def rmse_diff_boot(pa, pb, cat_mask):
    """Paired catalyst bootstrap of RMSE(a) - RMSE(b) from per-catalyst SSE and row counts."""
    sa = np.bincount(groups, (y - pa) ** 2, minlength=n_cat)
    sb = np.bincount(groups, (y - pb) ** 2, minlength=n_cat)
    cnt = np.bincount(groups, minlength=n_cat).astype(float)
    idx = np.where(cat_mask)[0]; n = len(idx)
    rng = np.random.default_rng(BOOT_SEED); d = np.empty(N_BOOT)
    f = lambda i: np.sqrt(sa[i].sum() / cnt[i].sum()) - np.sqrt(sb[i].sum() / cnt[i].sum())
    for b in range(N_BOOT):
        d[b] = f(idx[rng.integers(0, n, n)])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {'n_catalysts': int(n), 'delta_point': float(f(idx)), 'delta_boot_mean': float(d.mean()),
            'ci95_low': float(lo), 'ci95_high': float(hi), 'clears_zero': bool(lo > 0 or hi < 0),
            'n_boot': N_BOOT, 'reading': 'detected' if (lo > 0 or hi < 0) else 'no detectable change'}


# ------------------------------------------------------------------------------- run
ROWP = {a: [] for a in ROW_ARMS}; GRIDP = {a: [] for a in ROW_ARMS}; BP = {a: [] for a in B_ARMS}
for s in SEEDS:
    for a in ROW_ARMS:
        yp, gp, secs = run_row_arm(a, s)
        ROWP[a].append(yp); GRIDP[a].append(gp)
        log(f"seed {s} arm {a}: row RMSE all_917 {rmse_split(yp)[0]:.4f} ({sum(secs):.0f}s)")
    for a in B_ARMS:
        BP[a].append(run_B(B_LABEL[a], s))
ROWA = {a: np.mean(ROWP[a], 0) for a in ROW_ARMS}
GRIDA = {a: np.mean(GRIDP[a], 0) for a in ROW_ARMS}
BA = {a: np.mean(BP[a], 0) for a in B_ARMS}
assert all(not np.isnan(ROWA[a]).any() and not np.isnan(GRIDA[a]).any() for a in ROW_ARMS)
log("all arms fitted")

# ---- variance decomposition, grouping by (catalyst, temperature) only
def within_share(rows):
    yy, cc = y[rows], CELL[rows]
    mu = (np.bincount(cc, yy, minlength=CELL.max() + 1) /
          np.maximum(np.bincount(cc, minlength=CELL.max() + 1), 1))[cc]
    wss, tss = ((yy - mu) ** 2).sum(), ((yy - yy.mean()) ** 2).sum()
    return {'n_rows': int(rows.sum()), 'n_cells': int(len(np.unique(cc))),
            'within_cell_share_pct': float(100 * wss / tss), 'rmse_floor': float(np.sqrt(wss / rows.sum()))}

VARDEC = {'grouping': '(catalyst, temperature)',
          'all_917_rows': within_share(ROW_POPS['all_917']),
          'ND_889_rows': within_share(ROW_POPS['ND_889']),
          'meaning': ('rmse_floor = row RMSE of a model that knows each cell mean exactly; not '
                      'attainable for an unseen catalyst'),
          'not_computed': ('no floor for grouping by (catalyst, temperature, flows); the old 0.595 came '
                           'entirely from the 1,906 duplicated slots of the 28 doubled catalysts')}

# ---- row RMSE
ROWRES = {}
for a in ROW_ARMS:
    blk = {}
    for pn, rows in ROW_POPS.items():
        per = [rmse_split(p, rows) for p in ROWP[a]]
        sa = rmse_split(ROWA[a], rows)
        blk[pn] = {'n_rows': int(rows.sum()),
                   'per_seed': {'rmse': [v[0] for v in per], 'cell_part': [v[1] for v in per],
                                'within_part': [v[2] for v in per]},
                   'per_seed_mean': {'rmse': float(np.mean([v[0] for v in per])),
                                     'cell_part': float(np.mean([v[1] for v in per])),
                                     'within_part': float(np.mean([v[2] for v in per]))},
                   'seed_averaged_predictions': {'rmse': sa[0], 'cell_part': sa[1], 'within_part': sa[2]}}
    ROWRES[a] = blk
RMSE_BOOT = {f'{a}_minus_{b}': {pn: rmse_diff_boot(ROWA[a], ROWA[b], m) for pn, m in POPS.items()}
             for a, b in [('A', 'C'), ('A', 'Cp')]}
log("row RMSE done")

# ---- screening
SCREEN = {}
for est in ESTIMATORS:
    SCREEN[est] = {}
    for a in ROW_ARMS + B_ARMS:
        if a in B_ARMS:
            per_preds, avg_pred = BP[a], BA[a]
        else:
            per_preds = [cat_score(a, est, yp, gp) for yp, gp in zip(ROWP[a], GRIDP[a])]
            avg_pred = cat_score(a, est, ROWA[a], GRIDA[a])
        SCREEN[est][a] = {}
        for pn, m in EST_POPS[est].items():
            per = [scr(TRUTH[est], p, m) for p in per_preds]
            SCREEN[est][a][pn] = {
                'n_catalysts': int(m.sum()),
                'per_seed': {k: [r[k] for r in per] for k in METRICS},
                'per_seed_mean': {k: float(np.mean([r[k] for r in per])) for k in METRICS},
                'seed_averaged_predictions': scr(TRUTH[est], avg_pred, m)}

PAIRS = [('C', 'A'), ('Cp', 'A'), ('C', 'B_max'), ('A', 'B_max'), ('C', 'B_median')]
BOOT = {}
for est in ESTIMATORS:
    BOOT[est] = {}
    pred = {a: (BA[a] if a in B_ARMS else cat_score(a, est, ROWA[a], GRIDA[a]))
            for a in ROW_ARMS + B_ARMS}
    for pn, m in EST_POPS[est].items():
        BOOT[est][pn] = {}
        for a, b in PAIRS:
            r = {}
            for met in ['spearman', 'enrichment']:
                bb = paired_catalyst_bootstrap(TRUTH[est], pred[a], pred[b], mask=m,
                                               n_boot=N_BOOT, seed=BOOT_SEED, metric=met)
                bb['reading'] = 'detected' if bb['clears_zero'] else 'no detectable change'
                r[met] = bb
            BOOT[est][pn][f'{a}_minus_{b}'] = r
    log(f"bootstraps done for {est}")

P9 = json.load(open('phase9_equal_effort_eval.json'))['results']
PHASE9_CHECK = {}
for a, key in [('B_max', 'real_model'), ('B_median', 'median_trained_model')]:
    for pn, p9 in [('all_917', 'all_917'), ('EE_771', 'equal_effort')]:
        mine = SCREEN['measured_rows_max'][a][pn]['seed_averaged_predictions']['spearman']
        PHASE9_CHECK[f'{a}_{pn}_spearman_seed_averaged'] = {
            'this_script': mine, 'phase9_json': P9[key][p9]['spearman'],
            'abs_diff': abs(mine - P9[key][p9]['spearman'])}

RES = {
    'meta': {
        'supersedes': ('committed phase11_condition_baseline.py/.json (commit a9fffc7). Withdrawn: the '
                       '0.595 floor (came entirely from the 1,906 duplicated slots of the 28 doubled '
                       'catalysts); the +0.0215 screening gain (it was the measured-rows diagnostic); '
                       'the "not seed noise" claim (it rested on seed SD, which is not a test).'),
        'protocol': 'catalyst-grouped 5-fold CV (catalyst_folds), TUNED LightGBM, n_jobs=2, '
                    'train-fold StandardScaler',
        'seeds': SEEDS, 'n_rows': int(N), 'n_catalysts': int(n_cat), 'dev_run': DEV,
        'tuned_params': TUNED, 'lightgbm_version': lgb.__version__,
        'features': {a: len(v) for a, v in FEATS.items()} | {b: len(EL) for b in B_ARMS},
        'populations': {k: int(v.sum()) for k, v in POPS.items()},
        'per_seed_vs_seed_averaged': ('per_seed = one value per seed; per_seed_mean = mean of those; '
                                      'seed_averaged_predictions = predictions averaged over seeds, '
                                      'then scored. Differences between arms use only the latter.'),
        'B_arms_note': 'B arms give one catalyst prediction; it is the same under every estimator.',
        'decision_rule': ('difference detected only if the 95% paired catalyst bootstrap CI excludes '
                          'zero (n_boot=3000, seed 7, seed-averaged OOF predictions); otherwise "no '
                          'detectable change". Seed SD is never used as a test.')},
    'variance_decomposition': VARDEC,
    'fixed_slot_set': {
        'rule': ('by coverage only, before any model output: first of S1, S2, S3 with >= 300 '
                 'catalysts that have every slot measured'),
        'candidates': S_COVERAGE, 'chosen': S_NAME, 'slots': S_SLOTS.tolist(),
        'truth_convention': ('truth_S = max over the measured rows in S; a doubled slot contributes '
                             'both rows (max over rows, as TRUE_MAX)')},
    'estimators': {
        'measured_rows_max': 'NON-DEPLOYABLE DIAGNOSTIC: max over the catalyst\'s measured rows; truth TRUE_MAX',
        'full_grid_max': ('deployable: max over all 135 slots; truth TRUE_MAX is truncated by excluded '
                          'slots, so biased against condition-aware arms'),
        'fixed_slot_set_max': 'HEADLINE: max over S; truth_S; complete-S catalysts only'},
    'row_rmse': ROWRES,
    'row_rmse_bootstrap_seed_averaged': RMSE_BOOT,
    'screening': SCREEN,
    'screening_bootstrap_seed_averaged': BOOT,
    'phase9_reproduction_check': PHASE9_CHECK,
}


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def write_outputs():
    if DEV:
        return
    RES['runtime_minutes'] = (time.time() - t0) / 60
    json.dump(RES, open(OUT_JSON, 'w'), indent=1, default=_json_default)


if not DEV:
    f32 = lambda v: np.asarray(v, np.float32)
    np.savez_compressed(
        OUT_NPZ,
        **{f'rows_{a}': f32(ROWA[a]) for a in ROW_ARMS},
        **{f'grid_{a}': f32(GRIDA[a]) for a in ROW_ARMS},
        **{f'slot_{c}': f32(GRID[c].values) for c in ['Temperature_C'] + AX + COND_COLS},
        **{f'cat_{a}': f32(BA[a]) for a in B_ARMS},
        true_max=f32(TRUE_MAX), grid_C_per_seed=f32(np.stack(GRIDP['C'])),
        seeds=np.array(SEEDS, np.int32), row_catalyst=groups.astype(np.int32),
        row_slot=ROW_SLOT.astype(np.int32), fixed_slot_set=S_SLOTS.astype(np.int32))
    RES['npz_contents'] = ('rows_{A,C,Cp}: seed-averaged OOF row predictions in df row order; '
                           'grid_{A,C,Cp}: seed-averaged OOF predictions (917, 135) in condition_grid '
                           'slot order; slot_*: the slot table; cat_{B_max,B_median}: seed-averaged B '
                           'predictions; true_max; grid_C_per_seed (5, 917, 135); row_catalyst, '
                           'row_slot, fixed_slot_set: int32 index arrays')
write_outputs()
log("main results written" if not DEV else "dev run: nothing written")

# ------------------------------------------------------------------------------- printed summary
R = lambda a, pn='all_917': ROWRES[a][pn]['per_seed_mean']
SP = lambda est, a, pn: SCREEN[est][a][pn]['seed_averaged_predictions']
print("\n=========== phase11: what the gas-flow conditions buy ===========")
print(f"Row RMSE, all 917 catalysts' rows, per-seed mean of {len(SEEDS)} seed(s); split by (catalyst, temperature) cell.")
print(f"Screening: Spearman / enrichment on seed-averaged predictions, all 917 (iii: the "
      f"{S_COMPLETE.sum()} complete-{S_NAME} catalysts).")
print(f"{'arm':9s} {'RMSE':>6s} {'cell':>6s} {'within':>6s} | {'(i) measured rows*':>18s} "
      f"{'(ii) full grid':>15s} {'(iii) fixed set':>16s}")
for a in ROW_ARMS + B_ARMS:
    r = (f"{R(a)['rmse']:6.3f} {R(a)['cell_part']:6.3f} {R(a)['within_part']:6.3f}"
         if a in ROW_ARMS else f"{'-':>6s} {'-':>6s} {'-':>6s}")
    cols = [SP(e, a, pn) for e, pn in [('measured_rows_max', 'all_917'), ('full_grid_max', 'all_917'),
                                        ('fixed_slot_set_max', 'all_917_S_complete')]]
    print(f"{a:9s} {r} | " + " ".join(f"{c['spearman']:9.4f} {c['enrichment']:5.2f}x" for c in cols))
print("* (i) is a non-deployable diagnostic.")
for k, pn in [('A_minus_C', 'all_917'), ('A_minus_Cp', 'all_917')]:
    b = RMSE_BOOT[k][pn]
    print(f"Row RMSE {k}, all 917, seed-averaged: {b['delta_point']:+.4f} "
          f"[{b['ci95_low']:+.4f}, {b['ci95_high']:+.4f}] {b['reading']}")
print("Paired catalyst bootstrap, seed-averaged predictions, delta [95% CI]  Spearman | enrichment:")
for est, pn, pair in [('fixed_slot_set_max', 'all_917_S_complete', 'C_minus_A'),
                      ('fixed_slot_set_max', 'EE_771_S_complete', 'C_minus_A'),
                      ('fixed_slot_set_max', 'ND_889_S_complete', 'C_minus_A'),
                      ('fixed_slot_set_max', 'all_917_S_complete', 'C_minus_B_median'),
                      ('full_grid_max', 'all_917', 'C_minus_A'),
                      ('measured_rows_max', 'all_917', 'C_minus_A')]:
    s_, e_ = BOOT[est][pn][pair]['spearman'], BOOT[est][pn][pair]['enrichment']
    print(f"  {est:18s} {pn:19s} {pair:16s} {s_['delta_point']:+.4f} [{s_['ci95_low']:+.4f}, "
          f"{s_['ci95_high']:+.4f}] | {e_['delta_point']:+.2f} [{e_['ci95_low']:+.2f}, {e_['ci95_high']:+.2f}]")
print("phase9 check (B, seed-averaged Spearman): " + ", ".join(
    f"{k.replace('_spearman_seed_averaged', '')} {v['this_script']:.4f} vs {v['phase9_json']:.4f}"
    for k, v in PHASE9_CHECK.items()))

if DEV:
    log("dev run done")
    sys.exit(0)

# ------------------------------------------------------------------------------- capacity check
def cap_metrics(yp, gp):
    t = rmse_split(yp)
    mr = scr(TRUE_MAX, cat_max_rows(yp), POPS['all_917'])
    fg = scr(TRUE_MAX, gp.max(1), POPS['all_917'])
    fs = scr(TRUTH_S, gp[:, S_SLOTS].max(1), S_COMPLETE)
    return {'row_rmse_all_917': t[0], 'row_rmse_cell_part': t[1], 'row_rmse_within_part': t[2],
            'measured_rows_max_all_917_spearman': mr['spearman'],
            'measured_rows_max_all_917_enrichment': mr['enrichment'],
            'full_grid_max_all_917_spearman': fg['spearman'],
            'full_grid_max_all_917_enrichment': fg['enrichment'],
            'fixed_slot_set_max_all_917_S_complete_spearman': fs['spearman'],
            'fixed_slot_set_max_all_917_S_complete_enrichment': fs['enrichment']}


i0 = SEEDS.index(0)
yp_t, gp_t = ROWP['C'][i0], GRIDP['C'][i0]
CAP = {'scope': 'seed 0 only, arm C, all metrics per single seed; descriptive, no test',
       'configs': {f"TUNED_trees{TUNED['n_estimators']}_leaves{TUNED['num_leaves']}": {
           'params_changed': {}, **cap_metrics(yp_t, gp_t),
           'source': 'the main run, seed 0 (identical config)'}}}
for nm, over in [('trees1500_leaves255', {'n_estimators': 1500, 'num_leaves': 255}),
                 ('trees5000_leaves1023', {'n_estimators': 5000, 'num_leaves': 1023})]:
    ts = time.time()
    try:
        yp_c, gp_c, secs = run_row_arm('C', 0, over, deadline=ts + CAP_LIMIT_MIN * 60)
        CAP['configs'][nm] = {'params_changed': over, **cap_metrics(yp_c, gp_c),
                              'runtime_min': (time.time() - ts) / 60}
    except _Deadline:
        CAP['configs'][nm] = {'params_changed': over, 'dropped': True,
                              'reason': f'did not finish 5 folds within {CAP_LIMIT_MIN:.0f} min'}
    log(f"capacity {nm}: {CAP['configs'][nm].get('row_rmse_all_917', 'dropped')}")

yp_f, gp_f, _ = run_row_arm('C', 0, {'subsample_freq': 1})
yp_s1, _, _ = run_row_arm('C', 0, {'subsample': 1.0}, only_fold=0)
f0 = catalyst_folds(0, n_cat)[groups] == 0
CAP['configs']['TUNED_plus_subsample_freq1'] = {
    'params_changed': {'subsample_freq': 1}, **cap_metrics(yp_f, gp_f),
    'oof_row_prediction_change_vs_TUNED': {
        'max_abs_diff': float(np.abs(yp_f - yp_t).max()),
        'rms_diff': float(np.sqrt(np.mean((yp_f - yp_t) ** 2))),
        'pearson': float(np.corrcoef(yp_f, yp_t)[0, 1])}}
CAP['subsample_inert_check'] = {
    'what': 'fold 0 of seed 0, arm C: TUNED (subsample=0.6, no subsample_freq) vs subsample=1.0',
    'max_abs_diff_fold0_rows': float(np.abs(yp_s1[f0] - yp_t[f0]).max()),
    'identical': bool(np.array_equal(yp_s1[f0], yp_t[f0]))}
RES['capacity_check'] = CAP
write_outputs()
log("capacity check written")

print("\nCapacity check, seed 0, arm C (one seed, descriptive): row RMSE all_917 | measured-rows "
      "Spearman, enrich | fixed-set Spearman, enrich")
for nm, c in CAP['configs'].items():
    if c.get('dropped'):
        print(f"  {nm:30s} dropped: {c['reason']}")
        continue
    print(f"  {nm:30s} {c['row_rmse_all_917']:.4f} | {c['measured_rows_max_all_917_spearman']:.4f} "
          f"{c['measured_rows_max_all_917_enrichment']:.2f}x | "
          f"{c['fixed_slot_set_max_all_917_S_complete_spearman']:.4f} "
          f"{c['fixed_slot_set_max_all_917_S_complete_enrichment']:.2f}x")
ch = CAP['configs']['TUNED_plus_subsample_freq1']['oof_row_prediction_change_vs_TUNED']
print(f"  subsample_freq=1 moves OOF row predictions by up to {ch['max_abs_diff']:.3f} "
      f"(RMS {ch['rms_diff']:.3f}); subsample=1.0 without freq identical on fold 0: "
      f"{CAP['subsample_inert_check']['identical']}")
log("done")
