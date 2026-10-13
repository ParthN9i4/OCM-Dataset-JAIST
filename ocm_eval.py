"""
ocm_eval.py — shared evaluation infrastructure for the OCM literature-transfer project.

Single implementation of: data loading, catalyst-identity grouping (917 groups), grouped/row fold
construction, per-fold train-only scaler + DRST domain classifier, Stage-1 label treatments, and
row-level + catalyst-level metrics. Extracted from taniike_validation.py so every experiment imports
one implementation instead of re-copying it.

PROTOCOL RULES (established after Prof. Taniike's validation round — see SESSION_CONTEXT.md §7):
- Catalyst-grouped CV is the DEFAULT protocol. Row-level CV exists only for comparison with
  historical numbers and must always be labeled as such.
- Anything fit on data (scaler, DRST classifier, Stage 1, Stage 2) sees training-fold data only.
- Catalyst-level metrics (max-yield Spearman, enrichment@10%, precision@20) are primary for the
  screening objective; row RMSE is secondary and must be labeled as such. Each (catalyst,
  temperature) cell holds up to 27 measurements taken under 27 DIFFERENT gas-flow conditions: an
  exact 3x3x3 design of total flow (10/15/20, units as recorded in the file) x Ar fraction (0.70/0.40/0.15) x CH4:O2
  (2/4/6), so 5 x 27 = 135 conditions per catalyst. Rows with poor mass balance were excluded by
  the lab, so cells hold <= 27 distinct conditions. DATA_PATH does not record the conditions;
  load_with_conditions() restores them. With composition and temperature only, 19.9% of yield
  variance sits within cells, flooring row-level RMSE at 1.757 (1.651 without the 28 catalysts
  described next). That floor assumes the model already knows each cell mean, so it is not
  attainable for an unseen catalyst.
- 28 catalysts carry a second run: 1,906 (catalyst, temperature, condition) slots appear twice.
  In a cell one run is higher at every condition (median ratio ~2.3x), so the pairs are two runs
  with a run-level offset, not repeat measurements, and must not be used as a noise estimate.
  doubled_catalyst_mask() identifies them; report screening with and without them.
- Significance of a difference between two models is a PAIRED CATALYST BOOTSTRAP
  (paired_catalyst_bootstrap), never the seed-to-seed SD on the same catalysts.

Usage:
    from ocm_eval import Data, grouped_folds, row_folds, run_fold, run_cv, lgb_params, xgb_params
    d = Data.load()
    res = run_cv(d, grouped_folds(d, seed=0), seed=0, model='baseline')
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.stats import rankdata, spearmanr
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb

TARGET = 'Y(C2), %'
DATA_PATH = 'OCM_lab_data_and_literature_datal.csv'


# ---------------------------------------------------------------------------- data
@dataclass
class Data:
    X_lab: np.ndarray; y_lab: np.ndarray
    X_lit: np.ndarray; y_lit: np.ndarray
    groups: np.ndarray            # catalyst id (int code) per lab row; 917 unique
    n_cat: int
    features: list
    dl_lab: pd.DataFrame = field(repr=False, default=None)
    dl_lit: pd.DataFrame = field(repr=False, default=None)

    @classmethod
    def load(cls, path=DATA_PATH):
        df = pd.read_csv(path)
        dl_lab = df[df.year == 2025].reset_index(drop=True)
        dl_lit = df[df.year <= 2019].reset_index(drop=True)
        el = [c for c in df.columns if c not in ['Preparation', 'Temperature_C', TARGET, 'year']]
        le = LabelEncoder().fit(df.Preparation)
        dl_lab['prep_enc'] = le.transform(dl_lab.Preparation)
        dl_lit['prep_enc'] = le.transform(dl_lit.Preparation)
        feats = ['Temperature_C', 'prep_enc'] + el
        # catalyst identity = preparation + full element-loading vector (temperature is a condition)
        cat_str = dl_lab[['Preparation'] + el].astype(str).agg('|'.join, axis=1)
        groups, uniques = pd.factorize(cat_str)
        return cls(X_lab=dl_lab[feats].values.astype(float), y_lab=dl_lab[TARGET].values,
                   X_lit=dl_lit[feats].values.astype(float), y_lit=dl_lit[TARGET].values,
                   groups=groups, n_cat=len(uniques), features=feats,
                   dl_lab=dl_lab, dl_lit=dl_lit)


# ------------------------------------------------------------------ conditions dataset
CONDITIONS_PATH = 'OCM_data_with_conditions.csv'
NAME_MAP_PATH = 'catalyst_name_to_composition.json'
COND_COLS = ['Ar_flow', 'CH4_flow', 'O2_flow']


def load_with_conditions(cond_path=CONDITIONS_PATH, map_path=NAME_MAP_PATH, path=DATA_PATH):
    """Lab data with the gas-flow conditions restored (supplied by Prof. Taniike, Sept 2026).

    The original export dropped the reaction conditions to make the lab data compatible with the
    literature set. This rejoins them. Each catalyst was run at 5 temperatures x 27 gas-flow
    settings = 135 conditions; rows with poor mass balance were excluded, so cells hold <= 27.

    The flow columns make within-cell variation learnable: grouped-CV row RMSE improves 2.897 ->
    2.661 (phase11_condition_baseline.py). They do NOT give a demonstrated catalyst-screening
    gain; see phase11 for how screening must be scored with conditions.

    The join is by catalyst, not by row: the two files hold the same 89,074 yields in a different
    order, and 672 rows share a (temperature, yield) key with another row (607 beyond the first
    row of each key; 551 of the 672 are zero yields), so row matching is ambiguous. Each catalyst's (temperature, yield) multiset is the fingerprint instead. The map is
    built and checked by build_name_map.py, and the multiset match is re-asserted on every load.

    Returns (df, feat_cols, groups, n_cat) where feat_cols is composition + temperature + the three
    flow columns, and groups matches the catalyst coding used by Data.load() elsewhere.
    """
    import json as _json
    cond = pd.read_csv(cond_path)
    name_map = _json.load(open(map_path))
    df_all = pd.read_csv(path)
    lab = df_all[df_all.year == 2025].reset_index(drop=True)
    el = [c for c in df_all.columns if c not in ['Preparation', 'Temperature_C', TARGET, 'year']]

    comp_key = lab[['Preparation'] + el].astype(str).agg('|'.join, axis=1)
    comp_lookup = lab.assign(_k=comp_key).drop_duplicates('_k').set_index('_k')

    cond = cond.assign(_k=cond.Name.map(name_map))
    if cond._k.isna().any():
        raise ValueError('unmapped catalyst names in the conditions file')

    comp = comp_lookup.loc[cond._k.values, el].reset_index(drop=True)
    out = pd.concat([comp, cond[['Temperature'] + COND_COLS + ['C2y', 'Name', '_k']]
                     .reset_index(drop=True)], axis=1)
    out = out.rename(columns={'Temperature': 'Temperature_C', 'C2y': TARGET})

    # every catalyst's (temperature, yield) multiset must agree between the two files
    a = (lab.assign(_k=comp_key)[['_k', 'Temperature_C', TARGET]]
         .sort_values(['_k', 'Temperature_C', TARGET]).reset_index(drop=True))
    b = (cond[['_k', 'Temperature', 'C2y']]
         .sort_values(['_k', 'Temperature', 'C2y']).reset_index(drop=True))
    if (len(a) != len(b) or not (a._k.values == b._k.values).all()
            or not np.allclose(a[['Temperature_C', TARGET]].values.astype(float),
                               b[['Temperature', 'C2y']].values.astype(float))):
        raise ValueError('conditions file and lab data disagree on a catalyst yield multiset')

    # catalyst codes aligned to the same ordering Data.load() produces
    _, uniques = pd.factorize(comp_key)
    code_of = {k: i for i, k in enumerate(uniques)}
    groups = out._k.map(code_of).values
    feat_cols = el + ['Temperature_C'] + COND_COLS
    return out, feat_cols, groups, len(uniques)


# ------------------------------------------------------------------ shared evaluation helpers
# phase8/9/10/11 each carry their own copy of the fold and bootstrap logic; scripts from phase11
# onward use these. catalyst_folds reproduces those copies exactly, so folds stay paired across
# scripts (and with formulation B) for the same seed.
MIN_CELL = 20   # equal-effort rule from phase9: >= MIN_CELL rows in at least one temperature cell


def catalyst_folds(seed, n_cat, k=5):
    """Fold index per catalyst code (0..n_cat-1)."""
    r = np.random.default_rng(seed); perm = r.permutation(n_cat); f = np.empty(n_cat, int)
    for i, ch in enumerate(np.array_split(perm, k)): f[ch] = i
    return f


def equal_effort_mask(groups, temps, n_cat, min_cell=MIN_CELL):
    """Boolean per catalyst: >= min_cell rows in at least one (catalyst, temperature) cell."""
    max_cell = (pd.DataFrame({'c': groups, 't': temps}).groupby(['c', 't']).size()
                .groupby(level=0).max().reindex(range(n_cat)).fillna(0).values)
    return max_cell >= min_cell


def design_axes(df):
    """Derived design axes of the 3x3x3 gas-flow grid, rounded (raw flows carry ~1e-9 noise)."""
    ar, ch4, o2 = (df[c].values.astype(float) for c in COND_COLS)
    tot = ar + ch4 + o2
    return pd.DataFrame({'total_flow': np.round(tot, 6), 'ar_fraction': np.round(ar / tot, 2),
                         'ch4_o2': np.round(ch4 / o2, 1)}, index=df.index)


def slot_keys(df):
    """(temperature, total_flow, ar_fraction, ch4_o2) per row: one of 135 condition slots."""
    ax = design_axes(df)
    return pd.MultiIndex.from_arrays([df['Temperature_C'].values.astype(float), ax.total_flow.values,
                                      ax.ar_fraction.values, ax.ch4_o2.values],
                                     names=['Temperature_C', 'total_flow', 'ar_fraction', 'ch4_o2'])


def condition_grid(df):
    """The 135 slots (5 temperatures x 27 gas-flow conditions) with one canonical raw flow triple
    each, plus a slot id. Returns a DataFrame with Temperature_C, COND_COLS, the design axes and
    'slot'."""
    keys = slot_keys(df)
    g = (df[['Temperature_C'] + COND_COLS].assign(_k=list(keys))
         .drop_duplicates('_k').sort_values('_k').reset_index(drop=True))
    ax = design_axes(g)
    g = pd.concat([g.drop(columns='_k'), ax], axis=1)
    g['slot'] = np.arange(len(g))
    return g


def doubled_catalyst_mask(df, groups, n_cat):
    """Boolean per catalyst: has at least one (temperature, condition) slot measured twice."""
    k = pd.DataFrame({'c': groups}).join(slot_keys(df).to_frame(index=False))
    dup = k.duplicated(keep=False)
    return np.isin(np.arange(n_cat), np.unique(groups[dup.values]))


def _enrichment(yt, yp, top_frac=0.10):
    K = max(1, int(round(top_frac * len(yt))))
    top_true = np.argsort(-yt, kind='stable')[:K]; top_pred = np.argsort(-yp, kind='stable')[:K]
    return len(np.intersect1d(top_true, top_pred)) / K / top_frac


def paired_catalyst_bootstrap(y_true, pred_a, pred_b, mask=None, n_boot=3000, seed=7,
                              metric='spearman'):
    """Paired bootstrap over CATALYSTS of metric(a) - metric(b), both against y_true.

    y_true, pred_a, pred_b are per-catalyst arrays (use seed-averaged out-of-fold predictions).
    With mask=None, n_boot=3000, seed=7 the resampling matches phase8_target_robustness.py.
    metric: 'spearman' or 'enrichment' (top-10% overlap / 0.10).
    """
    y_true, pred_a, pred_b = (np.asarray(v, float) for v in (y_true, pred_a, pred_b))
    idx_all = np.arange(len(y_true)) if mask is None else np.where(mask)[0]
    n = len(idx_all)
    rng = np.random.default_rng(seed)
    f = ((lambda t, p: spearmanr(t, p)[0]) if metric == 'spearman' else _enrichment)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        i = idx_all[rng.integers(0, n, n)]
        deltas[b] = f(y_true[i], pred_a[i]) - f(y_true[i], pred_b[i])
    point = f(y_true[idx_all], pred_a[idx_all]) - f(y_true[idx_all], pred_b[idx_all])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {'metric': metric, 'n_catalysts': int(n), 'delta_point': float(point),
            'delta_boot_mean': float(deltas.mean()), 'ci95_low': float(lo), 'ci95_high': float(hi),
            'p_a_better': float((deltas > 0).mean()), 'clears_zero': bool(lo > 0 or hi < 0),
            'n_boot': int(n_boot)}


# ---------------------------------------------------------------------------- params
def lgb_params(seed=42, **over):
    p = dict(n_estimators=500, learning_rate=0.05, num_leaves=63, max_depth=7, min_child_samples=20,
             subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
             random_state=seed, n_jobs=-1, verbosity=-1)
    p.update(over); return p


def xgb_params(seed=42, **over):
    p = dict(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
             reg_alpha=0.1, reg_lambda=1.0, random_state=seed, verbosity=0, n_jobs=-1)
    p.update(over); return p


def quantile_normalize_y(y_source, y_target):
    q = np.clip(rankdata(y_source, method='average') / (len(y_source) + 1), 0.01, 0.99)
    return np.quantile(y_target, q)


# ---------------------------------------------------------------------------- folds
def grouped_folds(d: Data, seed=0, n_folds=5):
    """All rows of a catalyst in exactly one fold; group->fold assignment shuffled by seed."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(d.n_cat)
    fold_of_group = np.empty(d.n_cat, dtype=int)
    for k, chunk in enumerate(np.array_split(perm, n_folds)):
        fold_of_group[chunk] = k
    fold = fold_of_group[d.groups]
    return [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(n_folds)]


def row_folds(d: Data, seed=0, n_folds=5):
    """Row-level split — HISTORICAL COMPARISON ONLY. Leaks catalyst identity across folds."""
    return list(KFold(n_folds, shuffle=True, random_state=seed).split(d.X_lab))


# ---------------------------------------------------------------------------- pipeline pieces
def drst_mask(Xlab_tr_sc, Xlit_sc, tau=0.30, seed=42):
    """Domain classifier trained on TRAIN-fold lab rows only -> lab-like literature mask."""
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(Xlab_tr_sc), size=min(10_000, len(Xlab_tr_sc)), replace=False)
    clf = LogisticRegression(C=0.5, max_iter=1000, random_state=42, n_jobs=-1).fit(
        np.vstack([Xlab_tr_sc[sub], Xlit_sc]),
        np.r_[np.ones(len(sub)), np.zeros(len(Xlit_sc))])
    return clf.predict_proba(Xlit_sc)[:, 1] >= tau


def stage1_data(kind, Xlit_f, ylit_f, Xlab_tr, ylab_tr):
    """Stage-1 design matrix + labels. WARNING: *_joint kinds include lab training rows in Stage 1 —
    under row-level splits this is the catalyst-identity leakage channel; use only with grouped folds
    or for historical comparison."""
    if kind == 'qn_joint':
        return np.vstack([Xlit_f, Xlab_tr]), np.concatenate([quantile_normalize_y(ylit_f, ylab_tr), ylab_tr])
    if kind == 'raw_joint':
        return np.vstack([Xlit_f, Xlab_tr]), np.concatenate([ylit_f, ylab_tr])
    if kind == 'qn_litonly':
        return Xlit_f, quantile_normalize_y(ylit_f, ylab_tr)
    if kind == 'rank_litonly':
        return Xlit_f, rankdata(ylit_f, method='average') / (len(ylit_f) + 1)
    raise ValueError(kind)


def run_fold(d: Data, tr, va, seed, model, s1_kind='qn_joint', lit_X=None, lit_y=None,
             use_filter=True, lgbp=None, xgbp=None):
    """Train on lab[tr] (+ literature per config), predict lab[va]. Scaler + DRST are train-only."""
    sc = StandardScaler().fit(d.X_lab[tr])
    Xtr, Xva = sc.transform(d.X_lab[tr]), sc.transform(d.X_lab[va])
    lgbp = lgbp or lgb_params(seed); xgbp = xgbp or xgb_params(seed)
    if model == 'baseline':
        m = lgb.LGBMRegressor(**lgbp).fit(Xtr, d.y_lab[tr])
        return m.predict(Xva)
    Xl = sc.transform(lit_X if lit_X is not None else d.X_lit)
    yl = lit_y if lit_y is not None else d.y_lit
    if use_filter:
        m = drst_mask(Xtr, Xl)
        Xl, yl = Xl[m], yl[m]
    Xs1, ys1 = stage1_data(s1_kind, Xl, yl, Xtr, d.y_lab[tr])
    pre = xgb.XGBRegressor(**xgbp).fit(Xs1, ys1)
    ptr = pre.predict(Xtr).reshape(-1, 1); pva = pre.predict(Xva).reshape(-1, 1)
    fin = lgb.LGBMRegressor(**lgbp).fit(np.hstack([Xtr, ptr]), d.y_lab[tr])
    return fin.predict(np.hstack([Xva, pva]))


# ---------------------------------------------------------------------------- metrics
def row_metrics(yt, yp):
    return dict(rmse=float(np.sqrt(mean_squared_error(yt, yp))),
                mae=float(mean_absolute_error(yt, yp)), r2=float(r2_score(yt, yp)))


def cat_metrics(cat_codes, yt, yp, top_frac=0.10, budget=20):
    """Catalyst-level screening metrics: max over each catalyst's test rows, predicted vs observed."""
    g = pd.DataFrame({'c': cat_codes, 't': yt, 'p': yp}).groupby('c').agg(t=('t', 'max'), p=('p', 'max'))
    n = len(g); K = max(1, int(round(top_frac * n)))
    top_true = set(g.nlargest(K, 't').index)
    prec = float(np.mean([c in top_true for c in g.nlargest(K, 'p').index]))
    B = min(budget, n)
    prec_budget = float(np.mean([c in top_true for c in g.nlargest(B, 'p').index]))
    return dict(n_catalysts=int(n),
                pearson_max=float(np.corrcoef(g.t, g.p)[0, 1]),
                spearman_max=float(spearmanr(g.t, g.p)[0]),
                precision_top10pct=prec, enrichment_top10pct=float(prec / top_frac),
                precision_at20_vs_top10pct=prec_budget)


def run_cv(d: Data, folds, seed, model, **kw):
    """Full CV: pooled row metrics + per-fold-mean RMSE + catalyst metrics over all test predictions."""
    yp_all = np.empty(len(d.y_lab)); per_fold = []
    for tr, va in folds:
        yp = run_fold(d, tr, va, seed, model, **kw)
        yp_all[va] = yp
        per_fold.append(np.sqrt(mean_squared_error(d.y_lab[va], yp)))
    out = row_metrics(d.y_lab, yp_all)
    out['rmse_foldmean'] = float(np.mean(per_fold))
    out.update(cat_metrics(d.groups, d.y_lab, yp_all))
    return out
