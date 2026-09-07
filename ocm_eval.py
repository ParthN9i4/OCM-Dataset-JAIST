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
- grouped_folds prevents a LAB catalyst from crossing train/validation, but literature is
  fold-invariant: a small number of lab catalysts (3 of 917, verified) have a near-exact composition
  match in the literature set. Stage-1 can still see a held-out catalyst's yield via that literature
  twin unless the caller passes run_fold(..., exclude_overlap=True); default is False (no existing
  caller currently sets it — none has needed to, since the only two ocm_eval-based callers,
  catalyst_level.py and grouped_tuning.py, only ever run model='baseline', which never reaches
  literature at all). See exclude_overlapping_lit for the mechanism.
- Catalyst-level metrics (max-yield Spearman, enrichment@10%, precision@20) are primary for the
  screening objective; row RMSE is secondary and must be labeled as such. Each (catalyst,
  temperature) cell holds up to 27 measurements under settings this file does not record
  (5 x 27 = 135 per catalyst). Those rows are not replicates: 19.9% of total yield variance sits
  within cells and is therefore unreachable from composition and temperature alone, which floors
  row-level RMSE at 1.757. That floor is a property of the missing feature columns, not measurement
  noise, and it assumes the model already knows each catalyst's cell means — so it is not attainable
  for an unseen catalyst.
- OPEN QUESTION, do not write past it (SESSION_CONTEXT.md §7 item 2): whether those 27 slots are 27
  DISTINCT REACTION CONDITIONS or 27 SUCCESSIVE TIME-ON-STREAM SAMPLES at one condition is NOT
  established. The file cannot settle it — every cell is stored sorted descending by yield, so row
  order encodes rank, not acquisition sequence (phase11_condition_grid_forensics.json). Only JAIST
  can answer it. phase10_ground_truth_invariance.json shows the ranking survives either reading
  (worst regret −0.017 Spearman), which bounds the risk but does not resolve the question.

Usage:
    from ocm_eval import Data, grouped_folds, row_folds, run_fold, run_cv, lgb_params, xgb_params
    d = Data.load()
    res = run_cv(d, grouped_folds(d, seed=0), seed=0, model='baseline')
    # to also close the narrow lab<->literature identity overlap noted above:
    res = run_cv(d, grouped_folds(d, seed=0), seed=0, model='pft', exclude_overlap=True)
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
    lab_cat_key: np.ndarray = field(default=None)   # str catalyst-identity key per GROUP id, len n_cat
    lit_cat_key: np.ndarray = field(default=None)   # str catalyst-identity key per LITERATURE ROW
    dl_lab: pd.DataFrame = field(repr=False, default=None)
    dl_lit: pd.DataFrame = field(repr=False, default=None)

    @classmethod
    def load(cls, path=DATA_PATH):
        df = pd.read_csv(path)
        dl_lab = df[df.year == 2025].reset_index(drop=True)
        dl_lit = df[df.year <= 2019].reset_index(drop=True)
        # The two filters must partition the file with nothing left over. True today only because no
        # row's year falls in 2020-2024; a future data refresh that adds one there would otherwise be
        # silently dropped from both frames with no error and no visible symptom.
        assert len(dl_lab) + len(dl_lit) == len(df), (
            f"year filters cover {len(dl_lab) + len(dl_lit)} of {len(df)} rows -- "
            f"a row with year outside {{2025}} U (-inf, 2019] now exists; update the filters")
        el = [c for c in df.columns if c not in ['Preparation', 'Temperature_C', TARGET, 'year']]
        le = LabelEncoder().fit(df.Preparation)
        dl_lab['prep_enc'] = le.transform(dl_lab.Preparation)
        dl_lit['prep_enc'] = le.transform(dl_lit.Preparation)
        feats = ['Temperature_C', 'prep_enc'] + el
        # Catalyst identity = preparation + full element-loading vector (temperature is a condition).
        # Rounded before the string join: raw floats carry float32<->float64 round-trip noise (verified
        # on this file -- e.g. a nominal 10% loading stored as 9.999999999 in one row and 10.0 in
        # another for the SAME chemical catalyst) that would otherwise let a value like -0.0 vs 0.0, or
        # noise below the 6th decimal, silently fragment one catalyst into two pd.factorize groups.
        # Verified: rounding to 6 decimals still yields exactly 917 lab groups on the current data.
        cat_str = dl_lab[['Preparation'] + el].round(6).astype(str).agg('|'.join, axis=1)
        groups, uniques = pd.factorize(cat_str)
        # Same identity key for literature rows, used only by exclude_overlapping_lit -- literature
        # itself is never grouped into folds (there is exactly one "everyone's literature" pool).
        lit_cat_key = dl_lit[['Preparation'] + el].round(6).astype(str).agg('|'.join, axis=1).values
        return cls(X_lab=dl_lab[feats].values.astype(float), y_lab=dl_lab[TARGET].values,
                   X_lit=dl_lit[feats].values.astype(float), y_lit=dl_lit[TARGET].values,
                   groups=groups, n_cat=len(uniques), features=feats,
                   lab_cat_key=np.asarray(uniques), lit_cat_key=lit_cat_key,
                   dl_lab=dl_lab, dl_lit=dl_lit)


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


def exclude_overlapping_lit(d: Data, va, lit_mask=None):
    """Boolean keep-mask over d.X_lit/d.y_lit (or, if `lit_mask` is given, over the subset already
    addressed by it, positionally) that excludes any literature row whose catalyst-identity key
    exactly matches a lab catalyst held out in `va`.

    Why this exists: grouped_folds keeps a catalyst's lab rows out of both train and validation in the
    same fold, but literature is fold-invariant — d.X_lit/d.y_lit are identical for every fold. A
    handful of lab catalysts also appear, essentially verbatim, in the literature set (verified: 3 of
    917 on the current data — pure Ba, pure Ti, and 10%Ba/90%La; the third is only found because
    catalyst identity is built from ROUNDED values, see Data.load — the lab row records a nominal 10%
    Ba loading as 9.999999999, literature records it as 10.0). Without this filter, Stage-1 can still
    see a held-out catalyst's yield behaviour through its literature twin even though the catalyst
    itself never appears in the lab training rows for that fold.

    Opt-in via run_fold(..., exclude_overlap=True); default False, so no existing caller's behaviour
    changes."""
    va_keys = set(d.lab_cat_key[np.unique(d.groups[va])])
    keep = ~np.isin(d.lit_cat_key, list(va_keys))
    return keep if lit_mask is None else keep[lit_mask]


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
             use_filter=True, lgbp=None, xgbp=None, exclude_overlap=False):
    """Train on lab[tr] (+ literature per config), predict lab[va]. Scaler + DRST are train-only.
    exclude_overlap=True additionally drops literature rows that are an exact catalyst-identity match
    for a catalyst held out in `va` (see exclude_overlapping_lit) — default False, so this changes no
    existing caller's behaviour. Raises if combined with a custom lit_X/lit_y override, since this
    function has no way to map an already-transformed override back to catalyst identity; filter such
    an override yourself with exclude_overlapping_lit(d, va, lit_mask=...) before passing it in."""
    sc = StandardScaler().fit(d.X_lab[tr])
    Xtr, Xva = sc.transform(d.X_lab[tr]), sc.transform(d.X_lab[va])
    lgbp = lgbp or lgb_params(seed); xgbp = xgbp or xgb_params(seed)
    if model == 'baseline':
        m = lgb.LGBMRegressor(**lgbp).fit(Xtr, d.y_lab[tr])
        return m.predict(Xva)
    if exclude_overlap:
        if lit_X is not None or lit_y is not None:
            raise ValueError("exclude_overlap=True cannot be combined with a custom lit_X/lit_y "
                              "override; filter it yourself with exclude_overlapping_lit(d, va, "
                              "lit_mask=...) and pass the result in as lit_X/lit_y instead")
        keep = exclude_overlapping_lit(d, va)
        lit_X_raw, lit_y_raw = d.X_lit[keep], d.y_lit[keep]
    else:
        lit_X_raw = lit_X if lit_X is not None else d.X_lit
        lit_y_raw = lit_y if lit_y is not None else d.y_lit
    Xl = sc.transform(lit_X_raw)
    yl = lit_y_raw
    if use_filter:
        m = drst_mask(Xtr, Xl, seed=seed)
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
