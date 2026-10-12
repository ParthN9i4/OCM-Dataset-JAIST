"""
build_name_map.py -- rebuild catalyst_name_to_composition.json and check it.

WHAT IT DOES
OCM_data_with_conditions.csv names each catalyst (Name = M1-M2-M3-Support, e.g. K-K-K-BaO) and
holds the gas-flow conditions. The year==2025 rows of OCM_lab_data_and_literature_datal.csv hold
the same 89,074 yields with the 65 element loadings, but no names and no flows. This script maps
each Name to its composition key: Preparation + the 65 element columns, astype(str), joined by
'|'. That is the key ocm_eval.Data.load() and ocm_eval.load_with_conditions() use.

WHY THE JOIN IS BY CATALYST
The two files list the rows in a different order. 672 rows share an exact (temperature, yield)
key with at least one other row (65 keys; 607 rows beyond the first of each key, check 5), so a
row-by-row match is ambiguous. 551 of the 672 have yield exactly 0. Each catalyst's fingerprint
is instead its sorted multiset of (temperature, yield) pairs, compared as exact floats with no
rounding. A Name is joined to the composition key with the identical fingerprint.

PASS/FAIL RULES (any failure raises; every count goes to build_name_map.json)
  0.  Join: fingerprints are unique within each file, and the match is 1:1 for all 917 catalysts.
  0b. Name format: Name == M1-M2-M3-Support on every row, and M1/M2/M3/Support are constant
      within each Name: 917/917.
  1.  The Support element (e.g. Ba for BaO) has the strictly largest loading (a unique
      maximum): 917/917.
  2.  Every promoter in M1/M2/M3 that is not 'none' has loading > 0: 917/917.
  3.  Loadings rule: each non-'none' promoter slot adds 10/3 wt%, the support element adds 90 wt%
      (merged if it is the same element), and the vector is rescaled to sum to 100. All 65
      loadings must match within 1e-6: 917/917. Reported, not gated: how many Names match
      exactly one composition key by this rule alone (no yields used), and whether that map
      equals the fingerprint map. Only catalysts with 1 or 2 promoters test the rescale step.
  4.  After the map is written, ocm_eval.load_with_conditions() runs (it re-asserts the multiset
      match), and its catalyst codes equal Data.load().groups: same 917 codes, and each
      composition key gets the same code from both loaders. The two loaders list rows in
      different orders, so groups are NOT equal position by position (that count is stored for
      reference). The row-level check is on rows sorted by (code, temperature, yield).
  5.  Collision count (reported, must be the same in both files): rows whose exact (temperature,
      yield) key is shared with at least one other row, minus the number of such keys.
  Checks 0-3 and 5 run before the map is written; if one fails, the map is not written.

SERIALIZATION AND THE GIT DIFF
json.dump(mapping, indent=1), no trailing newline, the format of the committed file. The
committed key order appears to have come from iterating a Python set of hash strings; it is not
sorted by name or key, nor in file order, so it cannot be rebuilt from the CSVs. The values are
always rebuilt from the CSVs. The key order is taken, in this priority, from: the map in git
HEAD, the map on disk, or sorted names (when neither holds the same 917 names). The file is
written only when its bytes change, via a temp file and os.replace. Afterwards the script runs
"git diff HEAD --stat -- catalyst_name_to_composition.json". Expected: empty when the CSVs are
unchanged since the commit. A non-empty diff is recorded as a warning, not a failure, because a
real data change or a sorted-order rebuild legitimately produces one.

Run from the repo root:  python3 build_name_map.py
Population: all 917 catalysts / 89,074 lab rows. No model, no seeds, no estimator.
"""
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd

from ocm_eval import TARGET, Data, load_with_conditions

HERE = os.path.dirname(os.path.abspath(__file__))
COND_PATH = os.path.join(HERE, 'OCM_data_with_conditions.csv')
LAB_PATH = os.path.join(HERE, 'OCM_lab_data_and_literature_datal.csv')
MAP_NAME = 'catalyst_name_to_composition.json'
MAP_PATH = os.path.join(HERE, MAP_NAME)
OUT_PATH = os.path.join(HERE, 'build_name_map.json')

PROMOTER_WT = 10.0 / 3.0     # wt% per non-'none' promoter slot, before rescaling
SUPPORT_WT = 90.0            # wt% for the support element, before rescaling
TOL = 1e-6
N_EXPECTED = 917


def fingerprint(temps, ys):
    """Sorted multiset of exact (temperature, yield) pairs, as a hashable tuple."""
    return tuple(sorted(zip(np.asarray(temps, float).tolist(), np.asarray(ys, float).tolist())))


def collisions(temps, ys):
    """Rows sharing an exact (temperature, yield) key with another row, and the number of keys."""
    counts = pd.Series(1, index=pd.MultiIndex.from_arrays(
        [np.asarray(temps, float), np.asarray(ys, float)])).groupby(level=[0, 1]).size()
    shared = counts[counts >= 2]
    n_rows = int(shared.sum()); n_keys = int(len(shared))
    is_zero = shared.index.get_level_values(1) == 0.0
    zero, nonzero = shared[is_zero], shared[~is_zero]
    return {'keys_shared_by_2plus_rows': n_keys, 'rows_in_those_keys': n_rows,
            'collisions_rows_minus_keys': n_rows - n_keys,
            'of_which_keys_with_yield_exactly_0': int(len(zero)),
            'of_which_rows_with_yield_exactly_0': int(zero.sum()),
            'nonzero_yield_keys': int(len(nonzero)),
            'nonzero_yield_rows': int(nonzero.sum()),
            'nonzero_yield_collisions_rows_minus_keys': int(nonzero.sum() - len(nonzero))}


def support_element(support):
    m = re.match(r'([A-Z][a-z]?)', support)
    if m is None:
        raise ValueError(f'cannot parse support element from {support!r}')
    return m.group(1)


def expected_loadings(m1, m2, m3, sup, el_idx, n_el, rescale=True):
    """Loadings vector implied by the Name, or None if an element is not a column."""
    s_el = support_element(sup)
    proms = [p for p in (m1, m2, m3) if p != 'none']
    if s_el not in el_idx or any(p not in el_idx for p in proms):
        return None
    exp = np.zeros(n_el)
    exp[el_idx[s_el]] += SUPPORT_WT
    for p in proms:
        exp[el_idx[p]] += PROMOTER_WT
    if rescale:
        exp *= 100.0 / exp.sum()
    return exp


def git(*args):
    """Run a read-only git command in the repo; None if git is missing or fails."""
    try:
        r = subprocess.run(['git', *args], cwd=HERE, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def write_record(rec):
    with open(OUT_PATH, 'w') as f:
        json.dump(rec, f, indent=1)


def main():
    t0 = time.time()
    rec = {'script': 'build_name_map.py',
           'population': 'all 917 catalysts / 89,074 lab rows; no model, no seeds, no estimator',
           'status': 'running',
           'inputs': {'conditions_file': os.path.basename(COND_PATH),
                      'lab_file': os.path.basename(LAB_PATH) + ' (year == 2025 rows)'},
           'checks': {}, 'warnings': []}
    chk = rec['checks']

    # ----------------------------------------------------------------------------- load
    cond = pd.read_csv(COND_PATH)
    raw = pd.read_csv(LAB_PATH)
    lab = raw[raw.year == 2025].reset_index(drop=True)
    el = [c for c in raw.columns if c not in ['Preparation', 'Temperature_C', TARGET, 'year']]
    assert len(el) == 65, len(el)
    lab['_k'] = lab[['Preparation'] + el].astype(str).agg('|'.join, axis=1)
    rec['inputs'].update(rows_conditions_file=int(len(cond)), rows_lab_file=int(len(lab)),
                         element_columns=len(el))

    # --------------------------------------------------------------------- 0b. name format
    parts = cond.groupby('Name', sort=False)[['M1', 'M2', 'M3', 'Support']].first()
    n_parts_const = int((cond.groupby('Name')[['M1', 'M2', 'M3', 'Support']].nunique() == 1)
                        .all(axis=1).sum())
    name_is_join = bool((cond.Name == cond.M1 + '-' + cond.M2 + '-' + cond.M3 + '-'
                         + cond.Support).all())
    chk['name_format'] = {'rule': '0b. Name == M1-M2-M3-Support on every row, and '
                                  'M1/M2/M3/Support constant within each Name',
                          'name_equals_join_all_rows': name_is_join,
                          'names_with_constant_parts': n_parts_const, 'n_names': int(len(parts)),
                          'pass': name_is_join and n_parts_const == len(parts)}

    # --------------------------------------------------------------- 0. fingerprint join
    fp_cond = {n: fingerprint(g.Temperature, g.C2y) for n, g in cond.groupby('Name', sort=False)}
    fp_lab = {k: fingerprint(g.Temperature_C, g[TARGET]) for k, g in lab.groupby('_k', sort=False)}
    by_fp_lab = {f: k for k, f in fp_lab.items()}
    by_fp_cond = {f: n for n, f in fp_cond.items()}
    mapping = {n: by_fp_lab[f] for n, f in fp_cond.items() if f in by_fp_lab}
    n_matched = len(mapping)
    one_to_one = (n_matched == len(fp_cond) == len(fp_lab) == len(set(mapping.values())))
    chk['fingerprint_join'] = {
        'rule': '0. fingerprint = sorted multiset of exact (temperature, yield) pairs per '
                'catalyst; unique within each file; 1:1 match for every catalyst',
        'catalysts_conditions_file': len(fp_cond), 'catalysts_lab_file': len(fp_lab),
        'unique_fingerprints_conditions_file': len(by_fp_cond),
        'unique_fingerprints_lab_file': len(by_fp_lab),
        'names_matched': n_matched, 'distinct_composition_keys_matched': len(set(mapping.values())),
        'pass': bool(one_to_one and len(by_fp_cond) == len(fp_cond) == N_EXPECTED
                     and len(by_fp_lab) == len(fp_lab) == N_EXPECTED)}

    # ----------------------------------------------------------- 1-3. composition checks
    comp = lab.drop_duplicates('_k').set_index('_k')[el]
    el_idx = {e: i for i, e in enumerate(el)}
    n_sup = n_prom = n_rule = n_rule_norescale = 0
    worst_rule = 0.0
    fail_sup, fail_prom, fail_rule = [], [], []
    n_prom_hist = {}
    for name, key in mapping.items():
        v = comp.loc[key].values.astype(float)
        m1, m2, m3, sup = parts.loc[name]
        s_el = support_element(sup)
        proms = [p for p in (m1, m2, m3) if p != 'none']
        n_prom_hist[len(proms)] = n_prom_hist.get(len(proms), 0) + 1
        # 1. support element = dominant element (strict: unique maximum)
        order = np.argsort(-v, kind='stable')
        if el[order[0]] == s_el and v[order[0]] > v[order[1]]:
            n_sup += 1
        else:
            fail_sup.append(name)
        # 2. promoters present
        if all(p in el_idx and v[el_idx[p]] > 0 for p in proms):
            n_prom += 1
        else:
            fail_prom.append(name)
        # 3. loadings rule (and the same rule without the rescale step, for reference)
        exp = expected_loadings(m1, m2, m3, sup, el_idx, len(el))
        exp_raw = expected_loadings(m1, m2, m3, sup, el_idx, len(el), rescale=False)
        dev = float(np.max(np.abs(v - exp))) if exp is not None else float('inf')
        worst_rule = max(worst_rule, dev)
        if dev <= TOL:
            n_rule += 1
        else:
            fail_rule.append(name)
        if exp_raw is not None and float(np.max(np.abs(v - exp_raw))) <= TOL:
            n_rule_norescale += 1

    # 3, reported only: a map built from the loadings rule alone (no yields used)
    names = list(parts.index)
    keys_arr = np.asarray(comp.index)
    C = comp.values.astype(float)
    E = [expected_loadings(*parts.loc[n], el_idx, len(el)) for n in names]
    n_hits = np.zeros(len(names), int)
    rule_key = [None] * len(names)
    for i, e in enumerate(E):
        if e is None:
            continue
        hit = np.abs(C - e).max(axis=1) <= TOL
        n_hits[i] = int(hit.sum())
        if n_hits[i] == 1:
            rule_key[i] = keys_arr[hit.argmax()]
    n_rule_unique = int((n_hits == 1).sum())
    n_rule_agree = int(sum(rule_key[i] is not None and mapping.get(n) == rule_key[i]
                           for i, n in enumerate(names)))

    chk['support_is_dominant_element'] = {
        'rule': '1. element parsed from Support (e.g. La2O3 -> La) has the strictly largest '
                'loading (unique maximum)',
        'n_pass': n_sup, 'n_total': n_matched, 'failing_names': fail_sup[:10],
        'pass': n_sup == n_matched == N_EXPECTED}
    chk['promoters_present'] = {
        'rule': "2. every non-'none' promoter in M1/M2/M3 has loading > 0",
        'n_pass': n_prom, 'n_total': n_matched, 'failing_names': fail_prom[:10],
        'pass': n_prom == n_matched == N_EXPECTED}
    chk['loadings_rule'] = {
        'rule': "3. each non-'none' promoter slot adds 10/3 wt%, the support element adds 90 wt% "
                '(merged if same element), rescaled to sum to 100; all 65 loadings within 1e-6',
        'n_pass': n_rule, 'n_total': n_matched, 'tolerance': TOL,
        'max_abs_deviation_wt_pct': worst_rule, 'failing_names': fail_rule[:10],
        'pass': n_rule == n_matched == N_EXPECTED,
        'reported_not_gated': {
            'catalysts_by_number_of_promoters': {str(k): v for k, v in sorted(n_prom_hist.items())},
            'n_pass_without_rescaling': n_rule_norescale,
            'note_rescaling': 'With 3 promoters the raw sum is already 100, so only catalysts '
                              'with 1 or 2 promoters test the rescale step.',
            'names_with_exactly_one_rule_satisfying_key': n_rule_unique,
            'names_with_zero_rule_satisfying_keys': int((n_hits == 0).sum()),
            'names_with_2plus_rule_satisfying_keys': int((n_hits >= 2).sum()),
            'rule_only_map_agrees_with_fingerprint_map': n_rule_agree,
            'note_rule_only_map': 'Built from Name and loadings alone, with no yields; an '
                                  'independent check on the fingerprint join.'}}

    # ------------------------------------------------------------------- 5. collisions
    col_cond = collisions(cond.Temperature, cond.C2y)
    col_lab = collisions(lab.Temperature_C, lab[TARGET])
    chk['row_key_collisions'] = {
        'definition': '5. Count rows whose exact (Temperature, C2y) key is shared with at least '
                      'one other row, minus the number of such keys. Equivalently: rows that are '
                      'not the first occurrence of their key. This is the "607 collisions" '
                      'figure (672 rows in 65 shared keys; 672 - 65 = 607); it is why rows cannot '
                      'be matched one by one between the two files. Most of it is zero yields: '
                      '551 of the 672 rows have C2y exactly 0, in 5 keys (one per temperature). '
                      'Non-zero yields alone give 60 keys / 121 rows / 61 collisions.',
        'conditions_file': col_cond, 'lab_file_year_2025': col_lab,
        'collisions': col_cond['collisions_rows_minus_keys'],
        'same_in_both_files': col_cond == col_lab,
        'pass': col_cond == col_lab}
    # keep the definition text honest if the data ever change
    if (col_cond['rows_in_those_keys'], col_cond['keys_shared_by_2plus_rows'],
            col_cond['of_which_rows_with_yield_exactly_0'], col_cond['nonzero_yield_keys'],
            col_cond['nonzero_yield_rows']) != (672, 65, 551, 60, 121):
        rec['warnings'].append('collision counts differ from the numbers quoted in the '
                               'row_key_collisions definition text; update the text')

    failed = [k for k, c in chk.items() if not c['pass']]
    if failed:
        rec['map_written'] = False
        rec['status'] = f'failed before writing the map: {failed}'
        rec['all_checks_pass'] = False
        write_record(rec)
        raise AssertionError(f'checks failed before writing the map: {failed}')

    # ------------------------------------------------------------------- write the map
    old_bytes = open(MAP_PATH, 'rb').read() if os.path.exists(MAP_PATH) else None
    old_map = json.loads(old_bytes) if old_bytes is not None else {}
    head_bytes = git('show', f'HEAD:./{MAP_NAME}')
    head_map = json.loads(head_bytes) if head_bytes else None
    if head_map is not None and set(head_map) == set(mapping):
        keys = list(head_map)
        rec['key_order'] = 'taken from the map in git HEAD (same 917 names)'
    elif old_bytes is not None and set(old_map) == set(mapping):
        keys = list(old_map)
        rec['key_order'] = 'taken from the map on disk (git HEAD unavailable or different names)'
        rec['warnings'].append('key order not taken from git HEAD')
    else:
        keys = sorted(mapping)
        rec['key_order'] = 'sorted by name (neither git HEAD nor the disk file has the same names)'
        rec['warnings'].append('key order sorted by name; expect a reorder diff against HEAD')
    out = {k: mapping[k] for k in keys}
    new_bytes = json.dumps(out, indent=1).encode('ascii')
    rec['existing_file'] = {
        'present': old_bytes is not None,
        'names_with_same_value': int(sum(old_map.get(k) == v for k, v in out.items())),
        'bytes_identical_to_rebuild': old_bytes == new_bytes,
        'sha256_before': hashlib.sha256(old_bytes).hexdigest() if old_bytes else None}
    rec['git_head_file'] = {
        'available': head_map is not None,
        'names_with_same_value': (int(sum(head_map.get(k) == v for k, v in out.items()))
                                  if head_map is not None else None),
        'bytes_identical_to_rebuild': (head_bytes == new_bytes) if head_bytes else None}
    if old_bytes == new_bytes:
        rec['map_written'] = False
        rec['map_write_note'] = 'skipped: rebuilt bytes identical to the file on disk'
    else:
        fd, tmp = tempfile.mkstemp(dir=HERE, prefix='.' + MAP_NAME + '.', suffix='.tmp')
        with os.fdopen(fd, 'wb') as f:
            f.write(new_bytes)
        os.replace(tmp, MAP_PATH)
        rec['map_written'] = True
        rec['map_write_note'] = 'written atomically (temp file + os.replace)'
    rec['sha256_after'] = hashlib.sha256(new_bytes).hexdigest()

    # record progress so a crash in check 4 cannot leave a stale passing record behind
    rec['status'] = 'map done; check 4 running'
    rec['all_checks_pass'] = None
    write_record(rec)

    # ------------------------------------------- 4. loader runs and codes match Data.load()
    try:
        df, feats, groups, n_cat = load_with_conditions(cond_path=COND_PATH, map_path=MAP_PATH,
                                                        path=LAB_PATH)
    except Exception as exc:
        chk['loader_codes_match_data_load'] = {'rule': '4. load_with_conditions() runs',
                                               'error': repr(exc), 'pass': False}
        rec['status'] = 'failed in check 4: load_with_conditions() raised'
        rec['all_checks_pass'] = False
        write_record(rec)
        raise
    d = Data.load(LAB_PATH)
    key_d = d.dl_lab[['Preparation'] + el].astype(str).agg('|'.join, axis=1)
    code_d = pd.DataFrame({'k': key_d, 'c': d.groups}).drop_duplicates()
    one_code_per_key = bool(code_d.k.is_unique and code_d.c.is_unique)
    code_of = dict(zip(code_d.k, code_d.c))
    n_rows_same_code = int((df._k.map(code_of).values == groups).sum())
    a = (pd.DataFrame({'g': groups, 't': df.Temperature_C.values.astype(float),
                       'y': df[TARGET].values.astype(float)})
         .sort_values(['g', 't', 'y']).reset_index(drop=True))
    b = (pd.DataFrame({'g': d.groups, 't': d.dl_lab.Temperature_C.values.astype(float),
                       'y': d.y_lab.astype(float)})
         .sort_values(['g', 't', 'y']).reset_index(drop=True))
    sorted_equal = bool(len(a) == len(b) and (a.values == b.values).all())
    counts_equal = bool((np.bincount(groups, minlength=n_cat)
                         == np.bincount(d.groups, minlength=d.n_cat)).all()) if n_cat == d.n_cat else False
    same_len = len(groups) == len(d.groups)
    pos_groups = int((np.asarray(groups) == d.groups).sum()) if same_len else None
    pos_yields = int((df[TARGET].values.astype(float) == d.y_lab.astype(float)).sum()) if same_len else None
    chk['loader_codes_match_data_load'] = {
        'rule': '4. load_with_conditions() runs (re-asserts the multiset match); same n_cat as '
                'Data.load(); each composition key gets the same code from both loaders; rows '
                'sorted by (code, temperature, yield) are identical arrays',
        'load_with_conditions_rows': int(len(df)), 'load_with_conditions_n_cat': int(n_cat),
        'load_with_conditions_features': len(feats),
        'data_load_rows': int(len(d.y_lab)), 'data_load_n_cat': int(d.n_cat),
        'one_code_per_key_in_data_load': one_code_per_key,
        'rows_with_same_code_as_data_load': n_rows_same_code,
        'rows_per_code_equal': counts_equal,
        'sorted_code_temperature_yield_arrays_equal': sorted_equal,
        'pass': bool(n_cat == d.n_cat == N_EXPECTED and one_code_per_key
                     and n_rows_same_code == len(df) and counts_equal and sorted_equal),
        'reported_not_gated': {
            'groups_elementwise_equal': (pos_groups == len(groups)) if same_len else False,
            'rows_equal_by_position_groups': pos_groups,
            'rows_equal_by_position_yields': pos_yields,
            'note': 'The two loaders list rows in different orders, so groups from '
                    'load_with_conditions() must not be used with row-ordered arrays from '
                    'Data.load(). Codes agree per catalyst, not per row position.'}}

    # ----------------------------------------------------------- git diff of the map vs HEAD
    diff = git('diff', 'HEAD', '--stat', '--', MAP_NAME)
    rec['git_diff_cmd'] = f'git diff HEAD --stat -- {MAP_NAME}'
    rec['git_diff_stat_map'] = diff.decode().strip() if diff is not None else None
    rec['git_diff_stat_map_empty'] = (rec['git_diff_stat_map'] == '') if diff is not None else None
    if rec['git_diff_stat_map_empty'] is None:
        rec['warnings'].append('git diff could not run (git missing or not a repo)')
    elif not rec['git_diff_stat_map_empty']:
        rec['warnings'].append('git diff HEAD of the map is not empty: ' + rec['git_diff_stat_map'])

    rec['all_checks_pass'] = all(c['pass'] for c in chk.values())
    rec['status'] = 'complete'
    rec['runtime_seconds'] = round(time.time() - t0, 1)
    write_record(rec)

    # ------------------------------------------------------------------------- print
    j, c4, c5 = chk['fingerprint_join'], chk['loader_codes_match_data_load'], chk['row_key_collisions']
    l3 = chk['loadings_rule']['reported_not_gated']
    print('All 917 catalysts / 89,074 lab rows. No model, no seeds, no estimator.')
    print(f"0.  join: {j['names_matched']}/{N_EXPECTED} names matched 1:1; unique fingerprints "
          f"{j['unique_fingerprints_conditions_file']} (conditions file) / "
          f"{j['unique_fingerprints_lab_file']} (lab file)")
    print(f"0b. name format: {chk['name_format']['names_with_constant_parts']}/"
          f"{chk['name_format']['n_names']} names; Name == M1-M2-M3-Support on all rows: "
          f"{chk['name_format']['name_equals_join_all_rows']}")
    for k, label in [('support_is_dominant_element', '1.  support element = dominant element'),
                     ('promoters_present', '2.  promoters present'),
                     ('loadings_rule', '3.  loadings rule (1e-6)')]:
        print(f"{label}: {chk[k]['n_pass']}/{chk[k]['n_total']}")
    print(f"    loadings rule max |deviation| = {chk['loadings_rule']['max_abs_deviation_wt_pct']:.2e} wt%")
    print(f"    catalysts by number of promoters {l3['catalysts_by_number_of_promoters']}; "
          f"pass without rescaling {l3['n_pass_without_rescaling']}/{N_EXPECTED}")
    print(f"    rule alone (no yields): {l3['names_with_exactly_one_rule_satisfying_key']}/"
          f"{N_EXPECTED} names match exactly one key; agrees with fingerprint map "
          f"{l3['rule_only_map_agrees_with_fingerprint_map']}/{N_EXPECTED}")
    r4 = c4['reported_not_gated']
    print(f"4.  load_with_conditions codes = Data.load() codes: {c4['pass']} "
          f"(n_cat {c4['load_with_conditions_n_cat']} vs {c4['data_load_n_cat']}; same code per "
          f"key on {c4['rows_with_same_code_as_data_load']:,}/{c4['load_with_conditions_rows']:,} rows)")
    print(f"    row positions differ between loaders: groups equal by position on "
          f"{r4['rows_equal_by_position_groups']:,}/{c4['load_with_conditions_rows']:,} rows (expected)")
    cc = c5['conditions_file']
    print(f"5.  collisions (conditions file): {cc['keys_shared_by_2plus_rows']} keys hold "
          f"{cc['rows_in_those_keys']} rows -> {c5['collisions']} rows beyond the first of each key; "
          f"{cc['of_which_rows_with_yield_exactly_0']} of the {cc['rows_in_those_keys']} rows are "
          f"yield 0; same in lab file: {c5['same_in_both_files']}")
    print(f"map: key order {rec['key_order']}; {rec['map_write_note']}; "
          f"git diff HEAD --stat empty: {rec['git_diff_stat_map_empty']}")
    for w in rec['warnings']:
        print('WARNING:', w)
    print(f"all checks pass: {rec['all_checks_pass']}  ({rec['runtime_seconds']} s)")
    if not rec['all_checks_pass']:
        raise AssertionError('check 4 failed after writing the map; see build_name_map.json')


if __name__ == '__main__':
    main()
