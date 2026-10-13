"""
doc_number_trace.py -- every number quoted in ocm_worknote_v3.md and ocm_email_reply_v3.md, traced to
the JSON that stores it.

For each entry: the text as printed, the JSON file and key path, and the rounding used in the text. The
script asserts that the stored value rounds to the printed text, and that the printed text appears in
the documents that quote it. Derived numbers (percentages, differences) are recomputed from their stored
inputs. Run it after any change to the documents or the JSONs; it exits non-zero on any mismatch.
"""
import json, re, sys

DOCS = {'note': open('ocm_worknote_v3.md').read(), 'email': open('ocm_email_reply_v3.md').read()}
J = {}


def get(path):
    f, *keys = path.split(':')
    if f not in J:
        J[f] = json.load(open(f))
    v = J[f]
    for k in keys[0].split('/'):
        v = v[int(k)] if isinstance(v, list) else v[k]
    return v


P11 = 'phase11_condition_baseline.json'
P12 = 'phase12_runs_and_coverage.json'
P8B = 'phase8b_label_sweep.json'
P13 = 'phase13_condition_value.json'
P14 = 'phase14_crossprep_pft.json'
REC = 'a_condition_recommendation'
FIXED = 'screening_bootstrap_seed_averaged/fixed_slot_set_max/all_917_S_complete/C_minus_A/spearman'
LBL = 'verdict/sensitivity_table/rows'

# (printed text, source, decimals, docs that must contain the text)
ROWS = [
    ('2.90', f'{P11}:row_rmse/A/all_917/per_seed_mean/rmse', 2, ['note', 'email']),
    ('2.66', f'{P11}:row_rmse/C/all_917/per_seed_mean/rmse', 2, ['note', 'email']),
    ('0.24 points', f'{P11}:row_rmse_bootstrap_seed_averaged/A_minus_C/all_917/delta_point', 2, ['note']),
    ('0.22 to', f'{P11}:row_rmse_bootstrap_seed_averaged/A_minus_C/all_917/ci95_low', 2, ['note']),
    ('to 0.26', f'{P11}:row_rmse_bootstrap_seed_averaged/A_minus_C/all_917/ci95_high', 2, ['note']),
    ('2.94.', 'grouped_tuning.json:confirmation/default/rmse_foldmean_mean', 2, ['note']),
    ('same 18 conditions', f'{P11}:fixed_slot_set/candidates/1/n_slots', 0, ['note', 'email']),
    ('465 of', f'{P11}:{FIXED}/n_catalysts', 0, ['note']),
    ('0.732 with', f'{P11}:screening/fixed_slot_set_max/C/all_917_S_complete/seed_averaged_predictions/spearman', 3, ['note']),
    ('0.727 without', f'{P11}:screening/fixed_slot_set_max/A/all_917_S_complete/seed_averaged_predictions/spearman', 3, ['note']),
    ('+0.005 (95% CI', f'{P11}:{FIXED}/delta_point', 3, ['note']),
    ('−0.007 to', f'{P11}:{FIXED}/ci95_low', 3, ['note']),
    ('to +0.015', f'{P11}:{FIXED}/ci95_high', 3, ['note']),
    ('worse (−0.014)', f'{P11}:screening_bootstrap_seed_averaged/full_grid_max/all_917/C_minus_A/spearman/delta_point', 3, ['note']),
    ('1.75 points', f'{P13}:{REC}/regret/headline/k3/all_917/model_C_per_seed/mean_regret_seed_averaged', 2, ['note']),
    ('1.90 points', f'{P13}:{REC}/regret/headline/k3/all_917/family_rule/mean_regret_seed_averaged', 2, ['note']),
    ('0.15 points', f'{P13}:{REC}/rule_a/primary_k3_headline_all_917/mean_edge', 2, ['note']),
    ('0.15 yield points', f'{P13}:{REC}/rule_a/primary_k3_headline_all_917/mean_edge', 2, ['email']),
    ('0.07 to', f'{P13}:{REC}/rule_a/primary_k3_headline_all_917/ci95/0', 2, ['note']),
    ('to 0.23', f'{P13}:{REC}/rule_a/primary_k3_headline_all_917/ci95/1', 2, ['note']),
    ('10.4 points', f'{P13}:{REC}/mean_truth/all_917', 1, ['note']),
    ('0.969', f'{P13}:b_reduced_protocol/rule_b/results/most_often_max_K10/designed_seed_mean_spearman', 3, ['note', 'email']),
    ('0.917 on average', f'{P13}:b_reduced_protocol/rule_b/results/most_often_max_K10/random_seed_mean_mean', 3, ['note', 'email']),
    ('percentile 0.925', f'{P13}:b_reduced_protocol/rule_b/results/most_often_max_K10/random_seed_mean_p97_5', 3, ['note']),
    ('gave 0.939', f'{P13}:b_reduced_protocol/rule_b/results/greedy_K10/designed_seed_mean_spearman', 3, ['note']),
    ('0.009 lower', f'{P13}:b_reduced_protocol/composition_model_on_K_slot_labels/greedy_K10_label/all_917/minus_B_median_spearman_seed_averaged/delta_point', 3, ['note']),
    ('−0.018 to', f'{P13}:b_reduced_protocol/composition_model_on_K_slot_labels/greedy_K10_label/all_917/minus_B_median_spearman_seed_averaged/ci95_low', 3, ['note']),
    ('to +0.001)', f'{P13}:b_reduced_protocol/composition_model_on_K_slot_labels/greedy_K10_label/all_917/minus_B_median_spearman_seed_averaged/ci95_high', 3, ['note']),
    ('0.01 points', f'{P8B}:condition_mix_facts/single_highest_row_influence/labels/q0.50/median_abs_change_yield_pct', 2, ['note']),
    ('0.38 points', f'{P8B}:condition_mix_facts/single_highest_row_influence/labels/q1.00/median_abs_change_yield_pct', 2, ['note']),
    ('(+0.006, 95%', f'{P8B}:{LBL}/q0.75/all_917/spearman/vs_q1.00/delta_boot_mean', 3, ['note']),
    ('CI +0.001 to', f'{P8B}:{LBL}/q0.75/all_917/spearman/vs_q1.00/ci95/0', 3, ['note']),
    ('+0.012), but', f'{P8B}:{LBL}/q0.75/all_917/spearman/vs_q1.00/ci95/1', 3, ['note']),
    ('(+0.001, 95%', f'{P8B}:{LBL}/q0.75/all_917/spearman/vs_q0.50/delta_boot_mean', 3, ['note']),
    ('CI −0.002 to', f'{P8B}:{LBL}/q0.75/all_917/spearman/vs_q0.50/ci95/0', 3, ['note']),
    ('to +0.005). We', f'{P8B}:{LBL}/q0.75/all_917/spearman/vs_q0.50/ci95/1', 3, ['note']),
    ('84.5%', f'{P8B}:condition_mix_facts/peak_ratio/share_peak_at_ch4o2_2', 3, ['note']),
    ('0.388', f'{P14}:arms_T2_1003/C2_merge/per_seed_mean/spearman_max', 3, ['note', 'email']),
    ('0.422', f'{P14}:arms_T2_1003/C7_merge_qn_labels/per_seed_mean/spearman_max', 3, ['note', 'email']),
    ('+0.034', f'{P14}:key_numbers/T2_1003.C7_merge_qn_labels_minus_C2_merge.spearman.seed_averaged.delta', 3, ['note']),
    ('CI +0.018 to', f'{P14}:key_numbers/T2_1003.C7_merge_qn_labels_minus_C2_merge.spearman.seed_averaged.ci95/0', 3, ['note']),
    ('to +0.050)', f'{P14}:key_numbers/T2_1003.C7_merge_qn_labels_minus_C2_merge.spearman.seed_averaged.ci95/1', 3, ['note']),
    ('For 28 catalysts', f'{P12}:top_decile_inflation/n_doubled', 0, ['note', 'email']),
    ('1,906', f'{P12}:two_run_structure/n_duplicated_slots', 0, ['note']),
    ('factor of 2.3', f'{P12}:two_run_structure/hi_lo_ratio/pooled_median_over_pairs/overall', 1, ['note', 'email']),
    ('10 of the 92', f'{P12}:top_decile_inflation/raw_max/count_doubled_in_top_K', 0, ['note']),
    ('give 2.8', f'{P12}:top_decile_inflation/expected_count_if_random', 1, ['note']),
    ('4 remain', f'{P12}:top_decile_inflation/duplicated_slots_as_pair_mean/count_doubled_in_top_K', 0, ['note']),
    ('is −0.29', f'{P12}:exclusion_test/between_catalyst/spearman_missing_slots_of_135_incl_absent_cells_vs_observed_max', 2, ['note']),
    ('Only 46%', f'{P12}:exclusion_test/rule_R4/scope/share_scored', 2, ['note']),
    ('only 46%', f'{P12}:exclusion_test/rule_R4/scope/share_scored', 2, ['note']),
]


def fmt(v, d, text):
    if text.endswith('%'):
        return f'{100 * v:.{d - 2}f}%'
    if d == 0:
        return f'{int(round(v)):,}'
    s = f'{v:.{d}f}'
    if text[0] in '+−':
        s = ('+' if v >= 0 else '−') + s.lstrip('-')
    return s


fails = []
for text, src, d, docs in ROWS:
    try:
        v = get(src)
    except (KeyError, IndexError, TypeError) as e:
        fails.append(f'MISSING KEY {src}: {e!r}'); continue
    num = re.search(r'[+−]?\d[\d,]*(?:\.\d+)?%?', text).group(0)
    if ' lower' in text:            # magnitude stated in words ('ranked 0.009 lower')
        v = abs(v)
    want = fmt(v, d, num)
    ok_val = want == num
    ok_doc = all(text in DOCS[k] for k in docs)
    print(f"{'ok ' if ok_val and ok_doc else 'BAD'} {text:>14s}  stored {v!r:<24} -> {want:<10} {src.split(':')[0]}")
    if not ok_val:
        fails.append(f'{text}: stored value formats as {want}')
    if not ok_doc:
        fails.append(f'{text}: not found in {[k for k in docs if text not in DOCS[k]]}')

# derived numbers, recomputed from stored inputs
a = get(f'{P11}:row_rmse/A/all_917/per_seed_mean/rmse'); c = get(f'{P11}:row_rmse/C/all_917/per_seed_mean/rmse')
pct = 100 * (a - c) / a
edge = get(f'{P13}:{REC}/rule_a/primary_k3_headline_all_917/mean_edge'); truth = get(f'{P13}:{REC}/mean_truth/all_917')
cov = [p['coverage'] for p in get(f'{P13}:b_reduced_protocol/designed/most_often_max/K10/all_917/per_seed')]
import collections
_ch = get(f'{P13}:b_reduced_protocol/choices_per_seed_fold')
_cnt = collections.Counter(x for r in _ch for x in r['most_often_max_slots'][:10])
_unm = get(f'{P13}:{REC}/share_of_top_k_picks_unmeasured/k3/all_917')
DERIVED = [(f'{sum(1 for v in _cnt.values() if v == len(_ch))} conditions were chosen in all {len(_ch)}',
            '7 conditions were chosen in all 25', ['note']),
           ('About 16%', f"About {100 * (_unm['model_C_per_seed'] + _unm['family_rule']) / 2:.0f}%", ['note']),
           ('0.24 points (8.1%)', f'{a - c:.2f} points ({pct:.1f}%)', ['note']),
           ('92% of catalysts', f'{100 * sum(cov) / len(cov):.0f}% of catalysts', ['note']),
           ('8.1%', f'{pct:.1f}%', ['note', 'email']), ('1.4% of it', f'{100 * edge / truth:.1f}% of it', ['note']),
           ('About 6 of the 92', f'About {10 - 4} of the 92', ['note']), ('About 6 of\n   the 92', f'About {10 - 4} of\n   the 92', ['email'])]
for text, want, docs in DERIVED:
    ok = text == want and all(text in DOCS[k] for k in docs)
    print(f"{'ok ' if ok else 'BAD'} {text:>14s}  recomputed {want}")
    if not ok:
        fails.append(f'derived {text} vs {want}')

print('\nall numbers trace' if not fails else '\nFAILURES:\n  ' + '\n  '.join(fails))
sys.exit(1 if fails else 0)
