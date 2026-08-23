"""
build_campaign.py — the recommended 17-catalyst synthesis campaign.

Ranking by predicted yield alone returns 20 near-identical Ba/Mo catalysts, which spends a whole
campaign testing one hypothesis twenty times. This splits the budget deliberately:

  Tier A (12) — the highest-predicted IN_SUPPORT candidates. This is where the expected hits are;
                it is the model's actual best guess and preserves the retrospective hit rate.
  Tier B (5)  — the best candidate for each of the five strongest NON-Ba supports. These cost
                predicted yield, but they test whether the model's strong Ba preference reflects
                real chemistry or the fact that 78% of the lab's top decile already contains Ba.

Reads phase6_candidates.json (produced by phase6_candidates.py). Output: campaign_shortlist.csv
"""
import json
import pandas as pd

C = json.load(open('phase6_candidates.json'))
AUD = json.load(open('phase5_target_audit.json'))
full = pd.read_csv('phase6_candidates.csv')
insup = full[full.tier == 'IN_SUPPORT'].sort_values('predicted_max_yield', ascending=False)

tierA = insup.head(12).copy()
tierA['tier_role'] = 'A: model optimum'

# best non-Ba support, five strongest.
# NOTE: phase6_candidates.csv holds only the top 500 rows and every one of them is Ba-supported,
# so the per-support winners must come from the JSON's best_per_support block (computed over all
# 26,414 candidates), not from the CSV.
best_per_support = pd.DataFrame([r for r in C['best_per_support'] if r['support'] != 'Ba'])
best_per_support = best_per_support.sort_values('predicted_max_yield', ascending=False).head(5).copy()
best_per_support['tier_role'] = 'B: support diversity'

camp = pd.concat([tierA, best_per_support], ignore_index=True)
cols = ['tier_role', 'formula', 'support', 'promoters', 'predicted_max_yield',
        'ensemble_std', 'nn_distance', 'min_element_support']
camp[cols].to_csv('campaign_shortlist.csv', index=False)

ci = AUD['E_bootstrap_ci']['precision_at20_ci95']
print(f"{'#':>2s} {'role':22s} {'formula':40s} {'pred':>6s} {'+/-':>5s}")
for i, r in camp.iterrows():
    print(f"{i+1:2d} {r['tier_role']:22s} {r['formula']:40s} {r['predicted_max_yield']:6.2f} {r['ensemble_std']:5.2f}")
print(f"\nTotal: {len(camp)} catalysts ({len(tierA)} tier A + {len(best_per_support)} tier B)")
print(f"Tier A predicted range: {tierA.predicted_max_yield.min():.2f}-{tierA.predicted_max_yield.max():.2f}%")
print(f"Tier B predicted range: {best_per_support.predicted_max_yield.min():.2f}-{best_per_support.predicted_max_yield.max():.2f}%")
print(f"Supports covered: {sorted(camp.support.unique())}")
print(f"\nHonest expectation for tier A, from retrospective grouped-CV precision@20:")
print(f"  {ci[0]:.2f}-{ci[1]:.2f} of them truly top-decile (95% CI) -> roughly {ci[0]*12:.0f}-{ci[1]*12:.0f} of the 12")
print("  Tier B is an information buy, not a yield buy: if several tier-B catalysts outperform their")
print("  predictions, the model's Ba preference is partly an artifact of the lab's existing coverage.")
