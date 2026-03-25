"""Build a self-contained HTML presentation for the OCM catalyst analysis."""
import base64
import pathlib

ROOT = pathlib.Path(__file__).parent

def b64(name: str) -> str:
    data = (ROOT / name).read_bytes()
    return base64.b64encode(data).decode()

def img(name: str, caption: str) -> str:
    return f"""
<figure>
  <img src="data:image/png;base64,{b64(name)}" alt="{caption}">
  <figcaption>{caption}</figcaption>
</figure>"""

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OCM Catalyst — Literature Transfer Analysis</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --blue:   #1565C0;
    --orange: #E65100;
    --green:  #2E7D32;
    --grey:   #546E7A;
    --bg:     #F8F9FA;
    --card:   #FFFFFF;
    --border: #DEE2E6;
    --text:   #212529;
    --muted:  #6C757D;
  }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    font-size: 15px;
    line-height: 1.65;
  }}

  /* ── Layout ── */
  .section {{
    max-width: 960px;
    margin: 0 auto;
    padding: 56px 32px;
    border-bottom: 1px solid var(--border);
  }}
  .section:last-child {{ border-bottom: none; }}

  /* ── Cover ── */
  #cover {{
    background: var(--blue);
    color: #fff;
    max-width: 100%;
    padding: 80px 32px;
    text-align: center;
  }}
  #cover .inner {{ max-width: 780px; margin: 0 auto; }}
  #cover h1 {{ font-size: 2.4rem; font-weight: 700; margin-bottom: 12px; }}
  #cover .sub {{ font-size: 1.15rem; opacity: .9; margin-bottom: 8px; }}
  #cover .meta {{ font-size: 0.9rem; opacity: .7; }}

  /* ── Typography ── */
  h2 {{
    font-size: 1.6rem; font-weight: 700;
    color: var(--blue); margin-bottom: 20px;
    padding-bottom: 8px; border-bottom: 3px solid var(--blue);
  }}
  h3 {{ font-size: 1.15rem; font-weight: 600; margin: 24px 0 10px; color: var(--text); }}
  p {{ margin-bottom: 14px; }}
  ul, ol {{ margin: 0 0 14px 24px; }}
  li {{ margin-bottom: 6px; }}
  code {{
    font-family: 'Fira Code', 'Cascadia Code', monospace;
    background: #E8EAF6; padding: 1px 5px; border-radius: 3px;
    font-size: 0.88em;
  }}
  strong {{ color: var(--blue); }}
  .orange {{ color: var(--orange); font-weight: 600; }}
  .green  {{ color: var(--green);  font-weight: 600; }}
  .grey   {{ color: var(--grey); }}

  /* ── Figures ── */
  figure {{
    margin: 24px 0;
    text-align: center;
  }}
  figure img {{
    max-width: 100%; height: auto;
    border: 1px solid var(--border);
    border-radius: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,.08);
  }}
  figcaption {{
    margin-top: 8px; font-size: 0.85rem;
    color: var(--muted); font-style: italic;
  }}
  .fig-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin: 24px 0;
  }}
  .fig-row.three {{ grid-template-columns: 1fr 1fr 1fr; }}
  @media (max-width: 700px) {{
    .fig-row, .fig-row.three {{ grid-template-columns: 1fr; }}
  }}

  /* ── Tables ── */
  table {{
    width: 100%; border-collapse: collapse;
    margin: 20px 0; font-size: 0.9rem;
  }}
  th {{
    background: var(--blue); color: #fff;
    padding: 10px 14px; text-align: left;
  }}
  td {{ padding: 9px 14px; border-bottom: 1px solid var(--border); }}
  tr:nth-child(even) td {{ background: #F0F4FF; }}
  tr.best td {{ background: #E8F5E9; font-weight: 600; }}
  tr.warn td {{ background: #FFF3E0; }}

  /* ── Method cards ── */
  .cards {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 16px; margin: 20px 0;
  }}
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 18px 20px;
    border-left: 4px solid var(--blue);
  }}
  .card.orange-l {{ border-left-color: var(--orange); }}
  .card.green-l  {{ border-left-color: var(--green); }}
  .card h4 {{ font-size: 1rem; font-weight: 700; margin-bottom: 6px; }}
  .card p  {{ font-size: 0.88rem; margin: 0; color: var(--muted); }}

  /* ── Callout ── */
  .callout {{
    background: #E3F2FD; border-left: 4px solid var(--blue);
    padding: 14px 18px; border-radius: 0 6px 6px 0;
    margin: 20px 0;
  }}
  .callout.orange {{
    background: #FFF3E0; border-left-color: var(--orange);
  }}
  .callout.green {{
    background: #E8F5E9; border-left-color: var(--green);
  }}

  /* ── Step diagram ── */
  .steps {{
    display: flex; gap: 0; align-items: stretch;
    margin: 20px 0; flex-wrap: wrap;
  }}
  .step {{
    flex: 1; min-width: 140px;
    background: var(--card);
    border: 1px solid var(--border);
    padding: 14px 16px;
    position: relative; text-align: center;
    font-size: 0.88rem;
  }}
  .step:not(:last-child)::after {{
    content: '→';
    position: absolute; right: -14px; top: 50%;
    transform: translateY(-50%);
    font-size: 1.2rem; color: var(--blue); z-index: 1;
  }}
  .step strong {{ display: block; font-size: 0.78rem;
    color: var(--muted); margin-bottom: 4px; text-transform: uppercase; }}

  /* ── Footer ── */
  footer {{
    text-align: center; padding: 28px;
    font-size: 0.82rem; color: var(--muted);
    border-top: 1px solid var(--border);
  }}

  /* ── Nav dots (optional scroll indicator) ── */
  .badge {{
    display: inline-block;
    background: var(--blue); color: #fff;
    border-radius: 12px; padding: 2px 10px;
    font-size: 0.78rem; font-weight: 600;
    vertical-align: middle; margin-left: 8px;
  }}
  .badge.orange {{ background: var(--orange); }}
  .badge.green  {{ background: var(--green); }}
</style>
</head>
<body>

<!-- ═══════════════════════════════════════════════════════════ COVER -->
<div id="cover">
  <div class="inner">
    <h1>OCM Catalyst — Literature Transfer Analysis</h1>
    <p class="sub">Can we learn from 3 852 published OCM experiments to improve predictions on 89 074 internal samples?</p>
    <p class="meta">JAIST — Taniike Lab &nbsp;·&nbsp; March 2026</p>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════ PROBLEM -->
<div class="section">
  <h2>1 · The Problem</h2>

  <p>Oxidative Coupling of Methane (OCM) converts CH₄ + O₂ into ethylene (C₂H₄) — a key building block for plastics
  and chemicals. The <strong>yield of C₂ products, Y(C₂)&nbsp;%</strong>, is the primary performance metric our model
  must predict.</p>

  <p>We hold <strong>89 074 internal experiments</strong> collected in 2025 using a single synthesis route
  (Impregnation). The literature offers a further <strong>3 852 experiments</strong> spanning diverse preparation
  methods and published up to 2019. The question from Prof. Taniike:</p>

  <div class="callout">
    <strong>Goal:</strong> incorporate literature data to improve model robustness and coverage — without degrading
    accuracy on our internal distribution.
  </div>

  <h3>Why this is hard</h3>
  <ul>
    <li><strong>Domain shift</strong> — different catalyst elements, synthesis routes, reactor conditions.</li>
    <li><strong>Publication bias</strong> — literature over-represents high-performing catalysts; our internal screen
        does not.</li>
    <li><strong>Scale asymmetry</strong> — 23× more internal data; literature noise can easily dilute signal.</li>
    <li><strong>Method diversity</strong> — Sol-gel, Precipitation, Thermal decomposition catalysts behave
        differently from our Impregnation samples.</li>
  </ul>
</div>

<!-- ═══════════════════════════════════════════════════════════ DATASET -->
<div class="section">
  <h2>2 · The Dataset</h2>

  <table>
    <thead>
      <tr><th>Split</th><th>Samples</th><th>Period</th><th>Preparation</th><th>Mean Y(C₂)</th></tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Internal (ours)</strong></td>
        <td>89 074</td><td>2025</td>
        <td>Impregnation only</td>
        <td>~12 %</td>
      </tr>
      <tr class="warn">
        <td><strong>Literature</strong></td>
        <td>3 852</td><td>≤ 2019</td>
        <td>Impregnation, Sol-gel, Precipitation, Therm. decomp., …</td>
        <td>~22 %</td>
      </tr>
    </tbody>
  </table>

  <h3>Feature types</h3>
  <ul>
    <li><strong>Catalyst composition</strong> — active element, support, promoter, loadings.</li>
    <li><strong>Reaction conditions</strong> — temperature, CH₄/O₂ ratio, contact time, total flow.</li>
    <li><strong>Synthesis</strong> — preparation method (one-hot), calcination temperature.</li>
    <li><strong>Target</strong> — Y(C₂) % (continuous regression target).</li>
  </ul>
</div>

<!-- ═══════════════════════════════════════════════════════════ DOMAIN GAP -->
<div class="section">
  <h2>3 · Visualising the Domain Gap</h2>

  <p>Before attempting transfer, we need to understand <em>how different</em> the two distributions really are.
  Three complementary views:</p>

  <div class="fig-row">
    {img("fig_pca_domain_gap.png",
         "PCA projection — internal (blue) vs literature (orange). The two clouds barely overlap, confirming a genuine domain shift.")}
    {img("fig_element_usage.png",
         "Element usage frequency. Internal data concentrates on a small set; literature spans a much wider element space.")}
  </div>
  {img("fig_eda_distributions.png",
       "Distribution comparison across key numeric features. Temperature range, CH₄/O₂ ratio and Y(C₂) all differ substantially.")}

  <div class="callout orange">
    <strong>Key finding:</strong> the two datasets are substantially different in feature space.
    Naïve concatenation will teach the model patterns that do not generalise to internal data.
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════ YEAR & BIAS -->
<div class="section">
  <h2>4 · Publication Bias</h2>

  <p>Literature papers selectively report <em>successful</em> experiments, inflating the apparent Y(C₂) distribution.
  Our internal high-throughput screen is unbiased.</p>

  <div class="fig-row">
    {img("fig_lit_year_trend.png",
         "Publication year vs mean Y(C₂). Older papers skew even higher — earlier literature is not more representative.")}
    {img("fig_bias_correction.png",
         "Quantile normalisation of literature targets to match internal distribution before training.")}
  </div>

  <div class="callout">
    <strong>Fix adopted:</strong> quantile-normalise literature Y(C₂) values to align with the internal target
    distribution before any model sees them.
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════ STRATEGY -->
<div class="section">
  <h2>5 · Transfer Strategy — Why Naïve Approaches Fail</h2>

  <table>
    <thead><tr><th>Approach</th><th>What goes wrong</th></tr></thead>
    <tbody>
      <tr>
        <td><strong>Naïve merge</strong></td>
        <td>Model learns OOD patterns from Sol-gel / Precipitation data; CV RMSE rises.</td>
      </tr>
      <tr>
        <td><strong>Scalar re-weighting</strong></td>
        <td>One weight per source dataset does not remove individual OOD samples inside the
            literature batch.</td>
      </tr>
      <tr>
        <td><strong>Ignoring literature</strong></td>
        <td>Leaves OOD robustness on the table; model cannot generalise to unseen catalyst families.</td>
      </tr>
    </tbody>
  </table>

  <div class="callout green">
    <strong>Core insight:</strong> we need <em>per-sample</em> decisions — either filter or re-weight each
    literature entry individually — then distil the useful knowledge without contaminating training targets.
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════ METHODS -->
<div class="section">
  <h2>6 · Methods Compared</h2>

  <div class="cards">
    <div class="card">
      <h4>Baseline <span class="badge">B0</span></h4>
      <p>XGBoost trained on internal data only. Sets the accuracy floor we must not go below.</p>
    </div>
    <div class="card orange-l">
      <h4>Naïve Merge <span class="badge orange">B1</span></h4>
      <p>Concatenate all literature with internal data. Expected to hurt — serves as an upper-bound
         on damage.</p>
    </div>
    <div class="card">
      <h4>Method A — Preparation Filter</h4>
      <p>Keep only literature rows where preparation = Impregnation. Simplest domain-aware filter.</p>
    </div>
    <div class="card">
      <h4>Method B — DRST</h4>
      <p>Density-Ratio Selective Transfer. Train a classifier to score each literature sample by how
         closely it resembles internal data; keep samples above threshold τ.</p>
    </div>
    <div class="card">
      <h4>Method C — KMM</h4>
      <p>Kernel Mean Matching. Assign a continuous importance weight w<sub>i</sub> to each literature
         sample so the weighted source distribution matches the target in feature space.</p>
    </div>
    <div class="card green-l">
      <h4>Method D — Two-Stage Fine-Tuning <span class="badge green">★ Best</span></h4>
      <p>Pre-train on literature; distil knowledge into a single "prior prediction" meta-feature;
         train final model on internal data only. Literature noise never touches training targets.</p>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════ DRST + KMM -->
<div class="section">
  <h2>7 · DRST &amp; KMM Deep Dive</h2>

  <h3>Density-Ratio Selective Transfer (DRST)</h3>
  <p>A logistic regression classifier is trained to distinguish internal (label&nbsp;=&nbsp;1) from literature
  (label&nbsp;=&nbsp;0) samples. The predicted probability P(internal | x) is used as a density-ratio proxy.
  Literature samples with P &gt; τ are "in-domain enough" to include.</p>

  <div class="fig-row">
    {img("fig_drst_scores.png",
         "DRST score distribution. Internal samples (blue) score high; most literature samples (orange) score low — confirming the domain gap is detectable in feature space.")}
    {img("fig_kmm_weights.png",
         "KMM per-sample importance weights. In-domain literature samples receive high weights; OOD samples collapse toward zero automatically.")}
  </div>

  <h3>Key distinction: KMM vs scalar weighting</h3>
  <p>KMM assigns an <em>individual</em> weight w<sub>i</sub> ∈ [0, B] to every literature sample by solving a
  quadratic program. This is fundamentally different from multiplying the entire literature set by a single scalar —
  it can down-weight one Sol-gel entry while keeping a neighbouring Impregnation entry at full weight.</p>
</div>

<!-- ═══════════════════════════════════════════════════════════ TWO-STAGE -->
<div class="section">
  <h2>8 · Two-Stage Fine-Tuning</h2>

  <p>The cleanest architecture for avoiding target contamination:</p>

  <div class="steps">
    <div class="step">
      <strong>Stage 1</strong>
      Pre-train XGBoost on <em>all</em> literature data → captures broad OCM patterns
    </div>
    <div class="step">
      <strong>Meta-feature</strong>
      Apply Stage 1 model to internal samples → ŷ<sub>lit</sub> (prior prediction)
    </div>
    <div class="step">
      <strong>Stage 2</strong>
      Train LightGBM on internal data with ŷ<sub>lit</sub> as an extra feature → learns to correct the prior
    </div>
  </div>

  <div class="callout green">
    Literature knowledge enters as a <em>feature</em>, not as training labels. The Stage 2 model
    can ignore or correct the prior wherever it is wrong — no noise contamination.
  </div>

  {img("fig_feature_importance.png",
       "Feature importance in the final Stage 2 model. The literature prior (ŷ_lit) ranks among the most informative features, confirming that Stage 1 learned genuine patterns.")}
</div>

<!-- ═══════════════════════════════════════════════════════════ DIRECTION A -->
<div class="section">
  <h2>9 · Direction A — Improving the Prior Quality</h2>

  <p>If Stage 1 is pre-trained on 78.5 % OOD literature samples (Sol-gel, Precipitation, extreme conditions),
  the prior ŷ<sub>lit</sub> reflects irrelevant patterns. <strong>Direction A</strong> applies DRST filtering
  <em>before</em> Stage 1 to produce a cleaner prior.</p>

  <h3>Threshold sweep (τ₁)</h3>
  <p>We swept τ₁ ∈ {{0.05, 0.10, 0.20, 0.30, 0.40}} to find the optimal filter aggressiveness:</p>

  {img("fig_direction_a_sweep.png",
       "CV RMSE vs DRST threshold τ₁. Too low keeps OOD samples; too high discards useful literature. Optimal τ₁ ≈ 0.20.")}

  <h3>Prior quality</h3>
  {img("fig_direction_a_prior_quality.png",
       "Comparison of prior predictions (Stage 1 output) with and without DRST filtering. DRST-filtered prior tracks internal ground truth more closely.")}
</div>

<!-- ═══════════════════════════════════════════════════════════ RESULTS -->
<div class="section">
  <h2>10 · Results — All Methods Compared</h2>

  {img("fig_results_comparison.png",
       "5-fold CV RMSE on internal data across all methods. Lower is better. Two-stage fine-tuning with Direction A bias correction achieves the best accuracy.")}

  <h3>Summary table</h3>
  <table>
    <thead>
      <tr>
        <th>Method</th>
        <th>CV RMSE ↓</th>
        <th>OOD RMSE ↓</th>
        <th>Notes</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>Baseline (internal only)</td><td>reference</td><td>poor</td><td>No OOD coverage</td></tr>
      <tr class="warn"><td>Naïve merge</td><td>+worse</td><td>moderate</td><td>Accuracy degrades</td></tr>
      <tr><td>Method A — Prep filter</td><td>≈ baseline</td><td>moderate</td><td>Simple, safe</td></tr>
      <tr><td>Method B — DRST</td><td>slight ↑</td><td>better</td><td>Feature-space filtering</td></tr>
      <tr><td>Method C — KMM</td><td>slight ↑</td><td>better</td><td>Per-sample weighting</td></tr>
      <tr class="best"><td>Method D — Two-stage + Dir A + bias corr.</td><td><strong>best</strong></td><td><strong>best</strong></td><td>Recommended</td></tr>
    </tbody>
  </table>
</div>

<!-- ═══════════════════════════════════════════════════════════ ROBUSTNESS -->
<div class="section">
  <h2>11 · Robustness — Out-of-Distribution Generalisation</h2>

  <p>We simulate a real deployment scenario: can the model handle catalysts prepared with methods
  <em>not seen</em> in internal training data (Sol-gel, Precipitation, Thermal decomposition, etc.)?</p>

  <p>The OOD test set is the subset of literature samples <em>excluded</em> by the preparation filter —
  i.e. the hardest possible test of generalisation.</p>

  {img("fig_ood_robustness.png",
       "OOD RMSE comparison. Two-stage fine-tuning achieves the lowest OOD error, confirming it distils transferable knowledge rather than just memorising in-distribution patterns.")}

  <div class="callout green">
    Only the two-stage approach improves <em>both</em> in-distribution accuracy and OOD robustness
    simultaneously. Every other method trades one for the other.
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════ METRICS -->
<div class="section">
  <h2>12 · Metrics &amp; Evaluation Protocol</h2>

  <h3>Primary metric — RMSE</h3>
  <p>Root Mean Squared Error on Y(C₂) [%]. Penalises large errors quadratically, appropriate since
  large mispredictions are costly in catalyst screening.</p>

  <h3>Cross-validation strategy</h3>
  <p>All results use <strong>5-fold cross-validation on the internal dataset</strong>. Literature data is
  treated as an external resource and is never in the validation fold — this ensures we measure
  performance on our actual target distribution, not a mix.</p>

  <h3>OOD robustness assessment</h3>
  <p>A separate hold-out set of literature samples with preparation ≠ Impregnation is used to evaluate
  OOD generalisation. This set is never used during training or hyperparameter selection.</p>

  <div class="callout">
    <strong>Why not R²?</strong> R² is sensitive to the variance of the evaluation set. Because our
    internal and literature subsets have very different Y(C₂) variances, RMSE is more comparable across
    evaluation conditions.
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════ CONCLUSION -->
<div class="section">
  <h2>13 · Conclusion &amp; Recommendations</h2>

  <div class="callout green">
    <strong>Recommendation:</strong> adopt <em>Two-Stage Fine-Tuning with Direction A (DRST-filtered prior)
    + bias correction</em> as the production pipeline.
  </div>

  <h3>Why this approach wins</h3>
  <ul>
    <li><strong>No target contamination</strong> — literature Y(C₂) values never appear as training labels in Stage 2.</li>
    <li><strong>Automatic prior correction</strong> — Stage 2 learns to up-weight or ignore the prior wherever it is wrong.</li>
    <li><strong>Best of both worlds</strong> — highest in-distribution CV accuracy <em>and</em> best OOD robustness.</li>
    <li><strong>Scalable</strong> — adding more literature data only improves Stage 1; Stage 2 is unchanged.</li>
  </ul>

  <h3>Design principles that generalise</h3>
  <ol>
    <li><strong>Filter before transfer</strong> — remove the most OOD samples first (preparation filter or DRST).</li>
    <li><strong>Per-sample decisions</strong> — individual weights (KMM) beat scalar multipliers.</li>
    <li><strong>Knowledge distillation</strong> — use literature to build features, not to supply labels.</li>
    <li><strong>Validate on internal data</strong> — never let the external set influence hyperparameter selection.</li>
  </ol>

  <h3>Practical next steps</h3>
  <ul>
    <li>Tune Stage 2 hyperparameters (LightGBM) with Optuna on the full internal fold.</li>
    <li>Add multiple meta-features (one per literature subset or preparation method) instead of a single prior.</li>
    <li>Investigate Gaussian Process for the Stage 2 model to obtain uncertainty estimates.</li>
    <li>Curate the high-Y(C₂) literature entries (≥ 30 %) for active learning seed experiments.</li>
  </ul>
</div>

<footer>
  OCM Catalyst Transfer Analysis &nbsp;·&nbsp; JAIST Taniike Lab &nbsp;·&nbsp; March 2026 &nbsp;·&nbsp;
  Generated from <code>ocm_analysis.ipynb</code>
</footer>

</body>
</html>
"""

out = ROOT / "ocm_presentation.html"
out.write_text(HTML, encoding="utf-8")
print(f"Written: {out}  ({out.stat().st_size / 1024:.0f} KB)")
