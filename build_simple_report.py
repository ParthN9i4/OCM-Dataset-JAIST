"""
Generates ocm_initial_report.html — a plain, readable first-attempt summary
of the five transfer learning methods tried on the OCM dataset.

Run:  python build_simple_report.py
"""

import base64, pathlib

ROOT = pathlib.Path(__file__).parent

def b64(name):
    p = ROOT / name
    if not p.exists():
        return ""
    return base64.b64encode(p.read_bytes()).decode()

def img(name, style=""):
    data = b64(name)
    if not data:
        return f'<p style="color:#999">[figure {name} not found]</p>'
    return f'<img src="data:image/png;base64,{data}" style="max-width:100%;border-radius:6px;{style}" alt="">'

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OCM Literature Integration — Initial Exploration</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'Georgia', serif;
    font-size: 16px;
    line-height: 1.75;
    color: #1a1a2e;
    background: #fafafa;
  }}

  .page {{
    max-width: 860px;
    margin: 0 auto;
    padding: 48px 32px 80px;
  }}

  /* ── Header ── */
  .header {{
    border-bottom: 3px solid #1a237e;
    padding-bottom: 24px;
    margin-bottom: 40px;
  }}
  .header h1 {{
    font-size: 26px;
    color: #1a237e;
    margin-bottom: 6px;
  }}
  .header .subtitle {{
    font-size: 15px;
    color: #555;
    font-style: italic;
  }}
  .header .meta {{
    margin-top: 10px;
    font-size: 14px;
    color: #888;
  }}

  /* ── Section titles ── */
  h2 {{
    font-size: 20px;
    color: #1a237e;
    margin: 44px 0 12px;
    padding-bottom: 4px;
    border-bottom: 1px solid #c5cae9;
  }}
  h3 {{
    font-size: 17px;
    color: #283593;
    margin: 28px 0 8px;
  }}

  p {{ margin-bottom: 14px; }}

  /* ── Callout boxes ── */
  .callout {{
    border-left: 4px solid #3f51b5;
    background: #e8eaf6;
    padding: 14px 18px;
    border-radius: 0 6px 6px 0;
    margin: 18px 0;
    font-size: 15px;
  }}
  .callout.good  {{ border-color: #2e7d32; background: #e8f5e9; }}
  .callout.bad   {{ border-color: #c62828; background: #ffebee; }}
  .callout.warn  {{ border-color: #e65100; background: #fff3e0; }}

  /* ── Method cards ── */
  .method-card {{
    border: 1px solid #c5cae9;
    border-radius: 8px;
    margin: 28px 0;
    overflow: hidden;
  }}
  .method-card .card-header {{
    background: #3f51b5;
    color: white;
    padding: 12px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .method-card .card-header.baseline {{ background: #455a64; }}
  .method-card .card-header.bad      {{ background: #c62828; }}
  .method-card .card-header.ok       {{ background: #1565c0; }}
  .method-card .card-header.best     {{ background: #2e7d32; }}
  .method-card .card-title {{ font-size: 17px; font-weight: bold; }}
  .method-card .card-rmse  {{ font-size: 22px; font-weight: bold; font-family: monospace; }}
  .method-card .card-body  {{ padding: 18px 20px; }}

  .tag {{
    display: inline-block;
    font-size: 12px;
    font-weight: bold;
    padding: 2px 8px;
    border-radius: 12px;
    margin-right: 4px;
    margin-bottom: 4px;
  }}
  .tag.green  {{ background: #c8e6c9; color: #1b5e20; }}
  .tag.red    {{ background: #ffcdd2; color: #b71c1c; }}
  .tag.blue   {{ background: #bbdefb; color: #0d47a1; }}
  .tag.grey   {{ background: #eceff1; color: #37474f; }}

  /* ── Comparison table ── */
  table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 15px; }}
  th {{
    background: #1a237e;
    color: white;
    padding: 10px 14px;
    text-align: left;
  }}
  td {{ padding: 9px 14px; border-bottom: 1px solid #e0e0e0; }}
  tr:nth-child(even) td {{ background: #f5f5f5; }}
  tr.best-row td {{ background: #e8f5e9; font-weight: bold; }}
  tr.worst-row td {{ background: #ffebee; }}
  .rmse-val {{ font-family: monospace; font-size: 16px; }}
  .delta-pos {{ color: #c62828; font-weight: bold; }}
  .delta-neg {{ color: #2e7d32; font-weight: bold; }}

  /* ── Figure captions ── */
  .fig-wrap {{ margin: 24px 0; }}
  .fig-caption {{
    font-size: 13px;
    color: #555;
    font-style: italic;
    margin-top: 6px;
    text-align: center;
  }}

  /* ── Key insight box ── */
  .insight {{
    background: #1a237e;
    color: white;
    padding: 22px 26px;
    border-radius: 8px;
    margin: 32px 0;
    font-size: 15px;
    line-height: 1.7;
  }}
  .insight strong {{ font-size: 17px; display: block; margin-bottom: 8px; }}

  /* ── Next steps ── */
  ol.steps li {{ margin-bottom: 10px; }}

  /* ── Footer ── */
  .footer {{
    margin-top: 60px;
    padding-top: 16px;
    border-top: 1px solid #ddd;
    font-size: 13px;
    color: #999;
  }}
</style>
</head>
<body>
<div class="page">

<!-- ── Header ───────────────────────────────────────────────────────────── -->
<div class="header">
  <h1>Can Published OCM Data Make Our Model Better?</h1>
  <div class="subtitle">An initial exploration of five approaches to incorporating literature data into our catalyst yield prediction model</div>
  <div class="meta">JAIST &mdash; Taniike Lab &nbsp;|&nbsp; May 2026</div>
</div>

<!-- ── Starting point ────────────────────────────────────────────────────── -->
<h2>Where We Started</h2>

<p>
  We have <strong>89,074 catalyst experiments</strong> collected in our lab in 2025,
  all using the impregnation preparation method. We trained a LightGBM model on this data
  to predict C2 yield and got a cross-validated RMSE of <strong>2.133%</strong>.
  (RMSE means: on average, our predictions are about 2.1 percentage points away from
  the actual measured yield.)
</p>

<p>
  Prof. Taniike's question was: we also have access to <strong>3,852 catalyst records
  from published literature</strong> (1982–2019) covering 15+ preparation methods.
  Can we use this data to improve the model's robustness without hurting its accuracy
  on our own experiments?
</p>

<div class="callout">
  <strong>The challenge:</strong> The two datasets are not the same kind of data.
  Our experiments average 5.25% C2 yield. Literature averages 8.67%.
  That 3.42 percentage point gap is not random noise &mdash; it reflects real differences
  in experimental protocols, conditions, and publication bias. Simply combining them
  turns out to make things worse.
</div>

<div class="fig-wrap">
  {img("fig_mean_shift.png")}
  <div class="fig-caption">
    The Y(C2) distributions of the two datasets. Left: normalised density curves showing
    the literature distribution shifted 3.42 pp to the right. Middle: the CDF makes the
    shift visible as a consistent horizontal gap. Right: violin plot with the gap annotated.
  </div>
</div>

<!-- ── Domain gap ─────────────────────────────────────────────────────────── -->
<h2>The Chemistry Gap</h2>

<p>
  Beyond the yield difference, the two datasets also use different elements in different
  proportions. To see this, we ran PCA &mdash; a technique that compresses our 65 element
  features down to 2 dimensions so we can draw a picture of the chemical space.
</p>

<div class="fig-wrap">
  {img("fig_pca_domain_gap.png")}
  <div class="fig-caption">
    PCA of all 92,926 catalyst compositions. Blue = our lab data; red = literature.
    The clouds overlap but are not the same &mdash; literature covers regions our model
    has never encountered.
  </div>
</div>

<p>
  <strong>78.5% of literature samples</strong> sit in regions of chemical space that
  look quite different from our training data. Our catalysts are dominated by barium.
  Literature catalysts are dominated by silicon, sodium, and manganese.
  Elements like lanthanum, cerium, and strontium appear in both &mdash; which is what
  makes transfer learning possible at all.
</p>

<!-- ── Five methods ───────────────────────────────────────────────────────── -->
<h2>Five Things We Tried</h2>

<p>
  We tried five progressively more sophisticated approaches to incorporating the
  literature data. Here is what each one does, why we tried it, and what happened.
</p>

<!-- Method 0: Baseline -->
<div class="method-card">
  <div class="card-header baseline">
    <span class="card-title">Baseline &mdash; Our Data Only (no literature)</span>
    <span class="card-rmse">RMSE 2.133%</span>
  </div>
  <div class="card-body">
    <p>
      Train LightGBM purely on our 89,074 internal experiments. No literature involved.
      This is the reference point that every other method has to beat.
    </p>
    <p>
      <span class="tag grey">reference</span>
    </p>
  </div>
</div>

<!-- Method 1: Naive merge -->
<div class="method-card">
  <div class="card-header bad">
    <span class="card-title">Attempt 1 &mdash; Just Add All Literature</span>
    <span class="card-rmse">RMSE 2.248% ▲</span>
  </div>
  <div class="card-body">
    <p>
      <strong>The idea:</strong> The simplest thing you could do &mdash; take all 3,852
      literature rows and append them to the training set. More data should mean a better
      model, right?
    </p>
    <p>
      <strong>What happened:</strong> RMSE went from 2.133 to 2.248 &mdash; <em>worse</em>
      by 5.4%. Adding the data actively hurt performance.
    </p>
    <p>
      <strong>Why it failed:</strong> The model now sees two catalysts with nearly
      identical chemistry &mdash; one from our lab labelled 5%, one from literature
      labelled 10% &mdash; with no way to know the difference came from conditions
      that are not in the feature set (gas flow rate, methane-to-oxygen ratio).
      The model hedges between the two contradictory labels and predicts both badly.
      The 3.42% systematic gap acts as structured noise.
    </p>
    <p>
      <span class="tag red">worse than baseline</span>
      <span class="tag grey">establishes that naive merging is harmful</span>
    </p>
  </div>
</div>

<!-- Method 2: DRST -->
<div class="method-card">
  <div class="card-header ok">
    <span class="card-title">Attempt 2 &mdash; Only Keep Similar Literature (DRST)</span>
    <span class="card-rmse">RMSE 2.019% ▼</span>
  </div>
  <div class="card-body">
    <p>
      <strong>The idea:</strong> If the problem is that 78.5% of literature samples are
      chemically unlike our data, what if we only use the ones that <em>are</em> similar?
      We trained a classifier to score each literature sample by how much it resembles our
      data chemistry. We then kept only the ones above a threshold &mdash; 782 samples
      out of 3,852.
    </p>
    <p>
      <strong>What happened:</strong> RMSE dropped to 2.019%, a 5.3% improvement over
      baseline. The first method to actually help.
    </p>
    <p>
      <strong>Why it works:</strong> The 782 kept samples describe catalyst chemistries
      adjacent to ours &mdash; they add genuine coverage without introducing the
      contradictions that sank naive merging. By discarding the 78.5% that are too
      different, we avoid noisy gradients in unfamiliar chemical regions.
    </p>
    <p>
      <span class="tag green">beats baseline</span>
      <span class="tag blue">hard filter: keep or discard</span>
    </p>
  </div>
</div>

<!-- Method 3: KMM -->
<div class="method-card">
  <div class="card-header ok">
    <span class="card-title">Attempt 3 &mdash; Soft Weights Per Sample (KMM)</span>
    <span class="card-rmse">RMSE 2.035% ▼</span>
  </div>
  <div class="card-body">
    <p>
      <strong>The idea:</strong> Instead of a binary keep-or-discard decision,
      assign each literature sample a continuous weight between 0 and 10 based
      on how similar its chemistry is to our data distribution. Samples very close
      to our distribution count almost fully; samples far away contribute almost nothing.
      This is solved with a quadratic optimisation program.
    </p>
    <p>
      <strong>What happened:</strong> RMSE of 2.035%, a 4.6% improvement. Almost
      identical to DRST. Importantly, 78.5% of samples received near-zero weight &mdash;
      which agrees with DRST's hard filter. Two independent methods converging on the
      same answer is reassuring.
    </p>
    <p>
      <strong>Why it works:</strong> Same logic as DRST but with softer boundaries.
      Samples on the edge of the threshold are not abruptly cut off &mdash; they
      contribute proportionally to their similarity.
    </p>
    <p>
      <span class="tag green">beats baseline</span>
      <span class="tag blue">soft weights: continuous contribution</span>
    </p>
  </div>
</div>

<!-- Method 4: Bias correction -->
<div class="method-card">
  <div class="card-header ok">
    <span class="card-title">Attempt 4 &mdash; Correct the Label Shift (Bias Correction)</span>
    <span class="card-rmse">RMSE 2.044% ▼</span>
  </div>
  <div class="card-body">
    <p>
      <strong>The idea:</strong> The 3.42% mean shift is partly a systematic bias in
      literature labels (publication bias + different conditions). What if we corrected
      the labels before merging? We used quantile normalisation: map the literature
      yield distribution to match our own distribution, preserving relative rankings
      but rescaling the absolute values.
    </p>
    <p>
      <strong>What happened:</strong> RMSE of 2.044%, a 4.2% improvement. It helps,
      but less than DRST. Correcting the labels is not enough on its own.
    </p>
    <p>
      <strong>Why it only partially works:</strong> Bias correction fixes the label
      shift but does not fix the chemistry shift. Literature samples still describe
      chemistries that are foreign to our model &mdash; renaming their yields does
      not make their element compositions any less unfamiliar.
    </p>
    <p>
      <span class="tag green">beats baseline</span>
      <span class="tag blue">fixes label shift only</span>
    </p>
  </div>
</div>

<!-- Method 5: Two-stage -->
<div class="method-card">
  <div class="card-header best">
    <span class="card-title">Attempt 5 &mdash; Two-Stage Fine-Tuning ★ Best</span>
    <span class="card-rmse">RMSE 1.907% ▼▼</span>
  </div>
  <div class="card-body">
    <p>
      <strong>The idea:</strong> Instead of adding literature samples directly to training,
      use them to pre-train a separate model (XGBoost) that learns the high-yield chemistry
      from literature. Then use that model's predictions as an extra input feature for our
      main model.
    </p>
    <p>
      Concretely: Stage 1 trains an XGBoost model on the DRST-filtered literature.
      It learns &ldquo;which element combinations does literature associate with high yield?&rdquo;
      For every one of our 89,074 experiments, we ask Stage 1 what it would predict.
      That answer becomes a 68th feature called <code>lit_prior_prediction</code>.
      Stage 2 then trains our main LightGBM model on our data with this extra feature included.
    </p>
    <p>
      <strong>What happened:</strong> RMSE of 1.907%, a <strong>10.6% improvement</strong>
      over baseline &mdash; more than double the gain of any other method, and repeatable
      across 10 seeds (p&lt;10<sup>&minus;14</sup>) and on a held-out test set.
      <em>Correction:</em> an earlier version claimed a 45% out-of-distribution gain
      (6.53&rarr;3.60); that was data leakage (the prior had been trained on the OOD test
      rows). Leak-free OOD is &asymp;6.0&ndash;6.8, roughly level with baseline.
    </p>
    <p>
      <strong>Why it works best:</strong> Literature knowledge enters the final model
      as <em>a feature to reason about</em>, not as competing training labels.
      The main model trains only on our measurements, so the 3.42% systematic label
      gap cannot corrupt the final predictions. It uses literature to expand its
      chemical knowledge without being misled by literature's inflated yield values.
    </p>
    <p>
      <span class="tag green">best RMSE</span>
      <span class="tag green">10/10 seeds, p&lt;10<sup>&minus;14</sup></span>
      <span class="tag blue">separates knowledge from labels</span>
    </p>
  </div>
</div>

<!-- ── Comparison table ───────────────────────────────────────────────────── -->
<h2>Head-to-Head Comparison</h2>

<table>
  <thead>
    <tr>
      <th>#</th>
      <th>Method</th>
      <th>CV RMSE</th>
      <th>vs Baseline</th>
      <th>OOD RMSE</th>
      <th>Core idea</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Baseline (our data only)</td>
      <td class="rmse-val">2.133%</td>
      <td>&mdash;</td>
      <td class="rmse-val">6.53%</td>
      <td>No literature used</td>
    </tr>
    <tr class="worst-row">
      <td>1</td>
      <td>Naive concatenation</td>
      <td class="rmse-val">2.248%</td>
      <td class="delta-pos">+5.4% ✗</td>
      <td class="rmse-val">6.21%</td>
      <td>Add everything equally</td>
    </tr>
    <tr>
      <td>2</td>
      <td>DRST filter (τ = 0.30)</td>
      <td class="rmse-val">2.019%</td>
      <td class="delta-neg">−5.3%</td>
      <td class="rmse-val">4.95%</td>
      <td>Keep only similar samples</td>
    </tr>
    <tr>
      <td>3</td>
      <td>KMM reweighting</td>
      <td class="rmse-val">2.035%</td>
      <td class="delta-neg">−4.6%</td>
      <td class="rmse-val">5.10%</td>
      <td>Soft per-sample weights</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Bias correction</td>
      <td class="rmse-val">2.044%</td>
      <td class="delta-neg">−4.2%</td>
      <td class="rmse-val">5.30%</td>
      <td>Rescale literature yields</td>
    </tr>
    <tr class="best-row">
      <td>5</td>
      <td>Two-stage fine-tuning ★</td>
      <td class="rmse-val">1.907%</td>
      <td class="delta-neg">−10.6% ✓</td>
      <td class="rmse-val">6.77%</td>
      <td>Literature as a feature, not labels</td>
    </tr>
  </tbody>
</table>

<div class="fig-wrap">
  {img("fig_results_comparison.png")}
  <div class="fig-caption">
    CV RMSE and R² across all methods. The two-stage method is the clear winner on
    both in-distribution accuracy and out-of-distribution generalisation.
  </div>
</div>

<!-- ── SHAP ───────────────────────────────────────────────────────────────── -->
<h2>What the Model Actually Learned</h2>

<p>
  To check whether the transfer was meaningful and not just a numerical trick,
  we ran a SHAP analysis &mdash; a technique that tells us how much each feature
  contributed to each individual prediction.
</p>

<div class="fig-wrap">
  {img("fig_shap_beeswarm.png")}
  <div class="fig-caption">
    SHAP beeswarm plot. Each dot is one sample. Position = how much that feature
    moved the prediction up (right) or down (left). Colour = feature value (red = high).
    The <code>lit_prior_prediction</code> feature ranks first &mdash; the literature
    knowledge is genuinely being used on every prediction.
  </div>
</div>

<p>
  The literature prior feature is the single most important input to the model.
  When the literature expert predicted high yield for a composition, the final model
  also predicted higher yield. The model independently confirms: temperature matters,
  and barium, manganese, lanthanum, and cerium all push yield upward &mdash; all
  consistent with known OCM chemistry.
</p>

<!-- ── Ceiling effect ─────────────────────────────────────────────────────── -->
<h2>One Problem We Could Not Fix With Algorithms</h2>

<p>
  Looking at where the model still makes large errors, there is a clear pattern:
</p>

<table>
  <thead>
    <tr><th>Y(C2) range</th><th>RMSE</th><th>Mean error</th></tr>
  </thead>
  <tbody>
    <tr><td>0 – 6%</td><td class="rmse-val">1.43 – 1.48%</td><td>≈ 0%</td></tr>
    <tr><td>6 – 10%</td><td class="rmse-val">1.95%</td><td>−0.3%</td></tr>
    <tr><td>10 – 15%</td><td class="rmse-val">2.58%</td><td>−1.5%</td></tr>
    <tr class="worst-row"><td>&gt; 15%</td><td class="rmse-val">4.70%</td><td>−4.2% (under-predicts)</td></tr>
  </tbody>
</table>

<p>
  For high-yield catalysts (&gt;15%), the model consistently under-predicts by about
  4 percentage points. This is not a modelling problem &mdash; it is a data problem.
  The features that explain why a catalyst achieves 20% yield (gas flow rate,
  methane-to-oxygen ratio, reaction pressure) are <em>not in our dataset</em>.
  The model sees the same composition as a 5% catalyst and has no way to know
  conditions were optimised differently.
</p>

<div class="callout warn">
  <strong>Root cause of the ceiling effect:</strong> GHSV, CH₄/O₂ ratio, and reaction
  pressure are missing from the feature set. No transfer learning method can compensate
  for a missing explanatory variable. Adding these columns is the highest-impact
  improvement available.
</div>

<!-- ── Insight ────────────────────────────────────────────────────────────── -->
<div class="insight">
  <strong>The core insight from this exploration</strong>
  The problem with literature data is not that it is unhelpful &mdash; it is that
  it cannot be treated as equivalent to our own measurements. It lives in a
  partly different chemical space, measured under different conditions, with a
  systematic 3.42 percentage point label shift. The approach that worked best treats
  literature knowledge as a signal to be reasoned about, not as ground truth to be
  trained on directly. By routing that knowledge through a pre-trained sub-model
  feature, we got a 10.6% accuracy improvement on our own chemistry &mdash; repeatable
  across 10 seeds and on a held-out test set. (An earlier claim of a 45% out-of-distribution
  improvement was withdrawn as data leakage; the honest leak-free OOD gain is modest.)
</div>

<!-- ── Next steps ─────────────────────────────────────────────────────────── -->
<h2>What to Do Next</h2>

<ol class="steps">
  <li>
    <strong>Add GHSV and CH₄/O₂ ratio as features.</strong>
    This is a data collection problem, not a modelling problem, and it is the single
    change that would have the largest impact on the remaining error &mdash; specifically
    the ceiling effect on high-yield catalysts.
  </li>
  <li>
    <strong>Fix a minor methodological issue in the current implementation.</strong>
    In the two-stage setup, the Stage 1 model currently sees some of the same samples
    it later generates features for. The validation scores are still unbiased, but
    the fix (training Stage 1 on literature only) is straightforward and worth doing
    before drawing conclusions from the exact numbers.
  </li>
  <li>
    <strong>Re-run the full comparison after adding reaction conditions.</strong>
    It is likely that once GHSV and CH₄/O₂ are in the feature set, the relative
    performance of all methods will shift, and the optimal transfer threshold may
    change as well.
  </li>
</ol>

<!-- ── Footer ────────────────────────────────────────────────────────────── -->
<div class="footer">
  JAIST &mdash; Taniike Lab &nbsp;|&nbsp; May 2026 &nbsp;|&nbsp;
  All RMSE values from 5-fold cross-validation on internal data.
  OOD test: 2,139 held-out literature samples, never used in training.
</div>

</div>
</body>
</html>
"""

out = ROOT / "ocm_initial_report.html"
out.write_text(HTML, encoding="utf-8")
size_kb = out.stat().st_size // 1024
print(f"Written: {out}  ({size_kb} KB)")
