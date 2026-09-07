"""Build a horizontal slide-based HTML presentation for OCM catalyst analysis."""
import base64, pathlib

ROOT = pathlib.Path(__file__).parent

def b64(name: str) -> str:
    return base64.b64encode((ROOT / name).read_bytes()).decode()

def img(name: str, cls: str = "slide-fig") -> str:
    return f'<img src="data:image/png;base64,{b64(name)}" class="{cls}" alt="">'

# ── Slides ────────────────────────────────────────────────────────────────────

S1 = """
<section class="slide cover active">
  <div class="cover-body">
    <span class="pill">JAIST · Taniike Lab · 2026</span>
    <h1>OCM Catalyst Prediction</h1>
    <p class="cover-sub">
      Can 40 years of published literature improve a model<br>
      trained on 89,074 internal experiments?
    </p>
    <div class="cover-rule"></div>
    <p class="cover-meta">Literature Transfer Analysis — Progress Report</p>
  </div>
</section>"""

S2 = """
<section class="slide">
  <div class="inner">
    <h2>Two datasets — very different worlds</h2>
    <div class="cols-2 mt-4">
      <div class="data-card blue-card">
        <div class="eyebrow">Internal Data</div>
        <div class="hero-n">89,074</div>
        <div class="hero-sub">samples</div>
        <ul class="fact-list">
          <li>Our lab · 2025</li>
          <li>Impregnation synthesis only</li>
          <li>Mean Y(C₂) = <b>5.25 %</b></li>
          <li>Systematic, unbiased screen</li>
        </ul>
      </div>
      <div class="data-card amber-card">
        <div class="eyebrow">Published Literature Data</div>
        <div class="hero-n">3,852</div>
        <div class="hero-sub">samples</div>
        <ul class="fact-list">
          <li>Published 1982 – 2019</li>
          <li>15 + synthesis methods</li>
          <li>Mean Y(C₂) = <b>8.67 %</b></li>
          <li>Publication bias — labs report wins</li>
        </ul>
      </div>
    </div>
    <div class="callout mt-4">
      <b>Goal:</b> extract useful chemistry from literature without
      contaminating our model's accuracy on internal data.
    </div>
  </div>
</section>"""

S3 = f"""
<section class="slide">
  <div class="inner">
    <h2>The domain gap is real and large</h2>
    <div class="cols-2 mt-4 align-mid">
      <div class="fig-wrap">
        {img("fig_pca_domain_gap.png")}
        <p class="fig-cap">PCA of all 92,926 samples. Blue = lab data, orange = literature data.</p>
      </div>
      <div>
        <p class="lead">The two datasets live in<br><b>different corners of chemical space.</b></p>
        <div class="big-stat mt-3">
          <span class="big-n">78.5 %</span>
          <span class="big-desc">of literature samples are out-of-distribution relative to our lab</span>
        </div>
        <ul class="body-list mt-3">
          <li>Different element profiles — Ba, Ce (literature) vs Na, Mn (ours)</li>
          <li>Different synthesis routes alter catalyst microstructure</li>
          <li>Literature Y(C₂) is <b>65 % higher</b> on average — publication bias</li>
        </ul>
        <div class="callout warn mt-3">
          Naïve concatenation teaches the model patterns from a different world.
        </div>
      </div>
    </div>
  </div>
</section>"""

S4 = """
<section class="slide">
  <div class="inner">
    <h2>Every standard approach made things worse</h2>
    <table class="res-table mt-4">
      <thead>
        <tr><th>Method</th><th>Idea</th><th>CV RMSE</th><th>vs Baseline</th></tr>
      </thead>
      <tbody>
        <tr class="tr-base">
          <td><b>Baseline</b></td><td>Lab data only — no literature</td>
          <td><b>2.133</b></td><td>—</td>
        </tr>
        <tr class="tr-bad">
          <td>Naïve merge</td><td>Add all literature as training rows</td>
          <td>2.241</td><td class="td-bad">+5.1 % ↑</td>
        </tr>
        <tr class="tr-bad">
          <td>Preparation filter</td><td>Keep only Impregnation literature</td>
          <td>2.214</td><td class="td-bad">+3.8 % ↑</td>
        </tr>
        <tr class="tr-bad">
          <td>DRST filter</td><td>ML similarity filtering — keep top 20 %</td>
          <td>2.163</td><td class="td-bad">+1.4 % ↑</td>
        </tr>
        <tr class="tr-bad">
          <td>KMM weighting</td><td>Per-sample importance weights</td>
          <td>2.261</td><td class="td-bad">+6.0 % ↑</td>
        </tr>
        <tr class="tr-good">
          <td><b>Two-stage (PFT)</b></td><td>Literature as a feature, not a label</td>
          <td><b>1.912</b></td><td class="td-good">−9.7 % ↓</td>
        </tr>
      </tbody>
    </table>
    <p class="fig-cap mt-3">
      <b>Row-level 5-fold CV — historical only.</b> This split lets the same catalyst appear in both
      training and validation. Under catalyst-grouped CV the baseline is 2.9425 and PFT is 2.9955,
      i.e. <b>1.8 % worse</b>. See the next-but-one slide.
    </p>
  </div>
</section>"""

S5 = """
<section class="slide">
  <div class="inner">
    <h2>The method that exposed the leak — literature as a feature, not a label</h2>
    <div class="pipeline mt-4">
      <div class="pnode blue-node">
        <div class="pnode-title">Literature Data</div>
        <div class="pnode-sub">3,852 samples<br>noisy labels, publication bias</div>
      </div>
      <div class="parr">→</div>
      <div class="pnode grey-node">
        <div class="pnode-title">Stage 1: Pre-train</div>
        <div class="pnode-sub">XGBoost learns broad OCM patterns — which elements help, temperature effects</div>
      </div>
      <div class="parr">→</div>
      <div class="pnode grey-node">
        <div class="pnode-title">Prior Prediction</div>
        <div class="pnode-sub">Apply to our samples → one new feature column per sample</div>
      </div>
    </div>
    <div class="pipe-join">↓ added as extra input feature ↓</div>
    <div class="pipeline">
      <div class="pnode green-node">
        <div class="pnode-title">Lab Data</div>
        <div class="pnode-sub">89,074 samples<br>clean, unbiased labels</div>
      </div>
      <div class="parr">→</div>
      <div class="pnode gold-node">
        <div class="pnode-title">Stage 2: Fine-tune</div>
        <div class="pnode-sub">LightGBM trained on our labels only — learns when to trust or override the prior</div>
      </div>
      <div class="parr">→</div>
      <div class="pnode green-node">
        <div class="pnode-title">Final Model</div>
        <div class="pnode-sub">Literature noise never touches training targets</div>
      </div>
    </div>
    <div class="callout green mt-4">
      <b>Result:</b> the literature prior became the <b>#1 most important feature</b> out of 68 —
      Stage 1 captured real OCM chemistry.
    </div>
  </div>
</section>"""

S6 = f"""
<section class="slide">
  <div class="inner">
    <h2>Results</h2>
    <div class="cols-2 mt-4 align-mid">
      <div>
        <div class="kpi-grid">
          <div class="kpi">
            <div class="kpi-n">+1.8 %</div>
            <div class="kpi-label">PFT vs baseline, catalyst-grouped</div>
            <div class="kpi-sub">RMSE 2.9425 → 2.9955 — <b>worse</b></div>
          </div>
          <div class="kpi">
            <div class="kpi-n green">3.77×</div>
            <div class="kpi-label">enrichment, equal-effort set</div>
            <div class="kpi-sub">Spearman 0.724 on 771 catalysts</div>
          </div>
        </div>
        <div class="callout mt-4">
          The published <b>−9.7 %</b> was catalyst-identity leakage: under a row split the Stage-1
          prior had already seen the test catalysts' own yields. The same models, same data, under
          catalyst-grouped CV are <b>1.8 % worse</b> than baseline. The difference is the protocol,
          not the modelling.
        </div>
        <div class="callout green mt-4">
          What survives: the composition-only model still ranks <b>unseen</b> catalysts usefully —
          enrichment 3.77× (95 % CI 3.04–4.89×) — and that holds under either reading of the open
          question about how the data was collected.
        </div>
      </div>
      <div class="fig-wrap">
        {img("fig_protocol_comparison.png")}
      </div>
    </div>
  </div>
</section>"""

S7 = f"""
<section class="slide">
  <div class="inner">
    <h2>Does the model agree with known OCM chemistry?</h2>
    <div class="cols-2 mt-4 align-mid">
      <div class="fig-wrap">
        {img("fig_shap_beeswarm.png")}
        <p class="fig-cap">SHAP values — red = high feature value, right = raises predicted Y(C₂).</p>
      </div>
      <div>
        <p class="lead">SHAP explains <em>why</em> each prediction was made,
        letting us validate against chemistry knowledge.</p>
        <div class="check-list mt-3">
          <div class="chk chk-ok">
            ✓ <b>Ba, Mn</b> high loadings → positive impact<br>
            <span class="muted-sm">Consistent with known Mn–Na–W–O catalyst families</span>
          </div>
          <div class="chk chk-ok">
            ✓ <b>La, Ce</b> positive at moderate loadings<br>
            <span class="muted-sm">Rare-earth promoters known to improve selectivity</span>
          </div>
          <div class="chk chk-ok">
            ✓ <b>Literature prior</b> is the #1 most important feature<br>
            <span class="muted-sm">Stage 1 pre-training learned genuine OCM patterns</span>
          </div>
          <div class="chk chk-warn">
            ⚠ Under-predicts when Y(C₂) &gt; 15 %<br>
            <span class="muted-sm">Dataset missing GHSV / CH₄:O₂ — known limitation</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>"""

S8 = """
<section class="slide">
  <div class="inner">
    <h2>What should we prioritise next?</h2>
    <div class="cols-3 mt-4">
      <div class="opt-card">
        <span class="opt-badge green-badge">Priority 1</span>
        <h3>Ask for the condition columns</h3>
        <p>One email to JAIST. Each catalyst is run at 5 temperatures under ~27 unrecorded condition settings. Recovering them turns 917 training examples back into 89,074 and unlocks 19.9 % of currently unreachable variance.</p>
        <div class="opt-time">one email</div>
      </div>
      <div class="opt-card">
        <span class="opt-badge amber-badge">Priority 2</span>
        <h3>Settle conditions vs time-on-stream</h3>
        <p>Provably unanswerable from the file: every cell is stored sorted by yield, so row order records rank, not acquisition order. Only the lab can answer. The risk is already bounded — the shortlist barely moves either way.</p>
        <div class="opt-time">same email</div>
      </div>
      <div class="opt-card">
        <span class="opt-badge blue-badge">Priority 3</span>
        <h3>Re-scope the campaign</h3>
        <p>20 runs per catalyst instead of 135 reproduces the full ranking at ρ = 0.949 — roughly 72 catalysts screened plus a randomised control arm, for the reactor budget of 17 exhaustive ones.</p>
        <div class="opt-time">before reactor time</div>
      </div>
    </div>
    <div class="callout mt-4">
      <b>Today's question:</b> all three are questions for the lab rather than modelling work.
      The modelling has gone as far as this feature set allows.
    </div>
  </div>
</section>"""

SLIDES = [S1, S2, S3, S4, S5, S6, S7, S8]
N = len(SLIDES)
DOTS = "".join(
    f'<span class="dot{" active" if i == 0 else ""}" onclick="go({i})"></span>'
    for i in range(N)
)

# ── Full HTML ─────────────────────────────────────────────────────────────────

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OCM Catalyst — Literature Transfer</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

:root {{
  --navy:  #0F2044;
  --blue:  #1D4ED8;
  --green: #059669;
  --amber: #B45309;
  --red:   #DC2626;
  --bg:    #FFFFFF;
  --surf:  #F1F5F9;
  --text:  #0F172A;
  --muted: #64748B;
  --bdr:   #E2E8F0;
  --nav:   52px;
}}

html, body {{
  height: 100%; width: 100%;
  overflow: hidden;
  background: #0a1628;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  font-size: 15px;
  color: var(--text);
  line-height: 1.55;
}}

/* ── Slide deck ── */
.deck {{
  position: fixed;
  top: 0; left: 0; right: 0;
  bottom: var(--nav);
  overflow: hidden;
}}

.slide {{
  position: absolute;
  inset: 0;
  background: var(--bg);
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.22s ease;
}}
.slide.active {{ opacity: 1; pointer-events: all; }}

/* ── Cover ── */
.cover {{ background: var(--navy); }}
.cover-body {{
  text-align: center;
  max-width: 660px;
  padding: 0 40px;
}}
.pill {{
  display: inline-block;
  background: rgba(255,255,255,.12);
  color: rgba(255,255,255,.85);
  padding: 4px 14px;
  border-radius: 20px;
  font-size: 0.75rem;
  letter-spacing: .07em;
  text-transform: uppercase;
  margin-bottom: 22px;
}}
.cover h1 {{
  font-size: 2.6rem;
  font-weight: 700;
  color: #fff;
  line-height: 1.15;
  margin-bottom: 18px;
}}
.cover-sub {{
  font-size: 1.1rem;
  color: rgba(255,255,255,.75);
  line-height: 1.65;
  margin-bottom: 28px;
}}
.cover-rule {{
  width: 48px; height: 3px;
  background: rgba(255,255,255,.3);
  margin: 0 auto 22px;
  border-radius: 2px;
}}
.cover-meta {{
  font-size: 0.82rem;
  color: rgba(255,255,255,.45);
  letter-spacing: .03em;
}}

/* ── Content slides ── */
.inner {{
  width: 100%;
  max-width: 1080px;
  padding: 40px 64px;
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: center;
}}

h2 {{
  font-size: 1.6rem;
  font-weight: 700;
  color: var(--navy);
  padding-bottom: 12px;
  border-bottom: 3px solid var(--blue);
  flex-shrink: 0;
}}

/* ── Grid layouts ── */
.cols-2 {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 32px;
}}
.cols-3 {{
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 20px;
}}
.align-mid {{ align-items: center; }}
.mt-3 {{ margin-top: 18px; }}
.mt-4 {{ margin-top: 26px; }}

/* ── Dataset cards ── */
.data-card {{
  border-radius: 10px;
  padding: 24px 28px;
  border: 1px solid var(--bdr);
}}
.blue-card  {{ background: #EFF6FF; border-left: 5px solid var(--blue); }}
.amber-card {{ background: #FFFBEB; border-left: 5px solid var(--amber); }}
.eyebrow {{
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: var(--muted);
  margin-bottom: 6px;
}}
.hero-n {{
  font-size: 2.8rem;
  font-weight: 800;
  color: var(--navy);
  line-height: 1;
}}
.hero-sub {{
  font-size: 0.85rem;
  color: var(--muted);
  margin-bottom: 14px;
}}
.fact-list {{
  list-style: none;
  font-size: 0.88rem;
  line-height: 1.9;
}}
.fact-list li::before {{ content: "· "; color: var(--muted); }}

/* ── Callout ── */
.callout {{
  background: #EFF6FF;
  border-left: 4px solid var(--blue);
  padding: 11px 16px;
  border-radius: 0 6px 6px 0;
  font-size: 0.88rem;
}}
.callout.green {{ background: #ECFDF5; border-left-color: var(--green); }}
.callout.warn  {{ background: #FFFBEB; border-left-color: var(--amber); }}

/* ── Domain gap slide ── */
.lead {{ font-size: 1.05rem; line-height: 1.6; }}
.big-stat {{
  display: flex;
  align-items: baseline;
  gap: 12px;
  background: var(--surf);
  padding: 12px 16px;
  border-radius: 8px;
}}
.big-n   {{ font-size: 2rem; font-weight: 800; color: var(--blue); }}
.big-desc {{ font-size: 0.82rem; color: var(--muted); line-height: 1.4; }}
.body-list {{
  padding-left: 18px;
  font-size: 0.88rem;
  line-height: 1.9;
}}

/* ── Results table ── */
.res-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
}}
.res-table th {{
  background: var(--navy);
  color: #fff;
  padding: 9px 14px;
  text-align: left;
  font-weight: 600;
}}
.res-table td {{ padding: 9px 14px; border-bottom: 1px solid var(--bdr); }}
.tr-base td {{ background: #F8FAFC; }}
.tr-bad  td {{ color: var(--muted); }}
.tr-good td {{ background: #ECFDF5; font-weight: 600; }}
.td-bad  {{ color: var(--red);   font-weight: 700; }}
.td-good {{ color: var(--green); font-weight: 700; }}
.fig-cap {{
  font-size: 0.75rem;
  color: var(--muted);
  font-style: italic;
  text-align: center;
  margin-top: 6px;
}}

/* ── Pipeline diagram ── */
.pipeline {{ display: flex; align-items: stretch; }}
.pnode {{
  flex: 1;
  padding: 14px 16px;
  border-radius: 8px;
  text-align: center;
}}
.pnode-title {{ font-size: 0.82rem; font-weight: 700; margin-bottom: 5px; }}
.pnode-sub   {{ font-size: 0.74rem; line-height: 1.4; opacity: .85; }}
.blue-node  {{ background: #DBEAFE; }}
.grey-node  {{ background: #F1F5F9; }}
.green-node {{ background: #D1FAE5; }}
.gold-node  {{ background: #FEF3C7; }}
.parr {{
  display: flex;
  align-items: center;
  padding: 0 8px;
  font-size: 1.3rem;
  color: var(--muted);
}}
.pipe-join {{
  text-align: center;
  color: var(--blue);
  font-size: 0.8rem;
  font-weight: 600;
  padding: 7px 0;
  letter-spacing: .04em;
}}

/* ── KPI grid ── */
.kpi-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}}
.kpi {{
  background: var(--surf);
  border-radius: 10px;
  padding: 18px 20px;
  text-align: center;
}}
.kpi-n {{
  font-size: 2.4rem;
  font-weight: 800;
  line-height: 1;
}}
.kpi-n.green {{ color: var(--green); }}
.kpi-label   {{ font-size: 0.82rem; font-weight: 600; margin: 4px 0 2px; }}
.kpi-sub     {{ font-size: 0.75rem; color: var(--muted); }}

/* ── Figures ── */
.fig-wrap {{ text-align: center; }}
.slide-fig {{
  max-width: 100%;
  max-height: 330px;
  object-fit: contain;
  border: 1px solid var(--bdr);
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0,0,0,.07);
}}

/* ── SHAP checklist ── */
.check-list {{ display: flex; flex-direction: column; gap: 10px; }}
.chk {{
  padding: 9px 13px;
  border-radius: 6px;
  font-size: 0.86rem;
  line-height: 1.5;
}}
.chk-ok   {{ background: #ECFDF5; border-left: 3px solid var(--green); }}
.chk-warn {{ background: #FFFBEB; border-left: 3px solid var(--amber); }}
.muted-sm {{ font-size: 0.8rem; color: var(--muted); }}

/* ── Option cards ── */
.opt-card {{
  background: var(--bg);
  border: 1px solid var(--bdr);
  border-radius: 10px;
  padding: 20px 18px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,.05);
}}
.opt-badge {{
  display: inline-block;
  padding: 3px 11px;
  border-radius: 20px;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: .05em;
  text-transform: uppercase;
  width: fit-content;
}}
.blue-badge  {{ background: #DBEAFE; color: var(--blue);  }}
.amber-badge {{ background: #FEF3C7; color: var(--amber); }}
.green-badge {{ background: #D1FAE5; color: var(--green); }}
.opt-card h3 {{
  font-size: 1rem;
  font-weight: 700;
  color: var(--navy);
}}
.opt-card p {{
  font-size: 0.85rem;
  color: var(--muted);
  line-height: 1.55;
  flex: 1;
}}
.opt-time {{
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--muted);
  border-top: 1px solid var(--bdr);
  padding-top: 8px;
}}

/* ── Navigation bar ── */
.nav-bar {{
  position: fixed;
  bottom: 0; left: 0; right: 0;
  height: var(--nav);
  background: var(--navy);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  z-index: 100;
}}
.nav-btn {{
  background: rgba(255,255,255,.1);
  border: 1px solid rgba(255,255,255,.2);
  color: #fff;
  padding: 6px 20px;
  border-radius: 5px;
  cursor: pointer;
  font-size: 0.83rem;
  font-family: inherit;
  transition: background .15s;
}}
.nav-btn:hover    {{ background: rgba(255,255,255,.2); }}
.nav-btn:disabled {{ opacity: .3; cursor: default; }}
.dots {{ display: flex; gap: 8px; align-items: center; }}
.dot {{
  width: 7px; height: 7px;
  border-radius: 50%;
  background: rgba(255,255,255,.25);
  cursor: pointer;
  transition: background .15s, transform .15s;
}}
.dot.active {{ background: #fff; transform: scale(1.35); }}
.counter {{ color: rgba(255,255,255,.5); font-size: 0.8rem; min-width: 44px; text-align: right; }}
</style>
</head>
<body>

<div class="deck">
{"".join(SLIDES)}
</div>

<nav class="nav-bar">
  <button class="nav-btn" id="prev" onclick="go(cur-1)">← Prev</button>
  <div class="dots">{DOTS}</div>
  <div style="display:flex;align-items:center;gap:16px">
    <span class="counter" id="counter">1 / {N}</span>
    <button class="nav-btn" id="next" onclick="go(cur+1)">Next →</button>
  </div>
</nav>

<script>
const slides = document.querySelectorAll('.slide');
const dots   = document.querySelectorAll('.dot');
const prev   = document.getElementById('prev');
const next   = document.getElementById('next');
const ctr    = document.getElementById('counter');
let cur = 0;

function go(n) {{
  slides[cur].classList.remove('active');
  dots[cur].classList.remove('active');
  cur = Math.max(0, Math.min(n, slides.length - 1));
  slides[cur].classList.add('active');
  dots[cur].classList.add('active');
  ctr.textContent = (cur + 1) + ' / ' + slides.length;
  prev.disabled = cur === 0;
  next.disabled = cur === slides.length - 1;
}}

document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowRight' || e.key === ' ')  {{ e.preventDefault(); go(cur + 1); }}
  if (e.key === 'ArrowLeft')                     {{ e.preventDefault(); go(cur - 1); }}
}});

go(0);
</script>
</body>
</html>
"""

out = ROOT / "ocm_presentation.html"
out.write_text(HTML, encoding="utf-8")
print(f"Written: {out}  ({out.stat().st_size / 1024:.0f} KB)")
