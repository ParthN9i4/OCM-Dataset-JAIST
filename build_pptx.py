"""
Build ocm_presentation.pptx — 12-slide PowerPoint deck.

Mirrors the 8-chapter walkthrough notebook narrative:
  1  Title
  2  The Problem — two datasets, three systematic differences
  3  Visual evidence — KDE + PCA + element usage
  4  How We Measure Success — asymmetric 5-fold CV
  5  Baseline + Why Naive Merging Fails
  6  DRST — filtering by chemical similarity
  7  KMM — soft weights, same conclusion
  8  Prior Feature Transfer pipeline
  9  Results — all 5 methods bar chart
 10  SHAP — did it learn real chemistry
 11  Limitations & Next Steps
 12  Summary

Run: python build_pptx.py
Produces: ocm_presentation.pptx  (and ocm_presentation.pdf via LibreOffice)
"""

import pathlib
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

ROOT = pathlib.Path(__file__).parent

# ── Colour palette ──────────────────────────────────────────────────────────
NAVY   = RGBColor(0x0F, 0x1E, 0x42)
ACCENT = RGBColor(0x00, 0x78, 0xC8)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT  = RGBColor(0xF5, 0xF6, 0xF8)
GREEN  = RGBColor(0x10, 0xB9, 0x81)
RED    = RGBColor(0xDC, 0x3C, 0x3C)
AMBER  = RGBColor(0xF5, 0x9E, 0x0B)
DGRAY  = RGBColor(0x50, 0x59, 0x6E)
LGRAY  = RGBColor(0xD0, 0xD5, 0xDF)
BLACK  = RGBColor(0x1A, 0x1A, 0x2E)
GDARK  = RGBColor(0x1B, 0x5E, 0x20)
RDARK  = RGBColor(0x8B, 0x1A, 0x1A)

W = Inches(13.33)
H = Inches(7.5)

prs = Presentation()
prs.slide_width  = W
prs.slide_height = H
BLANK = prs.slide_layouts[6]

# ── Helpers ──────────────────────────────────────────────────────────────────
def add_rect(slide, x, y, w, h, fill=None, line=None):
    shape = slide.shapes.add_shape(1, x, y, w, h)
    if fill:
        shape.fill.solid(); shape.fill.fore_color.rgb = fill
    else:
        shape.fill.background()
    if line:
        shape.line.color.rgb = line; shape.line.width = Pt(0.75)
    else:
        shape.line.fill.background()
    return shape

def add_text(slide, text, x, y, w, h, size=14, bold=False, color=BLACK,
             align=PP_ALIGN.LEFT, wrap=True, italic=False):
    txb = slide.shapes.add_textbox(x, y, w, h)
    tf  = txb.text_frame; tf.word_wrap = wrap
    p   = tf.paragraphs[0]; p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size  = Pt(size)
    run.font.bold  = bold
    run.font.color.rgb = color
    run.font.italic = italic
    return txb

def add_img(slide, fname, x, y, w, h=None):
    path = ROOT / fname
    if not path.exists():
        return
    if h:
        slide.shapes.add_picture(str(path), x, y, w, h)
    else:
        slide.shapes.add_picture(str(path), x, y, w)

def nav_bar(slide, current, total):
    add_rect(slide, 0, H - Inches(0.38), W, Inches(0.38), fill=NAVY)
    add_text(slide, f"OCM Dataset Integration   |   Slide {current} / {total}",
             Inches(0.2), H - Inches(0.36), W - Inches(0.4), Inches(0.34),
             size=9, color=RGBColor(0xCC, 0xCC, 0xCC))

def slide_title_bar(slide, title, subtitle=None):
    bar_h = Inches(0.95) if not subtitle else Inches(1.2)
    add_rect(slide, 0, 0, W, bar_h, fill=NAVY)
    add_text(slide, title, Inches(0.4), Inches(0.18), W - Inches(0.8), Inches(0.55),
             size=22, bold=True, color=WHITE)
    if subtitle:
        add_text(slide, subtitle, Inches(0.4), Inches(0.7), W - Inches(0.8), Inches(0.45),
                 size=12, color=RGBColor(0xBB, 0xCC, 0xEE), italic=True)

def bullets(slide, items, x, y, w, h, size=12, spacing=0.34, color=BLACK):
    for i, item in enumerate(items):
        add_text(slide, "•  " + item, x, y + Inches(i * spacing), w, Inches(spacing + 0.05),
                 size=size, color=color)

def badge(slide, text, x, y, w, h, fill, text_color=WHITE, size=11):
    add_rect(slide, x, y, w, h, fill=fill)
    add_text(slide, text, x + Inches(0.05), y + Inches(0.02),
             w - Inches(0.1), h - Inches(0.04),
             size=size, bold=True, color=text_color, align=PP_ALIGN.CENTER)

def callout(slide, text, x, y, w, h, fill=LIGHT, border=ACCENT,
            text_color=DGRAY, size=11, italic=True):
    add_rect(slide, x, y, w, h, fill=fill, line=border)
    add_text(slide, text, x + Inches(0.12), y + Inches(0.08),
             w - Inches(0.24), h - Inches(0.16),
             size=size, color=text_color, italic=italic)

TOTAL = 12

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Title
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
add_rect(s, 0, 0, W, H, fill=NAVY)
add_rect(s, 0, Inches(2.95), W, Inches(0.06), fill=ACCENT)

add_text(s, "JAIST — Taniike Lab",
         Inches(1.2), Inches(1.4), Inches(10.9), Inches(0.5),
         size=13, color=RGBColor(0xAA, 0xBB, 0xDD), align=PP_ALIGN.CENTER)
add_text(s, "Integrating OCM Literature Data\ninto Lab-Scale ML Models",
         Inches(1.2), Inches(1.9), Inches(10.9), Inches(1.6),
         size=36, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
add_text(s, "A Walkthrough of Five Transfer-Learning Approaches",
         Inches(1.2), Inches(3.55), Inches(10.9), Inches(0.6),
         size=16, color=RGBColor(0x99, 0xBB, 0xEE), align=PP_ALIGN.CENTER, italic=True)
add_rect(s, Inches(4.5), Inches(4.35), Inches(4.33), Inches(0.04),
         fill=RGBColor(0x44, 0x66, 0xAA))
add_text(s, "Partha Nupam Nagar",
         Inches(1.2), Inches(4.55), Inches(10.9), Inches(0.45),
         size=15, color=RGBColor(0xCC, 0xDD, 0xFF), align=PP_ALIGN.CENTER)
add_text(s, "June 2026",
         Inches(1.2), Inches(5.0), Inches(10.9), Inches(0.4),
         size=13, color=RGBColor(0x88, 0x99, 0xBB), align=PP_ALIGN.CENTER)
nav_bar(s, 1, TOTAL)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — The Problem (two datasets + three differences)
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
add_rect(s, 0, 0, W, H, fill=WHITE)
slide_title_bar(s, "The Problem — Two Datasets, Three Systematic Differences")
nav_bar(s, 2, TOTAL)

# Left dataset card
add_rect(s, Inches(0.4), Inches(1.2), Inches(6.0), Inches(2.6),
         fill=RGBColor(0xE8, 0xF4, 0xFF), line=ACCENT)
add_rect(s, Inches(0.4), Inches(1.2), Inches(6.0), Inches(0.42), fill=ACCENT)
add_text(s, "Our Lab Data", Inches(0.55), Inches(1.23),
         Inches(5.7), Inches(0.36), size=13, bold=True, color=WHITE)
rows_l = [("Samples", "89,074"), ("Year", "2025"),
          ("Preparation", "Impregnation only"),
          ("Mean Y(C2)", "5.25%"), ("Conditions", "Fixed, one lab")]
for i, (k, v) in enumerate(rows_l):
    add_text(s, k, Inches(0.55), Inches(1.7) + Inches(i * 0.38),
             Inches(2.4), Inches(0.34), size=11, bold=True, color=DGRAY)
    add_text(s, v, Inches(2.95), Inches(1.7) + Inches(i * 0.38),
             Inches(3.3), Inches(0.34), size=11, color=BLACK)

# Right dataset card
add_rect(s, Inches(6.95), Inches(1.2), Inches(6.0), Inches(2.6),
         fill=RGBColor(0xFF, 0xF5, 0xE8), line=AMBER)
add_rect(s, Inches(6.95), Inches(1.2), Inches(6.0), Inches(0.42), fill=AMBER)
add_text(s, "Published Literature (≤2019)", Inches(7.1), Inches(1.23),
         Inches(5.7), Inches(0.36), size=13, bold=True, color=WHITE)
rows_r = [("Samples", "3,852"), ("Year", "1982 – 2019"),
          ("Preparation", "15+ methods"),
          ("Mean Y(C2)", "8.67%"), ("Conditions", "40 years of diverse labs")]
for i, (k, v) in enumerate(rows_r):
    add_text(s, k, Inches(7.1), Inches(1.7) + Inches(i * 0.38),
             Inches(2.4), Inches(0.34), size=11, bold=True, color=DGRAY)
    add_text(s, v, Inches(9.55), Inches(1.7) + Inches(i * 0.38),
             Inches(3.3), Inches(0.34), size=11, color=BLACK)

# Central question
add_text(s,
         "Can we use literature to improve our model — without hurting it?",
         Inches(0.4), Inches(3.95), Inches(12.5), Inches(0.4),
         size=14, bold=True, color=NAVY, align=PP_ALIGN.CENTER, italic=True)

# Three differences
diffs = [
    ("1. Label shift",
     "Literature mean is 3.42 pp higher (t = −32.2, p < 10⁻²⁰⁰) — systematic offset from publication bias and optimised conditions (Fanelli, 2012).",
     RED),
    ("2. Covariate shift",
     "78.5% of literature samples describe chemistry our lab has never tested (Pan & Yang, 2010, IEEE TKDE).",
     AMBER),
    ("3. Publication bias",
     "Literature is skewed, not just shifted — failed experiments are rarely reported, over-weighting high yields.",
     ACCENT),
]
for i, (head, body, color) in enumerate(diffs):
    y0 = Inches(4.5) + Inches(i * 0.62)
    badge(s, head, Inches(0.4), y0, Inches(2.7), Inches(0.42), color, size=12)
    add_text(s, body, Inches(3.25), y0 + Inches(0.04),
             Inches(9.7), Inches(0.4), size=11, color=DGRAY)

# Why ML? callout
callout(s,
        "Why ML? The catalyst search space (65 elements × varying loadings × temperature) is too large to screen physically. "
        "A surrogate model guides experiments toward high-yield compositions (Lookman et al., 2019, npj Comput. Mater.).",
        Inches(0.4), Inches(6.42), Inches(12.5), Inches(0.68),
        fill=LIGHT, border=NAVY, text_color=DGRAY, size=10, italic=False)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — Visual evidence
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
add_rect(s, 0, 0, W, H, fill=WHITE)
slide_title_bar(s, "Visualising the Gap — Label Shift, Covariate Shift, Chemistry")
nav_bar(s, 3, TOTAL)

# Top figure
add_img(s, "fig_walkthrough_label_pca.png", Inches(0.3), Inches(1.1), Inches(8.3))
# Bottom figure
add_img(s, "fig_element_usage.png", Inches(0.3), Inches(4.45), Inches(8.3))

# Right commentary
add_rect(s, Inches(8.9), Inches(1.1), Inches(4.1), Inches(0.4), fill=NAVY)
add_text(s, "Top: label & covariate shift", Inches(9.05), Inches(1.13),
         Inches(3.9), Inches(0.34), size=12, bold=True, color=WHITE)
bullets(s, [
    "Red band = 3.42 pp mean gap",
    "Orange tail = publication-bias skew",
    "PCA: orange spreads outside blue cluster",
], Inches(8.95), Inches(1.6), Inches(4.05), Inches(0.4), size=11, spacing=0.42, color=BLACK)

add_rect(s, Inches(8.9), Inches(4.45), Inches(4.1), Inches(0.4), fill=ACCENT)
add_text(s, "Bottom: which elements differ?", Inches(9.05), Inches(4.48),
         Inches(3.9), Inches(0.34), size=12, bold=True, color=WHITE)
bullets(s, [
    "Different active phases / promoters",
    "Concrete chemistry, not abstract",
    "Justifies selective filtering",
], Inches(8.95), Inches(4.95), Inches(4.05), Inches(0.4), size=11, spacing=0.42, color=BLACK)

callout(s,
        "These three figures jointly forecast that naive concatenation will harm the model — confirmed on slide 5.",
        Inches(8.9), Inches(6.3), Inches(4.1), Inches(0.7),
        fill=RGBColor(0xFF, 0xF0, 0xD8), border=AMBER,
        text_color=RGBColor(0x6B, 0x3A, 0x00), size=10)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — How We Measure Success
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
add_rect(s, 0, 0, W, H, fill=WHITE)
slide_title_bar(s, "How We Measure Success — Asymmetric 5-Fold CV")
nav_bar(s, 4, TOTAL)

# Left: rule + reasoning
add_rect(s, Inches(0.4), Inches(1.2), Inches(7.0), Inches(0.42), fill=NAVY)
add_text(s, "The validation rule", Inches(0.55), Inches(1.23),
         Inches(6.7), Inches(0.36), size=13, bold=True, color=WHITE)
bullets(s, [
    "Our 89,074 samples split into 5 folds",
    "Training fold = 4/5 of our data + any literature",
    "Validation fold = 1/5 of our data only",
    "Literature never appears in validation",
], Inches(0.45), Inches(1.7), Inches(6.9), Inches(0.4), size=12, spacing=0.4)

add_rect(s, Inches(0.4), Inches(3.65), Inches(7.0), Inches(0.42), fill=ACCENT)
add_text(s, "Why asymmetric?", Inches(0.55), Inches(3.68),
         Inches(6.7), Inches(0.36), size=13, bold=True, color=WHITE)
bullets(s, [
    "The question: how well does the model predict OUR experiments?",
    "Mixing literature into validation would dilute the answer",
    "Same CV harness used for every method — only training data changes",
], Inches(0.45), Inches(4.15), Inches(6.9), Inches(0.4), size=12, spacing=0.4)

# Right: metric box + fold diagram
add_rect(s, Inches(7.7), Inches(1.2), Inches(5.3), Inches(1.4),
         fill=RGBColor(0xE8, 0xF5, 0xE9), line=GREEN)
add_text(s, "Metric: RMSE",
         Inches(7.85), Inches(1.3), Inches(5.0), Inches(0.4),
         size=14, bold=True, color=GDARK)
add_text(s, "in units of percentage points of C₂ yield.\nAn RMSE of 2.1 means typical error ≈ 2 pp.",
         Inches(7.85), Inches(1.75), Inches(5.0), Inches(0.75),
         size=11, color=DGRAY)

# Fold visualization
add_text(s, "5-fold split (our data, n=89,074)",
         Inches(7.7), Inches(2.85), Inches(5.3), Inches(0.4),
         size=12, bold=True, color=NAVY)

fold_colors = [ACCENT, ACCENT, RED, ACCENT, ACCENT]
fold_labels = ["fold 1\ntrain", "fold 2\ntrain", "fold 3\nVAL",
               "fold 4\ntrain", "fold 5\ntrain"]
for i, (col, lab) in enumerate(zip(fold_colors, fold_labels)):
    x = Inches(7.7) + Inches(i * 1.05)
    add_rect(s, x, Inches(3.35), Inches(1.0), Inches(0.85), fill=col)
    add_text(s, lab, x, Inches(3.4), Inches(1.0), Inches(0.8),
             size=10, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

add_text(s, "+ literature (training only, never validation)",
         Inches(7.7), Inches(4.4), Inches(5.3), Inches(0.4),
         size=11, italic=True, color=DGRAY, align=PP_ALIGN.CENTER)

callout(s,
        "Repeat 5 times, rotating which fold is the validation set. Report mean RMSE ± std.",
        Inches(7.7), Inches(5.0), Inches(5.3), Inches(0.8),
        fill=LIGHT, border=NAVY, text_color=DGRAY, size=11, italic=True)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — Baseline + Naive Merge
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
add_rect(s, 0, 0, W, H, fill=WHITE)
slide_title_bar(s, "Baseline + Why Naive Merging Fails")
nav_bar(s, 5, TOTAL)

# Left: Step 1 + Step 2 tables
add_text(s, "Step 1 — establish baseline",
         Inches(0.4), Inches(1.3), Inches(6.2), Inches(0.4),
         size=14, bold=True, color=NAVY)
add_rect(s, Inches(0.4), Inches(1.8), Inches(6.2), Inches(0.42), fill=NAVY)
add_text(s, "Method", Inches(0.55), Inches(1.83), Inches(4.0), Inches(0.36),
         size=12, bold=True, color=WHITE)
add_text(s, "CV RMSE", Inches(4.7), Inches(1.83), Inches(1.8), Inches(0.36),
         size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
add_rect(s, Inches(0.4), Inches(2.22), Inches(6.2), Inches(0.5),
         fill=LIGHT, line=LGRAY)
add_text(s, "Baseline (our data only)",
         Inches(0.55), Inches(2.27), Inches(4.0), Inches(0.42),
         size=12, color=BLACK)
add_text(s, "2.133", Inches(4.7), Inches(2.27), Inches(1.8), Inches(0.42),
         size=12, bold=True, color=BLACK, align=PP_ALIGN.CENTER)
add_text(s, "Number every method must beat.",
         Inches(0.4), Inches(2.85), Inches(6.2), Inches(0.4),
         size=10, italic=True, color=DGRAY)

add_text(s, "Step 2 — try the obvious",
         Inches(0.4), Inches(3.55), Inches(6.2), Inches(0.4),
         size=14, bold=True, color=NAVY)
add_rect(s, Inches(0.4), Inches(4.05), Inches(6.2), Inches(0.42), fill=NAVY)
add_text(s, "Method", Inches(0.55), Inches(4.08), Inches(4.0), Inches(0.36),
         size=12, bold=True, color=WHITE)
add_text(s, "CV RMSE", Inches(4.7), Inches(4.08), Inches(1.8), Inches(0.36),
         size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
add_rect(s, Inches(0.4), Inches(4.47), Inches(6.2), Inches(0.5),
         fill=LIGHT, line=LGRAY)
add_text(s, "Baseline", Inches(0.55), Inches(4.52), Inches(4.0), Inches(0.42),
         size=12, color=BLACK)
add_text(s, "2.133", Inches(4.7), Inches(4.52), Inches(1.8), Inches(0.42),
         size=12, color=BLACK, align=PP_ALIGN.CENTER)
add_rect(s, Inches(0.4), Inches(4.97), Inches(6.2), Inches(0.5),
         fill=RGBColor(0xFF, 0xEB, 0xEB), line=RED)
add_text(s, "Naive merge (3,852 lit rows)",
         Inches(0.55), Inches(5.02), Inches(4.0), Inches(0.42),
         size=12, bold=True, color=BLACK)
add_text(s, "2.248", Inches(4.7), Inches(5.02), Inches(1.8), Inches(0.42),
         size=12, bold=True, color=RDARK, align=PP_ALIGN.CENTER)
add_text(s, "+5.4% worse than baseline.",
         Inches(0.4), Inches(5.6), Inches(6.2), Inches(0.4),
         size=12, bold=True, italic=True, color=RDARK)

# Right: explanation
add_rect(s, Inches(7.0), Inches(1.3), Inches(6.0), Inches(0.42), fill=ACCENT)
add_text(s, "Why did it fail?", Inches(7.15), Inches(1.33),
         Inches(5.7), Inches(0.36), size=13, bold=True, color=WHITE)
add_rect(s, Inches(7.0), Inches(1.72), Inches(6.0), Inches(2.7),
         fill=LIGHT, line=LGRAY)
add_text(s,
         "Model sees two catalysts with nearly identical composition:",
         Inches(7.15), Inches(1.85), Inches(5.7), Inches(0.35),
         size=12, color=BLACK)
bullets(s, [
    "One labelled 5.2%  (our lab)",
    "One labelled 8.7%  (literature)",
], Inches(7.25), Inches(2.25), Inches(5.6), Inches(0.4), size=12, spacing=0.4)
add_text(s,
         "No feature explains the gap — flow rate, gas ratio, etc. are absent. The model hedges and predicts both badly.",
         Inches(7.15), Inches(3.1), Inches(5.7), Inches(1.2),
         size=12, color=DGRAY)

callout(s,
        "The 3.42 pp offset is a SYSTEMATIC signal the model cannot account for — and incorrectly tries to fit.\n"
        "Ben-David et al. (2010, Mach. Learn.): when source and target distributions differ significantly, naive combination can perform WORSE than target-only training.\n\n"
        "⇒ We need to be much more selective.",
        Inches(7.0), Inches(4.35), Inches(6.0), Inches(1.9),
        fill=RGBColor(0xFF, 0xF0, 0xD8), border=AMBER,
        text_color=RGBColor(0x6B, 0x3A, 0x00), size=11, italic=False)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — DRST
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
add_rect(s, 0, 0, W, H, fill=WHITE)
slide_title_bar(s, "DRST — Filtering by Chemical Similarity")
nav_bar(s, 6, TOTAL)

add_img(s, "fig_drst_scores.png", Inches(0.3), Inches(1.25), Inches(7.2))
add_text(s, "Histogram of P(ours | x) for every literature sample.  Dashed lines = candidate thresholds.",
         Inches(0.3), Inches(4.5), Inches(7.2), Inches(0.4),
         size=10, italic=True, color=DGRAY, align=PP_ALIGN.CENTER)

# Right: how it works + sweep table
add_rect(s, Inches(7.8), Inches(1.25), Inches(5.2), Inches(0.42), fill=NAVY)
add_text(s, "How it works", Inches(7.95), Inches(1.28),
         Inches(4.9), Inches(0.36), size=13, bold=True, color=WHITE)
bullets(s, [
    "Train logistic classifier (finds element-space boundary):  ours=1, lit=0",
    "Score each lit sample:  P(ours | x)",
    "Keep samples scoring ≥ τ",
], Inches(7.85), Inches(1.75), Inches(5.1), Inches(0.4), size=11, spacing=0.4)
callout(s,
        "Logistic regression = finds a hyperplane in 67-dimensional feature space that best separates our samples from literature. "
        "Ref: Huang et al. (2006, NeurIPS); Sugiyama et al. (2007, JMLR).",
        Inches(7.8), Inches(3.0), Inches(5.2), Inches(0.65),
        fill=RGBColor(0xE8, 0xF5, 0xFF), border=ACCENT, text_color=DGRAY, size=9, italic=False)

# Sweep table (shifted down to make room for callout above)
add_text(s, "Threshold sweep",
         Inches(7.8), Inches(3.75), Inches(5.2), Inches(0.4),
         size=13, bold=True, color=NAVY)
sweep_rows = [
    ("τ = 0.10", "1168 kept", "2.068", False),
    ("τ = 0.20", "987 kept",  "2.034", False),
    ("τ = 0.30", "782 kept",  "2.019", True),
    ("τ = 0.40", "543 kept",  "2.045", False),
]
add_rect(s, Inches(7.8), Inches(4.2), Inches(5.2), Inches(0.4), fill=NAVY)
for j, (h, x, w) in enumerate(zip(["τ", "kept", "RMSE"],
                                    [Inches(7.95), Inches(9.3), Inches(11.4)],
                                    [Inches(1.3), Inches(2.0), Inches(1.5)])):
    add_text(s, h, x, Inches(4.23), w, Inches(0.34),
             size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
for i, (tau, kept, rmse, best) in enumerate(sweep_rows):
    y0 = Inches(4.6) + Inches(i * 0.4)
    bg = RGBColor(0xE8, 0xF5, 0xE9) if best else LIGHT
    add_rect(s, Inches(7.8), y0, Inches(5.2), Inches(0.4), fill=bg, line=LGRAY)
    add_text(s, tau, Inches(7.95), y0 + Inches(0.04), Inches(1.3), Inches(0.34),
             size=11, bold=best, color=GDARK if best else BLACK, align=PP_ALIGN.CENTER)
    add_text(s, kept, Inches(9.3), y0 + Inches(0.04), Inches(2.0), Inches(0.34),
             size=11, bold=best, color=GDARK if best else BLACK, align=PP_ALIGN.CENTER)
    add_text(s, rmse, Inches(11.4), y0 + Inches(0.04), Inches(1.5), Inches(0.34),
             size=11, bold=best, color=GDARK if best else BLACK, align=PP_ALIGN.CENTER)

callout(s,
        "Unsolved: hard cutoff is arbitrary — a sample at 0.29 is fully discarded; 0.31 gets full weight.",
        Inches(7.8), Inches(6.25), Inches(5.2), Inches(0.75),
        fill=RGBColor(0xFF, 0xF0, 0xD8), border=AMBER,
        text_color=RGBColor(0x6B, 0x3A, 0x00), size=10)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — KMM
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
add_rect(s, 0, 0, W, H, fill=WHITE)
slide_title_bar(s, "KMM — Soft Weights Instead of a Hard Filter")
nav_bar(s, 7, TOTAL)

add_img(s, "fig_kmm_weights.png", Inches(0.3), Inches(1.25), Inches(7.6))

# Right
add_rect(s, Inches(8.2), Inches(1.25), Inches(4.8), Inches(0.42), fill=NAVY)
add_text(s, "How it works", Inches(8.35), Inches(1.28),
         Inches(4.5), Inches(0.36), size=13, bold=True, color=WHITE)
bullets(s, [
    "Each lit sample gets weight wᵢ ∈ [0, 10]",
    "RBF kernel similarity: K(x,x′) = exp(−‖x−x′‖²/2σ²)",
    "Solve QP:  ½ wᵀK_ss w − κᵀw",
    "σ via median heuristic (parameter-free)",
], Inches(8.25), Inches(1.75), Inches(4.7), Inches(0.4), size=11, spacing=0.38)
callout(s,
        "RBF kernel = chemical similarity based on distance in element-composition space. "
        "Ref: Huang et al. (2006, NeurIPS); Gretton et al. (2009, Dataset Shift in ML).",
        Inches(8.2), Inches(3.35), Inches(4.8), Inches(0.62),
        fill=RGBColor(0xE8, 0xF5, 0xFF), border=ACCENT, text_color=DGRAY, size=9, italic=False)

# Result table (shifted down to make room for callout above)
add_text(s, "Result",
         Inches(8.2), Inches(4.08), Inches(4.8), Inches(0.4),
         size=13, bold=True, color=NAVY)
results_rows = [
    ("KMM CV RMSE",        "2.035"),
    ("Δ vs baseline",      "−4.6%"),
    ("Near-zero weights",  "78.5%"),
    ("Agreement w/ DRST",  "r = 0.79"),
]
for i, (k, v) in enumerate(results_rows):
    y0 = Inches(4.5) + Inches(i * 0.4)
    bg = LIGHT if i % 2 == 0 else WHITE
    add_rect(s, Inches(8.2), y0, Inches(4.8), Inches(0.4), fill=bg, line=LGRAY)
    add_text(s, k, Inches(8.35), y0 + Inches(0.05), Inches(3.0), Inches(0.34),
             size=11, color=DGRAY)
    add_text(s, v, Inches(11.4), y0 + Inches(0.05), Inches(1.5), Inches(0.34),
             size=11, bold=True, color=BLACK, align=PP_ALIGN.CENTER)

callout(s,
        "Reassuring: two unrelated methods flag the same 78.5% of samples ⇒ real signal, not artifact.\n"
        "Still unsolved: literature labels remain in training, so the 3.42 pp offset still bleeds in.",
        Inches(8.2), Inches(6.2), Inches(4.8), Inches(0.85),
        fill=LIGHT, border=ACCENT, text_color=DGRAY, size=10)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — Prior Feature Transfer Pipeline
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
add_rect(s, 0, 0, W, H, fill=WHITE)
slide_title_bar(s, "Prior Feature Transfer — The Winning Pipeline")
nav_bar(s, 8, TOTAL)

# Stage labels
add_text(s, "STAGE 1 — Pre-train on Literature Chemistry",
         Inches(0.3), Inches(1.2), Inches(12.7), Inches(0.38),
         size=13, bold=True, color=ACCENT)
add_text(s, "STAGE 2 — Fine-tune on Our Labels Only",
         Inches(0.3), Inches(3.9), Inches(12.7), Inches(0.38),
         size=13, bold=True, color=GREEN)

def pipeline_box(slide, x, y, w, h, title, subtitle, fill_c, text_c=WHITE):
    add_rect(slide, x, y, w, h, fill=fill_c, line=RGBColor(0xBB, 0xBB, 0xBB))
    add_text(slide, title, x + Inches(0.1), y + Inches(0.08),
             w - Inches(0.2), Inches(0.4), size=12, bold=True,
             color=text_c, align=PP_ALIGN.CENTER)
    if subtitle:
        add_text(slide, subtitle, x + Inches(0.1), y + Inches(0.5),
                 w - Inches(0.2), Inches(0.5),
                 size=10, color=text_c, align=PP_ALIGN.CENTER)

def arrow_h(slide, x, y, length, color=ACCENT):
    line = slide.shapes.add_connector(1, x, y, x + length, y)
    line.line.color.rgb = color; line.line.width = Pt(2.5)

bw = Inches(2.5); bh = Inches(1.0)
by1 = Inches(1.7); by2 = Inches(4.4)
gap = Inches(0.5)

# Stage 1 row
pipeline_box(s, Inches(0.4), by1, bw, bh,
             "Literature", "3,852 samples\n(DRST filtered)",
             RGBColor(0xCC, 0xE0, 0xFF), text_c=NAVY)
arrow_h(s, Inches(0.4) + bw, by1 + bh/2, gap)
pipeline_box(s, Inches(0.4) + bw + gap, by1, bw, bh,
             "Stage 1", "XGBoost\npre-train",
             ACCENT, text_c=WHITE)
arrow_h(s, Inches(0.4) + 2*bw + gap, by1 + bh/2, gap)
pipeline_box(s, Inches(0.4) + 2*(bw + gap), by1, bw, bh,
             "lit_prior", "prediction\n(68th feature)",
             AMBER, text_c=WHITE)

# Stage 2 row
pipeline_box(s, Inches(0.4), by2, bw, bh,
             "Our Lab Data", "89,074 samples",
             RGBColor(0xCC, 0xE8, 0xD0), text_c=GDARK)
arrow_h(s, Inches(0.4) + bw, by2 + bh/2, gap, color=GREEN)
pipeline_box(s, Inches(0.4) + bw + gap, by2, bw, bh,
             "Append", "68th feature",
             RGBColor(0xE0, 0xE8, 0xF8), text_c=NAVY)
arrow_h(s, Inches(0.4) + 2*bw + gap, by2 + bh/2, gap, color=GREEN)
pipeline_box(s, Inches(0.4) + 2*(bw + gap), by2, bw, bh,
             "Stage 2", "LightGBM\nour labels only",
             GREEN, text_c=WHITE)
arrow_h(s, Inches(0.4) + 3*bw + 2*gap, by2 + bh/2, gap, color=GREEN)
pipeline_box(s, Inches(0.4) + 3*(bw + gap), by2, bw, bh,
             "Prediction", "Y(C₂) %",
             NAVY, text_c=WHITE)

# Vertical arrow from prior to append
line = s.shapes.add_connector(1,
                              Inches(0.4) + 2*(bw + gap) + bw/2, by1 + bh,
                              Inches(0.4) + bw + gap + bw/2, by2)
line.line.color.rgb = AMBER; line.line.width = Pt(2.5)

# Key property callout
add_rect(s, Inches(0.4), Inches(5.65), Inches(12.5), Inches(1.45),
         fill=LIGHT, line=NAVY)
add_text(s, "Key property — feature-based transfer learning  (Pan & Yang, 2010, IEEE TKDE)",
         Inches(0.55), Inches(5.72), Inches(12.2), Inches(0.4),
         size=13, bold=True, color=NAVY)
add_text(s,
         "Stage 2 NEVER sees literature labels. The 3.42 pp offset affects the VALUE of the 68th feature — which Stage 2 can calibrate: "
         "\"when expert says X, my lab gets ~X − 3.4\". It does NOT corrupt the training loss (which it could not fix). "
         "Stage 1 uses XGBoost (Chen & Guestrin, 2016) for conservative behaviour on 782 samples; "
         "Stage 2 uses LightGBM (Ke et al., 2017) for efficiency on 89,074 rows.",
         Inches(0.55), Inches(6.12), Inches(12.2), Inches(0.9),
         size=10, color=DGRAY)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — Results
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
add_rect(s, 0, 0, W, H, fill=WHITE)
slide_title_bar(s, "Results — All Five Methods at a Glance")
nav_bar(s, 9, TOTAL)

add_img(s, "fig_walkthrough_results.png", Inches(0.3), Inches(1.25), Inches(7.6))

# Right: CV table + OOD table
add_text(s, "Cross-validation (our data)",
         Inches(8.1), Inches(1.3), Inches(5.0), Inches(0.4),
         size=14, bold=True, color=NAVY)
cv_rows = [
    ("Naive merge",        "2.248", "+5.4%",  False, True),
    ("Baseline",           "2.133", "—",      False, False),
    ("KMM weighted",       "2.035", "−4.6%",  False, False),
    ("DRST (τ=0.30)",      "2.019", "−5.3%",  False, False),
    ("Prior FT  ★",        "1.907", "−10.6%", True,  False),
]
add_rect(s, Inches(8.1), Inches(1.75), Inches(5.0), Inches(0.4), fill=NAVY)
for j, (h, x, w) in enumerate(zip(["Method", "RMSE", "Δ"],
                                    [Inches(8.2), Inches(10.3), Inches(11.4)],
                                    [Inches(2.0), Inches(1.1), Inches(1.6)])):
    add_text(s, h, x, Inches(1.78), w, Inches(0.34),
             size=11, bold=True, color=WHITE)
for i, (m, r, d, best, worst) in enumerate(cv_rows):
    y0 = Inches(2.15) + Inches(i * 0.4)
    bg = RGBColor(0xE8, 0xF5, 0xE9) if best else \
         RGBColor(0xFF, 0xEB, 0xEB) if worst else (LIGHT if i % 2 == 0 else WHITE)
    add_rect(s, Inches(8.1), y0, Inches(5.0), Inches(0.4), fill=bg, line=LGRAY)
    fc = GDARK if best else (RDARK if worst else BLACK)
    add_text(s, m, Inches(8.2), y0 + Inches(0.04), Inches(2.0), Inches(0.34),
             size=11, bold=best, color=fc)
    add_text(s, r, Inches(10.3), y0 + Inches(0.04), Inches(1.1), Inches(0.34),
             size=11, bold=best, color=fc, align=PP_ALIGN.CENTER)
    add_text(s, d, Inches(11.4), y0 + Inches(0.04), Inches(1.6), Inches(0.34),
             size=11, bold=best, color=fc, align=PP_ALIGN.CENTER)

# OOD
add_text(s, "OOD robustness (unseen lit)",
         Inches(8.1), Inches(4.5), Inches(5.0), Inches(0.4),
         size=14, bold=True, color=NAVY)
add_rect(s, Inches(8.1), Inches(4.95), Inches(5.0), Inches(0.4), fill=NAVY)
add_text(s, "Method", Inches(8.2), Inches(4.98), Inches(3.0), Inches(0.34),
         size=11, bold=True, color=WHITE)
add_text(s, "OOD RMSE", Inches(11.2), Inches(4.98), Inches(1.8), Inches(0.34),
         size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
add_rect(s, Inches(8.1), Inches(5.35), Inches(5.0), Inches(0.42),
         fill=LIGHT, line=LGRAY)
add_text(s, "Baseline", Inches(8.2), Inches(5.4), Inches(3.0), Inches(0.34),
         size=11, color=BLACK)
add_text(s, "6.53", Inches(11.2), Inches(5.4), Inches(1.8), Inches(0.34),
         size=11, color=BLACK, align=PP_ALIGN.CENTER)
add_rect(s, Inches(8.1), Inches(5.77), Inches(5.0), Inches(0.42),
         fill=RGBColor(0xE8, 0xF5, 0xE9), line=GREEN)
add_text(s, "Prior FT  ★", Inches(8.2), Inches(5.82), Inches(3.0), Inches(0.34),
         size=11, bold=True, color=GDARK)
add_text(s, "3.60  (−45%)", Inches(11.2), Inches(5.82), Inches(1.8), Inches(0.34),
         size=11, bold=True, color=GDARK, align=PP_ALIGN.CENTER)

add_text(s, "OOD: 2,139 non-Impregnation lit samples, unseen.",
         Inches(8.1), Inches(6.3), Inches(5.0), Inches(0.4),
         size=10, italic=True, color=DGRAY)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — SHAP
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
add_rect(s, 0, 0, W, H, fill=WHITE)
slide_title_bar(s, "SHAP — Did the Model Learn Real Chemistry?")
nav_bar(s, 10, TOTAL)

add_img(s, "fig_shap_beeswarm.png", Inches(0.3), Inches(1.25), Inches(6.8))

# Right: findings + residuals
add_rect(s, Inches(7.3), Inches(1.25), Inches(5.7), Inches(0.42), fill=NAVY)
add_text(s, "Top findings (3,000 sample subsample)",
         Inches(7.45), Inches(1.28), Inches(5.4), Inches(0.36),
         size=12, bold=True, color=WHITE)
bullets(s, [
    "#1: lit_prior_prediction — transfer is genuinely used",
    "Temperature — strong positive, T↑ → C₂↑",
    "Ba, Mn, La, Ce — known OCM active phases / promoters",
    "Li, K — mixed / suppressive effect",
], Inches(7.35), Inches(1.78), Inches(5.6), Inches(0.4), size=11, spacing=0.42)

# Residual table
add_text(s, "Residual pattern (ceiling effect)",
         Inches(7.3), Inches(4.0), Inches(5.7), Inches(0.4),
         size=13, bold=True, color=NAVY)
add_rect(s, Inches(7.3), Inches(4.45), Inches(5.7), Inches(0.42), fill=NAVY)
add_text(s, "Y(C₂) range", Inches(7.45), Inches(4.48), Inches(3.0), Inches(0.34),
         size=11, bold=True, color=WHITE)
add_text(s, "RMSE", Inches(10.7), Inches(4.48), Inches(2.0), Inches(0.34),
         size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
res_rows = [
    ("0 – 6%",   "1.43 – 1.48", False),
    ("6 – 10%",  "1.95",        False),
    ("10 – 15%", "2.58",        False),
    ("> 15%",    "4.70  (−4.2 pp bias)", True),
]
for i, (rng, rmse, bad) in enumerate(res_rows):
    y0 = Inches(4.87) + Inches(i * 0.4)
    bg = RGBColor(0xFF, 0xEB, 0xEB) if bad else (LIGHT if i % 2 == 0 else WHITE)
    add_rect(s, Inches(7.3), y0, Inches(5.7), Inches(0.4), fill=bg, line=LGRAY)
    fc = RDARK if bad else BLACK
    add_text(s, rng, Inches(7.45), y0 + Inches(0.04), Inches(3.0), Inches(0.34),
             size=11, bold=bad, color=fc)
    add_text(s, rmse, Inches(10.7), y0 + Inches(0.04), Inches(2.0), Inches(0.34),
             size=11, bold=bad, color=fc, align=PP_ALIGN.CENTER)

add_text(s,
         "Root cause: GHSV, CH₄/O₂ ratio, pressure absent from dataset. "
         "SHAP reference: Lundberg & Lee (2017, NeurIPS); Shapley (1953).",
         Inches(7.3), Inches(6.45), Inches(5.7), Inches(0.55),
         size=9, italic=True, color=DGRAY)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 11 — Limitations & Next Steps
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
add_rect(s, 0, 0, W, H, fill=WHITE)
slide_title_bar(s, "Limitations & Next Steps")
nav_bar(s, 11, TOTAL)

options = [
    ("Option A", "Easy", GREEN, "Add reaction conditions",
     "Include GHSV, CH₄/O₂, and pressure as features.",
     "Expected to fix the >15% ceiling effect and reduce high-yield RMSE from 4.70 to ~2.5."),
    ("Option B", "Medium", AMBER, "Tune transfer threshold",
     "Sweep DRST τ and Stage 1 sample size jointly.",
     "Current best: τ = 0.30 (782 samples). Grid search may recover 0.3–0.5% additional RMSE."),
    ("Option C", "Hard", RED, "Neural domain adaptation",
     "Replace LightGBM with DANN or CORAL.",
     "High complexity — only justified after reaction conditions are added."),
]
col_w = Inches(4.1); col_gap = Inches(0.25)
for i, (lab, diff, color, head, sub, body) in enumerate(options):
    x0 = Inches(0.4) + i * (col_w + col_gap)
    add_rect(s, x0, Inches(1.25), col_w, Inches(3.4), fill=LIGHT, line=color)
    add_rect(s, x0, Inches(1.25), col_w, Inches(0.45), fill=color)
    add_text(s, f"{lab} — {diff}",
             x0 + Inches(0.15), Inches(1.28),
             col_w - Inches(0.3), Inches(0.4),
             size=13, bold=True, color=WHITE)
    add_text(s, head,
             x0 + Inches(0.15), Inches(1.8),
             col_w - Inches(0.3), Inches(0.4),
             size=13, bold=True, color=NAVY)
    add_text(s, sub,
             x0 + Inches(0.15), Inches(2.25),
             col_w - Inches(0.3), Inches(0.8),
             size=11, color=DGRAY)
    add_text(s, body,
             x0 + Inches(0.15), Inches(3.1),
             col_w - Inches(0.3), Inches(1.4),
             size=11, color=DGRAY, italic=True)

# Bottom two boxes
add_rect(s, Inches(0.4), Inches(4.9), Inches(6.2), Inches(2.0),
         fill=RGBColor(0xF5, 0xF5, 0xFF), line=ACCENT)
add_text(s, "Key Limitations",
         Inches(0.55), Inches(4.97), Inches(5.9), Inches(0.4),
         size=13, bold=True, color=NAVY)
bullets(s, [
    "Missing GHSV / CH₄:O₂ / pressure features",
    "Literature has 15+ prep methods — coverage uneven",
    "Publication bias: only successful experiments published",
    "OOD improvement may not generalise to new catalyst families",
], Inches(0.5), Inches(5.4), Inches(6.1), Inches(0.4), size=11, spacing=0.36, color=DGRAY)

add_rect(s, Inches(6.8), Inches(4.9), Inches(6.2), Inches(2.0),
         fill=RGBColor(0xE8, 0xF5, 0xE9), line=GREEN)
add_text(s, "Recommended Priority",
         Inches(6.95), Inches(4.97), Inches(5.9), Inches(0.4),
         size=13, bold=True, color=GDARK)
bullets(s, [
    "1. Collect GHSV + CH₄/O₂  (highest impact)",
    "2. Re-run Prior FT with expanded features",
    "3. Revisit τ tuning with updated data",
], Inches(6.9), Inches(5.4), Inches(6.1), Inches(0.4), size=11, spacing=0.42, color=GDARK)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 12 — Summary
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
add_rect(s, 0, 0, W, H, fill=WHITE)
slide_title_bar(s, "Summary")
nav_bar(s, 12, TOTAL)

# Big summary box
add_rect(s, Inches(1.0), Inches(1.4), Inches(11.3), Inches(4.6),
         fill=LIGHT, line=NAVY)
add_text(s,
         "What we asked",
         Inches(1.25), Inches(1.55), Inches(10.9), Inches(0.4),
         size=14, bold=True, color=NAVY)
add_text(s,
         "Can published OCM literature improve a lab-data model without harming it?",
         Inches(1.25), Inches(1.95), Inches(10.9), Inches(0.45),
         size=12, color=DGRAY)

add_text(s,
         "What we learned",
         Inches(1.25), Inches(2.6), Inches(10.9), Inches(0.4),
         size=14, bold=True, color=NAVY)
findings = [
    "Naive merging HARMS accuracy (+5.4% RMSE) — the systematic offset corrupts training.",
    "Selective filtering (DRST / KMM) recovers about 5% improvement — independent methods agree (r=0.79).",
    "Using literature as a PRIOR FEATURE (not a label) gives the biggest gain: −10.6% CV, −45% OOD.",
    "SHAP confirms the model uses real OCM chemistry: temperature, Ba, Mn, La, Ce, and the literature prior drive predictions.",
]
for i, txt in enumerate(findings):
    add_text(s, "•  " + txt,
             Inches(1.35), Inches(3.05) + Inches(i * 0.4),
             Inches(10.7), Inches(0.4),
             size=12, color=BLACK)

# Bottom emphasis box
add_rect(s, Inches(1.0), Inches(6.15), Inches(11.3), Inches(0.85),
         fill=RGBColor(0xE8, 0xF5, 0xE9), line=GREEN)
add_text(s, "Next step that matters most",
         Inches(1.25), Inches(6.23), Inches(11.0), Inches(0.4),
         size=13, bold=True, color=GDARK)
add_text(s,
         "Add GHSV and CH₄:O₂ to the feature set — the high-yield ceiling is data-bound, not model-bound.",
         Inches(1.25), Inches(6.6), Inches(11.0), Inches(0.4),
         size=12, color=GDARK)

# ── Save ──────────────────────────────────────────────────────────────────────
pptx_path = ROOT / "ocm_presentation.pptx"
prs.save(str(pptx_path))
print(f"Saved: {pptx_path}  ({pptx_path.stat().st_size // 1024} KB)")

# ── Export to PDF via LibreOffice ─────────────────────────────────────────────
import subprocess, shutil
lo = shutil.which("libreoffice") or shutil.which("soffice")
if lo:
    print("Converting to PDF via LibreOffice...")
    result = subprocess.run(
        [lo, "--headless", "--convert-to", "pdf",
         "--outdir", str(ROOT), str(pptx_path)],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode == 0:
        pdf = ROOT / "ocm_presentation.pdf"
        print(f"Saved: {pdf}  ({pdf.stat().st_size // 1024} KB)")
    else:
        print("LibreOffice PDF export failed:")
        print(result.stderr[:500])
else:
    print("LibreOffice not found — skipping PDF export.")
