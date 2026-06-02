"""
Build ocm_presentation.pptx — 8-slide PowerPoint deck.
Run: python build_pptx.py
Produces: ocm_presentation.pptx  (and ocm_presentation.pdf via LibreOffice)
"""

import io, pathlib, subprocess
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import pptx.util as util

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

# Slide dimensions: widescreen 16:9
W = Inches(13.33)
H = Inches(7.5)

prs = Presentation()
prs.slide_width  = W
prs.slide_height = H

BLANK = prs.slide_layouts[6]   # completely blank layout

# ── Helpers ──────────────────────────────────────────────────────────────────

def add_rect(slide, x, y, w, h, fill=None, line=None):
    shape = slide.shapes.add_shape(1, x, y, w, h)   # MSO_SHAPE_TYPE.RECTANGLE=1
    if fill:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    else:
        shape.fill.background()
    if line:
        shape.line.color.rgb = line
        shape.line.width = Pt(0.75)
    else:
        shape.line.fill.background()
    return shape

def add_text(slide, text, x, y, w, h,
             size=18, bold=False, color=BLACK, align=PP_ALIGN.LEFT,
             wrap=True, italic=False):
    txb = slide.shapes.add_textbox(x, y, w, h)
    tf  = txb.text_frame
    tf.word_wrap = wrap
    p   = tf.paragraphs[0]
    p.alignment = align
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
    """Bottom navy navigation bar with slide number."""
    bar = add_rect(slide, 0, H - Inches(0.38), W, Inches(0.38), fill=NAVY)
    add_text(slide, f"OCM Dataset Integration   |   {current} / {total}",
             Inches(0.2), H - Inches(0.36), W - Inches(0.4), Inches(0.34),
             size=9, color=RGBColor(0xCC, 0xCC, 0xCC), align=PP_ALIGN.LEFT)

def slide_title_bar(slide, title, subtitle=None):
    """Navy title bar at top."""
    bar_h = Inches(1.0) if not subtitle else Inches(1.25)
    add_rect(slide, 0, 0, W, bar_h, fill=NAVY)
    add_text(slide, title,
             Inches(0.4), Inches(0.1), W - Inches(0.8), Inches(0.65),
             size=24, bold=True, color=WHITE)
    if subtitle:
        add_text(slide, subtitle,
                 Inches(0.4), Inches(0.68), W - Inches(0.8), Inches(0.45),
                 size=13, color=RGBColor(0xBB, 0xCC, 0xEE), italic=True)

def bullet(slide, items, x, y, w, h, size=14, spacing=0.32):
    """Add a bullet list."""
    for i, item in enumerate(items):
        prefix = "•  " if not item.startswith("   ") else ""
        c = DGRAY if item.startswith("   ") else BLACK
        add_text(slide, prefix + item.lstrip(),
                 x, y + Inches(i * spacing), w, Inches(spacing + 0.05),
                 size=size, color=c)

TOTAL = 8

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Title
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)

# Full navy background
add_rect(s, 0, 0, W, H, fill=NAVY)

# Decorative accent bar
add_rect(s, 0, Inches(2.95), W, Inches(0.06), fill=ACCENT)

add_text(s, "JAIST — Taniike Lab",
         Inches(1.2), Inches(1.4), Inches(10.9), Inches(0.5),
         size=13, color=RGBColor(0xAA, 0xBB, 0xDD), align=PP_ALIGN.CENTER)

add_text(s, "Integrating OCM Literature Data\ninto Lab-Scale ML Models",
         Inches(1.2), Inches(1.9), Inches(10.9), Inches(1.6),
         size=36, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

add_text(s, "Domain-Adaptive Transfer Learning for Catalyst Yield Prediction",
         Inches(1.2), Inches(3.55), Inches(10.9), Inches(0.6),
         size=16, color=RGBColor(0x99, 0xBB, 0xEE), align=PP_ALIGN.CENTER, italic=True)

add_rect(s, Inches(4.5), Inches(4.35), Inches(4.33), Inches(0.04),
         fill=RGBColor(0x44, 0x66, 0xAA))

add_text(s, "Partha Nupam Nagar",
         Inches(1.2), Inches(4.55), Inches(10.9), Inches(0.45),
         size=15, color=RGBColor(0xCC, 0xDD, 0xFF), align=PP_ALIGN.CENTER)

add_text(s, "May 2026",
         Inches(1.2), Inches(5.0), Inches(10.9), Inches(0.4),
         size=13, color=RGBColor(0x88, 0x99, 0xBB), align=PP_ALIGN.CENTER)

nav_bar(s, 1, TOTAL)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — Two Datasets
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
add_rect(s, 0, 0, W, H, fill=WHITE)
slide_title_bar(s, "Two Very Different Datasets")
nav_bar(s, 2, TOTAL)

# Left card — Internal
add_rect(s, Inches(0.4), Inches(1.35), Inches(5.8), Inches(4.6),
         fill=RGBColor(0xE8, 0xF4, 0xFF), line=ACCENT)
add_rect(s, Inches(0.4), Inches(1.35), Inches(5.8), Inches(0.45), fill=ACCENT)
add_text(s, "Our Lab Data (Internal)", Inches(0.55), Inches(1.38),
         Inches(5.5), Inches(0.4), size=14, bold=True, color=WHITE)

rows_l = [
    ("Samples",      "89,074"),
    ("Year",         "2025"),
    ("Preparation",  "Impregnation only"),
    ("Mean Y(C2)",   "5.25%"),
    ("Conditions",   "Controlled, fixed protocol"),
]
for i, (k, v) in enumerate(rows_l):
    bg = RGBColor(0xF0, 0xF7, 0xFF) if i % 2 == 0 else RGBColor(0xE8, 0xF4, 0xFF)
    add_rect(s, Inches(0.4), Inches(1.8) + Inches(i * 0.52),
             Inches(5.8), Inches(0.52), fill=bg)
    add_text(s, k, Inches(0.55), Inches(1.83) + Inches(i * 0.52),
             Inches(2.4), Inches(0.48), size=13, bold=True, color=DGRAY)
    add_text(s, v, Inches(2.95), Inches(1.83) + Inches(i * 0.52),
             Inches(3.1), Inches(0.48), size=13, color=BLACK)

add_rect(s, Inches(0.4), Inches(4.4), Inches(5.8), Inches(0.5),
         fill=RGBColor(0xD0, 0xE8, 0xFF), line=ACCENT)
add_text(s, "✓  High volume, consistent protocol",
         Inches(0.55), Inches(4.43), Inches(5.5), Inches(0.44),
         size=12, color=RGBColor(0x1A, 0x5C, 0x2A))

add_rect(s, Inches(0.4), Inches(4.92), Inches(5.8), Inches(0.5),
         fill=RGBColor(0xFF, 0xEB, 0xEB), line=RED)
add_text(s, "✗  Narrow chemical diversity",
         Inches(0.55), Inches(4.95), Inches(5.5), Inches(0.44),
         size=12, color=RGBColor(0x8B, 0x1A, 0x1A))

# Right card — Literature
add_rect(s, Inches(6.53), Inches(1.35), Inches(6.4), Inches(4.6),
         fill=RGBColor(0xFF, 0xF5, 0xE8), line=AMBER)
add_rect(s, Inches(6.53), Inches(1.35), Inches(6.4), Inches(0.45), fill=AMBER)
add_text(s, "Literature Data (Public, ≤2019)", Inches(6.68), Inches(1.38),
         Inches(6.1), Inches(0.4), size=14, bold=True, color=WHITE)

rows_r = [
    ("Samples",      "3,852"),
    ("Year",         "1982 – 2019"),
    ("Preparation",  "15+ methods"),
    ("Mean Y(C2)",   "8.67%"),
    ("Conditions",   "Mixed labs, varied protocols"),
]
for i, (k, v) in enumerate(rows_r):
    bg = RGBColor(0xFF, 0xF8, 0xF0) if i % 2 == 0 else RGBColor(0xFF, 0xF5, 0xE8)
    add_rect(s, Inches(6.53), Inches(1.8) + Inches(i * 0.52),
             Inches(6.4), Inches(0.52), fill=bg)
    add_text(s, k, Inches(6.68), Inches(1.83) + Inches(i * 0.52),
             Inches(2.6), Inches(0.48), size=13, bold=True, color=DGRAY)
    add_text(s, v, Inches(9.28), Inches(1.83) + Inches(i * 0.52),
             Inches(3.5), Inches(0.48), size=13, color=BLACK)

add_rect(s, Inches(6.53), Inches(4.4), Inches(6.4), Inches(0.5),
         fill=RGBColor(0xD0, 0xE8, 0xFF), line=ACCENT)
add_text(s, "✓  Chemical diversity, high-yield recipes",
         Inches(6.68), Inches(4.43), Inches(6.1), Inches(0.44),
         size=12, color=RGBColor(0x1A, 0x5C, 0x2A))

add_rect(s, Inches(6.53), Inches(4.92), Inches(6.4), Inches(0.5),
         fill=RGBColor(0xFF, 0xEB, 0xEB), line=RED)
add_text(s, "✗  Publication bias, missing conditions",
         Inches(6.68), Inches(4.95), Inches(6.1), Inches(0.44),
         size=12, color=RGBColor(0x8B, 0x1A, 0x1A))

# Gap callout
add_rect(s, Inches(0.4), Inches(6.0), Inches(12.53), Inches(0.72),
         fill=RGBColor(0xFF, 0xF0, 0xF0), line=RED)
add_text(s,
         "Challenge:  Δ mean Y(C2) = 8.67 − 5.25 = 3.42 pp    (t = −32.2,  p = 2.3 × 10⁻²⁰²)  —  datasets are statistically different",
         Inches(0.6), Inches(6.04), Inches(12.1), Inches(0.64),
         size=13, bold=True, color=RGBColor(0x8B, 0x1A, 0x1A), align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — Domain Gap
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
add_rect(s, 0, 0, W, H, fill=WHITE)
slide_title_bar(s, "Domain Gap — Why We Cannot Just Mix the Data")
nav_bar(s, 3, TOTAL)

add_img(s, "fig_pca_domain_gap.png", Inches(0.3), Inches(1.2), Inches(6.2))

# Right panel
bx = Inches(6.8)
add_rect(s, bx, Inches(1.2), Inches(6.1), Inches(0.4), fill=ACCENT)
add_text(s, "What the PCA shows", bx + Inches(0.15), Inches(1.22),
         Inches(5.8), Inches(0.36), size=14, bold=True, color=WHITE)

bullet(s, [
    "65 element features compressed to 2D",
    "Blue = our 89k samples;   Red = literature 3.8k",
    "Literature spans regions our model has never seen",
], bx + Inches(0.1), Inches(1.72), Inches(5.9), Inches(0.4), size=13, spacing=0.38)

add_rect(s, bx, Inches(3.15), Inches(6.1), Inches(0.4), fill=NAVY)
add_text(s, "Key facts", bx + Inches(0.15), Inches(3.17),
         Inches(5.8), Inches(0.36), size=14, bold=True, color=WHITE)

facts = [
    "78.5% of literature samples are out-of-distribution",
    "Our data: Ba-dominant   |   Literature: Si / Na / Mn dominant",
    "La, Ce, Sr, Mg shared between both — transfer is possible",
    "Gap is in proportions, not in completely foreign chemistry",
]
for i, f in enumerate(facts):
    bg = RGBColor(0xF0, 0xF2, 0xFF) if i % 2 == 0 else WHITE
    add_rect(s, bx, Inches(3.57) + Inches(i * 0.48),
             Inches(6.1), Inches(0.48), fill=bg)
    add_text(s, "•  " + f, bx + Inches(0.12), Inches(3.6) + Inches(i * 0.48),
             Inches(5.85), Inches(0.44), size=12, color=BLACK)

add_rect(s, bx, Inches(5.62), Inches(6.1), Inches(0.68),
         fill=RGBColor(0xFF, 0xF0, 0xD8), line=AMBER)
add_text(s,
         "Naive concatenation inflates training error — literature samples vote equally despite representing a different measurement regime.",
         bx + Inches(0.15), Inches(5.66), Inches(5.8), Inches(0.6),
         size=11, color=RGBColor(0x6B, 0x3A, 0x00), italic=True)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — Methods RMSE Table
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
add_rect(s, 0, 0, W, H, fill=WHITE)
slide_title_bar(s, "Five Approaches — What We Tried and What Happened")
nav_bar(s, 4, TOTAL)

# Table header
cols = [Inches(0.3), Inches(3.6), Inches(7.8), Inches(9.7), Inches(11.1)]
col_w = [Inches(3.25), Inches(4.15), Inches(1.85), Inches(1.35), Inches(1.95)]
headers = ["Method", "Core Idea", "CV RMSE", "Δ vs Baseline", "Result"]
for j, (h_txt, cx, cw) in enumerate(zip(headers, cols, col_w)):
    add_rect(s, cx, Inches(1.15), cw, Inches(0.42), fill=NAVY)
    add_text(s, h_txt, cx + Inches(0.05), Inches(1.17),
             cw - Inches(0.1), Inches(0.38), size=12, bold=True,
             color=WHITE, align=PP_ALIGN.CENTER)

table_data = [
    ("Baseline\n(no literature)",
     "LightGBM on 89k internal samples only",
     "2.133%", "—", "Reference",
     RGBColor(0xF0,0xF0,0xF5), BLACK, BLACK),

    ("Naive\nconcatenation",
     "Add all 3,852 literature rows with equal weight",
     "2.248%", "+5.4%  ✗", "WORSE",
     RGBColor(0xFF,0xEB,0xEB), RED, RED),

    ("DRST  (τ = 0.30)",
     "Keep 782 similar literature samples via density-ratio classifier",
     "2.019%", "−5.3%", "Better",
     RGBColor(0xF5,0xF8,0xFF), BLACK, RGBColor(0x1A,0x6E,0x40)),

    ("KMM reweighting",
     "QP solver: soft per-sample weights; 78.5% → near zero",
     "2.035%", "−4.6%", "Better",
     RGBColor(0xF5,0xF8,0xFF), BLACK, RGBColor(0x1A,0x6E,0x40)),

    ("Bias correction",
     "Quantile-normalise literature Y(C2) to our distribution",
     "2.044%", "−4.2%", "Better",
     RGBColor(0xF5,0xF8,0xFF), BLACK, RGBColor(0x1A,0x6E,0x40)),

    ("Two-stage\nfine-tuning  ★",
     "XGBoost prior on filtered literature → 68th feature → LightGBM on our labels",
     "1.907%", "−10.6%  ✓", "BEST",
     RGBColor(0xE8,0xF5,0xE9), BLACK, RGBColor(0x1B,0x5E,0x20)),
]

row_h = Inches(0.78)
for i, row in enumerate(table_data):
    txt_vals, bg, tc, dc = row[:5], row[5], row[6], row[7]
    for j, (txt, cx, cw) in enumerate(zip(txt_vals, cols, col_w)):
        add_rect(s, cx, Inches(1.57) + Inches(i * row_h / Inches(1)),
                 cw, row_h, fill=bg, line=LGRAY)
        align = PP_ALIGN.CENTER if j >= 2 else PP_ALIGN.LEFT
        col_color = dc if j in (2, 3, 4) else tc
        add_text(s, txt,
                 cx + Inches(0.07),
                 Inches(1.62) + Inches(i * row_h / Inches(1)),
                 cw - Inches(0.1), row_h - Inches(0.08),
                 size=11, color=col_color, align=align)

# OOD note
add_rect(s, Inches(0.3), Inches(6.35), Inches(12.73), Inches(0.46),
         fill=RGBColor(0xE8, 0xF5, 0xE9), line=GREEN)
add_text(s,
         "OOD test (2,139 held-out non-Impregnation literature samples, never trained on):   Baseline 6.53%  →  Two-stage 3.60%     −45%",
         Inches(0.45), Inches(6.38), Inches(12.4), Inches(0.4),
         size=12, bold=True, color=RGBColor(0x1B, 0x5E, 0x20), align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — Pipeline
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
add_rect(s, 0, 0, W, H, fill=WHITE)
slide_title_bar(s, "Two-Stage Fine-Tuning — How It Works")
nav_bar(s, 5, TOTAL)

# Stage labels
add_text(s, "STAGE 1 — Pre-train on Literature Chemistry",
         Inches(0.3), Inches(1.2), Inches(12.7), Inches(0.38),
         size=13, bold=True, color=ACCENT)
add_text(s, "STAGE 2 — Fine-tune on Our Labels",
         Inches(0.3), Inches(3.85), Inches(12.7), Inches(0.38),
         size=13, bold=True, color=GREEN)

def pipeline_box(slide, x, y, w, h, title, subtitle, fill_c, text_c=WHITE):
    add_rect(slide, x, y, w, h, fill=fill_c, line=RGBColor(0xBB,0xBB,0xBB))
    add_text(slide, title, x + Inches(0.1), y + Inches(0.08),
             w - Inches(0.2), h * 0.55, size=12, bold=True, color=text_c,
             align=PP_ALIGN.CENTER)
    if subtitle:
        add_text(slide, subtitle, x + Inches(0.1),
                 y + h * 0.52, w - Inches(0.2), h * 0.45,
                 size=10, color=text_c, align=PP_ALIGN.CENTER)

def arrow_h(slide, x, y, length, color=ACCENT):
    line = slide.shapes.add_connector(1, x, y, x + length, y)
    line.line.color.rgb = color
    line.line.width = Pt(2.5)

bw = Inches(2.4)
bh = Inches(0.95)
by1 = Inches(1.65)
by2 = Inches(4.3)
gap = Inches(0.55)

pipeline_box(s, Inches(0.3), by1, bw, bh,
             "Literature (782)", "DRST-filtered samples",
             RGBColor(0x01,0x57,0x9B))
arrow_h(s, Inches(0.3) + bw, by1 + bh//2, gap)
pipeline_box(s, Inches(0.3) + bw + gap, by1, bw, bh,
             "Stage 1  XGBoost", "Pre-train on literature",
             ACCENT)
arrow_h(s, Inches(0.3) + 2*bw + gap, by1 + bh//2, gap)
pipeline_box(s, Inches(0.3) + 2*bw + 2*gap, by1, bw, bh,
             "lit_prior_prediction", "New 68th feature",
             AMBER)

# Stage 2
pipeline_box(s, Inches(0.3), by2, bw, bh,
             "Lab Data (89,074)", "Internal experiments",
             RGBColor(0x1B, 0x5E, 0x20))
arrow_h(s, Inches(0.3) + bw, by2 + bh//2, gap)
pipeline_box(s, Inches(0.3) + bw + gap, by2, bw, bh,
             "Merge Features", "67 original + prior",
             RGBColor(0x33, 0x69, 0x1E))
arrow_h(s, Inches(0.3) + 2*bw + gap, by2 + bh//2, gap)
pipeline_box(s, Inches(0.3) + 2*bw + 2*gap, by2, bw, bh,
             "Stage 2  LightGBM", "Train on OUR labels only",
             GREEN)
arrow_h(s, Inches(0.3) + 3*bw + 2*gap, by2 + bh//2, gap)
pipeline_box(s, Inches(0.3) + 3*bw + 3*gap, by2, bw, bh,
             "Y(C2)% Prediction", "Final output",
             NAVY)

# Diagonal arrow from prior box down to merge box
from pptx.enum.shapes import MSO_CONNECTOR
x1 = Inches(0.3) + 2*bw + 2*gap + bw//2
y1_top = by1 + bh
x2 = Inches(0.3) + bw + gap + bw//2
y2_bot = by2
conn = s.shapes.add_connector(1, x1, y1_top, x2, y2_bot)
conn.line.color.rgb = AMBER
conn.line.width = Pt(2.0)

# Key property callout
add_rect(s, Inches(7.8), Inches(1.5), Inches(5.2), Inches(3.8),
         fill=RGBColor(0xE8, 0xF4, 0xFF), line=ACCENT)
add_text(s, "Key Property", Inches(7.95), Inches(1.55),
         Inches(4.9), Inches(0.4), size=14, bold=True, color=NAVY)
props = [
    "Stage 2 trains only on our labels",
    "Literature bias cannot corrupt\nfinal predictions",
    "Prior feature = chemistry signal\nnot competing label",
    "Model learns to trust / discount\nthe prior per composition",
]
for i, p in enumerate(props):
    add_rect(s, Inches(7.85), Inches(2.05) + Inches(i * 0.72),
             Inches(5.1), Inches(0.65),
             fill=WHITE if i%2==0 else RGBColor(0xF0,0xF7,0xFF))
    add_text(s, "→  " + p,
             Inches(8.0), Inches(2.08) + Inches(i * 0.72),
             Inches(4.85), Inches(0.62), size=12, color=BLACK)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — Results
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
add_rect(s, 0, 0, W, H, fill=WHITE)
slide_title_bar(s, "Results at a Glance")
nav_bar(s, 6, TOTAL)

add_img(s, "fig_results_comparison.png", Inches(0.25), Inches(1.2), Inches(7.0))

# KPI cards — right side
kpis = [
    ("In-distribution RMSE", "2.133%  →  1.907%", "−10.6%", GREEN),
    ("Out-of-distribution RMSE", "6.53%  →  3.60%", "−45%", GREEN),
]
for i, (title, vals, delta, color) in enumerate(kpis):
    bx2 = Inches(7.6)
    by3 = Inches(1.4) + Inches(i * 2.2)
    add_rect(s, bx2, by3, Inches(5.4), Inches(1.9),
             fill=RGBColor(0xF5,0xF9,0xFF), line=ACCENT)
    add_text(s, title, bx2 + Inches(0.15), by3 + Inches(0.1),
             Inches(5.1), Inches(0.38), size=12, bold=True, color=NAVY)
    add_text(s, vals, bx2 + Inches(0.15), by3 + Inches(0.5),
             Inches(5.1), Inches(0.45), size=16, color=BLACK)
    add_rect(s, bx2 + Inches(0.15), by3 + Inches(1.05),
             Inches(2.2), Inches(0.55), fill=color)
    add_text(s, delta,
             bx2 + Inches(0.15), by3 + Inches(1.08),
             Inches(2.2), Inches(0.5),
             size=22, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

add_rect(s, Inches(7.6), Inches(5.9), Inches(5.4), Inches(0.72),
         fill=RGBColor(0xE8, 0xF5, 0xE9), line=GREEN)
add_text(s,
         "OOD test: 2,139 held-out samples, non-Impregnation\nmethods, never seen during any training fold",
         Inches(7.75), Inches(5.94), Inches(5.1), Inches(0.65),
         size=11, color=RGBColor(0x1B, 0x5E, 0x20), italic=True)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — SHAP
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
add_rect(s, 0, 0, W, H, fill=WHITE)
slide_title_bar(s, "What Drives the Model? — SHAP Analysis")
nav_bar(s, 7, TOTAL)

add_img(s, "fig_shap_beeswarm.png", Inches(0.2), Inches(1.15), Inches(6.4))

bx = Inches(6.85)
add_rect(s, bx, Inches(1.2), Inches(6.15), Inches(0.4), fill=ACCENT)
add_text(s, "Top findings (3,000 sample subsample)",
         bx + Inches(0.12), Inches(1.22),
         Inches(5.9), Inches(0.36), size=13, bold=True, color=WHITE)

findings = [
    "#1  lit_prior_prediction — literature knowledge\n     is actively used on every prediction",
    "Temperature — strong positive; higher T favours C2",
    "Ba, Mn, La, Ce — all raise predicted yield",
    "Li, K — mixed / context-dependent effect",
]
for i, f in enumerate(findings):
    bg = RGBColor(0xF0,0xF4,0xFF) if i%2==0 else WHITE
    add_rect(s, bx, Inches(1.62) + Inches(i*0.7),
             Inches(6.15), Inches(0.68), fill=bg)
    add_text(s, "→  " + f,
             bx + Inches(0.12), Inches(1.65) + Inches(i*0.7),
             Inches(5.9), Inches(0.64), size=12, color=BLACK)

add_rect(s, bx, Inches(4.45), Inches(6.15), Inches(0.38), fill=NAVY)
add_text(s, "Residual analysis (ceiling effect)",
         bx + Inches(0.12), Inches(4.47),
         Inches(5.9), Inches(0.34), size=12, bold=True, color=WHITE)

res_rows = [
    ("0 – 6%",   "1.43 – 1.48%", "≈ 0%",    WHITE),
    ("6 – 10%",  "1.95%",         "−0.3%",   RGBColor(0xFF,0xF8,0xE8)),
    ("10 – 15%", "2.58%",         "−1.5%",   RGBColor(0xFF,0xF0,0xD8)),
    ("> 15%",    "4.70%",         "−4.2%",   RGBColor(0xFF,0xEB,0xEB)),
]
rh = Inches(0.42)
for i, (rng, rmse, bias, bg) in enumerate(res_rows):
    add_rect(s, bx, Inches(4.85) + Inches(i*rh/Inches(1)),
             Inches(6.15), rh, fill=bg, line=LGRAY)
    for j, (txt, cw2, cx2) in enumerate([
        (rng,  Inches(1.6),  bx+Inches(0.1)),
        (rmse, Inches(1.5),  bx+Inches(1.85)),
        (bias, Inches(1.5),  bx+Inches(3.5)),
    ]):
        col = RED if i==3 and j==2 else BLACK
        add_text(s, txt, cx2, Inches(4.87) + Inches(i*rh/Inches(1)),
                 cw2, rh - Inches(0.04), size=11,
                 color=col, align=PP_ALIGN.CENTER)

add_rect(s, bx, Inches(6.55), Inches(6.15), Inches(0.58),
         fill=RGBColor(0xFF,0xF0,0xD8), line=AMBER)
add_text(s, "Root cause: GHSV, CH₄/O₂ ratio, pressure absent from dataset",
         bx + Inches(0.12), Inches(6.58),
         Inches(5.9), Inches(0.52), size=11,
         color=RGBColor(0x6B, 0x3A, 0x00), italic=True)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — Next Steps
# ════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
add_rect(s, 0, 0, W, H, fill=WHITE)
slide_title_bar(s, "Limitations & Next Steps")
nav_bar(s, 8, TOTAL)

options = [
    ("Option A", "Easy", GREEN,
     "Add Reaction Conditions",
     "Collect GHSV, CH₄/O₂ ratio, and pressure as features. Expected to fix the >15% ceiling effect and reduce high-yield RMSE from 4.70% to ~2.5%."),
    ("Option B", "Medium", AMBER,
     "Tune Transfer Threshold",
     "Grid-search DRST τ via nested CV. Current τ=0.30 was manually swept. Proper search may recover 0.3–0.5% RMSE."),
    ("Option C", "Hard", RED,
     "Neural Domain Adaptation",
     "Replace XGBoost with domain-adversarial neural network (DANN/CORAL). High complexity — only justified after reaction conditions are added."),
]

for i, (opt, diff, col, title, desc) in enumerate(options):
    cx3 = Inches(0.3) + Inches(i * 4.35)
    add_rect(s, cx3, Inches(1.3), Inches(4.1), Inches(4.85),
             fill=RGBColor(0xF8,0xF9,0xFF), line=col)
    add_rect(s, cx3, Inches(1.3), Inches(4.1), Inches(0.55), fill=col)
    add_text(s, f"{opt}  —  {diff}",
             cx3 + Inches(0.1), Inches(1.33),
             Inches(3.9), Inches(0.48),
             size=14, bold=True, color=WHITE)
    add_text(s, title,
             cx3 + Inches(0.12), Inches(1.98),
             Inches(3.86), Inches(0.5),
             size=15, bold=True, color=NAVY)
    add_text(s, desc,
             cx3 + Inches(0.12), Inches(2.52),
             Inches(3.86), Inches(2.0),
             size=12, color=DGRAY)

# Limitations + Recommendation
add_rect(s, Inches(0.3), Inches(6.25), Inches(6.0), Inches(0.88),
         fill=RGBColor(0xF5,0xF5,0xFF), line=ACCENT)
add_text(s, "Key Limitations",
         Inches(0.45), Inches(6.27), Inches(5.7), Inches(0.35),
         size=12, bold=True, color=NAVY)
add_text(s, "Missing GHSV / CH₄:O₂ / pressure  •  Publication bias  •  OOD coverage uneven",
         Inches(0.45), Inches(6.6), Inches(5.7), Inches(0.48),
         size=11, color=DGRAY)

add_rect(s, Inches(6.6), Inches(6.25), Inches(6.4), Inches(0.88),
         fill=RGBColor(0xE8, 0xF5, 0xE9), line=GREEN)
add_text(s, "Recommended Priority",
         Inches(6.75), Inches(6.27), Inches(6.1), Inches(0.35),
         size=12, bold=True, color=RGBColor(0x1B,0x5E,0x20))
add_text(s, "1. Add GHSV + CH₄/O₂  (highest impact)     2. Re-run two-stage     3. Tune threshold",
         Inches(6.75), Inches(6.6), Inches(6.1), Inches(0.48),
         size=11, color=RGBColor(0x1B, 0x5E, 0x20))

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
    print("LibreOffice not found — skipping PDF export")
