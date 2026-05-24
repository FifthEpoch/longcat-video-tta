#!/usr/bin/env python3
"""
Generate all assets for the one-page AdaSteer TTA summary.

Produces:
  - filmstrip_comparison.png  (GT vs Baseline vs AdaSteer frame grids)
  - fvd_comparison.png        (FVD bar chart)
  - compute_cost.png          (stacked time bar chart)
  - one_pager.html            (self-contained HTML with base64-embedded images + KaTeX)

Usage:
    python3 generate_one_pager.py [--diagram PATH_TO_DIAGRAM_IMAGE]
"""
from __future__ import annotations

import argparse
import base64
import glob
import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).parent
VIDEO_BASE = Path(
    "/Users/macrohard/Downloads/generated_video/using_longcat/04_17_2026"
)

VIDEOS = [
    {"idx": 96, "caption": "Person holding plastic animals", "psnr_gain": "+0.21 dB"},
    {"idx": 380, "caption": "News report about a pool", "psnr_gain": "+0.20 dB"},
    {"idx": 174, "caption": "Living room interior", "psnr_gain": "+0.13 dB"},
]

NUM_FRAMES = 6


def find_video(directory: Path, idx: int) -> Path | None:
    pattern = str(directory / f"{idx}_*")
    matches = glob.glob(pattern)
    return Path(matches[0]) if matches else None


def extract_frames(video_path: Path, n: int = NUM_FRAMES) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def make_filmstrip(out_path: Path) -> None:
    thumb_h, thumb_w = 108, 187
    label_w = 120
    pad = 3
    border = 2

    n_videos = len(VIDEOS)
    row_h = thumb_h + pad
    block_h = 3 * row_h + 18
    total_h = n_videos * block_h + 10
    strip_w = NUM_FRAMES * (thumb_w + pad) - pad
    total_w = label_w + strip_w + 10

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        font_caption = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font
        font_caption = font

    sources = [
        ("Ground Truth", VIDEO_BASE / "gt_panda_cut" / "gt_videos"),
        ("No-TTA Baseline", VIDEO_BASE / "notta_baseline_g4" / "annotated"),
        ("AdaSteer (Ours)", VIDEO_BASE / "adasteer_bare_g4" / "annotated"),
    ]

    row_colors = [
        (230, 245, 230),
        (245, 230, 230),
        (230, 235, 250),
    ]

    for vi, vinfo in enumerate(VIDEOS):
        y_base = vi * block_h
        draw.text(
            (label_w, y_base),
            f"{vinfo['caption']}  ({vinfo['psnr_gain']})",
            fill=(80, 80, 80), font=font_caption,
        )
        y_base += 16

        for ri, (label, src_dir) in enumerate(sources):
            y = y_base + ri * row_h
            draw.rectangle([0, y, label_w - 5, y + thumb_h], fill=row_colors[ri])
            draw.text((6, y + thumb_h // 2 - 8), label, fill=(40, 40, 40),
                      font=font_small)

            vpath = find_video(src_dir, vinfo["idx"])
            if vpath is None:
                draw.text((label_w + 10, y + 20),
                          f"[Video not found: {vinfo['idx']}]",
                          fill=(200, 0, 0), font=font)
                continue

            frames = extract_frames(vpath, NUM_FRAMES)
            for fi, frame in enumerate(frames):
                x = label_w + fi * (thumb_w + pad)
                thumb = Image.fromarray(frame).resize(
                    (thumb_w, thumb_h), Image.LANCZOS)
                draw.rectangle(
                    [x - border, y - border,
                     x + thumb_w + border, y + thumb_h + border],
                    outline=row_colors[ri], width=border)
                canvas.paste(thumb, (x, y))

    canvas.save(out_path, dpi=(300, 300))
    print(f"  Saved filmstrip: {out_path}")


def make_fvd_chart(out_path: Path) -> None:
    methods = ["No-TTA\nBaseline", "LoRA\n(best)", "AdaSteer\n(Ours)"]
    fvd = [641.1, 644.6, 561.1]
    colors = ["#AAAAAA", "#F0C75E", "#4E9A5E"]

    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    bars = ax.bar(methods, fvd, color=colors, width=0.55, edgecolor="white",
                  linewidth=1.2)
    for bar, val in zip(bars, fvd):
        delta = val - 641.1
        label = f"{val:.0f}"
        if delta < 0:
            label += f"\n({delta:+.0f})"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                label, ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("FVD (lower = better)", fontsize=10)
    ax.set_ylim(450, 700)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(y=641.1, color="#AAAAAA", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title("Video Quality (FVD)", fontsize=12, fontweight="bold", pad=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white",
                transparent=False)
    plt.close(fig)
    print(f"  Saved FVD chart: {out_path}")


def make_compute_chart(out_path: Path) -> None:
    methods = ["No-TTA\nBaseline", "LoRA\n(best)", "AdaSteer\n(Ours)"]
    gen_time = [80, 80, 80]
    tta_time = [0, 19, 54]

    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.bar(methods, gen_time, color="#B0C4DE", width=0.55,
           label="Generation (fixed)", edgecolor="white", linewidth=1.2)
    ax.bar(methods, tta_time, bottom=gen_time, color="#4E79A7",
           width=0.55, label="TTA Training", edgecolor="white", linewidth=1.2)

    totals = [g + t for g, t in zip(gen_time, tta_time)]
    for i, total in enumerate(totals):
        ax.text(i, total + 2, f"{total}s", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_ylabel("Time per Video (seconds)", fontsize=10)
    ax.set_ylim(0, 170)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.set_title("Compute Cost", fontsize=12, fontweight="bold", pad=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white",
                transparent=False)
    plt.close(fig)
    print(f"  Saved compute chart: {out_path}")


def img_to_base64(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "svg": "image/svg+xml"}.get(suffix, "image/png")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{data}"


def build_html(diagram_path=None):
    filmstrip_b64 = img_to_base64(OUT_DIR / "filmstrip_comparison.png")
    fvd_b64 = img_to_base64(OUT_DIR / "fvd_comparison.png")
    compute_b64 = img_to_base64(OUT_DIR / "compute_cost.png")

    if diagram_path and os.path.isfile(diagram_path):
        diagram_b64 = img_to_base64(Path(diagram_path))
        diagram_html = f'<img src="{diagram_b64}" class="diagram-img" />'
    else:
        diagram_html = ('<div class="diagram-placeholder">'
                        '[Method Diagram -- provide via --diagram flag]'
                        '</div>')

    # Use percent-style formatting to avoid issues with CSS braces
    html_template = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AdaSteer: Test-Time Adaptation for Video Diffusion</title>
<!-- KaTeX for LaTeX-style equation rendering -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {delimiters: [
    {left: '$$', right: '$$', display: true},
    {left: '\\(', right: '\\)', display: false}
  ]});"></script>
<style>
  @page { size: letter; margin: 0.4in 0.5in 0.35in 0.5in; }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 8.5pt; line-height: 1.32; color: #222;
    max-width: 7.5in; margin: 0 auto;
  }
  .title-block {
    text-align: center; margin-bottom: 6px;
    padding-bottom: 5px; border-bottom: 2px solid #4E9A5E;
  }
  .title-block h1 { font-size: 14pt; font-weight: 700; color: #1a1a1a; margin-bottom: 1px; }
  .title-block .subtitle { font-size: 8.5pt; color: #444; font-style: italic; margin-bottom: 1px; }
  .title-block .authors { font-size: 8pt; color: #666; }

  .filmstrip-section { margin-bottom: 6px; }
  .filmstrip-img { width: 100%%; display: block; }

  h2 { font-size: 9pt; color: #333; margin-bottom: 3px;
    border-left: 3px solid #4E9A5E; padding-left: 5px; margin-top: 2px; }

  .two-col { display: flex; gap: 12px; margin-bottom: 5px; }
  .col-left { flex: 0 0 40%%; }
  .col-right { flex: 1; }

  .diagram-img { width: 100%%; border: 1px solid #ddd; border-radius: 3px; }
  .diagram-placeholder {
    width: 100%%; height: 160px; border: 2px dashed #ccc; border-radius: 4px;
    display: flex; align-items: center; justify-content: center;
    color: #999; font-style: italic; font-size: 8pt;
  }

  .section-text { margin-bottom: 4px; }
  .section-text p { margin-bottom: 3px; text-align: justify; }
  .eq-block { margin: 3px 0 3px 12px; font-size: 9pt; }

  .charts-row { display: flex; gap: 10px; margin-bottom: 5px; }
  .chart-container { flex: 1; text-align: center; }
  .chart-container img { width: 100%%; max-height: 185px; object-fit: contain; }

  .key-numbers {
    display: flex; gap: 8px; justify-content: center;
    padding: 5px 0; border-top: 1px solid #ddd;
  }
  .kn-box {
    text-align: center; padding: 3px 14px;
    background: #f5f9f5; border-radius: 4px; border: 1px solid #d5e5d5;
  }
  .kn-box .kn-val { font-size: 13pt; font-weight: 700; color: #4E9A5E; }
  .kn-box .kn-label { font-size: 7pt; color: #666; display: block; }
</style>
</head>
<body>

<!-- TITLE -->
<div class="title-block">
  <h1>AdaSteer: Adaptive Shared Timestep Embedding Efficient Residual</h1>
  <div class="subtitle">Test-Time Adaptation for Video Diffusion Transformers</div>
  <div class="authors">LongCat-Video Project &nbsp;|&nbsp; NYU &nbsp;|&nbsp; February 2026</div>
</div>

<!-- FILMSTRIP -->
<div class="filmstrip-section">
  <h2>Visual Comparison: Ground Truth vs. Baseline vs. AdaSteer (Ours)</h2>
  <img src="%%FILMSTRIP_B64%%" class="filmstrip-img" />
</div>

<!-- METHOD DIAGRAM + INTRO + METHODOLOGY -->
<div class="two-col">
  <div class="col-left">
    <h2>Method</h2>
    %%DIAGRAM_HTML%%

    <h2>Methodology</h2>
    <div class="section-text">
      <p>
        AdaSteer operates on the timestep embedding pathway. In LongCat-Video,
        the timestep embedder produces \(t \in \mathbb{R}^{512}\), which each
        block's adaLN projects into shift, scale, and gate vectors:
      </p>
      <div class="eq-block">
        $$t \;\xrightarrow{\;\text{adaLN}_i\;}\;
        \bigl[\gamma_i^{\text{msa}},\, \beta_i^{\text{msa}},\, \alpha_i^{\text{msa}},\,
        \gamma_i^{\text{mlp}},\, \beta_i^{\text{mlp}},\, \alpha_i^{\text{mlp}}\bigr]
        \in \mathbb{R}^{6 \times 4096}$$
      </div>
      <p>
        AdaSteer learns a single additive offset
        \(\delta \in \mathbb{R}^{512}\) applied before this projection:
      </p>
      <div class="eq-block">
        $$t' = t + \delta$$
      </div>
      <p>
        Because each block has its own frozen adaLN projection
        (SiLU&nbsp;+&nbsp;Linear: 512&rarr;24,576), the same \(\delta\)
        produces <em>block-specific</em> perturbations to modulation vectors.
        This is a <strong>structured weight-tying strategy</strong>: all
        48 blocks share 512 trainable inputs, but the pretrained projections
        act as learned de-tying transforms, providing per-block
        specialization for free.
      </p>
      <p>
        We also explore a <strong>TinyLoRA</strong> variant operating on
        attention weights via truncated SVD:
      </p>
      <div class="eq-block">
        $$y = Wx + \tfrac{\alpha}{r}\, U_r\!\bigl(v \odot (V_r^\top x)\bigr)$$
      </div>
      <p>
        With \(r\!=\!2\), 48 blocks, and 2 target modules per block,
        TinyLoRA trains 192 scalars. Full weight tying reduces this to 4.
        Both methods use 10 steps of Adam (lr=0.005) on the flow-matching
        denoising loss over the 14 conditioning frames. No future-frame
        supervision. Adapted parameters are discarded after generation.
      </p>
    </div>
  </div>

  <div class="col-right">
    <h2>Introduction</h2>
    <div class="section-text">
      <p>
        Video diffusion transformers achieve state-of-the-art generation quality,
        but their inference is instance-agnostic: the same frozen weights process
        every test video regardless of its content, motion, and style.
        Test-time adaptation (TTA) offers a remedy&mdash;briefly adapting a
        pretrained model on a single test instance before generation&mdash;but
        applying TTA to large-scale video DiTs (10B+ parameters) presents a
        fundamental challenge: the extreme asymmetry between model capacity
        and the available adaptation signal (typically 14 frames,
        ~300K latent values).
      </p>
      <p>
        We introduce <strong>AdaSteer</strong> (<strong>Ada</strong>ptive
        <strong>S</strong>hared <strong>T</strong>imestep
        <strong>E</strong>mbedding <strong>E</strong>fficient
        <strong>R</strong>esidual), a TTA method that steers a frozen video
        DiT by learning a single additive perturbation
        \(\delta \in \mathbb{R}^{512}\) to the timestep embedding
        shared across all 48 transformer blocks. Each block's frozen adaLN
        projection then maps this shared perturbation into block-specific
        modulation of the 4096-dim hidden states (shift, scale, and gate for
        self-attention and MLP). This is a <strong>structured weight-tying
        strategy</strong>: all blocks share one 512-dim trainable input, but
        the pretrained adaLN projections act as learned de-tying transforms,
        producing 48 distinct high-dimensional modulations from one compact
        perturbation.
      </p>
      <p>
        We evaluate on LongCat-Video (13.6B-parameter DiT, 48 blocks,
        hidden dim 4096) in the video continuation setting: given 14 observed
        frames at 480p, the model generates 28 future frames. We compare
        against LoRA (ranks 1&ndash;8, 5&ndash;20 steps), full-model
        fine-tuning (13.6B params), and a no-TTA baseline on Panda-70M
        and UCF-101.
      </p>
    </div>

    <h2>Experimental Results</h2>
    <div class="section-text">
      <p>
        On Panda-70M (100 videos, 480p), AdaSteer achieves
        <strong>FVD&nbsp;=&nbsp;561.1</strong>, a reduction of 80 points
        (12.5%%) over the no-TTA baseline (641.1), at +54s compute per video
        (67%% overhead). Per-frame metrics (PSNR, SSIM, LPIPS) remain
        unchanged, indicating AdaSteer improves <em>distributional</em>
        quality&mdash;temporal coherence and motion realism&mdash;rather
        than per-pixel fidelity.
      </p>
      <p>
        LoRA-based TTA fails to improve FVD across all tested configs.
        Sweeping rank&nbsp;\(\in\{1,4,8\}\), block subsets,
        step counts (5&ndash;20), and LRs (\(10^{-5}\) to
        \(2\!\times\!10^{-4}\)), the best LoRA achieves FVD&nbsp;=&nbsp;644.6
        (baseline: 641.1). Higher LRs cause catastrophic overfitting
        (PSNR drops from 18.6 to 17.4). Full-model TTA shows +0.01 PSNR.
      </p>
      <p>
        An 8-way ablation isolates auxiliary features: early stopping gains
        marginal FVD (&minus;85 vs &minus;80) but doubles training time;
        augmentation improves PSNR (+0.05 dB) but <em>hurts</em> FVD by 26
        points; CLIP gating and gradient accumulation provide no benefit.
        Bare AdaSteer is the Pareto-optimal configuration.
      </p>
    </div>

    <h2>Next Steps</h2>
    <div class="section-text">
      <p>
        (1)&nbsp;1000-video full-scale Panda-70M evaluation (running);
        (2)&nbsp;TinyLoRA ablation: 13 configs exploring SVD rank, weight
        tying, target scope, block subsets (running);
        (3)&nbsp;SAVi-DNO comparison adapted for DiT backbone;
        (4)&nbsp;investigating the distributional-vs-pointwise quality
        tradeoff under rank-1 constraints.
      </p>
    </div>
  </div>
</div>

<!-- CHARTS -->
<div class="charts-row">
  <div class="chart-container">
    <img src="%%FVD_B64%%" />
  </div>
  <div class="chart-container">
    <img src="%%COMPUTE_B64%%" />
  </div>
</div>

<!-- KEY NUMBERS -->
<div class="key-numbers">
  <div class="kn-box">
    <span class="kn-val">561</span>
    <span class="kn-label">FVD (&minus;80 vs baseline)</span>
  </div>
  <div class="kn-box">
    <span class="kn-val">512</span>
    <span class="kn-label">Trainable Parameters</span>
  </div>
  <div class="kn-box">
    <span class="kn-val">+54s</span>
    <span class="kn-label">Added Compute / Video</span>
  </div>
  <div class="kn-box">
    <span class="kn-val">0</span>
    <span class="kn-label">Architecture Changes</span>
  </div>
</div>

</body>
</html>'''

    html = html_template.replace("%%FILMSTRIP_B64%%", filmstrip_b64)
    html = html.replace("%%FVD_B64%%", fvd_b64)
    html = html.replace("%%COMPUTE_B64%%", compute_b64)
    html = html.replace("%%DIAGRAM_HTML%%", diagram_html)

    out_path = OUT_DIR / "one_pager.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"  Saved HTML: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagram", type=str, default=None,
                        help="Path to method diagram image")
    args = parser.parse_args()

    print("Generating one-pager assets...")
    print()

    print("[1/4] Extracting filmstrip frames...")
    make_filmstrip(OUT_DIR / "filmstrip_comparison.png")

    print("[2/4] Generating FVD chart...")
    make_fvd_chart(OUT_DIR / "fvd_comparison.png")

    print("[3/4] Generating compute cost chart...")
    make_compute_chart(OUT_DIR / "compute_cost.png")

    print("[4/4] Building HTML...")
    build_html(args.diagram)

    print()
    print("Done! Open one_pager.html in a browser and print to PDF.")
    print(f"Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
