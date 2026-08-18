# Wan I2V test-time controller — briefing

**Audience:** weekly recap / PI update.
**Date:** 2026-08-18.
**Status:** search-while-sick is the best *handcrafted* median (2.764)
and hits the 11/16 + 24 checklist; hybrid is still cheapest (173 s).
Official VBench is incomplete (15959601 scored do-nothing only).
Do not start test-time training.

This note is the full picture: what we generate, how we measure drift,
the exact equations, how the controller uses those numbers, and the
headline results. Lower composite scores are better.

---

## 1. What we are trying to do

Long video generation from a still image **drifts**. On Wan 1.3B at
30 seconds the typical clip **sharpens** (median +167%) and **freezes**
(median motion −60%). That is the headroom.

The paper claim is not “search four random seeds.” It is a
**drift-gated, ground-truth-free test-time controller**: decide, from
the video itself, **when extra compute is worth it**, without a real
video to compare against.

Locked comparison, same images and seeds:

| Method | What it does |
|---|---|
| Do-nothing | Generate each later piece once (default seed). |
| Always-search | Generate four candidates every later piece; keep the best. |
| Gated-search | Search four ways only when a drift alarm says the last second looks sick. |

If gated-search **beats** always-search on quality → controller paper.
If it **ties and is cheaper** → efficiency paper (where we are).
If it **loses** → drop the gating claim.

---

## 2. Setup

- **Model:** Wan2.1-T2V-1.3B + Self-Forcing causal DMD, image-to-video
  continuation (not text-to-video from scratch).
- **Images:** VBench-I2V stills, first 32 unique names, seed 0.
- **Horizon:** 30 seconds at 16 fps → **481 pixel frames**
  (1 condition frame + 480 generated).
- **Latents:** 1 condition + 120 generated. Split into **five pieces**
  of 24 generated latents (~6 seconds) each.
- **Piece 0** is always the default seed. Everyone shares this prefix.
  The first 16 frames **after** the still (1 second) are the
  **reference** — “what this video looked like when it was still honest.”
- **Pieces 1–4** are where methods may search.

We cannot pass a growing prefix through the official
`CausalInferencePipeline.inference()` (it only caches frame 0). The
runner replays committed latents into the KV cache, then denoises the
next piece.

---

## 3. How we measure drift (no real video)

All scores are **ground-truth-free**. We never look at a real 30-second
clip. We compare generated frames to the first-second-after-the-still
reference.

Frames \(x_t \in [0,1]^{H \times W \times 3}\). Gray:

\[
g_t = 0.299\, r_t + 0.587\, g_t + 0.114\, b_t
\]

### 3.1 Sharpness — Laplacian variance

Discrete Laplacian on gray (4-neighbour), then variance, then mean
over the window of \(T\) frames:

\[
L(g) = 4g - \mathrm{roll}_y(g) - \mathrm{roll}_{-y}(g) - \mathrm{roll}_x(g) - \mathrm{roll}_{-x}(g)
\]

\[
\mathrm{sharp}(x) = \frac{1}{T}\sum_{t=1}^{T} \mathrm{Var}\!\big(L(g_t)_{\mathrm{interior}}\big)
\]

High sharpness vs the reference = oversharpening (Wan’s 30 s signature).

### 3.2 Colorfulness — Hasler–Süsstrunk

Per frame, then mean over the window:

\[
rg = r-g,\quad yb = \tfrac{1}{2}(r+g)-b
\]

\[
\mathrm{color}(x) = \sqrt{\sigma_{rg}^2 + \sigma_{yb}^2} + 0.3\sqrt{\mu_{rg}^2 + \mu_{yb}^2}
\]

### 3.3 Contrast — luma standard deviation

\[
\mathrm{contrast}(x) = \frac{1}{T}\sum_{t=1}^{T} \sigma(g_t)
\]

### 3.4 Temporal motion — mean absolute frame difference

\[
\mathrm{motion}(x) = \frac{1}{T-1}\sum_{t=2}^{T} \mathrm{mean}\!\big(|x_t - x_{t-1}|\big)
\]

Low motion vs the reference = freeze.

### 3.5 Seam — cut between the last committed frame and the new piece

\[
\mathrm{seam} = \mathrm{mean}\big(|x^{\mathrm{new}}_1 - x^{\mathrm{prev}}_{\mathrm{last}}|\big)
\]

Normalized by **reference** motion so a frozen video is not rewarded
for a small jump:

\[
\mathrm{seam\text{-}term} = \frac{\mathrm{seam}}{\mathrm{motion}(x^{\mathrm{ref}}) + 10^{-6}}
\]

---

## 4. Turning signals into one number

For each signal \(k \in \{\mathrm{sharp},\,\mathrm{color},\,\mathrm{contrast},\,\mathrm{motion}\}\):

\[
d_k = \frac{\big|s_k(x) - s_k(x^{\mathrm{ref}})\big|}{\big|s_k(x^{\mathrm{ref}})\big| + 10^{-6}}
\]

Two-sided: being *too sharp* and *too soft* both cost. Freeze is not
a win.

**Incoming drift** (gate input) — last 1 second already committed,
**no seam** (\(\lambda = 0\)):

\[
\mathrm{incoming} = \sum_k d_k
\]

**Candidate / last-piece score** (what we minimize when we search) —
the new piece vs the reference, **with seam** (\(\lambda = 1\)):

\[
\mathrm{score} = \sum_k d_k + \lambda\cdot\mathrm{seam\text{-}term}
\]

**Outgoing drift** — last 1 second **after** we commit this piece,
no seam. Used to decide whether the video recovered.

\[
\mathrm{recovery} = \mathrm{incoming} - \mathrm{outgoing}
\]

Positive recovery = the last second looks closer to the reference than
it did before this piece.

We **cite medians**, not means. Video 26 (spiral galaxy) scores 85.6
under both search methods vs 5.1 do-nothing and wrecks every average.

---

## 5. How the controller uses those numbers

On pieces 1–4, gated-search fires (generate 4 seeds, keep lowest
`score`) if any alarm is true:

1. **Early.** This is piece 1 and \(\mathrm{incoming} > 0.8\).
2. **Late.** \(\mathrm{incoming} > 2.0\).
3. **Trend.** \(\mathrm{incoming} - \mathrm{incoming}_{\mathrm{prev}} > 0.5\)
   and \(\mathrm{incoming}_{\mathrm{prev}} > 0.5\).

Otherwise generate once (default seed). Candidate 0 is always that
default seed, so a search that picks 0 paid for three extra tries and
learned nothing.

**Stay-on (tested):** after the first alarm, search every later piece.
Caught highway (03) and busy street (24). Became always-search on 21/32
videos. Erased smoke (11) and book-on-fire (16), where hybrid had
**slept after a recovery**. Cost 256 s vs always-search 258 s.

**Search-while-sick (in flight, job 15959146):** stay-on, but turn
memory off if \(\mathrm{recovery} > 0.5\) or
\(\mathrm{outgoing} < 1.0\). A later alarm can still wake the video.
Goal: keep 11/16 and 03/24.

---

## 6. Headline numbers (cite these)

**Drift, do-nothing, N=16** (head 1 s vs tail 1 s, skip the still):

| Horizon | Sharpness (median) | Motion (median) |
|---|---|---|
| 5 s | +11% | −14% |
| 30 s | **+167%** | **−60%** |

**Last-piece composite, N=32, 30 s** (lower better; video 26 excluded
from the “exclude-26” row only):

| Method | Median | Mean wall | vs always-search |
|---|---|---|---|
| Do-nothing | 3.68 | 92 s | — |
| Always-search (k=4) | **2.97** | 258 s | 25/32 beat do-nothing (median Δ −0.43) |
| Hybrid gated | 3.04 | **173 s** | 9 / 10 / 13, median Δ **0**, **33% cheaper** |
| Stay-on gated | 2.99 | 256 s | 6 / **21** / 5, median Δ 0 (delayed always-search) |

Search works. Hybrid keeps the typical gain and spends one third less.
Stay-on copied always-search and spent the saving. That is why the
honest line is still **efficiency, not a quality win**.

Always-search **hurts** do-nothing on fries (06), beach (07), church
(30), and the galaxy (26). Hybrid still saves 06 and 07 by not
searching them early.

---

## 7. What we are not doing

- The **controller loop** does not look at a real 30-second video.
  Incoming / score / outgoing stay GT-free. That is not the same as
  skipping official metrics on the finished mp4s — see §9.
- We do not train the model at test time. 11 and 16 showed that
  “intervene more” can destroy a good prefix.
- We do not slide one cutoff on every piece of every video. 07 at 0.68
  (must skip) and 05 at 0.87 (must catch) sit next to each other.

---

## 8. Equations at a glance

\[
d_k = \frac{|s_k - s_k^{\mathrm{ref}}|}{|s_k^{\mathrm{ref}}| + 10^{-6}}
\qquad
\mathrm{incoming} = \sum_k d_k
\qquad
\mathrm{score} = \sum_k d_k + \frac{\mathrm{seam}}{\mathrm{motion}^{\mathrm{ref}} + 10^{-6}}
\]

\[
\mathrm{fire} =
(\mathrm{piece}{=}1 \wedge \mathrm{incoming}>0.8)
\;\vee\;
(\mathrm{incoming}>2)
\;\vee\;
(\Delta\mathrm{incoming}>0.5 \wedge \mathrm{incoming}_{\mathrm{prev}}>0.5)
\]

Search-while-sick extra: after a search, turn memory off if
\(\mathrm{incoming}-\mathrm{outgoing}>0.5\) or
\(\mathrm{outgoing}<1\).

Code: `wan_experiment/scripts/i2v_verifier.py`,
`wan_experiment/scripts/run_i2v_chunked.py`.
Results tables:
`2026-08-17_wan_i2v_bon32_hybrid.md`,
`2026-08-18_wan_i2v_bon32_sticky.md`,
`2026-08-17_wan_i2v_notta16_drift.md`.

---

## 9. Outcome scorecard (added same day)

The hybrid / sticky / sick numbers above are **handcrafted last-piece
composites**. They are allowed inside the controller. They are **not**
enough to claim the videos got better.

Official outcome eval (no new generation): VBench quality dims on the
hybrid 32 mp4s, `last5` then `full`. Spec:
`2026-08-18_wan_i2v_official_eval_spec.md`.

```bash
bash wan_experiment/sbatch/submit_i2v_vbench_hybrid32.sh
```

These 32 stills have no paired 30 s GT. Do not invent PSNR. If
Spearman(last-chunk, VBench) is near zero, the verifier is not a
quality proxy and the efficiency story is unverified.
