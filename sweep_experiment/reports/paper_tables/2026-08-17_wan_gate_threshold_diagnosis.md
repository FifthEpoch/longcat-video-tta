# Gated vs always-on — can a new threshold win? (2026-08-17)

**Source:** `i2v_bon_16v` summaries (jobs 15884598/599/600). Incoming
from gated logs (NOTTA prefix until first fire). always-on first
divergence vs last-chunk composite.

## The pattern

always-on first leaves cand0 at **chunk 1 on 13/16 videos**. Gated with
T=2.0 usually skips that chunk (incoming 0.2–1.3). The last-chunk gap
is decided there, not at T=2.0.

| i | always first-div chunk | incoming then | always−NOTTA (last) | gated−always | Read |
|---|---|---|---|---|---|
| 06 | 1 | **0.20** | **+0.61** (always hurt) | −0.74 | skip was right |
| 14 | 1 | 0.25 | −0.01 | +0.01 | don't care |
| 15 | 1 | 0.40 | −0.37 | +0.06 | small |
| 07 | 1 | **0.68** | **+0.66** (always hurt) | −0.66 | skip was right |
| 05 | 1 | **0.87** | −1.08 | +0.42 | **miss** |
| 09 | 1 | **0.89** | −0.97 | +0.33 | **miss** |
| 02 | 1 | **0.90** | −1.11 | +1.11 | **miss** |
| 13 | 1 | 1.05 | −0.22 | +0.22 | small |
| 12 | 2 | **1.09** | −3.12 | +2.60 | **miss** (late) |
| 03 | 1 | **1.27** | −1.23 | +1.12 | **miss** |
| 01 | 1 | 1.27 | −0.03 | +0.03 | tiny |
| 08 | 2 | 1.36 | −0.50 | +0.10 | late miss |
| 10 | 4 | 2.25 | −0.30 | 0 | caught |
| 04 | 1 | 2.41 | −4.18 | 0 | caught |
| 11 | 1 | 2.38 | −6.87 | −2.16 | fired; gated won |
| 00 | 3 | 2.46 | −0.52 | 0 | caught |

Correct-skip incoming (always hurt): **0.20 and 0.68**.
Big-miss incoming (always gain >0.9): **0.87–1.27**.

Those two bands touch. A global T cannot keep 07 (0.68) skipped and
still fire 05/09/02 (0.87–0.90) with any margin.

## What T=0.8 would do (from incoming only)

Would fire chunk 1 on 02, 03, 05, 09, 01, 13 (incoming ≥0.87).
Would still skip 06 (0.20) and 07 at chunk 1 (0.68).
Would fire 07 at chunk 3 (incoming 0.98) — risk of recreating the
always-on hurt, but on a NOTTA prefix, not always-on's prefix.
Would fire 12 at chunk 2 (1.09) — the +2.60 miss.

This is the only single-T candidate worth simulating. Need always-on's
**chunk-1 candidate scores** (shared prefix → valid offline).

## Why threshold-only is the wrong lever

Gated loses when always-on finds a better **early** continuation while
incoming still looks healthy. That is a forecasting problem, not a
level-crossing problem. Ideas that can beat always-on without copying it:

1. **Trend gate** — fire if Δincoming > ~0.4 even if level < 2. Video 12
   rises 0.53→1.09→1.78→3.69. Video 07 is 0.68→0.61→0.98→1.54 (flat
   then slow). Separates 12 from 07 better than a level.
2. **Per-signal gate** — fire on sharpness or motion deviation, not the
   4-way composite. Wan's 30 s signature is sharpen+freeze.
3. **Stay-on hysteresis** — once fired, keep searching the rest of the
   video (11 already does this well; 12 fired too late to recover).
4. **Do not** “always search chunk 1.” That is how always-on hurt 06/07.

Do not retune T and resubmit 16v until the chunk-1 score dump below
shows that always-on's ch1 best actually beats cand0 on the miss videos.
