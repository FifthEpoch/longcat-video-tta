# Caption WAVE=1 outcomes (updated 2026-08-25 02:34)

Series `v2v_panda_caption_32v`. Prompt = first-segment Panda
caption (`prompt_source=metadata_csv` on every harvested sidecar).
Do **not** mix with stem-prompt tables.

Official quality is VBench / VBench++ on the **full clip**.
**16310330** still **R** (~1h22). `joined.json` exists for notta /
rolling / rewind only — provisional until 330 COMPLETED 0:0.
Remaining methods stay blank. Tails are a generate diagnostic.
Dated table: `2026-08-25_wan_v2v_caption_always_adasteer.md`.

## Jobs

| Job | Method | State | n |
|---|---|---|---|
| **16310318** | notta | COMPLETED 0:0 1h47 | **32/32** |
| **16310319** | rolling_notta | COMPLETED 0:0 27m | **32/32** |
| **16310320** | sf_rewind | COMPLETED 0:0 1h13 | **32/32** |
| **16310321** | sf_sick_search | COMPLETED 0:0 1h37 | **32/32** |
| **16310322** | sf_pseudo | COMPLETED 0:0 2h44 | **32/32** |
| **16310323** | sf_sink | COMPLETED | **32/32** |
| **16310324** | sf_always_search | COMPLETED 0:0 3h08 | **32/32** |
| **16310325–329** | RF always / rewind / sick / pseudo / sink | COMPLETED | **32/32** |
| **16310330** | VBench full clip | **R ~1h22** | 3/12 methods written |
| **16314667–669** | AdaSteer 8v (first) | FAILED 2:0 inference_mode | closed |
| **16321558 / 560 / 562** | AdaSteer 8v (retry) | FAILED 2:0 inplace IM | closed |
| **16321563** | AdaSteer VBench | CANCELLED | — |
| **16326033** | ada_fixed | COMPLETED 0:0 19m | **8/8** |
| **16326034** | ada_stream | COMPLETED 0:0 21m | **8/8** |
| **16326035** | ada_resid | COMPLETED 0:0 18m | **8/8** |
| **16326036** | AdaSteer VBench | COMPLETED 0:0 20m | **8/8** |

## vs caption Self-Forcing (SF-hosted)

| Method | n | tail | vs SF | W/L/tie | Official VBench |
|---|---:|---:|---:|---|---|
| notta (SF) | 32 | **0.01164** | — | — | 0.700 · 71.54 · 0 · 0.989 |
| rolling_notta (RF host) | 32 | 0.01423 | +22% | 23/9/0 | 0.694 · 70.22 · 0 · 0.985 |
| sf_rewind | 32 | 0.01262 | +8% | 23/5/4 | 0.698 · 70.89 · 0 · 0.988 |
| sf_sick_search | 32 | 0.01164 | **+0%** | 19/4/9 | 330 scoring |
| sf_pseudo | 32 | 0.01492 | **+28%** | **23/0/9** | 330 scoring |
| sf_sink | 32 | 0.01907 | +64% | 31/1/0 | 330 scoring |
| sf_always_search | **32** | **0.01623** | **+39%** | **30/2/0** | 330 scoring |

VBench cells are subject · IQ · Dyn · flicker. The three written
rows are provisional while 330 is R.

## vs caption Rolling Forcing (RF-hosted)

Median vs **caption rolling 0.01423**, and vs caption SF.

| Method | n | tail | vs RF host | W/L vs RF | vs caption SF | W/L vs SF |
|---|---:|---:|---:|---|---:|---|
| rolling_notta | 32 | 0.01423 | — | — | +22% | 23/9/0 |
| rf_rewind | 32 | 0.01505 | +6% | 16/9/7 | +29% | 24/8/0 |
| rf_sick_search | 32 | 0.01408 | **−1%** | 12/11/9 | +21% | 21/11/0 |
| rf_pseudo | 32 | 0.01534 | +8% | 9/3/20 | +32% | 24/8/0 |
| rf_sink | 32 | 0.02017 | +42% | 29/3/0 | +73% | 29/3/0 |
| rf_always_search | 32 | 0.01775 | +25% | 23/9/0 | +52% | 25/7/0 |

## Read

- **Always 32/32:** tail +39% vs caption SF (30/2). Pseudo is +28%
  (23/0/9). Hold-out is not inert. No Always call until VBench.
- **Caption RF is not the quality-better host.** Provisional
  VBench: subject 0.694 vs SF 0.700, IQ **70.22 vs 71.54 (−1.32,
  fails the −1 bar)**, Dyn 0. Stem “RF wins identity/IQ” does not
  copy. Tail +22% still does.
- **Rewind** letter holds on the written dims (IQ −0.65, subject
  −0.002) plus tail +8%. Wait for 330. Do not scale.
- **Sick** still a median tail tie. **Sink** still the large tail
  mover. **RF pseudo** 20/32 exact host — dead gate, same as stem.
- Caption SF baseline tail 0.01164 is below stem 0.0135; official
  SF identity/IQ are **above** stem (0.700 / 71.54 vs 0.665 / 69.65).

## AdaSteer N=8 — NO (not a crash)

`|δ|` ≈ 0.84. 8/8 mp4. vs caption notta first 8 (tail 0.01291):

| Arm | tail vs SF8 | IQ | Call |
|---|---|---:|---|
| ada_fixed | −22% | **42.67** | **NO** |
| ada_stream | +11% | **51.48** | **NO** |
| ada_resid | −49% | **17.75** (Dyn 1) | **NO** |

δ fits and wrecks imaging. Do not scale. Do not retune tonight.

## Do not

Replace stem talk VBench with this partial caption table.
HOLD Always / Pseudo / Sink from tails. Submit WAVE=2.
Relaunch AdaSteer N=32.
