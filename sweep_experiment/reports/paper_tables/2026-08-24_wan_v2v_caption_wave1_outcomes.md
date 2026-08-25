# Caption WAVE=1 outcomes (2026-08-24 22:45)

Series `v2v_panda_caption_32v`. Prompt = first-segment Panda
caption (`prompt_source=metadata_csv` on every harvested sidecar).
Do **not** mix with stem-prompt tables.

Official quality is VBench / VBench++ on the **full clip** after
**16310330**. That job is still waiting on `sf_pseudo` and
`sf_always_search`. Subject / IQ / dynamic degree / flickering stay
blank. Tails below are a generate diagnostic only.

## Jobs

| Job | Method | State | n |
|---|---|---|---|
| **16310318** | notta | COMPLETED 0:0 1h47 | **32/32** |
| **16310319** | rolling_notta | COMPLETED 0:0 27m | **32/32** |
| **16310320** | sf_rewind | COMPLETED 0:0 1h13 | **32/32** |
| **16310321** | sf_sick_search | COMPLETED 0:0 1h37 | **32/32** |
| **16310322** | sf_pseudo | COMPLETED 0:0 2h44 | **32/32** |
| **16310323** | sf_sink | generate 32/32 in harvest | **32/32** |
| **16310324** | sf_always_search | R ~2h55 | **29/32** |
| **16310325–329** | RF always / rewind / sick / pseudo / sink | generate 32/32 in harvest | **32/32** |
| **16310330** | VBench full clip | PD (Dependency) | — |
| **16314667–669** | AdaSteer 8v (first) | FAILED 2:0 inference_mode | closed |
| **16321558 / 560 / 562** | AdaSteer 8v (retry) | **FAILED 2:0** inplace IM cache | crash |
| **16321563** | AdaSteer VBench | CANCELLED | — |

## vs caption Self-Forcing (SF-hosted)

| Method | n | tail | vs SF | W/L/tie | Official VBench |
|---|---:|---:|---:|---|---|
| notta (SF) | 32 | **0.01164** | — | — | pending |
| rolling_notta (RF host) | 32 | 0.01423 | +22% | 23/9/0 | pending |
| sf_rewind | 32 | 0.01262 | +8% | 23/5/4 | pending |
| sf_sick_search | 32 | 0.01164 | **+0%** | 19/4/9 | pending |
| sf_pseudo | 32 | 0.01492 | **+28%** | **23/0/9** | pending |
| sf_sink | 32 | 0.01907 | +64% | 31/1/0 | pending |
| sf_always_search | **29** | 0.01591 | +36% | 27/2/0 | running |

`notta` “fire 28” in the harvest script is a false count (`last_sick`
on do-nothing chunks). Ignore it.

## vs caption Rolling Forcing (RF-hosted)

Median vs **caption rolling 0.01423**, and vs caption SF for the
same row.

| Method | n | tail | vs RF host | vs caption SF | W/L vs SF |
|---|---:|---:|---:|---:|---|
| rolling_notta | 32 | 0.01423 | — | +22% | 23/9/0 |
| rf_rewind | 32 | 0.01505 | +6% | +29% | 24/8/0 |
| rf_sick_search | 32 | 0.01408 | **−1%** | +21% | 21/11/0 |
| rf_pseudo | 32 | 0.01534 | +8% | +32% | 24/8/0 |
| rf_sink | 32 | 0.02017 | +42% | +73% | 29/3/0 |
| rf_always_search | 32 | 0.01775 | +25% | +52% | 25/7/0 |

RF-paired W/L was not in this paste.

## Read (generate only)

- **Sick on SF** is a median tie. 19 small wins do not move the
  typical video. Same story as stem (−1%).
- **Rewind** still a small typical plus (+8%, 23/5/4).
- **Sink** is still the large tail mover on both hosts (+64% SF /
  +42% vs RF). Official identity / flicker unknown until VBench.
- **Pseudo** is now 32/32: +28% vs caption SF, **23/0/9**.
  Stem was +37% / 25/2/5. Caption win set is all wins or ties,
  no losses. VBench still required.
- **Always-search** is 29/32, +36% / 27/2. Not a gate-vs-pick
  call until 32.
- **RF host** still beats caption SF (+22%). RF sick is a wash vs
  that host. RF always +25% vs rolling is the first caption look at
  “gate vs pick” on RF — VBench still required.
- Caption SF baseline is lower than stem SF (0.01164 vs 0.0135).

## AdaSteer — second crash, still not a null

Retry **16321558 / 560 / 562** FAILED 2:0 in ~2 m. 0 mp4.
`Inplace update to inference tensor outside InferenceMode` — KV
caches were allocated under the runner’s IM, then written after we
left IM. Fix: AdaSteer generate is not wrapped in IM; caches are
dropped and re-allocated; inputs cloned. Resubmit N=8 after pull.
Do not write “dead on Wan.” Do not submit N=32.

## Do not

Call HOLD/NO. Cite partials as N=32. Mix stem numbers into this
table. Submit WAVE=2. Relaunch AdaSteer N=32.
