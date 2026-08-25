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
| **16310322** | sf_pseudo | left squeue 00:37 — harvest | was 27/32 |
| **16310323** | sf_sink | generate 32/32 in harvest | **32/32** |
| **16310324** | sf_always_search | R 2h51 gh119 | was 9/32 |
| **16310325–329** | RF always / rewind / sick / pseudo / sink | generate 32/32 in harvest | **32/32** |
| **16310330** | VBench full clip | PD (Dependency) | — |
| **16314667–669** | AdaSteer 8v | **FAILED 2:0** (~3m, 0 mp4) | crash |
| **16314670** | AdaSteer VBench | CANCELLED (afterok) | — |

## vs caption Self-Forcing (SF-hosted)

| Method | n | tail | vs SF | W/L/tie | Official VBench |
|---|---:|---:|---:|---|---|
| notta (SF) | 32 | **0.01164** | — | — | pending |
| rolling_notta (RF host) | 32 | 0.01423 | +22% | 23/9/0 | pending |
| sf_rewind | 32 | 0.01262 | +8% | 23/5/4 | pending |
| sf_sick_search | 32 | 0.01164 | **+0%** | 19/4/9 | pending |
| sf_pseudo | **27** | 0.01494 | +28% | 19/0/8 | running |
| sf_sink | 32 | 0.01907 | +64% | 31/1/0 | pending |
| sf_always_search | **9** | 0.01494 | +22% | 9/0/0 | running |

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
- **Pseudo** is incomplete (27). Partial +28% / 19/0/8. Do not
  compare to stem +37% until 32.
- **Always-search** is n=9. Same partial median as the 27-video
  pseudo (0.01494). Not a gate-vs-pick call.
- **RF host** still beats caption SF (+22%). RF sick is a wash vs
  that host. RF always +25% vs rolling is the first caption look at
  “gate vs pick” on RF — VBench still required.
- Caption SF baseline is lower than stem SF (0.01164 vs 0.0135).

## AdaSteer — crashed, not a null

All three arms **FAILED 2:0** in ~3 minutes. 0 mp4. 9 json = 8
per-video error stubs + `summary.json` (`n_ok != n` → exit 2).
Captions loaded (`metadata_csv`). VBench 670 cancelled. This is a
hook / fit crash, **not** “AdaSteer is dead on Wan.” Do not scale.
Do not resubmit until the first error json is read.

## Do not

Call HOLD/NO. Cite partials as N=32. Mix stem numbers into this
table. Submit WAVE=2. Relaunch AdaSteer N=32.
