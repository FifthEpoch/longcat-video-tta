# Caption Always 32/32 + AdaSteer N=8 NO (2026-08-25 02:34)

Series: `v2v_panda_caption_32v`, `v2v_panda_adasteer_8v`.
Prompt = `metadata_csv` on every harvested sidecar. Do **not** mix
stem-prompt VBench into this table.

Jobs: Always **16310324** COMPLETED 0:0 3h08. AdaSteer generate
**16326033–035** COMPLETED 0:0 18–21 min. AdaSteer VBench
**16326036** COMPLETED 0:0 20 min. Caption VBench **16310330**
still **R** (~1h22). `joined.json` already exists for caption
notta / rolling / rewind — treat those three as **provisional**
until 330 exits 0:0.

Cite **medians**. Official quality = full-clip VBench. Tails are
diagnostic. Locked bars vs caption SF: tail ↑, IQ ≥ SF−1.0,
subject ≥ SF−0.02.

## Caption generate (N=32, all 32/32)

| Method | tail | vs SF | W/L/tie | vs RF host | Official VBench (subj · IQ · Dyn · flick) |
|---|---:|---:|---|---:|---|
| notta (SF) | **0.01164** | — | — | — | **0.700 · 71.54 · 0 · 0.989** (provisional) |
| rolling_notta | 0.01423 | +22% | 23/9/0 | — | **0.694 · 70.22 · 0 · 0.985** (provisional) |
| sf_rewind | 0.01262 | +8% | 23/5/4 | — | **0.698 · 70.89 · 0 · 0.988** (provisional) |
| sf_sick_search | 0.01164 | +0% | 19/4/9 | — | 330 still scoring |
| sf_pseudo | 0.01492 | +28% | 23/0/9 | — | 330 still scoring |
| sf_sink | 0.01907 | +64% | 31/1/0 | — | 330 still scoring |
| **sf_always_search** | **0.01623** | **+39%** | **30/2/0** | — | 330 still scoring |
| rf_rewind | 0.01505 | +29% | 24/8/0 | +6% (16/9/7) | 330 still scoring |
| rf_sick_search | 0.01408 | +21% | 21/11/0 | **−1%** (12/11/9) | 330 still scoring |
| rf_pseudo | 0.01534 | +32% | 24/8/0 | +8% (9/3/20) | 330 still scoring |
| rf_sink | 0.02017 | +73% | 29/3/0 | +42% (29/3/0) | 330 still scoring |
| rf_always_search | 0.01775 | +52% | 25/7/0 | +25% (23/9/0) | 330 still scoring |

RF-vs-RF W/L from `harvest_caption_wave1.py` 02:34.

## Provisional caption VBench vs caption SF

| Method | subject | IQ | Dyn | flicker | Δ subject | Δ IQ |
|---:|---:|---:|---:|---:|---:|---:|
| notta (SF) | 0.700 | 71.54 | 0 | 0.989 | — | — |
| rolling_notta | 0.694 | 70.22 | 0 | 0.985 | −0.006 | **−1.32** |
| sf_rewind | 0.698 | 70.89 | 0 | 0.988 | −0.002 | −0.65 |

Stem-prompt SF was 0.665 / 69.65. Real captions **raised** the
SF baseline (identity + IQ). Caption RF is no longer the
quality-better host: IQ fails the −1.0 bar vs caption SF, subject
slightly down, Dyn still 0. Tail +22% remains. Do not cite stem
“RF subject 0.702 / IQ 70.44 beats SF” as the caption story.

Rewind letter vs caption SF: tail +8%, IQ −0.65 (hold), subject
−0.002 (hold), Dyn 0. Not a scale-up. Wait for the rest of 330
before a family call.

## Always vs Pseudo (generate only)

| | tail | vs SF | W/L/tie |
|---|---:|---:|---|
| sf_pseudo | 0.01492 | +28% | 23/0/9 |
| sf_always_search | 0.01623 | +39% | 30/2/0 |

Hold-out is **not inert** (Always ≠ Pseudo). Always converts most
of Pseudo’s 9 exact-SF ties into tail wins and takes 2 losses.
That is a gate-vs-pick split on the handcrafted tail only. **No
call** until Always VBench lands. Do not mix stem always-search.

## AdaSteer N=8 — measured NO

Cite vs **caption notta first 8** tail **0.01291**. Sidecars
`metadata_csv`. `|δ|` ≈ 0.84–0.95 (fit ran). VBench **16326036**
full clip on the 8 mp4s.

| Arm | tail | vs SF8 | W/L | subject | IQ | Dyn | flicker | Call |
|---|---:|---:|---|---:|---:|---:|---:|---|
| ada_fixed | 0.01004 | **−22%** | 5/3/0 | 0.632 | **42.67** | 0.50 | 0.988 | **NO** |
| ada_stream | 0.01430 | +11% | 5/3/0 | 0.618 | **51.48** | 0 | 0.985 | **NO** |
| ada_resid | 0.00660 | **−49%** | 2/6/0 | 0.675 | **17.75** | **1.00** | 0.993 | **NO** |

Locked IQ bar is SF−1 ≈ 70 if compared to caption SF-32 (71.54).
Even against a generous first-8 guess, IQ 18–51 is a collapse, not
a −0.6 dip. Stream’s +11% tail does not save the letter. Resid
Dyn 1 + IQ 17 is a broken picture, not recovered motion.

This is a **Wan measurement**, not “dead because LongCat was
flat.” δ fits and wrecks imaging. **Do not submit N=32.** Do not
retune S/LR tonight. Closed as a confirmation null-with-damage.

## Do not

Replace the stem-prompt talk table with this partial caption
VBench. Call HOLD/NO on Always / Pseudo / Sink from tails.
Submit WAVE=2. Scale AdaSteer. Mix stem and caption in one
official row.
