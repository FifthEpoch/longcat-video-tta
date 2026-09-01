# Weekly Recap — Pseudo-future Search (Wan V2V)

**Period:** Monday 2026-08-25 → Tuesday 2026-09-01
**For:** week talk (canvas `week-recap-0901`)
**Stack:** Wan2.1-T2V-1.3B + Self-Forcing. Caption V2V, `metadata.csv`.
**Locks:** No I2V scale. No TTC / LoRA-at-test-time. Cite medians.
Dynamic Degree = **percent of clips**, not the median of 0/1.

Talk: open [week recap](/Users/macrohard/.cursor/projects/Users-macrohard-Desktop-longcat-video-tta/canvases/week-recap-0901.canvas.tsx) beside the chat.

---

## TL;DR

- **Pseudo-future Search is the cite method.** Opening hold-out of the last 3 prefix latents; fire k=4 if an extra seed beats do-nothing on that real B; motion+trust pick after. γ=0, k=4. Code `sf_pseudo`.
- **Caption 128 VBench is complete.** Pseudo tail **0.0157 ≈ Rolling 0.0158**. Dyn% **47.7% (61/128)** vs Self Forcing 32.8% (42) and Rolling 28.9% (37). IQ **72.38**. Subject **0.660** vs Rolling **0.685**. Always **50.8% (65)** / 0.661 / 72.19. Gate **90 fire / 38 skip** costs 4 Dyn clips.
- **Cost:** cite caption-128 generate wall / 96 new clips (first 32 hardlinked). Pseudo **294 s**, Always **354 s**, Rolling **47 s**, Self Forcing **108 s**. The gate is 17% cheaper than Always and still ~6.3× Rolling.
- **Pixels DONE (2026-09-01).** Paired 30 s leftover, n=128: SF **9.25**, Pseudo **9.22**, Always **9.21**, Rolling **7.98**. Search does not pay a reconstruction tax. Rolling does. Headline stays VBench + Dyn%.
- **Closed this week:** AdaSteer, Prefix-match, Pseudo-on-Rolling, lastmix / bpseudo / restep, intra-chunk, keep-picture (14 arms), CachedSearch, re-gate. Mid-chunk rewrite is an experiment paragraph.

---



## 1. Caption N=32 (official, 2026-08-25)

Cite Dyn as percent of clips. Median was 0 on almost every row and hid the story.


| Method       | Dyn%           | Tail   | Subject | IQ           | Call                 |
| ------------ | -------------- | ------ | ------- | ------------ | -------------------- |
| Self Forcing | 21.9% (7/32)   | 0.0116 | 0.700   | 71.54        | Baseline             |
| Rolling      | 18.8% (6)      | 0.0142 | 0.694   | 70.22        | Host; IQ vs SF fails |
| Pseudo       | **40.6% (13)** | 0.0149 | 0.701   | 71.66        | HOLD                 |
| Always       | **43.8% (14)** | 0.0162 | 0.687   | 71.16        | Ablation             |
| Prefix-match | 21.9% (7)      | −18%   | 0.746   | 70.54        | **NO** (freezes)     |
| AdaSteer N=8 | —              | —      | —       | 43 / 51 / 18 | **NO**               |


`rf_sink` subject 0.709 vs its host but IQ 70.15 (−1.39 vs SF). Not ours.

Tables: `2026-08-25_wan_v2v_caption_official_complete.md`,
`2026-08-25_wan_v2v_caption_dyn_percent.md`,
`2026-08-25_pseudo_future_search.md`.

---



## 2. Caption 128 (paper-size row)

- VBench 7/7. Do not remake videos.
- **Tail motion** is not VBench. After the real ~2 s prefix we invent
  ~30 s. Tail = median, over 128 clips, of mean |pixel change|
  between consecutive invented frames (range ~0–1). Higher = the
  continuation moves more (or morphs). Self Forcing 0.0119 is the
  quiet/freeze tail; Rolling and Pseudo are ~+33%. Dynamic Degree
  is VBench’s RAFT “is this clip alive?” count (we cite percent of
  clips, not the median of 0/1). They can disagree.

  |                      | Self Forcing | Rolling    | Pseudo     | Always         |
  | -------------------- | ------------ | ---------- | ---------- | -------------- |
  | tail motion (invented 30 s) | 0.0119 | **0.0158** | 0.0157     | 0.0168         |
  | Subject Consistency  | 0.666        | **0.685**  | 0.660      | 0.661          |
  | Background Consistency | 0.801      | **0.802**  | 0.792      | 0.790          |
  | Aesthetic Quality    | 0.499        | **0.529**  | 0.510      | 0.503          |
  | Imaging Quality      | 72.07        | 71.52      | **72.38**  | 72.19          |
  | Motion Smoothness    | **0.992**    | 0.991      | 0.991      | 0.990          |
  | Dynamic Degree       | 32.8% (42)   | 28.9% (37) | 47.7% (61) | **50.8% (65)** |
  | Temporal Flickering  | **0.987**    | 0.983      | 0.984      | 0.982          |
  | mean s / clip (n=128) | 108         | **47**     | **294**    | 354            |
  | PSNR                 | **9.25**     | 7.98       | 9.22       | 9.21           |
  | SSIM                 | **0.279**    | 0.250      | 0.268      | 0.266          |


Always − Pseudo = **+4 dynamic clips**. Subject/IQ match. Gate is the cost cut.
Mean seconds = generate wall / 96 new clips (`2026-09-01_wan_v2v_cite128_wall.md`).

Tables: `2026-08-31_wan_v2v_cite128_complete.md`,
`2026-08-31_wan_v2v_cite128_all_metrics.md`,
`2026-09-01_wan_v2v_cite128_pixel.md`.

---



## 3. Closed methods (same-wave + harvest)


| Attempt                                                    | N            | Call                                                  |
| ---------------------------------------------------------- | ------------ | ----------------------------------------------------- |
| AdaSteer (`ada_fixed` / `stream` / `resid`)                | 8            | **NO.** |δ|≈0.84. IQ 43 / 51 / 18.                    |
| Prefix-match / appear pick                                 | 32           | **NO.** Identity damper.                              |
| Pseudo on Rolling                                          | 32           | **NO.** Gate dead.                                    |
| lastmix / bpseudo / restep                                 | 8 (restep 5) | **NO.** Identity or subject 0.575.                    |
| Intra-chunk                                                | 8            | **NO.** Gated ≡ always. Subject 0.632.                |
| Keep-picture (nudge / next-seed / wiggle / latmot × SF+RF) | 8 × 14       | **NO.** All miss subject 0.68. RF IQ 66–67.           |
| CachedSearch                                               | 8            | **NO.** Tails match; wall **higher** (389 vs 360).    |
| Re-gate each chunk                                         | 8            | **NO.** Fire lives (6/5/6/7/8/6). No lift. +53% wall. |


Keep letter was subject ≥ 0.68 and IQ ≥ 70.5. Do not loosen.
Mid-chunk rewrite is closed. Harvest:
`2026-08-31_wan_v2v_keep_intra_closed.md`,
`2026-08-31_wan_v2v_pseudo_next8_harvest.md`,
`2026-08-31_wan_pseudo_improvements_tried.md`.

---



## 4. Success bar (locked 2026-08-30)

**Wanted:** about as good as Rolling, much cheaper than Always (354 s). Stretch: near Rolling (47 s). Ours = controller on frozen Self Forcing, not a host swap.

128 quality: tail tie, Dyn% + IQ win, subject loss. Cost still ~6.3× Rolling. Remaining paper is **cheapen** (search-early or prune k) or **Rolling window-exit** (`2026-08-30_wan_rf_intervene.md`). Not another seed search. Neighbors: Video-T1 (prune), TANGO critic without LoRA, TTC as the AdaSteer / Prefix-match exhibit. Do not build an agent.

---



## 5. Still open

- **FVD:** aligned 30 s tails only, `--force`. Do not score the full mp4.
- **LPIPS:** `import lpips` fails in `self_forcing`. Env gap.
- **Look:** 10 matched IDs × 4 methods staged (40 files, 777 MB) at `v2v_panda_caption_128v_examples`. Local dest `~/Desktop/caption128_compare`.

Queue empty. Do not remake 128 videos.

---



## 6. Do not launch

I2V-32 / I2V-200. TTC. LoRA-at-test-time. AdaSteer scale. Keep / intra / denoise remakes. γ or k retune on cite 128. More LongCat TTC.