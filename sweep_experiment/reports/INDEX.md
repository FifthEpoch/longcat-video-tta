# AdaSteer Experiment Index

**Purpose:** Single source of truth for "what experiments exist, where their
results live, what is paper-quality vs discovery, and what remains to be
run." Every agent / human working on this paper should read this first.

**Update rule:** Append a row whenever a new experiment series completes,
update the Status / Findings columns when re-merged. NEVER delete rows
even if results are superseded — mark them `superseded` and keep them
for audit trail.

**Owners:** Wenchen (PI) and any active agent. Last updated: 2026-08-23.

---

## Headline 1000v paper-grade experiments (the 4 we'd publish today)

| Series | Dataset | N | Frames | Methods | Status | Cluster path | Paper table | Key finding |
|---|---|---|---|---|---|---|---|---|
| `panda_1000v_standard` | Panda-70M | 999 | 28 | NOTTA, ADA, LORA_R8_TTA, TL_BARE_R2, TL_TIED_R2 | DONE + VBench backfilled (2026-06-05) | `sweep_experiment/results/panda_1000v_standard/`, `delta_experiment/results/tinylora_panda_1000v_standard/` | Table 1 of [`paper_tables/2026-06-08_headline_1000v.md`](paper_tables/2026-06-08_headline_1000v.md) | AdaSteer ≈ NoTTA on every metric. LoRA shifts distribution (Aes↑, IQ↓). |
| `ucf101_932v_standard` | UCF-101 | 932 | 28 | NOTTA, ADA, LORA_R8_TTA, TL_BARE_R2, TL_TIED_R2 | DONE + VBench backfilled (2026-06-05) | `sweep_experiment/results/ucf101_932v_standard/`, `delta_experiment/results/tinylora_ucf101_932v_standard/` | Table 2 | Same saturation pattern. 932v not 1000v because some chunks failed. |
| `ucf101_932v_retrieval` | UCF-101 | 932 | 28 | K5_SIM, K5_RAND, K10_SIM, K10_RAND | DONE + VBench backfilled (2026-06-05) | `sweep_experiment/results/ucf101_932v_retrieval/` | Table 2 | All 4 retrieval variants ≈ NOTTA. UCF class-block layout means SIM and RAND retrieve same-class neighbours. NOT a useful retrieval testbed. |
| `panda_1000v_retrieval` | Panda-70M | 999 | 28 | K5_SIM, K5_RAND, K10_SIM, K10_RAND | DONE + 7-dim VBench 2026-07-05 (pool `panda_2048_480p`) | `sweep_experiment/results/panda_1000v_retrieval/` | [`paper_tables/2026-07-05_panda_1000v_retrieval.md`](paper_tables/2026-07-05_panda_1000v_retrieval.md) | SIM≈RAND; PSNR/FVD ≤ ADA; LoRA-like Aes↑ IQ↓ (VB total≈0.778 vs ADA 0.773). |
| `panda_longctx_1000v` | Panda-70M | 999 | 76 | NOTTA, ADA_S10, LORA_R8, PANDA_TL_LAST24 | DONE + VBench backfilled (2026-06-05) | `sweep_experiment/results/panda_longctx_1000v/`, `delta_experiment/results/tinylora_longctx_1000v/` | Table 3 | Saturated at PSNR ~12.77. Subj drops 0.907→0.774 vs std (drift effect). AdaSteer preserves Subj (0.775); LoRA worsens it (0.757). |
| `ucf101_683v_longhorizon` | UCF-101 | 683 | 76 | NOTTA, ADA, LORA_R8_TTA | DONE + VBench backfilled (2026-06-08) | `sweep_experiment/results/ucf101_683v_longhorizon/` | Table 4 | All within 0.02 PSNR. LoRA Aes↑ (0.394→0.433), IQ↓ (0.450→0.430). 683 not 1000 because original chunked submit hit class-name skip. |

---

## Config-routing pilot (N=200 OOD-stratified) — current headline thread

| Series | Dataset | N | Methods | Status | Cluster path | Paper table | Key finding |
|---|---|---|---|---|---|---|---|
| `panda_ood_budget_pilot` | Panda-70M | 200 (40/quintile × 5 OOD) | 12 AdaSteer configs (S{2,5,10,20} × LR{1e-3,5e-3,1e-2}); NOTTA joined by id from `panda_1000v_standard/NOTTA` | DONE + 7-dim VBench backfilled | `sweep_experiment/results/panda_ood_budget_pilot/` | [`2026-07-15_pilot_config_vs_notta_routing.md`](paper_tables/2026-07-15_pilot_config_vs_notta_routing.md), [`2026-07-09_deploy_psnr_router.md`](paper_tables/2026-07-09_deploy_psnr_router.md), [`2026-07-09_deploy_router_aux_metrics.md`](paper_tables/2026-07-09_deploy_router_aux_metrics.md) | Same-200-video: oracle PSNR routing **+0.95 dB vs no-TTA**, **+0.75 vs best fixed config** (also ↑SSIM ↓LPIPS); no single config wins across OOD quintiles. Learned 9-d OOF router realizes ≈7.2% (PSNR) / ≈20.8% (VBench) of oracle gap. VBench/PSNR objectives decouple. |
| `panda_ood_budget_1000v_preview` | Panda segment pool | 1000 | same 12-config grid (+ NOTTA in flight) | **12 configs DONE + merged 2026-07-19** (10 chunks each, mp4s saved). NOTTA submitted 2026-07-19 (jobs 14319937–946). VBench backfill + routers pending. | `sweep_experiment/results/panda_ood_budget_1000v_preview/` | [`2026-07-19_budget_grid_1000v_preview.md`](paper_tables/2026-07-19_budget_grid_1000v_preview.md) | Population metrics FLAT across all 12 configs (PSNR 19.37–19.49, spread 0.11 dB); only train time scales (15→128 s). Aggressive S20_LR1e2 worst → mild over-adaptation. Motivates per-video router + 13th skip-TTA candidate. |
| `panda_ood_budget_1000v_preview` (router matrix) | Panda segment pool | 898 paired | 7 feature blocks (A/B/C + subsets) × {12,13 actions} × {PSNR,VBench}; NOTTA VBench backfilled | DONE 2026-07-21 (offline OOF ridge) | same series; features `per_video_analysis/2026-07-12` | [`2026-07-21_router_full_matrix_1000v.md`](paper_tables/2026-07-21_router_full_matrix_1000v.md), [`2026-07-21_router_1000v_feature_model_suite.md`](paper_tables/2026-07-21_router_1000v_feature_model_suite.md) | **No deployable router beats best fixed config or no-TTA** on either metric, any block. PSNR all cells −0.004…−0.018; best PSNR config = S2 (least-adaptive). VBench 12-action ~flat (cap −0.5%), 13-action uniformly −0.13 (skip-averse). Aug-oracle +1.03 over no-TTA but is max-of-fat-tailed-noise ⇒ un-routable (signal ceiling, not tuning). Open: verify NO-TTA VBench fat tail = genuine variance-reduction vs coverage artifact. |

---

## Long-horizon drift + test-time control (2026-08 — current headline thread)

Native LongCat window (13-cond/80-gen), true autoregressive rollout (feed the
model's own generated tail back). Judge by GT-free per-chunk drift + per-video
paired sign-flip test (`compare_drift_paired.py`); GT pixel metrics span only
~1-2 chunks (source clips short) so they are gating, not paper numbers. All N=8
here is a GATING sample.

| Series | N | Chunks / horizon | Method | Status | Cluster path | Key finding |
|---|---|---|---|---|---|---|
| `longhorizon_sweep_notta_native_12ch` | 8 | 12 / ~60 s | NOTTA baseline | DONE 2026-08-09 | `sweep_experiment/results/longhorizon_sweep_notta_native_12ch/` | Native drift COMPOUNDS with horizon: sharpness +48%, motion +45%, contrast −16% (vs +28/+8/+3 at 6ch/30 s). Headroom is real at ~1 min. |
| `longhorizon_sweep_delta_stream_native_12ch` | 8 | 12 / ~60 s | AdaSteer δ re-fit each chunk on generated window | DONE 2026-08-09 | same root | NULL under paired test (p≥0.26); population "flattening" was cancellation (raised per-video volatility). |
| `longhorizon_sweep_delta_stream_clean_native_12ch` | 8 | 12 / ~60 s | δ re-fit toward clean chunk-0 latents | DONE 2026-08-10 | same root | NULL (p≥0.53); fixes saturation, overshoots contrast fade. 3rd delta recipe to fail; delta line CLOSED (+ ramp contraindicated, routing = noise ceiling). |
| `longhorizon_sweep_bestof_k4_native_12ch` | 8 | 12 / ~60 s | best-of-4 GT-free drift verifier (cand0 = NOTTA seed) | DONE 2026-08-11 | `sweep_experiment/results/longhorizon_sweep_bestof_k4_native_12ch/` | **FIRST credible positive.** Verifier picks non-NOTTA 75%; on 11 GT chunks chosen beats RANDOM by **+0.833 dB PSNR** (81% of by-PSNR oracle), −0.032 LPIPS — passes the credibility gate routing failed. Per-signal oracle capture: sharpness 96%, motion 76%, contrast 29%, color 10%. BUT end-to-end paired |drift| vs NOTTA not yet significant at N=8 (sharpness/motion lean right, contrast wrong). Worth SCALING. |

---

## Wan 1.3B I2V continuation (2026-08 — discovery only; do not scale)

LongCat 13.6B stays as the saturated-large-model audit. The I2V-32
30 s series is a **discovery / stress** run on our protocol. It is
**not** the field long-horizon table (that table is T2V, 128 MovieGen,
VBench-Long). **Do not scale I2V-32 or I2V-200.** **Do not add TTC.**
Stop lock: [`paper_tables/2026-08-18_wan_protocol_stop.md`](paper_tables/2026-08-18_wan_protocol_stop.md).

| Series | N | Horizon | Method | Status | Cluster path | Key finding |
|---|---|---|---|---|---|---|
| `i2v_notta_smoke` | 2 | 5 s (85 px) | NOTTA | **DONE 2026-08-16** job 15880611 | `wan_experiment/results/i2v_notta_smoke/h5s_shard0/` | First working generate. n_ok=2, mp4s 5.9/3.9 MB, 8–12 s/clip. Frame-0 MAE vs cond 5.56 / 3.71 (I2V, not noise). Autograd-off fixed the 138 GB OOM. |
| `i2v_notta_16v` | 16 | 5 s + 30 s | NOTTA | **DONE 2026-08-16** n_ok=16/16 both | `wan_experiment/results/i2v_notta_16v/{h5s,h30s}_shard0/` | 5 s mean 9.61 s/clip; 30 s mean 38.32 s/clip. Drift (skip f0, 1 s windows): 5 s median sharp +11% / motion −14%; **30 s median sharp +167% / motion −60%**. Headroom real. Table: [`2026-08-17_wan_i2v_notta16_drift.md`](paper_tables/2026-08-17_wan_i2v_notta16_drift.md). |
| `i2v_chunked_smoke` | 2 | 30 s (5×24 lat) | NOTTA, always-BoN k=4 | **DONE 2026-08-17** jobs 15883525/526 | `wan_experiment/results/i2v_chunked_smoke/` | n_ok=2, search alive (5/8 chunks left cand0). NOTTA scores worsen later. **Not a paired quality result** — chunk 0 cand0 already differed (unseeded add_noise). Table: [`2026-08-17_wan_i2v_chunked_bon_smoke.md`](paper_tables/2026-08-17_wan_i2v_chunked_bon_smoke.md). |
| `i2v_bon_16v` | 16 | 30 s (5×24 lat) | NOTTA, always-BoN k=4, gated-BoN k=4 (gate=2.0) | **DONE 2026-08-17** jobs 15884598/599/600 | `wan_experiment/results/i2v_bon_16v/` | Seed match 16/16. Last-chunk: NOTTA 4.43 / always 3.23 / gated 3.38. Search works. Gated vs always: mean +0.152, median −0.13, 6/16 better-or-tie. Efficiency paper, not a quality win. Tables: [`2026-08-17_wan_i2v_bon16.md`](paper_tables/2026-08-17_wan_i2v_bon16.md), [`2026-08-17_wan_i2v_bon16_lastchunk.md`](paper_tables/2026-08-17_wan_i2v_bon16_lastchunk.md). |
| `i2v_bon_32v_hybrid` | 32 | 30 s (5×24 lat) | NOTTA, always-BoN k=4, gated-BoN hybrid (ch1>0.8 / late>2.0 / Δ>0.5∧prev>0.5) | **DONE 2026-08-17** n_ok=32/32. Official VBench **DONE** jobs 15959601+15984561 | `wan_experiment/results/i2v_bon_32v_hybrid/` | Cite medians, not means (video 26 = 85.6). Median last-chunk NOTTA 3.68 / always 2.97 / gated 3.04. gated−always −0.041 mean / 0 median, 19/32 better-or-tie. 33% cheaper (173 vs 258 s). First-16 pairing held; hybrid flipped T=2.0 gated−always +0.15 → −0.12. Handcrafted last-chunk looked like search helping (med 3.68 / 2.97 / 3.04). Official last5 VBench does **not**: do-nothing wins IQ (68.2 vs 66.4) and background; `dynamic_degree` median 0 for all three; gated Aes 0.522 vs always 0.548. Verifier anti-aligned with IQ (ρ +0.23 to +0.33). Efficiency-only, and weaker than the handcrafted line. Tables: [`2026-08-17_wan_i2v_bon32_hybrid.md`](paper_tables/2026-08-17_wan_i2v_bon32_hybrid.md), VBench [`2026-08-18_wan_i2v_bon32_vbench_read.md`](paper_tables/2026-08-18_wan_i2v_bon32_vbench_read.md). |
| `i2v_bon_32v_sticky` | 32 | 30 s (5×24 lat) | gated-BoN hybrid + stay-on (reuse hybrid NOTTA / always-search) | **DONE 2026-08-18** n_ok=32/32 | `wan_experiment/results/i2v_bon_32v_sticky/` | 03/24 caught (exact ties). 06/07 still skipped early. 21/32 exact ties with always-search. Wall 256 vs 258 s — spent the 33% savings. Erased hybrid’s unique wins on 11 and 16. Not a quality win. Hybrid stays the efficiency method. Table: [`2026-08-18_wan_i2v_bon32_sticky.md`](paper_tables/2026-08-18_wan_i2v_bon32_sticky.md). 11/16 diagnosis: [`2026-08-18_wan_i2v_11_16_diagnosis.md`](paper_tables/2026-08-18_wan_i2v_11_16_diagnosis.md). |
| `i2v_bon_32v_sick` | 32 | 30 s (5×24 lat) | gated-BoN search-while-sick (sticky + off if recovered>0.5 or outgoing<1.0) | **DONE 2026-08-18** job 15959146 n_ok=32/32 | `wan_experiment/results/i2v_bon_32v_sick/` | Checklist pass on the handcrafted score. Median 2.764 vs always 2.966 / hybrid 3.036. 11=1.830 (beat hybrid 2.16), 16=2.656 exact hybrid, 24 exact always, 03 near (1.755 vs 1.57), 06/07/30 saved, wall 204 s. 9/14/9 vs always — not a strict quality win. Table: [`2026-08-18_wan_i2v_bon32_sick.md`](paper_tables/2026-08-18_wan_i2v_bon32_sick.md). |

Timing: [`paper_tables/2026-08-16_wan_i2v_smoke.md`](paper_tables/2026-08-16_wan_i2v_smoke.md), [`paper_tables/2026-08-16_wan_i2v_notta16.md`](paper_tables/2026-08-16_wan_i2v_notta16.md)

---

## Missing / not-yet-run experiments (paper-blocking or paper-relevant)

| Series | Why it's needed | Cluster status | Decision |
|---|---|---|---|
| `panda_1000v_retrieval` (K5/K10 × SIM/RAND) | UCF retrieval is uninformative due to class-block layout. Panda hash-ordered pool would give a clean retrieval signal. | **DONE** 2026-07-05 (40 jobs, pool `panda_2048_480p`, 999v merged). | **CLOSED** — SIM≈RAND; no PSNR/FVD win vs ADA. See [`paper_tables/2026-07-05_panda_1000v_retrieval.md`](paper_tables/2026-07-05_panda_1000v_retrieval.md). |
| 200v "gain disappears" comparison | Show research partner that small-N gains compress at scale. | Existing 26-100v discovery runs available; no actual N=200 series. | Skip or use 100v `panda_cover_candidates` as proxy. |
| Larger Panda retrieval pool (25K segments) | Original ambition: 25K segments from full Panda metadata for richer retrieval. | Phase 2A: 3K-segment pool built. Phase 2B: full-metadata download started but never completed → 25K. | Decide after Panda 1000v retrieval result. |
| `i2v_bon_32v_hybrid` official VBench | Paper-blocking outcome scorecard. **Cite the full clip** (comparable VBench++). last5 is diagnostic only. | **DONE** 15959601+15984561. full + last5, 32 paired. | Full-clip **tie** (Aes 0.587/0.593/0.591, IQ 71.24/71.28/71.19, dynamic median 0). last5 IQ drop is not the paper number. Read: [`2026-08-18_wan_i2v_bon32_vbench_read.md`](paper_tables/2026-08-18_wan_i2v_bon32_vbench_read.md). **Discovery protocol only — not a standard long-horizon result.** |
| `i2v_bon_32v_hybrid` VBench 5 s windows | Trend: VBench++ on 0–5 … 25–30 of the same 32 mp4s. Full clip stays official. | **DONE** job **16009916**. 18/18 `joined.json`. | All 7 dims: [`2026-08-19_wan_i2v_bon32_vbench_alldims.md`](paper_tables/2026-08-19_wan_i2v_bon32_vbench_alldims.md). Per-dim source: [`2026-08-19_wan_i2v_bon32_vbench_trend.md`](paper_tables/2026-08-19_wan_i2v_bon32_vbench_trend.md). Aes 0.651→0.538, IQ 72.9→68.1 (do-nothing). Search does not reverse it; tail IQ favors do-nothing. Dynamic median 0 in every window. Read: [`2026-08-19_wan_i2v_vbench_windows_read.md`](paper_tables/2026-08-19_wan_i2v_vbench_windows_read.md). |
| `i2v_notta_16v` VBench 5 s vs 30 s + first16/last16 | Same windows as the handpicked drift table, on official dims. 16-frame clips are diagnostic. | **DONE** job **16010032**. 8/8 `joined.json`. | **Cite entire clips:** [`2026-08-19_wan_i2v_notta16_vbench_fullclip.md`](paper_tables/2026-08-19_wan_i2v_notta16_vbench_fullclip.md) — 5 s full vs 30 s full, subject 0.932→0.842. 16-frame Δrel does **not** copy sharp +167% / motion −60%. Table: [`2026-08-19_wan_i2v_notta16_vbench_headtail.md`](paper_tables/2026-08-19_wan_i2v_notta16_vbench_headtail.md). |
| I2V-32 / I2V-200 30 s scale-up | Larger N on the *current* stills would tighten our error bars only. | **NOT SUBMITTED. Do not submit.** | **CLOSED 2026-08-18.** Task/N/suite are not the field recipe (I2V-from-still, N=32, `custom_input`). Stop: [`2026-08-18_wan_protocol_stop.md`](paper_tables/2026-08-18_wan_protocol_stop.md). |
| VBench-I2V protocol vs our scoring | Papers generate with image+caption and report `i2v_subject` / `i2v_background` / `camera_motion` + quality 7. | Generation already uses `i2v-bench-info.json` captions. Scoring was quality-7 `custom_input` only. | Not a new generate. Optional: score I2V dims on existing mp4s. Memo: [`2026-08-19_vbench_i2v_what_papers_do.md`](paper_tables/2026-08-19_vbench_i2v_what_papers_do.md). **Do not scale I2V-32.** |
| `t2v_bon_128v_vbenchlong` | Optional T2V *comparison* to Relax Forcing / SF++ / FreqForcing so gating can sit on a standard bench. **Not a task lock.** | **SUBMIT-READY 2026-08-18.** Runner + 4-shard submit landed. Not yet launched on cluster. | User asked to run it anyway. Smoke first (`SMOKE=1`), then 128. Spec: [`2026-08-18_wan_t2v_vbenchlong_128_spec.md`](paper_tables/2026-08-18_wan_t2v_vbenchlong_128_spec.md). |
| V2V prefix-continuation (real video history → 30 s AR) | Closer match to the claim than T2V. Condition on 1–2 s of real video (Panda / UCF / Kinetics), generate 30 s, VBench on the full clip. | **SMOKE+PROBE DONE 2026-08-20.** Jobs **16069897** (7m) / **16069898** (5m), both 0:0. | Probe: shift/CFG are no-ops (motion 0.01626 on all 9 cells). Table: [`2026-08-20_wan_v2v_probe.md`](paper_tables/2026-08-20_wan_v2v_probe.md). |
| `v2v_panda_bakeoff_8v` | notta, seed_bon, motion_bon, backtrack. **No shift_search** (probe dead). | **DONE 2026-08-20.** Generate 16092846–849 + VBench `joined.json` n=8. | N=8 seed_bon +35% / Dyn 0→0.5 **superseded** by N=32 fail. Keep as discovery. Table: [`2026-08-20_wan_v2v_bakeoff_8v_vbench.md`](paper_tables/2026-08-20_wan_v2v_bakeoff_8v_vbench.md). |
| `v2v_panda_confirm_32v` | notta vs seed_bon only. N=32. Same V2V protocol. | **DONE 2026-08-21.** Generate + VBench 16122823. | **seed_bon FAIL.** Also: prompts were filename stems (`panda 0013`), not scene captions. [`stem prompt`](paper_tables/2026-08-24_wan_v2v_panda_stem_prompt.md). |
| `v2v_panda_tricks_8v` | hinge_bon, late_bon, hist_drop, good_backtrack, cached_bon, sink. Same 8. | **GENERATE DONE.** VBench **16122824** hist_drop+hinge DONE. | Both **PASS N=8 bars** (IQ −0.15 / +0.42, Dyn 0.50). Same family as seed_bon-8. **Do not scale to 32.** |
| `v2v_panda_quiet_32v` | quiet_bon (search iff prefix_motion<0.018). | **DONE 2026-08-21** job **16124386** 2h29 0:0. n=32. | Tail **0.01089 vs notta 0.01353 (−19%)**. Gate hits some hots (exact notta). Still loses. No VBench. |
| `v2v_panda_tail_8v` | tail_hist k=1 last-3 replay. | **DONE 2026-08-21** job **16124387** 15m 0:0. n=8. | **+0.8% vs notta**, 2/8 vs hist_drop. Short history is not the hist_drop win. Unsticks 0002; erases 0000/0006/0007 lifts. |
| `v2v_panda_lineage_8v` | live_bon, live_hist, longlive_notta, longlive_sink, longlive_prefix_sink, longlive_live_bon, rolling_notta. Same 8. | **DONE 2026-08-21.** All 808–816 COMPLETED 0:0. 8/8 + VBench. | live_* N=8 +37% killed at 32. longlive_sink = longlive_notta. prefix_sink FAIL IQ. rolling_notta best host N=8 (do not scale tonight). Table: [`2026-08-21_wan_v2v_lineage_ideas_done.md`](paper_tables/2026-08-21_wan_v2v_lineage_ideas_done.md). |
| Sampling-space idea screen (9 TTA/TTC proposals) | Which interventions are actually in our V2V lock. | Analysis only. | Keep #1 pseudo-future gate, #5 appearance≠motion anchor, #3 U_t after ε probe. Drop weight TTA and late-horizon δ. Memo: [`2026-08-21_wan_v2v_sampling_ideas.md`](paper_tables/2026-08-21_wan_v2v_sampling_ideas.md). |
| `v2v_panda_ideas_8v` | appear_bon, live_appear, pseudo_gate, pseudo_appear, noise_probe, noise_bon. Same 8 as bake-off. | **DONE 2026-08-21.** 125–131 COMPLETED 0:0. 8/8 + VBench. | N=8 PROMOTE trap. noise_probe = notta. appear = noise_bon = pseudo_appear. **Do not scale.** Table: [`2026-08-21_wan_v2v_lineage_ideas_done.md`](paper_tables/2026-08-21_wan_v2v_lineage_ideas_done.md). |
| `v2v_panda_live_32v` | live_bon only. N=32. Same prefix set as confirm_32v. live_min=0.012. | **DONE 2026-08-21.** **16147007** 1h59 0:0 32 mp4; VBench **16147008** 24m. | **NO.** Searches bit-match seed_bon; 4/6 live clips on 0020–31 lost. Skip half works. Script N=32 was summary stubs. Verdict: [`2026-08-21_wan_v2v_live32_verdict.md`](paper_tables/2026-08-21_wan_v2v_live32_verdict.md). |
| `v2v_panda_forward_32v` | rolling_notta + appear_bon. N=32. Reuse confirm notta. | **DONE 2026-08-22.** 16179112 27m, 16179113 3h09, VBench 16179114. 32/32 + VBench. | **rolling YES** on locked bars (tail +31% / 21/11 / IQ+subj up; Dyn 0). **appear NO** (mean −2%, 15/17, 12/32=seed, subj +0.065). Table: [`2026-08-22_wan_v2v_forward32_verdict.md`](paper_tables/2026-08-22_wan_v2v_forward32_verdict.md). |
| `v2v_panda_rolling_128v` | SF notta + rolling_notta. N=128. Same prefix set as N=32. | **OFFICIAL 7-DIM DONE 2026-08-23.** **16259396** 0:0 + join-only n=128 all dims. | Locked bars PASS. Flicker 0.982 vs SF 0.986. Dyn first32 med 0 / 128 med 1. Host, not ours. [`vbench7`](paper_tables/2026-08-23_wan_v2v_rolling128_vbench7_read.md). |
| RF-sick rewind (Family A) | Controller on RF: rewind a window if it collapsed. Offline chunk-trace first. | **SPEC READY.** Login: `python3 -u wan_experiment/scripts/resim_v2v_rf_chunk_trace.py --only n128`. No GPU until GO. | Paper baseline stays SF notta. Ablation zero if GPU = rolling_notta. Spec: [`2026-08-23_wan_v2v_rf_sick_rewind_spec.md`](paper_tables/2026-08-23_wan_v2v_rf_sick_rewind_spec.md). |
| `v2v_panda_family_32v` | Family A/B/C/D on RF: `rf_rewind`, `rf_sick_search` k=4, `rf_pseudo` k=4, `rf_sink`. **N=32**. | **DONE 2026-08-24.** 32/32 + VBench 7/7. Dual wave 007–011 / 080–087 0:0. | Analyzer PROMOTE vs SF is wrong. vs RF: rewind/sick HOLD, pseudo NO, sink HOLD no-scale. [`verdict`](paper_tables/2026-08-24_wan_v2v_family32_verdict.md). |
| `v2v_panda_sf_family_32v` | Same four widgets on **SF chunked**: `sf_rewind`, `sf_sick_search` k=4, `sf_pseudo` k=4, `sf_sink`. **N=32**. | **DONE 2026-08-24.** 32/32 + VBench. 878 fail/resume 992; 879–881; VBench **16268053**. | **pseudo HOLD** +37% vs SF, 25/2, Dyn 0.5, beats RF median. **rewind HOLD** +6%. **sink HOLD no-scale** +72% / subject on the line / flicker 0.977. **sick NO**. [`verdict`](paper_tables/2026-08-24_wan_v2v_sf_family32_verdict.md). |
| `v2v_panda_sf_always_32v` | `sf_always_search` k=4. Same pick as pseudo, no prefix gate. **N=32**. | **IN FLIGHT 2026-08-24.** **16288113** R 2h40, mp4=26/32. | Splits gate vs pick. k=4. [`k`](paper_tables/2026-08-24_wan_v2v_always_search_k.md). |
| `v2v_panda_rf_always_32v` | `rf_always_search` k=4 on RF rolling. Same pick as rf_sick/rf_pseudo, no gate. **N=32**. | **GENERATE DONE 2026-08-24.** **16288114** COMPLETED 0:0 53m, mp4=32/32. VBench **16288115** waits on 113. | Host twin. Cite vs `rolling_notta`. Do not call until VBench + SF twin. Stem prompts. |
| `v2v_panda_caption_32v` | Caption replay WAVE=1: notta, rolling, SF/RF family, both always-search. **N=32**. | **02:34 2026-08-25.** Generate **32/32** all arms. Always tail **+39%** 30/2. VBench **16310330** R; notta/rolling/rewind written (SF 0.700/71.54; RF IQ −1.32). | [`outcomes`](paper_tables/2026-08-24_wan_v2v_caption_wave1_outcomes.md) · [`always+ada`](paper_tables/2026-08-25_wan_v2v_caption_always_adasteer.md). No Always call. |
| `v2v_panda_caption_prefix_32v` | Caption Prefix-match: `seed_bon`, `live_bon`, `appear_bon`. **N=32**. | **IN FLIGHT 2026-08-25 02:52.** **16328464–466** PD h200. VBench **16328467** afterok. | Cite vs caption notta. Spec: [`prefix32`](paper_tables/2026-08-25_wan_v2v_caption_prefix32_spec.md). |
| `v2v_panda_caption_closed_32v` | WAVE=2: seed/quiet/live/appear + host-split. **N=32**. | **DO NOT SUBMIT.** Prefix-only wave covers the slide. Host-split / quiet stay closed. | `sf_roll` is NO. |
| `v2v_panda_caption_8v` | WAVE=3: remaining N=8 discovery. | **READY after WAVE=1.** | No shift_search / knob_probe. |
| `v2v_panda_caption_128v` | WAVE=4: caption notta + rolling. **N=128**. | **READY after WAVE=1 harvest.** | Hosts only. Do not mix with stem rolling-128. |
| `v2v_panda_adasteer_8v` | AdaSteer on SF V2V: `ada_fixed`, `ada_stream`, `ada_resid`. Captions. **N=8**. | **DONE 2026-08-25.** 033–035 COMPLETED 0:0 18–21m 8/8. VBench **16326036** 20m. `|δ|`≈0.84. | **NO.** IQ 42.7 / 51.5 / 17.8. Stream tail +11% fails letter. Do not scale. [`always+ada`](paper_tables/2026-08-25_wan_v2v_caption_always_adasteer.md). |
| `v2v_panda_rolling_leftovers_8v` | rolling_rho_lo/hi, rolling_adapt, rolling_look. N=8. | **DONE 2026-08-22.** 129–132 9–11m, VBench 133 27m. 8/8 + VBench. | ρ knob **lives** but IQ fails vs host (hi −3.8). look HOLD N=8 only. Verdict: [`2026-08-22_wan_v2v_leftovers8_verdict.md`](paper_tables/2026-08-22_wan_v2v_leftovers8_verdict.md). |
| `v2v_panda_host_split_32v` | H1 `sf_roll`/`rf_chunk` + H4 `sf_recache`/`rf_recache`. **N=32**. H2/H3 offline. | **H1+H4 DONE 2026-08-23.** 197/199/200 + rf_chunk **16228103** 32/32 + VBench **16228104**. | Crosses twitch (tail 0.028 / Dyn 1). Do not scale. Read: [`2026-08-23_wan_v2v_host_split32_h1_read.md`](paper_tables/2026-08-23_wan_v2v_host_split32_h1_read.md). |
| V2V coverage audit (ran tests) | Sidecar inventory + offline router. | **DONE 2026-08-22 12:25.** Login CPU. | Always-rolling +31% beats still→notta/live→rolling +9%. N=8 search ρ is lucky-8. Trust resim was a key miss. Read: [`2026-08-22_wan_v2v_coverage_audit_read.md`](paper_tables/2026-08-22_wan_v2v_coverage_audit_read.md). |
| collapse+band picker resim | Offline on seed_bon-8/32 + live_bon-32 cand logs. | **DONE 2026-08-22.** No GPU. | 0002/0003/0027/0028 pass. **0022 fails** (band_appear cand1, same as seed). damp=0. **No generate.** Read: [`2026-08-22_wan_v2v_collapse_band_read.md`](paper_tables/2026-08-22_wan_v2v_collapse_band_read.md). |
| Wan Panda-prefix pixel audit | Optional later. PSNR/SSIM/LPIPS/FVD need paired long GT; VBench-I2V stills do not have it. | Not started. Would reuse Panda-70M / UCF prefixes already on cluster. | Do **not** invent PSNR on the current 32. New generate series only. |

---

## Active discovery / ablation experiments (not paper-grade, kept for audit)

These exist but should NOT be mixed with headline tables. They are kept to
document the methodology trail (how we picked LR / steps / target blocks).
Per-series N is small; FVD/FID values are sample-size-biased.

| Series | N | Methods | Purpose | Status |
|---|---|---|---|---|
| `panda_adasteer_ablation` | 100 | AS_CLIP_T10, AS_CLIP_T15 | CLIP threshold sweep | Discovery |
| `panda_cover_candidates` | 26 | NOTTA, DV_BARE, LORA_R8_S10 | LoRA-collapse cover | Discovery |
| `panda_longctx` | 50 | NOTTA, ADA_S10, LORA_R8 | Long-context discovery (precursor to `panda_longctx_1000v`) | Superseded by 1000v |
| `ucf_longctx` | 50 | NOTTA, ADA_S10, LORA_R8 | UCF long-ctx discovery | Superseded by `ucf101_683v_longhorizon` |
| `ucf500_lora_collapse_cover` | 30 | NOTTA, LORA_R8_S50, ADA_S10_AREG_D2 | LoRA collapse documentation on UCF | Discovery |
| `delta_a_iter_sweep`, `delta_a_lr_sweep` | 99 | DA1-DA10 | AdaSteer hyperparameter discovery | Superseded by `panda_1000v_standard/ADA` |
| `delta_b_*`, `delta_c_*` | 93-99 | DB1-DB11, DC1-DC5 | Variant family ablations | Discovery |
| `full_iter_sweep`, `full_lr_sweep` | 99 | F1-F9 | Full fine-tune ablation | Discovery |
| `lora_rank_sweep` | 99 | L1-L5 | LoRA rank sweep | Discovery |
| `tinylora_sweep` | 100 | TL_* (13 variants) | TinyLoRA discovery | Superseded by `tinylora_panda_1000v_standard/{TL_BARE_R2, TL_TIED_R2}` |

---

## Datasets and retrieval pools

### Eval sets

| Name | Cluster path | N | Notes |
|---|---|---|---|
| Panda 1000v eval | `datasets/panda_1000_480p/` | 1000 | Used for all Panda eval runs |
| Panda 100v eval | `datasets/panda_100_480p/` | 100 | Discovery |
| UCF-101 1000v eval | `datasets/ucf101_1000_480p/` | 1000 | Used for `ucf101_932v_*` runs |
| UCF-101 std eval | `datasets/ucf101_std_480p/` | (varies) | Used by `submit_retrieval_1000v_chunked.sh` for UCF retrieval |
| UCF-101 test eval | `datasets/ucf101_test_480p/` | (varies) | Older runs |

### Retrieval pools — embedding-database status

The retrieval-augmented sweeps require pre-computed `caption_embeddings.npy` +
`caption_embeddings.json` in the pool directory. Without these, `K_SIM` runs
fall back to encoding captions per-job (~30-60 s/job overhead). **Verify
embedding presence before any retrieval submission.**

| Pool name | Cluster path | Pool size (entries) | Embeddings precomputed? | Used by |
|---|---|---|---|---|
| Panda 2048-clip pool | `datasets/panda_2048_480p/` | 2048 | Yes (per submit_retrieval_1000v_chunked.sh header docstring; verify with `ls .../caption_embeddings.*`) | `panda_1000v_retrieval` (default in submit script) |
| Panda segment pool (Phase 2A) | `datasets/panda_segment_pool/` | ~3000 | Status UNCONFIRMED — verify on cluster | not yet wired into any submit script |
| Panda segment pool (Phase 2B target) | (would be `datasets/panda_segment_pool_25k/` or similar) | 25000+ | NOT BUILT — Phase 2B started late May, never completed | future: replace `panda_2048_480p` in retrieval submit script if built |
| UCF-101 max chunked pool | `datasets/ucf101_pool_max/` | ~26000 | Yes (used successfully by completed `ucf101_932v_retrieval` sweep) | `ucf101_932v_retrieval` |

**CURRENT GAP:** Panda retrieval submitted today uses the 2K-entry pool, not
25K. UCF retrieval was already on a 26K pool. If the 2K-pool Panda result
shows no gain, we still need the 25K Panda pool to fully claim "retrieval
doesn't help" — pool diversity could be the confound.

### Verify embedding-database presence (run on cluster)

```bash
cd /scratch/$USER/longcat-video-tta
for pool in datasets/panda_2048_480p \
            datasets/panda_segment_pool \
            datasets/ucf101_pool_max; do
    echo "=== $pool ==="
    if [ -d "$pool" ]; then
        ls -la "$pool"/caption_embeddings.* 2>&1 | head -5
        if [ -f "$pool/caption_embeddings.npy" ]; then
            python -c "
import numpy as np, json
e = np.load('$pool/caption_embeddings.npy')
with open('$pool/caption_embeddings.json') as f: m = json.load(f)
print(f'  shape={e.shape} dtype={e.dtype} entries={len(m) if isinstance(m, list) else len(m.get(\"captions\", m))}')"
        fi
    else
        echo "  (pool dir does not exist)"
    fi
    echo
done
```

---

## Pending merges and in-flight sweeps (UPDATE WHEN STATUS CHANGES)

| Sweep / job | Submit date | Job IDs | Expected wall | Next-step command |
|---|---|---|---|---|
| 1. Panda full metadata download (`panda_metadata_full/panda70m_training_full.csv`, 12 GB CSV / 2.6 GB ZIP) | 2026-06-08 (no-op skip) | 10616455 (COMPLETED 35s — file already on disk from June 1) | n/a | DONE — proceed to step 2. The metadata had been on disk under `datasets/panda_metadata_full/` the whole time; earlier verification looked at the wrong path. |
| 2. Panda 25K segment pool build (extends existing 3.3K pool to ~22-25K via full metadata) | 2026-06-09 (1:38 AM UTC+8 relaunch) | 10619044 (RUNNING; previous attempt 10617270 FAILED at 49s on csv field-size-limit, fixed in commit 5d565d4) | ~1-3 h on 16 CPU workers (per Phase 2A baseline); 12h hard cap; idempotent | After done: verify `ls datasets/panda_segment_pool/videos/*.mp4 \| wc -l` ≈ 22K+ and `cat datasets/panda_segment_pool/validation_report.json`, then submit step 3 |
| 3. Panda 25K embedding precompute | After step 2 | TBD | ~30 min on 1 GPU | After done: verify `caption_embeddings.npy` shape ≈ (25000+, 384), then launch step 4 |
| 4. Panda 1000v retrieval sweep (40 jobs, K5/K10 × SIM/RAND, against 25K pool) | After step 3 | TBD | ~3 days with 2-way GPU cap | Merge: `python sweep_experiment/scripts/merge_chunks.py --results-dir sweep_experiment/results/panda_1000v_retrieval --recursive`; then `python scripts/update_merged_with_vbench.py --series-dir sweep_experiment/results/panda_1000v_retrieval --force`; then `python scripts/build_paper_tables.py --regime panda_std --output sweep_experiment/reports/paper_tables/$(date +%Y-%m-%d)_panda_retrieval_followup.md` |

**Pivot rationale (2026-06-08):** the original same-day plan was to submit
step 4 against the 2048-clip pool, but verification showed neither a 25K
nor any other Panda pool exists at the user's stated target size. We
pivoted to a 4-step pipeline so the actual experiment lines up with the
paper claim. Records of this pivot are in `ANALYSIS_LOG.md` (entry 2026-06-08).

**Cancellation note (2026-06-08, 12:15 AM UTC+8 next day):** the user
submitted the original 2K-pool sweep (job IDs 10615946–10616023, all
`t1kr_panda_*`) before the pivot landed. All 40 jobs were cancelled
before any chunk completed. The `sweep_experiment/results/panda_1000v_retrieval/`
directory was wiped to avoid mixing 2K-pool and 25K-pool partial outputs.

---

## Code commits relevant to result reproducibility

| Commit | Description | Affected series |
|---|---|---|
| `64f608a` | Fix `batch_method=random` -> `sequential` in retrieval submit script | `ucf101_932v_retrieval/K*_RAND` |
| `4cf8b57` | VBench backfill env: pin opencv-python-headless==4.11.0.86, setuptools<80 | All 1000v VBench dims |
| `4aba71f` | VBench backfill sbatch: use `--gres=gpu:h200:1` + preemption comment | All 1000v VBench backfill jobs |
| `514237f` | VBench backfill submit script: propagate `PARTITION` env | (subsequent backfill submissions) |

---

## Where today's results live

- **Per-method merged summaries:** `*/results/<series>/<METHOD>/merged_summary.json` on cluster
- **Daily raw output logs:** `sweep_experiment/reports/experiment_outputs/YYYY-MM-DD.md`
- **Paper-ready tables:** `sweep_experiment/reports/paper_tables/`
- **Analysis log (decisions, findings):** [`ANALYSIS_LOG.md`](ANALYSIS_LOG.md)
- **VBench cache (compute reuse):** `/scratch/$USER/vbench-cache/` on cluster
- **Backfill targets TSVs:** `sweep_experiment/reports/vbench_backfill_targets*.tsv`
