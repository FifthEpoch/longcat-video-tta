# In-chunk denoise hooks — spec (2026-08-28)

**Status:** SUBMIT-READY. Caption N=8. Same-wave twins in one paste.
Appear punch stays **1.5×** prefix (sharp / color / sat). Do not retune.
**Do not scancel** SF intra `16471672–677`. **Do not rerun RF intra.**
**No TTC. No I2V. No AdaSteer. No CachedSearch this wave.**

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
SMOKE=1 WAVE=lastmix bash wan_experiment/sbatch/submit_v2v_denoise8.sh
WAVE=lastmix bash wan_experiment/sbatch/submit_v2v_denoise8.sh
```

GRES is tight while intra is PD. Start with **`WAVE=lastmix`**. Then
`WAVE=bpseudo` and `WAVE=restep`. `WAVE=all` is 9 generate jobs.

---

## Why

A 21-latent chunk is four DMD steps × seven 3-latent blocks, decoded
once at the end. Intra-chunk (`sf_intra`) already retries a **finished**
block. These three tests hook the **denoise steps themselves**.

CFG / shift and first-step `U_t` (`noise_bon`) are closed. This is not
another chunk-level controller.

---

## What we run

| Method | Host | Hook | Gate |
|---|---|---|---|
| `sf_lastmix` | SF | last of 4 DMD steps: `0.5 * step3 + 0.5 * step4` | appear punch |
| `sf_lastmix_always` | SF | same mix | always |
| `rf_lastmix` | RF | 0.5-mix last 3 latents with previous 3 | appear punch on last 3 |
| `rf_lastmix_always` | RF | same mix | always |
| `sf_bpseudo` | SF | hide last **3** committed latents; extra seed rewrites B; restore; next block uses that seed if it wins MAE | extra beats cand 0 |
| `sf_bpseudo_always` | SF | same hold-out | always pick best-B seed |
| `rf_bpseudo` | RF | hold out last 3 of the span; if extra MAE wins, rewind the 21-span | extra beats cand 0 |
| `sf_restep` | SF | redo last **2 of 4** DMD steps with extra seeds | appear punch |
| `sf_restep_always` | SF | same redo | always, keep least-punch |
| `rf_restep` | RF | reroll last 3 latents of the span | appear punch |
| `rf_restep_always` | RF | same reroll | always |

**No `rf_bpseudo_always`.** That is `rf_intra_always`, already **NO**.

k=4 on SF search arms. Block-pseudo B is one full block (3 latents) —
DMD cannot denoise a single latent. γ = 0 (`pseudo_gamma`).

Series: `v2v_panda_caption_denoise_8v`. Prompts = `metadata.csv`.
Cite vs caption Self Forcing (`v2v_panda_caption_32v/notta`).

---

## Call after harvest

HOLD only if tail or Dyn% beats SF **and** IQ / subject hold the
caption bars (IQ ≥ 70.54, subject ≥ 0.680). Always-on says whether
the pick or the gate did the work. RF twins are span-level on
purpose. Do not retune 1.5× on this 8.

---

## Not this wave

CachedSearch prune. Mid-denoise RGB on a 50-step sampler. Prefix
hold-out retune. N=32. Intra resubmit.
