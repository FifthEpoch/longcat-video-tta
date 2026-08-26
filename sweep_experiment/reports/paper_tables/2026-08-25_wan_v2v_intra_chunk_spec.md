# Intra-chunk motion + appearance probe — spec (2026-08-25)

**Status:** SUBMIT-READY. Caption N=8. Same-wave twins in one paste.
**Do not retune** 0.8 / 1.5 after seeing 8. Harvest decides the call.
**No TTC. No I2V. No WAVE=2 leftovers. No AdaSteer.**

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
SMOKE=1 bash wan_experiment/sbatch/submit_v2v_intra8.sh
bash wan_experiment/sbatch/submit_v2v_intra8.sh
```

---

## Why

A seed is picked (or defaulted) for a **21-latent chunk (~5 s)**. We
decode once at the end. Saturation / sharpness can punch *inside* that
write, and motion can die, with no hook until the chunk commits.

Pseudo-future Search only decides **whether the video gets k=4**. It
cannot abort a block that goes neon or still after the seed is locked.

---

## What we run

| Method | Host | Grain | Gate |
|---|---|---|---|
| `sf_intra` | SF | **3 latents (~0.7 s)** | Fire if motion **or** appear sick; try other seeds |
| `sf_intra_always` | SF | 3 latents | No gate; k=4 every block |
| `rf_intra` | RF | 21-latent span | Same sick test; rewind span (RF windows overlap) |
| `rf_intra_always` | RF | 21-latent span | Always try an alt seed |

**Appear sick** (vs the real 2 s prefix): sharpness **> 1.5×** or
colorfulness **> 1.5×** or saturation **> 1.5×**.
**Motion sick:** block motion **< 0.8×** previous block (same 0.8 as
Rewind / trust).

Fire = motion **or** appear. We want both failure modes, not AND.

Series: `v2v_panda_caption_intra_8v`. Prompts = `metadata.csv`.
Cite vs caption Self Forcing (`v2v_panda_caption_32v/notta`).

---

## Call after harvest

HOLD only if tail or Dyn% beats SF **and** IQ / subject hold the
caption bars (IQ ≥ 70.54, subject ≥ 0.680). If appear-gate fires
every block, the 1.5× is too tight — log it, do **not** retune on
the same 8. Always-on says whether the pick or the gate did the work.

RF twin is span-level on purpose. If SF intra HOLDs and RF does
nothing, that is a host result, not a retune.

---

## Not this wave

Mid-denoise RGB (Early Failure Detection) on 4-step DMD. CachedSearch
inside the block. Prefix hold-out re-tune. N=32.
