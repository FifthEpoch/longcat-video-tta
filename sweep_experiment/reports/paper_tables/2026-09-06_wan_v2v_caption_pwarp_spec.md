# Caption N=8 slide the guessed picture — spec (2026-09-06)

**Status:** SUBMIT-READY. Pred-only (not extras). Self Forcing
host. Same-wave twins: always-on + leftover-live gate.
Prompts = `metadata.csv`. Do not remake cite-128.
**No TTC. No I2V. No 8-GPU DMD.**

Why extras died (not this recipe):
`2026-09-06_nwarp_vs_gwf_why_iq_died.md`.

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
SMOKE=1 bash wan_experiment/sbatch/submit_v2v_caption_pwarp.sh
bash wan_experiment/sbatch/submit_v2v_caption_pwarp.sh
```

---

## Recipe (the guessed-picture idea)

Pass 1 of each 3-latent strip is ordinary. Then **slide
that guess** by 1 latent pixel along the leftover’s
dominant axis (sign of Farneback mean flow). Remaining
passes finish the shifted scene. Extras stay ordinary
white. New edge = repeat, not wrap, not noise.

Leftover speed on the truck clip was 0.004 cells / frame
— a faithful leftover slide is a no-op. The 1-cell floor
is the force. Direction still comes from the opening.

| Method | When the guess slides |
|---|---|
| `sf_pwarp` | Always |
| `sf_pwarp_live` | Only if leftover `prefix_motion >= 0.012` |

Series: `v2v_panda_caption_pwarp_8v`.
Cite vs caption Self Forcing first-8
(`v2v_panda_caption_32v/notta`).

This is not Go-with-the-Flow. They warp starting noise
and LoRA the video model. lastmix / restep already edited
a mid-strip guess. The leftover in memory does not slide
with the picture (KV seam).

---

## Hold

Dyn% (or tail motion) up **and** Imaging Quality /
flicker / subject hold the caption Self Forcing bars
(IQ ≥ 70.54, subject ≥ 0.680, flicker off the 0.978
twitch band). A Dyn-only lift is **NO**.

Kill test. If it paints or no-ops, stop. Do not combine
with `sf_nwarp` on this wave. Do not retune extra γ.

---

## Do not

Warp extras in the same job. Wrap the grid. Start on
Rolling Forcing. Remake cite-128. Launch 8-GPU DMD.
