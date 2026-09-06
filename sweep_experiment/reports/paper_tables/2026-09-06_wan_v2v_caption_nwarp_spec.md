# Caption N=8 leftover-flow HIWYN extras — spec (2026-09-06)

**Status:** SUBMIT-READY. Extra-only (not `pred`). Self Forcing
host. Same-wave twins: always-on + leftover-live gate.
Prompts = `metadata.csv`. Do not remake cite-128.
**No TTC. No I2V. No 8-GPU DMD.**

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
SMOKE=1 bash wan_experiment/sbatch/submit_v2v_caption_nwarp.sh
bash wan_experiment/sbatch/submit_v2v_caption_nwarp.sh
```

---

## Recipe (the extra-only idea, with hole fixes)

Pass 1 of each 3-latent block is ordinary white
opening static. Do **not** slide `pred`. After pass 1,
every `extra` is HIWYN-style: spatially white per
frame, integer-shifted by a **frozen leftover mean
velocity** (Farneback on the real 2 s leftover, once),
holes resampled (no wrap), mixed with plain snow at
\(\gamma=0.5\). The particle field carries across
blocks and chunks. Host = Self Forcing. k=1.

| Method | When extras drift |
|---|---|
| `sf_nwarp` | Always |
| `sf_nwarp_live` | Only if leftover `prefix_motion >= 0.012` |

Series: `v2v_panda_caption_nwarp_8v`.
Cite vs caption Self Forcing first-8
(`v2v_panda_caption_32v/notta`).

---

## Hold

Dyn% (or tail motion) up **and** Imaging Quality /
flicker / subject hold the caption Self Forcing bars
(IQ ≥ 70.54, subject ≥ 0.680, flicker off the 0.978
twitch band). A Dyn-only lift is **NO**.

This is a kill test of frozen-student drifting snow.
It is not Go-with-the-Flow (they LoRA the video
model). If it paints or no-ops, stop.

---

## Do not

Move `pred`. Wrap the grid. Start on Rolling Forcing.
Remake cite-128. Launch 8-GPU DMD.
