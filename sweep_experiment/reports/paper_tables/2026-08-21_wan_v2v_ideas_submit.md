# V2V sampling-space ideas 1 / 5 / 3 — submit spec (2026-08-21)

**Status:** SUBMIT-READY. Queue behind `v2v_panda_lineage_8v`
(16140808–816). **Do not scancel lineage.**

**Series:** `wan_experiment/results/v2v_panda_ideas_8v/`

**Videos:** same N=8 as `v2v_panda_bakeoff_8v` (`discover_v2v_items`
sorted path order). Prefix 9 latents, 6 × 21 generated, 30 s tail.

**Baseline:** bake-off `notta`. Analyze with `--baseline-dir
v2v_panda_bakeoff_8v --allow-partial`. Sidecars, not unpaired
`summary.json`.

**Submit (after `git pull --ff-only origin main`):**

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
bash wan_experiment/sbatch/submit_v2v_ideas.sh
```

2-way H200 cap: extras queue. That is intended.

---

## Methods

| Method | Idea | k | What it does |
|---|---|---:|---|
| `appear_bon` | 5 | 4 | Always search. Pick min appearance+seam. **Motion is not in the pick.** |
| `live_appear` | 5 + live gate | 4 | Search iff `prefix_motion >= 0.012`. Same appear pick. |
| `pseudo_gate` | 1 | 4 | Held-out last-3 prefix latents vs real B (pixel MAE). Search tail iff some extra seed beats notta MAE. Two-sided prefix-match pick when searching. |
| `pseudo_appear` | 1 + 5 | 4 | Same gate. Appearance pick on the tail. |
| `noise_probe` | 3 | 1 | notta + first-step residual stats (`eps_mean`, `eps_mean_abs`, `eps_std`) in sidecars. |
| `noise_bon` | 3 | 4 | Always cand0. Extra seeds iff cand0 `eps_mean_abs >= 0.04`. Appear pick. |

Overrides: `LIVE_MIN=0.012`, `PSEUDO_GAMMA=0.0` (strict MAE win, extra seed ≠ 0), `NOISE_TAU=0.04`.

U_t here is **not** TANGO ε. It is the first denoising-step residual
`noisy_input − denoised_pred` on 4-step DMD. If `noise_probe` shows U_t
flat across chunks/videos, `noise_bon` is a no-op or a random gate —
do not scale it.

---

## Promote past N=8

Same bars as lineage: tail motion > notta **and** IQ not worse by ≥1.0
**and** subject not worse by ≥0.02. Dyn must not freeze (median 0 with
notta > 0 is a fail).

---

## Not in this wave

Weight TTA, LoRA-at-test-time, rolling ρ on vanilla SF, lookahead beam,
horizon-increasing δ, full hybrid router, hist_drop-32, I2V scale-up.
