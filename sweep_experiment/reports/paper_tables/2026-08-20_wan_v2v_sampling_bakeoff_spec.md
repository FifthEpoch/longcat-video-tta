# Spec — V2V sampling-space bake-off (beyond gating)

**Status:** SUBMIT-READY (2026-08-20). Host is **V2V prefix-continuation**,
not I2V-from-still and not T2V-from-scratch. No TTC. Do not scale I2V-32.
Do not retune hybrid/sticky/sick as a quality claim.

Cluster (after `git pull`):
```
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
SMOKE=1 bash wan_experiment/sbatch/submit_v2v_bakeoff.sh    # N=2 NOTTA
PROBE=1 bash wan_experiment/sbatch/submit_v2v_bakeoff.sh    # shift/CFG probe
bash wan_experiment/sbatch/submit_v2v_bakeoff.sh            # N=8 wave-1
```

---

## Why this exists

I2V-32 seed-BoN + gating was a **full-clip VBench tie**. Changing the seed
does not change a collapsed trajectory. The 2026-08-18 memo
([`2026-08-18_wan_nonweight_next.md`](2026-08-18_wan_nonweight_next.md))
listed sampling-space interventions. This series hosts them on a **real
video prefix** (the claim), then runs every cheap live method in parallel.

## Protocol (locked)

| Knob | Value |
|---|---|
| Model | Wan2.1-T2V-1.3B + Self-Forcing causal DMD |
| Task | **V2V**: first 9 latents = real Panda prefix (~33 px / 2.1 s), then AR |
| Source | `datasets/panda_1000_480p/` |
| Prompt | Panda caption if found; else clip stem |
| Generate | 6 × 21 latents after the prefix (~30 s tail). Total ~135 latents / 537 px |
| Piece 0 | Real prefix. Never searched. |
| Seed | 0, paired across methods |
| Official score | VBench quality 7 on the **full generated clip** (tail+prefix). last5 diagnostic |
| Drift | Generated tail only (skip the real prefix) |
| Series | smoke `v2v_panda_smoke` / bake-off `v2v_panda_bakeoff_8v` |

## ROI ranking (wave 1 vs later)

Wave 1 (cheap, parallel after the knob probe):

1. `notta` — do-nothing baseline
2. `seed_bon` k=4 — old seed search (control)
3. `motion_bon` k=4 — pick **more** `|Δframe|`, not two-sided deviation
4. `shift_search` — `{5, 8, 12}` if the probe shows pixels move
5. `backtrack` — rewind one chunk if outgoing explodes

Wave 2 (do not implement until wave 1 is scored):

- Attention sink (NVIDIA `sink=5 + window=7 + rerope` preset / new ckpt)
- CachedSearch
- Fractional history guidance
- CFG search, unless the probe shows a real pixel move (DMD default `guidance_scale=1.0`)

## Decision rule (Phase 4)

Cite **medians**. Promote a method to N>8 only if it beats `notta` on
tail motion **and** does not tank imaging quality or subject consistency
on full-clip VBench. A motion win that freezes identity is a fail.

No PSNR on these clips unless a later audit uses paired long GT.
No TTC. No I2V-32 scale-up.
