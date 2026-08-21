# Spec — V2V N=32 confirm (notta vs seed_bon)

**Status:** SUBMIT-READY 2026-08-20. User said go.
**Series:** `v2v_panda_confirm_32v`
**Why:** N=8 seed_bon passed the promote rule (+35% tail motion, IQ −0.60,
subject held, Dyn med 0→0.5). N=8 Dyn is a coin-flip. Confirm or kill.

```
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
CONFIRM=1 bash wan_experiment/sbatch/submit_v2v_bakeoff.sh
```

Two jobs: notta (~4 h wall) and seed_bon k=4 (~8 h wall). Same first 32
Panda clips as `discover_v2v_items` (the N=8 set is the prefix).

## Locked

- Methods: **notta** and **seed_bon** only
- No motion_bon, backtrack, shift_search, CFG, TTC
- Same V2V protocol: 9-latent prefix, 6×21 gen, seed 0
- Official score after generate: full-clip VBench 7 + tail `|Δframe|`
- Cite medians. Same promote bars: IQ worse ≥1.0 or subject worse ≥0.02
  → fail, even if motion is up
- If the motion / Dyn win dies at N=32: N=8 was noise. Do not write the
  paper around seed-BoN.

No I2V-32 scale-up.
