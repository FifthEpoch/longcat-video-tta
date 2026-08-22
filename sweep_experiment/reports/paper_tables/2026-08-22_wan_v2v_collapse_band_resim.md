# Collapse + band picker resim (2026-08-22)

**Zero GPU.** Replay a collapse gate + motion band on existing
candidate logs. Not a counterfactual 30 s tail.

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
python3 -u wan_experiment/scripts/resim_v2v_collapse_band.py \
  --method-dir wan_experiment/results/v2v_panda_bakeoff_8v/seed_bon_h30s_shard0 \
  --method-dir wan_experiment/results/v2v_panda_confirm_32v/seed_bon_h30s_shard0 \
  --method-dir wan_experiment/results/v2v_panda_live_32v/live_bon_h30s_shard0
```

Policy (per chunk, cand0 = NOTTA twin):

1. `prefix < 0.001` → keep cand0 (do not match a hold).
2. Else if `cand0_motion ≥ 0.7 × prefix` → keep cand0 (not collapsed).
3. Else feasible = cands in `[0.85, 1.15] × prefix`; pick min appearance.
4. If empty, pick `argmin |motion − prefix|`.

## Earn a generate only if

- Bake-off 0002/0003 stay cand0 (no damp).
- 0007 / 0026 recover (pick above cand0).
- Confirm hots **0022 / 0027 / 0028** stay at cand0 (do not pick the
  seed damper).

If 0022 still damps, this family does not get a GPU.
