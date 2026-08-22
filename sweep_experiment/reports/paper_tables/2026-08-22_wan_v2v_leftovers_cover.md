# Cover remaining ideas (2026-08-22)

`rolling_notta` passed N=32 locked bars (host, not a controller).
This wave covers the leftover ideas without reopening closed search.

## (1) Depends on RF host — N=8 GPU

Series: `v2v_panda_rolling_leftovers_8v`. Same 8 as bake-off.
Compare to lineage `rolling_notta`, not SF notta.

| Method | Idea | What |
|---|---|---|
| `rolling_rho_lo` | 4 | per-block init noise × (h/H)^0.5 |
| `rolling_rho_hi` | 4 | × (h/H)^2.0 |
| `rolling_adapt` | 4 | ρ from prefix (still=2, mid=1, hot=0.5) |
| `rolling_look` | 6+7 | k=4 on every 7th new-noise window; seam pick; reject if motion < 0.8× cand0 |

Kill #4 if `rho_lo` / `rho_hi` bit-match native RF tails (knob dead, same as SF shift).
Kill #6 if look never leaves cand0 or IQ drops ≥1 vs rolling_notta.
Do **not** scale from this N=8 table.

N=128 host scale-up is separate (`submit_v2v_rolling128.sh`).

## (2) Cover the rest — no new weight TTA

| Idea | Status |
|---|---|
| 1, 3, 5 | Already run. Closed as paper methods (N=32 or lucky-8). |
| 2 Self-rollout TTA | **Weights locked.** On 9-latent prefix, B is one 3-latent block — AR rewrite = idea 1. No distinct GPU. |
| 7 Trust region | Offline resim on existing cand logs + baked into `rolling_look`. |
| 8 Horizon δ | Already falsified by `late_bon`. |
| 9 Hybrid router | Offline: oracle best-of-arms + prefix rule (still→SF notta, live→rolling). |

## (3) Analysis on ran tests (login CPU, no GPU)

```bash
python3 -u wan_experiment/scripts/audit_v2v_coverage.py
python3 -u wan_experiment/scripts/resim_v2v_trust_hybrid.py
```

## Submit

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
python3 -u wan_experiment/scripts/audit_v2v_coverage.py
python3 -u wan_experiment/scripts/resim_v2v_trust_hybrid.py
bash wan_experiment/sbatch/submit_v2v_rolling128.sh
bash wan_experiment/sbatch/submit_v2v_rolling_leftovers.sh
```

2-way H200: 128 takes both GPUs first; leftovers queue. Paste analysis stdout + sbatch job IDs.
