# Family wave — A/B/C/D at once (2026-08-23)

One login paste: offline chunk-trace on 8/32/128 **and** N=32 GPU
jobs for all four families. **Pack-2 OOMed** (16261273–276).
Resume with `VIDEO_WORKERS=1` (skip-existing). VBench is **L40S
afterok**. See `2026-08-23_wan_gpu_pack2_oom.md`.

## Baselines

| Role | Method |
|---|---|
| Paper / field primary | Self-Forcing do-nothing (`v2v_panda_confirm_32v/notta`) |
| Ablation zero | `rolling_notta` (`v2v_panda_forward_32v`) |
| Required | report both. Analyzer PROMOTE vs SF is not the call. |

## Arms (series `v2v_panda_family_32v`)

| Family | Method | k | What it is |
|---|---|---:|---|
| A | `rf_rewind` | 1 | At each 21-latent freeze, if motion `< 0.8×` previous chunk, resample that 21 once; reject if quieter |
| B | `rf_sick_search` | 4 | Search only after a sick freeze; pick max motion among cands with motion ≥ `0.8×` cand0 |
| C | `rf_sink` | 1 | LongLive-style sink on RF. **Not HG-f** (that code is not in-repo). Pixel-move probe at N=32 |
| D | `rf_pseudo` | 4 | Hold out last 3 prefix latents; if extra seed beats native MAE on B, search the tail like B |

`DROP=0.8` pre-registered. Do not retune.

Promote an arm only if median tail beats **`rolling_notta`** and IQ not
worse by ≥1.0 and subject not worse by ≥0.02. Also print vs SF.

## Submit

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
python3 -u wan_experiment/scripts/resim_v2v_rf_chunk_trace.py --only all
bash wan_experiment/sbatch/submit_v2v_family_wave.sh
```

Paste both stdouts (trace + job IDs). No TTC. No I2V. Do not
resubmit 16259396.
