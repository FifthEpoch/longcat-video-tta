# Caption mixed lock + context noise — SUBMIT-READY (2026-09-04)

Not the paper method. Mixed inference is Liu et al. Appendix E
(named, not run). Context noise is a key-value (KV) write
timestep, not leftover ρ. Do not remake cite-128.

## Arms (k=1, caption N=8)

Host checkpoints stay paired with their native sampler unless
the gate fires. First span / chunk is always native.

| Method | Host | What it does | Twin |
|---|---|---|---|
| `rf_mix` | Rolling | After a sick 21-latent lock (motion < 0.8× previous), next span is chunked on the same weights | `rf_mix_always` |
| `rf_mix_always` | Rolling | Every span after the first is chunked | always-on |
| `sf_mix` | Self Forcing | After a sick chunk, next chunk rolls on the same weights | `sf_mix_always` |
| `sf_mix_always` | Self Forcing | Every chunk after the first rolls | always-on / other host |
| `rolling_ctx` | Rolling | `context_noise=50` on KV write, including the real prefix | `sf_ctx` |
| `sf_ctx` | Self Forcing | Same KV dirt on the Self Forcing host | other host |

Sick = same `RF_SICK_DROP=0.8` as rewind. `50` is a small
flow-matching t (clean is 0; live Rolling floor is 556). This
is not leftover ρ (that scaled new-block init noise).

Cite Rolling arms versus caption Rolling Forcing first-8.
Cite Self Forcing arms versus caption Self Forcing first-8.
Analyzer HOLD / FAIL versus `notta` is not the letter.

## Submit

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
SMOKE=1 bash wan_experiment/sbatch/submit_v2v_caption_mixctx.sh   # optional 2-video
bash wan_experiment/sbatch/submit_v2v_caption_mixctx.sh
```

Cancel this wave only (print the JobIDs from submit):
`scancel <ids>`. Do not remake cite-128. No I2V. No TTC.
Do not start 8-GPU Distribution Matching Distillation (DMD).

## Harvest

8/8 + `prompt_source=metadata_csv` (truck hood).
`rf_mix` sidecar `mix_logs` / `n_chunked`.
`rolling_ctx` / `sf_ctx` sidecar `context_noise=50`.
Pair tails versus the matching caption-32 host.
Promote past N=8 only if median tail beats that host **and**
Imaging Quality is not worse by ≥1.0 and Subject Consistency
is not worse by ≥0.02.

A twitch (high tail, flicker collapse) is **NO**, same as
`sf_roll` / `rf_chunk`.
