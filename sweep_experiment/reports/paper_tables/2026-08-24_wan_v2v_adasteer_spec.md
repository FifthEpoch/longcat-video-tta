# AdaSteer on Wan SF V2V — confirmation series (2026-08-24)

LongCat AdaSteer (Delta-A, 13.6B, 14→14) was **saturated ≈ NOTTA**.
Streaming δ on LongCat native AR was **NULL**. This is a new stack:
Wan2.1-T2V-1.3B + Self-Forcing chunks, Panda V2V 9→126 latents, **real
captions**. Question: does any δ-update rule move tail / quality vs
caption SF notta?

## Paper AdaSteer (what we port)

Huang-style LongCat method, not a new invention:

- One vector δ, one clip, discarded after.
- `t' = t + δ` into frozen per-block adaLN.
- Wan hook: `CausalWanModel.time_embedding` (then frozen
  `time_projection` → 6×dim modulation).
- AdamW on δ only. Loss = student few-step x0 reconstruction on
  **observed latents only**. No future GT. No LoRA. No TTC.

## Every LongCat update rule we already tried

| Family | Short name | Update | LongCat call | Wan wave |
|---|---|---|---|---|
| Paper | `ADA` / S10 LR 5e-3 | AdamW on prefix FM, **hold δ** | ≈ NOTTA at 1000v | **`ada_fixed`** |
| Budget | S{2,5,10,20} × LR grid | Same update, knobs | Population flat | parked (N=8 first) |
| Placement | `ADA_ADALN` vs `ADA_RESID` | δ on t vs mid-late residual | both lost to NOTTA | **`ada_resid`** |
| Stream | `delta_stream` | per-chunk refit, `δ ← (1−λ)δ_refit + λ δ₀` | NULL | **`ada_stream`** |
| Stream hold | `delta` long-horizon | fit once, hold across AR chunks | no flatten | same as `ada_fixed` |
| Retrieval | K5/K10 SIM/RAND | shared δ, one video / step | failed | parked |
| Aux | ES, AREG, anchor-x0, CLIP gate | extra loss / skip | discovery | parked |
| Other methods | Delta-B/C, FiLM, NormTune | not AdaSteer-1 | discovery | **not this series** |
| Weights | LoRA / TinyLoRA / full TTA | not a vector | separate | **no** |
| Sample | TTC / SAVi-DNO | not activations | separate | **no** |

## Confirmation arms (N=8, then N=32 if any moves)

Series: `v2v_panda_adasteer_8v`. Captions from `metadata.csv`.
S=10, LR=5e-3 (Panda paper defaults). Stream λ=0.5, refit 5.

| method | rule |
|---|---|
| `ada_fixed` | fit on 9-latent prefix, hold for 6 chunks |
| `ada_stream` | refit on last 9 committed latents each chunk, blend to δ₀ |
| `ada_resid` | same as fixed, δ on blocks 55–80% depth |

Cite vs **caption** SF notta (`v2v_panda_caption_32v/notta`, first 8).
Do not mix stem-prompt numbers. Sidecar must be `metadata_csv`.

Call: HOLD only if tail vs caption notta is up and IQ/subject stay
inside locked bars. Else **NO**. Do not scale a null.

Submit: `bash wan_experiment/sbatch/submit_v2v_adasteer.sh`

**IN FLIGHT 2026-08-24 19:20.** **16314667** ada_fixed, **16314668**
ada_stream, **16314669** ada_resid, VBench **16314670** afterok.
Queues behind caption WAVE=1. No TTC. No I2V.
