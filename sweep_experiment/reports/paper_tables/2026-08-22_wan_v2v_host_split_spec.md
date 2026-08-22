# Four host hypotheses (2026-08-22)

`rolling_notta` is someone else's host, not our controller.
These four tests ask whether a **cheap** method (no LoRA, no backprop)
can isolate or reuse that win. Do **not** rebrand RF as AdaSteer / TTA.

**First GPU run is N=32** (same first 32 as confirm/forward).
N=8 was the draft and was never launched — seed / live / look all
printed wins at 8 and died at 32. Do not start another lucky-8.

## H1 — Split the host win: weights vs sampler (GPU N=32)

Crossed pair. Host is the checkpoint; sampler is the unroll.
`method.startswith("rolling")` no longer implies the RF ckpt.

| Method | Weights | Sampler | Question |
|---|---|---|---|
| `sf_roll` | Self-Forcing DMD | RF rolling window | Does the window unroll move SF pixels? |
| `rf_chunk` | Rolling Forcing DMD | SF 6×21 chunks | Does RF still win without its sampler? |

Compare to confirm **SF notta** and forward **`rolling_notta`**.
Do not scale to 128 from a weak 32.

| Result | Call |
|---|---|
| `sf_roll` bit-matches notta | Sampler dead on SF (same class as shift/CFG) |
| `rf_chunk` bit-matches rolling | Sampler is a no-op; the ckpt is the method |
| `sf_roll` beats notta on locked bars | Cheap method = change the unroll, not θ |
| Only RF ckpt moves the tail | Cheap TTA is "pick a student." Not ours. |

## H2 — One-chunk bake-off (offline, no GPU)

Both 30 s videos already exist (8 / 32 / 128). Score first-chunk
motion from the mp4 (skip `prefix_pix`, next 81 frames = 21 latents).
Keep the full video whose chunk-0 motion won.

This is **not** "still→SF, live→RF" (that rule already lost: +9% vs
always-RF +31%). It is argmax of **generated** chunk 0.

YES only if bake median tail **beats always-RF**. If Spearman(Δc0, Δtail)
is weak, chunk-0 rank does not predict 30 s — do not GPU a router.

## H3 — Always RF, SF only as a veto (offline, same mp4s)

Default RF. Switch to SF only if RF chunk-0 `< 0.8 ×` SF chunk-0
(`ROLL_TRUST_FRAC`). Same counterfactual tails.

YES only if veto beats always-RF. A veto that mostly stays on RF is
a no-op. A veto that throws away RF stills is the prefix gate again.

## H4 — Recache from pixels, not poisoned KV (GPU N=32)

`sink` / `tail_hist` only shortened attention (+0% / +0.8%).
This decodes the last ~2 s (9 latents), **VAE-encodes**, writes those
latents back, and the next unroll replays KV from the round-tripped
latents.

| Method | Host | When |
|---|---|---|
| `sf_recache` | SF chunked | after every gen chunk except the last |
| `rf_recache` | RF rolling | every 21 frozen tail latents |

Compare each to **its own host** (notta / rolling_notta).
IQ drop ≥1.0 vs host = NO (prefix_sink class). Bit-match = no-op.

## Series / submit

GPU series: `v2v_panda_host_split_32v`. Same first 32 Panda videos
as `v2v_panda_confirm_32v` / `v2v_panda_forward_32v`.
Wall 8 h generate / 12 h VBench (N=8 leftovers were 9–11 min; N=32
rolling was 27 min; SF chunked-32 is the long arm).
2-way H200: queues behind 128 VBench **16209128** if still running.

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
python3 -u wan_experiment/scripts/resim_v2v_host_switch.py
bash wan_experiment/sbatch/submit_v2v_host_split.sh
```

Paste resim stdout + sbatch job IDs. No leftover scale-up. No TTC.
No I2V scale-up. Do not retune `live_min`.
