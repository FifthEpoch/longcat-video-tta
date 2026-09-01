# Caption leftover ρ / look — SUBMIT-READY (2026-09-01)

The Aug-22 leftover pack (`v2v_panda_rolling_leftovers_8v` +
lineage `rolling_notta`) used filename stems (`panda 0013`). T5
heard “panda,” so every tail is infected. Those videos are an
**audit**. Do not look at them as scene continuation.

Caption-128 / caption-32 Rolling already used `metadata.csv`.
**Do not remake cite-128.** If those clips still show a panda,
that clip’s real caption or the leftover T5 takeover is a
different fact — not a missing caption.

## What this wave remakes

New series: `v2v_panda_caption_leftovers_8v`. Same first-8 path
order as bake-off / caption-32. Prompt =
`prompt_source=metadata_csv`.

| Method | k | What | Stem call (do not cite) |
|---|---:|---|---|
| `rolling_rho_lo` | 1 | init-noise × (h/H)^0.5 | IQ −1.66 **NO** |
| `rolling_rho_hi` | 1 | × (h/H)^2.0 | IQ −3.77 **NO** |
| `rolling_adapt` | 1 | ρ from prefix motion | IQ −1.39 **NO** |
| `rolling_look` | 4 | lookahead + trust reject | HOLD n=8 only |

Host = existing `v2v_panda_caption_32v/rolling_notta` (first 8).
Do **not** regenerate native Rolling. Same seed / captions /
prefix should already be on disk.

## What this wave does **not** remake

Already caption-conditioned and closed:

- cite-128 hosts + Pseudo + Always
- keep / intra / denoise / AdaSteer / Pseudo-next
- prefix-match / crossed host

WAVE=3 (`v2v_panda_caption_8v`) dumps 19 extra discovery
methods. **Do not submit WAVE=3.** This script is the ρ / look
slice only.

## Lock

- New series name. Stem leftover dirs stay on scratch as audit.
- Cite vs **caption** Rolling, not stem lineage Rolling.
- k=4 only on `rolling_look`. Do not retune ρ or the look reject.
- ρ is RF-window-specific. No SF twin (not a host-swap claim).
- `rolling_look` is always-on lookahead, not a Pseudo gate.
  Host is the twin.
- First sidecar must be `metadata_csv`. If stem, scancel.
- No TTC. No I2V. `VIDEO_WORKERS=1`. VBench `afterok` full clip.

## Hypothesis

Stem “panda NNNN” made every leftover tail a panda morph. That
can fake motion (ρ_hi +40% tail) and wreck IQ. Caption replay
asks whether the knob still moves pixels **and** still kills
image quality when T5 hears the scene. Harvest can flip the
letter; do not import the stem IQ deltas.

## Submit

**IN FLIGHT 2026-09-01 14:46.** Generate **16734909–912**
(`rho_lo` / `rho_hi` / `adapt` / `look`). VBench **16734913**
afterok. Preflight PASS: 1000 captions, 8/8 first-segment,
0 stem. Cancel this wave only:
`scancel 16734909 16734910 16734911 16734912 16734913`.

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
bash wan_experiment/sbatch/submit_v2v_caption_leftovers.sh
```

Look at **caption** Rolling now (no GPU): first 8 of
`v2v_panda_caption_32v/rolling_notta_h30s_shard0` or
`v2v_panda_caption_128v/rolling_notta_h30s_shard0`.

## Harvest

8/8 mp4 + sidecar `prompt_source=metadata_csv` + VBench full
clip. Pair tails vs caption Rolling. Watch `rho_hi` next to
Rolling on the same ID. Do not mix numbers into
`2026-08-22_wan_v2v_leftovers8_verdict.md`.
