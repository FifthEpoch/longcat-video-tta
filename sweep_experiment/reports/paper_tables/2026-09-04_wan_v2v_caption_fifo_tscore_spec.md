# Caption FIFO lookahead + lock-score — SUBMIT-READY (2026-09-04)

Diagnostic. Not leftover ρ. Not mixed host. Do not remake cite-128.
Kim et al. FIFO-Diffusion already named lookahead. The lock-score
is DMD-shaped, not a new loss.

## Talk-through

**FIFO lookahead (`rolling_fifo`).** Rolling already denoises a
window whose first block is almost clean and whose last block is
almost noise. FIFO’s training-free move: draft the **noisier half**
once, put those frames back at the same t, then run the emit
forward so the about-to-lock head sees a less raw future. Same
student, same list. About 2× Rolling wall, still << Always-search.
This is Kim et al. on our host, not a paper title.

`rolling_fifo_sick` is the gated twin: extra pass only after a
sick lock (motion < 0.8× previous).

**Lock-score (`rf_tscore` / `sf_tscore`).** DMD’s signal is
teacher-score minus critic on a noised clip. We cannot host
Wan-14B next to the 39 GB rolling KV on one H200. The launchable
score is the **1.3B student as a freeze-score** (`s_fake` role):
noise the just-locked 21 latents at the mid list step, one
forward, mean |pred − clean|. Lower is better. Gated: redraw if
this span is 1.2× worse than the previous span. Always-on: always
draw a second seed and keep the better score.

That is not Wan-14B real-score. If N=8 lives, a later wave can
swap the scorer. A twitch or Imaging Quality collapse is **NO**.

## Arms (k=1, caption N=8)

| Method | Host | Twin |
|---|---|---|
| `rolling_fifo` | Rolling | `rolling_fifo_sick` |
| `rolling_fifo_sick` | Rolling | gated |
| `rf_tscore` | Rolling | `rf_tscore_always` |
| `rf_tscore_always` | Rolling | always-on |
| `sf_tscore` | Self Forcing | `sf_tscore_always` |
| `sf_tscore_always` | Self Forcing | always-on / other host |

Cite Rolling arms versus caption Rolling first-8. Cite Self
Forcing arms versus caption Self Forcing first-8.

## Submit

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
bash wan_experiment/sbatch/submit_v2v_caption_fifo_tscore.sh
```

`scancel` this wave only. No I2V. No TTC. Do not remake cite-128.

## Harvest

8/8 + `metadata_csv`. `rolling_fifo` sidecar `fifo_n` > 0.
`rf_tscore` sidecar `rewind_logs` with `score0`. Bars: tail vs
matching host; Imaging Quality not worse by ≥1.0; Subject
Consistency not worse by ≥0.02.
