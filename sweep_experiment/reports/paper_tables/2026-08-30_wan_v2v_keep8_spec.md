# Keep the picture — mid-chunk N=8 (2026-08-30)

**Status:** IN FLIGHT. Smoke **16616159–173**. N=8 **16616174–188**.
Gate = **latent travel** (how much the last latent of a block differs
from the first). Fire if travel < 0.8× the previous block. Never use
sharpness or color. Do not retune. k=4.

Do not scancel cite **16615748–750** or crash reruns **16615741–747**.

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
SMOKE=1 bash wan_experiment/sbatch/submit_v2v_keep8.sh
bash wan_experiment/sbatch/submit_v2v_keep8.sh
```

## Why these, not the old ones

The failed mid-chunk runs **replaced the picture** (50/50 mix, redraw
the last 0.7 s, redo two of four steps). Subject and image quality
dropped. These methods keep the default picture and only add a small
change, or they never edit a block already written.

## What we submit (14 generate + 1 score job)

| What it does in English | Code names |
|---|---|
| Last denoising step: keep 90% of the finished block, blend 10% of the previous step | `sf_nudge`, `sf_nudge_always` |
| Same 10% blend on Rolling’s last 3 frames | `rf_nudge`, `rf_nudge_always` |
| Leave this block alone. If it looks stuck, the **next** block uses a different seed | `sf_nextseed`, `sf_nextseed_always` |
| Keep the default block. Add 20% of (most-travelling seed − default). Lock the first latent to the default so who entered does not change | `sf_wiggle`, `sf_wiggle_always` |
| Same residual + lock on Rolling last 3 | `rf_wiggle`, `rf_wiggle_always` |
| Try 4 seeds. Keep the one whose last latent differs most from its first. Then copy the default first latent on top (subject lock) | `sf_latmot`, `sf_latmot_always` |
| Same first-vs-last pick + lock on Rolling last 3 | `rf_latmot`, `rf_latmot_always` |

Motion-only trigger is the **gate** on every gated arm, not its own
job. Always-on says whether the edit or the gate did the work.

## Subject lock (your consistency idea)

We do **not** pick the seed that matches the opening (that froze
motion before). We pick the seed that **moved most in latent space**,
then copy the default block’s **first** latent onto the winner. The
person who walked into the block stays; the rest of the block can
travel. Wiggle does the same lock after the 20% residual.

## Keep / drop after harvest

Keep only if tail motion or percent-dynamic beats Self Forcing **and**
subject ≥ 0.68 and image quality ≥ 70.5 (same 8-video bar as before).
Do not grow past 8 if those bars fail.

## Not this wave

CFG / shift (they do not move pixels here). 50/50 mix. Sharpness
trigger. Rewriting a finished block with a brand-new seed. Rolling
“next seed” (Rolling is one long roll, not blocks we can leave alone).
CLIP / DINO subject models.
