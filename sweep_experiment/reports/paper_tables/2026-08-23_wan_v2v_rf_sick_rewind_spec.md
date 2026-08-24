# Family A — RF-sick rewind (2026-08-23)

Controller on Rolling Forcing, not a new student. First step is
**offline** on the existing 128 mp4s. No GPU until that print says GO.

## Baselines (locked)

| Role | Method | Why |
|---|---|---|
| **Paper / field primary** | Self-Forcing do-nothing | What Relax / Deep Forcing / FreqForcing subtract from |
| **Ablation zero if we GPU** | `rolling_notta` | The method is “rewind on RF,” so the zero is RF k=1 |
| Required comparison | both | A win vs SF only is not enough to call this the contribution |

Do not drop SF from the table. Do not rebrand RF as our method.

## Offline kill (login, no GPU)

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
python3 -u wan_experiment/scripts/resim_v2v_rf_chunk_trace.py --only n128
```

Same 81-frame windows as H2 (`prefix_pix`, then 6 × 21 latents).
`DROP=0.8` is pre-registered (same number as H3). Do not retune after
seeing the print.

| Call | Rule |
|---|---|
| **NO GPU** | Late-drop rate on RF-losses ≤ 25%, or losses were quieter from chunk 0 |
| **GO** | Late-drop rate on losses ≥ 50% **and** at least 20 points above the win rate |
| **HOLD** | Mixed — read named clips (0004 / 0027 / 0035 / 0044 / 0087) before a job |

Late drop = last-chunk motion `< 0.8 ×` chunk-0 motion.

If RF losses look like SF invented motion (0004-class) and RF was
quieter from the first window, salvage is the wrong story. Say so.
Do not rewind a correct refusal.

## If GO — GPU N=32 only

Same first 32 as confirm/forward. Method: after each RF window, if
the just-written 21 latents dropped vs the previous window by DROP,
discard them, restore KV, resample once. k=1 otherwise.

Promote only if median tail beats **`rolling_notta`** and IQ not worse
by ≥1.0 and subject not worse by ≥0.02. Analyzer PROMOTE vs SF is
ignored. Also report vs SF so the field number is in the table.

No look / recache / ρ / H2 router. No TTC. No I2V.
