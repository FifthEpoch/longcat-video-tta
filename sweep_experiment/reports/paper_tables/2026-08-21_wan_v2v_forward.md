# Forward leftovers + mixed-result audit (2026-08-21)

`live_bon` N=32 is **NO**. Do not retune `live_min`. Do not scale
`live_hist` / `pseudo_*` / `noise_bon` / `live_appear`.

## Forward (GPU, N=32 vs confirm notta)

These are the only two that are not the dead live-gate and that
passed N=8 motion **and** quality (or Dyn).

| Method | Why it is still a candidate | Submit |
|---|---|---|
| `rolling_notta` | Host, not our picker. N=8 +29% tail, Dyn 0.5, IQ/subject hold | **YES — N=32** |
| `appear_bon` | Appearance pick (motion dropped). N=8 +7%, Dyn 0.5, IQ hold. Not a live_bon twin on VBench | **YES — N=32** |

Same yes/no bars as live_bon-32. Sidecars, not summary stubs.
Series: `v2v_panda_forward_32v`.

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
python3 -u wan_experiment/scripts/diagnose_v2v_mixed.py \
  --baseline-dir wan_experiment/results/v2v_panda_bakeoff_8v \
  --series-dir wan_experiment/results/v2v_panda_lineage_8v \
  --series-dir wan_experiment/results/v2v_panda_ideas_8v
bash wan_experiment/sbatch/submit_v2v_forward.sh
```

## Investigate first (no new N=32)

| Method | Win | Fail | Question |
|---|---|---|---|
| `longlive_prefix_sink` | tail +84%, Dyn 1.0 | IQ −2.0, flicker 0.971 | Is tail flicker? If `d_tail` anti-correlates with flicker, **close**. If flicker is flat and IQ drops on 1–2 clips, then a fix is worth talking about. |
| `longlive_notta` | IQ +2, subject +0.036 | tail −10%, Dyn 0 | Is quality just more identity / less motion? If no tail wins and Dyn stays 0, **not a Dyn fix**. |

Do **not** submit prefix_sink-32 or LongLive-32 until that paste says the gain is content motion.

## Not moving

`live_bon`, `live_hist`, `pseudo_gate` (became seed_bon), `noise_bon`
(= appear), `pseudo_appear` (= appear), `longlive_sink` (no-op).
