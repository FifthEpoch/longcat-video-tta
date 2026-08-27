# Intra-chunk N=8 harvest (2026-08-27)

**Series:** `v2v_panda_caption_intra_8v` (+ smoke)
**Jobs:** 16371523–527 smoke; 16371530–536 N=8
**Prompt:** `metadata_csv` (RF). SF wrote error json only.
**Cite vs caption SF** `v2v_panda_caption_32v/notta` (paired N=8).
**Do not retune** 0.8 / 1.5 on this 8.

## Call

| Method | Generate | vs SF | Call |
|---|---|---|---|
| `sf_intra` | **FAILED** 2:0 ~3 min, 8 json / 0 mp4 | — | **DEAD.** Need traceback. |
| `sf_intra_always` | **FAILED** 2:0 ~3 min, 8 json / 0 mp4 | — | **DEAD.** Same crash. |
| `rf_intra` | 8/8 + VBench | tail 0.0169 vs 0.0129; subj **0.645** / IQ **66.33** / Dyn **1** / flick 0.983 | **NO.** Quality collapse. |
| `rf_intra_always` | 8/8 + VBench | **bit-match rf_intra** on every official dim | **NO.** Gate did not split. |

Locked bars vs caption SF 0.700 / 71.54: IQ ≥ 70.54, subject ≥ 0.680. RF fails both. Dyn median 1 is the twitch signature, not a motion win.

Smoke: RF 2/2 mp4 in ~8 min. SF 2 json / 0 mp4. Same crash at N=2.

VBench jobs **16371527 / 536 FAILED** (empty SF). RF `vbench_full/joined.json` still landed.

## What this means

The thing we actually wanted — **SF block abort** — never wrote a video. The RF twin is span-level and here it is a no-split quality kill: gated and always-on are the same row. That is either “appear 1.5× fired every span” or “always-on always kept the same pick as the gate.” Do **not** loosen 1.5× on these 8. Do **not** scale.

`pair_v2v_tails.py` crashed because SF mappings are empty (`first is None`). Guard landed; rerun after SF jsons are real.

## Next paste (SF traceback only)

```bash
python3 - <<'PY'
import json
from pathlib import Path
d = Path("wan_experiment/results/v2v_panda_caption_intra_8v/sf_intra_h30s_shard0")
ps = sorted(p for p in d.glob("*.json") if p.name not in {"summary.json","joined.json"} and "vbench" not in p.name)
print("n", len(ps))
if ps:
    rec = json.loads(ps[0].read_text())
    print("ok", rec.get("ok"), "error", rec.get("error"))
    print((rec.get("traceback") or "")[-2500:])
PY
echo "===== SLURM ====="
ls -t wan_experiment/slurm_log/wan_v2v_chunk_16371530.* 2>/dev/null | head
tail -80 wan_experiment/slurm_log/wan_v2v_chunk_16371530.err
```

No resubmit until that paste. No WAVE=2. No I2V. No TTC.
