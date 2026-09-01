# Caption 128 — all official metrics, four methods (2026-08-31)

**Series:** `v2v_panda_caption_128v`. Prompt = `metadata_csv`.
n=128. Cite VBench **medians** except Dynamic Degree =
**percent of clips** (`population.mean`). PSNR/SSIM = paired
30 s tail vs real leftover (median). Headline stays VBench +
Dyn%. LongCat already taught us PSNR and VBench can disagree.

| | Self Forcing | Rolling | Pseudo | Always |
|---|---:|---:|---:|---:|
| tail motion | 0.0119 | **0.0158** | 0.0157 | 0.0168 |
| mean s / clip (N=32) | 196 | **45** | 304 | 348 |
| **VBench++** | | | | |
| subject_consistency | 0.666 | **0.685** | 0.660 | 0.661 |
| background_consistency | — | — | — | — |
| aesthetic_quality | 0.499 | 0.529 | 0.510 | — |
| imaging_quality | 72.07 | 71.52 | **72.38** | 72.19 |
| motion_smoothness | — | — | — | — |
| dynamic_degree % | 32.8 (42) | 28.9 (37) | 47.7 (61) | **50.8 (65)** |
| temporal_flickering | 0.987 | 0.983 | 0.984 | 0.982 |
| **Paired pixels** | | | | |
| PSNR | 9.25 | — | — | — |
| SSIM | 0.279 | — | — | — |
| LPIPS | — | — | — | — |
| **Distribution** | | | | |
| FVD (aligned tails) | — | — | — | — |

## What is missing, and why

- **background_consistency** and **motion_smoothness** are on
  every `vbench_full/joined.json`. Not copied into earlier
  harvests. Login dump fills them. Always **aesthetic** is on
  the same join (**16674378**).
- **PSNR/SSIM** for Rolling / Pseudo / Always: pixel job
  **16678705** preempted after Self Forcing. Resubmit
  skip-existing. Do not remake videos.
- **LPIPS:** `lpips` is not in the self_forcing env (SF row is
  `None`). Install or skip. Not a method delta.
- **FVD:** not run. Must be I3D on **aligned 30 s tails**,
  `--force` (n=128 < 256). Do **not** score the full mp4 — that
  includes the real prefix and is not a future metric.

Self Forcing PSNR **9.25** / SSIM **0.279** is one method only.
Do not cite it as a bake-off until the other three land.

## Dump (login, no GPU)

```bash
cd /scratch/wc3013/longcat-video-tta
python3 - <<'PY'
import json
from pathlib import Path
ROOT = Path("wan_experiment/results/v2v_panda_caption_128v")
DIMS = (
    "subject_consistency", "background_consistency", "aesthetic_quality",
    "imaging_quality", "motion_smoothness", "dynamic_degree",
    "temporal_flickering",
)
print("method n subj bg aes iq smooth dyn_mean ndyn flick psnr ssim lpips")
for name in ("notta", "rolling_notta", "sf_pseudo", "sf_always_search"):
    d = ROOT / f"{name}_h30s_shard0"
    p = d / "vbench_full" / "joined.json"
    pix = d / "pixel_full" / "summary.json"
    if not p.is_file():
        print(name, "NO_JOIN")
        continue
    pop = json.loads(p.read_text()).get("population") or {}
    cells = []
    for dim in DIMS:
        cell = pop.get(dim) or {}
        if dim == "dynamic_degree":
            mean = cell.get("mean")
            n = cell.get("n")
            ndyn = int(round(float(mean) * int(n))) if mean is not None and n else None
            cells.append(f"{mean} {ndyn}/{n}")
        else:
            cells.append(str(cell.get("median")))
    if pix.is_file():
        s = json.loads(pix.read_text())
        cells.append(f"psnr={s.get('psnr')} ssim={s.get('ssim')} lpips={s.get('lpips')} n={s.get('n')}")
    else:
        cells.append("pixel=MISSING")
    print(name, *cells)
PY
```

FVD after pixel tails exist. Do not launch it on full mp4s.
