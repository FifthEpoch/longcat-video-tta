# Caption 128 LPIPS + aligned-tail FVD (2026-09-01)

**Status:** **IN FLIGHT 16737041** (L40S `l40s_mren`, PD at
submit). PSNR/SSIM already **DONE**. Do not remake videos. Do
not recompute PSNR. `scancel 16737041` only — leftover
16734912–913 stay up.

LPIPS was `None` because `import lpips` failed in `self_forcing`.
FVD was never launched. This job fills both on the existing 128
clips.

## Protocol

**LPIPS.** Same pairing as PSNR. AlexNet, every 8th aligned frame
plus the last. Writes into the existing
`pixel_full/*.json` (`--fill-lpips`). Cite **median**.

**FVD.** I3D Kinetics-400 TorchScript (`eval_fvd.py` weights).
**Aligned 30 s tails only** — skip the real 33-frame prefix.
GT = source after frame 33, time-resampled to 16 fps.
I3D clip = **16 consecutive frames**. Primary score = all
non-overlapping windows on the tail (~31 × 128). Also write
`fvd_last16` (last window only, n=128). Do not feed the full
mp4 to I3D.

Caches (resume after L40S preemption):
`pixel_full/fvd_i3d/{stem}.npz` and `fvd_gt_i3d/{stem}.npz`.

## Submit

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
bash wan_experiment/sbatch/submit_v2v_lpips_fvd.sh
```

L40S. Installs `lpips` **`--no-deps`** into
`/scratch/$USER/pip-extras/lpips-nodeps` (conda keeps torch).
16737041 resolved a second CUDA torch into `pip-extras/lpips` —
scancel that job only and resubmit. Skip-existing LPIPS and FVD
features.

## Harvest

```bash
cd /scratch/wc3013/longcat-video-tta
python3 - <<'PY'
import json
from pathlib import Path
root = Path("/scratch/wc3013/longcat-video-tta/wan_experiment/results/v2v_panda_caption_128v")
print("| Method | n | PSNR | SSIM | LPIPS | FVD | last16 FVD |")
print("|---|---:|---:|---:|---:|---:|---:|")
for name in ("notta", "rolling_notta", "sf_pseudo", "sf_always_search"):
    s = json.loads((root / f"{name}_h30s_shard0/pixel_full/summary.json").read_text())
    fp = root / f"{name}_h30s_shard0/pixel_full/fvd.json"
    f = json.loads(fp.read_text()) if fp.is_file() else {}
    print(
        f"| {name} | {s.get('n')} | {s.get('psnr')} | {s.get('ssim')} | "
        f"{s.get('lpips')} | {f.get('fvd')} | {f.get('fvd_last16')} |"
    )
PY
```
