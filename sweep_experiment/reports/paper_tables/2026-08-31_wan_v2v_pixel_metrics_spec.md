# Caption 128 paired pixel metrics (2026-08-31)

**Status:** duration audit **DONE**. Scorer ready. Not yet submitted.

Source lengths (imageio, self_forcing env): **n=128 / 0 err**,
min **54.8 s**, median **314 s**, max **1824 s**. `ge_32s=128/128`.
`ge_120s=113/128`. Every cite clip has a real 30 s leftover.

## Protocol

- Opening used at generate time: first **33** source frames
  (`1+4×8`). Do not score that prefix (VAE round-trip).
- Invented tail: generated mp4 after `prefix_pix`, **16 fps**.
- GT tail: source after frame 33, **same duration in seconds**,
  resampled to 16 fps and to the generated H×W.
- Cite **medians**. PSNR / SSIM on every aligned frame. LPIPS
  AlexNet every 8th frame (plus last).
- Official headline stays full-clip VBench + Dyn%. These numbers
  are the LongCat-style reconstruction check. They can disagree.

Do not compare the tail to the opening. That is Prefix-match.

## Submit (no new videos)

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
bash wan_experiment/sbatch/submit_v2v_pixel128.sh
```

L40S job. Skip-existing per-video json. Methods: Self Forcing,
Rolling, Pseudo, Always. FVD is a second step (`eval_fvd.py
--force` on aligned tails) after this lands — I3D wants 16-frame
clips; do not feed it the full mp4 or it will score the prefix.

Cancel that job only. Do not scancel **16674378**.
