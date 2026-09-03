# Non-linear Rolling Forcing timestep list — SUBMIT-READY (2026-09-03)

Inference smoke on the **existing** Rolling Forcing student.
Not leftover ρ. ρ scaled injected Gaussian. This list is
`c_noise` / adaLN. Same T as the live `denoising_step_list`.

## Do we need Distribution Matching Distillation (DMD)?

**Probably yes for it to work.** Distilled few-step models
are tied to the noise levels they saw. `sf_roll` twitched.
Caption leftover ρ killed Imaging Quality without changing
the list. Stream Forcing trains a path *to* the inference
schedule.

**We still smoke without DMD first.** That is the only way
to know. If linger-high / dump-early hold Imaging Quality
and Subject Consistency versus caption Rolling Forcing
first-8, we got a cheap test-time move. If Imaging Quality
dies (likely), that *is* the evidence that a short
Distribution Matching Distillation (DMD) is required — then
a go/no-go on 8-GPU training, not the first GPU.

Do not start a student tonight.

## Arms

Host = existing `v2v_panda_caption_32v/rolling_notta` first 8.
Do **not** remake native Rolling Forcing or cite-128.

| Method | List if native is `[1000,800,600,400,200]` | Shape |
|---|---|---|
| `rolling_linger` | 1000, 920, 800, 520, 200 | stay noisy, dump at the end |
| `rolling_dump` | 1000, 520, 360, 260, 200 | jump down, linger near-clean |

If native is four-step `[1000,750,500,250]`: linger
`1000,875,650,250`; dump `1000,500,350,250`. Other T: keep
endpoints, warp interior (`u^2` / `u^0.5`). Slurm log must
print `rf_step_list native=... used=...`.

N=8, `metadata_csv`, k=1. Cite versus caption Rolling Forcing
first-8 (not analyzer-versus-Self-Forcing). Bars: tail versus
host; Imaging Quality not worse by ≥1.0; Subject Consistency
not worse by ≥0.02.

## Submit

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
bash wan_experiment/sbatch/submit_v2v_caption_schedule8.sh
```

Cancel this wave only (print the JobIDs from submit):
`scancel <linger> <dump> <vbench>`.
Do not scancel leftover 16734909–913 (already DONE).
Do not scancel Learned Perceptual Image Patch Similarity
(LPIPS) **16738784** unless that job already finished.

## Harvest

8/8 + `metadata_csv` + first sidecar `denoising_step_kind`
`linger` / `dump` + Visual Benchmark (VBench) full clip.
Pair tails versus caption Rolling Forcing first-8.
