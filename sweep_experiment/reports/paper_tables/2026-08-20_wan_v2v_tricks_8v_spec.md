# V2V N=8 sampling-space tricks — submit spec

**Date:** 2026-08-20
**Series:** `v2v_panda_tricks_8v`
**Cluster:** `/scratch/wc3013/longcat-video-tta/wan_experiment/results/v2v_panda_tricks_8v`
**Protocol:** same as `v2v_panda_bakeoff_8v` (first 8 Panda clips, 9-latent real prefix, 6×21-latent tail).
**Baseline:** pair against bake-off `notta` / `seed_bon` via `--baseline-dir`.
**No TTC. Do not resubmit motion_bon, dead-tail backtrack, or shift/CFG.**
**Do not scancel** confirm jobs 16113805 / 16113806.

## Submit

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
TRICKS=1 bash wan_experiment/sbatch/submit_v2v_bakeoff.sh
```

2-way H200 cap: N=32 confirm already holds two slots. These six queue.

## Methods

| Method | What it tests | k | Pick | Expected wall (N=8) |
|---|---|---:|---|---|
| `hinge_bon` | H-match: motion hinge vs two-sided `seed_bon` | 4 | min `prefix_match_score` | ~51 min |
| `late_bon` | Search only if incoming motion < 0.7× prefix **or** last 2 chunks | 1–4 | two-sided (same as seed_bon) | ~25–51 min |
| `hist_drop` | History Guidance without CFG: full vs last-3 latents vs extra seeds | 4 | hinge | ~51 min |
| `good_backtrack` | Resample only if this chunk collapsed **and** previous commit was good (≥0.8× prefix) | 1+retry | n/a | ~20–30 min |
| `cached_bon` | Efficiency: replay KV once, snapshot, restore per seed. Same pick as seed_bon | 4 | two-sided | ≤51 min if snap works |
| `sink` | Replay prefix + last 21 latents only (sink+window approx; no rerope ckpt) | 1 | n/a | ~18 min |

## Promote rule (same lock)

Past N=8 only if median tail motion beats bake-off `notta` **and** IQ not worse by ≥1.0 **and** subject not worse by ≥0.02.

`hinge_bon` vs `seed_bon` is the causal test: if hinge wins, H-match is the story; if it ties, the N=8 win was “four seeds.”
`cached_bon` should **match** `seed_bon` quality. A quality gap is a snapshot bug, not a new method.
`sink` matching `notta` means local attention never saw the pinned prefix (need NVIDIA rerope ckpt).

## After generate

```bash
python wan_experiment/scripts/analyze_v2v_bakeoff.py \
  --series-dir /scratch/wc3013/longcat-video-tta/wan_experiment/results/v2v_panda_tricks_8v \
  --baseline-dir /scratch/wc3013/longcat-video-tta/wan_experiment/results/v2v_panda_bakeoff_8v
# official VBench, space-separated METHODS, no commas in --export
```
