# Next Actions

## Immediate Cluster Actions

1. Wait for the four 1000-video validation jobs to finish, then merge/log Panda `S10_LR005` and UCF `S5_LR0025`.
2. Inspect failed retrieval-batch `K5`/`K10` SLURM logs before resubmitting retrieval jobs.
3. Decide whether Panda `AREG02` deserves a 500-video or 1000-video validation run after current 1000-video validation finishes.
4. Audit why UCF pointwise metrics are `nan` in raw summaries while exporter pointwise values are finite.
5. If revisiting anchor gating, fix the failed `G_OFF` controls and tune thresholds from observed anchor-improvement quantiles.
6. **NEW (May 23, 2026, validation):** Run `scripts/recompute_fvd_fid_from_stats.py` against the four long-context Panda 999v runs (No-TTA, AdaSteer S10, LoRA R8, TinyLoRA LAST24). Cluster command sequence is in `sweep_experiment/reports/experiment_metrics_log.md` under "May 23, 2026 - FVD/FID Chunked-Merge Validation". The local numerical unit test (`scripts/test_chunked_fvd_equivalence.py`) already proved the merge math is identical to single-pass FVD/FID to 1e-15 relative precision; this cluster step is the implementation-level check against the actual stored sufficient statistics. Acceptance: <1e-4 relative agreement between recomputed FVD/FID and `merged_summary.json` for all four runs.
7. **NEW (May 23, 2026, Phase A):** After step 6 confirms the +5.4 FVD regression is real (not an artifact), run the long-horizon failure-mode diagnostic on the corrected 999-video Panda long-context No-TTA vs AdaSteer S10 runs (`scripts/diagnose_long_horizon_failures.py`) to identify caption themes where AdaSteer regressed. Findings drive horizon-aware config design in Phase B below.

## Phase A: Long-Horizon Failure Diagnostics (Sat May 23 → Sun May 24)

Goal: find which caption themes / quality buckets drive the +5.4 FVD regression on long-context Panda 999-video so that subsequent AdaSteer variants can be designed against them rather than guessed at.

Per-video data:
- `notta-dir`: long-context Panda 999v No-TTA run output (chunk_N/summary.json).
- `treat-dir`: long-context Panda 999v AdaSteer S10 run output (chunk_N/summary.json).
- `dataset-dir`: `datasets/panda_1000_480p/` (provides metadata.csv with captions).

Cluster command (after `git pull`):

```bash
cd $LONGCAT_REPO
python scripts/diagnose_long_horizon_failures.py \
  --notta-dir /path/to/results/panda_long_1000v_notta \
  --treat-dir /path/to/results/panda_long_1000v_ada_s10 \
  --dataset-dir /path/to/datasets/panda_1000_480p \
  --out-csv sweep_experiment/reports/long_horizon_failure_panda_1000v.csv \
  --treat-label ada_s10 --top-k 25 \
  | tee sweep_experiment/reports/long_horizon_failure_panda_1000v.txt
```

Then push the CSV + txt back to GitHub for local theme aggregation and copying into `experiment_metrics_log.md`.

What to look for in the printout:
- Themes (sport, dance_music, cooking, nature, animal, vehicle, talking_head, crowd, indoor_misc, other) whose mean `dPSNR` or `dSSIM` is strongly negative or whose mean `dLPIPS` is strongly positive. These are the failure modes AdaSteer's long-horizon residual collides with.
- Quintile buckets on No-TTA PSNR: if AdaSteer hurts the lowest-PSNR quintile most, the issue is "easy" videos with little conditioning signal; if it hurts the highest-PSNR quintile most, the issue is overfitting to clean conditioning frames.

## Phase B: Long-Horizon AdaSteer Configs (Pending Phase A)

Designs we have *plausible* a-priori hypotheses for; the actual selection will be informed by Phase A and we'll discuss the full sweep before submitting any cluster jobs.

- **Conditioning-window-aware step schedule:** scale `delta_steps` and learning rate by the ratio of conditioning frames to total horizon (e.g., fewer steps when conditioning is a smaller fraction of the predicted future).
- **Anchor regularization at long horizon:** repeat the 200-video Panda `AREG02` recipe on the long-context 28-frame -> 93-frame setting; we already added `--anchor-reg-weight` support and saw a Panda 200v FVD win at standard horizon. The mechanism (differentiable fixed-sigma anchor loss) is exactly the regularization that should help long-horizon stability.
- **Theme-conditioned residual:** if Phase A identifies one or two themes where AdaSteer reliably hurts, train a small "skip if theme matches" CLIP gate. Less attractive than horizon-aware regularization because it's bandage-shaped, but easy and cheap.
- **Multi-noise / multi-sigma objective:** already implemented via `--anchor-reg-noise-draws`. We have not yet swept this *at long horizon*. Strong candidate after Phase A.

## Discussion After Initial Changes

- Anchor gating result: simple binary/soft anchor gates did not improve the 200-video Pareto frontier; keep as diagnostic unless we tune thresholds from quantiles.
- Horizon-aware objective result: Panda anchor regularization (`AREG02`) is promising at 200 videos; UCF anchor regularization is not.

## Rule

Only promote configs to 1000-video runs if they improve FVD and do not regress pointwise metrics on discovery.

