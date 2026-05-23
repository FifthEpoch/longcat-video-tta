# Next Actions

## Immediate Cluster Actions

1. Wait for the four 1000-video validation jobs to finish, then merge/log Panda `S10_LR005` and UCF `S5_LR0025`.
2. Inspect failed retrieval-batch `K5`/`K10` SLURM logs before resubmitting retrieval jobs.
3. Decide whether Panda `AREG02` deserves a 500-video or 1000-video validation run after current 1000-video validation finishes.
4. Audit why UCF pointwise metrics are `nan` in raw summaries while exporter pointwise values are finite.
5. If revisiting anchor gating, fix the failed `G_OFF` controls and tune thresholds from observed anchor-improvement quantiles.

## Discussion After Initial Changes

- Anchor gating result: simple binary/soft anchor gates did not improve the 200-video Pareto frontier; keep as diagnostic unless we tune thresholds from quantiles.
- Horizon-aware objective result: Panda anchor regularization (`AREG02`) is promising at 200 videos; UCF anchor regularization is not.

## Rule

Only promote configs to 1000-video runs if they improve FVD and do not regress pointwise metrics on discovery.

