# Next Actions

## Immediate Cluster Actions

1. Decide whether to promote the 200-video winners to 1000-video validation:
   - Panda standard: `S10_LR005`
   - UCF standard: `S5_LR0025` as balanced candidate, `S5_LR001` as FVD-only candidate
2. Audit why UCF pointwise metrics are `nan` in raw summaries while exporter pointwise values are finite.
3. If revisiting anchor gating, fix the failed `G_OFF` controls and tune thresholds from observed anchor-improvement quantiles.
4. Discuss horizon-aware objective implementation details before adding new method code.
5. Submit retrieval-batch discovery only after deciding whether the 1000-video validation should run first.

## Discussion After Initial Changes

- Anchor gating result: simple binary/soft anchor gates did not improve the 200-video Pareto frontier; keep as diagnostic unless we tune thresholds from quantiles.
- Horizon-aware objective: decide whether to implement multi-noise consistency first or rollout self-consistency.

## Rule

Only promote configs to 1000-video runs if they improve FVD and do not regress pointwise metrics on discovery.

