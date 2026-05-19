# Next Actions

## Immediate Cluster Actions

1. Resume the six checkpointed 10-step discovery jobs:
   - Panda: `S10_LR001`, `S10_LR0025`, `S10_LR005`
   - UCF: `S10_LR001`, `S10_LR0025`, `S10_LR005`
2. Audit why UCF pointwise metrics are `nan` in raw summaries.
3. After 10-step summaries are complete, select Panda and UCF winners.
4. Submit retrieval-batch discovery only after the 200-video step/LR sweep is fully interpreted.

## Discussion After Initial Changes

- Gating implementation: choose between baseline-quality gate, TTA-loss-slope gate, or prompt-alignment gate.
- Horizon-aware objective: decide whether to implement multi-noise consistency first or rollout self-consistency.

## Rule

Only promote configs to 1000-video runs if they improve FVD and do not regress pointwise metrics on discovery.

