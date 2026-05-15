# Next Actions

## Immediate Implementation

1. Create deterministic 200-video subsets:
   - `datasets/panda_200_480p`
   - `datasets/ucf101_200_480p`
2. Submit standard/short-horizon AdaSteer discovery sweeps on both datasets.
3. Submit retrieval-batch discovery only after the 200-video eval sets exist.

## Discussion After Initial Changes

- Gating implementation: choose between baseline-quality gate, TTA-loss-slope gate, or prompt-alignment gate.
- Horizon-aware objective: decide whether to implement multi-noise consistency first or rollout self-consistency.

## Rule

Only promote configs to 1000-video runs if they improve FVD and do not regress pointwise metrics on discovery.

