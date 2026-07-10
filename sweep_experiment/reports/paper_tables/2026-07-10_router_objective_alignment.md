# Router objective alignment — VBench vs PSNR @ N=200

**Date:** 2026-07-10  
**Series:** `panda_ood_budget_pilot` · same 9-d Block A features  
**Cluster path:** `per_video_analysis/2026-07-06/router_objective_alignment/summary.md`  
**Script:** `scripts/analyze_router_objective_alignment.py`

## Question

How much do VBench-targeted and PSNR-targeted routers **agree** on the chosen config per video?

## Pick overlap

| Metric | Value |
|--------|------:|
| **Router pick agreement** (same config) | **12.5%** (25/200) |
| Oracle agreement (VB oracle = PSNR oracle) | 15.0% |
| When oracles agree → routers agree | 10.0% |
| VB router ↔ VB oracle match | 18.5% |
| PSNR router ↔ PSNR oracle match | 15.5% |
| Config-set Jaccard (configs ever used) | 0.75 |

## When routers disagree (175 videos)

| Check | Result |
|-------|--------|
| VB-router pick ≥ PSNR-router pick on **realized VBench** | 90/175 (**51.4%**) |
| PSNR-router pick ≥ VB-router pick on **realized PSNR** | 97/175 (**55.4%**) |

## Realized-metric correlation (per video)

| ρ | Value |
|---|------:|
| VBench(VB pick) vs VBench(PSNR pick) | **0.995** |
| PSNR(PSNR pick) vs PSNR(VB pick) | **0.987** |

## Headline findings

1. **Low overlap in config labels (12.5%).** VB and PSNR routers pick the **same** step×LR only 1/8 of the time. Oracle configs agree even less (15%) — the objectives are genuinely different at the config level.

2. **Near-identical realized outcomes (ρ≈0.99).** Despite different picks, per-video VBench and PSNR from the two routers' choices are almost perfectly rank-correlated. On this 12-config grid, many configs are **metrically redundant** per video — disagreements are mostly label swaps within a flat local landscape.

3. **Weak specialization when disagreeing.** Each router beats the other on its own metric only ~51–55% when picks differ (not ~80%+). Neither router is sharply identifying a unique optimum — consistent with flat per-video metric curves.

4. **Shared config palette (Jaccard 0.75).** Both routers draw from similar high-LR / mid-step configs (`S20_LR1e2`, `S10_LR1e2`, `S2_LR1e2` dominate crosstab). Top agreeing pair: **`S20_LR1e2` → `S20_LR1e2`** (22 videos).

## Paper narrative

- **Config space:** VBench vs PSNR routing **do not align** (12.5% pick agreement).
- **Outcome space:** They **mostly agree** on what you get (ρ≈0.99) — explains why VB router barely moves PSNR (+0.009 dB) while still capturing VB headroom: alternate picks are often metrically equivalent.
- **Implication:** Gains come from moving off fixed S10 into a **better region** of the grid; fine-grained objective choice matters less than coarse routing. Pick objective (VB vs PSNR) mainly shifts **which equivalent-ish config** you land on.

## Related results

| Router | VB cap % | PSNR Δ vs fixed | PSNR cap % |
|--------|---------:|------------------:|-----------:|
| VBench-targeted | **20.8%** | +0.009 dB | 1.2% |
| PSNR-targeted | 5.6% | **+0.054 dB** | **7.2%** |

Different picks, similar outcomes on average — but PSNR-targeted router still shifts the **mean** PSNR upward because its picks systematically favor slightly higher-PSNR configs in aggregate.

## Reproduce

```bash
bash scripts/run_router_objective_alignment.sh
```
