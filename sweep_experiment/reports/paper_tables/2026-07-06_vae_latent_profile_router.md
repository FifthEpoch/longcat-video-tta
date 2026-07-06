# VAE latent profile router @ N=200 (2026-07-06)

**Series:** `panda_ood_budget_pilot` · **Features:** 130-d VAE ctx/tgt/full pools + probe  
**Baseline:** exp7 ridge Phase-0 + probe @ **12.8%** captured  
**Cluster:** `per_video_analysis/2026-07-06/vae_latent_profile_router/`

| Router | # features | Captured % | Oracle match % | Δ vs exp7 |
|---|---:|---:|---:|---:|
| baseline_exp7 (Phase-0 + probe) | 47 | **12.8** | 21.5 | — |
| vae_profile_probe (VAE + probe) | 138 | 12.2 | 16.5 | −0.6 |
| vae_profile_full (all stacked) | 177 | 4.2 | 17.5 | −8.6 |

## Conclusion

Rich **hand-pooled LongCat-VAE latents do not improve** honest OOF VBench-total routing. Stacking 177 features at N=200 **overfits** (4.2%). Native VAE representation path **closed** for routing.

**Remaining ceiling (non-deployable):** GT quality on probe outputs ~17–18% (exp10 / exp14_full). Next lever if pursued: **learned verifier on probe mp4s**, not more VAE CSV dims.
