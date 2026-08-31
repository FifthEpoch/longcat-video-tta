# Pseudo-next N=8 — cheapen + re-gate (IN FLIGHT)

Jobs (2026-08-31 16:24): smoke **16679371–375**; N=8 **16679376–379**
+ VBench **16679380**. Do not scancel **16674378** or **16678705**.

User 16:17: implement **both** upgrades from
[`2026-08-31_wan_pseudo_next.md`](2026-08-31_wan_pseudo_next.md)
and fire first. Do **not** wait for Always-128 official
(**16674378**). Do **not** jump to 128. Caption N=8.

γ stays **0.0**. k stays **4**. Pick stays motion+trust
(`sick_motion`). No RF twins (Pseudo-on-RF gate was dead).
No mid-chunk rewrite. No TTC. No I2V.

## Methods (same wave)

| Code | What | Twin |
|---|---|---|
| `sf_pseudo_cached` | Once-on-opening gate (same as `sf_pseudo`). If fire, k=4 with CachedSearch KV snap. | `sf_always_cached` |
| `sf_always_cached` | Always k=4 CachedSearch. Same pick. | — |
| `sf_repseudo` | Re-hold-out last 3 committed latents before **every** chunk. Chunk 0 = current prefix gate. | caption-32 `sf_always_search` (do not remake) |
| `sf_repseudo_cached` | Re-gate + CachedSearch when it fires. | `sf_always_cached` |

Do **not** remake `sf_pseudo` or `sf_always_search`. Cite first 8 of
`v2v_panda_caption_32v`.

Series: `v2v_panda_caption_pseudo_next_8v`.
Baseline: caption-32 `notta` first 8 — subject **0.700** / IQ **71.54**
/ tail **0.0129**. Keep letter if promoting: subject ≥ 0.68, IQ ≥ 70.5,
tail or Dyn% beats SF.

## Gate (re-eval)

A = `hist_end − 3`. Generate B. MAE vs real B. Fire iff an extra
seed beats cand0 by more than γ=0. Cite `chunks[i].pseudo_fire` and
`pseudo_rows[].hist_end`. Top-level sidecar `pseudo_fire` is not
the harvest key.

Probe replays to A once, then snaps KV (same CachedSearch pay as
the fired path). Not a pixel rewrite.

## Cost read

CachedSearch is how we *pay* for a fired clip, not a new picker.
`sf_pseudo_cached` vs caption-32 `sf_pseudo` is the cheapen test.
`sf_repseudo` vs `sf_pseudo` is the skip-recovery test. Stacked
arm is `sf_repseudo_cached`.

## Submit

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
SMOKE=1 bash wan_experiment/sbatch/submit_v2v_pseudo_next8.sh
bash wan_experiment/sbatch/submit_v2v_pseudo_next8.sh
```

Do not scancel **16674378** or **16678705**.
