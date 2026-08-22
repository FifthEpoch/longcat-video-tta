# rolling-128 scale-up (2026-08-22)

`rolling_notta` is the **only** method that passed the N=32 locked
bars. It is the Rolling Forcing **host**, not a sampling-space
controller. `appear_bon` / `live_bon` / `seed_bon` stay closed.

## What ran at N=32

| Method | tail median | mean Δ | win/loss | IQ | Subj | Dyn | Call |
|---|---:|---:|---|---:|---:|---:|---|
| notta (SF) | 0.0135 | — | — | 69.65 | 0.665 | 0 | baseline |
| rolling_notta | 0.0178 (+31%) | +13% | 21/11 | +0.79 | +0.037 | 0 | **YES on locked bars** |

## Scale

Series: `v2v_panda_rolling_128v`. First **128** of `panda_1000_480p`
(N=32 is a prefix). Fresh SF `notta` + `rolling_notta` in the same
series. Do **not** reuse confirm_32v (no notta on videos 32–127).

Same protocol: 9-latent prefix, 6×21 tail, seed 0, k=1, no search.

32 clips ≈ 27 min → 128 ≈ 2 h per method. 2-way H200 runs them
together. VBench full afterany (~2–3 h).

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
bash wan_experiment/sbatch/submit_v2v_rolling128.sh
```

## Pre-registered kill (do not cite analyzer PROMOTE)

YES only if **all** hold on the honest 128-way sidecar pair:

1. median tail > notta **and** mean tail > notta
2. win/loss majority (wins > losses)
3. still prefixes (`prefix_motion < 0.012`) not mass-damped (still
   win-rate ≥ 0.5)
4. IQ ≥ notta−1.0 and subject ≥ notta−0.02
5. Dyn 0/0 does **not** decide (notta already 0)

If median holds but mean or win-rate flips, **NO**. Do not retcon.
Do not add search on top of this host until 128 is read.

200 / 1000 wait on this gate.
