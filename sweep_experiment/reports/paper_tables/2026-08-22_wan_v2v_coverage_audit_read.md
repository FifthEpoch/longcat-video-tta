# Coverage audit read (2026-08-22 12:25)

Login-node sidecars. No new generate. Jobs **16209126–133** submitted
after this paste (128 + leftovers-8v).

## What holds

`rolling_notta` N=32 is still the only method that beats notta on
**median and still-prefix win-rate** without bit-matching notta:
tail +31%, still **15/6**, live 6/5, bit=notta **0**.

N=8 search “wins” are the lucky-8 trap. They die at 32:

| Method | N=8 tail | N=32 tail | N=32 still W/L | N=32 live W/L | bit=notta |
|---|---:|---:|---|---|---:|
| seed_bon | +34% | **−9%** | 7/14 | 5/6 | 0 |
| live_bon | +37% | +8% | 0/0 (skip) | **5/6** | **21** |
| appear_bon | +7% | +3% | 10/11 | 5/6 | 0 |
| quiet_bon | — | **−19%** | 7/14 | 0/1 | 10 |
| rolling_notta | +29% | **+31%** | **15/6** | 6/5 | 0 |

`sink` and `noise_probe` are exact notta (8/8). `late_bon` −10%
(already closed). `prefix_sink` +83% stays closed on IQ.

## Spearman(prefix, Δtail)

N=8 search is **live-recovery**: seed/live/hist ρ ≈ +0.8…+0.9.
That sign **dies at 32** (seed +0.08, live −0.12, appear +0.01).
rolling N=32 ρ = **−0.11** — the host helps stills more than hots.
Do not build a “search when live” gate on the N=8 ρ.

## Idea 9 router — the prefix rule is worse than always-rolling

Paired N=32 tails already on disk:

| Policy | tail median | vs notta |
|---|---:|---:|
| notta | 0.0135 | — |
| **always rolling** | **0.0178** | **+31%** |
| still→notta, live→rolling | 0.0148 | +9% |
| oracle best-of-arms | 0.0197 | +46% |

The rule we wrote **throws away the still-prefix rolling wins**.
Always-RF is the deployable host policy. Oracle +46% is **not**
deployable (it picks seed 4× and appear 5×).

## What did **not** resolve

- **Idea 7 trust resim:** 0 search chunks. Script looked for
  `cands`; sidecars use `candidates`. Not a scientific result.
  Re-run after the key fix; still not a 30 s tail.
- **Idea 3 U_t:** `noise_probe` sidecars have no `eps_mean_abs`.
  Cannot say U_t is flat from this paste.

## Do not do

Do not gate rolling on `prefix_motion ≥ 0.012`. Do not scale
leftovers from N=8. Do not treat the oracle as a method.
