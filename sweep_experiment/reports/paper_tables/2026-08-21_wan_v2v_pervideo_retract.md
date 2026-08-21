# V2V per-video pairing — retract the N=32 −26% kill

**Date:** 2026-08-21
**Supersedes:** the N=32 “seed_bon −26% KILL” line in
`2026-08-21_wan_v2v_confirm_32_and_tricks.md`. That median compared
**unpaired** `summary.json` rows. Many rows are `skipped: true` (preempt
/ requeue) and have no `tail_motion`. Printed `n=32` was ok-row count,
not scored-row count.
**Does not supersede:** the N=8 tricks generate numbers (those 8 sidecars
are complete).

## 1. Confirm N=32 — only 12 videos actually paired

Intersection of rows that have `tail_motion` on both sides:
**panda_0020–0031 only.** panda_0000–0019 are missing (skip stubs).

| | notta | seed_bon | seed−notta |
|---|---:|---:|---:|
| paired N | 12 | 12 | |
| median tail motion | 0.01380 | **0.01424** | **+3%** |
| seed_bon > notta | | | **5/12** |

| video | notta | seed_bon | Δ |
|---|---:|---:|---:|
| panda_0020 | 0.00646 | 0.00656 | +0.00010 |
| panda_0021 | 0.00981 | 0.00856 | −0.00125 |
| panda_0022 | 0.03579 | 0.02678 | −0.00901 |
| panda_0023 | 0.01368 | 0.01406 | +0.00038 |
| panda_0024 | 0.01093 | 0.01442 | +0.00350 |
| panda_0025 | 0.02637 | 0.02449 | −0.00188 |
| panda_0026 | 0.01392 | 0.02287 | +0.00895 |
| panda_0027 | 0.03519 | 0.02720 | −0.00799 |
| panda_0028 | 0.02026 | 0.01472 | −0.00554 |
| panda_0029 | 0.00877 | 0.01064 | +0.00187 |
| panda_0030 | 0.01066 | 0.00871 | −0.00196 |
| panda_0031 | 0.01618 | 0.00922 | −0.00696 |

Honest read: **tie / coin-flip on the 12 we can pair.** The earlier
0.01018 seed_bon median is a different (larger, unmatched) subset and
must not be compared to notta 0.01380. **Retract KILL. Retract PROMOTE.
Status: incomplete until sidecars for 0000–0019 are folded in.**

32 mp4s exist in both dirs. Per-video `{stem}.json` written on first
finish should still have `tail_motion`. Rebuild the pair from sidecars,
not from `summary.json`.

## 2. N=8 hist_drop is not one clip

Same 8 as the bake-off. Complete.

**hist_drop vs notta:** 6/8 win, median 0.01675 → 0.02377 (+42%).

| video | notta | hist_drop | Δ | seed_bon | hist−seed |
|---|---:|---:|---:|---:|---:|
| panda_0000 | 0.01486 | 0.02761 | +0.01275 | 0.02761 | 0 |
| panda_0001 | 0.00641 | 0.01048 | +0.00406 | 0.00933 | +0.00114 |
| panda_0002 | 0.02350 | 0.01958 | −0.00393 | 0.01787 | +0.00171 |
| panda_0003 | 0.02236 | 0.01993 | −0.00243 | 0.01917 | +0.00076 |
| panda_0004 | 0.03062 | 0.03368 | +0.00306 | 0.03061 | +0.00307 |
| panda_0005 | 0.00686 | 0.01101 | +0.00416 | 0.00671 | +0.00430 |
| panda_0006 | 0.01864 | 0.02967 | +0.01104 | 0.02584 | +0.00384 |
| panda_0007 | 0.01210 | 0.02856 | +0.01646 | 0.02617 | +0.00239 |

Losses vs notta are the two already-hot prefixes (0002, 0003). The +42%
is three large lifts (0000/0006/0007) plus three small ones — not a
single outlier.

**hist_drop vs seed_bon:** 7/8 (0000 exact tie). Every other clip is a
small consistent bump (+0.0008 to +0.0043). History dropout is an
increment on the seed picker, not a different mode.

**hinge vs notta:** 5/8, +11% median. 0000 matches hist_drop/seed_bon
exactly (0.02761). 0004 is a large loss (−0.013). Hinge is not better
than two-sided on this set.

## 3. What is still true / what is not

Still true:
- `cached_bon` = bake-off seed_bon (KV snap works).
- `sink` = bake-off notta (replay sink is a no-op).
- late_bon / good_backtrack lost on generate metrics.
- hist_drop is the only new motion win, and it is **broad on these 8**.

Not true (retracted):
- “N=32 seed_bon −26% killed four-seed search.”

Open:
- True paired N=32 from sidecars (0000–0019).
- Full-clip VBench on hist_drop (IQ / subject / Dyn). Do not scale
  hist_drop to 32 until that lands.

## 4. Sidecar rebuild (run on cluster)

```bash
cd /scratch/wc3013/longcat-video-tta
python3 <<'PY'
import json, statistics
from pathlib import Path

def sidecar_rows(series, method):
    d = Path(f"wan_experiment/results/{series}/{method}_h30s_shard0")
    out = {}
    for p in sorted(d.glob("*.json")):
        if p.name == "summary.json":
            continue
        rec = json.loads(p.read_text())
        if not rec.get("ok") or rec.get("tail_motion") is None:
            continue
        key = rec.get("file_name") or rec.get("stem") or p.stem
        out[key] = float(rec["tail_motion"])
    return out

nb = sidecar_rows("v2v_panda_confirm_32v", "notta")
sb = sidecar_rows("v2v_panda_confirm_32v", "seed_bon")
keys = sorted(set(nb) & set(sb))
print(f"sidecar paired n={len(keys)}  notta_sidecars={len(nb)} seed_sidecars={len(sb)}")
if keys:
    nt = [nb[k] for k in keys]
    st = [sb[k] for k in keys]
    wins = sum(1 for a,b in zip(nt,st) if b>a)
    print(f"med {statistics.median(nt):.5f} -> {statistics.median(st):.5f}  seed>notta {wins}/{len(keys)}")
    for k in keys:
        print(f"  {k}: {nb[k]:.5f} -> {sb[k]:.5f}  d={sb[k]-nb[k]:+.5f}")
PY
```
