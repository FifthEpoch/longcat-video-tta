# rolling-128 VBench 6/7 (2026-08-23 20:55)

Login join of per-dim files after **16228045** died at flickering.
GPU resume **16259396** (4 h wall, skip-existing). Subject n=128
(the 09128 91/128 file was completed, not left incomplete).

Cite medians. Official VBench++ is the **full clip**. Flickering
is still missing — do not publish a 7-dim table yet.

## Locked bars — PASS (flicker outstanding)

| Clause | N=128 rolling vs SF notta | Call |
|---|---|---|
| median tail > notta | 0.0177 vs 0.0136 (+31%, 88/40) | **Yes** |
| IQ ≥ notta − 1.0 | 70.91 vs 70.20 (**+0.71**) | **Yes** |
| subject ≥ notta − 0.02 | 0.687 vs 0.648 (**+0.039**) | **Yes** |

Analyzer `PROMOTE` is vs SF notta. That is the correct baseline
here. This is still **someone else’s host**, not our controller.

## Full-clip VBench (6/7)

| Method | N | subject | background | aesthetic | IQ | smoothness | dynamic | flicker |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| notta | 128 | 0.6482 | 0.8049 | 0.5073 | 70.20 | 0.9919 | **0.000** | 0.9858 |
| rolling_notta | 128 | **0.6871** | 0.8093 | **0.5403** | **70.91** | 0.9910 | **1.000** | — |

Rolling mean Dyn **0.531** ⇒ about 68/128 clips are VBench-dynamic.
N=32 forward rolling was Dyn **median 0**. That line does **not**
automatically carry. First 32 of this 128 series bit-matched the
forward-32 **tails**; split Dyn on `joined.json` before rewriting
“RF preserves freeze.”

N=32 rolling (forward series, for scale): subj 0.702 / IQ 70.44 /
Dyn 0. The extra 96 pull subject down a little (0.702→0.687) and
IQ stays up.

## What this is not

- Not a reason to scale `sf_roll` / `rf_chunk` (those also Dyn 1,
  but subject **fails vs this host**).
- Not `rf_recache` / leftovers / H2 / H3.
- Not “our TTA.” Native RF schedule, k=1.

## Next

Wait for **16259396**. Then re-join (skip will add flickering) and
cite 7/7. Optional login split while it runs:

```bash
python3 - <<'PY'
import json
from pathlib import Path
from statistics import mean, median
p = Path("wan_experiment/results/v2v_panda_rolling_128v/rolling_notta_h30s_shard0/vbench_full/joined.json")
rows = sorted(json.loads(p.read_text())["per_video"],
              key=lambda r: r.get("file_name") or r.get("stem") or "")
vals = [r.get("vbench", {}).get("dynamic_degree") for r in rows]
vals = [v for v in vals if v is not None]
print("n", len(vals), "n_dyn", sum(v >= 0.5 for v in vals),
      "mean", mean(vals), "med", median(vals))
for name, sl in ("first32", vals[:32]), ("last96", vals[32:]):
    print(name, "n_dyn", sum(v >= 0.5 for v in sl),
          "mean", mean(sl), "med", median(sl))
PY
```

Cancel flickering only: `scancel 16259396`.
