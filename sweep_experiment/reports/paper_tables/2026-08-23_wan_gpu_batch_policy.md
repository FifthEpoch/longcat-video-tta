# GPU policy — pack H200 or leave it (2026-08-23)

User request: use an H200 to the fullest with batching; if we
cannot batch, request a lesser GPU.

## Why candidate tensor-batch is off

V2V 30 s allocates a **137-frame KV cache** (~39 GB bf16,
`enlarge_kv_cache`, 48 GB safety cap). Search k=4 would need
four copies of that cache. 4 × 39 GB misses a 141 GB H200.
KV `global_end_index` is also a scalar — the official cache is
built for one sequence at a time.

Do **not** shrink that cache to fake a batch. H1 already showed
host/sampler mismatch twitches.

## What we do instead

| Job | GPU | Why |
|---|---|---|
| V2V generate (`run_v2v_chunked.sbatch`) | **H200** + `VIDEO_WORKERS=2` | KV needs >48 GB. Two independent videos share one card (same pixels as serial). MPS on. |
| VBench (`run_i2v_vbench.sbatch`) | **L40S** (`--constraint=l40s`) | CLIP/DINO/RAFT. Must not sit on the 2-way H200 cap. |
| 16259396 flickering | H200 (already queued) | Do not resubmit. |

L40S is 48 GB. Generate cannot move there while the 137-frame
KV stays. `torch_pr_36_mren` is the mren L40S stakeholder
account — VBench belongs on that pool.

If `VIDEO_WORKERS=2` OOMs, resubmit that method with
`VIDEO_WORKERS=1`. Do not retune the method.

NYU `gh*` (H200) cancels jobs under **60%** GPU util. Packing
two videos is also the utilization fix.

## Family wave

Same paste as before. Generate jobs pick up workers=2 from
`run_v2v_chunked.sbatch`. VBench is L40S afterany.
