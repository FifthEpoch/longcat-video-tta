# In-chunk OOM is still GPU KV clones (2026-08-30)

**Tails:** 16546045 / 048 (SF intra), 16546059 (SF restep),
16546053 (RF bpseudo). All H200 `total: 150110011392`. Free
70 MB–1.6 GB. Failed allocs 331 MB / 417 MB / 562–658 MB /
1.65 GB / 2.1 GB. Same family as 16471675.

The one-live-snap fix (`2d84a74`) dropped k=4 post-block GPU
clones. Peak was still **pre-block snap + live cache + post-block
snap** on device. Used-prefix ≈ 40 GB late in 30 s → three copies
miss 141 GB. SF restep holds `mid_kv` + winner snap + live. RF
bpseudo called `_initialize_kv_cache` while the rolling list was
still referenced (second full RF cache) and cloned the whole
`output` each cand.

**Fix (this turn):** `_snapshot_kv` / `_snap_kv` copy used tensors
to CPU (view → host, no extra GPU clone). `_rf_replay_clean`
resets in place via `_reset_caches`. `rf_bpseudo` reuses live
`output` + last-3 save. Score decode unchanged (do not retune
1.5×).

Smoke only. Scored siblings are already **NO**. Do not expect a
letter pass. Do not relaunch RF intra.

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
SMOKE=1 WAVE=sf bash wan_experiment/sbatch/submit_v2v_intra8.sh
SMOKE=1 WAVE=crash bash wan_experiment/sbatch/submit_v2v_denoise8.sh
```

N=8 only if smoke writes mp4. Paper-size next is still `WAVE=cite`.
