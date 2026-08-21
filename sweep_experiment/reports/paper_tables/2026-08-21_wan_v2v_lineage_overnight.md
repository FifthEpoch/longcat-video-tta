# V2V lineage overnight — all remaining tests (2026-08-21)

**Status:** implemented, submit-ready. User is out; one cluster command.

Same V2V protocol: Wan family, 9-latent Panda prefix, 6×21 gen latents
(~30 s), N=8 (first 8 of the bake-off set). No TTC. No hist_drop-32.

```bash
cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
bash wan_experiment/sbatch/submit_v2v_lineage.sh
```

Series: `wan_experiment/results/v2v_panda_lineage_8v/`
Compare tails to bake-off `notta` (`v2v_panda_bakeoff_8v`). Cite medians.

## Jobs (all at once)

| Method | Student | What | k | GPU dep |
|---|---|---|---:|---|
| `live_bon` | SF DMD | search iff `prefix_motion ≥ 0.012` | 4 | immediate |
| `live_hist` | SF DMD | live gate + hist_drop candidates | 4 | immediate |
| download | CPU | LongLive v1.0 + RF clones + ckpts + peft | — | immediate |
| `longlive_notta` | LongLive-1.3B | trained sink=3 / window=12, k=1 | 1 | after download |
| `longlive_sink` | LongLive-1.3B | prefix+window replay (sink is trained-in) | 1 | after download |
| `longlive_prefix_sink` | LongLive-1.3B | notta with `sink_size=9` (whole prefix pinned) | 1 | after download |
| `longlive_live_bon` | LongLive-1.3B | live_bon on LongLive | 4 | after download |
| `rolling_notta` | Rolling Forcing | native rolling window after prefix | 1 | after download |
| VBench full | — | all of the above + bake-off notta | — | afterany generate |

2-way H200 cap: extras queue. SF pair starts now; host jobs wait on the
~14 GB download then serialize.

## Read when you get back

```bash
python wan_experiment/scripts/analyze_v2v_bakeoff.py \
  --series-dir wan_experiment/results/v2v_panda_lineage_8v \
  --baseline-dir wan_experiment/results/v2v_panda_bakeoff_8v \
  --allow-partial
```

Promote past N=8 only if tail motion beats bake-off notta **and** IQ not
worse by ≥1.0 **and** subject not worse by ≥0.02. Use sidecars, not
unpaired `summary.json` medians.

## What each test answers

1. **live_bon** — was quiet_bon’s sign the only bug?
2. **live_hist** — does hist_drop’s extra candidate still help when we
   only search living prefixes?
3. **longlive_notta** — is 30 s Dyn collapse a Self-Forcing DMD fact or
   a causal-1.3B fact?
4. **longlive_sink / prefix_sink** — does a *trained* sink + pinning the
   real prefix move pixels? (replay-sink on vanilla SF was a no-op)
5. **longlive_live_bon** — is the live gate student-agnostic?
6. **rolling_notta** — does Rolling Forcing’s joint-denoise window keep
   tail motion on the same V2V protocol?
