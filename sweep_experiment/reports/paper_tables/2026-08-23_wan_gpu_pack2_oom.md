# Pack-2 OOM — family wave 16261273–276 (2026-08-23 22:05)

Two processes × 137-frame KV (~39 GB) did not fit. One worker
wrote ~16 mp4s; the other died. `VIDEO_WORKERS` default is **1**.

| Job | Method | Elapsed | Exit | mp4 / 32 |
|---|---|---|---|---:|
| 16261273 | rf_rewind | 24m | 2:0 | 15 |
| 16261274 | rf_sick_search | 22m | 2:0 | 16 |
| 16261275 | rf_pseudo | 23m | 1:0 | 16 |
| 16261276 | rf_sink | 8m | 2:0 | 5 |
| 16261277 | VBench `afterany` | — | **scancel** | was scoring incomplete dirs |

VBench must be `afterok`, not `afterany`. Skip-existing resumes
the mp4s. Do not delete videos. Do not retry pack-2.
