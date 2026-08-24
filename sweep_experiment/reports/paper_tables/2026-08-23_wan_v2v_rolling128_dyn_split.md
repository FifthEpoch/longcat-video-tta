# rolling-128 Dyn split (2026-08-23 20:59)

Login count on `v2v_panda_rolling_128v/rolling_notta` `joined.json`.
Cite medians. Flickering still **16259396** (PD QOSMaxGRESPerUser).

| Split | n | n_dyn | mean | median |
|---|---:|---:|---:|---:|
| all 128 | 128 | 68 | 0.531 | **1.0** |
| first 32 (same videos as N=32) | 32 | 14 | 0.438 | **0.0** |
| last 96 | 96 | 54 | 0.562 | **1.0** |

N=32 forward rolling Dyn **median 0** is the first-32 slice, not
“RF never trips VBench dynamic.” 14/32 of that slice are already
dynamic; median stays 0 because 14 < 16. The extra 96 sit just
over half (54/96) and flip the 128 median.

SF notta 128 Dyn median is still **0**. The host is what creates
VBench-dynamic clips. Locked IQ/subject bars already passed
(`2026-08-23_wan_v2v_rolling128_vbench6_read.md`). Official 7-dim
waits on flickering.

Do not resubmit **16259396**. Do not scale crosses. Host, not ours.
