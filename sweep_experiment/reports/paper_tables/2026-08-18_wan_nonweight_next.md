# Non-weight next steps (after a standard bench exists)

**Status:** brainstorm locked 2026-08-18. Do **not** implement TTC.
Do **not** retune the I2V-32 gate and call that a paper quality win.
Do this work on the T2V 128 / VBench-Long bench, not by scaling I2V-32.

---

## What the field already treats as “when to spend compute”

| Idea | Paper | Axis |
|---|---|---|
| Intervene only if failure is predicted | Early Failure Detection (2026), Wan 1.3B/14B | gate *whether* to spend |
| Explore cheap, recommit winner | CachedSearch (2026), Wan 1.3B–14B | gate *how* search is paid |
| Skip a denoising step vs compute | BAG / NaviCache | NFE, not chunk seed |
| Prune on partial latents | LatSearch | do not decode every candidate |
| Rewind to a good prefix | Temporal Backtracking Search | not root-only BoN |

StreamingT2V: long AR either breaks or **stands still**. History
Guidance: vanilla history-CFG helps identity and **kills dynamics**;
fractional/frequency HG puts motion back. Same trade as our high
smoothness + `dynamic_degree` median 0.

Nobody’s next-step paragraph is “handcrafted sharpness-deviation gate
on I2V stills.” Several are “detect failure, then allocate compute”
and “do not trust a consistency score that rewards freeze.”

---

## What our I2V-32 run forbids us to forget

1. Last-chunk composite is **anti-aligned with imaging quality**
   (last5 ρ +0.23 to +0.33). Do not search that score and expect
   VBench IQ to rise.
2. `dynamic_degree` is 0/1. Median 0 = most clips fail RAFT’s
   dynamic test. Seed-BoN unfroze ~2 last5 clips. The attractor is
   freeze; four seeds of the same score stay there.
3. Full-piece pick-score can lie (11/16). Incoming/outgoing last
   second is the less-wrong *controller* signal, not a quality claim.
4. First-second-after-a-still is a **bad motion target**. Two-sided
   deviation regularizes toward mild I2V motion. Official dynamic
   degree wants large motion.

Late AR chunks sample a posterior that has collapsed toward a still.
Changing the **seed** is not changing the **trajectory**.

---

## Ideas to try (no new weights)

1. **Failure-gated CachedSearch**  
   Gate *whether* the chunk is sick (incoming/outgoing or a
   VBench-tracking scorer). If sick: cheap-cache BoN, recommit only
   the winner. If healthy: one seed. Early Failure Detection +
   CachedSearch on the chunked AR loop.

2. **Motion verifier, not sharpness-deviation**  
   Score by RAFT / dynamic_degree proxy or ImageReward (CachedSearch).
   Gate on “about to freeze.” Directly attacks the 0 median.

3. **Test-time history / shift / CFG search**  
   Search `{shift, cfg, sink on/off}` rather than k=4 seeds. Wan
   `shift` already moves dynamics. Rolling/Relax Forcing keep an
   attention sink of the first frames.

4. **Prefix backtrack**  
   If outgoing explodes, rewind one piece and resample. Do not
   stay-on into a poisoned prefix (11/16).

5. **Approximate fractional history guidance**  
   Two forwards (text vs low-passed prefix) only if the causal
   pipeline can do a second pass. Vanilla history-CFG can freeze;
   that is why HG-f exists.

6. **No TTC / LoRA-at-test-time.** Field long-horizon wins are KV
   memory, sinks, teacher correction, and guidance.

---

## Order

Standard T2V 128 / VBench-Long bench first
([`2026-08-18_wan_t2v_vbenchlong_128_spec.md`](2026-08-18_wan_t2v_vbenchlong_128_spec.md)).
Then try (1)+(2) on that bench. Not on a larger I2V-32.
