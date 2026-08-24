# SF-family dissection playbook (2026-08-24)

Locked **before** harvest. The question is not only promote / no.
Every arm must produce a **mechanism read** and a **next action**,
including a clean miss.

Paper baseline = SF notta (`v2v_panda_confirm_32v`). RF
`rolling_notta` is a comparison row. Analyzer `PROMOTE` is the
correct call this time (methods sit on SF). Cite **medians**.
Official VBench = **full clip**. Do not retune `DROP=0.8` after
seeing 32. Do not scale RF-hosted 32. No TTC. No I2V.

Harvest script (login CPU):

```bash
python3 -u wan_experiment/scripts/analyze_v2v_sf_family_dissect.py \
  --family-dir wan_experiment/results/v2v_panda_sf_family_32v \
  --notta-dir wan_experiment/results/v2v_panda_confirm_32v \
  --rolling-dir wan_experiment/results/v2v_panda_forward_32v
```

Also run the usual `analyze_v2v_bakeoff.py` + `pair_v2v_tails.py`.

---

## 1. Three numbers, then the mechanism

Do not collapse an arm to one Δ. Print in this order:

| Layer | What it answers | Kill / keep |
|---|---|---|
| **Headline** | Locked bars vs SF: tail ↑, IQ ≥ SF−1, subject ≥ SF−0.02 | Promote letter |
| **Coverage** | Did the widget fire? Exact-SF rate | Dead gate vs live gate |
| **Conditional** | When it fired, did tail move? | Sensor OK, actuator dead |
| **Quality tax** | IQ / subject / flicker / Dyn | Motion bought with identity |
| **Heterogeneity** | Which videos win / lose | One wound vs 32-wide |
| **Host gap** | Same Δ vs RF rolling | Closing RF ≠ inventing |

A method can fail the letter and still be the next invention if
coverage is high and the miss is a named tax we can veto.

---

## 2. Outcome cells → next action (every arm)

Pick **one** primary cell from the tables. Always write a next
action. “No” is not the end of the page.

| Cell | How we know | Next action (pre-registered) |
|---|---|---|
| **A. Dead gate** | Exact-SF ≥ 16/32, or fire < 8/32 | New **sensor**, not a bigger k. Prefix-relative motion, last-2-chunk trend, or a different hold-out. |
| **B. Fire, no act** | Sick/search fires; accept or trust-pick ≈ 0 | New **actuator**: second resample, different seed family, rewind two chunks. Sensor is live. |
| **C. Act, tail down** | Fire+accept, median tail ≤ SF, many losses | Pick-score **anti-aligned** (I2V lesson). Veto with a second signal; do not climb motion blindly. |
| **D. Tail up, quality tax** | Letter fails IQ or subject, or flicker ≤ 0.972 | Keep the motion lever. Add an **identity veto** (appear / first-frame). Do not scale. |
| **E. Tail up, mixed videos** | Letter may pass;  W/L near 16/16 or 2–3 named wounds dominate | Cluster wounds. Next widget is **for that cluster**, not a 32-wide retune. |
| **F. Clean letter win** | Tail ↑ and IQ/subject hold; fire ≥ 8/32 or always-on; not 18/32 exact-SF | **HOLD N=32.** Mechanism table must show the widget did the work. Do **not** scale tonight. Ablate, then N=128. |
| **G. H1 twitch** | Dyn 0→1 **and** flicker ~0.972 **and** subject down | **NO.** Crossed sampler. Not a motion method. |
| **H. Always-on harm** | Sink-like; tail down or tax on almost every video | Drop the widget on SF. Record that the lever is host-specific. |

RF-family memory (do not reuse as SF evidence): rewind recovered
**0027**; sink recovered **0004**; pseudo was 18/32 exact RF
(cell A). Same cells apply on SF.

---

## 3. Per-method questions (use the sidecars)

Chunk json already has `gate_reason`, `last_sick`, `chunk_motion`,
`rewind {mot0,mot1,ref,accepted}`, `pseudo_fire`, `search_k`,
`chosen_cand`. That is enough. Do not invent a new metric tonight.

### `sf_rewind` (Family A)

1. Videos with ≥1 sick trigger. Accept rate among triggers.
2. Conditional tail: accepted ≥1 vs never-accepted vs never-sick.
3. After accept, does `mot1 ≥ mot0` stick to the **tail**, or does
   a later chunk re-freeze?
4. Named: **0027** (RF rewind story). Does SF rewind recover it
   vs SF notta?

| If we see | We learned | Next |
|---|---|---|
| Rarely sick | SF freezes less / differently than RF | Prefix-relative or 2-chunk trend sensor |
| Sick, mostly reject | Resample is not better than the freeze | Two retries or rewind chunk−1 as well |
| Accept, later freeze | One-shot rewind | Stay-on / sick-search after accept |
| Accept, tail up, flicker 0.972 | Twitch | **NO** (cell G) |
| Accept, tail up, quality holds | Widget works on SF | HOLD; ablate accept-only vs always-resample |

### `sf_sick_search` (Family B)

1. Exact-SF rate (never-sick ⇒ bit-match notta).
2. Search chunks / video. Trust-reject rate (`look_trust_reject`).
3. When k=4, does pick beat cand0 on tail, or only on the chunk?
4. Overlap with rewind wins. High overlap ⇒ **one sensor**, keep
   one actuator.

| If we see | We learned | Next |
|---|---|---|
| Exact-SF ≥ 16 | Gate sleeps | Same as rewind cell A |
| Search fires, trust rejects extras | Feasible set empty | Trust 0.8 is the brake; try max-motion among all k, log the tax |
| Search helps chunk, not tail | One-chunk memory | Stay-on for the next 1–2 chunks |
| Wins = rewind wins | Duplicate | Keep the cheaper of the two |

### `sf_pseudo` (Family D)

1. Fire rate on the prefix hold-out (RF was low).
2. Fire ⇒ tail vs no-fire ⇒ exact-SF.
3. Does B-MAE predict 30 s tail at all? (rank correlation on the 32)

| If we see | We learned | Next |
|---|---|---|
| Fire < 8/32 | Prefix B is a dead sensor | Hold out a **later** span, or fire on prefix motion, not MAE |
| Fires, tail flat | Extra seed is a coin flip | **NO** (RF pseudo). Do not raise k |
| Fires on easy videos only | Gate anti-selected | Invert: search when B-MAE **loses** |

### `sf_sink` (Family C)

Always-on. No fire rate. Every video should differ from notta.

| If we see | We learned | Next |
|---|---|---|
| Tail ≈ SF, exact-ish | `sink_size` is a no-op on SF KV | Dead lever on this host |
| Tail ↑, subject/flicker tax | Same RF story: pixel-move probe | Identity veto, or **0004-only** use. Not HG-f. Not scale |
| Tail ↑, quality holds | Host-specific win | HOLD; this is the first SF-native always-on lever |
| Helps 0004, hurts live-hot | Orthogonal to rewind | Combine with rewind **only** if Jaccard of wins < 0.5 |

---

## 4. Mixed metrics (do not average them away)

- **Cite medians.** One outlier (I2V video 26) already lied to us.
- **Mean vs median disagree** → name the videos that move the mean.
- **Tail ↑ / Dyn still 0** → we moved the handcrafted tail, not
  VBench dynamic-degree. Honest: tail-motion method, not Dyn.
- **Dyn 0→1** → check flicker. ~0.982 + subject hold = recovered
  motion. ~0.972 + subject down = H1 twitch (**NO**).
- **IQ −0.9, subject +0.03** → letter passes; still record the IQ
  dip. Do not call it free.
- **W/L 17/15** → not a method. Need the conditional-on-fire split
  before any “small win” sentence.
- **Vs RF better, vs SF worse** → impossible if SF is the host
  unless a bug. Recheck pairing.
- **Vs SF better, vs RF worse** → expected. That is “we moved SF
  toward RF,” not “we beat the field host.”

---

## 5. If an arm is a win

Do this **in order**. Do not skip to 128.

1. **Mechanism proof.** Coverage + conditional tables. If 18/32
   exact-SF, it is not a win (cell A) even if median Δ is positive.
2. **Quality proof.** All 7 VBench dims. Flicker vs H1 0.972.
3. **Named wounds.** If two videos explain the median, it is a
   case study, not a method. HOLD, do not scale.
4. **Complementarity.** Jaccard of “beat SF by >10%” vs the other
   HOLD arms. Jaccard ≥ 0.7 ⇒ pick the cheaper one. Jaccard < 0.5
   ⇒ a **combine** experiment is allowed (one new arm, N=32).
5. **Ablation (N=32, same videos), not a retune.**
   - Rewind: accept-only already; compare to always-resample (upper).
   - Sick: k=4 vs k=2 (cost). Do not change DROP.
   - Sink: `sink_size` 3 vs 1 vs off (already have off = notta).
6. **Invention sentence** (required before 128):
   “On Self-Forcing chunked, *widget* improves SF notta by *X*
   because *sensor* fires on *Y* and *actuator* does *Z*.”
   If we cannot fill Y and Z from the sidecar, we do not scale.
7. **N=128** only after 1–6. Same first 128 as rolling-128.
   Same locked bars vs SF. Still no TTC.

A win vs SF that still loses to RF is publishable as **method-on-SF**.
It is not a reason to drop the RF comparison row.

---

## 6. If every arm fails

We still leave with four facts:

1. **SF freeze signature** — sick-rate vs RF family. If SF rarely
   trips `DROP=0.8`, the RF widgets were host-tuned. Next sensor
   is built on SF chunk traces, not RF ones.
2. **Named SF wounds** — videos where notta tail is high and we
   still lose, and videos where notta is dead and nothing woke it.
   Those two lists are the next method’s spec.
3. **Score alignment** — did higher chunk motion predict higher
   tail and higher Dyn? If not, stop picking on that scalar.
4. **What not to try again** — `sf_roll`, pack-2, retune DROP on
   the same 32, TTC, I2V scale-up.

Then the next wave is **one** new sensor or **one** combine, N=32,
same playbook.

---

## 7. Cross-method table (required)

For videos 0004, 0027, and the worst 3 / best 3 vs SF:

| Video | SF | RF | rewind | sick | pseudo | sink | note |
|---|---:|---:|---:|---:|---:|---:|---|

Plus: Jaccard of win-sets; count of videos all four lose (SF
attractor we cannot move); count only sink moves (identity /
still cluster).

---

## 8. What this playbook forbids

- Declaring a method dead without a cell (A–H) and a next action.
- Scaling on analyzer `PROMOTE` alone.
- Retuning `DROP` / trust 0.8 after seeing these 32.
- Citing last5 VBench or PSNR.
- Using RF-hosted +X vs SF as the SF-family claim.
- Adding TTC or LoRA-at-test-time because “the family failed.”
