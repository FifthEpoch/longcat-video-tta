# Sticky gated-search, 32 videos (2026-08-18)

**Source:** cluster `analyze_i2v_bon.py` on
`i2v_bon_32v_sticky` vs baseline `i2v_bon_32v_hybrid` (user paste 10:47).
**What changed:** same three alarms as hybrid. After the first alarm on
a video, later pieces keep searching four ways (`already_on`).
**Comparison:** same 32 images and seeds. Do-nothing and always-search
are the hybrid jobs. Lower last-piece score is better.
**Do not cite raw means** — video 26 is still 85.6 for both search
methods.

Regenerate:
```
python wan_experiment/scripts/analyze_i2v_bon.py \
  --series-dir wan_experiment/results/i2v_bon_32v_sticky \
  --baseline-dir wan_experiment/results/i2v_bon_32v_hybrid
```

## Cite this

| Method | Median last-piece | Mean wall | vs always-search |
|---|---|---|---|
| Do-nothing | 3.679 | 91.7 s | — |
| Always-search (4 candidates every later piece) | **2.966** | 258.1 s | — |
| Hybrid gated (alarms only) | 3.036 | **172.8 s** | 9 / 10 / 13, median Δ 0 |
| **Sticky gated** (alarms + stay on) | **2.989** | 256.4 s | 6 / **21** / 5, median Δ 0 |

Sticky last-piece median is almost always-search (2.989 vs 2.966).
Cost is almost always-search (256 vs 258 s). **21 of 32 videos are an
exact tie with always-search.** We spent the 33% savings and became a
delayed-start copy of always-search.

| Contrast (sticky) | Mean Δ | Median Δ | better | tie | worse |
|---|---|---|---|---|---|
| always-search − do-nothing | +1.709 | −0.434 | 25 | 0 | 7 |
| sticky − do-nothing | +1.697 | −0.310 | 26 | 2 | 4 |
| sticky − always-search | −0.012 | **0** | 6 | **21** | 5 |

Exclude video 26: do-nothing 3.92 / always-search 3.08 / sticky 3.07.

## The three checks we set

| Check | Result |
|---|---|
| 03 (highway) and 24 (busy street) catch always-search | **Yes.** Both exact ties (1.567 and 2.315). That is what stay-on was for. |
| 06 / 07 / 28 stay skipped on piece 1 | **Yes.** Incoming 0.20 / 0.68 / 0.25. 06 and 07 still beat always-search. 28 got a bit better (1.931 vs hybrid 2.061). |
| 30 stay safe | **No.** Piece 1 already tripped the early alarm (incoming 1.41) in hybrid, which then went back to sleep and matched do-nothing (1.444). Stay-on kept searching and copied always-search’s worse ending (1.688). |
| No second video-26 explosion | **Yes.** 26 is still 85.6, same path. No new blow-up. |

## Sticky vs hybrid gated (same videos)

Negative Δ = sticky better than hybrid.

| i | What happened | hybrid | sticky | always | Δ sticky−hybrid |
|---|---|---|---|---|---|
| 03 | stay-on did its job | 2.827 | 1.567 | 1.567 | **−1.260** |
| 24 | stay-on did its job | 3.179 | 2.315 | 2.315 | **−0.864** |
| 12, 13, 08, 29 | finished matching always-search | — | = always | — | small wins |
| 25 | hybrid had *hurt* do-nothing; sticky matches always-search | 3.176 | 2.652 | 2.652 | −0.524 |
| 14 | late stay-on beat both | 1.242 | **0.894** | 1.234 | −0.348 |
| 22, 28 | extra late search helped | — | better than both | — | −0.30 / −0.13 |
| **11** | hybrid’s big unique win **erased** | **2.157** | 4.319 | 4.319 | **+2.162** |
| **16** | hybrid’s other big unique win **erased** | **2.656** | 5.047 | 5.047 | **+2.391** |
| 01 | lost a small hybrid edge | 1.936 | 2.104 | 2.104 | +0.168 |
| 30 | un-saved; now copies always-search harm | 1.444 | 1.688 | 1.688 | +0.244 |
| 17 | still never wakes (incoming 0.76–1.20) | 3.006 | 3.006 | 1.553 | 0 |

Net sticky vs hybrid: 10 better, 18 tie, 4 worse, mean **+0.028**
(slightly worse). The 03/24 wins are cancelled by destroying 11 and 16.

Always-search still hurts do-nothing on 06, 07, 16, 19, 26, 28, 30.
Sticky still saves 06 and 07 (never woke early). It **no longer saves
30**, and it **joins** always-search’s harm on 16.

## Locked read

- Stay-on **worked as designed** on 03 and 24. It is not a failed
  implementation.
- It is **not a quality win vs always-search.** 21 exact ties, median
  Δ 0, cost within 2 seconds. The paper cannot say sticky gated-search
  beats always-search.
- It **spent the efficiency story.** Hybrid was 33% cheaper. Sticky is
  not.
- The important new fact: **keeping the search on can erase a good
  prefix.** Videos 11 and 16 were hybrid’s best unique wins. Local
  scores kept improving (`chosen−cand0` down to −4.1 and −3.6) while
  the ending got worse. Local piece score still does not predict the
  ending.
- Hybrid gated-search remains the method to cite if the claim is
  “almost the same typical quality, one third cheaper.”
- Do not stack another stay-on variant. Do not start test-time
  training. Video 17 is still the never-wake miss; that is a different
  alarm problem, and a weaker alarm will have false positives.

## Per-video last-piece (sticky)

| i | key | do-nothing | always | sticky | sticky−always |
|---|---|---|---|---|---|
| 00 | abstract black and white | 4.948 | 4.427 | 4.427 | 0 |
| 01 | boiling pot | 2.129 | 2.104 | 2.104 | 0 |
| 02 | flower bud | 5.275 | 4.164 | 4.164 | 0 |
| 03 | highway cars | 2.798 | 1.567 | 1.567 | 0 |
| 04 | bald eagle | 8.874 | 4.696 | 4.696 | 0 |
| 05 | bar with chairs | 4.990 | 3.908 | 3.908 | 0 |
| 06 | french fries | 2.729 | 3.338 | 2.594 | **−0.743** |
| 07 | beach buildings | 1.963 | 2.620 | 1.963 | **−0.657** |
| 08 | woman in blue sari | 4.145 | 3.641 | 3.641 | 0 |
| 09 | bicycle at fence | 5.648 | 4.681 | 4.681 | 0 |
| 10 | bird with fish | 3.782 | 3.477 | 3.477 | 0 |
| 11 | blue and white smoke | 11.192 | 4.319 | 4.319 | 0 |
| 12 | blue car dirt road | 5.494 | 2.375 | 2.375 | 0 |
| 13 | fishing boat | 2.263 | 2.044 | 2.044 | 0 |
| 14 | blue train | 1.242 | 1.234 | 0.894 | **−0.340** |
| 15 | boat on shore | 3.393 | 3.021 | 3.077 | +0.056 |
| 16 | book on fire | 4.776 | 5.047 | 5.047 | 0 |
| 17 | bridge in a river | 3.006 | 1.553 | 3.006 | **+1.453** |
| 18 | bridge over water | 2.844 | 2.080 | 2.786 | +0.707 |
| 19 | cow eating hay | 2.247 | 3.067 | 3.067 | 0 |
| 20 | brown bear | 4.158 | 2.911 | 3.210 | +0.299 |
| 21 | building on a hillside | 2.401 | 1.904 | 1.644 | −0.260 |
| 22 | food on a grill | 3.804 | 3.458 | 2.971 | −0.487 |
| 23 | houses on a hillside | 2.760 | 2.228 | 2.526 | +0.298 |
| 24 | busy street | 4.036 | 2.315 | 2.315 | 0 |
| 25 | butterfly on a flower | 2.663 | 2.652 | 2.652 | 0 |
| 26 | spiral galaxy | 5.063 | 85.630 | 85.630 | 0 |
| 27 | snowy castle | 6.630 | 6.525 | 6.525 | 0 |
| 28 | chair in a room | 2.061 | 2.656 | 1.931 | −0.724 |
| 29 | chef with mushrooms | 4.149 | 2.308 | 2.308 | 0 |
| 30 | church on a hill | 1.444 | 1.688 | 1.688 | 0 |
| 31 | city bus in snow | 3.577 | 3.535 | 3.535 | 0 |
