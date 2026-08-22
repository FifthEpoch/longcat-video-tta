# live_bon N=32 verdict: **NO** (2026-08-21 21:42)

**Jobs:** generate **16147007** COMPLETED 1h59 0:0, 32 mp4s.
VBench **16147008** COMPLETED 24m 0:0.

**The test method is not the controller.** Do not retune `live_min`.
Do not scale live_hist / appear / pseudo on the back of this.

## Locked rule (pre-registered)

YES only if (1) paired tail > confirm notta, (2) still prefixes
(`< 0.012`) not mass-damped, (3) VBench IQ not worse by ≥1.0 and
subject not worse by ≥0.02.

## What the script printed (do not cite)

`analyze_v2v_bakeoff.py` said paired N=32, tail 0.0138 → 0.0146
(+6%), PROMOTE. That `N=32` is the confirm **summary.json stub
count**. `pair_v2v_tails.py` only saw **0020–0031** (n=12), the
same 12 as the retracted seed_bon pairing. notta median 0.01380
**is that 12-video median**. live_bon 0.0146 is the 12-video
median (0.01442/0.01472). Mean on those 12: notta 0.0173,
live_bon 0.0163 (**−6%**).

VBench is on the 32 mp4s and is real.

| | notta | live_bon | Δ |
|---|---:|---:|---|
| tail median (n=12, only honest pair) | 0.0138 | 0.0146 | +6% |
| tail **mean** (same 12) | 0.0173 | 0.0163 | **−6%** |
| subject | 0.6652 | 0.6737 | +0.0085 |
| IQ | 69.65 | 69.52 | −0.13 |
| Dyn | 0 | 0 | 0 |

Quality bars pass. That is a safer skip, not a motion win.

## The 12 we can pair (0020–0031)

| video | prefix | gate | notta | seed | live_bon | vs notta |
|---|---:|---|---:|---:|---:|---|
| 0020 | 0.00086 | skip | 0.00646 | 0.00656 | 0.00646 | tie |
| 0021 | 0.00889 | skip | 0.00981 | 0.00856 | 0.00981 | tie (avoided seed damp) |
| 0022 | 0.06834 | search | 0.03579 | 0.02678 | 0.02678 | **loss** |
| 0023 | 0.00417 | skip | 0.01368 | 0.01406 | 0.01368 | tie |
| 0024 | 0.02274 | search | 0.01093 | 0.01442 | 0.01442 | win |
| 0025 | 0.00076 | skip | 0.02637 | 0.02449 | 0.02637 | tie (avoided seed damp) |
| 0026 | 0.04472 | search | 0.01392 | 0.02287 | 0.02287 | win |
| 0027 | 0.04763 | search | 0.03519 | 0.02720 | 0.02720 | **loss** |
| 0028 | 0.03534 | search | 0.02026 | 0.01472 | 0.01472 | **loss** |
| 0029 | 0.00653 | skip | 0.00877 | 0.01064 | 0.00877 | tie (0000-style FN) |
| 0030 | 0.01293 | search | 0.01066 | 0.00871 | 0.00871 | **loss** |
| 0031 | 0.00170 | skip | 0.01618 | 0.00922 | 0.01618 | tie (avoided seed damp) |

Skips (6): **exact notta**. The still-prefix half of the gate works.

Searches (6): **exact seed_bon**. 2 wins, **4 losses**. Live-and-hot
prefixes (0022/0027/0028) already had high notta tails; prefix-match
search damped them. That is the N=32 seed_bon failure, now restricted
to `prefix ≥ 0.012`.

N=8 hid this: 0001/0006/0007 were live **and collapsed**. N=32 live
is often live **and already moving**.

## Verdict

| Clause | Result |
|---|---|
| Tail | **NO.** n=12 mean −6%; median +6% is the high-tail losses staying above the median. Not a motion method. |
| Stills | Pass. Skips bit-match notta. |
| VBench | Pass. IQ −0.13, subject +0.009. Dyn 0/0 (notta already 0). |

**Overall: NO.** live_bon is a skip that prevents still-prefix
damping. It is not a test-time motion controller. Close this gate.
0000–0019 sidecars can lock a paper tail number; they will not
change the mechanism call.

Analyzer now prefers sidecars and drops skip stubs (`allow_partial`
no longer trusts `ok:true` stubs).
