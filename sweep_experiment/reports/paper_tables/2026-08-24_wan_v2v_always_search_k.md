# Always-search width (2026-08-24)

Search width **k** = candidates generated before the pick, not N videos.

## What the field uses on this model

CachedSearch (2026, Wan2.1-T2V-1.3B — our stack) treats:

| Width | Role in that paper |
|---|---|
| **N=4** | Full-compute best-of-4. The cheap / budget baseline. |
| **N=8** | Headline search. Their API default (`n=8`). |

Both are conventional. N=16+ is a cost sweep, not the default.

Our family widgets (`sf_pseudo`, `sf_sick_search`, seed_bon) were
locked at **k=4**. The gate-vs-pick ablation must stay **k=4** or
it is a different experiment.

## This wave

`sf_always_search` and `rf_always_search` use **k=4**.

- Matches the family arms we are ablating.
- Matches CachedSearch’s published BoN-4 budget.
- k=8 is the CachedSearch headline; that is a **later** width
  sweep, not this split. Do not retune k after seeing 32.

## Same-wave rule (locked)

When we submit a gated method, submit the obvious twins in the
**same paste**: always-on (no gate), and the other host if the
claim is host-specific. Do not wait for harvest to invent the
ablation. Harvest still decides the call.
