# V2V text prompt was the filename (2026-08-24)

`v2v_panda_confirm_32v` sidecars:

| file | prompt_source | prompt |
|---|---|---|
| 013_panda_0013 | stem | `panda 0013` |
| 001_panda_0001 | stem | `panda 0001` |
| 020_panda_0020 | stem | `panda 0020` |
| 005_panda_0005 | stem | `panda 0005` |

`datasets/panda_1000_480p/` has **no** `captions.json`. The 2026-08-24
16:41 `ls *caption* *.json` also missed `metadata.csv` (357525 bytes,
same size as `datasets/panda_1000/metadata.csv`). Videos live in
`videos/`. The runner only loaded JSON, so it never saw the CSV.

A later hunt script printed `panda_100/metadata.csv` first (79-byte
header, 0 matches) and stopped. **Do not treat that as “no captions.”**
Peek `panda_1000_480p/metadata.csv` before a re-run.

`discover_v2v_items` falls back to `stem.replace("_", " ")`. Every
V2V arm on this pool (notta, seed_bon, RF rolling, SF/RF family,
always-search) was T5-conditioned on **“panda NNNN”** for the whole
30 s. The prefix is real footage. The tail is pulled toward a panda.
That is why freeze-demo clips morph into pandas.

This is **not** a Panda-70M caption-conditioned protocol.

## What still holds

Same-prompt comparisons (method vs SF notta, vs RF) stay valid.
They all heard the same bad string.

## What does not hold

- “These are Panda-70M scene captions.”
- Reading a panda morph as the freeze attractor alone.
- A paper sentence that we continued “the video’s caption.”

## Confirmed 16:56 — real scene captions exist

`metadata.csv`: 1000 rows, 1000 list captions, 0 empty. First-segment
resolution (LongCat TTA rule). 1000 mp4s.

| stem | first-segment caption | old stem prompt |
|---|---|---|
| panda_0001 | A young woman is standing in front of a kitchen counter. she has tattoos on her arms and is wearing a black tank top. | `panda 0001` |
| panda_0005 | The person is holding a plastic bowl filled with cherry tomatoes on a kitchen counter. | `panda 0005` |
| panda_0013 | There is a small spot on the ceiling of a bathroom caused by an accumulation of moisture due to insufficient ventilation. | `panda 0013` |
| panda_0020 | A close up of a flashlight in a box. | `panda 0020` |

0013 is a bathroom stain, not a panda. The morph-to-panda in the
downloaded freeze demos is T5 takeover from the filename string.

## Next (WAITING GO)

Caption-conditioned N=32 of the paper arms. Spec:
`2026-08-24_wan_v2v_caption_rerun_spec.md`. Do not submit until GO.
Do not scancel always-search (stem-prompt ablation). No TTC. No I2V
scale-up.
