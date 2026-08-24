# V2V text prompt was the filename (2026-08-24)

`v2v_panda_confirm_32v` sidecars:

| file | prompt_source | prompt |
|---|---|---|
| 013_panda_0013 | stem | `panda 0013` |
| 001_panda_0001 | stem | `panda 0001` |
| 020_panda_0020 | stem | `panda 0020` |
| 005_panda_0005 | stem | `panda 0005` |

`datasets/panda_1000_480p/` has **no** `captions.json`. Only
`dynamic_degree.json` / `dynamic_degree_long.json`.

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

## Next (not submitted)

1. Find or write real captions →
   `datasets/panda_1000_480p/captions.json`.
2. Runner now **refuses** `panda_*` stem prompts.
3. Re-run the paper baseline + lead methods only after GO.
4. Do not scancel always-search; it is the same-prompt ablation.

No TTC. No I2V scale-up.
