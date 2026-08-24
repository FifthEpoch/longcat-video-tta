# Caption-conditioned V2V N=32 — WAITING GO (2026-08-24)

Every finished V2V arm on `panda_1000_480p` used filename stems
(`panda 0013`). Real first-segment captions are in
`datasets/panda_1000_480p/metadata.csv` (1000/1000, 0 empty). Runner
now loads that CSV and refuses panda stems.

This re-run is the first caption-conditioned protocol. Same-prompt
stem tables stay as a confound audit, not the paper caption claim.

## Lock

- N=32, same first-32 paths as confirm_32v (sorted `rglob`).
- Prefix 9 latents, 6 × 21 tail, 30 s.
- Prompt = first list caption from `metadata.csv`.
- Sidecar must show `prompt_source=metadata_csv` (or `caption_json`).
- k=4. No TTC. No I2V. Do not scale to 128 tonight.

## Arms (same-wave)

If GO includes a gated method, launch the twins in the same paste.

| method | host | why |
|---|---|---|
| `notta` | SF | paper baseline |
| `rolling_notta` | RF | host comparison row |
| `sf_pseudo` | SF | gated lead on stem tables |
| `sf_always_search` | SF | always-on twin |
| `rf_always_search` | RF | other-host always twin |

Do **not** add `sf_sink` to this wave (subject on the −0.02 line;
no-scale). Do not add rewind/sick.

Series: `v2v_panda_caption_32v`.

Cite vs caption-conditioned SF notta. Do not mix stem-prompt numbers
into the caption table.

## Status

**WAITING GO.** Always-search 16288113–115 is the stem-prompt
ablation; let it finish.
