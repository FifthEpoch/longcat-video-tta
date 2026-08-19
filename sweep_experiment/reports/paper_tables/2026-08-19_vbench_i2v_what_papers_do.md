# What I2V papers actually do with VBench++ (2026-08-19)

**Question:** Should we include a text prompt for VBench++? Is that
standard for I2V?

**Short answer:** At **generation**, yes — official VBench-I2V is
image + paired caption (except image-only models like SVD). At
**quality scoring**, the seven dims we ran do **not** read the prompt.
The I2V-specific columns (`i2v_subject`, `i2v_background`,
`camera_motion`) are what need the condition image (and, for camera,
the prompt). We generated with captions. We did not score those
I2V columns.

## What we already do

`discover_items` loads `caption` from `i2v-bench-info.json` and both
`run_i2v_continuation.py` and `run_i2v_chunked.py` pass
`text_prompts=[prompt]` into Self-Forcing. Each still is animated with
its VBench-I2V caption. If the json is missing, the fallback is the
filename stem — confirm on cluster before claiming every clip was
caption-conditioned.

`score_i2v_vbench.py` then runs `mode=custom_input` on the seven
**video-quality** dims. DINO / CLIP / MUSIQ / RAFT never see the
text. Passing the prompt into `evaluate()` would not move
`dynamic_degree`. The script **drops** `i2v_subject` /
`i2v_background` with a warning.

## Official VBench-I2V (VBench++ I2V leaderboard)

Huang et al., VBench++ §III: *“Text Prompts Paired with Images.”*
Captions start from CoCa/BLIP2, then humans add **motion details**.
Camera-motion items append “camera pans left” / “zooms in” / etc.
Sampling code is `(image_path, prompt_en)` for every clip; 5 samples
per pair. Filenames are `$prompt-$index.mp4`.

Leaderboard length:

| Era | Models | Length |
|---|---|---|
| Original suite (Table III) | DynamiCrafter, ConsistI2V, VideoCrafter-I2V, SVD-XT, … | **16 frames / ~2 s** |
| Current Wan-class | WAN2.1-I2V-14B, Wan2.2-I2V-A14B | **81 frames / ~5 s** |

Neither is 30 s / N=32.

Two dim groups (Table III columns):

| Group | Dims | Uses text at score time? | Uses the input image? |
|---|---|---|---|
| Video-condition (I2V-specific) | `i2v_subject`, `i2v_background`, `camera_motion` | camera_motion: yes (instruction in prompt/filename). Others: no | **Yes** (DINO / DreamSim vs the still) |
| Video-quality | subject/background consistency, flicker, smoothness, dynamic degree, aesthetic, imaging | **No** | No |

`i2v_subject` ≠ `subject_consistency`. GitHub issue #34: the first is
**image↔video** identity; the second is **frame↔frame** inside the
video. We only scored the second.

SVD-XT-1.1 is on the same table with **MotionCamera = “-”** (image-only
I2V). Text is standard for captioned I2V models, not a universal
requirement of the quality scorer.

Wan-class papers (Wan2.1-I2V-14B on the VBench-I2V leaderboard) still
generate with image + prompt (often Qwen prompt-extend) at **5 s**,
then run `evaluate_i2v.py`.

## VBench++ Table III (official ~2 s I2V suite)

Percentages as published. Dynamic degree is a population fraction, same
0/1 RAFT test we used.

| Model | I2V subj | I2V bg | Camera | Subj (frames) | Dynamic |
|---|---:|---:|---:|---:|---:|
| DynamiCrafter-1024 | 96.71 | 96.05 | 35.44 | 95.69 | **47.40** |
| SVD-XT-1.1 | 97.51 | 97.62 | — | 95.42 | **43.17** |
| SEINE | 94.85 | 94.02 | 23.36 | 94.20 | 34.31 |
| I2VGen-XL | 96.74 | 95.44 | 13.32 | 96.36 | 24.96 |
| VideoCrafter-I2V | 90.97 | 90.51 | 33.58 | 97.86 | 22.60 |
| ConsistI2V | 94.69 | 94.57 | 33.60 | 95.27 | **18.62** |
| Animate-Anything | 98.54 | 96.88 | 12.56 | 98.90 | **2.68** |

Our 5 s mean dynamic **25%** (8/32) sits next to I2VGen-XL / VideoCrafter,
not next to DynamiCrafter. Animate-Anything is the cautionary high-consistency
/ almost-still extreme. Adding the prompt at `evaluate()` will not move us
toward DynamiCrafter’s 47%.

## What would actually match “standard I2V VBench++”

1. Keep generating with the paired caption (already done, pending json check).
2. Score **`i2v_subject` / `i2v_background`** on the existing mp4s with
   `vbench2_beta_i2v` and name-matched stills. No new generate. This is
   the missing official I2V column.
3. Do **not** treat 30 s / N=32 as a VBench-I2V leaderboard number
   (official is ~2 s or 5 s, hundreds of images). Full-clip quality 7 on
   30 s stays our diagnostic / hybrid lock.
4. If the goal is a **high dynamic-degree** table, that is T2V
   MovieGen-128 (action in the prompt), not more text on these stills.

No TTC. No I2V-32 scale-up. Scoring I2V dims on the mp4s we have is
allowed outcome eval.
