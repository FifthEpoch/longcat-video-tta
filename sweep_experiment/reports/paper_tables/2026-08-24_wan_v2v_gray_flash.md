# Gray flash in V2V mp4s is the 21-latent / 81-frame horizon (2026-08-24)

Local freeze demos: `Downloads/generated_video/using_wan/v2v_notta_freeze_demos/`
(stem-prompt SF notta, 537 frames @ 16 fps).

The flash is **in the pixels**, not a player-only keyframe glitch.
It sits on **latent 21** (frames 81–84, **t = 5.06–5.25 s**):

| clip | interframe MAE at f82 | sat f80 → f82 |
|---|---:|---|
| 0001 | 18.3 | 40.7 → 31.5 |
| 0005 | 14.8 | 35.7 → 29.2 |
| 0020 | 36.2 | 16.5 → 24.3 (also the biggest jump) |

0001 detail: frames 70–80 are calm (MAE ~1–3). Frame 81 starts
latent 21 (MAE 10). Frame 82 peaks (MAE 18). Frames 83–84 stay
washed-out. Frame 85 (latent 22) recovers.

That is one Wan VAE group (4 pixels). It is **not**:

- the prefix seam (9 latents → frame 33 / 2.06 s)
- a generate-chunk seam (30 latents → frame 117 / 7.31 s)
- x264 keyint (250 frames → 15.625 s on most of these files)

Self-Forcing’s native unit is **21 latents → 81 frames → 5.0625 s**.
Latent 21 is the first step past that trained clip. Same clock on
every video, so every clip flashes once there.

Stem-prompt panda morph is a separate bug (`2026-08-24_wan_v2v_panda_stem_prompt.md`).
