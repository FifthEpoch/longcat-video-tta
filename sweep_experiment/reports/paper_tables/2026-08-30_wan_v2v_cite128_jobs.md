# Jobs submitted 2026-08-30 14:31

Pull `2d84a74..69eb346`. Do not scancel. Do not remake Self Forcing
or Rolling Forcing on 128.

| Job | What | Videos |
|---|---|---|
| **16615741** | Watch every tiny block and resample (Self Forcing) | 8 |
| **16615742** | Same, always on | 8 |
| 16615743 | Quality scores after 741–742 | — |
| **16615744** | Redo last two denoising steps (Self Forcing) | 8 |
| **16615745** | Same, always on | 8 |
| **16615746** | Rewrite last tiny block (Rolling) | 8 |
| 16615747 | Quality scores after 744–746 | — |
| **16615748** | Pseudo-future Search (first 32 copied) | 128 |
| **16615749** | Always-on search (first 32 copied) | 128 |
| 16615750 | Quality scores after 748–749 | — |

**01:19 31 Aug squeue:** **749** R 8h50 on gh111. **750** still
PD Dependency. **741–748** gone — do not assume COMPLETED until
`sacct`. Do not scancel 749/750.

Cancel one wave only:
`scancel 16615741 16615742 16615743` (8-video resample)
`scancel 16615744 16615745 16615746 16615747` (8-video redo/rewrite)
`scancel 16615748 16615749 16615750` (128 search)
