# Keep-picture jobs submitted 2026-08-30 14:56

**2-video check:** Self Forcing **FAIL** (0 videos, ~2 min, exit 2:0).
Rolling **PASS** (2 videos). The code looked for `nudge` but the
name was still `sf_nudge`. Fixed. Resubmit Self Forcing only.

**Rolling 8-video:** already wrote 8 videos (one still finishing).
Do not remake Rolling.

## Self Forcing resubmit 17:05 (kind fix `deee3ad`)

2-video: **16620355–362**, scorer **16620363**.  
**PASS 17:27:** six COMPLETED with 2 videos. `360` / `362` still writing (1 video each).  
8-video: **16620364–371**, scorer **16620372**. Leave running.

Cancel this resubmit only:
`scancel 16620355 16620356 16620357 16620358 16620359 16620360 16620361 16620362 16620363`
`scancel 16620364 16620365 16620366 16620367 16620368 16620369 16620370 16620371 16620372`

---

# Keep-picture jobs submitted 2026-08-30 14:56

Pull `69eb346..384e72a`. Do not scancel. Do not cancel
**16615741–750** (crash reruns + 128 search).

## 2-video check

| Job | What |
|---|---|
| 16616159 / 16616160 | 90/10 last-step mix (stuck / always) |
| 16616161 / 16616162 | Next-block seed (stuck / always) |
| 16616163 / 16616164 | Residual + first-frame lock (stuck / always) |
| 16616165 / 16616166 | First-vs-last latent pick + lock (stuck / always) |
| 16616167 / 16616168 | Rolling 90/10 mix (stuck / always) |
| 16616169 / 16616170 | Rolling residual + lock (stuck / always) |
| 16616171 / 16616172 | Rolling first-vs-last pick + lock (stuck / always) |
| 16616173 | Quality scores after the 2-video jobs |

Cancel 2-video wave only:
`scancel 16616159 16616160 16616161 16616162 16616163 16616164 16616165 16616166 16616167 16616168 16616169 16616170 16616171 16616172 16616173`

## 8-video run

| Job | What |
|---|---|
| 16616174 / 16616175 | 90/10 last-step mix (stuck / always) |
| 16616176 / 16616177 | Next-block seed (stuck / always) |
| 16616178 / 16616179 | Residual + first-frame lock (stuck / always) |
| 16616180 / 16616181 | First-vs-last latent pick + lock (stuck / always) |
| 16616182 / 16616183 | Rolling 90/10 mix (stuck / always) |
| 16616184 / 16616185 | Rolling residual + lock (stuck / always) |
| 16616186 / 16616187 | Rolling first-vs-last pick + lock (stuck / always) |
| 16616188 | Quality scores after the 8-video jobs |

Cancel 8-video wave only:
`scancel 16616174 16616175 16616176 16616177 16616178 16616179 16616180 16616181 16616182 16616183 16616184 16616185 16616186 16616187 16616188`
