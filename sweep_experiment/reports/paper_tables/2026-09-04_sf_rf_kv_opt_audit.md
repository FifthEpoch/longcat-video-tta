# Self Forcing / Rolling Forcing KV + compute audit (2026-09-04)

Not a submit. Read of official
[guandeh17/Self-Forcing](https://github.com/guandeh17/Self-Forcing)
and [TencentARC/RollingForcing](https://github.com/TencentARC/RollingForcing)
(`causal_inference.py`, `rolling_forcing_inference.py`,
`wan/modules/causal_model.py`) against our V2V wrappers
(`run_v2v_chunked.py`, `v2v_hosts.py`, `run_i2v_continuation.py`).

**Verdict:** the **quality** KV / sink / RoPE / window mechanisms live
in the third-party kernels we already call. Our loops match official
inference (with a prefix offset on Rolling because their public
`inference_rolling_forcing()` overwrites a multi-frame prefix).
**Do not retune `enlarge_kv_cache`, `local_attn_size`, or sink on
`notta` / `rolling_notta`.** That would change the host vs cite-128.

Training-only items (gradient truncation, 50% Self Forcing mix,
hide-first-chunk for eviction) are **not** inference. Do not start
8-GPU DMD.

Canvas: `canvases/sf-rf-kv-audit.canvas.tsx`.

---

## How our hosts actually run

| | Self Forcing (`notta`) | Rolling Forcing (`rolling_notta`) |
|---|---|---|
| Weights | `self_forcing_dmd.pt` via official `CausalInferencePipeline` | `rolling_forcing_dmd.pt` via official `CausalInferencePipeline` |
| Sampler loop | Our `generate_chunked_v2v` — official `inference()` is one-shot T2V/I2V | Our `generate_rolling_v2v` — official rolling overwrites prefix |
| KV write after lock | Full just-finished 3-latent block at `context_noise` (default **0**) | **First block of the window only** + `updating_cache=True` (official) |
| Prefix write | Replay 9 latents in blocks of 3 at t=0 | Same, then roll only the tail with `current_start` offset |
| Attention kernel | Official SF `CausalWanSelfAttention` | Official RF `CausalWanSelfAttention` (sink + Dynamic RoPE **hardcoded**) |

Official Rolling comment: “only cache the first block.” We do that.
Official Self Forcing writes the whole finished chunk. We do that.

---

## Inference mechanisms (quality)

| Mechanism | Paper | Official code | Our path | Status |
|---|---|---|---|---|
| AR unroll with KV on **self-generated** history | Both | Generator + `kv_cache1` / `kv_cache_clean` | Replay committed, then denoise | **In** |
| Clean KV write after lock (`context_noise`) | Both; default **0** | SF step 3.3; RF `updating_cache=True` | Same; `sf_ctx` / `rolling_ctx` are the t=50 probe | **In** (0) |
| Last-`L` attention (`L`=21 latents = 32760 tokens) | SF rolling KV | `max_attention_size = 32760` when `local_attn_size=-1` | Kernel slice; we do not bypass it | **In** |
| Causal chunk, one `t` per 3 latents | SF | `inference()` inner loop | `_denoise_chunk` / `_run_one_chunk` | **In** |
| Bidirectional window, monotone diagonal, lock at exit | RF | `inference_rolling_forcing` window construction | `generate_rolling_v2v` same windows + prefix offset | **In** |
| First-block attention sink | RF | `sink_tokens = 1 * block_length` **hardcoded**; first block stored **unroped** | Kernel; `apply_sink_size` does **not** change this | **In** (native) |
| Dynamic RoPE on the sink at read time | RF | `causal_rope_apply` on tokens `0:block` with `rope_start_frame` just behind the working window | Kernel | **In** |
| Cross-attn cache (T5, 512 tokens) | Both | `_initialize_crossattn_cache` | Reset + reuse | **In** |
| Few-step list + `warp_denoising_step` | SF `[1000,750,500,250]`; RF `[1000,800,600,400,200]` | Config + warp to scheduler | Loaded from their yaml; live RF floor is **556**, not paper 200 | **In** (native list) |

`sf_sink` / `rf_sink` / replay-`sink` were **extra** levers. Native
Rolling already sinks the first block. Setting `module.sink_size` on
official RF is a no-op because the forward ignores that attribute.
Replay-sink without their RoPE path matched `notta`. Leave extra sink
closed.

---

## Compute / memory shortcuts

| Shortcut | Official | Ours | Notes |
|---|---|---|---|
| KV **allocation** | SF 21 frames (`32760` tokens); RF **24** frames (`1560*24`) | `enlarge_kv_cache` to prefix+tail+2 ≈ **137** frames (~39 GB) | Attention still reads last 21 (+ RF sink). Extra store is memory, not a different window. Needed so SF (`local_attn_size=-1`, no eviction) does not write past a 21-frame buffer at 30 s. |
| Rolling eviction | SF only if `local_attn_size != -1` (DMD yaml does **not** set it → **-1**). RF evicts when the 24-frame buffer fills; keeps first block. | SF: no eviction, huge buffer. RF: buffer never fills at 30 s, so no eviction; sink stays at index 0 anyway. | Same attended tokens as official at 30 s. Do **not** flip `local_attn_size` on the cite hosts. |
| `torch.compile(flex_attention)` | SF: `max-autotune-no-cudagraphs` at import. RF: **commented out**. | We replace with eager `flex_attention` (job 15877786 / 138 GB compile). | Speed only. Official RF already ships without compile. |
| flash-attn | `model.py` calls `flash_attention` by name | Env skips compile (`SKIP_FLASH=1`); `install_sdpa_attention_fallback` | H200 SDPA uses the built-in flash kernel. Padding `k_lens` ignored. |
| T5 `DynamicSwapInstaller` | `demo_utils.memory` | `v2v_hosts._swap_text_encoder` | **In** |
| VAE `cached_decode` | Official decode `use_cache=False` | Our pixel decode is not their streaming cache | Match official. |
| Gradient checkpoint / truncation | **Train** | We do not train | Out of scope. |

---

## Training-only (correctly absent at test)

- Stochastic last-step gradient truncation (SF).
- Non-overlapping window gradients (RF).
- 50% Self Forcing loss mix (RF).
- Hide first chunk in the train cache so eviction is seen (SF).
- Holistic video DMD / critic / Wan-14B real-score.

These are how they **fit** the student. The student we load already
has them. Re-implementing them at inference is not a host bug.

---

## What we will not do after this read

1. Shrink KV to official 21/24 without an N=1 pixel-match vs
   `notta` / `rolling_notta`. A mismatch would poison cite-128.
2. Turn `local_attn_size=21` on Self Forcing to “enable the paper
   rolling KV.” Attention already windows to 21. Eviction is a
   memory opt; the yaml default is `-1`.
3. Re-enable `torch.compile` on flex-attention.
4. Treat `apply_sink_size` as Rolling’s paper sink. The paper sink
   is already on.
5. Start 8-GPU DMD.

FIFO lookahead and lock-score stay **on top of** this host, not a
replacement for it.
