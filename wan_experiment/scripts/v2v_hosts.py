"""Load LongLive / Rolling Forcing students for V2V. No TTC.

LongLive-1.3B is a Self-Forcing fork with sink_size=3, local_attn_size=12,
and a LoRA on top of longlive_base.pt. Rolling Forcing uses a different
sampler (inference_rolling_forcing) and kv_cache_clean.
"""
from __future__ import annotations

from pathlib import Path

from run_i2v_continuation import (
    enlarge_kv_cache,
    ensure_wan_symlink,
    _cuda_mem,
)


def apply_sink_size(pipeline, sink_size: int, local_attn_size: int | None = None) -> dict:
    info = {"sink_size": int(sink_size)}
    if local_attn_size is not None:
        info["local_attn_size"] = int(local_attn_size)
    model = getattr(getattr(pipeline, "generator", None), "model", None)
    targets = []
    if model is not None:
        targets.append(model)
        for mod in model.modules():
            targets.append(mod)
    for mod in targets:
        if hasattr(mod, "sink_size"):
            mod.sink_size = int(sink_size)
        if local_attn_size is not None and hasattr(mod, "local_attn_size"):
            mod.local_attn_size = int(local_attn_size)
        if local_attn_size is not None and hasattr(mod, "max_attention_size"):
            try:
                frame_seq = int(getattr(pipeline, "frame_seq_length", 1560))
                mod.max_attention_size = int(local_attn_size) * frame_seq
            except Exception:
                pass
    print("host_sink:", info, flush=True)
    return info


def _swap_text_encoder(pipeline, device) -> None:
    try:
        from utils.memory import DynamicSwapInstaller
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
        print("text_encoder: DynamicSwapInstaller (utils.memory)")
        return
    except Exception as e:
        print(f"text_encoder: utils.memory skip ({type(e).__name__})")
    try:
        from demo_utils.memory import DynamicSwapInstaller
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
        print("text_encoder: DynamicSwapInstaller (demo_utils)")
        return
    except Exception as e:
        pipeline.text_encoder.to(device=device)
        print(f"text_encoder: full GPU ({type(e).__name__}: {e})")


def load_longlive_pipeline(
    ll_root: Path,
    wan_dir: Path,
    base_ckpt: Path,
    lora_ckpt: Path,
    device,
    n_cache_frames: int,
    sink_size: int = 3,
    local_attn_size: int = 12,
):
    import torch
    from omegaconf import OmegaConf
    from pipeline import CausalInferencePipeline

    ensure_wan_symlink(ll_root, wan_dir)
    cfg_path = ll_root / "configs" / "longlive_inference.yaml"
    config = OmegaConf.load(str(cfg_path))
    config.model_kwargs.sink_size = int(sink_size)
    config.model_kwargs.local_attn_size = int(local_attn_size)
    config.generator_ckpt = str(base_ckpt)
    config.lora_ckpt = str(lora_ckpt)
    config.use_ema = False
    config.context_noise = int(getattr(config, "context_noise", 0) or 0)
    config.global_sink = True
    if not hasattr(config, "independent_first_frame"):
        config.independent_first_frame = False
    else:
        config.independent_first_frame = False

    print(
        f"longlive load sink={sink_size} local_attn={local_attn_size} "
        f"base={base_ckpt} lora={lora_ckpt}",
        flush=True,
    )
    pipeline = CausalInferencePipeline(config, device=device)
    state = torch.load(str(base_ckpt), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and (
        "generator" in state or "generator_ema" in state
    ):
        raw = state["generator_ema" if config.use_ema else "generator"]
        raw = {
            k.replace("_fsdp_wrapped_module.", ""): v for k, v in raw.items()
        }
        missing, unexpected = pipeline.generator.load_state_dict(raw, strict=False)
        print(
            f"longlive base: missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    elif isinstance(state, dict) and "model" in state:
        pipeline.generator.load_state_dict(state["model"], strict=False)
    else:
        pipeline.generator.load_state_dict(state, strict=False)
    del state

    from utils.lora_utils import configure_lora_for_model
    import peft

    adapter = config.adapter
    if OmegaConf.is_config(adapter):
        adapter = OmegaConf.to_container(adapter, resolve=True)
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=adapter,
        is_main_process=True,
    )
    lora_state = torch.load(str(lora_ckpt), map_location="cpu", weights_only=False)
    if isinstance(lora_state, dict) and "generator_lora" in lora_state:
        peft.set_peft_model_state_dict(
            pipeline.generator.model, lora_state["generator_lora"],
        )
    else:
        peft.set_peft_model_state_dict(pipeline.generator.model, lora_state)
    del lora_state
    print("longlive LoRA loaded", flush=True)

    pipeline = pipeline.to(dtype=torch.bfloat16)
    _swap_text_encoder(pipeline, device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    enlarge_kv_cache(pipeline, n_cache_frames)
    apply_sink_size(pipeline, sink_size, local_attn_size)
    _cuda_mem("after_longlive_load")
    return pipeline


def load_rolling_pipeline(
    rf_root: Path,
    wan_dir: Path,
    rf_ckpt: Path,
    device,
    n_cache_frames: int,
):
    import torch
    from omegaconf import OmegaConf
    from pipeline import CausalInferencePipeline

    ensure_wan_symlink(rf_root, wan_dir)
    default_cfg = OmegaConf.load(str(rf_root / "configs" / "default_config.yaml"))
    dmd_cfg = OmegaConf.load(str(rf_root / "configs" / "rolling_forcing_dmd.yaml"))
    config = OmegaConf.merge(default_cfg, dmd_cfg)
    config.independent_first_frame = False
    if not hasattr(config, "context_noise"):
        config.context_noise = 0

    pipeline = CausalInferencePipeline(config, device=device)
    state = torch.load(str(rf_ckpt), map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise KeyError(f"unexpected Rolling Forcing ckpt type {type(state)}")
    if "generator_ema" in state:
        raw = state["generator_ema"]
    elif "generator" in state:
        raw = state["generator"]
    else:
        raise KeyError(f"RF ckpt keys={list(state)[:12]}")
    raw = {k.replace("_fsdp_wrapped_module.", ""): v for k, v in raw.items()}
    pipeline.generator.load_state_dict(raw)
    del state

    pipeline = pipeline.to(dtype=torch.bfloat16)
    _swap_text_encoder(pipeline, device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    enlarge_kv_cache(pipeline, n_cache_frames)
    _cuda_mem("after_rolling_load")
    return pipeline
