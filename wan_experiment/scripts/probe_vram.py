#!/usr/bin/env python3
"""Dump what occupies the H200 after a Self-Forcing pipeline load.

Do not generate video. Writes wan_experiment/results/setup_healthcheck/vram_probe.json
and prints torch.cuda.memory_summary plus the largest CUDA tensors.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _tensor_report(limit: int = 25):
    import torch
    import gc
    rows = []
    for obj in gc.get_objects():
        try:
            if not torch.is_tensor(obj) or not obj.is_cuda:
                continue
            rows.append({
                "mb": round(obj.numel() * obj.element_size() / 1e6, 2),
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
            })
        except Exception:
            continue
    rows.sort(key=lambda r: r["mb"], reverse=True)
    return rows[:limit]


def main() -> int:
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

    scratch = Path(os.environ.get("SCRATCH_BASE", f"/scratch/{os.environ['USER']}"))
    project = scratch / "longcat-video-tta"
    sys.path.insert(0, str(project))
    from wan_experiment.scripts.run_i2v_continuation import (
        FRAME_SEQ_PER_LATENT,
        _cuda_mem,
        enlarge_kv_cache,
        ensure_wan_symlink,
        install_sdpa_attention_fallback,
        load_pipeline,
    )

    sf_root = scratch / "third_party" / "Self-Forcing"
    wan_dir = scratch / "wan-checkpoints" / "Wan2.1-T2V-1.3B"
    sf_ckpt = scratch / "wan-checkpoints" / "self_forcing_dmd.pt"
    out = project / "wan_experiment" / "results" / "setup_healthcheck" / "vram_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(sf_root))
    os.chdir(sf_root)

    import torch
    torch._dynamo.config.disable = True
    device = torch.device("cuda")
    print("torch", torch.__version__, "gpu", torch.cuda.get_device_name(0))
    _cuda_mem("empty")

    install_sdpa_attention_fallback()
    try:
        import wan.modules.causal_model as _cm
        from torch.nn.attention.flex_attention import flex_attention as _eager_fa
        _cm.flex_attention = _eager_fa
        print("flex_attention: eager")
    except Exception as e:
        print("flex_attention leave-as-is", type(e).__name__, e)
    _cuda_mem("after_import")

    ensure_wan_symlink(sf_root, wan_dir)
    n_frames = 24
    pipeline = load_pipeline(sf_root, wan_dir, sf_ckpt, device, n_cache_frames=n_frames)
    _cuda_mem("after_load_fn")

    # Force the same cache init inference() uses.
    pipeline._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=device)
    pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=device)
    k0 = pipeline.kv_cache1[0]["k"]
    print(f"kv_cache k shape={tuple(k0.shape)} dtype={k0.dtype} "
          f"FRAME_SEQ_PER_LATENT={FRAME_SEQ_PER_LATENT} "
          f"n_frames={n_frames} "
          f"pipeline.frame_seq_length={getattr(pipeline, 'frame_seq_length', None)}")
    kv_gb = (
        len(pipeline.kv_cache1) * 2 * k0.numel() * k0.element_size() / 1e9
    )
    print(f"kv_cache total {kv_gb:.2f} GB")
    _cuda_mem("after_forced_kv_init")

    param_gb = sum(p.numel() * p.element_size() for p in pipeline.parameters()) / 1e9
    print(f"pipeline parameters {param_gb:.2f} GB")

    top = _tensor_report(20)
    print("top CUDA tensors (MB):")
    for r in top:
        print(f"  {r['mb']:10.1f}  {r['dtype']:18s}  {r['shape']}")

    summary = torch.cuda.memory_summary()
    print(summary)

    report = {
        "kv_k_shape": list(k0.shape),
        "kv_gb": round(kv_gb, 3),
        "param_gb": round(param_gb, 3),
        "frame_seq_per_latent": FRAME_SEQ_PER_LATENT,
        "pipeline_frame_seq_length": getattr(pipeline, "frame_seq_length", None),
        "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 3),
        "top_tensors": top,
        "memory_summary": summary[-4000:],
    }
    out.write_text(json.dumps(report, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
