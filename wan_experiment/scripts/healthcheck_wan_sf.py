#!/usr/bin/env python3
"""GPU health check for the Wan2.1-1.3B + Self-Forcing overnight setup.

Writes a JSON report. Exit 0 only if the *required* artifacts load; VBench-I2V
images and a 1-clip smoke gen are recorded but do not fail the job if Drive
rate-limited the image download (models are the blocker).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


def _ok(name, **extra):
    return {"name": name, "ok": True, **extra}


def _fail(name, err, **extra):
    return {"name": name, "ok": False, "error": err, **extra}


def check_files(wan_dir: Path, sf_ckpt: Path) -> list:
    rows = []
    cfg = wan_dir / "config.json"
    rows.append(_ok("wan_config", path=str(cfg)) if cfg.is_file()
                else _fail("wan_config", f"missing {cfg}"))
    t5 = list(wan_dir.glob("models_t5*.pth")) + list(wan_dir.glob("*t5*"))
    rows.append(_ok("wan_t5", n=len(t5), bytes=sum(p.stat().st_size for p in t5 if p.is_file()))
                if t5 else _fail("wan_t5", "no T5 weight file under Wan dir"))
    vae = list(wan_dir.glob("*vae*")) + list(wan_dir.glob("*VAE*"))
    rows.append(_ok("wan_vae", n=len(vae)) if vae else _fail("wan_vae", "no VAE file under Wan dir"))
    if sf_ckpt.is_file() and sf_ckpt.stat().st_size > 10_000_000:
        rows.append(_ok("sf_ckpt", path=str(sf_ckpt), bytes=sf_ckpt.stat().st_size))
    else:
        rows.append(_fail("sf_ckpt", f"missing or tiny: {sf_ckpt}"))
    return rows


def check_i2v_images(i2v_dir: Path, n_try: int = 8) -> dict:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    imgs = [p for p in i2v_dir.rglob("*") if p.suffix.lower() in exts]
    decoded = 0
    err = None
    if imgs:
        try:
            from PIL import Image
            for p in imgs[:n_try]:
                with Image.open(p) as im:
                    im.convert("RGB").size
                decoded += 1
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
    return {
        "name": "vbench_i2v_images",
        "ok": decoded > 0,
        "n_found": len(imgs),
        "n_decoded": decoded,
        "error": err,
        "required": False,
    }


def check_torch_load(wan_dir: Path, sf_ckpt: Path, device: str) -> list:
    rows = []
    try:
        import torch
        rows.append(_ok("torch", version=torch.__version__,
                        cuda=bool(torch.cuda.is_available()),
                        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
    except Exception as e:
        return [_fail("torch", f"{type(e).__name__}: {e}")]

    # Load SF checkpoint onto CPU first (maps), then peek keys.
    try:
        ckpt = torch.load(sf_ckpt, map_location="cpu", weights_only=False)
        keys = list(ckpt.keys()) if isinstance(ckpt, dict) else [type(ckpt).__name__]
        n_tensors = 0
        if isinstance(ckpt, dict):
            inner = ckpt.get("state_dict", ckpt.get("model", ckpt))
            if isinstance(inner, dict):
                n_tensors = sum(1 for v in inner.values() if hasattr(v, "shape"))
        rows.append(_ok("sf_ckpt_load", top_keys=keys[:12], n_tensors=n_tensors))
        del ckpt
    except Exception as e:
        rows.append(_fail("sf_ckpt_load", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))

    # Minimal Wan config / tokenizer presence (full DiT load is the smoke gen).
    try:
        cfg_path = wan_dir / "config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
        rows.append(_ok("wan_config_parse", keys=list(cfg)[:16],
                        model_type=cfg.get("model_type") or cfg.get("_class_name")))
    except Exception as e:
        rows.append(_fail("wan_config_parse", f"{type(e).__name__}: {e}"))
    return rows


def smoke_t2v(sf_root: Path, wan_dir: Path, sf_ckpt: Path, out_mp4: Path) -> dict:
    """Best-effort 1-clip generation via Self-Forcing inference.py if present."""
    infer = sf_root / "inference.py"
    cfg = sf_root / "configs" / "self_forcing_dmd.yaml"
    if not infer.is_file():
        return {"name": "smoke_t2v", "ok": False, "required": False,
                "error": f"no {infer}"}
    import subprocess
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    prompt_file = out_mp4.parent / "smoke_prompt.txt"
    prompt_file.write_text("A golden retriever running across a grassy field, cinematic, 4k.\n")
    cmd = [
        sys.executable, str(infer),
        "--config_path", str(cfg) if cfg.is_file() else "",
        "--output_folder", str(out_mp4.parent),
        "--checkpoint_path", str(sf_ckpt),
        "--data_path", str(prompt_file),
        "--use_ema",
    ]
    cmd = [c for c in cmd if c]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(sf_root), capture_output=True, text=True, timeout=2400)
        mp4s = list(out_mp4.parent.glob("*.mp4"))
        return {
            "name": "smoke_t2v",
            "ok": p.returncode == 0 and bool(mp4s),
            "required": False,
            "rc": p.returncode,
            "seconds": round(time.time() - t0, 1),
            "n_mp4": len(mp4s),
            "stdout_tail": (p.stdout or "")[-1500:],
            "stderr_tail": (p.stderr or "")[-1500:],
        }
    except Exception as e:
        return {"name": "smoke_t2v", "ok": False, "required": False,
                "error": f"{type(e).__name__}: {e}",
                "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wan-dir", required=True)
    ap.add_argument("--sf-ckpt", required=True)
    ap.add_argument("--sf-root", required=True)
    ap.add_argument("--i2v-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-gen", action="store_true")
    args = ap.parse_args()

    wan_dir, sf_ckpt = Path(args.wan_dir), Path(args.sf_ckpt)
    sf_root, i2v_dir = Path(args.sf_root), Path(args.i2v_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    checks = []
    checks.extend(check_files(wan_dir, sf_ckpt))
    checks.append(check_i2v_images(i2v_dir))
    checks.extend(check_torch_load(wan_dir, sf_ckpt, "cuda"))
    if not args.skip_gen:
        checks.append(smoke_t2v(sf_root, wan_dir, sf_ckpt, out.parent / "smoke" / "out.mp4"))

    required_fail = [c for c in checks if c.get("required", True) and not c.get("ok")]
    report = {
        "ok": not required_fail,
        "n_fail_required": len(required_fail),
        "checks": checks,
    }
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if required_fail:
        print("REQUIRED CHECKS FAILED:", [c["name"] for c in required_fail], file=sys.stderr)
        return 2
    print("All required checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
