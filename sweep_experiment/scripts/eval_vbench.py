#!/usr/bin/env python3
"""
Run VBench++ evaluation on a directory of generated videos.

Evaluates key quality dimensions relevant to video continuation:
  - subject_consistency: does continuation match conditioning content?
  - motion_smoothness: smooth transition from conditioned to generated?
  - temporal_flickering: artifacts in generated frames?
  - aesthetic_quality: overall visual quality
  - imaging_quality: per-frame sharpness and detail

Requires: pip install vbench

Usage:
    python sweep_experiment/scripts/eval_vbench.py \
        --video-dir sweep_experiment/results/full_lr_sweep/F3/videos \
        --output sweep_experiment/results/full_lr_sweep/F3/vbench_scores.json \
        --dimensions subject_consistency motion_smoothness temporal_flickering
"""
import argparse, json, os, sys
from pathlib import Path


DEFAULT_DIMENSIONS = [
    "subject_consistency",
    "motion_smoothness",
    "temporal_flickering",
    "aesthetic_quality",
    "imaging_quality",
]


def run_vbench_evaluation(video_dir: str, dimensions: list, 
                           output_path: str, mode: str = "i2v"):
    """Run VBench++ evaluation on videos."""
    try:
        from vbench import VBench
    except ImportError:
        print("ERROR: vbench not installed. Run: pip install vbench", file=sys.stderr)
        sys.exit(1)

    video_files = sorted(Path(video_dir).glob("*.mp4"))
    if not video_files:
        print(f"ERROR: No .mp4 files found in {video_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(video_files)} videos in {video_dir}")
    print(f"Evaluating dimensions: {dimensions}")
    print(f"Mode: {mode}")

    # Initialize VBench
    vbench = VBench(device="cuda", full_json_dir=None)

    # Prepare video paths
    video_paths = [str(vp) for vp in video_files]

    results = {}
    for dim in dimensions:
        print(f"\nEvaluating: {dim}...")
        try:
            score = vbench.evaluate(
                videos_path=video_paths,
                name=dim,
                dimension_list=[dim],
                mode=mode,
            )
            results[dim] = float(score) if isinstance(score, (int, float)) else score
            print(f"  {dim}: {results[dim]}")
        except Exception as e:
            print(f"  WARNING: Failed to evaluate {dim}: {e}", file=sys.stderr)
            results[dim] = None

    # Save results
    output = {
        "video_dir": video_dir,
        "num_videos": len(video_files),
        "mode": mode,
        "scores": results,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="VBench++ video quality evaluation")
    parser.add_argument("--video-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dimensions", nargs="+", default=DEFAULT_DIMENSIONS,
                        help=f"Evaluation dimensions (default: {DEFAULT_DIMENSIONS})")
    parser.add_argument("--mode", type=str, default="i2v",
                        choices=["t2v", "i2v"],
                        help="Evaluation mode (i2v for continuation)")
    args = parser.parse_args()

    run_vbench_evaluation(args.video_dir, args.dimensions, args.output, args.mode)


if __name__ == "__main__":
    main()
