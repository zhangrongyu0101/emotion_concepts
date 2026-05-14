#!/usr/bin/env python3
"""
Step 1b: Extract emotion vectors from pre-generated stories.

Use this after running 00_generate_stories.py with a fast backend (vLLM, SGLang).
This script loads the saved JSONL files and runs activation extraction using
the HuggingFace backend, which is the only one that exposes internal activations.

This is the recommended two-backend workflow for GPU servers:
  Step 0: python scripts/00_generate_stories.py --backend vllm  (fast, concurrent)
  Step 1b: python scripts/01b_extract_from_stories.py           (HF activations)

Usage:
    python scripts/01b_extract_from_stories.py
    python scripts/01b_extract_from_stories.py --stories-dir results/stories
    python scripts/01b_extract_from_stories.py --model Qwen/Qwen2.5-7B-Instruct
    python scripts/01b_extract_from_stories.py --no-resume  # recompute all vectors
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.emotion_vectors import EmotionVectorExtractor
from src.models import get_backend


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract emotion vectors from pre-generated JSONL stories (HF backend)"
    )
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--model", help="HuggingFace model name (overrides config)")
    p.add_argument("--device", help="Device: cuda / mps / cpu (overrides config)")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"])
    p.add_argument(
        "--stories-dir",
        default=None,
        help="Directory containing <emotion>.jsonl files (overrides config stories_dir)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Where to save .pt vector files (overrides config output_dir)",
    )
    p.add_argument("--no-resume", action="store_true", help="Recompute even if .pt files exist")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = args.model or cfg["model"]["hf_model_name"]
    device     = args.device or cfg["model"]["device"]
    dtype      = args.dtype  or cfg["model"]["dtype"]
    stories_dir = args.stories_dir or cfg["emotion_vectors"].get("stories_dir") or "results/stories"
    output_dir  = args.output_dir  or cfg["emotion_vectors"]["output_dir"]
    aggregation  = cfg["emotion_vectors"]["aggregation"]
    target_layers = cfg["emotion_vectors"]["target_layers"]

    stories_path = Path(stories_dir)
    if not stories_path.exists():
        print(f"ERROR: stories-dir '{stories_dir}' does not exist.")
        print("Run 00_generate_stories.py first to generate stories.")
        sys.exit(1)

    jsonl_files = list(stories_path.glob("*.jsonl"))
    if not jsonl_files:
        print(f"ERROR: No .jsonl files found in '{stories_dir}'.")
        print("Run 00_generate_stories.py first to generate stories.")
        sys.exit(1)

    n_emotions = sum(1 for f in jsonl_files if f.stem != "neutral")
    print(f"Found {len(jsonl_files)} JSONL files ({n_emotions} emotions + neutral) in {stories_dir}/")

    gpu_ids = cfg["model"].get("hf_gpu_ids")
    if gpu_ids:
        print(f"\nMulti-GPU pool: {len(gpu_ids)} replica(s) of {model_name} on GPUs {gpu_ids}")
    else:
        print(f"\nLoading {model_name} on {device} ({dtype})")
    backend = get_backend(
        "hf",
        model_name=model_name,
        gpu_ids=gpu_ids,
        device=device,
        dtype=dtype,
    )

    extractor = EmotionVectorExtractor(
        backend=backend,
        aggregation=aggregation,
        target_layers=target_layers,
    )

    print(f"\nExtracting activations from {stories_dir}/ → {output_dir}/")
    vectors = extractor.extract_from_stories(
        stories_dir=stories_dir,
        output_dir=output_dir,
        resume=not args.no_resume,
    )

    print(f"\nDone. Extracted {len(vectors)} emotion vectors.")
    print(f"Layers per vector: {sorted(next(iter(vectors.values())).keys())[:5]}...")
    print(f"Hidden size: {backend.hidden_size}")
    print(f"Vectors saved to: {output_dir}/")
    print(f"\nNext steps:")
    print(f"  python scripts/02_validate_vectors.py")
    print(f"  python scripts/05_steering_experiment.py")


if __name__ == "__main__":
    main()
