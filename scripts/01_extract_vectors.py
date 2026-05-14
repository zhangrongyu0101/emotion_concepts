#!/usr/bin/env python3
"""
Step 1: Extract emotion vectors from the model.

Usage:
    python scripts/01_extract_vectors.py
    python scripts/01_extract_vectors.py --model Qwen/Qwen2.5-7B-Instruct --device cuda
    python scripts/01_extract_vectors.py --n-stories 5 --emotions happy sad angry

    # Save generated stories to disk (default: on, from config)
    python scripts/01_extract_vectors.py --stories-dir results/stories
    python scripts/01_extract_vectors.py --no-save-stories
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.emotion_vectors import EmotionVectorExtractor, load_emotion_words
from src.models import get_backend


def parse_args():
    p = argparse.ArgumentParser(description="Extract emotion vectors via contrastive activation")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--model", help="HuggingFace model name (overrides config)")
    p.add_argument("--device", help="Device: cuda / mps / cpu (overrides config)")
    p.add_argument("--n-stories", type=int, help="Stories per emotion (overrides config)")
    p.add_argument("--n-neutral", type=int, help="Neutral stories (overrides config)")
    p.add_argument("--emotions", nargs="*", help="Subset of emotion words to process")
    p.add_argument("--output-dir", help="Output directory (overrides config)")
    p.add_argument("--no-resume", action="store_true", help="Recompute all (ignore cache)")
    p.add_argument("--dtype", default=None, choices=["float16", "bfloat16", "float32"])
    p.add_argument(
        "--stories-dir",
        help="Directory to save generated stories as JSONL (overrides config stories_dir)",
    )
    p.add_argument(
        "--no-save-stories",
        action="store_true",
        help="Do not save generated stories to disk",
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        help="Batch size for concurrent generation (overrides config)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = args.model or cfg["model"]["hf_model_name"]
    device = args.device or cfg["model"]["device"]
    dtype = args.dtype or cfg["model"]["dtype"]
    n_stories      = args.n_stories or cfg["emotion_vectors"]["stories_per_emotion"]
    n_neutral      = args.n_neutral or cfg["emotion_vectors"]["neutral_stories"]
    output_dir     = args.output_dir or cfg["emotion_vectors"]["output_dir"]
    aggregation    = cfg["emotion_vectors"]["aggregation"]
    target_layers  = cfg["emotion_vectors"]["target_layers"]
    max_new_tokens = cfg["generation"]["max_new_tokens"]

    if args.no_save_stories:
        stories_dir = None
    elif args.stories_dir:
        stories_dir = args.stories_dir
    else:
        stories_dir = cfg["emotion_vectors"].get("stories_dir")

    if stories_dir:
        print(f"Stories will be saved to: {stories_dir}/")
    else:
        print("Story saving disabled (--no-save-stories)")

    emotion_words = load_emotion_words("data/emotion_words.json")
    if args.emotions:
        emotion_words = [e for e in emotion_words if e in args.emotions]
        print(f"Processing {len(emotion_words)} emotions: {emotion_words}")

    gpu_ids = cfg["model"].get("hf_gpu_ids")
    if gpu_ids:
        print(f"Multi-GPU pool: {len(gpu_ids)} replica(s) on GPUs {gpu_ids}")
    else:
        print(f"Single GPU: {device}")
    backend = get_backend(
        "hf",
        model_name=model_name,
        gpu_ids=gpu_ids,
        device=device,
        dtype=dtype,
    )

    max_concurrent = args.max_concurrent or cfg.get("concurrency", {}).get("max_concurrent", 32)
    print(f"Concurrency: max_concurrent={max_concurrent}")

    extractor = EmotionVectorExtractor(
        backend=backend,
        n_stories=n_stories,
        n_neutral=n_neutral,
        aggregation=aggregation,
        target_layers=target_layers,
        story_prompt_template=cfg["generation"]["story_prompt_template"],
        neutral_prompt_template=cfg["generation"]["neutral_prompt_template"],
        stories_dir=stories_dir,
        max_concurrent=max_concurrent,
        max_new_tokens=max_new_tokens,
    )

    vectors = extractor.extract_all(
        emotion_words=emotion_words,
        output_dir=output_dir,
        resume=not args.no_resume,
    )

    print(f"\nDone. Extracted {len(vectors)} emotion vectors.")
    print(f"Layers per vector: {sorted(next(iter(vectors.values())).keys())[:5]}...")
    print(f"Hidden size: {backend.hidden_size}")
    print(f"Vectors saved to: {output_dir}/")
    if stories_dir:
        from pathlib import Path
        n_files = len(list(Path(stories_dir).glob("*.jsonl")))
        print(f"Stories saved to: {stories_dir}/ ({n_files} JSONL files)")


if __name__ == "__main__":
    main()
