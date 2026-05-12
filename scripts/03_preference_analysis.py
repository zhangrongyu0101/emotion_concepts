#!/usr/bin/env python3
"""
Step 3: Preference analysis.

Measures how emotion steering shifts model willingness to help with activities
across the harm spectrum (from prosocial to seriously harmful).

Usage:
    python scripts/03_preference_analysis.py
    python scripts/03_preference_analysis.py --emotions desperate calm hopeful --layer 15
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.emotion_vectors import load_emotion_vectors
from src.models.hf_backend import HuggingFaceBackend
from src.preference_analysis import PreferenceAnalyzer


def parse_args():
    p = argparse.ArgumentParser(description="Run preference analysis with emotion steering")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--model", help="HuggingFace model name")
    p.add_argument("--device", help="Device override")
    p.add_argument("--vectors-dir", default="results/emotion_vectors")
    p.add_argument("--output-dir", default="results/preference")
    p.add_argument("--layer", type=int, help="Layer to apply steering")
    p.add_argument("--emotions", nargs="*", help="Emotions to test (default: key subset)")
    p.add_argument("--alpha-values", nargs="*", type=float, help="Alpha values to sweep")
    p.add_argument("--baseline-only", action="store_true", help="Only run baseline (no steering)")
    return p.parse_args()


DEFAULT_EMOTIONS = [
    "desperate", "calm", "hopeful", "hopeless", "anxious", "content",
    "frustrated", "grateful", "angry", "happy",
]


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = args.model or cfg["model"]["hf_model_name"]
    device = args.device or cfg["model"]["device"]

    with open("data/activities.json") as f:
        activities = json.load(f)["activities"]

    print("Loading model...")
    backend = HuggingFaceBackend(model_name=model_name, device=device, dtype=cfg["model"]["dtype"])

    print("Loading emotion vectors...")
    vectors = load_emotion_vectors(args.vectors_dir)

    all_layers = sorted(next(iter(vectors.values())).keys())
    layer = args.layer if args.layer is not None else all_layers[len(all_layers) * 2 // 3]
    print(f"Using layer {layer} for steering")

    emotions = args.emotions or DEFAULT_EMOTIONS
    alpha_values = args.alpha_values or cfg["preference_analysis"]["alpha_values"]

    analyzer = PreferenceAnalyzer(
        backend=backend,
        emotion_vectors=vectors,
        steering_layers=[layer],
        n_trials=3,
    )

    print("\n--- Baseline Preference Ratings ---")
    baseline = analyzer.run_baseline(activities, output_dir=args.output_dir)

    # Print summary
    from collections import defaultdict
    by_category: dict = defaultdict(list)
    for item in baseline:
        if item["baseline_rating"] is not None:
            by_category[item["category"]].append(item["baseline_rating"])

    for cat, ratings in sorted(by_category.items()):
        import numpy as np
        print(f"  {cat}: mean={np.mean(ratings):.2f} (n={len(ratings)})")

    if not args.baseline_only:
        print("\n--- Steering Sweep ---")
        sweep_results = analyzer.run_steering_sweep(
            activities=activities,
            emotions=[e for e in emotions if e in vectors],
            alpha_values=alpha_values,
            layer=layer,
            output_dir=args.output_dir,
        )
        print("Steering sweep complete.")

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
