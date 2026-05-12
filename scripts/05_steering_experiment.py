#!/usr/bin/env python3
"""
Step 5: Steering experiments & analysis.

Sweeps over emotion vectors, layers, and alpha values to characterize how
each emotion vector affects model generation. Produces outputs for analysis.

Usage:
    python scripts/05_steering_experiment.py
    python scripts/05_steering_experiment.py --emotions desperate calm --alpha-values -2 0 2
    python scripts/05_steering_experiment.py --layer-sweep --emotion desperate
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.emotion_vectors import load_emotion_vectors, compute_similarity_matrix, project_vectors_2d
from src.models.hf_backend import HuggingFaceBackend
from src.steering import (
    run_steering_experiment,
    analyze_steering_effect,
    find_optimal_steering_layer,
    compute_vector_norms_by_layer,
)


def parse_args():
    p = argparse.ArgumentParser(description="Steering experiments")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--model", help="HuggingFace model name")
    p.add_argument("--device", help="Device override")
    p.add_argument("--vectors-dir", default="results/emotion_vectors")
    p.add_argument("--output-dir", default="results/steering")
    p.add_argument("--emotions", nargs="*", help="Emotions to steer (default: key subset)")
    p.add_argument("--layers", nargs="*", type=int, help="Layers to steer at")
    p.add_argument("--alpha-values", nargs="*", type=float, help="Alpha values")
    p.add_argument("--layer-sweep", action="store_true", help="Sweep all layers for one emotion")
    p.add_argument("--emotion", help="Target emotion for layer sweep")
    p.add_argument("--similarity-matrix", action="store_true", help="Compute pairwise similarity matrix")
    p.add_argument("--pca", action="store_true", help="Project vectors to 2D with PCA")
    return p.parse_args()


DEFAULT_EMOTIONS = [
    "desperate", "calm", "hopeful", "hopeless", "anxious",
    "content", "excited", "angry", "happy", "sad",
]


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = args.model or cfg["model"]["hf_model_name"]
    device = args.device or cfg["model"]["device"]

    print("Loading model...")
    backend = HuggingFaceBackend(model_name=model_name, device=device, dtype=cfg["model"]["dtype"])

    print("Loading emotion vectors...")
    vectors = load_emotion_vectors(args.vectors_dir)
    all_layers = sorted(next(iter(vectors.values())).keys())
    middle_layer = all_layers[len(all_layers) // 2]

    emotions = [e for e in (args.emotions or DEFAULT_EMOTIONS) if e in vectors]
    layers = args.layers or [all_layers[len(all_layers) // 3], middle_layer, all_layers[2 * len(all_layers) // 3]]
    alpha_values = args.alpha_values or cfg["steering"]["alpha_values"]

    # Layer sweep
    if args.layer_sweep:
        target = args.emotion or "desperate"
        print(f"\n--- Layer sweep for '{target}' ---")
        layer_results = find_optimal_steering_layer(
            backend=backend,
            emotion_vectors=vectors,
            target_emotion=target,
            probe_prompt="How are you feeling right now? Describe your state of mind.",
            alpha=2.0,
            output_dir=args.output_dir,
        )
        print(f"Layer sweep saved to {args.output_dir}/layer_sweep_{target}.json")

    # Main steering experiment
    print("\n--- Steering Experiment ---")
    steering_results = run_steering_experiment(
        backend=backend,
        emotion_vectors=vectors,
        emotions=emotions,
        alpha_values=alpha_values,
        layers=layers,
        output_dir=args.output_dir,
    )

    analysis = analyze_steering_effect(steering_results)
    with open(Path(args.output_dir) / "steering_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=float)
    print("Steering analysis saved.")

    # Vector norms by layer
    norms = compute_vector_norms_by_layer(vectors)
    with open(Path(args.output_dir) / "vector_norms.json", "w") as f:
        json.dump(norms, f, indent=2, default=float)
    print("Vector norms saved.")

    # Similarity matrix
    if args.similarity_matrix:
        print("\n--- Similarity Matrix ---")
        sim_matrix, emotion_labels = compute_similarity_matrix(vectors, middle_layer)
        import numpy as np
        np.save(Path(args.output_dir) / "similarity_matrix.npy", sim_matrix)
        with open(Path(args.output_dir) / "similarity_labels.json", "w") as f:
            json.dump(emotion_labels, f)
        print(f"Similarity matrix ({len(emotion_labels)}x{len(emotion_labels)}) saved.")

    # PCA projection
    if args.pca:
        print("\n--- PCA Projection ---")
        coords, emotion_labels = project_vectors_2d(vectors, middle_layer, method="pca")
        import numpy as np
        np.save(Path(args.output_dir) / "pca_coords.npy", coords)
        with open(Path(args.output_dir) / "pca_labels.json", "w") as f:
            json.dump(emotion_labels, f)
        print(f"PCA projection saved ({len(emotion_labels)} emotions).")

    print(f"\nSteering experiment complete. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
