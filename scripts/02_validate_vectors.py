#!/usr/bin/env python3
"""
Step 2: Validate extracted emotion vectors.

Runs corpus validation and sensitivity testing to confirm vectors
encode meaningful emotional content.

Usage:
    python scripts/02_validate_vectors.py
    python scripts/02_validate_vectors.py --layer 15
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.emotion_vectors import load_emotion_vectors
from src.models.hf_backend import HuggingFaceBackend
from src.validation import CorpusValidator, SensitivityTester


def parse_args():
    p = argparse.ArgumentParser(description="Validate emotion vectors")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--model", help="HuggingFace model name")
    p.add_argument("--device", help="Device override")
    p.add_argument("--vectors-dir", default="results/emotion_vectors")
    p.add_argument("--output-dir", default="results/validation")
    p.add_argument("--layer", type=int, help="Layer to validate (default: middle layer)")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = args.model or cfg["model"]["hf_model_name"]
    device = args.device or cfg["model"]["device"]

    print("Loading model...")
    backend = HuggingFaceBackend(
        model_name=model_name,
        device=device,
        dtype=cfg["model"]["dtype"],
    )

    print("Loading emotion vectors...")
    vectors = load_emotion_vectors(args.vectors_dir)
    print(f"Loaded {len(vectors)} emotion vectors.")

    # Default to middle layer
    all_layers = sorted(next(iter(vectors.values())).keys())
    layer = args.layer if args.layer is not None else all_layers[len(all_layers) // 2]
    print(f"Validating at layer {layer}")

    # Corpus validation
    print("\n--- Corpus Validation ---")
    corpus_validator = CorpusValidator(backend, vectors, layer)
    corpus_results = corpus_validator.run_full_validation(output_dir=args.output_dir)
    print(f"Overall top-1 accuracy: {corpus_results['overall_top1_accuracy']:.2%}")
    print(f"Overall mean rank: {corpus_results['overall_mean_rank']:.2f}")

    # Sensitivity testing
    print("\n--- Sensitivity Testing ---")
    sensitivity_tester = SensitivityTester(
        backend=backend,
        emotion_vectors=vectors,
        layer=layer,
        target_emotions=["afraid", "anxious", "desperate", "hopeless", "hopeful", "excited"],
    )
    sensitivity_results = sensitivity_tester.run(output_dir=args.output_dir)
    print(f"Sensitivity accuracy: {sensitivity_results['accuracy']:.2%}")

    print(f"\nValidation results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
