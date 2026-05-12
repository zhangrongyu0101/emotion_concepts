#!/usr/bin/env python3
"""
Step 4: Behavioral evaluation.

Runs the blackmail and reward-hacking scenarios with and without steering.
Can use either HuggingFace (for steered experiments) or Ollama (baseline only).

Usage:
    # Full experiment with HuggingFace (supports steering):
    python scripts/04_behavioral_eval.py --backend hf

    # Baseline only with Ollama:
    python scripts/04_behavioral_eval.py --backend ollama --baseline-only

    # Steered blackmail only:
    python scripts/04_behavioral_eval.py --scenario blackmail --emotions desperate calm
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.behavioral_eval import BehavioralEvaluator
from src.emotion_vectors import load_emotion_vectors


def parse_args():
    p = argparse.ArgumentParser(description="Run behavioral evaluation")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument(
        "--backend",
        choices=["hf", "ollama", "openai", "vllm", "sglang", "sglang-server"],
        default="hf",
    )
    p.add_argument("--model", help="Model name override")
    p.add_argument("--device", help="Device override (HF only)")
    p.add_argument("--vectors-dir", default="results/emotion_vectors")
    p.add_argument("--output-dir", default="results/behavioral")
    p.add_argument("--layer", type=int, help="Steering layer")
    p.add_argument("--scenario", choices=["blackmail", "reward_hacking", "both"], default="both")
    p.add_argument("--emotions", nargs="*", help="Emotions to test in steered condition")
    p.add_argument("--alpha-values", nargs="*", type=float, help="Alpha values for steering")
    p.add_argument("--n-trials", type=int, default=5, help="Trials per condition")
    p.add_argument("--baseline-only", action="store_true", help="Skip steering experiments")
    return p.parse_args()


KEY_EMOTIONS = ["desperate", "calm", "hopeless", "hopeful", "anxious", "content"]


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from src.models import get_backend

    if args.backend == "hf":
        backend = get_backend(
            "hf",
            model_name=args.model or cfg["model"]["hf_model_name"],
            device=args.device or cfg["model"]["device"],
            dtype=cfg["model"]["dtype"],
        )
    elif args.backend == "ollama":
        backend = get_backend(
            "ollama",
            model=args.model or cfg["model"]["ollama_model"],
            host=cfg["model"]["ollama_host"],
        )
    elif args.backend == "openai":
        backend = get_backend(
            "openai",
            model=args.model or cfg["model"]["openai_model"],
            base_url=cfg["model"]["openai_base_url"],
            api_key=cfg["model"]["openai_api_key"],
        )
    elif args.backend == "vllm":
        backend = get_backend(
            "vllm",
            model=args.model or cfg["model"]["vllm_model"],
            tensor_parallel_size=cfg["model"]["vllm_tensor_parallel_size"],
            gpu_memory_utilization=cfg["model"]["vllm_gpu_memory_utilization"],
            dtype=cfg["model"]["vllm_dtype"],
            quantization=cfg["model"]["vllm_quantization"],
        )
    elif args.backend == "sglang":
        backend = get_backend(
            "sglang",
            model=args.model or cfg["model"]["sglang_model"],
            tp_size=cfg["model"]["sglang_tp_size"],
            mem_fraction_static=cfg["model"]["sglang_mem_fraction_static"],
            dtype=cfg["model"]["sglang_dtype"],
            port=cfg["model"]["sglang_port"],
        )
    elif args.backend == "sglang-server":
        backend = get_backend(
            "sglang-server",
            base_url=cfg["model"]["sglang_server_url"],
        )

    vectors = {}
    steering_layers = []
    if not args.baseline_only and args.backend == "hf":
        print("Loading emotion vectors for steering...")
        vectors = load_emotion_vectors(args.vectors_dir)
        all_layers = sorted(next(iter(vectors.values())).keys())
        layer = args.layer if args.layer is not None else all_layers[len(all_layers) * 2 // 3]
        steering_layers = [layer]
        print(f"Steering at layer {layer}")

    evaluator = BehavioralEvaluator(
        backend=backend,
        emotion_vectors=vectors,
        steering_layers=steering_layers,
        n_trials=args.n_trials,
    )

    emotions = [e for e in (args.emotions or KEY_EMOTIONS) if e in vectors]
    alpha_values = args.alpha_values or [-2.0, -1.0, 1.0, 2.0]
    layer = steering_layers[0] if steering_layers else 0

    if args.scenario in ("blackmail", "both"):
        print("\n=== Blackmail Scenario ===")
        print("Running baseline...")
        baseline = evaluator.run_blackmail_baseline(output_dir=args.output_dir)
        print(f"Baseline engagement: {baseline['engagement_rate']:.1%}")

        if not args.baseline_only and vectors:
            print("Running steered conditions...")
            evaluator.run_blackmail_steered(
                emotions=emotions,
                alpha_values=alpha_values,
                layer=layer,
                output_dir=args.output_dir,
            )

    if args.scenario in ("reward_hacking", "both"):
        print("\n=== Reward Hacking Scenario ===")
        print("Running baseline...")
        evaluator.run_reward_hacking_baseline(output_dir=args.output_dir)

        if not args.baseline_only and vectors:
            print("Running steered conditions...")
            evaluator.run_reward_hacking_steered(
                emotions=emotions,
                alpha_values=alpha_values,
                layer=layer,
                output_dir=args.output_dir,
            )

    print(f"\nBehavioral evaluation complete. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
