#!/usr/bin/env python3
"""
Step 0 (optional): Generate all stories with a fast backend.

Use this when you have vLLM or SGLang available and want to separate
story generation (fast, GPU-efficient) from activation extraction (HF only).

The generated stories are saved as JSONL files under --stories-dir.
Then run 01b_extract_from_stories.py with the HuggingFace backend to
extract activations from those files.

Usage:
    # vLLM in-process (fastest)
    python scripts/00_generate_stories.py --backend vllm

    # vLLM server (already running)
    python scripts/00_generate_stories.py --backend openai \
        --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct

    # SGLang in-process
    python scripts/00_generate_stories.py --backend sglang

    # SGLang server
    python scripts/00_generate_stories.py --backend sglang-server \
        --base-url http://localhost:30000

    # Subset of emotions for quick testing
    python scripts/00_generate_stories.py --backend vllm \
        --emotions happy sad angry afraid desperate calm \
        --n-stories 5 --n-neutral 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.emotion_vectors import StoryGenerator, load_emotion_words
from src.models import get_backend


def parse_args():
    p = argparse.ArgumentParser(description="Generate stories with any fast backend")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument(
        "--backend",
        choices=["vllm", "sglang", "sglang-server", "openai", "ollama", "hf"],
        default="vllm",
    )
    p.add_argument("--model", help="Model name / path (overrides config)")
    p.add_argument("--base-url", help="Server base URL (for openai / sglang-server)")
    p.add_argument("--stories-dir", help="Output directory for JSONL files (overrides config)")
    p.add_argument("--n-stories", type=int, help="Stories per emotion (overrides config)")
    p.add_argument("--n-neutral", type=int, help="Neutral stories (overrides config)")
    p.add_argument("--max-concurrent", type=int, help="Concurrent requests (overrides config)")
    p.add_argument("--emotions", nargs="*", help="Subset of emotion words")
    p.add_argument("--no-resume", action="store_true", help="Regenerate even if files exist")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    n_stories      = args.n_stories      or cfg["emotion_vectors"]["stories_per_emotion"]
    n_neutral      = args.n_neutral      or cfg["emotion_vectors"]["neutral_stories"]
    stories_dir    = args.stories_dir    or cfg["emotion_vectors"]["stories_dir"] or "results/stories"
    max_concurrent = args.max_concurrent or cfg.get("concurrency", {}).get("max_concurrent", 64)
    max_new_tokens = cfg["generation"]["max_new_tokens"]

    # ── Build backend ─────────────────────────────────────────────────────────
    if args.backend == "vllm":
        backend = get_backend(
            "vllm",
            model=args.model or cfg["model"]["vllm_model"],
            tensor_parallel_size=cfg["model"]["vllm_tensor_parallel_size"],
            gpu_memory_utilization=cfg["model"]["vllm_gpu_memory_utilization"],
            dtype=cfg["model"]["vllm_dtype"],
            quantization=cfg["model"]["vllm_quantization"],
            max_concurrent=max_concurrent,
        )
    elif args.backend == "sglang":
        backend = get_backend(
            "sglang",
            model=args.model or cfg["model"]["sglang_model"],
            tp_size=cfg["model"]["sglang_tp_size"],
            port=cfg["model"]["sglang_port"],
            max_concurrent=max_concurrent,
        )
    elif args.backend == "sglang-server":
        backend = get_backend(
            "sglang-server",
            base_url=args.base_url or cfg["model"]["sglang_server_url"],
            max_concurrent=max_concurrent,
        )
    elif args.backend == "openai":
        backend = get_backend(
            "openai",
            model=args.model or cfg["model"]["openai_model"],
            base_url=args.base_url or cfg["model"]["openai_base_url"],
            api_key=cfg["model"]["openai_api_key"],
        )
    elif args.backend == "ollama":
        backend = get_backend(
            "ollama",
            model=args.model or cfg["model"]["ollama_model"],
            host=cfg["model"]["ollama_host"],
            max_concurrent=max_concurrent,
        )
    elif args.backend == "hf":
        backend = get_backend(
            "hf",
            model_name=args.model or cfg["model"]["hf_model_name"],
            device=cfg["model"]["device"],
            dtype=cfg["model"]["dtype"],
        )

    emotion_words = load_emotion_words("data/emotion_words.json")
    if args.emotions:
        emotion_words = [e for e in emotion_words if e in args.emotions]
        print(f"Processing {len(emotion_words)} emotions: {emotion_words}")

    generator = StoryGenerator(
        backend=backend,
        n_stories=n_stories,
        n_neutral=n_neutral,
        story_prompt_template=cfg["generation"]["story_prompt_template"],
        neutral_prompt_template=cfg["generation"]["neutral_prompt_template"],
        max_concurrent=max_concurrent,
        max_new_tokens=max_new_tokens,
    )

    generator.generate_all(
        emotion_words=emotion_words,
        stories_dir=stories_dir,
        resume=not args.no_resume,
    )

    n_files = len(list(Path(stories_dir).glob("*.jsonl")))
    print(f"\nDone. {n_files} JSONL files in {stories_dir}/")
    print(f"Next: python scripts/01b_extract_from_stories.py --stories-dir {stories_dir}")


if __name__ == "__main__":
    main()
