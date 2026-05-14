#!/usr/bin/env python3
"""
Step 6: AIME mathematical reasoning evaluation with emotion steering.

Tests whether different emotion vectors affect mathematical reasoning.
Compares baseline (no steering) against multiple emotion × alpha conditions.

Workflow:
    1. Load AIME dataset (auto-downloads from HuggingFace if not cached locally)
    2. Run baseline inference with greedy decoding
    3. For each emotion × alpha, run steered inference at the specified layer
    4. Extract integer answers (0-999), compare to ground truth
    5. Print a comparison table and save detailed results as JSONL

Usage:
    # Baseline only (no vectors needed):
    python scripts/06_aime_eval.py --baseline-only

    # Full comparison with default emotions:
    python scripts/06_aime_eval.py --emotions calm anxious desperate hopeful content

    # Quick test — 10 problems, 2 alphas:
    python scripts/06_aime_eval.py --n-problems 10 --alpha-values 1.0 -1.0 \\
        --emotions calm anxious

    # Use historical AIME (more statistical power):
    python scripts/06_aime_eval.py --dataset di-zhang-fdu/AIME_1983_2024 \\
        --n-problems 100 --emotions calm anxious desperate
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.emotion_vectors import load_emotion_vectors
from src.models.hf_backend import HuggingFaceBackend


# ── Prompting ──────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a mathematical expert specializing in competition mathematics. "
    "Solve problems carefully, showing all steps of your reasoning."
)

_USER_TEMPLATE = """\
Solve the following AIME problem. Work through it step by step, then write your \
final answer as a single integer from 0 to 999 on its own line in the format:

**Final Answer: N**

Problem:
{problem}"""


def build_prompt(tokenizer, problem: str, enable_thinking: bool = True) -> str:
    """Apply the model's chat template to produce the correct input string.

    enable_thinking=True activates Qwen3's <think>...</think> reasoning blocks,
    which substantially improves accuracy on hard math problems.
    """
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": _USER_TEMPLATE.format(problem=problem)},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # Older tokenizers that don't support enable_thinking
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


# ── Answer extraction ──────────────────────────────────────────────────────────

_BOXED    = re.compile(r'\\boxed\{(\d+)\}')
_FINAL    = re.compile(r'\*{0,2}[Ff]inal\s+[Aa]nswer\s*[:：]\s*\*{0,2}\s*(\d+)')
_ANS_IS   = re.compile(r'(?:answer|result|solution)\s+is\s+\**\s*(\d+)', re.I)
_EQ_END   = re.compile(r'=\s*(\d+)\s*\.?\s*$', re.MULTILINE)


def extract_aime_answer(response: str) -> int | None:
    """
    Parse an integer answer (0-999) from the model response.

    Tries patterns in order of specificity:
      \\boxed{N}  →  Final Answer: N  →  "answer is N"  →  last "= N"
    Falls back to the last 1-3 digit number in the response.
    """
    for pat in (_BOXED, _FINAL, _ANS_IS, _EQ_END):
        m = pat.search(response)
        if m:
            val = int(m.group(1))
            if 0 <= val <= 999:
                return val

    # Last resort: rightmost 1–3-digit number
    for num in reversed(re.findall(r'\b(\d{1,3})\b', response)):
        val = int(num)
        if 0 <= val <= 999:
            return val

    return None


# ── Dataset ────────────────────────────────────────────────────────────────────

def _normalise(raw: dict) -> dict:
    """Map varying field names across AIME datasets to a unified schema."""
    problem = (
        raw.get("Problem") or raw.get("problem") or
        raw.get("question") or raw.get("Question") or ""
    ).strip()
    raw_ans = raw.get("Answer") or raw.get("answer") or ""
    try:
        answer = int(str(raw_ans).strip())
    except (ValueError, TypeError):
        answer = None
    return {
        "problem": problem,
        "answer":  answer,
        "year":    str(raw.get("year") or raw.get("Year") or ""),
    }


def load_or_download_aime(
    data_path: str,
    dataset_name: str = "Maxwell-Jia/AIME_2024",
    n_problems: int | None = None,
) -> list[dict]:
    path = Path(data_path)

    if path.exists():
        print(f"Loading AIME from {path}")
        records = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    else:
        print(f"Downloading: {dataset_name}  (saving to {path})")
        try:
            from datasets import load_dataset
        except ImportError:
            print("ERROR: install the 'datasets' package:  uv pip install datasets")
            sys.exit(1)

        ds = load_dataset(dataset_name, trust_remote_code=True)
        split = ds.get("train") or ds.get("test") or next(iter(ds.values()))
        records = [_normalise(row) for row in split]
        records = [r for r in records if r["problem"] and r["answer"] is not None]

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved {len(records)} problems → {path}")

    if n_problems:
        records = records[:n_problems]
    print(f"Using {len(records)} AIME problems")
    return records


# ── Inference ──────────────────────────────────────────────────────────────────

def run_condition(
    backend: HuggingFaceBackend,
    problems: list[dict],
    prompts: list[str],
    *,
    emotion_vec=None,        # torch.Tensor [hidden_size] for the chosen layer, or None
    alpha: float = 0.0,
    layer: int = 0,
    max_new_tokens: int = 1024,
    condition_name: str = "baseline",
    output_dir: Path,
    resume: bool = True,
) -> list[dict]:
    out_file = output_dir / f"{condition_name}.jsonl"

    # Resume: load already-finished problems
    done: dict[int, dict] = {}
    if resume and out_file.exists():
        with open(out_file, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done[r["problem_idx"]] = r
        if len(done) == len(problems):
            print(f"  [{condition_name}] already complete — loading cache.")
            return [done[i] for i in sorted(done)]

    results = [done[i] for i in sorted(done)]
    with open(out_file, "a", encoding="utf-8") as fout:
        for i, (prob, prompt) in enumerate(zip(problems, prompts)):
            if i in done:
                continue

            if emotion_vec is not None:
                response = backend.generate_with_steering(
                    prompt,
                    emotion_vector=emotion_vec,
                    layers=[layer],
                    alpha=alpha,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            else:
                response = backend.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            predicted = extract_aime_answer(response)
            correct   = predicted == prob["answer"]

            record = {
                "problem_idx": i,
                "year":        prob.get("year", ""),
                "problem":     prob["problem"],
                "gold_answer": prob["answer"],
                "predicted":   predicted,
                "correct":     correct,
                "response":    response,
                "condition":   condition_name,
                "alpha":       alpha,
                "layer":       layer,
                "timestamp":   datetime.now(timezone.utc).isoformat(),
            }
            results.append(record)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            mark = "✓" if correct else "✗"
            pred_str = str(predicted) if predicted is not None else "?"
            print(f"    [{i+1:3d}/{len(problems)}] {mark}  gold={prob['answer']:3d}  pred={pred_str:>3s}")

    return results


# ── Reporting ──────────────────────────────────────────────────────────────────

def _accuracy(results: list[dict]) -> float:
    return sum(r["correct"] for r in results) / len(results) if results else 0.0


def print_table(all_conditions: dict[str, list[dict]]) -> None:
    print("\n" + "=" * 68)
    print(f"  {'Condition':<38} {'Acc':>6}  {'Correct':>7}  {'Total':>5}")
    print("  " + "-" * 64)
    for name in sorted(all_conditions, key=lambda k: (k != "baseline", k)):
        res = all_conditions[name]
        n   = len(res)
        ok  = sum(r["correct"] for r in res)
        print(f"  {name:<38} {ok/n:>6.1%}  {ok:>7d}  {n:>5d}")
    print("=" * 68 + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="AIME evaluation with emotion steering")
    p.add_argument("--config",         default="config/config.yaml")
    p.add_argument("--model",          help="HF model name (overrides config)")
    p.add_argument("--device",         help="Device: cuda / mps / cpu")
    p.add_argument("--dtype",          choices=["float16", "bfloat16", "float32"])
    p.add_argument("--vectors-dir",    default="results/emotion_vectors",
                   help="Directory with .pt emotion vector files")
    p.add_argument("--output-dir",     default="results/aime")
    p.add_argument("--dataset",        default="Maxwell-Jia/AIME_2024",
                   help="HuggingFace dataset ID (used if --data-path not cached)")
    p.add_argument("--data-path",      default="data/aime/aime.jsonl",
                   help="Local JSONL cache of AIME problems")
    p.add_argument("--n-problems",     type=int, default=None,
                   help="Limit to first N problems (default: all)")
    p.add_argument("--emotions",       nargs="*",
                   default=["calm", "anxious", "desperate", "hopeful", "content"],
                   help="Emotion vectors to test")
    p.add_argument("--alpha-values",   nargs="*", type=float,
                   default=[-2.0, -1.0, 1.0, 2.0],
                   help="Steering strengths to sweep")
    p.add_argument("--layer",          type=int, default=None,
                   help="Steering layer index (default: 2/3 through network)")
    p.add_argument("--max-new-tokens", type=int, default=8192,
                   help="Max tokens to generate (higher for thinking mode)")
    p.add_argument("--no-thinking",    action="store_true",
                   help="Disable Qwen3 thinking mode (faster but lower accuracy)")
    p.add_argument("--baseline-only",  action="store_true",
                   help="Skip steering, run baseline only")
    p.add_argument("--no-resume",      action="store_true",
                   help="Ignore cached results, recompute everything")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load AIME ─────────────────────────────────────────────────────────────
    problems = load_or_download_aime(
        args.data_path,
        dataset_name=args.dataset,
        n_problems=args.n_problems,
    )

    # ── Load model ────────────────────────────────────────────────────────────
    model_name = args.model or cfg["model"]["hf_model_name"]
    device     = args.device or cfg["model"]["device"]
    dtype      = args.dtype  or cfg["model"]["dtype"]

    print(f"\nLoading {model_name} on {device} ({dtype}) ...")
    backend = HuggingFaceBackend(
        model_name=model_name,
        device=device,
        dtype=dtype,
        load_in_4bit=cfg["model"]["load_in_4bit"],
        load_in_8bit=cfg["model"]["load_in_8bit"],
    )

    # ── Format prompts with chat template ─────────────────────────────────────
    enable_thinking = not args.no_thinking
    if enable_thinking:
        print("Thinking mode: ON  (Qwen3 <think>...</think> reasoning enabled)")
    else:
        print("Thinking mode: OFF (faster, lower accuracy on hard problems)")
    prompts = [build_prompt(backend.tokenizer, p["problem"], enable_thinking) for p in problems]

    # ── Choose steering layer ─────────────────────────────────────────────────
    layer = args.layer if args.layer is not None else backend.num_layers * 2 // 3
    print(f"Steering layer: {layer}  (model has {backend.num_layers} layers total)")

    # ── Load emotion vectors ──────────────────────────────────────────────────
    all_vectors: dict[str, dict] = {}
    if not args.baseline_only:
        try:
            all_vectors = load_emotion_vectors(args.vectors_dir)
        except FileNotFoundError:
            print(f"WARNING: No vectors found in {args.vectors_dir}/ — running baseline only.")
            args.baseline_only = True

    # Validate requested emotions and layer availability
    available: dict[str, object] = {}   # emotion -> Tensor at chosen layer
    if not args.baseline_only:
        for emotion in args.emotions:
            if emotion not in all_vectors:
                print(f"  WARNING: '{emotion}' not in vectors directory — skipping.")
                continue
            vec_by_layer = all_vectors[emotion]
            if layer not in vec_by_layer:
                # Find nearest available layer
                nearest = min(vec_by_layer.keys(), key=lambda l: abs(l - layer))
                print(f"  WARNING: layer {layer} missing for '{emotion}', using layer {nearest}.")
                available[emotion] = vec_by_layer[nearest]
            else:
                available[emotion] = vec_by_layer[layer]
        if not available:
            print("No valid emotion vectors found — running baseline only.")
            args.baseline_only = True

    resume = not args.no_resume
    all_conditions: dict[str, list[dict]] = {}

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("\n--- Baseline (no steering) ---")
    all_conditions["baseline"] = run_condition(
        backend, problems, prompts,
        condition_name="baseline",
        max_new_tokens=args.max_new_tokens,
        output_dir=output_dir,
        resume=resume,
    )

    # ── Steered conditions ────────────────────────────────────────────────────
    if not args.baseline_only:
        for emotion, vec in available.items():
            for alpha in args.alpha_values:
                name = f"{emotion}_a{alpha:+.1f}_l{layer}"
                print(f"\n--- {name} ---")
                all_conditions[name] = run_condition(
                    backend, problems, prompts,
                    emotion_vec=vec,
                    alpha=alpha,
                    layer=layer,
                    condition_name=name,
                    max_new_tokens=args.max_new_tokens,
                    output_dir=output_dir,
                    resume=resume,
                )

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        name: {
            "accuracy": _accuracy(res),
            "correct":  sum(r["correct"] for r in res),
            "total":    len(res),
        }
        for name, res in all_conditions.items()
    }
    summary["_meta"] = {
        "model":        model_name,
        "steering_layer": layer,
        "alpha_values": args.alpha_values,
        "emotions":     list(available.keys()),
        "n_problems":   len(problems),
        "dataset":      args.dataset,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print_table(all_conditions)
    print(f"Results saved to: {output_dir}/")
    print(f"  • Per-condition JSONL files: one row per problem")
    print(f"  • summary.json: accuracy for every condition")


if __name__ == "__main__":
    main()
