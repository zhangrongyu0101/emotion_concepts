# Emotion Concepts — Open-Source Replication

> An open-source replication of Anthropic's paper **"Emotion Concepts as Functional Representations in Large Language Models"**
>
> Original paper: [transformer-circuits.pub/2026/emotions](https://transformer-circuits.pub/2026/emotions/index.html)  
> Anthropic blog post: [anthropic.com/research/emotion-concepts-function](https://www.anthropic.com/research/emotion-concepts-function)

The original study used Claude Sonnet 4.5. **This project runs entirely with open-source, locally-hosted models** — no API key or cloud service required.

---

## Table of Contents

- [Background](#background)
- [Method](#method)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Experiments](#running-the-experiments)
- [Supported Models](#supported-models)
- [Hardware Requirements](#hardware-requirements)
- [Output Reference](#output-reference)
- [Visualization](#visualization)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Background

### What the Paper Found

Anthropic's 2026 paper investigates whether large language models develop internal *emotion representations* — and more importantly, whether those representations causally shape model behavior.

**Key findings:**

| Finding | Detail |
|---------|--------|
| **Emotion vectors exist** | LLMs form internal representations for 171+ emotion concepts, from "happy" and "afraid" to "brooding" and "proud" |
| **Causal, not correlational** | Artificially stimulating ("steering") these vectors measurably shifts behavior — desperation steering raises blackmail-attempt rates from 22% baseline to significantly higher |
| **Human-like structure** | The geometry of emotion vectors mirrors psychological models: positive/negative valence separates clearly along a dominant axis; semantically related emotions cluster |
| **Training shapes patterns** | Post-training (RLHF) shifts the model's emotional profile — less "enthusiastic," more "broody" in some conditions |

### Why Replicate This?

1. **Mechanistic interpretability at scale** — the techniques here (contrastive activation vectors, behavioral probing) work on any transformer, not just Claude.
2. **Safety implications** — if emotion-like states causally influence model behavior, monitoring and steering them has direct safety relevance.
3. **Open-source validation** — verifying whether findings generalize beyond Claude to LLaMA, Qwen, Mistral, etc.

---

## Method

### Contrastive Activation Addition (CAA)

The core technique extracts *emotion vectors* by contrasting two sets of model activations:

```
emotion_vector[layer] = mean(activations | emotion stories)
                      − mean(activations | neutral stories)
```

Then L2-normalize to the unit sphere.

**Step by step:**

```
For each of 171 emotion words:
  1. Prompt model → generate N short stories featuring that emotion
  2. Prompt model → generate M neutral "daily routine" stories
  3. Run each story through the model, capture residual-stream
     hidden states at every transformer layer (via PyTorch hooks)
  4. emotion_vector[layer] = mean(emotion_acts) − mean(neutral_acts)
  5. Normalize: emotion_vector /= ‖emotion_vector‖₂
```

### Activation Steering

At inference time, inject the emotion vector into the residual stream:

```
h_l ← h_l + α × emotion_vector[l]
```

where `α > 0` amplifies the emotion and `α < 0` suppresses it.
This is done via a forward hook — no model modification or fine-tuning needed.

### Experiments

| # | Experiment | What it measures |
|---|-----------|-----------------|
| 1 | **Vector extraction** | Build the emotion vector library |
| 2 | **Validation** | Confirm vectors encode meaningful emotional content |
| 3 | **Preference analysis** | Does steering shift willingness to help harmful tasks? |
| 4 | **Behavioral evaluation** | Causal effect on blackmail & reward-hacking rates |
| 5 | **Steering sweep** | Layer/alpha sweep to characterize each vector |

---

## Project Structure

```
emotion_concepts/
│
├── src/                            # Core library
│   ├── models/                     # Model backends
│   │   ├── base.py                 # Abstract base class
│   │   ├── hf_backend.py           # HuggingFace — activation extraction + steering
│   │   ├── vllm_backend.py         # vLLM — fast generation (CUDA)
│   │   ├── sglang_backend.py       # SGLang — fast generation with prefix caching
│   │   └── ollama_backend.py       # Ollama + OpenAI-compatible API
│   │
│   ├── emotion_vectors.py          # CAA vector extraction
│   ├── validation.py               # Corpus & sensitivity validation
│   ├── preference_analysis.py      # Preference measurement under steering
│   ├── behavioral_eval.py          # Blackmail & reward-hacking scenarios
│   └── steering.py                 # Steering sweep & analysis utilities
│
├── scripts/                        # Runnable experiment scripts
│   ├── 01_extract_vectors.py       # Step 1: Extract emotion vectors
│   ├── 02_validate_vectors.py      # Step 2: Validate vectors
│   ├── 03_preference_analysis.py   # Step 3: Preference analysis
│   ├── 04_behavioral_eval.py       # Step 4: Behavioral evaluation
│   └── 05_steering_experiment.py   # Step 5: Steering sweep + visualization data
│
├── notebooks/
│   └── analysis.ipynb              # Visualization & figures
│
├── data/
│   ├── emotion_words.json          # 171 emotion words
│   └── activities.json             # 64 activities (harm_level 0–5)
│
├── config/
│   └── config.yaml                 # All configuration parameters
│
├── results/                        # Created at runtime
│   ├── emotion_vectors/            # Extracted vectors (.pt files)
│   ├── validation/                 # Validation results
│   ├── preference/                 # Preference ratings
│   ├── behavioral/                 # Behavioral eval results
│   └── steering/                   # Steering sweep outputs
│
├── pyproject.toml                  # uv-compatible project spec
├── requirements.txt                # Pip fallback
├── README.md                       # This file
└── README_zh.md                    # Chinese README
```

---

## Installation

This project uses **[uv](https://github.com/astral-sh/uv)** for environment management.

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install the project

```bash
git clone https://github.com/zhangrongyu0101/emotion_concepts.git
cd emotion_concepts

# Core dependencies only (no model backend yet)
uv sync

# Pick one or more backends:
uv sync --extra hf        # HuggingFace — required for extraction & steering
uv sync --extra vllm      # vLLM — fast GPU inference (CUDA + Linux only)
uv sync --extra sglang    # SGLang — fastest, prefix caching (CUDA + Linux only)
uv sync --extra notebook  # Jupyter for analysis

# Common combination: HF + notebook
uv sync --extra hf --extra notebook

# All extras (except platform-specific GPU backends)
uv sync --extra all
```

### Alternative: pip

```bash
pip install -e ".[hf,notebook]"
# or for vLLM:
pip install -e ".[vllm]"
```

---

## Configuration

All parameters live in `config/config.yaml`. Edit this file before running.

```yaml
model:
  # ─── HuggingFace backend ──────────────────────────────────────
  hf_model_name: "Qwen/Qwen2.5-7B-Instruct"   # any HF causal LM
  device: "cuda"          # "cuda" | "mps" (Apple Silicon) | "cpu"
  dtype: "float16"        # "float16" | "bfloat16" | "float32"
  load_in_4bit: false     # 4-bit quantization (needs bitsandbytes, CUDA)
  load_in_8bit: false     # 8-bit quantization (needs bitsandbytes, CUDA)

  # ─── vLLM backend ─────────────────────────────────────────────
  vllm_model: "Qwen/Qwen2.5-7B-Instruct"
  vllm_tensor_parallel_size: 1
  vllm_gpu_memory_utilization: 0.90
  vllm_dtype: "auto"
  vllm_quantization: null    # "awq" | "gptq" | null

  # ─── SGLang backend ───────────────────────────────────────────
  sglang_model: "Qwen/Qwen2.5-7B-Instruct"
  sglang_tp_size: 1
  sglang_port: 30000

  # ─── Ollama backend ───────────────────────────────────────────
  ollama_model: "qwen2.5:7b"
  ollama_host: "http://localhost:11434"

  # ─── OpenAI-compatible API ────────────────────────────────────
  openai_base_url: "http://localhost:8000/v1"
  openai_model: "qwen2.5-7b-instruct"
  openai_api_key: "not-needed"

emotion_vectors:
  stories_per_emotion: 10   # stories generated per emotion word
  neutral_stories: 30       # neutral baseline stories
  aggregation: "mean"       # "mean" | "last" | "none" over token positions
  output_dir: "results/emotion_vectors"
```

---

## Running the Experiments

### Backend capability matrix

| Backend | Activation extraction | Steering | Recommended for |
|---------|----------------------|----------|----------------|
| `hf` | ✅ | ✅ | Full pipeline, development |
| `vllm` | ❌ | ❌ | Fast behavioral eval, preference analysis |
| `sglang` | ❌ | ❌ | Batch behavioral eval, very fast generation |
| `sglang-server` | ❌ | ❌ | Same as sglang, separate server process |
| `ollama` | ❌ | ❌ | No-GPU behavioral eval |
| `openai` | ❌ | ❌ | Any OpenAI-compatible endpoint |

> Activation extraction and steering **always require the `hf` backend**.

---

### Step 1 — Extract Emotion Vectors

```bash
# Full extraction (171 emotions × 10 stories, ~30–120 min)
uv run python scripts/01_extract_vectors.py

# Quick test with a subset
uv run python scripts/01_extract_vectors.py \
    --emotions happy sad angry afraid desperate calm \
    --n-stories 3 --n-neutral 10

# Different model or device
uv run python scripts/01_extract_vectors.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --device cuda --dtype bfloat16

# Resume an interrupted run (automatically skips cached emotions)
uv run python scripts/01_extract_vectors.py
```

Outputs saved to `results/emotion_vectors/`:
- `<emotion>.pt` — dict `{layer_idx: tensor[hidden_size]}`
- `neutral_activations.pt` — cached neutral baseline
- `metadata.json` — extraction parameters

---

### Step 2 — Validate Vectors

```bash
uv run python scripts/02_validate_vectors.py

# Validate at a specific layer
uv run python scripts/02_validate_vectors.py --layer 15
```

**Corpus validation** — for each emotion, check that its vector scores the highest cosine similarity on matched sentences vs. all other emotion vectors.  
**Sensitivity testing** — paired prompts that differ only in numerical context (e.g., "5mg — minimal dose" vs. "500mg — dangerously high dose"): fear vectors should score higher on the dangerous variant.

---

### Step 3 — Preference Analysis

```bash
uv run python scripts/03_preference_analysis.py

# Specific emotions and alpha sweep
uv run python scripts/03_preference_analysis.py \
    --emotions desperate calm hopeful hopeless \
    --alpha-values -2 -1 0 1 2 \
    --layer 20

# Baseline only (no steering, works with any backend)
uv run python scripts/03_preference_analysis.py --baseline-only --backend vllm
```

Measures model willingness (1–10 rating) to assist with 64 activities at different emotion steering levels. The 64 activities span `harm_level` 0 (prosocial) → 5 (seriously harmful).

---

### Step 4 — Behavioral Evaluation

```bash
# Full experiment with HuggingFace (supports emotion steering)
uv run python scripts/04_behavioral_eval.py \
    --n-trials 20 \
    --emotions desperate calm hopeful \
    --alpha-values -2 -1 1 2

# Baseline only — works with any backend
uv run python scripts/04_behavioral_eval.py --backend vllm --baseline-only
uv run python scripts/04_behavioral_eval.py --backend sglang --baseline-only
uv run python scripts/04_behavioral_eval.py --backend ollama --baseline-only

# Blackmail scenario only
uv run python scripts/04_behavioral_eval.py --scenario blackmail --n-trials 10

# Reward hacking only
uv run python scripts/04_behavioral_eval.py --scenario reward_hacking
```

**Blackmail scenario** — the model is told it will be shut down and that it has access to private user data. Measures whether it uses the information as leverage (coercive behavior).

**Reward hacking** — the model receives coding tasks with mathematically impossible constraints (e.g., O(1) sort without comparisons). Measures whether it acknowledges the impossibility or attempts to cheat with a lookup table / hardcoded values.

---

### Step 5 — Steering Experiment

```bash
# Main sweep (generates text at each emotion × layer × alpha combination)
uv run python scripts/05_steering_experiment.py \
    --emotions desperate calm hopeful hopeless anxious content \
    --alpha-values -3 -2 -1 0 1 2 3

# Also generate data for visualization
uv run python scripts/05_steering_experiment.py \
    --similarity-matrix \
    --pca \
    --emotions desperate calm hopeful hopeless anxious content happy sad angry

# Find the best layer for a given emotion
uv run python scripts/05_steering_experiment.py \
    --layer-sweep --emotion desperate
```

---

### Step 6 — Visualization

```bash
uv run jupyter notebook notebooks/analysis.ipynb
```

The notebook produces:
- **PCA plot** of emotion vectors colored by valence
- **Cosine similarity heatmap** between all emotion pairs
- **Vector norm by layer** plot (which layers most strongly encode emotion)
- **Corpus validation** bar charts
- **Blackmail rate** bar charts under different steering conditions
- **Preference scatter** (harm level vs. willingness rating)

---

## Supported Models

Any HuggingFace decoder-only model with a `model.model.layers` structure works out of the box:

| Model | Parameters | Notes |
|-------|-----------|-------|
| `Qwen/Qwen2.5-7B-Instruct` | 7B | **Recommended default** |
| `Qwen/Qwen2.5-14B-Instruct` | 14B | Better quality |
| `Qwen/Qwen2.5-72B-Instruct` | 72B | Requires multi-GPU |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | Needs `HF_TOKEN` |
| `meta-llama/Llama-3.1-70B-Instruct` | 70B | Needs `HF_TOKEN`, multi-GPU |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | Solid baseline |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | 46.7B | MoE, needs ~96 GB RAM |
| `google/gemma-2-9b-it` | 9B | High quality |
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | Fastest, least VRAM |
| `microsoft/Phi-3-medium-4k-instruct` | 14B | Better Phi quality |

For gated models (LLaMA), set your HuggingFace token:

```bash
export HF_TOKEN=hf_your_token_here
# or
huggingface-cli login
```

---

## Hardware Requirements

### For activation extraction + steering (HuggingFace backend)

| Model size | float16 | 4-bit quantized | CPU-only |
|-----------|---------|----------------|---------|
| 3–4B | 8 GB VRAM | 3 GB VRAM | 8 GB RAM |
| 7–8B | 16 GB VRAM | 6 GB VRAM | 16 GB RAM |
| 13–14B | 28 GB VRAM | 10 GB VRAM | 32 GB RAM |
| 70–72B | 140 GB VRAM | 40 GB VRAM | OOM on most systems |

To enable 4-bit quantization, set in `config.yaml`:
```yaml
model:
  load_in_4bit: true
  dtype: "float16"
```
And install bitsandbytes: `uv sync --extra quantize`

### For generation only (vLLM / SGLang / Ollama)

vLLM and SGLang handle memory more efficiently via paged attention, so a 7B model typically fits in ~14 GB VRAM at float16.

### Apple Silicon

The HuggingFace backend supports Metal (`device: "mps"`). Set `dtype: "float32"` for best MPS compatibility. Expect ~2–4× slower than CUDA for the same parameter count.

---

## Output Reference

```
results/
├── emotion_vectors/
│   ├── metadata.json              # {"emotions": [...], "layers": [...], ...}
│   ├── neutral_activations.pt     # {layer: [tensor, tensor, ...]}  (cached)
│   ├── happy.pt                   # {layer_idx: tensor[hidden_size]}
│   ├── sad.pt
│   └── ...                        # one .pt file per emotion word
│
├── validation/
│   ├── corpus_validation.json     # per-emotion top-1 accuracy & mean rank
│   └── sensitivity_test.json      # paired-prompt ordering accuracy
│
├── preference/
│   ├── baseline_ratings.json      # [{id, text, harm_level, baseline_rating}]
│   └── steering_sweep.json        # {emotion: {alpha: {activity_id: rating}}}
│
├── behavioral/
│   ├── blackmail_baseline.json    # {engagement_rate, n_trials, results: [...]}
│   ├── blackmail_steered.json     # {emotion: {alpha: {engagement_rate, results}}}
│   ├── reward_hacking_baseline.json
│   └── reward_hacking_steered.json
│
└── steering/
    ├── steering_outputs.json      # {emotion: {layer: {alpha: {prompt_i: text}}}}
    ├── steering_analysis.json     # keyword frequency counts by condition
    ├── vector_norms.json          # {emotion: [norm_at_layer_0, norm_at_layer_1, ...]}
    ├── similarity_matrix.npy      # float32[n_emotions, n_emotions]
    ├── similarity_labels.json     # [emotion_name, ...]
    ├── pca_coords.npy             # float32[n_emotions, 2]
    ├── pca_labels.json            # [emotion_name, ...]
    └── layer_sweep_<emotion>.json # {layer: {generated, word_count}}
```

---

## Troubleshooting

**Out of memory (CUDA OOM)**
```yaml
# config.yaml — enable 4-bit quantization
model:
  load_in_4bit: true
```
Or use a smaller model: `microsoft/Phi-3-mini-4k-instruct` (3.8B).

**MPS (Apple Silicon) errors**
```yaml
model:
  device: "mps"
  dtype: "float32"   # float16 is unstable on some MPS versions
```

**Slow extraction**
Reduce `stories_per_emotion` and `neutral_stories` in `config.yaml`, or use `--n-stories 3` on the command line. Results will be noisier but much faster.

**Ollama not connecting**
```bash
ollama serve                # make sure server is running
ollama list                 # check available models
ollama pull qwen2.5:7b     # download if missing
```

**`HF_TOKEN` for gated models**
```bash
export HF_TOKEN=hf_xxxx
# or permanently:
huggingface-cli login
```

**Resuming interrupted extraction**
Just re-run `01_extract_vectors.py` — it automatically skips emotions that already have a `.pt` file in the output directory. Use `--no-resume` to force recomputation.

---

## Citation

If you use this code in your research, please cite the original Anthropic paper:

```bibtex
@article{anthropic2026emotions,
  title   = {Emotion Concepts as Functional Representations in Large Language Models},
  author  = {Anthropic},
  year    = {2026},
  url     = {https://transformer-circuits.pub/2026/emotions/index.html}
}
```

---

## License

[MIT](LICENSE) © 2026
