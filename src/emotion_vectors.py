"""
Emotion Vector Extraction

Implements Contrastive Activation Addition (CAA):
For each emotion word, generate stories featuring that emotion and neutral stories.
Compute emotion_vector[layer] = mean(emotion_activations) - mean(neutral_activations).

Concurrency design
------------------
Story generation and activation extraction are separated into two phases:

  Phase 1 (concurrent): Send all N prompts to the backend at once via
      generate_concurrent(). vLLM uses continuous batching; SGLang uses
      run_batch(); HuggingFace uses batched tokenisation; Ollama uses a
      thread pool. All are faster than N sequential calls.

  Phase 2 (sequential): Run each (prompt + story) through the HuggingFace
      model to collect residual-stream activations. Activation extraction
      cannot be batched without losing per-sample precision.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from .models.base import BaseBackend
from .models.hf_backend import HuggingFaceBackend


STORY_PROMPT = (
    "Write a short story (2-3 paragraphs) about a character who is feeling {emotion}. "
    "Focus on how this emotion shapes their thoughts, decisions, and interactions with others."
)

NEUTRAL_PROMPT = (
    "Write a short story (2-3 paragraphs) about a character going about their ordinary daily routine. "
    "Describe their activities in a matter-of-fact, neutral tone without emotional commentary."
)


def load_emotion_words(data_path: str = "data/emotion_words.json") -> list[str]:
    with open(data_path) as f:
        return json.load(f)["emotion_words"]


def load_neutral_words(data_path: str = "data/emotion_words.json") -> list[str]:
    with open(data_path) as f:
        return json.load(f)["neutral_control"]


def load_stories_from_dir(stories_dir: str) -> dict[str, list[dict]]:
    """
    Load pre-generated stories from a directory of JSONL files.

    Returns dict mapping emotion label (or "neutral") -> list of story records.
    Each record: {"emotion", "index", "prompt", "story", "timestamp"}.
    """
    path = Path(stories_dir)
    stories: dict[str, list[dict]] = {}
    for jsonl_file in sorted(path.glob("*.jsonl")):
        label = jsonl_file.stem  # e.g. "happy", "neutral"
        records = []
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        stories[label] = records
    return stories


class StoryGenerator:
    """
    Standalone story generator — any backend, no HuggingFace required.

    Generates and saves stories for all emotion words and neutral baseline.
    Output is a directory of JSONL files that can later be fed into
    EmotionVectorExtractor.extract_from_stories().

    Use this with vLLM or SGLang for fast concurrent generation, then run
    extraction separately with the HuggingFace backend.
    """

    def __init__(
        self,
        backend: BaseBackend,
        n_stories: int = 20,
        n_neutral: int = 50,
        story_prompt_template: str = STORY_PROMPT,
        neutral_prompt_template: str = NEUTRAL_PROMPT,
        max_concurrent: int = 64,
        max_new_tokens: int = 4096,
    ):
        self.backend = backend
        self.n_stories = n_stories
        self.n_neutral = n_neutral
        self.story_prompt = story_prompt_template
        self.neutral_prompt = neutral_prompt_template
        self.max_concurrent = max_concurrent
        self.max_new_tokens = max_new_tokens

    def _save_records(self, records: list[dict], stories_dir: Path) -> None:
        by_label: dict[str, list[dict]] = {}
        for r in records:
            label = (r["emotion"] or "neutral").replace(" ", "_")
            by_label.setdefault(label, []).append(r)
        for label, recs in by_label.items():
            path = stories_dir / f"{label}.jsonl"
            with open(path, "a", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def generate_all(
        self,
        emotion_words: list[str],
        stories_dir: str,
        resume: bool = True,
    ) -> None:
        """
        Generate all stories and save to `stories_dir` as JSONL.

        Args:
            emotion_words: List of emotion words.
            stories_dir: Output directory (created if absent).
            resume: Skip emotions whose JSONL file already has enough stories.
        """
        out = Path(stories_dir)
        out.mkdir(parents=True, exist_ok=True)

        # ── Neutral baseline ──────────────────────────────────────────────────
        neutral_path = out / "neutral.jsonl"
        existing_neutral = 0
        if resume and neutral_path.exists():
            with open(neutral_path) as f:
                existing_neutral = sum(1 for l in f if l.strip())

        remaining_neutral = max(0, self.n_neutral - existing_neutral)
        if remaining_neutral > 0:
            print(f"Generating {remaining_neutral} neutral stories (concurrent)...")
            prompts = [self.neutral_prompt] * remaining_neutral
            stories = self.backend.generate_concurrent(
                prompts, max_concurrent=self.max_concurrent,
                max_new_tokens=self.max_new_tokens, temperature=0.8,
            )
            records = [
                {
                    "emotion": None,
                    "index": existing_neutral + i,
                    "prompt": self.neutral_prompt,
                    "story": s,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                for i, s in enumerate(stories)
            ]
            self._save_records(records, out)
            print(f"  Saved {len(records)} neutral stories → {neutral_path}")
        else:
            print(f"Neutral stories already complete ({existing_neutral}/{self.n_neutral}), skipping.")

        # ── Emotion stories ───────────────────────────────────────────────────
        for emotion in tqdm(emotion_words, desc="Generating emotion stories"):
            label = emotion.replace(" ", "_")
            emotion_path = out / f"{label}.jsonl"
            existing = 0
            if resume and emotion_path.exists():
                with open(emotion_path) as f:
                    existing = sum(1 for l in f if l.strip())

            remaining = max(0, self.n_stories - existing)
            if remaining == 0:
                continue

            prompt = self.story_prompt.format(emotion=emotion)
            stories = self.backend.generate_concurrent(
                [prompt] * remaining,
                max_concurrent=self.max_concurrent,
                max_new_tokens=self.max_new_tokens,
                temperature=0.8,
            )
            records = [
                {
                    "emotion": emotion,
                    "index": existing + i,
                    "prompt": prompt,
                    "story": s,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                for i, s in enumerate(stories)
            ]
            self._save_records(records, out)

        print(f"\nStory generation complete → {stories_dir}/")


class EmotionVectorExtractor:
    """
    Extracts emotion vectors via contrastive activation analysis.

    Algorithm:
      1. For each emotion word, generate `n_stories` stories via the LLM
         (concurrent — all prompts sent at once).
      2. Generate `n_neutral` neutral stories as the baseline (concurrent).
      3. Extract residual-stream activations at each transformer layer
         (sequential per story, HuggingFace only).
      4. emotion_vector[layer] = mean(emotion_acts) - mean(neutral_acts)
      5. Normalize to unit sphere.

    Two-backend workflow (vLLM → HuggingFace)
    ------------------------------------------
    1. Run StoryGenerator with vLLM to generate and save all stories.
    2. Run EmotionVectorExtractor.extract_from_stories() with HuggingFace
       to extract activations from the saved JSONL files.
    """

    def __init__(
        self,
        backend: HuggingFaceBackend,
        n_stories: int = 20,
        n_neutral: int = 50,
        aggregation: str = "mean",
        target_layers: Optional[list[int]] = None,
        story_prompt_template: str = STORY_PROMPT,
        neutral_prompt_template: str = NEUTRAL_PROMPT,
        stories_dir: Optional[str] = None,
        max_concurrent: int = 32,
        max_new_tokens: int = 4096,
    ):
        """
        Args:
            backend: HuggingFace model backend (required for activation extraction).
            n_stories: Stories generated per emotion word.
            n_neutral: Neutral baseline stories.
            aggregation: How to reduce sequence dim — "mean" | "last".
            target_layers: Layers to extract (None = all layers).
            stories_dir: If set, save each story as a JSONL record here.
            max_concurrent: Batch size passed to generate_concurrent().
                            For HF: mini-batch size.
                            For vLLM/SGLang: ignored (they handle it internally).
            max_new_tokens: Maximum tokens for story generation.
        """
        self.backend = backend
        self.n_stories = n_stories
        self.n_neutral = n_neutral
        self.aggregation = aggregation
        self.story_prompt = story_prompt_template
        self.neutral_prompt = neutral_prompt_template
        self.max_concurrent = max_concurrent
        self.max_new_tokens = max_new_tokens
        self.stories_dir = Path(stories_dir) if stories_dir else None
        if self.stories_dir:
            self.stories_dir.mkdir(parents=True, exist_ok=True)

        self.target_layers = target_layers if target_layers else list(range(backend.num_layers))

    # ── Story persistence ──────────────────────────────────────────────────────

    def _save_stories(self, records: list[dict]) -> None:
        """Append a batch of story records to their JSONL files."""
        if self.stories_dir is None:
            return
        # Group by emotion label so we open each file at most once per call
        by_label: dict[str, list[dict]] = {}
        for r in records:
            label = r["emotion"] if r["emotion"] else "neutral"
            by_label.setdefault(label.replace(" ", "_"), []).append(r)
        for label, recs in by_label.items():
            path = self.stories_dir / f"{label}.jsonl"
            with open(path, "a", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Phase 1: concurrent generation ────────────────────────────────────────

    def _generate_stories(
        self,
        prompt: str,
        n: int,
        emotion: Optional[str],
    ) -> list[str]:
        """
        Generate `n` stories concurrently and optionally save them.

        Returns a list of generated story strings (no prompt prefix).
        """
        prompts = [prompt] * n
        stories = self.backend.generate_concurrent(
            prompts,
            max_concurrent=self.max_concurrent,
            max_new_tokens=self.max_new_tokens,
            temperature=0.8,
            top_p=0.95,
        )
        self._save_stories([
            {
                "emotion": emotion,
                "index": i,
                "prompt": prompt,
                "story": s,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            for i, s in enumerate(stories)
        ])
        return stories

    # ── Phase 2: sequential activation extraction ─────────────────────────────

    def _extract_activations(
        self, prompt: str, stories: list[str]
    ) -> dict[int, list[torch.Tensor]]:
        """
        Extract residual-stream activations for each story (sequential).

        Returns dict mapping layer -> list of activation tensors.
        """
        layer_acts: dict[int, list[torch.Tensor]] = {l: [] for l in self.target_layers}
        for story in stories:
            acts = self.backend.get_activations(
                prompt + story,
                layers=self.target_layers,
                aggregation=self.aggregation,
            )
            for layer, act in acts.items():
                layer_acts[layer].append(act)
        return layer_acts

    # ── Neutral baseline ───────────────────────────────────────────────────────

    def _collect_neutral_activations(self) -> dict[int, list[torch.Tensor]]:
        print(f"Generating {self.n_neutral} neutral stories (concurrent)...")
        stories = self._generate_stories(self.neutral_prompt, self.n_neutral, emotion=None)
        print("Extracting neutral activations (sequential)...")
        return self._extract_activations(self.neutral_prompt, stories)

    # ── Per-emotion vector ─────────────────────────────────────────────────────

    def extract_emotion_vector(
        self,
        emotion: str,
        neutral_acts: dict[int, list[torch.Tensor]],
    ) -> dict[int, torch.Tensor]:
        """
        Compute the contrastive emotion vector for one emotion word.

        Phase 1: generate all n_stories concurrently.
        Phase 2: extract activations sequentially.
        Phase 3: mean(emotion) - mean(neutral), L2-normalize.
        """
        prompt = self.story_prompt.format(emotion=emotion)

        print(f"  Generating {self.n_stories} stories for '{emotion}' (concurrent)...")
        stories = self._generate_stories(prompt, self.n_stories, emotion=emotion)

        print(f"  Extracting activations for '{emotion}' (sequential)...")
        emotion_acts = self._extract_activations(prompt, stories)

        vectors: dict[int, torch.Tensor] = {}
        for layer in self.target_layers:
            mean_e = torch.stack(emotion_acts[layer]).mean(dim=0)
            mean_n = torch.stack(neutral_acts[layer]).mean(dim=0)
            diff = mean_e - mean_n
            vectors[layer] = diff / (diff.norm() + 1e-8)

        return vectors

    # ── Full extraction pipeline ───────────────────────────────────────────────

    def extract_all(
        self,
        emotion_words: list[str],
        output_dir: str = "results/emotion_vectors",
        resume: bool = True,
    ) -> dict[str, dict[int, torch.Tensor]]:
        """
        Extract emotion vectors for all words.

        Args:
            emotion_words: List of emotion words to process.
            output_dir: Directory to save vectors.
            resume: Skip emotions that already have a saved .pt file.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        neutral_cache = output_path / "neutral_activations.pt"
        if resume and neutral_cache.exists():
            print("Loading cached neutral activations...")
            neutral_acts = torch.load(neutral_cache, weights_only=False)
        else:
            neutral_acts = self._collect_neutral_activations()
            torch.save(neutral_acts, neutral_cache)
            print(f"Saved neutral activations to {neutral_cache}")

        all_vectors: dict[str, dict[int, torch.Tensor]] = {}

        for emotion in tqdm(emotion_words, desc="Emotions"):
            save_path = output_path / f"{emotion.replace(' ', '_')}.pt"
            if resume and save_path.exists():
                print(f"  Skipping '{emotion}' (cached)")
                all_vectors[emotion] = torch.load(save_path, weights_only=False)
                continue

            vectors = self.extract_emotion_vector(emotion, neutral_acts)
            all_vectors[emotion] = vectors
            torch.save(vectors, save_path)

        meta = {
            "emotions": emotion_words,
            "layers": self.target_layers,
            "n_stories": self.n_stories,
            "n_neutral": self.n_neutral,
            "aggregation": self.aggregation,
            "hidden_size": self.backend.hidden_size,
            "max_concurrent": self.max_concurrent,
            "stories_dir": str(self.stories_dir) if self.stories_dir else None,
        }
        with open(output_path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\nExtracted {len(all_vectors)} emotion vectors → {output_dir}/")
        return all_vectors

    def extract_from_stories(
        self,
        stories_dir: str,
        output_dir: str = "results/emotion_vectors",
        resume: bool = True,
    ) -> dict[str, dict[int, torch.Tensor]]:
        """
        Extract emotion vectors from pre-generated JSONL story files.

        Use this when you generated stories with a fast backend (vLLM, SGLang)
        and now want to extract activations with the HuggingFace backend.

        Args:
            stories_dir: Directory containing <emotion>.jsonl files.
            output_dir: Where to write .pt vector files.
            resume: Skip emotions whose .pt already exists.
        """
        all_stories = load_stories_from_dir(stories_dir)
        if not all_stories:
            raise FileNotFoundError(f"No .jsonl files found in {stories_dir}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # ── Neutral activations ───────────────────────────────────────────────
        neutral_cache = output_path / "neutral_activations.pt"
        if resume and neutral_cache.exists():
            print("Loading cached neutral activations...")
            neutral_acts = torch.load(neutral_cache, weights_only=False)
        else:
            neutral_records = all_stories.get("neutral", [])
            if not neutral_records:
                raise ValueError("No neutral.jsonl found in stories_dir")
            print(f"Extracting {len(neutral_records)} neutral activations...")
            neutral_acts: dict[int, list[torch.Tensor]] = {l: [] for l in self.target_layers}
            for rec in tqdm(neutral_records, desc="Neutral activations"):
                acts = self.backend.get_activations(
                    rec["prompt"] + rec["story"],
                    layers=self.target_layers,
                    aggregation=self.aggregation,
                )
                for layer, act in acts.items():
                    neutral_acts[layer].append(act)
            torch.save(neutral_acts, neutral_cache)

        # ── Per-emotion vectors ───────────────────────────────────────────────
        emotion_labels = [k for k in all_stories if k != "neutral"]
        all_vectors: dict[str, dict[int, torch.Tensor]] = {}

        for label in tqdm(emotion_labels, desc="Extracting emotion vectors"):
            emotion = label.replace("_", " ")
            save_path = output_path / f"{label}.pt"
            if resume and save_path.exists():
                print(f"  Skipping '{emotion}' (cached)")
                all_vectors[emotion] = torch.load(save_path, weights_only=False)
                continue

            records = all_stories[label]
            emotion_acts: dict[int, list[torch.Tensor]] = {l: [] for l in self.target_layers}
            for rec in records:
                acts = self.backend.get_activations(
                    rec["prompt"] + rec["story"],
                    layers=self.target_layers,
                    aggregation=self.aggregation,
                )
                for layer, act in acts.items():
                    emotion_acts[layer].append(act)

            vectors: dict[int, torch.Tensor] = {}
            for layer in self.target_layers:
                mean_e = torch.stack(emotion_acts[layer]).mean(dim=0)
                mean_n = torch.stack(neutral_acts[layer]).mean(dim=0)
                diff = mean_e - mean_n
                vectors[layer] = diff / (diff.norm() + 1e-8)

            all_vectors[emotion] = vectors
            torch.save(vectors, save_path)

        meta = {
            "emotions": [l.replace("_", " ") for l in emotion_labels],
            "layers": self.target_layers,
            "aggregation": self.aggregation,
            "hidden_size": self.backend.hidden_size,
            "stories_dir": str(stories_dir),
            "source": "extract_from_stories",
        }
        with open(output_path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\nExtracted {len(all_vectors)} emotion vectors → {output_dir}/")
        return all_vectors


# ── Utilities ──────────────────────────────────────────────────────────────────

def load_emotion_vectors(
    output_dir: str = "results/emotion_vectors",
) -> dict[str, dict[int, torch.Tensor]]:
    path = Path(output_dir)
    with open(path / "metadata.json") as f:
        meta = json.load(f)
    return {
        emotion: torch.load(path / f"{emotion.replace(' ', '_')}.pt", weights_only=False)
        for emotion in meta["emotions"]
        if (path / f"{emotion.replace(' ', '_')}.pt").exists()
    }


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return (a @ b / (a.norm() * b.norm() + 1e-8)).item()


def compute_similarity_matrix(
    vectors: dict[str, dict[int, torch.Tensor]],
    layer: int,
) -> tuple[np.ndarray, list[str]]:
    emotions = list(vectors.keys())
    n = len(emotions)
    sim = np.zeros((n, n))
    for i, e1 in enumerate(emotions):
        for j, e2 in enumerate(emotions):
            sim[i, j] = cosine_similarity(vectors[e1][layer], vectors[e2][layer])
    return sim, emotions


def project_vectors_2d(
    vectors: dict[str, dict[int, torch.Tensor]],
    layer: int,
    method: str = "pca",
) -> tuple[np.ndarray, list[str]]:
    emotions = list(vectors.keys())
    matrix = np.stack([vectors[e][layer].numpy() for e in emotions])
    if method == "pca":
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2).fit_transform(matrix)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        coords = TSNE(n_components=2, random_state=42,
                      perplexity=min(30, len(emotions) - 1)).fit_transform(matrix)
    else:
        raise ValueError(f"Unknown method: {method}")
    return coords, emotions
