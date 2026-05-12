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
    """

    def __init__(
        self,
        backend: HuggingFaceBackend,
        n_stories: int = 10,
        n_neutral: int = 30,
        aggregation: str = "mean",
        target_layers: Optional[list[int]] = None,
        story_prompt_template: str = STORY_PROMPT,
        neutral_prompt_template: str = NEUTRAL_PROMPT,
        stories_dir: Optional[str] = None,
        max_concurrent: int = 32,
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
        """
        self.backend = backend
        self.n_stories = n_stories
        self.n_neutral = n_neutral
        self.aggregation = aggregation
        self.story_prompt = story_prompt_template
        self.neutral_prompt = neutral_prompt_template
        self.max_concurrent = max_concurrent
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
            max_new_tokens=300,
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
