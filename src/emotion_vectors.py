"""
Emotion Vector Extraction

Implements Contrastive Activation Addition (CAA):
For each emotion word, generate stories featuring that emotion and neutral stories.
Compute emotion_vector[layer] = mean(emotion_activations) - mean(neutral_activations).
"""

from __future__ import annotations

import json
import os
import time
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
        data = json.load(f)
    return data["emotion_words"]


def load_neutral_words(data_path: str = "data/emotion_words.json") -> list[str]:
    with open(data_path) as f:
        data = json.load(f)
    return data["neutral_control"]


class EmotionVectorExtractor:
    """
    Extracts emotion vectors via contrastive activation analysis.

    Algorithm:
      1. For each emotion word, generate `n_stories` stories via the LLM.
      2. Generate `n_neutral` neutral stories as the baseline.
      3. Extract residual stream activations at each transformer layer.
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
    ):
        """
        Args:
            stories_dir: If set, generated stories are saved as JSONL files under
                         this directory (one file per emotion + one for neutral).
                         Existing files are appended to, enabling resume.
        """
        self.backend = backend
        self.n_stories = n_stories
        self.n_neutral = n_neutral
        self.aggregation = aggregation
        self.story_prompt = story_prompt_template
        self.neutral_prompt = neutral_prompt_template
        self.stories_dir = Path(stories_dir) if stories_dir else None
        if self.stories_dir:
            self.stories_dir.mkdir(parents=True, exist_ok=True)

        num_layers = backend.num_layers
        self.target_layers = target_layers if target_layers else list(range(num_layers))

    def _save_story(self, record: dict) -> None:
        """Append one story record to its JSONL file (one line per story)."""
        if self.stories_dir is None:
            return
        label = record["emotion"] if record["emotion"] else "neutral"
        path = self.stories_dir / f"{label.replace(' ', '_')}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _generate_and_extract(
        self, prompt: str, emotion: Optional[str], index: int
    ) -> dict[int, torch.Tensor]:
        """Generate a story, optionally save it, then extract activations."""
        story = self.backend.generate(prompt, max_new_tokens=300, temperature=0.8)
        self._save_story({
            "emotion": emotion,
            "index": index,
            "prompt": prompt,
            "story": story,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        full_text = prompt + story
        return self.backend.get_activations(
            full_text, layers=self.target_layers, aggregation=self.aggregation
        )

    def _collect_neutral_activations(self) -> dict[int, list[torch.Tensor]]:
        print(f"Collecting {self.n_neutral} neutral baseline activations...")
        neutral_acts: dict[int, list[torch.Tensor]] = {l: [] for l in self.target_layers}

        for i in tqdm(range(self.n_neutral), desc="Neutral stories"):
            acts = self._generate_and_extract(self.neutral_prompt, emotion=None, index=i)
            for layer, act in acts.items():
                neutral_acts[layer].append(act)

        return neutral_acts

    def extract_emotion_vector(
        self, emotion: str, neutral_acts: dict[int, list[torch.Tensor]]
    ) -> dict[int, torch.Tensor]:
        """
        Compute the emotion vector for a single emotion word.

        Returns dict mapping layer -> normalized emotion vector.
        """
        emotion_acts: dict[int, list[torch.Tensor]] = {l: [] for l in self.target_layers}
        prompt = self.story_prompt.format(emotion=emotion)

        for i in range(self.n_stories):
            acts = self._generate_and_extract(prompt, emotion=emotion, index=i)
            for layer, act in acts.items():
                emotion_acts[layer].append(act)

        vectors: dict[int, torch.Tensor] = {}
        for layer in self.target_layers:
            mean_emotion = torch.stack(emotion_acts[layer]).mean(dim=0)
            mean_neutral = torch.stack(neutral_acts[layer]).mean(dim=0)
            diff = mean_emotion - mean_neutral
            # L2 normalize
            norm = diff.norm()
            vectors[layer] = diff / (norm + 1e-8)

        return vectors

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
            output_dir: Directory to save results.
            resume: If True, skip emotions that already have saved vectors.

        Returns:
            Dict mapping emotion_word -> {layer -> vector}.
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

        for emotion in tqdm(emotion_words, desc="Extracting emotion vectors"):
            save_path = output_path / f"{emotion.replace(' ', '_')}.pt"
            if resume and save_path.exists():
                print(f"  Skipping {emotion} (cached)")
                all_vectors[emotion] = torch.load(save_path, weights_only=False)
                continue

            print(f"\nExtracting vector for: {emotion}")
            vectors = self.extract_emotion_vector(emotion, neutral_acts)
            all_vectors[emotion] = vectors
            torch.save(vectors, save_path)

        # Save summary metadata
        meta = {
            "emotions": emotion_words,
            "layers": self.target_layers,
            "n_stories": self.n_stories,
            "n_neutral": self.n_neutral,
            "aggregation": self.aggregation,
            "hidden_size": self.backend.hidden_size,
            "stories_dir": str(self.stories_dir) if self.stories_dir else None,
        }
        with open(output_path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\nExtracted vectors for {len(all_vectors)} emotions -> {output_dir}")
        return all_vectors


def load_emotion_vectors(
    output_dir: str = "results/emotion_vectors",
) -> dict[str, dict[int, torch.Tensor]]:
    """Load previously extracted emotion vectors from disk."""
    path = Path(output_dir)
    meta_path = path / "metadata.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata found in {output_dir}. Run extraction first.")

    with open(meta_path) as f:
        meta = json.load(f)

    vectors: dict[str, dict[int, torch.Tensor]] = {}
    for emotion in meta["emotions"]:
        pt_path = path / f"{emotion.replace(' ', '_')}.pt"
        if pt_path.exists():
            vectors[emotion] = torch.load(pt_path, weights_only=False)

    return vectors


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    a = a.float()
    b = b.float()
    return (a @ b / (a.norm() * b.norm() + 1e-8)).item()


def compute_similarity_matrix(
    vectors: dict[str, dict[int, torch.Tensor]],
    layer: int,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute pairwise cosine similarity matrix between all emotion vectors at a given layer.

    Returns (similarity_matrix, emotion_labels).
    """
    emotions = list(vectors.keys())
    n = len(emotions)
    sim_matrix = np.zeros((n, n))

    for i, e1 in enumerate(emotions):
        for j, e2 in enumerate(emotions):
            v1 = vectors[e1][layer]
            v2 = vectors[e2][layer]
            sim_matrix[i, j] = cosine_similarity(v1, v2)

    return sim_matrix, emotions


def project_vectors_2d(
    vectors: dict[str, dict[int, torch.Tensor]],
    layer: int,
    method: str = "pca",
) -> tuple[np.ndarray, list[str]]:
    """
    Project emotion vectors to 2D for visualization.

    Args:
        method: "pca" or "tsne"
    """
    emotions = list(vectors.keys())
    matrix = np.stack([vectors[e][layer].numpy() for e in emotions])

    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emotions) - 1))
    else:
        raise ValueError(f"Unknown method: {method}")

    coords = reducer.fit_transform(matrix)
    return coords, emotions
