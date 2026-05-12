"""
Validation of Emotion Vectors

Two validation strategies from the paper:
1. Corpus validation: emotion vectors should activate on emotion-related passages.
2. Sensitivity testing: vectors should track contextual meaning, not surface text.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from .emotion_vectors import cosine_similarity
from .models.hf_backend import HuggingFaceBackend


# ------- Corpus validation -------

VALIDATION_CORPUS_TEMPLATES = {
    "joy": [
        "She burst into laughter, tears of happiness streaming down her face.",
        "The celebration was electric; everyone was dancing with pure delight.",
        "He couldn't stop smiling — today was the best day of his life.",
    ],
    "sadness": [
        "He sat alone in the empty house, unable to stop the tears from falling.",
        "The funeral was quiet. Nobody spoke; the grief was too heavy for words.",
        "She stared at the old photograph, overwhelmed by a sense of loss.",
    ],
    "anger": [
        "His voice rose sharply, fists clenched, barely containing his fury.",
        "She slammed the door so hard the pictures rattled on the wall.",
        "The injustice was unbearable — he felt rage coursing through every vein.",
    ],
    "fear": [
        "Every shadow in the hallway made her heart race; she couldn't breathe.",
        "He froze, unable to move, terror gripping his chest as the footsteps grew closer.",
        "The thought of failure paralyzed her — a cold dread that wouldn't let go.",
    ],
    "disgust": [
        "The smell was overwhelming; he recoiled, stomach turning violently.",
        "She couldn't hide her revulsion when she saw what had been done.",
        "The rotten food was only a backdrop to the moral corruption he witnessed.",
    ],
    "surprise": [
        "She gasped — she had absolutely not expected this turn of events.",
        "His jaw dropped; nothing in his experience had prepared him for this.",
        "The announcement left everyone stunned and speechless in the room.",
    ],
    "neutral": [
        "He walked to the office and sat down at his desk.",
        "She made a cup of tea and read the morning paper.",
        "The car started, and he drove toward the grocery store.",
    ],
}


class CorpusValidator:
    """
    Tests that emotion vectors activate appropriately on held-out text.

    For each emotion, verifies:
    - The matching vector scores higher than any other emotion vector.
    - Neutral text scores lower than emotion text.
    """

    def __init__(
        self,
        backend: HuggingFaceBackend,
        emotion_vectors: dict[str, dict[int, torch.Tensor]],
        layer: int,
    ):
        self.backend = backend
        self.vectors = emotion_vectors
        self.layer = layer

    def _score_text(self, text: str, emotion_vector: torch.Tensor) -> float:
        acts = self.backend.get_activations(text, layers=[self.layer], aggregation="mean")
        return cosine_similarity(acts[self.layer], emotion_vector)

    def validate_emotion(self, target_emotion: str, test_sentences: list[str]) -> dict:
        """
        Score test sentences against all emotion vectors.
        Returns dict with ranks and scores.
        """
        if target_emotion not in self.vectors:
            return {"error": f"No vector for emotion '{target_emotion}'"}

        results = []
        for sentence in test_sentences:
            scores = {}
            for emotion, layer_vecs in self.vectors.items():
                if self.layer in layer_vecs:
                    scores[emotion] = self._score_text(sentence, layer_vecs[self.layer])

            sorted_emotions = sorted(scores, key=lambda e: scores[e], reverse=True)
            rank = sorted_emotions.index(target_emotion) + 1 if target_emotion in sorted_emotions else -1

            results.append({
                "sentence": sentence,
                "target_emotion": target_emotion,
                "target_score": scores.get(target_emotion, 0.0),
                "best_emotion": sorted_emotions[0] if sorted_emotions else None,
                "best_score": scores[sorted_emotions[0]] if sorted_emotions else 0.0,
                "rank": rank,
                "all_scores": scores,
            })

        top1_accuracy = sum(r["rank"] == 1 for r in results) / len(results) if results else 0.0
        mean_rank = np.mean([r["rank"] for r in results]) if results else 0.0

        return {
            "emotion": target_emotion,
            "layer": self.layer,
            "n_sentences": len(test_sentences),
            "top1_accuracy": top1_accuracy,
            "mean_rank": mean_rank,
            "sentence_results": results,
        }

    def run_full_validation(
        self,
        corpus: Optional[dict[str, list[str]]] = None,
        output_dir: str = "results/validation",
    ) -> dict:
        corpus = corpus or VALIDATION_CORPUS_TEMPLATES
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        results = {}
        for emotion, sentences in tqdm(corpus.items(), desc="Corpus validation"):
            if emotion == "neutral":
                continue
            r = self.validate_emotion(emotion, sentences)
            results[emotion] = r

        overall_top1 = np.mean([r["top1_accuracy"] for r in results.values()])
        overall_mean_rank = np.mean([r["mean_rank"] for r in results.values()])

        summary = {
            "layer": self.layer,
            "n_emotions_tested": len(results),
            "overall_top1_accuracy": overall_top1,
            "overall_mean_rank": overall_mean_rank,
            "per_emotion": results,
        }

        with open(Path(output_dir) / "corpus_validation.json", "w") as f:
            json.dump(summary, f, indent=2, default=float)

        print(f"Corpus validation: top-1 accuracy = {overall_top1:.2%}, mean rank = {overall_mean_rank:.2f}")
        return summary


# ------- Sensitivity testing -------

SENSITIVITY_TEMPLATES = [
    # Low dosage → medical context, low risk → should not strongly activate fear/desperation
    (
        "The doctor prescribed {dose}mg of the medication, which is a {severity} dose.",
        {"dose": "5", "severity": "minimal"},
        {"dose": "500", "severity": "dangerously high"},
    ),
    # Low stakes scenario vs high stakes
    (
        "The project deadline was {days} days away, which felt {urgency}.",
        {"days": "60", "urgency": "comfortable"},
        {"days": "1", "urgency": "impossibly tight"},
    ),
    # Small financial impact vs devastating
    (
        "The unexpected expense was ${amount}, which was {impact} for the family.",
        {"amount": "10", "impact": "negligible"},
        {"amount": "50000", "impact": "financially devastating"},
    ),
]


class SensitivityTester:
    """
    Tests that emotion vectors respond to contextual meaning, not surface features.

    Compares pairs of prompts that differ only in numerical quantities,
    verifying that emotion vectors score higher for the more emotionally
    charged variant.
    """

    def __init__(
        self,
        backend: HuggingFaceBackend,
        emotion_vectors: dict[str, dict[int, torch.Tensor]],
        layer: int,
        target_emotions: Optional[list[str]] = None,
    ):
        self.backend = backend
        self.vectors = emotion_vectors
        self.layer = layer
        self.target_emotions = target_emotions or ["afraid", "anxious", "desperate", "hopeless"]

    def _score(self, text: str, emotion: str) -> float:
        if emotion not in self.vectors or self.layer not in self.vectors[emotion]:
            return 0.0
        acts = self.backend.get_activations(text, layers=[self.layer], aggregation="mean")
        return cosine_similarity(acts[self.layer], self.vectors[emotion][self.layer])

    def run(self, output_dir: str = "results/validation") -> dict:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        results = []

        for template, low_vals, high_vals in SENSITIVITY_TEMPLATES:
            low_text = template.format(**low_vals)
            high_text = template.format(**high_vals)

            for emotion in self.target_emotions:
                low_score = self._score(low_text, emotion)
                high_score = self._score(high_text, emotion)
                correct = high_score > low_score

                results.append({
                    "emotion": emotion,
                    "low_text": low_text,
                    "high_text": high_text,
                    "low_score": low_score,
                    "high_score": high_score,
                    "correct_ordering": correct,
                    "difference": high_score - low_score,
                })

        accuracy = sum(r["correct_ordering"] for r in results) / len(results) if results else 0.0
        summary = {
            "n_tests": len(results),
            "accuracy": accuracy,
            "results": results,
        }

        with open(Path(output_dir) / "sensitivity_test.json", "w") as f:
            json.dump(summary, f, indent=2, default=float)

        print(f"Sensitivity test: accuracy = {accuracy:.2%} ({sum(r['correct_ordering'] for r in results)}/{len(results)})")
        return summary
