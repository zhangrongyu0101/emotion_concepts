from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import torch


class BaseBackend(ABC):
    """Abstract base class for model backends."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> str:
        """Generate text given a single prompt."""
        ...

    def generate_concurrent(
        self,
        prompts: list[str],
        max_concurrent: int = 32,
        **kwargs,
    ) -> list[str]:
        """
        Generate text for multiple prompts concurrently.

        Default implementation uses a ThreadPoolExecutor, which is suitable for
        network-bound backends (Ollama, OpenAI-compatible APIs).  Subclasses
        override this with backend-native batching (vLLM, SGLang, HuggingFace).

        Args:
            prompts: List of input prompts.
            max_concurrent: Maximum number of concurrent workers / batch size.
            **kwargs: Forwarded to generate().

        Returns:
            List of generated strings, same order as prompts.
        """
        results: list[str | None] = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_to_idx = {
                executor.submit(self.generate, p, **kwargs): i
                for i, p in enumerate(prompts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results  # type: ignore[return-value]

    def get_activations(
        self,
        text: str,
        layers: Optional[list[int]] = None,
    ) -> dict[int, torch.Tensor]:
        """Extract residual-stream activations. Only available for HuggingFaceBackend."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support activation extraction. "
            "Use HuggingFaceBackend for this."
        )

    def generate_with_steering(
        self,
        prompt: str,
        emotion_vector: torch.Tensor,
        layers: list[int],
        alpha: float,
        max_new_tokens: int = 300,
    ) -> str:
        """Generate with emotion vector steering. Only available for HuggingFaceBackend."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support activation steering. "
            "Use HuggingFaceBackend for this."
        )

    @property
    def num_layers(self) -> int:
        raise NotImplementedError

    @property
    def hidden_size(self) -> int:
        raise NotImplementedError
