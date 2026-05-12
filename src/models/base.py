from abc import ABC, abstractmethod
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
    ) -> str:
        """Generate text given a prompt."""
        ...

    def get_activations(
        self,
        text: str,
        layers: Optional[list[int]] = None,
    ) -> dict[int, torch.Tensor]:
        """Extract residual stream activations. Only available for HuggingFace backend."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support activation extraction. "
            "Use HuggingFaceBackend for this functionality."
        )

    def generate_with_steering(
        self,
        prompt: str,
        emotion_vector: torch.Tensor,
        layers: list[int],
        alpha: float,
        max_new_tokens: int = 300,
    ) -> str:
        """Generate text with emotion vector steering applied to residual stream."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support activation steering. "
            "Use HuggingFaceBackend for this functionality."
        )

    @property
    def num_layers(self) -> int:
        """Number of transformer layers."""
        raise NotImplementedError

    @property
    def hidden_size(self) -> int:
        """Hidden dimension size."""
        raise NotImplementedError
