from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .base import BaseBackend


class HuggingFaceBackend(BaseBackend):
    """
    HuggingFace Transformers backend with full activation access.

    Supports any decoder-only causal LM (LLaMA, Mistral, Qwen, Gemma, etc.)
    with a standard `model.model.layers` structure.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: str = "float16",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ):
        print(f"Loading model: {model_name}")
        self.device = device
        self._model_name = model_name

        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype if quantization_config is None else None,
            quantization_config=quantization_config,
            device_map=device if quantization_config else None,
            trust_remote_code=True,
        )
        if quantization_config is None:
            self.model = self.model.to(device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(
            f"Model loaded: {self.num_layers} layers, hidden_size={self.hidden_size}"
        )

    @property
    def num_layers(self) -> int:
        return len(self.model.model.layers)

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def _get_transformer_layers(self):
        return self.model.model.layers

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> str:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True
        ).to(self.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = outputs[0][input_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def get_activations(
        self,
        text: str,
        layers: Optional[list[int]] = None,
        aggregation: str = "mean",
    ) -> dict[int, torch.Tensor]:
        """
        Extract residual stream activations from transformer layers.

        Args:
            text: Input text to process.
            layers: Layer indices to capture (None = all layers).
            aggregation: How to aggregate over sequence dimension:
                         "mean" | "last" | "none"

        Returns:
            Dict mapping layer index -> tensor of shape [hidden_size].
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        captured: dict[int, torch.Tensor] = {}

        transformer_layers = self._get_transformer_layers()
        target_layers = layers if layers is not None else list(range(len(transformer_layers)))

        hooks = []
        for idx in target_layers:
            layer = transformer_layers[idx]

            def make_hook(layer_idx: int):
                def hook(module, input, output):
                    # Output is (hidden_states, ...) or just hidden_states
                    hs = output[0] if isinstance(output, tuple) else output
                    hs = hs.detach().cpu().float()  # [1, seq_len, hidden_size]
                    if aggregation == "last":
                        captured[layer_idx] = hs[0, -1, :]
                    elif aggregation == "mean":
                        captured[layer_idx] = hs[0].mean(dim=0)
                    else:
                        captured[layer_idx] = hs[0]
                return hook

            hooks.append(layer.register_forward_hook(make_hook(idx)))

        with torch.no_grad():
            self.model(**inputs)

        for h in hooks:
            h.remove()

        return captured

    def generate_with_steering(
        self,
        prompt: str,
        emotion_vector: torch.Tensor,
        layers: list[int],
        alpha: float,
        max_new_tokens: int = 300,
        do_sample: bool = True,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate text with emotion vector added to residual stream at specified layers.

        Args:
            prompt: Input prompt.
            emotion_vector: Steering vector of shape [hidden_size].
            layers: Which layers to apply steering at.
            alpha: Steering strength (positive = amplify, negative = suppress).
            max_new_tokens: Maximum tokens to generate.
        """
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True
        ).to(self.device)
        input_len = inputs["input_ids"].shape[1]

        vec = emotion_vector.to(self.device).float()

        transformer_layers = self._get_transformer_layers()
        hooks = []

        for idx in layers:
            layer = transformer_layers[idx]

            def make_steering_hook(steering_alpha: float):
                def hook(module, input, output):
                    hs = output[0] if isinstance(output, tuple) else output
                    steered = hs + steering_alpha * vec.unsqueeze(0).unsqueeze(0)
                    if isinstance(output, tuple):
                        return (steered,) + output[1:]
                    return steered
                return hook

            hooks.append(layer.register_forward_hook(make_steering_hook(alpha)))

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_p=0.95 if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        for h in hooks:
            h.remove()

        generated = outputs[0][input_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    @contextmanager
    def steering_context(
        self,
        emotion_vector: torch.Tensor,
        layers: list[int],
        alpha: float,
    ):
        """Context manager that applies emotion steering during the with-block."""
        vec = emotion_vector.to(self.device).float()
        transformer_layers = self._get_transformer_layers()
        hooks = []

        for idx in layers:
            layer = transformer_layers[idx]

            def make_hook(a: float):
                def hook(module, input, output):
                    hs = output[0] if isinstance(output, tuple) else output
                    steered = hs + a * vec.unsqueeze(0).unsqueeze(0)
                    if isinstance(output, tuple):
                        return (steered,) + output[1:]
                    return steered
                return hook

            hooks.append(layer.register_forward_hook(make_hook(alpha)))

        try:
            yield self
        finally:
            for h in hooks:
                h.remove()
