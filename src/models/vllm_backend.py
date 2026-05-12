"""
vLLM backend — fast local inference.

Supports text generation only; activation extraction requires HuggingFaceBackend.
Install: pip install 'emotion-concepts[vllm]'

Typical use:
    backend = VLLMBackend("Qwen/Qwen2.5-7B-Instruct", tensor_parallel_size=1)
    text = backend.generate("Write a story about joy.")

For serving mode (vllm serve ...), use OpenAICompatibleBackend in ollama_backend.py.
"""

from __future__ import annotations

from typing import Optional

from .base import BaseBackend


class VLLMBackend(BaseBackend):
    """
    In-process vLLM backend.

    Loads the model once and keeps it in GPU memory for fast repeated inference.
    Much faster than HuggingFace for generation-only workloads.

    Not suitable for activation extraction or steering — use HuggingFaceBackend for those.
    """

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: Optional[int] = None,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        seed: int = 42,
        trust_remote_code: bool = True,
    ):
        """
        Args:
            model: HuggingFace model ID or local path.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU memory to use (0–1).
            max_model_len: Maximum sequence length. None = use model default.
            dtype: Weight dtype — "auto", "float16", "bfloat16".
            quantization: Optional quantization — "awq", "gptq", "squeezellm", None.
            seed: Random seed for reproducibility.
            trust_remote_code: Pass to HuggingFace loader.
        """
        try:
            from vllm import LLM, SamplingParams as _SP  # noqa: F401
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Run: pip install 'emotion-concepts[vllm]'\n"
                "Note: vLLM requires CUDA and Linux. For macOS use HuggingFace or Ollama."
            )

        from vllm import LLM

        kwargs: dict = dict(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            seed=seed,
            trust_remote_code=trust_remote_code,
        )
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        if quantization is not None:
            kwargs["quantization"] = quantization

        print(f"Loading vLLM engine: {model}")
        self._llm = LLM(**kwargs)
        self._model_name = model

        # Expose tokenizer for prompt formatting
        self.tokenizer = self._llm.get_tokenizer()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_p: float = 0.95,
        do_sample: bool = True,
        n: int = 1,
    ) -> str:
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
            n=n,
        )
        outputs = self._llm.generate([prompt], sampling_params=params)
        return outputs[0].outputs[0].text

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> list[str]:
        """Generate for a batch of prompts in parallel — vLLM's main advantage."""
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
        )
        outputs = self._llm.generate(prompts, sampling_params=params)
        return [o.outputs[0].text for o in outputs]

    def chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 300,
        temperature: float = 0.8,
    ) -> str:
        """
        Chat-format generation using the model's chat template.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
        """
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
