"""
vLLM backend — fast local inference with native concurrency.

Install: pip install 'emotion-concepts[vllm]'

Two classes:
  VLLMBackend     — wraps the sync LLM engine; generate_concurrent() submits
                    all prompts as one batch (continuous batching internally).
  VLLMAsyncBackend — wraps AsyncLLMEngine; ideal when you want fine-grained
                    asyncio control or a persistent server-style engine.

For server mode (vllm serve ...), use OpenAICompatibleBackend instead.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Optional

from .base import BaseBackend


# ─── Sync batch engine ────────────────────────────────────────────────────────

class VLLMBackend(BaseBackend):
    """
    In-process vLLM backend using the synchronous LLM engine.

    vLLM's LLM.generate() already uses PagedAttention + continuous batching,
    so passing a list of prompts is the most efficient path: all requests are
    scheduled together in a single engine step rather than sequentially.

    generate_concurrent() overrides the ThreadPoolExecutor default and sends
    ALL prompts as one batch.
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
        max_concurrent: int = 256,
    ):
        """
        Args:
            model: HuggingFace model ID or local path.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU memory to use (0–1).
            max_model_len: Maximum sequence length. None = model default.
            dtype: Weight dtype — "auto", "float16", "bfloat16".
            quantization: "awq", "gptq", or None.
            max_concurrent: Default batch size for generate_concurrent().
        """
        try:
            from vllm import LLM  # noqa: F401
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Run: pip install 'emotion-concepts[vllm]'\n"
                "Note: vLLM requires CUDA and Linux."
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
        self._default_max_concurrent = max_concurrent
        self.tokenizer = self._llm.get_tokenizer()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> str:
        return self.generate_batch(
            [prompt],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )[0]

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> list[str]:
        """Submit all prompts as one vLLM batch (continuous batching)."""
        from vllm import SamplingParams
        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
        )
        outputs = self._llm.generate(prompts, sampling_params=params)
        # vLLM preserves input order
        return [o.outputs[0].text for o in outputs]

    def generate_concurrent(
        self,
        prompts: list[str],
        max_concurrent: int | None = None,
        **kwargs,
    ) -> list[str]:
        """
        Override: send ALL prompts as a single vLLM batch.

        vLLM's scheduler handles the concurrency internally via PagedAttention
        and continuous batching — no thread pool needed.  max_concurrent is
        accepted for API compatibility but ignored (vLLM self-tunes).
        """
        return self.generate_batch(prompts, **kwargs)

    def chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 300,
        temperature: float = 0.8,
    ) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)

    def chat_batch(
        self,
        batch_messages: list[list[dict]],
        max_new_tokens: int = 300,
        temperature: float = 0.8,
    ) -> list[str]:
        """Chat-format batch generation."""
        prompts = [
            self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in batch_messages
        ]
        return self.generate_batch(prompts, max_new_tokens=max_new_tokens, temperature=temperature)


# ─── Async engine ─────────────────────────────────────────────────────────────

class VLLMAsyncBackend(BaseBackend):
    """
    In-process vLLM backend using AsyncLLMEngine.

    Requests are submitted concurrently via asyncio.gather(); the engine
    processes them with continuous batching.  Use this when you want asyncio-
    native control or need to interleave generation with other async work.

    All public methods are synchronous wrappers; call run_async() directly
    if you are already inside an event loop.
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
        try:
            from vllm import AsyncLLMEngine, AsyncEngineArgs  # noqa: F401
        except ImportError:
            raise ImportError("pip install 'emotion-concepts[vllm]'")

        from vllm import AsyncLLMEngine, AsyncEngineArgs
        from transformers import AutoTokenizer

        engine_args = AsyncEngineArgs(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            quantization=quantization,
            seed=seed,
            trust_remote_code=trust_remote_code,
        )
        print(f"Loading vLLM AsyncLLMEngine: {model}")
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        self._model_name = model
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    async def _generate_one_async(
        self,
        prompt: str,
        sampling_params,
    ) -> str:
        request_id = str(uuid.uuid4())
        results_generator = self._engine.generate(prompt, sampling_params, request_id)
        final = None
        async for output in results_generator:
            final = output
        return final.outputs[0].text  # type: ignore[union-attr]

    async def generate_many_async(
        self,
        prompts: list[str],
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> list[str]:
        """Submit all prompts concurrently; await all results."""
        from vllm import SamplingParams
        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
        )
        tasks = [self._generate_one_async(p, params) for p in prompts]
        return list(await asyncio.gather(*tasks))

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> str:
        return asyncio.run(
            self.generate_many_async(
                [prompt], max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p, do_sample=do_sample,
            )
        )[0]

    def generate_concurrent(
        self,
        prompts: list[str],
        max_concurrent: int | None = None,
        **kwargs,
    ) -> list[str]:
        """Submit all prompts concurrently via asyncio (ignores max_concurrent)."""
        return asyncio.run(self.generate_many_async(prompts, **kwargs))
