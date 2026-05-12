"""
SGLang backend — fast inference with RadixAttention prefix caching.

Install: pip install 'emotion-concepts[sglang]'

Two classes:
  SGLangBackend       — in-process runtime; generate_concurrent() uses
                        run_batch(num_threads=max_concurrent) for true parallelism.
  SGLangServerBackend — HTTP client for a running `python -m sglang.launch_server`;
                        generate_concurrent() sends requests via ThreadPoolExecutor.

Start server:
    python -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-7B-Instruct \
        --port 30000 --tp 1
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests

from .base import BaseBackend


# ─── In-process runtime ───────────────────────────────────────────────────────

class SGLangBackend(BaseBackend):
    """
    In-process SGLang runtime with RadixAttention prefix caching.

    For experiments that share long system prompts across many generations
    (e.g., behavioral evaluation with the same BLACKMAIL_SYSTEM prompt across
    all 20 × N trials), SGLang caches the KV for the shared prefix and only
    computes the unique suffix — giving substantial speed-ups vs. vLLM.

    generate_concurrent() maps directly to run_batch(num_threads=max_concurrent),
    which processes all requests concurrently inside the SGLang scheduler.
    """

    def __init__(
        self,
        model: str,
        tp_size: int = 1,
        dtype: str = "auto",
        mem_fraction_static: float = 0.88,
        max_total_tokens: Optional[int] = None,
        trust_remote_code: bool = True,
        port: int = 30000,
        max_concurrent: int = 128,
    ):
        """
        Args:
            model: HuggingFace model ID or local path.
            tp_size: Tensor parallel size (number of GPUs).
            mem_fraction_static: Fraction of GPU memory for KV cache.
            max_concurrent: Default for generate_concurrent().
        """
        try:
            import sglang as sgl  # noqa: F401
        except ImportError:
            raise ImportError(
                "SGLang is not installed. Run: pip install 'emotion-concepts[sglang]'\n"
                "Note: SGLang requires CUDA."
            )

        import sglang as sgl

        print(f"Launching SGLang runtime: {model} (tp={tp_size})")
        runtime_kwargs: dict = dict(
            model_path=model,
            tp_size=tp_size,
            dtype=dtype,
            mem_fraction_static=mem_fraction_static,
            trust_remote_code=trust_remote_code,
            port=port,
        )
        if max_total_tokens is not None:
            runtime_kwargs["max_total_tokens"] = max_total_tokens

        self._runtime = sgl.Runtime(**runtime_kwargs)
        sgl.set_default_backend(self._runtime)
        self._sgl = sgl
        self._model_name = model
        self._default_max_concurrent = max_concurrent

        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    def _make_sgl_fn(self, max_new_tokens: int, temperature: float, top_p: float, do_sample: bool):
        """Return a compiled sgl.function for generation."""
        sgl = self._sgl

        @sgl.function
        def _gen(s, prompt_text):
            s += prompt_text
            s += sgl.gen(
                "response",
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 0.0,
                top_p=top_p if do_sample else 1.0,
            )

        return _gen

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> str:
        fn = self._make_sgl_fn(max_new_tokens, temperature, top_p, do_sample)
        state = fn.run(prompt_text=prompt)
        return state["response"]

    def generate_concurrent(
        self,
        prompts: list[str],
        max_concurrent: int | None = None,
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_p: float = 0.95,
        do_sample: bool = True,
        **kwargs,
    ) -> list[str]:
        """
        Override: use SGLang's run_batch() for native concurrent execution.

        All prompts are scheduled by the SGLang runtime simultaneously;
        num_threads controls how many Python threads submit requests in parallel.
        Combined with RadixAttention, shared prefixes (e.g. system prompts) are
        computed only once.
        """
        n_threads = max_concurrent or self._default_max_concurrent
        fn = self._make_sgl_fn(max_new_tokens, temperature, top_p, do_sample)
        states = fn.run_batch(
            [{"prompt_text": p} for p in prompts],
            num_threads=min(n_threads, len(prompts)),
            progress_bar=True,
        )
        return [s["response"] for s in states]

    def chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 300,
        temperature: float = 0.8,
    ) -> str:
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)

    def chat_concurrent(
        self,
        batch_messages: list[list[dict]],
        max_concurrent: int | None = None,
        **kwargs,
    ) -> list[str]:
        """Batch chat-format generation."""
        prompts = [
            self._tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in batch_messages
        ]
        return self.generate_concurrent(prompts, max_concurrent=max_concurrent, **kwargs)

    def shutdown(self):
        self._runtime.shutdown()


# ─── Server HTTP client ───────────────────────────────────────────────────────

class SGLangServerBackend(BaseBackend):
    """
    HTTP client for a running SGLang server.

    generate_concurrent() sends all requests in parallel via a thread pool,
    letting the server-side scheduler (with RadixAttention) batch them.

    Start the server:
        python -m sglang.launch_server \\
            --model-path Qwen/Qwen2.5-7B-Instruct \\
            --port 30000 --tp 1
    """

    def __init__(
        self,
        base_url: str = "http://localhost:30000",
        model: Optional[str] = None,
        timeout: int = 120,
        max_concurrent: int = 128,
    ):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._default_max_concurrent = max_concurrent
        self._model = model or self._detect_model()

    def _detect_model(self) -> str:
        try:
            resp = requests.get(f"{self._base_url}/v1/models", timeout=5)
            resp.raise_for_status()
            models = resp.json().get("data", [])
            if models:
                return models[0]["id"]
        except Exception:
            pass
        return "unknown"

    def _post_completion(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float, do_sample: bool) -> str:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": temperature if do_sample else 0.0,
            "top_p": top_p if do_sample else 1.0,
        }
        resp = requests.post(
            f"{self._base_url}/v1/completions",
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["text"]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> str:
        return self._post_completion(prompt, max_new_tokens, temperature, top_p, do_sample)

    def generate_concurrent(
        self,
        prompts: list[str],
        max_concurrent: int | None = None,
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_p: float = 0.95,
        do_sample: bool = True,
        **kwargs,
    ) -> list[str]:
        """
        Send all prompts concurrently to the SGLang HTTP server.

        The server's RadixAttention scheduler handles shared-prefix caching
        across concurrent requests automatically.
        """
        n_workers = max_concurrent or self._default_max_concurrent
        results: list[str | None] = [None] * len(prompts)

        with ThreadPoolExecutor(max_workers=min(n_workers, len(prompts))) as executor:
            future_to_idx = {
                executor.submit(
                    self._post_completion, p, max_new_tokens, temperature, top_p, do_sample
                ): i
                for i, p in enumerate(prompts)
            }
            for future in as_completed(future_to_idx):
                results[future_to_idx[future]] = future.result()

        return results  # type: ignore[return-value]

    def chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 300,
        temperature: float = 0.8,
    ) -> str:
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        resp = requests.post(
            f"{self._base_url}/v1/chat/completions",
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
