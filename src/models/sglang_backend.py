"""
SGLang backend — fast inference with structured generation support.

Supports text generation only; activation extraction requires HuggingFaceBackend.
Install: pip install 'emotion-concepts[sglang]'

Two usage modes:
  1. In-process runtime (SGLangBackend) — launches SGLang engine in the same process.
  2. Server mode (SGLangServerBackend) — connects to a running `python -m sglang.launch_server`.

Typical use:
    # In-process
    backend = SGLangBackend("Qwen/Qwen2.5-7B-Instruct")
    text = backend.generate("Write a story about joy.")

    # Server mode
    backend = SGLangServerBackend("http://localhost:30000")
    text = backend.generate("Write a story about joy.")
"""

from __future__ import annotations

from typing import Optional

import requests

from .base import BaseBackend


class SGLangBackend(BaseBackend):
    """
    In-process SGLang runtime.

    SGLang provides RadixAttention for prefix caching — excellent for experiments
    that share long system prompts (e.g., repeated behavioral evaluation runs).

    Not suitable for activation extraction or steering — use HuggingFaceBackend for those.
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
    ):
        """
        Args:
            model: HuggingFace model ID or local path.
            tp_size: Tensor parallel size (number of GPUs).
            dtype: Weight dtype — "auto", "float16", "bfloat16".
            mem_fraction_static: Fraction of GPU memory for KV cache.
            max_total_tokens: Maximum total token budget. None = auto.
            trust_remote_code: Pass to HuggingFace loader.
            port: Port for internal SGLang server.
        """
        try:
            import sglang as sgl  # noqa: F401
        except ImportError:
            raise ImportError(
                "SGLang is not installed. Run: pip install 'emotion-concepts[sglang]'\n"
                "Note: SGLang requires CUDA. For macOS use HuggingFace or Ollama."
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

        # Get tokenizer for chat template application
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> str:
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

        state = _gen.run(prompt_text=prompt)
        return state["response"]

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> list[str]:
        """Generate for multiple prompts in parallel — SGLang's main advantage."""
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

        states = _gen.run_batch(
            [{"prompt_text": p} for p in prompts],
            num_threads=len(prompts),
            progress_bar=True,
        )
        return [s["response"] for s in states]

    def chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 300,
        temperature: float = 0.8,
    ) -> str:
        """Chat-format generation using the model's chat template."""
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)

    def shutdown(self):
        """Shut down the SGLang runtime and release GPU memory."""
        self._runtime.shutdown()


class SGLangServerBackend(BaseBackend):
    """
    HTTP client for a running SGLang server.

    Start the server separately:
        python -m sglang.launch_server \\
            --model-path Qwen/Qwen2.5-7B-Instruct \\
            --port 30000 --tp 1

    Then connect:
        backend = SGLangServerBackend("http://localhost:30000")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:30000",
        model: Optional[str] = None,
        timeout: int = 120,
    ):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
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

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> str:
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
