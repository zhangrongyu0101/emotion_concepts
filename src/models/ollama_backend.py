from __future__ import annotations

import json
from typing import Optional

import requests

from .base import BaseBackend


class OllamaBackend(BaseBackend):
    """
    Ollama backend for local model inference.

    Supports text generation only — no activation extraction or steering.
    Use this backend for behavioral evaluation experiments.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        host: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._verify_connection()

    def _verify_connection(self):
        try:
            resp = requests.get(f"{self._host}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if not any(self._model in m for m in models):
                print(
                    f"Warning: model '{self._model}' not found in Ollama. "
                    f"Available: {models}. Run: ollama pull {self._model}"
                )
        except requests.exceptions.ConnectionError:
            print(
                f"Warning: Could not connect to Ollama at {self._host}. "
                "Make sure Ollama is running: ollama serve"
            )

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
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": temperature if do_sample else 0.0,
                "top_p": top_p if do_sample else 1.0,
            },
        }
        resp = requests.post(
            f"{self._host}/api/generate",
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()["response"]

    def chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 300,
        temperature: float = 0.8,
    ) -> str:
        """Send a chat-format request (list of {role, content} dicts)."""
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": temperature,
            },
        }
        resp = requests.post(
            f"{self._host}/api/chat",
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


class OpenAICompatibleBackend(BaseBackend):
    """
    Backend for any OpenAI-compatible API endpoint (vLLM, LM Studio, etc.).

    Use for behavioral evaluation with any locally-hosted model.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "not-needed",
        timeout: int = 120,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._timeout = timeout

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> str:
        response = self._client.completions.create(
            model=self._model,
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
        )
        return response.choices[0].text

    def chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 300,
        temperature: float = 0.8,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
