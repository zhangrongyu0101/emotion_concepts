"""
HuggingFace Model Pool — one model replica per GPU.

Designed for nodes with multiple high-VRAM GPUs (e.g. 8× H200-141G) where
each card can hold a complete model copy. Work is distributed via a
thread-safe queue: workers check out a backend, run inference, return it.

Activation extraction across N stories becomes N independent tasks dispatched
to N GPUs simultaneously, giving near-linear speedup.

Usage:
    pool = HuggingFaceModelPool("Qwen/Qwen3-32B", gpu_ids=[0,1,2,3,4,5,6,7])
    # Drop-in for HuggingFaceBackend — same interface
    acts = pool.get_activations(text, layers=[20, 40, 60])
    # Parallel bulk extraction across all GPUs
    all_acts = pool.get_activations_many(texts, layers=[20, 40, 60])
"""

from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Optional

import torch

from .hf_backend import HuggingFaceBackend


class HuggingFaceModelPool:
    """
    Thread-safe pool of HuggingFaceBackend instances, one per GPU.

    Each GPU runs a complete, independent model copy. Callers interact with
    the pool exactly like a single HuggingFaceBackend — the pool handles
    routing internally. Adds `get_activations_many()` for bulk parallel work.
    """

    def __init__(
        self,
        model_name: str,
        gpu_ids: list[int],
        dtype: str = "bfloat16",
    ):
        if not gpu_ids:
            raise ValueError("gpu_ids must be a non-empty list")

        self.gpu_ids = list(gpu_ids)
        self._backends: list[HuggingFaceBackend] = []
        self._queue: queue.Queue = queue.Queue()

        print(f"Initializing model pool: {len(gpu_ids)} replica(s) of {model_name}")
        for gpu_id in gpu_ids:
            print(f"  cuda:{gpu_id} — loading ...")
            b = HuggingFaceBackend(
                model_name=model_name,
                device=f"cuda:{gpu_id}",
                dtype=dtype,
            )
            self._backends.append(b)
            self._queue.put(b)

        print(f"Pool ready: {len(gpu_ids)} × {model_name} on GPUs {gpu_ids}")

    # ── Pool management ────────────────────────────────────────────────────────

    @contextmanager
    def _acquire(self):
        """Check out one backend; automatically returned after the with-block."""
        backend = self._queue.get()
        try:
            yield backend
        finally:
            self._queue.put(backend)

    @property
    def num_replicas(self) -> int:
        return len(self._backends)

    # ── Properties delegated to the first replica ──────────────────────────────

    @property
    def num_layers(self) -> int:
        return self._backends[0].num_layers

    @property
    def hidden_size(self) -> int:
        return self._backends[0].hidden_size

    @property
    def tokenizer(self):
        return self._backends[0].tokenizer

    # ── Single-item operations (dispatched to any free GPU) ────────────────────

    def generate(self, prompt: str, **kwargs) -> str:
        with self._acquire() as b:
            return b.generate(prompt, **kwargs)

    def get_activations(
        self,
        text: str,
        layers: Optional[list[int]] = None,
        aggregation: str = "mean",
    ) -> dict[int, torch.Tensor]:
        with self._acquire() as b:
            return b.get_activations(text, layers=layers, aggregation=aggregation)

    def generate_with_steering(
        self,
        prompt: str,
        emotion_vector: torch.Tensor,
        layers: list[int],
        alpha: float,
        **kwargs,
    ) -> str:
        with self._acquire() as b:
            return b.generate_with_steering(
                prompt,
                emotion_vector=emotion_vector,
                layers=layers,
                alpha=alpha,
                **kwargs,
            )

    # ── Bulk parallel operations ───────────────────────────────────────────────

    def get_activations_many(
        self,
        texts: list[str],
        layers: Optional[list[int]] = None,
        aggregation: str = "mean",
    ) -> list[dict[int, torch.Tensor]]:
        """
        Extract activations for multiple texts in parallel across all GPU replicas.

        Each text is sent to whichever GPU is free. Results are returned in
        the same order as the input list.
        """
        results: list[Optional[dict]] = [None] * len(texts)

        with ThreadPoolExecutor(max_workers=self.num_replicas) as ex:
            future_to_idx = {
                ex.submit(self.get_activations, text, layers, aggregation): i
                for i, text in enumerate(texts)
            }
            for future in as_completed(future_to_idx):
                results[future_to_idx[future]] = future.result()

        return results  # type: ignore[return-value]

    def generate_concurrent(
        self,
        prompts: list[str],
        max_concurrent: Optional[int] = None,
        **kwargs,
    ) -> list[str]:
        """
        Generate for multiple prompts in parallel across all GPU replicas.

        Results are returned in the same order as input prompts.
        """
        workers = max_concurrent or self.num_replicas
        results: list[Optional[str]] = [None] * len(prompts)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            future_to_idx = {
                ex.submit(self.generate, p, **kwargs): i
                for i, p in enumerate(prompts)
            }
            for future in as_completed(future_to_idx):
                results[future_to_idx[future]] = future.result()

        return results  # type: ignore[return-value]

    def parallel_run(
        self,
        fn,
        items: list,
    ) -> list:
        """
        Generic parallel dispatch: call fn(item) for each item across the pool.

        fn receives (backend, item) — use this for custom operations that need
        direct backend access (e.g. generate_with_steering in a batch).
        Results returned in input order.
        """
        results: list = [None] * len(items)
        lock = threading.Lock()

        def _worker(i, item):
            with self._acquire() as b:
                return i, fn(b, item)

        with ThreadPoolExecutor(max_workers=self.num_replicas) as ex:
            future_to_idx = {ex.submit(_worker, i, item): i for i, item in enumerate(items)}
            for future in as_completed(future_to_idx):
                i, result = future.result()
                results[i] = result

        return results
