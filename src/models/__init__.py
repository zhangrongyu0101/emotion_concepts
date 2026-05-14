from .base import BaseBackend
from .hf_backend import HuggingFaceBackend
from .hf_pool import HuggingFaceModelPool
from .ollama_backend import OllamaBackend, OpenAICompatibleBackend

__all__ = [
    "BaseBackend",
    "HuggingFaceBackend",
    "HuggingFaceModelPool",
    "OllamaBackend",
    "OpenAICompatibleBackend",
    # vLLM and SGLang are imported lazily to avoid hard dependency errors
    # from .vllm_backend import VLLMBackend
    # from .sglang_backend import SGLangBackend, SGLangServerBackend
]


def get_backend(backend_type: str, **kwargs) -> BaseBackend:
    """
    Factory function for creating a model backend by name.

    For the "hf" backend, pass gpu_ids=[0,1,...,7] to automatically create a
    HuggingFaceModelPool (one replica per GPU) instead of a single-GPU backend.

    Args:
        backend_type: One of "hf", "hf-pool", "ollama", "openai", "vllm",
                      "sglang", "sglang-server".
        **kwargs: Passed to the backend constructor.

    Examples:
        # Single GPU
        backend = get_backend("hf", model_name="Qwen/Qwen3-32B", device="cuda:0")
        # 8-GPU pool (one model copy per GPU)
        backend = get_backend("hf", model_name="Qwen/Qwen3-32B", gpu_ids=[0,1,2,3,4,5,6,7])
        # Explicit pool type
        backend = get_backend("hf-pool", model_name="Qwen/Qwen3-32B", gpu_ids=[0,1,2,3])
        backend = get_backend("vllm", model="Qwen/Qwen3-32B")
        backend = get_backend("sglang-server", base_url="http://localhost:30000")
    """
    if backend_type in ("hf", "hf-pool"):
        gpu_ids = kwargs.pop("gpu_ids", None)
        if gpu_ids:
            kwargs.pop("device", None)   # pool uses gpu_ids, not a single device
            return HuggingFaceModelPool(gpu_ids=gpu_ids, **kwargs)
        return HuggingFaceBackend(**kwargs)
    elif backend_type == "ollama":
        return OllamaBackend(**kwargs)
    elif backend_type == "openai":
        return OpenAICompatibleBackend(**kwargs)
    elif backend_type == "vllm":
        from .vllm_backend import VLLMBackend
        return VLLMBackend(**kwargs)
    elif backend_type == "sglang":
        from .sglang_backend import SGLangBackend
        return SGLangBackend(**kwargs)
    elif backend_type == "sglang-server":
        from .sglang_backend import SGLangServerBackend
        return SGLangServerBackend(**kwargs)
    else:
        raise ValueError(
            f"Unknown backend: '{backend_type}'. "
            "Choose from: hf, hf-pool, ollama, openai, vllm, sglang, sglang-server"
        )
