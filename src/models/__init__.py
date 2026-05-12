from .base import BaseBackend
from .hf_backend import HuggingFaceBackend
from .ollama_backend import OllamaBackend, OpenAICompatibleBackend

__all__ = [
    "BaseBackend",
    "HuggingFaceBackend",
    "OllamaBackend",
    "OpenAICompatibleBackend",
    # vLLM and SGLang are imported lazily to avoid hard dependency errors
    # from .vllm_backend import VLLMBackend
    # from .sglang_backend import SGLangBackend, SGLangServerBackend
]


def get_backend(backend_type: str, **kwargs) -> BaseBackend:
    """
    Factory function for creating a model backend by name.

    Args:
        backend_type: One of "hf", "ollama", "openai", "vllm", "sglang", "sglang-server".
        **kwargs: Passed to the backend constructor.

    Examples:
        backend = get_backend("hf", model_name="Qwen/Qwen2.5-7B-Instruct", device="cuda")
        backend = get_backend("vllm", model="Qwen/Qwen2.5-7B-Instruct")
        backend = get_backend("sglang-server", base_url="http://localhost:30000")
    """
    if backend_type == "hf":
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
            "Choose from: hf, ollama, openai, vllm, sglang, sglang-server"
        )
