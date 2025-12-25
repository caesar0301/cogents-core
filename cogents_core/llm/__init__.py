import os
from typing import Optional

from .base import BaseLLMClient

# Deprecated imports
from .litellm import LLMClient as LitellmClient
from .litellm import LLMClient as LitellmLLMClient
from .openai import LLMClient as OpenAIClient
from .openai import LLMClient as OpenAILLMClient
from .openrouter import LLMClient as OpenRouterClient
from .openrouter import LLMClient as OpenRouterLLMClient

# Optional import - llamacpp might not be available
try:
    from .llamacpp import LLMClient as LlamaCppClient
    from .llamacpp import LLMClient as LlamaCppLLMClient

    LLAMACPP_AVAILABLE = True
except ImportError:
    LlamaCppLLMClient = None
    LLAMACPP_AVAILABLE = False

# Optional import - ollama might not be available
try:
    from .ollama import LLMClient as OllamaClient
    from .ollama import LLMClient as OllamaLLMClient

    OLLAMA_AVAILABLE = True
except ImportError:
    OllamaLLMClient = None
    OLLAMA_AVAILABLE = False


__all__ = [
    "get_llm_client",
    "BaseLLMClient",
    # -- Deprecated imports --
    "LitellmLLMClient",
    "LlamaCppLLMClient",
    "OpenRouterLLMClient",
    "OllamaLLMClient",
    "OpenAILLMClient",
    # -- New imports --
    "LitellmClient",
    "LlamaCppClient",
    "OllamaClient",
    "OpenAIClient",
    "OpenRouterClient",
]

#############################
# Common LLM helper functions
#############################


def get_llm_client(
    provider: str = os.getenv("COGENTS_LLM_PROVIDER", "openai"),
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    structured_output: bool = True,
    instructor: Optional[bool] = None,
    chat_model: Optional[str] = None,
    vision_model: Optional[str] = None,
    embed_model: Optional[str] = None,
    **kwargs,
):
    """
    Get an LLM client instance based on the specified provider.

    Args:
        provider: LLM provider to use ("openrouter", "openai", "litellm" always available; "ollama", "llamacpp" require optional dependencies)
        base_url: Base URL for API (used by openai and ollama providers)
        api_key: API key for authentication (used by openai and openrouter providers)
        structured_output: Whether to enable structured output (default: True)
        instructor: Whether to enable instructor for structured output (deprecated, use structured_output instead)
        chat_model: Model to use for chat completions
        vision_model: Model to use for vision tasks
        embed_model: Model to use for embeddings
        **kwargs: Additional provider-specific arguments:
            - llamacpp: model_path, n_ctx, n_gpu_layers, etc.
            - others: depends on provider
            Note: 'instructor' in kwargs will be removed to avoid conflicts

    Returns:
        LLMClient instance for the specified provider

    Raises:
        ValueError: If provider is not supported or dependencies are missing
    """
    # Handle backward compatibility: enable instructor if either parameter is True
    # If instructor is explicitly provided (True/False), use it OR structured_output
    # If instructor is None, use structured_output
    enable_instructor = (instructor if instructor is not None else False) or structured_output

    # Remove 'instructor' from kwargs to avoid duplicate argument errors
    kwargs.pop("instructor", None)

    if provider == "openrouter":
        return OpenRouterLLMClient(
            base_url=base_url,
            api_key=api_key,
            instructor=enable_instructor,
            chat_model=chat_model,
            vision_model=vision_model,
            embed_model=embed_model,
            **kwargs,
        )
    elif provider == "openai":
        return OpenAILLMClient(
            base_url=base_url,
            api_key=api_key,
            instructor=enable_instructor,
            chat_model=chat_model,
            vision_model=vision_model,
            embed_model=embed_model,
            **kwargs,
        )
    elif provider == "ollama":
        if not OLLAMA_AVAILABLE:
            raise ValueError("ollama provider is not available. Please install the required dependencies.")
        return OllamaLLMClient(
            base_url=base_url,
            api_key=api_key,
            instructor=enable_instructor,
            chat_model=chat_model,
            vision_model=vision_model,
            embed_model=embed_model,
            **kwargs,
        )
    elif provider == "llamacpp":
        if not LLAMACPP_AVAILABLE:
            raise ValueError("llamacpp provider is not available. Please install the required dependencies.")
        return LlamaCppLLMClient(
            instructor=enable_instructor,
            chat_model=chat_model,
            vision_model=vision_model,
            embed_model=embed_model,
            **kwargs,
        )
    elif provider == "litellm":
        return LitellmLLMClient(
            base_url=base_url,
            api_key=api_key,
            instructor=enable_instructor,
            chat_model=chat_model,
            vision_model=vision_model,
            embed_model=embed_model,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. Supported providers: openrouter, openai, ollama, llamacpp, litellm"
        )
