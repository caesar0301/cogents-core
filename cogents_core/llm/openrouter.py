"""
LLM utilities for Cogents using OpenRouter via OpenAI SDK.

This module provides:
- Chat completion using various models via OpenRouter
- Text embeddings using OpenAI text-embedding-3-small
- Image understanding using vision models
- Instructor integration for structured output

"""

import os
from typing import Optional, TypeVar

from cogents_core.consts import GEMINI_FLASH
from cogents_core.llm.openai import LLMClient as OpenAILLMClient
from cogents_core.tracing.opik_tracing import configure_opik
from cogents_core.utils.logging import get_logger

# Only import OPIK if tracing is enabled
OPIK_AVAILABLE = False
track = lambda func: func  # Default no-op decorator
if os.getenv("COGENTS_OPIK_TRACING", "false").lower() == "true":
    try:
        pass

        OPIK_AVAILABLE = True
    except ImportError:
        pass


T = TypeVar("T")

logger = get_logger(__name__)


class LLMClient(OpenAILLMClient):
    """Client for interacting with LLMs via OpenRouter using OpenAI SDK."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        instructor: bool = False,
        chat_model: Optional[str] = None,
        vision_model: Optional[str] = None,
        embed_model: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the LLM client.

        Args:
            base_url: Base URL for the OpenRouter API (defaults to OpenRouter's URL)
            api_key: API key for authentication (defaults to OPENROUTER_API_KEY env var)
            instructor: Whether to enable instructor for structured output
            chat_model: Model to use for chat completions (defaults to gemini-flash)
            vision_model: Model to use for vision tasks (defaults to gemini-flash)
            **kwargs: Additional arguments to pass to OpenAILLMClient
        """
        self.openrouter_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            raise ValueError(
                "OpenRouter API key is required. Provide api_key parameter or set OPENROUTER_API_KEY environment variable."
            )

        self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        # Model configurations (can be overridden by environment variables)
        self.chat_model = chat_model or os.getenv("OPENROUTER_CHAT_MODEL", GEMINI_FLASH)
        self.vision_model = vision_model or os.getenv("OPENROUTER_VISION_MODEL", GEMINI_FLASH)
        self.embed_model = embed_model or os.getenv("OPENROUTER_EMBED_MODEL", "text-embedding-3-small")

        super().__init__(
            base_url=self.base_url,
            api_key=self.openrouter_api_key,
            instructor=instructor,
            chat_model=self.chat_model,
            vision_model=self.vision_model,
            embed_model=self.embed_model,
            **kwargs,
        )

        # Configure Opik tracing for observability only if enabled
        if OPIK_AVAILABLE:
            configure_opik()
            self._opik_provider = "openrouter"
        else:
            self._opik_provider = None
