"""
LLM utilities for Cogents using OpenAI-compatible APIs.

This module provides:
- Chat completion using various models via OpenAI-compatible endpoints
- Image understanding using vision models
- Instructor integration for structured output

- Configurable base URL and API key for OpenAI-compatible services
"""

import base64
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

# Import instructor for structured output
from instructor import Instructor, Mode, patch
from openai import OpenAI

from cogents_core.llm.base import BaseLLMClient
from cogents_core.tracing import (
    configure_opik,
    estimate_token_usage,
    extract_token_usage_from_openai_response,
    get_token_tracker,
    is_opik_enabled,
)
from cogents_core.utils.logging import get_logger

# Only import OPIK if tracing is enabled
OPIK_AVAILABLE = False
track = lambda func: func  # Default no-op decorator
track_openai = lambda client: client  # Default no-op function
if os.getenv("COGENTS_OPIK_TRACING", "false").lower() == "true":
    try:
        from opik import track
        from opik.integrations.openai import track_openai

        OPIK_AVAILABLE = True
    except ImportError:
        pass


T = TypeVar("T")

logger = get_logger(__name__)


class LLMClient(BaseLLMClient):
    """Client for interacting with OpenAI-compatible LLM services."""

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
            base_url: Base URL for the OpenAI-compatible API (defaults to OpenAI's URL)
            api_key: API key for authentication (defaults to OPENAI_API_KEY env var)
            instructor: Whether to enable instructor for structured output
            chat_model: Model to use for chat completions (defaults to gpt-3.5-turbo)
            vision_model: Model to use for vision tasks (defaults to gpt-4-vision-preview)
            embed_model: Model to use for embeddings (defaults to text-embedding-3-small)
            **kwargs: Additional arguments to pass to the LLM client
        """
        super().__init__(**kwargs)
        # Configure Opik tracing for observability only if enabled
        if OPIK_AVAILABLE:
            configure_opik()
            self._opik_provider = "openai"
        else:
            self._opik_provider = None

        # Set API key from parameter or environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Provide api_key parameter or set OPENAI_API_KEY environment variable."
            )

        # Set base URL (defaults to OpenAI if not provided)
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        # Initialize OpenAI client
        client_kwargs = {"api_key": self.api_key, **kwargs}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        base_client = OpenAI(**client_kwargs)

        # Wrap with Opik tracking if available
        self.client = track_openai(base_client) if OPIK_AVAILABLE and is_opik_enabled() else base_client

        # Model configurations
        self.chat_model = chat_model or os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")
        self.vision_model = vision_model or os.getenv("OPENAI_VISION_MODEL", "gpt-4-vision-preview")
        self.embed_model = embed_model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

        # Initialize instructor if requested
        self.instructor = None
        if instructor:
            # Create instructor instance for structured output
            patched_client = patch(self.client, mode=Mode.JSON)
            self.instructor = Instructor(
                client=patched_client,
                create=patched_client.chat.completions.create,
                mode=Mode.JSON,
            )

    @track
    def completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate chat completion using the configured model.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional arguments to pass to OpenAI API

        Returns:
            Generated text response or streaming response
        """

        try:
            if self.debug:
                logger.debug(f"Chat completion: {messages}")
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs,
            )
            if stream:
                return response
            else:
                # Log token usage if available
                self._log_token_usage_if_available(response)
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise

    @track
    def structured_completion(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[T],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        attempts: int = 2,
        backoff: float = 0.5,
        **kwargs,
    ) -> T:
        """
        Generate structured completion using instructor.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            response_model: Pydantic model class for structured output
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            attempts: Number of attempts to make
            backoff: Backoff factor for exponential backoff
            **kwargs: Additional arguments to pass to instructor

        Returns:
            Structured response as the specified model type
        """
        if not self.instructor:
            raise ValueError("Instructor is not enabled. Initialize LLMClient with instructor=True")

        if self.debug:
            logger.debug(f"Structured completion: {messages}")

        last_err = None
        for i in range(attempts):
            try:
                # Capture token usage by enabling detailed response
                kwargs_with_usage = kwargs.copy()
                kwargs_with_usage.setdefault("stream", False)

                result = self.instructor.create(
                    model=self.chat_model,
                    messages=messages,
                    response_model=response_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs_with_usage,
                )

                # Try to capture token usage from instructor's underlying response
                # The instructor library usually stores the raw response
                if hasattr(result, "_raw_response"):
                    self._log_token_usage_if_available(result._raw_response, "structured")
                else:
                    # If no raw response, try to estimate usage
                    try:
                        prompt_text = "\n".join([msg.get("content", "") for msg in messages])
                        completion_text = str(result)
                        if hasattr(result, "model_dump_json"):
                            completion_text = result.model_dump_json()

                        usage = estimate_token_usage(prompt_text, completion_text, self.chat_model, "structured")
                        get_token_tracker().record_usage(usage)
                        logger.debug(f"Estimated token usage for structured completion: {usage.total_tokens} tokens")
                    except Exception as e:
                        logger.debug(f"Could not estimate token usage: {e}")

                return result
            except Exception as e:
                last_err = e
                if i < attempts - 1:
                    time.sleep(backoff * (2**i))
                else:
                    logger.error(f"Error in structured completion: {e}")
                    raise
        raise last_err

    @track
    def understand_image(
        self,
        image_path: Union[str, Path],
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Analyze an image using the configured vision model.

        Args:
            image_path: Path to the image file
            prompt: Text prompt describing what to analyze in the image
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments

        Returns:
            Analysis of the image
        """

        try:
            # Read and encode the image
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                image_base64 = base64.b64encode(image_data).decode("utf-8")

            # Create message with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        },
                    ],
                }
            ]

            if self.debug:
                logger.debug(f"Understand image: {messages}")

            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            # Record token usage for vision call
            self._log_token_usage_if_available(response, "vision")
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            raise

    @track
    def understand_image_from_url(
        self,
        image_url: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Analyze an image from URL using the configured vision model.

        Args:
            image_url: URL of the image
            prompt: Text prompt describing what to analyze in the image
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments

        Returns:
            Analysis of the image
        """

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ]

            if self.debug:
                logger.debug(f"Understand image from url: {messages}")

            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            # Record token usage for vision URL call
            self._log_token_usage_if_available(response, "vision")
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error analyzing image from URL: {e}")
            raise

    def _log_token_usage_if_available(self, response, call_type: str = "completion"):
        """Extract and record token usage from OpenAI response if available."""
        try:
            usage = extract_token_usage_from_openai_response(response, self.chat_model, call_type)
            if usage:
                get_token_tracker().record_usage(usage)
                logger.debug(
                    f"Token usage - Prompt: {usage.prompt_tokens}, "
                    f"Completion: {usage.completion_tokens}, "
                    f"Total: {usage.total_tokens} (model: {usage.model_name})"
                )
        except Exception as e:
            logger.debug(f"Could not extract token usage: {e}")

    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings using OpenAI's embedding model.

        Args:
            text: Text to embed

        Returns:
            List of embedding values
        """
        try:
            response = self.client.embeddings.create(
                model=self.embed_model,
                input=text,
                dimensions=self.get_embedding_dimensions(),
            )

            # Record token usage if available
            try:
                if hasattr(response, "usage") and response.usage:
                    usage_data = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": 0,  # Embeddings don't have completion tokens
                        "total_tokens": response.usage.total_tokens,
                        "model_name": self.embed_model,
                        "call_type": "embedding",
                    }
                    from cogents_core.tracing import TokenUsage

                    usage = TokenUsage(**usage_data)
                    get_token_tracker().record_usage(usage)
                    logger.debug(f"Token usage for embedding: {usage.total_tokens} tokens")
            except Exception as e:
                logger.debug(f"Could not track embedding token usage: {e}")

            embedding = response.data[0].embedding

            # Validate embedding dimensions
            expected_dims = self.get_embedding_dimensions()
            if len(embedding) != expected_dims:
                logger.warning(
                    f"Embedding has {len(embedding)} dimensions, expected {expected_dims}. "
                    f"Consider setting COGENTS_EMBEDDING_DIMS={len(embedding)} or "
                    f"using a different embedding model."
                )

            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding with OpenAI: {e}")
            raise

    def embed_batch(self, chunks: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using OpenAI.

        Args:
            chunks: List of texts to embed

        Returns:
            List of embedding lists
        """
        try:
            response = self.client.embeddings.create(
                model=self.embed_model,
                input=chunks,
                dimensions=self.get_embedding_dimensions(),
            )

            # Record token usage if available
            try:
                if hasattr(response, "usage") and response.usage:
                    usage_data = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": 0,
                        "total_tokens": response.usage.total_tokens,
                        "model_name": self.embed_model,
                        "call_type": "embedding",
                    }
                    from cogents_core.tracing import TokenUsage

                    usage = TokenUsage(**usage_data)
                    get_token_tracker().record_usage(usage)
                    logger.debug(f"Token usage for batch embedding: {usage.total_tokens} tokens")
            except Exception as e:
                logger.debug(f"Could not track batch embedding token usage: {e}")

            embeddings = [item.embedding for item in response.data]

            # Validate embedding dimensions
            expected_dims = self.get_embedding_dimensions()
            for i, embedding in enumerate(embeddings):
                if len(embedding) != expected_dims:
                    logger.warning(
                        f"Embedding at index {i} has {len(embedding)} dimensions, expected {expected_dims}. "
                        f"Consider setting COGENTS_EMBEDDING_DIMS={len(embedding)} or "
                        f"using a different embedding model."
                    )

            return embeddings

        except Exception as e:
            logger.error(f"Error generating batch embeddings with OpenAI: {e}")
            # Fallback to individual calls
            embeddings = []
            for chunk in chunks:
                embedding = self.embed(chunk)
                embeddings.append(embedding)
            return embeddings

    def rerank(self, query: str, chunks: List[str]) -> List[Tuple[float, int, str]]:
        """
        Rerank chunks based on their relevance to the query using embeddings.

        Note: OpenAI doesn't have a native reranking API, so this implementation
        uses a similarity-based approach with embeddings.

        Args:
            query: The query to rank against
            chunks: List of text chunks to rerank

        Returns:
            List of tuples (similarity_score, original_index, chunk_text)
            sorted by similarity score in descending order
        """
        try:
            # Get embeddings for query and chunks
            query_embedding = self.embed(query)
            chunk_embeddings = self.embed_batch(chunks)

            from cogents_core.utils.statistics import cosine_similarity

            # Calculate similarities and sort
            similarities = []
            for i, chunk_embedding in enumerate(chunk_embeddings):
                similarity = cosine_similarity(query_embedding, chunk_embedding)
                similarities.append((similarity, i, chunks[i]))

            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[0], reverse=True)

            # Return sorted tuples
            return similarities

        except Exception as e:
            logger.error(f"Error reranking with OpenAI: {e}")
            # Fallback: return original order with zero similarities
            return [(0.0, i, chunk) for i, chunk in enumerate(chunks)]
