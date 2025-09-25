"""
LiteLLM provider for Cogents.

This module provides:
- Unified interface to multiple LLM providers via LiteLLM
- Chat completion using various models through LiteLLM
- Image understanding using vision models
- Instructor integration for structured output

- Support for OpenAI, Anthropic, Cohere, Ollama, and many other providers
"""

import base64
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import litellm
from instructor import Instructor, Mode, patch

from cogents_core.llm.base import BaseLLMClient
from cogents_core.tracing import configure_opik, estimate_token_usage, get_token_tracker, is_opik_enabled
from cogents_core.utils.logging import get_logger

# Only import OPIK if tracing is enabled
OPIK_AVAILABLE = False
track = lambda func: func  # Default no-op decorator
if os.getenv("COGENTS_OPIK_TRACING", "false").lower() == "true":
    try:
        from opik import track

        OPIK_AVAILABLE = True
    except ImportError:
        pass


T = TypeVar("T")

logger = get_logger(__name__)


class LLMClient(BaseLLMClient):
    """Client for interacting with multiple LLM services via LiteLLM."""

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
        Initialize the LiteLLM client.

        Args:
            base_url: Base URL for custom API endpoints (optional)
            api_key: API key for the provider (can be set via environment variables)
            instructor: Whether to enable instructor for structured output
            chat_model: Model to use for chat completions (e.g., "gpt-3.5-turbo", "claude-3-sonnet")
            vision_model: Model to use for vision tasks (e.g., "gpt-4-vision-preview", "claude-3-sonnet")
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        # Configure Opik tracing for observability only if enabled
        if OPIK_AVAILABLE:
            configure_opik()
            self._opik_provider = "litellm"
        else:
            self._opik_provider = None

        # Set base URL if provided
        self.base_url = base_url
        if self.base_url:
            litellm.api_base = self.base_url

        # Set API key if provided
        self.api_key = api_key
        if self.api_key:
            litellm.api_key = self.api_key

        # Model configurations
        self.chat_model = chat_model or os.getenv("LITELLM_CHAT_MODEL", "gpt-3.5-turbo")
        self.vision_model = vision_model or os.getenv("LITELLM_VISION_MODEL", "gpt-4-vision-preview")
        self.embed_model = embed_model or os.getenv("LITELLM_EMBED_MODEL", "text-embedding-ada-002")

        # Initialize instructor if requested
        self.instructor = None
        if instructor:
            try:
                from openai import OpenAI

                # Create a mock client for instructor
                mock_client = OpenAI(
                    api_key="litellm",
                    base_url="http://localhost:8000",  # LiteLLM proxy default
                )
                patched_client = patch(mock_client, mode=Mode.JSON)
                self.instructor = Instructor(
                    client=patched_client,
                    create=patched_client.chat.completions.create,
                    mode=Mode.JSON,
                )
            except ImportError:
                logger.warning("OpenAI package not available, structured completion will not work")

        # Configure LiteLLM settings
        litellm.drop_params = True  # Drop unsupported parameters
        litellm.set_verbose = False  # Reduce verbosity

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
        Generate chat completion using LiteLLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional arguments

        Returns:
            Generated text response or streaming response
        """
        # Add Opik tracing metadata
        opik_metadata = {}
        if is_opik_enabled():
            opik_metadata = {
                "provider": self._opik_provider,
                "model": self.chat_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
                "call_type": "completion",
            }

        try:
            if self.debug:
                logger.debug(f"Chat completion: {messages}")

            response = litellm.completion(
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
                # Extract token usage if available
                try:
                    if hasattr(response, "usage") and response.usage:
                        usage_data = {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens,
                            "model_name": self.chat_model,
                            "call_type": "completion",
                        }
                        from cogents_core.tracing import TokenUsage

                        usage = TokenUsage(**usage_data)
                        get_token_tracker().record_usage(usage)
                        logger.debug(f"Token usage for completion: {usage.total_tokens} tokens")
                    else:
                        # Fallback to estimation
                        prompt_text = "\n".join([msg.get("content", "") for msg in messages])
                        completion_text = response.choices[0].message.content
                        usage = estimate_token_usage(prompt_text, completion_text, self.chat_model, "completion")
                        get_token_tracker().record_usage(usage)
                        logger.debug(f"Estimated token usage for completion: {usage.total_tokens} tokens")
                except Exception as e:
                    logger.debug(f"Could not track token usage: {e}")

                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error in LiteLLM completion: {e}")
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
        Generate structured completion using instructor with LiteLLM.

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

        last_err = None
        for i in range(attempts):
            try:
                # Use LiteLLM directly with JSON mode for structured output
                # Add system message to enforce JSON structure
                structured_messages = messages.copy()
                if response_model.__doc__:
                    schema_prompt = f"Respond with JSON matching this schema: {response_model.model_json_schema()}"
                else:
                    schema_prompt = f"Respond with JSON matching this Pydantic model: {response_model.__name__}"

                # Add schema instruction to the last user message or create a new one
                if structured_messages and structured_messages[-1]["role"] == "user":
                    structured_messages[-1]["content"] += f"\n\n{schema_prompt}"
                else:
                    structured_messages.append({"role": "user", "content": schema_prompt})

                if self.debug:
                    logger.debug(f"Structured completion: {structured_messages}")

                response = litellm.completion(
                    model=self.chat_model,
                    messages=structured_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"} if "gpt" in self.chat_model.lower() else None,
                    **kwargs,
                )

                # Parse the JSON response into the Pydantic model
                import json

                response_text = response.choices[0].message.content
                response_json = json.loads(response_text)
                result = response_model.model_validate(response_json)

                # Estimate token usage for logging
                try:
                    prompt_text = "\n".join([msg.get("content", "") for msg in structured_messages])
                    completion_text = response_text
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
        Analyze an image using LiteLLM vision model.

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

            # Determine the image format
            image_format = image_path.suffix.lower().lstrip(".")
            if image_format == "jpg":
                image_format = "jpeg"

            # Prepare the message with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{image_base64}"}},
                    ],
                }
            ]

            if self.debug:
                logger.debug(f"Understand image: {messages}")

            response = litellm.completion(
                model=self.vision_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            # Estimate token usage for logging
            try:
                completion_text = response.choices[0].message.content
                usage = estimate_token_usage(prompt, completion_text, self.vision_model, "vision")
                get_token_tracker().record_usage(usage)
                logger.debug(f"Estimated token usage for vision: {usage.total_tokens} tokens")
            except Exception as e:
                logger.debug(f"Could not estimate token usage: {e}")

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error analyzing image with LiteLLM: {e}")
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
        Analyze an image from URL using LiteLLM vision model.

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
            # Prepare the message with image URL
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

            response = litellm.completion(
                model=self.vision_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            # Estimate token usage for logging
            try:
                completion_text = response.choices[0].message.content
                usage = estimate_token_usage(prompt, completion_text, self.vision_model, "vision")
                get_token_tracker().record_usage(usage)
                logger.debug(f"Estimated token usage for vision: {usage.total_tokens} tokens")
            except Exception as e:
                logger.debug(f"Could not estimate token usage: {e}")

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error analyzing image from URL with LiteLLM: {e}")
            raise

    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings using LiteLLM.

        Args:
            text: Text to embed

        Returns:
            List of embedding values
        """
        try:
            response = litellm.embedding(
                model=self.embed_model,
                input=[text],
                dimensions=self.get_embedding_dimensions(),
            )

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
            logger.error(f"Error generating embedding with LiteLLM: {e}")
            raise

    def embed_batch(self, chunks: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using LiteLLM.

        Args:
            chunks: List of texts to embed

        Returns:
            List of embedding lists
        """
        try:
            response = litellm.embedding(
                model=self.embed_model,
                input=chunks,
                dimensions=self.get_embedding_dimensions(),
            )

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
            logger.error(f"Error generating batch embeddings with LiteLLM: {e}")
            # Fallback to individual calls
            embeddings = []
            for chunk in chunks:
                embedding = self.embed(chunk)
                embeddings.append(embedding)
            return embeddings

    def rerank(self, query: str, chunks: List[str]) -> List[Tuple[float, int, str]]:
        """
        Rerank chunks based on their relevance to the query.

        Note: LiteLLM doesn't have a native reranking API, so this implementation
        uses a simple similarity-based approach with embeddings.

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
            logger.error(f"Error reranking with LiteLLM: {e}")
            # Fallback: return original order with zero similarities
            return [(0.0, i, chunk) for i, chunk in enumerate(chunks)]
