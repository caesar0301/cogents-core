"""
Ollama LLM provider for Cogents.

This module provides:
- Chat completion using Ollama models
- Image understanding using Ollama vision models
- Instructor integration for structured output

"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import ollama
from instructor import Instructor, Mode, patch

from cogents_core.llm.base import BaseLLMClient
from cogents_core.tracing import estimate_token_usage, get_token_tracker
from cogents_core.tracing.opik_tracing import configure_opik
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
    """Client for interacting with Ollama LLM services."""

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
        Initialize the Ollama LLM client.

        Args:
            base_url: Base URL for the Ollama API (defaults to http://localhost:11434)
            api_key: Not used for Ollama but kept for compatibility
            instructor: Whether to enable instructor for structured output
            chat_model: Model to use for chat completions (defaults to gemma3:4b)
            vision_model: Model to use for vision tasks (defaults to gemma3:4b)
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        # Configure Opik tracing for observability only if enabled
        if OPIK_AVAILABLE:
            configure_opik()
            self._opik_provider = "ollama"
        else:
            self._opik_provider = None

        # Set base URL (defaults to Ollama default)
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Initialize Ollama client
        self.client = ollama.Client(host=self.base_url)

        # Model configurations
        self.chat_model = chat_model or os.getenv("OLLAMA_CHAT_MODEL", "gemma3:4b")
        self.vision_model = vision_model or os.getenv("OLLAMA_VISION_MODEL", "gemma3:4b")
        self.embed_model = embed_model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")

        # Initialize instructor if requested
        self.instructor = None
        if instructor:
            # Create a mock OpenAI-compatible client for instructor
            try:
                from openai import OpenAI

                # Create a mock client that uses Ollama through OpenAI-compatible API
                mock_client = OpenAI(
                    base_url=f"{self.base_url}/v1",
                    api_key="ollama",  # Ollama doesn't require real API key
                )
                patched_client = patch(mock_client, mode=Mode.JSON)
                self.instructor = Instructor(
                    client=patched_client,
                    create=patched_client.chat.completions.create,
                    mode=Mode.JSON,
                )
            except ImportError:
                logger.warning("OpenAI package not available, structured completion will not work")

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
        Generate chat completion using Ollama.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional arguments

        Returns:
            Generated text response or streaming response
        """

        try:
            options = {
                "temperature": temperature,
            }
            if max_tokens:
                options["num_predict"] = max_tokens

            if self.debug:
                logger.debug(f"Chat completion: {messages}")

            response = self.client.chat(
                model=self.chat_model,
                messages=messages,
                stream=stream,
                options=options,
                **kwargs,
            )

            if stream:
                return response
            else:
                # Estimate token usage for logging
                try:
                    prompt_text = "\n".join([msg.get("content", "") for msg in messages])
                    completion_text = response["message"]["content"]
                    usage = estimate_token_usage(prompt_text, completion_text, self.chat_model, "completion")
                    get_token_tracker().record_usage(usage)
                    logger.debug(f"Estimated token usage for completion: {usage.total_tokens} tokens")
                except Exception as e:
                    logger.debug(f"Could not estimate token usage: {e}")

                return response["message"]["content"]

        except Exception as e:
            logger.error(f"Error in Ollama completion: {e}")
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
        Generate structured completion using instructor with Ollama.

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
                result = self.instructor.create(
                    model=self.chat_model,
                    messages=messages,
                    response_model=response_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )

                # Estimate token usage for logging
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
        Analyze an image using Ollama vision model.

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

            # Prepare the message with image
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_data],
                }
            ]

            options = {
                "temperature": temperature,
            }
            if max_tokens:
                options["num_predict"] = max_tokens

            if self.debug:
                logger.debug(f"Understand image: {messages}")

            response = self.client.chat(
                model=self.vision_model,
                messages=messages,
                options=options,
                **kwargs,
            )

            # Estimate token usage for logging
            try:
                completion_text = response["message"]["content"]
                usage = estimate_token_usage(prompt, completion_text, self.vision_model, "vision")
                get_token_tracker().record_usage(usage)
                logger.debug(f"Estimated token usage for vision: {usage.total_tokens} tokens")
            except Exception as e:
                logger.debug(f"Could not estimate token usage: {e}")

            return response["message"]["content"]

        except Exception as e:
            logger.error(f"Error analyzing image with Ollama: {e}")
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
        Analyze an image from URL using Ollama vision model.

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
            import requests

            # Download the image
            response = requests.get(image_url)
            response.raise_for_status()
            image_data = response.content

            # Prepare the message with image
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_data],
                }
            ]

            options = {
                "temperature": temperature,
            }
            if max_tokens:
                options["num_predict"] = max_tokens

            if self.debug:
                logger.debug(f"Understand image from url: {messages}")

            response = self.client.chat(
                model=self.vision_model,
                messages=messages,
                options=options,
                **kwargs,
            )

            # Estimate token usage for logging
            try:
                completion_text = response["message"]["content"]
                usage = estimate_token_usage(prompt, completion_text, self.vision_model, "vision")
                get_token_tracker().record_usage(usage)
                logger.debug(f"Estimated token usage for vision: {usage.total_tokens} tokens")
            except Exception as e:
                logger.debug(f"Could not estimate token usage: {e}")

            return response["message"]["content"]

        except Exception as e:
            logger.error(f"Error analyzing image from URL with Ollama: {e}")
            raise

    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings using Ollama.

        Args:
            text: Text to embed

        Returns:
            List of embedding values
        """
        try:
            response = self.client.embeddings(
                model=self.embed_model,
                prompt=text,
            )
            embedding = response["embedding"]

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
            logger.error(f"Error generating embedding with Ollama: {e}")
            raise

    def embed_batch(self, chunks: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using Ollama.

        Args:
            chunks: List of texts to embed

        Returns:
            List of embedding lists
        """
        embeddings = []
        for chunk in chunks:
            embedding = self.embed(chunk)
            embeddings.append(embedding)
        return embeddings

    def rerank(self, query: str, chunks: List[str]) -> List[Tuple[float, int, str]]:
        """
        Rerank chunks based on their relevance to the query.

        Note: Ollama doesn't have a native reranking API, so this implementation
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
            logger.error(f"Error reranking with Ollama: {e}")
            # Fallback: return original order with zero similarities
            return [(0.0, i, chunk) for i, chunk in enumerate(chunks)]
