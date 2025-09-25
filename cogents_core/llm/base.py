import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from cogents_core.consts import DEFAULT_EMBEDDING_DIMS

T = TypeVar("T")


class BaseLLMClient(ABC):
    """Client for interacting with LLMs via OpenRouter using OpenAI SDK."""

    def __init__(self, **kwargs):
        """
        Initialize the LLM client.

        Args:
            **kwargs: Additional arguments to pass to the LLM client
        """
        self.debug = os.getenv("COGENTS_DEBUG", "false").lower() == "true"

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embeddings for input text"""

    @abstractmethod
    def embed_batch(self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings for input text"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: List[str],
    ) -> List[Tuple[float, int, str]]:
        """
        Rerank chunks based on their relevance to the query.

        Args:
            query: The query to rank against
            chunks: List of text chunks to rerank

        Returns:
            List of tuples (similarity_score, original_index, chunk_text)
            sorted by similarity score in descending order
        """

    def get_embedding_dimensions(self) -> int:
        """
        Get the expected dimensions for embeddings from this provider.

        Returns:
            int: Expected embedding dimensions
        """
        return int(os.getenv("COGENTS_EMBEDDING_DIMS", str(DEFAULT_EMBEDDING_DIMS)))
