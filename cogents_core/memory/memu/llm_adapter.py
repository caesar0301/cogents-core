from typing import Any, Dict, List

from cogents_core.utils.logging import get_logger

logger = get_logger(__name__)

from cogents_core.llm import BaseLLMClient, get_llm_client

################################################################################
# MemU agent compatibility
################################################################################


class MemoryLLMAdapter:
    """Adapter to make cogents LLM clients compatible with memory agent system"""

    def __init__(self, llm_client: BaseLLMClient):
        """Initialize with the original LLM client"""
        self.llm_client = llm_client

    def simple_chat(self, message: str) -> str:
        """
        Simple chat method that wraps the completion method

        Args:
            message: The message to send to the LLM

        Returns:
            str: The LLM response
        """
        try:
            # Convert single message to messages format
            messages = [{"role": "user", "content": message}]

            # Call the completion method
            response = self.llm_client.completion(messages)

            # Return the response as string
            return str(response)

        except Exception as e:
            logger.error(f"Error in simple_chat: {e}")
            raise

    def chat_completion(self, messages: List[Dict[str, str]], tools=None, tool_choice=None, **kwargs) -> Any:
        """
        Chat completion method for automated memory processing

        Args:
            messages: List of message dictionaries
            tools: Optional tools for function calling
            tool_choice: Tool choice strategy
            **kwargs: Additional arguments

        Returns:
            Mock response object for memory agent compatibility
        """
        try:
            # For now, call the regular completion method
            # In a full implementation, this would handle tool calls properly
            response_text = self.llm_client.completion(messages, **kwargs)

            # Create a mock response object that the memory agent expects
            class MockResponse:
                def __init__(self, content, success=True):
                    self.success = success
                    self.content = content
                    self.tool_calls = []  # No function calling in this simplified version
                    self.error = None if success else "Mock error"

            return MockResponse(str(response_text))

        except Exception as e:
            logger.error(f"Error in chat_completion: {e}")

            class MockResponse:
                def __init__(self, error_msg):
                    self.success = False
                    self.content = ""
                    self.tool_calls = []
                    self.error = error_msg

            return MockResponse(str(e))

    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for text using the underlying LLM client

        Args:
            text: Text to embed

        Returns:
            List[float]: Embedding vector
        """
        return self.llm_client.embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using the underlying LLM client

        Args:
            texts: List of texts to embed

        Returns:
            List[List[float]]: List of embedding vectors
        """
        return self.llm_client.embed_batch(texts)

    def get_embedding_dimensions(self) -> int:
        """
        Get the embedding dimensions from the underlying LLM client

        Returns:
            int: Embedding dimensions
        """
        return self.llm_client.get_embedding_dimensions()


def _get_llm_client_memu_compatible(**kwargs) -> BaseLLMClient:
    """
    Get an LLM client with optional MemU system compatibility

    Args:
        **kwargs: Additional arguments to pass to the LLM client

    Returns:
        BaseLLMClient: Configured LLM client
    """
    return MemoryLLMAdapter(get_llm_client(structured_output=True, **kwargs))
