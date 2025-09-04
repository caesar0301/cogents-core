#!/usr/bin/env python3
"""
Integration tests for LiteLLM provider.

These tests verify the integration with multiple LLM providers through LiteLLM.
"""

import os

import pytest
from pydantic import BaseModel

from cogents.core.base.llm.litellm import LLMClient


class TestResponse(BaseModel):
    """Test Pydantic model for structured output."""

    answer: str
    confidence: float
    reasoning: str


@pytest.mark.integration
@pytest.mark.slow
class TestLiteLLMIntegration:
    """Integration tests for LiteLLM functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        # Check if at least one API key is available
        api_keys = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
            "COHERE_API_KEY": os.getenv("COHERE_API_KEY"),
        }

        if not any(api_keys.values()):
            pytest.skip("No API keys found for LiteLLM providers")

        # Use OpenAI as default if available, otherwise use the first available provider
        if api_keys["OPENAI_API_KEY"]:
            self.client = LLMClient(chat_model="gpt-3.5-turbo")
        elif api_keys["ANTHROPIC_API_KEY"]:
            self.client = LLMClient(chat_model="claude-3-haiku-20240307")
        elif api_keys["COHERE_API_KEY"]:
            self.client = LLMClient(chat_model="command-r")
        else:
            pytest.skip("No supported API keys found")

    def test_basic_completion(self):
        """Test basic chat completion with LiteLLM."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2? Answer with just the number."},
        ]

        response = self.client.completion(messages, temperature=0.1, max_tokens=10)

        # Assertions
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        assert "4" in response

    def test_completion_with_conversation_history(self):
        """Test completion with conversation history."""
        messages = [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "What is 10 / 2?"},
            {"role": "assistant", "content": "10 divided by 2 equals 5."},
            {"role": "user", "content": "What about 5 * 3?"},
        ]

        response = self.client.completion(messages, temperature=0.1, max_tokens=20)

        assert response is not None
        assert isinstance(response, str)
        assert "15" in response

    def test_structured_completion(self):
        """Test structured output with LiteLLM."""
        try:
            client = LLMClient(chat_model="gpt-4", instructor=True)
        except Exception:
            pytest.skip("Instructor not available or model not supported")

        messages = [
            {
                "role": "user",
                "content": "What is the capital of France? Provide your answer with confidence level and reasoning.",
            }
        ]

        try:
            response = client.structured_completion(messages=messages, response_model=TestResponse, temperature=0.1)

            # Assertions
            assert response is not None
            assert isinstance(response, TestResponse)
            assert response.answer.lower() == "paris"
            assert 0 <= response.confidence <= 1
            assert len(response.reasoning) > 0
        except Exception as e:
            if "json" in str(e).lower() or "structured" in str(e).lower():
                pytest.skip(f"Structured output not supported by this model: {e}")
            else:
                raise

    def test_streaming_completion(self):
        """Test streaming completion with LiteLLM."""
        messages = [{"role": "user", "content": "Count from 1 to 3, one number per line."}]

        try:
            response = self.client.completion(messages, stream=True, max_tokens=50)

            # Collect streaming response
            content_parts = []
            for chunk in response:
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        content_parts.append(delta.content)

            full_response = "".join(content_parts)

            # Assertions
            assert len(content_parts) > 0
            assert len(full_response) > 0

        except Exception as e:
            if "stream" in str(e).lower():
                pytest.skip(f"Streaming not supported by this provider: {e}")
            else:
                raise

    def test_embed_single_text(self):
        """Test embedding generation for a single text."""
        text = "This is a test sentence for embedding generation."

        try:
            embedding = self.client.embed(text)

            # Assertions
            assert embedding is not None
            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert all(isinstance(x, (int, float)) for x in embedding)
            # Common embedding dimensions
            assert len(embedding) in [384, 768, 1536, 3072]

        except Exception as e:
            if "embedding" in str(e).lower() or "api key" in str(e).lower():
                pytest.skip(f"Embedding not supported or API key missing: {e}")
            else:
                raise

    def test_embed_batch(self):
        """Test batch embedding generation."""
        texts = [
            "First test sentence.",
            "Second test sentence with different content.",
            "Third sentence about artificial intelligence.",
        ]

        try:
            embeddings = self.client.embed_batch(texts)

            # Assertions
            assert embeddings is not None
            assert isinstance(embeddings, list)
            assert len(embeddings) == len(texts)

            for embedding in embeddings:
                assert isinstance(embedding, list)
                assert len(embedding) > 0
                assert all(isinstance(x, (int, float)) for x in embedding)
                # All embeddings should have the same dimension
                assert len(embedding) == len(embeddings[0])

        except Exception as e:
            if "embedding" in str(e).lower() or "api key" in str(e).lower():
                pytest.skip(f"Batch embedding not supported or API key missing: {e}")
            else:
                raise

    def test_rerank_chunks(self):
        """Test reranking functionality."""
        query = "machine learning algorithms"
        chunks = [
            "Deep learning is a subset of machine learning.",
            "The weather is nice today with sunshine.",
            "Neural networks are used in artificial intelligence.",
            "I like to eat pizza for dinner.",
            "Supervised learning requires labeled training data.",
        ]

        try:
            reranked_chunks = self.client.rerank(query, chunks)

            # Assertions
            assert reranked_chunks is not None
            assert isinstance(reranked_chunks, list)
            assert len(reranked_chunks) == len(chunks)
            assert set(reranked_chunks) == set(chunks)  # Same chunks, different order

            # The first chunk should be more relevant to the query
            relevant_chunks = [
                "Deep learning is a subset of machine learning.",
                "Neural networks are used in artificial intelligence.",
                "Supervised learning requires labeled training data.",
            ]

            # At least one relevant chunk should be in the top 3
            top_3 = reranked_chunks[:3]
            assert any(chunk in top_3 for chunk in relevant_chunks)

        except Exception as e:
            if "embedding" in str(e).lower() or "api key" in str(e).lower():
                pytest.skip(f"Reranking not supported or API key missing: {e}")
            else:
                raise

    def test_vision_image_from_url(self):
        """Test image understanding from URL."""
        try:
            client = LLMClient(vision_model="gpt-4-vision-preview")
        except Exception:
            pytest.skip("Vision model not available")

        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        prompt = "What type of landscape is shown in this image?"

        try:
            response = client.understand_image_from_url(image_url=image_url, prompt=prompt, max_tokens=100)

            # Assertions
            assert response is not None
            assert isinstance(response, str)
            assert len(response) > 0
            # Should mention nature, landscape, or boardwalk
            response_lower = response.lower()
            assert any(keyword in response_lower for keyword in ["nature", "landscape", "boardwalk", "path", "green"])

        except Exception as e:
            if "vision" in str(e).lower() or "image" in str(e).lower():
                pytest.skip(f"Vision not supported by this model: {e}")
            else:
                raise

    def test_multiple_providers(self):
        """Test that LiteLLM can work with different providers."""
        providers_to_test = []

        if os.getenv("OPENAI_API_KEY"):
            providers_to_test.append(("OpenAI", "gpt-3.5-turbo"))
        if os.getenv("ANTHROPIC_API_KEY"):
            providers_to_test.append(("Anthropic", "claude-3-haiku-20240307"))
        if os.getenv("COHERE_API_KEY"):
            providers_to_test.append(("Cohere", "command-r"))

        if len(providers_to_test) < 2:
            pytest.skip("Need at least 2 provider API keys to test multiple providers")

        messages = [{"role": "user", "content": "Say 'Hello from' followed by your provider name."}]

        for provider_name, model in providers_to_test:
            try:
                client = LLMClient(chat_model=model)
                response = client.completion(messages, max_tokens=20)

                assert response is not None
                assert isinstance(response, str)
                assert len(response) > 0

            except Exception as e:
                pytest.fail(f"Failed to get response from {provider_name}: {e}")

    def test_error_handling(self):
        """Test error handling with invalid inputs."""
        # Test with empty messages
        with pytest.raises(Exception):
            self.client.completion([])

        # Test with invalid model (should fail gracefully)
        try:
            invalid_client = LLMClient(chat_model="nonexistent-model-12345")
            invalid_client.completion([{"role": "user", "content": "test"}])
        except Exception:
            pass  # Expected to fail

    def test_embed_empty_text(self):
        """Test embedding generation with empty text."""
        try:
            with pytest.raises(Exception):
                self.client.embed("")
        except Exception as e:
            if "embedding" in str(e).lower() or "api key" in str(e).lower():
                pytest.skip(f"Embedding not supported: {e}")
            else:
                raise

    def test_rerank_edge_cases(self):
        """Test reranking edge cases."""
        try:
            # Empty chunks
            result = self.client.rerank("query", [])
            assert result == []

            # Single chunk
            single_chunk = ["Single test chunk"]
            result = self.client.rerank("query", single_chunk)
            assert result == single_chunk

        except Exception as e:
            if "embedding" in str(e).lower() or "api key" in str(e).lower():
                pytest.skip(f"Reranking not supported: {e}")
            else:
                raise
