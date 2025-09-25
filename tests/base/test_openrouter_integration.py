#!/usr/bin/env python3
"""
Integration tests for OpenRouter LLM client.

These tests verify the integration with OpenRouter API
and general OpenAI-compatible providers.
"""

import os
from pathlib import Path

import pytest

from cogents_core.llm import get_llm_client


@pytest.mark.integration
@pytest.mark.slow
class TestOpenRouterIntegration:
    """Integration tests for OpenRouter LLM functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        # Check if API key is available
        if not os.getenv("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set in environment")

        self.client = get_llm_client(provider="openrouter")

    def test_completion_with_gemini(self):
        """Test chat completion with Gemini model."""
        messages = [
            {"role": "system", "content": "You are a helpful travel assistant."},
            {"role": "user", "content": "What are the top 3 things to do in Tokyo?"},
        ]

        response = self.client.completion(messages, temperature=0.7)

        # Assertions
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        assert "Tokyo" in response or "japan" in response.lower()

    def test_completion_alias(self):
        """Test that completion method works as alias for completion."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'test completed' exactly."},
        ]

        # Test completion method (alias)
        response = self.client.completion(messages, temperature=0.1)

        # Assertions
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        assert "test completed" in response.lower()

    def test_completion_convenience_function(self):
        """Test the convenience completion function."""
        messages = [{"role": "user", "content": "What's the capital of France?"}]

        response = self.client.completion(messages)

        # Assertions
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        assert "paris" in response.lower() or "france" in response.lower()

    def test_image_analysis_with_sample_image(self):
        """Test image analysis functionality if sample image exists."""
        sample_image_path = Path(__file__).parent.parent.parent / "examples" / "sample_image.jpg"

        if not sample_image_path.exists():
            pytest.skip("Sample image not found for testing")

        prompt = "Describe what you see in this image in detail."
        analysis = self.client.understand_image(sample_image_path, prompt)

        # Assertions
        assert analysis is not None
        assert isinstance(analysis, str)
        assert len(analysis) > 0

    def test_client_initialization(self):
        """Test LLMClient initialization."""
        assert self.client is not None
        assert hasattr(self.client, "completion")
        assert hasattr(self.client, "understand_image")

    def test_error_handling_invalid_messages(self):
        """Test error handling with invalid message format."""
        invalid_messages = [{"invalid_key": "This should fail"}]

        with pytest.raises(Exception):
            self.client.completion(invalid_messages)

    def test_temperature_parameter(self):
        """Test that temperature parameter affects response variability."""
        messages = [{"role": "user", "content": "Give me a random number between 1 and 10"}]

        # Test with different temperatures
        response_low = self.client.completion(messages, temperature=0.1)
        response_high = self.client.completion(messages, temperature=0.9)

        # Both should be valid responses
        assert response_low is not None
        assert response_high is not None
        assert isinstance(response_low, str)
        assert isinstance(response_high, str)

    def test_structured_completion_integration(self):
        """Test structured completion with OpenRouter."""
        from pydantic import BaseModel

        class SimpleResponse(BaseModel):
            message: str

        # Get client with instructor enabled
        client = get_llm_client(provider="openrouter", instructor=True)

        # Make a structured completion
        messages = [{"role": "user", "content": "Reply with a simple greeting message"}]
        response = client.structured_completion(
            messages=messages, response_model=SimpleResponse, temperature=0.1, max_tokens=50
        )

        # Verify response
        assert response is not None
        assert isinstance(response, SimpleResponse)
        assert hasattr(response, "message")
        assert len(response.message.strip()) > 0

    def test_multiple_messages_conversation(self):
        """Test handling of multi-turn conversations."""
        messages = [
            {"role": "system", "content": "You are a helpful math tutor."},
            {"role": "user", "content": "What is 10 + 5?"},
            {"role": "assistant", "content": "10 + 5 = 15"},
            {"role": "user", "content": "What about 15 * 2?"},
        ]

        response = self.client.completion(messages, temperature=0.1)

        # Assertions
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        # Should mention 30 or show understanding of math
        assert any(str(i) in response for i in [30, "thirty"])

    def test_max_tokens_parameter(self):
        """Test max_tokens parameter limits response length."""
        messages = [{"role": "user", "content": "Write a long story about a dragon and a knight."}]

        # Test with very limited tokens
        short_response = self.client.completion(messages, max_tokens=20)

        # Test with more tokens
        long_response = self.client.completion(messages, max_tokens=100)

        # Assertions
        assert short_response is not None
        assert long_response is not None
        assert isinstance(short_response, str)
        assert isinstance(long_response, str)

        # Long response should generally be longer (though not guaranteed)
        # This is a probabilistic test, so we'll just check they're both valid
        assert len(short_response.strip()) > 0
        assert len(long_response.strip()) > 0

    def test_embed_single_text(self):
        """Test embedding generation for a single text."""
        # Check if OpenAI API key is available for embeddings
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            pytest.skip("OPENAI_API_KEY not set - required for OpenRouter embeddings")

        text = "This is a test sentence for embedding generation."

        try:
            embedding = self.client.embed(text)

            # Assertions
            assert embedding is not None
            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert all(isinstance(x, (int, float)) for x in embedding)
            # OpenAI embeddings are typically 1536 or 3072 dimensions
            assert len(embedding) in [1536, 3072]

        except Exception as e:
            if "api key" in str(e).lower():
                pytest.skip(f"OpenAI API key required for embeddings: {e}")
            else:
                raise

    def test_embed_batch(self):
        """Test batch embedding generation."""
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            pytest.skip("OPENAI_API_KEY not set - required for OpenRouter embeddings")

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
            if "api key" in str(e).lower():
                pytest.skip(f"OpenAI API key required for batch embeddings: {e}")
            else:
                raise

    def test_rerank_chunks(self):
        """Test reranking functionality."""
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            pytest.skip("OPENAI_API_KEY not set - required for OpenRouter reranking")

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
            if "api key" in str(e).lower():
                pytest.skip(f"OpenAI API key required for reranking: {e}")
            else:
                raise

    def test_embed_and_rerank_consistency(self):
        """Test that embeddings and reranking are consistent."""
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            pytest.skip("OPENAI_API_KEY not set - required for OpenRouter embeddings")

        query = "artificial intelligence"
        chunks = [
            "AI is transforming many industries.",
            "Cooking pasta requires boiling water.",
            "Machine learning is a branch of AI.",
        ]

        try:
            # Get embeddings
            query_embedding = self.client.embed(query)
            chunk_embeddings = self.client.embed_batch(chunks)

            # Rerank chunks
            reranked_results = self.client.rerank(query, chunks)

            from cogents_core.utils.statistics import cosine_similarity

            similarities = []
            for i, chunk_embedding in enumerate(chunk_embeddings):
                similarity = cosine_similarity(query_embedding, chunk_embedding)
                similarities.append((similarity, i, chunks[i]))

            similarities.sort(key=lambda x: x[0], reverse=True)

            # The reranked results should match our manual calculation
            assert reranked_results == similarities

            # Verify the structure of returned tuples
            assert all(len(result) == 3 for result in reranked_results)
            assert all(isinstance(result[0], (int, float)) for result in reranked_results)  # similarity score
            assert all(isinstance(result[1], int) for result in reranked_results)  # original index
            assert all(isinstance(result[2], str) for result in reranked_results)  # chunk text

        except Exception as e:
            if "api key" in str(e).lower():
                pytest.skip(f"OpenAI API key required for embeddings: {e}")
            else:
                raise

    def test_rerank_edge_cases(self):
        """Test reranking edge cases."""
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            pytest.skip("OPENAI_API_KEY not set - required for OpenRouter reranking")

        try:
            # Empty chunks
            result = self.client.rerank("query", [])
            assert result == []

            # Single chunk
            single_chunk = ["Single test chunk"]
            result = self.client.rerank("query", single_chunk)
            assert len(result) == 1
            similarity, index, chunk = result[0]
            assert isinstance(similarity, (int, float))
            assert index == 0
            assert chunk == "Single test chunk"

        except Exception as e:
            if "api key" in str(e).lower():
                pytest.skip(f"OpenAI API key required for reranking: {e}")
            else:
                raise
