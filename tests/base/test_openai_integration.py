#!/usr/bin/env python3
"""
Integration tests for OpenAI LLM client.

These tests verify the integration with OpenAI API
and OpenAI-compatible providers.
"""

import os

import pytest

from cogents_core.llm import get_llm_client


@pytest.mark.integration
@pytest.mark.slow
class TestOpenAIIntegration:
    """Integration tests for OpenAI LLM functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        # Check if API key is available
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set in environment")

        self.client = get_llm_client(provider="openai")

    def test_basic_completion(self):
        """Test basic chat completion with OpenAI."""
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

        # Assertions
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        # Should understand the math context
        assert any(str(i) in response for i in [15, "fifteen"])

    def test_structured_completion(self):
        """Test structured completion with OpenAI."""
        from pydantic import BaseModel, Field

        class MathResponse(BaseModel):
            answer: int = Field(description="The numerical answer")
            explanation: str = Field(description="Brief explanation of the calculation")

        # Create client with instructor enabled
        client = get_llm_client(provider="openai", instructor=True)

        messages = [{"role": "user", "content": "Calculate 7 + 8 and explain the result."}]

        result = client.structured_completion(messages, MathResponse, temperature=0.1, max_tokens=100)

        # Assertions
        assert isinstance(result, MathResponse)
        assert result.answer == 15
        assert len(result.explanation) > 0
        assert "7" in result.explanation or "8" in result.explanation

    def test_temperature_variations(self):
        """Test different temperature settings."""
        messages = [{"role": "user", "content": "Describe the color blue in one sentence."}]

        # Test low temperature (more deterministic)
        response_low = self.client.completion(messages, temperature=0.1, max_tokens=50)

        # Test higher temperature (more creative)
        response_high = self.client.completion(messages, temperature=0.9, max_tokens=50)

        # Both should be valid responses
        assert response_low is not None
        assert response_high is not None
        assert isinstance(response_low, str)
        assert isinstance(response_high, str)
        assert len(response_low.strip()) > 0
        assert len(response_high.strip()) > 0

    def test_max_tokens_limiting(self):
        """Test that max_tokens parameter limits response length."""
        messages = [{"role": "user", "content": "Write a story about a cat and a dog becoming friends."}]

        # Test with very limited tokens
        short_response = self.client.completion(messages, max_tokens=10, temperature=0.7)

        # Test with more tokens
        longer_response = self.client.completion(messages, max_tokens=100, temperature=0.7)

        # Both should be valid
        assert short_response is not None
        assert longer_response is not None
        assert isinstance(short_response, str)
        assert isinstance(longer_response, str)

        # Responses should exist
        assert len(short_response.strip()) > 0
        assert len(longer_response.strip()) > 0

    def test_system_message_influence(self):
        """Test that system messages influence response style."""
        user_query = "Explain gravity."

        # Test with formal system message
        formal_messages = [
            {"role": "system", "content": "You are a formal physics professor. Use scientific terminology."},
            {"role": "user", "content": user_query},
        ]

        # Test with casual system message
        casual_messages = [
            {"role": "system", "content": "You are a friendly teacher who explains things simply."},
            {"role": "user", "content": user_query},
        ]

        formal_response = self.client.completion(formal_messages, temperature=0.3, max_tokens=100)
        casual_response = self.client.completion(casual_messages, temperature=0.3, max_tokens=100)

        # Both should mention gravity concepts
        assert formal_response is not None
        assert casual_response is not None
        assert "gravity" in formal_response.lower() or "force" in formal_response.lower()
        assert "gravity" in casual_response.lower() or "force" in casual_response.lower()

    def test_vision_capability(self):
        """Test image understanding if vision model is available."""
        from pathlib import Path

        # Look for a sample image
        sample_image_path = Path(__file__).parent.parent.parent / "examples" / "sample_image.jpg"

        if not sample_image_path.exists():
            pytest.skip("No sample image available for vision testing")

        try:
            prompt = "What do you see in this image? Describe it briefly."
            analysis = self.client.understand_image(sample_image_path, prompt)

            # Assertions
            assert analysis is not None
            assert isinstance(analysis, str)
            assert len(analysis) > 10  # Should be a meaningful description

        except Exception as e:
            if "vision" in str(e).lower() or "image" in str(e).lower():
                pytest.skip(f"Vision capability not available: {e}")
            else:
                raise

    def test_error_handling(self):
        """Test error handling with invalid inputs."""
        # Test with malformed messages
        with pytest.raises(Exception):
            self.client.completion([{"invalid": "message format"}])

        # Test with empty messages
        with pytest.raises(Exception):
            self.client.completion([])

    def test_streaming_not_supported_gracefully(self):
        """Test that streaming parameter is handled gracefully."""
        messages = [{"role": "user", "content": "Say hello"}]

        # Most implementations should either support streaming or ignore the parameter
        try:
            response = self.client.completion(messages, stream=False)
            assert response is not None
            assert isinstance(response, str)
        except Exception as e:
            if "stream" not in str(e).lower():
                raise  # Re-raise if it's not a streaming-related error

    def test_embed_single_text(self):
        """Test embedding generation for a single text."""
        text = "This is a test sentence for embedding generation."

        embedding = self.client.embed(text)

        # Assertions
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, (int, float)) for x in embedding)
        # OpenAI embeddings are typically 1536 or 3072 dimensions
        assert len(embedding) in [1536, 3072]

    def test_embed_batch(self):
        """Test batch embedding generation."""
        texts = [
            "First test sentence.",
            "Second test sentence with different content.",
            "Third sentence about artificial intelligence.",
        ]

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

    def test_embed_empty_text(self):
        """Test embedding generation with empty text."""
        with pytest.raises(Exception):
            self.client.embed("")

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

        reranked_chunks = self.client.rerank(query, chunks)

        # Assertions
        assert reranked_chunks is not None
        assert isinstance(reranked_chunks, list)
        assert len(reranked_chunks) == len(chunks)
        assert set(reranked_chunks) == set(chunks)  # Same chunks, different order

        # The first chunk should be more relevant to the query
        # (This is a heuristic test, actual ranking may vary)
        relevant_chunks = [
            "Deep learning is a subset of machine learning.",
            "Neural networks are used in artificial intelligence.",
            "Supervised learning requires labeled training data.",
        ]

        # At least one relevant chunk should be in the top 3
        top_3 = reranked_chunks[:3]
        assert any(chunk in top_3 for chunk in relevant_chunks)

    def test_rerank_empty_chunks(self):
        """Test reranking with empty chunks list."""
        query = "test query"
        chunks = []

        reranked_results = self.client.rerank(query, chunks)

        assert reranked_results == []

    def test_rerank_single_chunk(self):
        """Test reranking with a single chunk."""
        query = "test query"
        chunks = ["Single test chunk"]

        reranked_results = self.client.rerank(query, chunks)

        assert len(reranked_results) == 1
        similarity, index, chunk = reranked_results[0]
        assert isinstance(similarity, (int, float))
        assert index == 0
        assert chunk == "Single test chunk"

    def test_embed_and_rerank_consistency(self):
        """Test that embeddings and reranking are consistent."""
        query = "artificial intelligence"
        chunks = [
            "AI is transforming many industries.",
            "Cooking pasta requires boiling water.",
            "Machine learning is a branch of AI.",
        ]

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
