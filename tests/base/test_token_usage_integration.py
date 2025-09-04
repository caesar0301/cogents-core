#!/usr/bin/env python3
"""
Integration tests for token usage tracking system.

These tests verify token usage tracking across different LLM providers
and integration with various components like callbacks and agents.
"""

import os

import pytest

from cogents_core.llm import get_llm_client
from cogents_core.tracing import TokenUsage, TokenUsageCallback, get_token_tracker


@pytest.mark.integration
@pytest.mark.slow
class TestTokenUsageIntegration:
    """Integration tests for token usage tracking system."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        # Reset token tracker before each test
        get_token_tracker().reset()

        # Check if API key is available for real API tests
        self.has_api_key = bool(os.getenv("OPENROUTER_API_KEY"))
        if self.has_api_key:
            self.client = get_llm_client(provider="openrouter")

    def test_token_tracker_basic_functionality(self):
        """Test basic token tracker functionality."""
        tracker = get_token_tracker()

        # Should start with zero usage
        assert tracker.get_total_tokens() == 0
        assert tracker.call_count == 0

        # Record some usage
        usage = TokenUsage(
            prompt_tokens=50,
            completion_tokens=30,
            total_tokens=80,
            model_name="test-model",
            timestamp="2024-01-01T00:00:00",
            call_type="completion",
        )
        tracker.record_usage(usage)

        # Verify tracking
        assert tracker.get_total_tokens() == 80
        assert tracker.call_count == 1
        assert tracker.total_prompt_tokens == 50
        assert tracker.total_completion_tokens == 30

        # Test stats
        stats = tracker.get_stats()
        assert stats["total_calls"] == 1
        assert stats["total_tokens"] == 80
        assert len(stats["usage_history"]) == 1

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set")
    def test_openrouter_client_token_tracking(self):
        """Test that OpenRouter client properly tracks token usage."""
        tracker = get_token_tracker()
        initial_tokens = tracker.get_total_tokens()

        # Make a simple completion
        messages = [{"role": "user", "content": "Say 'Hello World' and nothing else."}]
        response = self.client.completion(messages, temperature=0.1, max_tokens=10)

        # Verify response
        assert response is not None
        assert isinstance(response, str)
        assert len(response.strip()) > 0

        # Verify token usage was tracked
        final_tokens = tracker.get_total_tokens()
        assert final_tokens > initial_tokens, "Token usage should have increased"

        # Check that at least one call was recorded
        stats = tracker.get_stats()
        assert stats["total_calls"] > 0
        assert len(stats["usage_history"]) > 0

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set")
    def test_structured_completion_token_tracking(self):
        """Test token tracking for structured completions."""
        from pydantic import BaseModel

        class SimpleResponse(BaseModel):
            message: str

        # Get client with instructor enabled
        client = get_llm_client(provider="openrouter", instructor=True)
        tracker = get_token_tracker()
        initial_tokens = tracker.get_total_tokens()

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

        # Verify token usage was tracked
        final_tokens = tracker.get_total_tokens()
        assert final_tokens > initial_tokens, "Token usage should have increased"

        # Check for structured completion in history
        stats = tracker.get_stats()
        structured_calls = [h for h in stats["usage_history"] if h.get("call_type") == "structured"]
        assert len(structured_calls) > 0, "Should have at least one structured completion call"

    def test_token_callback_integration(self):
        """Test TokenUsageCallback integration with token tracker."""
        callback = TokenUsageCallback(model_name="test-model", verbose=False)
        tracker = get_token_tracker()

        # Simulate some direct token tracker usage (from custom clients)
        usage = TokenUsage(
            prompt_tokens=40,
            completion_tokens=25,
            total_tokens=65,
            model_name="custom-client-model",
            timestamp="2024-01-01T00:00:00",
            call_type="completion",
        )
        tracker.record_usage(usage)

        # Simulate some callback usage (from LangChain LLMs)
        callback.total_prompt_tokens = 30
        callback.total_completion_tokens = 20
        callback.llm_calls = 1

        # Test combined session summary
        summary = callback.get_session_summary()

        # Should combine both sources
        assert summary["total_llm_calls"] == 2  # 1 from tracker + 1 from callback
        assert summary["total_prompt_tokens"] == 70  # 40 + 30
        assert summary["total_completion_tokens"] == 45  # 25 + 20
        assert summary["total_tokens"] == 115  # 65 + 50

        # Should have separate breakdowns
        assert summary["callback_stats"]["llm_calls"] == 1
        assert summary["tracker_stats"]["total_calls"] == 1

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set")
    def test_multiple_llm_calls_token_aggregation(self):
        """Test token tracking with multiple sequential LLM calls."""
        tracker = get_token_tracker()
        callback = TokenUsageCallback(verbose=False)

        initial_tokens = tracker.get_total_tokens()

        # Make multiple calls to simulate a conversation
        messages_list = [
            [{"role": "user", "content": "What is 5 + 3?"}],
            [{"role": "user", "content": "What is 10 - 2?"}],
            [{"role": "user", "content": "What is 4 * 2?"}],
        ]

        responses = []
        for messages in messages_list:
            response = self.client.completion(messages, temperature=0.1, max_tokens=20)
            responses.append(response)

        # Verify all responses
        for response in responses:
            assert response is not None
            assert isinstance(response, str)
            assert len(response.strip()) > 0

        # Verify token tracking aggregation
        final_tokens = tracker.get_total_tokens()
        assert final_tokens > initial_tokens, "Total token usage should have increased"

        # Should have tracked multiple calls
        stats = tracker.get_stats()
        assert stats["total_calls"] >= 3, "Should have tracked at least 3 LLM calls"

        # Test session summary includes all usage
        session_summary = callback.get_session_summary()
        assert session_summary["total_tokens"] >= (final_tokens - initial_tokens)

        print(f"Completed {len(messages_list)} LLM calls with total {final_tokens - initial_tokens} tokens")

    def test_token_callback_reset_functionality(self):
        """Test that callback reset properly clears both callback and tracker."""
        callback = TokenUsageCallback(verbose=False)
        tracker = get_token_tracker()

        # Add some data to both
        callback.total_prompt_tokens = 50
        callback.total_completion_tokens = 30
        callback.llm_calls = 2

        usage = TokenUsage(
            prompt_tokens=25,
            completion_tokens=15,
            total_tokens=40,
            model_name="test-model",
            timestamp="2024-01-01T00:00:00",
            call_type="completion",
        )
        tracker.record_usage(usage)

        # Verify data exists
        assert callback.total_tokens() == 80
        assert tracker.get_total_tokens() == 40

        # Reset
        callback.reset_session()

        # Verify both are reset
        assert callback.total_tokens() == 0
        assert tracker.get_total_tokens() == 0
        assert callback.llm_calls == 0
        assert tracker.call_count == 0

    def test_token_estimation_fallback(self):
        """Test token estimation when actual usage is not available."""
        from cogents_core.tracing import estimate_token_usage

        prompt_text = "This is a test prompt with some words"
        completion_text = "This is a test response with different words"

        usage = estimate_token_usage(prompt_text, completion_text, "test-model", "completion")

        # Should have reasonable estimates
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens > 0
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
        assert usage.estimated == True
        assert usage.model_name == "test-model"
        assert usage.call_type == "completion"

        # Should be roughly proportional to text length
        assert usage.prompt_tokens > 5  # Should be more than 5 tokens for that text
        assert usage.completion_tokens > 5

    def test_llamacpp_token_tracking(self):
        """Test token tracking works with llamacpp provider (uses estimation)."""
        # Clear any existing model path to test default model download
        old_model_path = os.environ.get("LLAMACPP_MODEL_PATH")
        if old_model_path:
            del os.environ["LLAMACPP_MODEL_PATH"]

        try:
            # This will use the default model with auto-download
            client = get_llm_client(provider="llamacpp")
            tracker = get_token_tracker()
            initial_tokens = tracker.get_total_tokens()

            messages = [
                {"role": "user", "content": "Say hello"},
            ]

            response = client.completion(messages, temperature=0.1, max_tokens=10)

            # Verify response
            assert response is not None
            assert isinstance(response, str)
            assert len(response.strip()) > 0

            # Should have some token usage recorded (estimated)
            assert tracker.get_total_tokens() > initial_tokens
            assert tracker.call_count > 0

            # Get latest usage
            latest_usage = tracker.usage_history[-1]
            assert latest_usage.estimated == True  # Should be estimated for llamacpp
            assert latest_usage.call_type == "completion"

        finally:
            # Restore original environment variable if it existed
            if old_model_path:
                os.environ["LLAMACPP_MODEL_PATH"] = old_model_path

    def test_cross_provider_token_aggregation(self):
        """Test that token usage is properly aggregated across different providers."""
        tracker = get_token_tracker()

        # Simulate usage from different providers
        openrouter_usage = TokenUsage(
            prompt_tokens=30,
            completion_tokens=20,
            total_tokens=50,
            model_name="openrouter/model",
            timestamp="2024-01-01T00:00:00",
            call_type="completion",
        )

        llamacpp_usage = TokenUsage(
            prompt_tokens=40,
            completion_tokens=25,
            total_tokens=65,
            model_name="llamacpp/model",
            timestamp="2024-01-01T00:01:00",
            call_type="completion",
            estimated=True,
        )

        tracker.record_usage(openrouter_usage)
        tracker.record_usage(llamacpp_usage)

        # Check aggregation
        assert tracker.get_total_tokens() == 115  # 50 + 65
        assert tracker.call_count == 2
        assert tracker.total_prompt_tokens == 70  # 30 + 40
        assert tracker.total_completion_tokens == 45  # 20 + 25

        # Check stats include both providers
        stats = tracker.get_stats()
        assert stats["total_calls"] == 2
        assert stats["total_tokens"] == 115
        assert len(stats["usage_history"]) == 2

        # Should have one estimated and one actual
        estimated_calls = [h for h in stats["usage_history"] if h.get("estimated")]
        actual_calls = [h for h in stats["usage_history"] if not h.get("estimated")]
        assert len(estimated_calls) == 1
        assert len(actual_calls) == 1
