"""
Unit tests for self-assessment routing strategy.

Tests cover SelfAssessmentStrategy implementation.
"""

from unittest.mock import Mock, patch

import pytest

from cogents_core.routing.strategies.self_assessment import SelfAssessmentStrategy
from cogents_core.routing.types import ModelTier, RoutingResult


class TestSelfAssessmentStrategy:
    """Test cases for SelfAssessmentStrategy."""

    @pytest.fixture
    def mock_llm_client(self):
        """Provide a mock LLM client for testing."""
        mock_client = Mock()
        return mock_client

    @pytest.fixture
    def default_config(self):
        """Provide default configuration for testing."""
        return {"temperature": 0.1, "max_tokens": 5, "lite_threshold": 1, "fast_threshold": 3}

    @pytest.fixture
    def strategy(self, mock_llm_client, default_config):
        """Provide a configured SelfAssessmentStrategy for testing."""
        return SelfAssessmentStrategy(lite_client=mock_llm_client, config=default_config)

    def test_init_requires_lite_client(self):
        """Test that SelfAssessmentStrategy requires a lite client."""
        with pytest.raises(ValueError, match="SelfAssessmentStrategy requires a lite_client"):
            SelfAssessmentStrategy(lite_client=None)

    def test_init_with_default_config(self, mock_llm_client):
        """Test initialization with default configuration."""
        strategy = SelfAssessmentStrategy(lite_client=mock_llm_client)

        assert strategy.lite_client == mock_llm_client
        assert strategy.temperature == 0.1
        assert strategy.max_tokens == 5
        assert strategy.lite_threshold == 1
        assert strategy.fast_threshold == 3

    def test_init_with_custom_config(self, mock_llm_client):
        """Test initialization with custom configuration."""
        custom_config = {"temperature": 0.3, "max_tokens": 10, "lite_threshold": 2, "fast_threshold": 4}

        strategy = SelfAssessmentStrategy(lite_client=mock_llm_client, config=custom_config)

        assert strategy.temperature == 0.3
        assert strategy.max_tokens == 10
        assert strategy.lite_threshold == 2
        assert strategy.fast_threshold == 4

    def test_get_strategy_name(self, strategy):
        """Test strategy name."""
        assert strategy.get_strategy_name() == "self_assessment"

    def test_create_assessment_prompt(self, strategy):
        """Test assessment prompt creation."""
        query = "What is the capital of France?"
        prompt = strategy._create_assessment_prompt(query)

        assert "request classifier" in prompt.lower()
        assert "1 = simple fact" in prompt
        assert "4-5 = deep reasoning" in prompt
        assert query in prompt
        assert "Output ONLY a number from 1 to 5" in prompt

    def test_parse_assessment_response_valid_scores(self, strategy):
        """Test parsing valid assessment responses."""
        # Test each valid score
        for expected_score in range(1, 6):
            response = str(expected_score)
            score, confidence = strategy._parse_assessment_response(response)
            assert score == expected_score
            assert confidence == 0.9  # High confidence for clean response

    def test_parse_assessment_response_noisy_but_valid(self, strategy):
        """Test parsing noisy but valid responses."""
        test_cases = [
            ("3", 3, 0.9),  # Clean response gets high confidence
            ("Score: 4", 4, 0.7),  # Extra text
            ("2.", 2, 0.7),  # With punctuation
            ("The answer is 5", 5, 0.7),  # Verbose response
            ("  3  ", 3, 0.9),  # Whitespace gets stripped so it's clean
        ]

        for response, expected_score, expected_confidence in test_cases:
            score, confidence = strategy._parse_assessment_response(response)
            assert score == expected_score
            assert confidence == expected_confidence

    def test_parse_assessment_response_invalid(self, strategy):
        """Test parsing invalid responses."""
        test_cases = [
            "6",  # Out of range
            "0",  # Out of range
            "abc",  # No digits
            "",  # Empty
            "no number here",  # No valid digits
        ]

        for response in test_cases:
            with patch("cogents_core.utils.logging.get_logger") as mock_logger:
                score, confidence = strategy._parse_assessment_response(response)
                assert score == 3  # Default fallback
                assert confidence == 0.3  # Low confidence

    def test_score_to_tier_mapping(self, strategy):
        """Test mapping of scores to tiers."""
        # Test with default thresholds (lite_threshold=1, fast_threshold=3)
        assert strategy._score_to_tier(1) == ModelTier.LITE
        assert strategy._score_to_tier(2) == ModelTier.FAST
        assert strategy._score_to_tier(3) == ModelTier.FAST
        assert strategy._score_to_tier(4) == ModelTier.POWER
        assert strategy._score_to_tier(5) == ModelTier.POWER

    def test_score_to_tier_custom_thresholds(self, mock_llm_client):
        """Test tier mapping with custom thresholds."""
        config = {"lite_threshold": 2, "fast_threshold": 4}
        strategy = SelfAssessmentStrategy(lite_client=mock_llm_client, config=config)

        assert strategy._score_to_tier(1) == ModelTier.LITE
        assert strategy._score_to_tier(2) == ModelTier.LITE
        assert strategy._score_to_tier(3) == ModelTier.FAST
        assert strategy._score_to_tier(4) == ModelTier.FAST
        assert strategy._score_to_tier(5) == ModelTier.POWER

    def test_route_success_lite(self, strategy):
        """Test successful routing to LITE tier."""
        strategy.lite_client.completion.return_value = "1"

        result = strategy.route("What is 2+2?")

        assert isinstance(result, RoutingResult)
        assert result.tier == ModelTier.LITE
        assert result.confidence == 0.9
        assert result.complexity_score.total == 1 / 5  # Normalized
        assert result.strategy == "self_assessment"
        assert "raw_assessment" in result.metadata

    def test_route_success_fast(self, strategy):
        """Test successful routing to FAST tier."""
        strategy.lite_client.completion.return_value = "3"

        result = strategy.route("Explain the causes of World War I.")

        assert result.tier == ModelTier.FAST
        assert result.confidence == 0.9
        assert result.complexity_score.total == 3 / 5  # Normalized

    def test_route_success_power(self, strategy):
        """Test successful routing to POWER tier."""
        strategy.lite_client.completion.return_value = "5"

        result = strategy.route("Write a detailed analysis comparing quantum mechanics and general relativity.")

        assert result.tier == ModelTier.POWER
        assert result.confidence == 0.9
        assert result.complexity_score.total == 5 / 5  # Normalized

    def test_route_llm_client_called_correctly(self, strategy):
        """Test that LLM client is called with correct parameters."""
        strategy.lite_client.completion.return_value = "2"
        query = "Test query"

        strategy.route(query)

        # Verify client was called with correct parameters
        strategy.lite_client.completion.assert_called_once()
        call_args = strategy.lite_client.completion.call_args

        assert call_args[1]["temperature"] == 0.1
        assert call_args[1]["max_tokens"] == 5

        # Check that the prompt contains the query
        messages = call_args[1]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert query in messages[0]["content"]

    @patch("cogents_core.utils.logging.get_logger")
    def test_route_exception_handling(self, mock_logger, strategy):
        """Test route method exception handling."""
        # Make the LLM client raise an exception
        strategy.lite_client.completion.side_effect = Exception("LLM error")

        result = strategy.route("Test query")

        # Should fallback to FAST tier with low confidence
        assert result.tier == ModelTier.FAST
        assert result.confidence == 0.0
        assert result.complexity_score.total == 0.5
        assert result.strategy == "self_assessment"
        assert "error" in result.metadata
        assert result.metadata["fallback"] is True

    def test_route_empty_query(self, strategy):
        """Test routing with empty query."""
        strategy.lite_client.completion.return_value = "1"

        result = strategy.route("")

        # Should still call the LLM and return a result
        assert isinstance(result, RoutingResult)
        strategy.lite_client.completion.assert_called_once()

    def test_route_whitespace_query(self, strategy):
        """Test routing with whitespace-only query."""
        strategy.lite_client.completion.return_value = "1"

        result = strategy.route("   \n\t   ")

        assert isinstance(result, RoutingResult)
        strategy.lite_client.completion.assert_called_once()

    def test_complexity_score_metadata(self, strategy):
        """Test that complexity score includes metadata."""
        strategy.lite_client.completion.return_value = "4"

        result = strategy.route("Complex reasoning task")

        assert result.complexity_score.metadata is not None
        assert "raw_score" in result.complexity_score.metadata
        assert result.complexity_score.metadata["raw_score"] == 4
        assert "raw_response" in result.complexity_score.metadata

    def test_multiple_queries_independent(self, strategy):
        """Test that multiple queries are processed independently."""
        # First query
        strategy.lite_client.completion.return_value = "1"
        result1 = strategy.route("Simple query")

        # Second query with different response
        strategy.lite_client.completion.return_value = "5"
        result2 = strategy.route("Complex query")

        # Results should be independent
        assert result1.tier == ModelTier.LITE
        assert result2.tier == ModelTier.POWER
        assert result1.complexity_score.total != result2.complexity_score.total
