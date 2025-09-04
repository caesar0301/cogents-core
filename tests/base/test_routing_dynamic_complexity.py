"""
Unit tests for dynamic complexity routing strategy.

Tests cover DynamicComplexityStrategy implementation including all complexity scoring components.
"""

from unittest.mock import Mock, patch

import pytest

from cogents_core.routing.strategies.dynamic_complexity import DynamicComplexityStrategy
from cogents_core.routing.types import ModelTier, RoutingResult


class TestDynamicComplexityStrategy:
    """Test cases for DynamicComplexityStrategy."""

    @pytest.fixture
    def mock_llm_client(self):
        """Provide a mock LLM client for testing."""
        mock_client = Mock()
        return mock_client

    @pytest.fixture
    def default_config(self):
        """Provide default configuration for testing."""
        return {
            "alpha": 0.4,
            "beta": 0.4,
            "gamma": 0.2,
            "lite_threshold": 0.3,
            "fast_threshold": 0.65,
            "temperature": 0.1,
            "reasoning_max_tokens": 3,
            "uncertainty_max_tokens": 64,
        }

    @pytest.fixture
    def strategy_with_client(self, mock_llm_client, default_config):
        """Provide a configured DynamicComplexityStrategy with LLM client."""
        return DynamicComplexityStrategy(lite_client=mock_llm_client, config=default_config)

    @pytest.fixture
    def strategy_no_client(self, default_config):
        """Provide a configured DynamicComplexityStrategy without LLM client."""
        return DynamicComplexityStrategy(lite_client=None, config=default_config)

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        strategy = DynamicComplexityStrategy()

        assert strategy.alpha == 0.4
        assert strategy.beta == 0.4
        assert strategy.gamma == 0.2
        assert strategy.lite_threshold == 0.3
        assert strategy.fast_threshold == 0.65

    def test_init_custom_config(self, mock_llm_client):
        """Test initialization with custom configuration."""
        config = {"alpha": 0.5, "beta": 0.3, "gamma": 0.2, "lite_threshold": 0.2, "fast_threshold": 0.7}

        strategy = DynamicComplexityStrategy(lite_client=mock_llm_client, config=config)

        assert strategy.alpha == 0.5
        assert strategy.beta == 0.3
        assert strategy.gamma == 0.2
        assert strategy.lite_threshold == 0.2
        assert strategy.fast_threshold == 0.7

    def test_weight_normalization_warning(self, mock_llm_client):
        """Test that weights are normalized if they don't sum to 1.0."""
        config = {"alpha": 0.6, "beta": 0.4, "gamma": 0.2}  # These sum to 1.2, should be normalized

        with patch("cogents_core.logging_config.get_logger") as mock_logger:
            strategy = DynamicComplexityStrategy(lite_client=mock_llm_client, config=config)

            # Weights should be normalized
            assert abs(strategy.alpha + strategy.beta + strategy.gamma - 1.0) < 0.001
            assert strategy.alpha == 0.6 / 1.2  # 0.5
            assert strategy.beta == 0.4 / 1.2  # ~0.333
            assert strategy.gamma == 0.2 / 1.2  # ~0.167

    def test_get_strategy_name(self, strategy_with_client):
        """Test strategy name."""
        assert strategy_with_client.get_strategy_name() == "dynamic_complexity"

    # Linguistic complexity tests
    def test_calculate_linguistic_score_simple_query(self, strategy_with_client):
        """Test linguistic score calculation for simple query."""
        query = "What is Python?"
        score = strategy_with_client._calculate_linguistic_score(query)

        assert 0.0 <= score <= 1.0
        assert score < 0.3  # Should be low for simple query

    def test_calculate_linguistic_score_complex_query(self, strategy_with_client):
        """Test linguistic score calculation for complex query."""
        query = """
        Analyze the comprehensive implications of quantum entanglement in distributed
        computing systems, considering the theoretical frameworks, practical limitations,
        and potential applications in cryptographic protocols, while also examining
        the intersection with classical computational paradigms and emerging technologies.
        """
        score = strategy_with_client._calculate_linguistic_score(query)

        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be high for complex query

    def test_calculate_linguistic_score_empty_query(self, strategy_with_client):
        """Test linguistic score for empty query."""
        score = strategy_with_client._calculate_linguistic_score("")

        assert score == 0.0

    def test_calculate_linguistic_score_error_handling(self, strategy_with_client):
        """Test linguistic score error handling."""
        with patch("re.findall", side_effect=Exception("Regex error")):
            with patch("cogents_core.logging_config.get_logger"):
                score = strategy_with_client._calculate_linguistic_score("test")
                assert score == 0.5  # Fallback value

    # Reasoning complexity tests
    def test_calculate_reasoning_score_with_client(self, strategy_with_client):
        """Test reasoning score calculation with LLM client."""
        strategy_with_client.lite_client.completion.return_value = "2"

        score = strategy_with_client._calculate_reasoning_score("Analyze this problem")

        assert score == 2 / 2  # Normalized (2/2 = 1.0)
        assert 0.0 <= score <= 1.0

    def test_calculate_reasoning_score_invalid_response(self, strategy_with_client):
        """Test reasoning score with invalid LLM response."""
        strategy_with_client.lite_client.completion.return_value = "invalid"

        with patch("cogents_core.logging_config.get_logger"):
            score = strategy_with_client._calculate_reasoning_score("test query")

        # Should fall back to keyword-based scoring
        assert 0.0 <= score <= 1.0

    def test_calculate_reasoning_score_no_client(self, strategy_no_client):
        """Test reasoning score calculation without LLM client."""
        query = "analyze and compare these complex systems"
        score = strategy_no_client._calculate_reasoning_score(query)

        assert 0.0 <= score <= 1.0
        # Should use fallback method

    def test_fallback_reasoning_score(self, strategy_with_client):
        """Test fallback reasoning score calculation."""
        # Query with reasoning keywords
        query = "analyze and compare the effects of this strategy"
        score = strategy_with_client._fallback_reasoning_score(query)

        assert score > 0  # Should detect reasoning keywords
        assert score <= 1.0

    def test_fallback_reasoning_score_no_keywords(self, strategy_with_client):
        """Test fallback reasoning score with no reasoning keywords."""
        query = "the cat sat on the mat"
        score = strategy_with_client._fallback_reasoning_score(query)

        assert score == 0.0  # No reasoning keywords

    # Uncertainty scoring tests
    def test_calculate_uncertainty_score_with_client(self, strategy_with_client):
        """Test uncertainty score calculation with LLM client."""
        strategy_with_client.lite_client.completion.return_value = "This is a normal response."

        score = strategy_with_client._calculate_uncertainty_score("test query")

        assert 0.0 <= score <= 1.0
        strategy_with_client.lite_client.completion.assert_called_once()

    def test_calculate_uncertainty_score_short_response(self, strategy_with_client):
        """Test uncertainty score with very short response."""
        strategy_with_client.lite_client.completion.return_value = "Yes."

        score = strategy_with_client._calculate_uncertainty_score("test query")

        assert score == 0.7  # High uncertainty for short response

    def test_calculate_uncertainty_score_long_response(self, strategy_with_client):
        """Test uncertainty score with very long response."""
        long_response = " ".join(["word"] * 50)  # 50 words
        strategy_with_client.lite_client.completion.return_value = long_response

        score = strategy_with_client._calculate_uncertainty_score("test query")

        assert score == 0.6  # High uncertainty for long response

    def test_calculate_uncertainty_score_hedging_words(self, strategy_with_client):
        """Test uncertainty score with hedging words."""
        strategy_with_client.lite_client.completion.return_value = "Maybe this could possibly work"

        score = strategy_with_client._calculate_uncertainty_score("test query")

        assert score > 0.3  # Should increase due to hedging words

    def test_calculate_uncertainty_score_no_client(self, strategy_no_client):
        """Test uncertainty score calculation without LLM client."""
        score = strategy_no_client._calculate_uncertainty_score("What is the answer?")

        assert 0.0 <= score <= 1.0

    def test_fallback_uncertainty_score_questions(self, strategy_with_client):
        """Test fallback uncertainty score with questions."""
        query = "What is this? How does it work? Why is it important?"
        score = strategy_with_client._fallback_uncertainty_score(query)

        assert score > 0.4  # Questions increase uncertainty

    def test_fallback_uncertainty_score_specific_vs_general(self, strategy_with_client):
        """Test fallback uncertainty score with specific vs general indicators."""
        # Specific query
        specific_query = "the exact precise specific method"
        specific_score = strategy_with_client._fallback_uncertainty_score(specific_query)

        # General query
        general_query = "a general overview of broad topics"
        general_score = strategy_with_client._fallback_uncertainty_score(general_query)

        assert general_score > specific_score

    # Integration tests
    def test_route_integration_lite(self, strategy_with_client):
        """Test complete routing to LITE tier."""
        # Mock LLM responses for reasoning and uncertainty
        strategy_with_client.lite_client.completion.side_effect = ["0", "short"]

        result = strategy_with_client.route("What is 2+2?")

        assert isinstance(result, RoutingResult)
        assert result.tier == ModelTier.LITE
        assert result.strategy == "dynamic_complexity"
        assert result.complexity_score.total < 0.3

    def test_route_integration_fast(self, strategy_with_client):
        """Test complete routing to FAST tier."""
        # Mock medium complexity responses
        strategy_with_client.lite_client.completion.side_effect = [
            "1",
            "This requires some reasoning but not too complex.",
        ]

        query = "Explain the basic principles of machine learning."
        result = strategy_with_client.route(query)

        assert result.tier == ModelTier.FAST
        assert 0.3 <= result.complexity_score.total < 0.65

    def test_route_integration_power(self, strategy_with_client):
        """Test complete routing to POWER tier."""
        # Mock high complexity responses
        strategy_with_client.lite_client.completion.side_effect = [
            "2",
            "This is a very complex question that requires extensive analysis and deep reasoning capabilities.",
        ]

        query = """
        Develop a comprehensive framework for analyzing the intersection of quantum computing
        and artificial intelligence, considering theoretical limitations, practical implementations,
        and potential breakthrough applications in cryptography and optimization.
        """
        result = strategy_with_client.route(query)

        assert result.tier == ModelTier.POWER
        assert result.complexity_score.total >= 0.65

    def test_route_detailed_complexity_breakdown(self, strategy_with_client):
        """Test that routing provides detailed complexity breakdown."""
        strategy_with_client.lite_client.completion.side_effect = ["1", "normal response"]

        result = strategy_with_client.route("Analyze this complex problem")

        # Check complexity score components
        assert result.complexity_score.linguistic is not None
        assert result.complexity_score.reasoning is not None
        assert result.complexity_score.uncertainty is not None
        assert result.complexity_score.metadata is not None

        # Check metadata structure
        metadata = result.complexity_score.metadata
        assert "weights" in metadata
        assert "components" in metadata
        assert "alpha" in metadata["weights"]

    def test_index_to_tier_mapping(self, strategy_with_client):
        """Test complexity index to tier mapping."""
        assert strategy_with_client._index_to_tier(0.2) == ModelTier.LITE
        assert strategy_with_client._index_to_tier(0.5) == ModelTier.FAST
        assert strategy_with_client._index_to_tier(0.8) == ModelTier.POWER

        # Test boundary values
        assert strategy_with_client._index_to_tier(0.3) == ModelTier.FAST  # At threshold
        assert strategy_with_client._index_to_tier(0.65) == ModelTier.POWER  # At threshold

    def test_calculate_confidence_consistent_scores(self, strategy_with_client):
        """Test confidence calculation with consistent component scores."""
        # Similar scores should give high confidence
        confidence = strategy_with_client._calculate_confidence(0.5, 0.5, 0.5)
        assert confidence > 0.8

    def test_calculate_confidence_inconsistent_scores(self, strategy_with_client):
        """Test confidence calculation with inconsistent component scores."""
        # Very different scores should give lower confidence
        confidence = strategy_with_client._calculate_confidence(0.1, 0.9, 0.5)
        assert confidence < 0.8

    @patch("cogents_core.logging_config.get_logger")
    def test_route_exception_handling(self, mock_logger, strategy_with_client):
        """Test route method exception handling."""
        # Make linguistic score calculation fail
        with patch.object(strategy_with_client, "_calculate_linguistic_score", side_effect=Exception("Test error")):
            result = strategy_with_client.route("test query")

            # Should fallback to FAST tier
            assert result.tier == ModelTier.FAST
            assert result.confidence == 0.0
            assert result.metadata["fallback"] is True

    def test_route_no_client_fallback(self, strategy_no_client):
        """Test routing without LLM client uses fallback methods."""
        result = strategy_no_client.route("analyze and compare these complex systems")

        assert isinstance(result, RoutingResult)
        # Should still work without client, using fallback methods
        assert result.complexity_score.reasoning is not None
        assert result.complexity_score.uncertainty is not None

    def test_multiple_queries_different_results(self, strategy_with_client):
        """Test that different queries produce different results."""
        # Simple query
        strategy_with_client.lite_client.completion.side_effect = ["0", "yes"]
        result1 = strategy_with_client.route("What is 1+1?")

        # Complex query
        strategy_with_client.lite_client.completion.side_effect = [
            "2",
            "This requires extensive analysis and consideration of multiple factors.",
        ]
        result2 = strategy_with_client.route(
            "Design a comprehensive solution for optimizing distributed systems performance."
        )

        assert result1.tier != result2.tier
        assert result1.complexity_score.total < result2.complexity_score.total
