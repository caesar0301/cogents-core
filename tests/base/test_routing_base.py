"""
Unit tests for routing base module.

Tests cover BaseRoutingStrategy abstract base class.
"""

from unittest.mock import Mock

import pytest

from cogents.core.base.routing.base import BaseRoutingStrategy
from cogents.core.base.routing.types import ComplexityScore, ModelTier, RoutingResult


class ConcreteRoutingStrategy(BaseRoutingStrategy):
    """Concrete implementation of BaseRoutingStrategy for testing."""

    def route(self, query: str) -> RoutingResult:
        """Mock implementation."""
        return self._create_result(
            tier=ModelTier.FAST, confidence=0.8, complexity_score=ComplexityScore(total=0.5), metadata={"test": "mock"}
        )

    def get_strategy_name(self) -> str:
        """Mock implementation."""
        return "concrete_test_strategy"


class TestBaseRoutingStrategy:
    """Test cases for BaseRoutingStrategy abstract base class."""

    @pytest.fixture
    def mock_llm_client(self):
        """Provide a mock LLM client for testing."""
        mock_client = Mock()
        mock_client.completion.return_value = "test response"
        return mock_client

    @pytest.fixture
    def sample_config(self):
        """Provide a sample configuration for testing."""
        return {"temperature": 0.5, "max_tokens": 100, "custom_param": "test_value"}

    def test_init_with_no_parameters(self):
        """Test BaseRoutingStrategy initialization with no parameters."""
        strategy = ConcreteRoutingStrategy()

        assert strategy.lite_client is None
        assert strategy.config == {}

    def test_init_with_lite_client(self, mock_llm_client):
        """Test BaseRoutingStrategy initialization with lite client."""
        strategy = ConcreteRoutingStrategy(lite_client=mock_llm_client)

        assert strategy.lite_client == mock_llm_client
        assert strategy.config == {}

    def test_init_with_config(self, sample_config):
        """Test BaseRoutingStrategy initialization with config."""
        strategy = ConcreteRoutingStrategy(config=sample_config)

        assert strategy.lite_client is None
        assert strategy.config == sample_config

    def test_init_with_all_parameters(self, mock_llm_client, sample_config):
        """Test BaseRoutingStrategy initialization with all parameters."""
        strategy = ConcreteRoutingStrategy(lite_client=mock_llm_client, config=sample_config)

        assert strategy.lite_client == mock_llm_client
        assert strategy.config == sample_config

    def test_init_with_none_config(self, mock_llm_client):
        """Test BaseRoutingStrategy initialization with None config."""
        strategy = ConcreteRoutingStrategy(lite_client=mock_llm_client, config=None)

        assert strategy.lite_client == mock_llm_client
        assert strategy.config == {}

    def test_route_method_implemented(self):
        """Test that route method is implemented in concrete class."""
        strategy = ConcreteRoutingStrategy()
        result = strategy.route("test query")

        assert isinstance(result, RoutingResult)
        assert result.tier == ModelTier.FAST
        assert result.confidence == 0.8
        assert result.strategy == "concrete_test_strategy"

    def test_get_strategy_name_implemented(self):
        """Test that get_strategy_name method is implemented in concrete class."""
        strategy = ConcreteRoutingStrategy()
        name = strategy.get_strategy_name()

        assert name == "concrete_test_strategy"

    def test_create_result_helper(self):
        """Test the _create_result helper method."""
        strategy = ConcreteRoutingStrategy()
        complexity_score = ComplexityScore(total=0.7, linguistic=0.6)
        metadata = {"test": "data"}

        result = strategy._create_result(
            tier=ModelTier.POWER, confidence=0.9, complexity_score=complexity_score, metadata=metadata
        )

        assert isinstance(result, RoutingResult)
        assert result.tier == ModelTier.POWER
        assert result.confidence == 0.9
        assert result.complexity_score == complexity_score
        assert result.strategy == "concrete_test_strategy"
        assert result.metadata == metadata

    def test_create_result_without_metadata(self):
        """Test _create_result without metadata."""
        strategy = ConcreteRoutingStrategy()
        complexity_score = ComplexityScore(total=0.3)

        result = strategy._create_result(tier=ModelTier.LITE, confidence=0.6, complexity_score=complexity_score)

        assert result.tier == ModelTier.LITE
        assert result.confidence == 0.6
        assert result.complexity_score == complexity_score
        assert result.strategy == "concrete_test_strategy"
        assert result.metadata is None

    def test_abstract_methods_must_be_implemented(self):
        """Test that abstract methods must be implemented."""

        class IncompleteStrategy(BaseRoutingStrategy):
            """Strategy missing required abstract methods."""

        # Should not be able to instantiate without implementing abstract methods
        with pytest.raises(TypeError):
            IncompleteStrategy()

    def test_partial_implementation_fails(self):
        """Test that partial implementation of abstract methods fails."""

        class PartialStrategy(BaseRoutingStrategy):
            """Strategy with only one abstract method implemented."""

            def route(self, query: str) -> RoutingResult:
                return self._create_result(
                    tier=ModelTier.FAST, confidence=0.5, complexity_score=ComplexityScore(total=0.5)
                )

            # Missing get_strategy_name implementation

        # Should not be able to instantiate without implementing all abstract methods
        with pytest.raises(TypeError):
            PartialStrategy()

    def test_config_is_mutable(self, sample_config):
        """Test that config can be modified after initialization."""
        strategy = ConcreteRoutingStrategy(config=sample_config)

        # Verify initial config
        assert strategy.config["temperature"] == 0.5

        # Modify config
        strategy.config["temperature"] = 0.7
        strategy.config["new_param"] = "new_value"

        # Verify changes
        assert strategy.config["temperature"] == 0.7
        assert strategy.config["new_param"] == "new_value"
        assert strategy.config["custom_param"] == "test_value"  # Original value preserved

    def test_config_isolation(self):
        """Test that config changes don't affect original dict."""
        original_config = {"param": "original"}
        strategy = ConcreteRoutingStrategy(config=original_config)

        # Modify strategy config
        strategy.config["param"] = "modified"
        strategy.config["new_param"] = "added"

        # Original config should be unchanged
        assert original_config == {"param": "original"}
        assert "new_param" not in original_config
