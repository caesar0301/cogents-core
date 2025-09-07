"""
Unit tests for model router.

Tests cover ModelRouter main class and its integration with strategies.
"""

from unittest.mock import Mock, patch

import pytest

from cogents_core.routing.base import BaseRoutingStrategy
from cogents_core.routing.router import ModelRouter
from cogents_core.routing.strategies import DynamicComplexityStrategy, SelfAssessmentStrategy
from cogents_core.routing.types import ComplexityScore, ModelTier, RoutingResult


class MockStrategy(BaseRoutingStrategy):
    """Mock strategy for testing."""

    def __init__(self, lite_client=None, config=None, return_tier=ModelTier.FAST):
        super().__init__(lite_client, config)
        self.return_tier = return_tier

    def route(self, query: str) -> RoutingResult:
        return self._create_result(
            tier=self.return_tier, confidence=0.8, complexity_score=ComplexityScore(total=0.5), metadata={"mock": True}
        )

    def get_strategy_name(self) -> str:
        return "mock_strategy"


class TestModelRouter:
    """Test cases for ModelRouter."""

    @pytest.fixture
    def mock_llm_client(self):
        """Provide a mock LLM client for testing."""
        mock_client = Mock()
        mock_client.completion.return_value = "2"
        return mock_client

    def test_init_default_strategy(self):
        """Test ModelRouter initialization with default strategy."""
        with patch("cogents_core.routing.router.get_llm_client") as mock_get_client:
            mock_get_client.return_value = Mock()

            router = ModelRouter()

            assert isinstance(router.strategy, DynamicComplexityStrategy)
            assert router.strategy.get_strategy_name() == "dynamic_complexity"

    def test_init_with_strategy_name(self):
        """Test ModelRouter initialization with strategy name."""
        with patch("cogents_core.routing.router.get_llm_client") as mock_get_client:
            mock_get_client.return_value = Mock()

            router = ModelRouter(strategy="self_assessment")

            assert isinstance(router.strategy, SelfAssessmentStrategy)

    def test_init_with_strategy_instance(self, mock_llm_client):
        """Test ModelRouter initialization with strategy instance."""
        strategy = MockStrategy(lite_client=mock_llm_client)

        with patch("cogents_core.routing.router.get_llm_client") as mock_get_client:
            mock_get_client.side_effect = Exception("Should not be called")

            router = ModelRouter(strategy=strategy)

            assert router.strategy == strategy

    def test_init_with_lite_client(self, mock_llm_client):
        """Test ModelRouter initialization with pre-configured lite client."""
        router = ModelRouter(lite_client=mock_llm_client)

        assert router.lite_client == mock_llm_client

    def test_init_with_lite_client_config(self):
        """Test ModelRouter initialization with lite client config."""
        config = {"provider": "openai", "chat_model": "gpt-4o-mini"}

        with patch("cogents_core.routing.router.get_llm_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            router = ModelRouter(lite_client_config=config)

            mock_get_client.assert_called_with(**config)
            assert router.lite_client == mock_client

    def test_init_with_strategy_config(self, mock_llm_client):
        """Test ModelRouter initialization with strategy config."""
        strategy_config = {"temperature": 0.5, "max_tokens": 100}

        router = ModelRouter(strategy="self_assessment", lite_client=mock_llm_client, strategy_config=strategy_config)

        assert router.strategy_config == strategy_config

    def test_init_invalid_strategy_name(self):
        """Test ModelRouter initialization with invalid strategy name."""
        with patch("cogents_core.routing.router.get_llm_client") as mock_get_client:
            mock_get_client.side_effect = Exception("Should not be called")

            with pytest.raises(ValueError, match="Unknown strategy"):
                ModelRouter(strategy="nonexistent_strategy")

    def test_lite_client_setup_fallback_llamacpp(self):
        """Test lite client setup with llamacpp fallback."""
        with patch("cogents_core.routing.router.get_llm_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            router = ModelRouter()

            # Should try llamacpp first
            mock_get_client.assert_called_with(provider="llamacpp")
            assert router.lite_client == mock_client

    def test_lite_client_setup_fallback_openai(self):
        """Test lite client setup with OpenAI fallback when llamacpp fails."""
        with patch("cogents_core.routing.router.get_llm_client") as mock_get_client:
            # First call (llamacpp) fails, second call (openai) succeeds
            mock_client = Mock()
            mock_get_client.side_effect = [Exception("llamacpp failed"), mock_client]

            router = ModelRouter()

            assert router.lite_client == mock_client
            # Should have called both providers
            assert mock_get_client.call_count == 2

    def test_lite_client_setup_complete_failure(self):
        """Test lite client setup when all providers fail."""
        with patch("cogents_core.routing.router.get_llm_client") as mock_get_client:
            mock_get_client.side_effect = Exception("All providers failed")

            with patch("cogents_core.utils.logging.get_logger"):
                router = ModelRouter()

                assert router.lite_client is None

    def test_route_success(self):
        """Test successful routing."""
        strategy = MockStrategy(return_tier=ModelTier.POWER)

        with patch("cogents_core.routing.router.get_llm_client") as mock_get_client:
            mock_get_client.side_effect = Exception("Should not be called")

            router = ModelRouter(strategy=strategy)

            result = router.route("Complex analysis task")

            assert isinstance(result, RoutingResult)
            assert result.tier == ModelTier.POWER
            assert result.confidence == 0.8
            assert result.strategy == "mock_strategy"

    def test_route_empty_query(self):
        """Test routing with empty query."""
        strategy = MockStrategy()

        with patch("cogents_core.routing.router.get_llm_client") as mock_get_client:
            mock_get_client.side_effect = Exception("Should not be called")

            router = ModelRouter(strategy=strategy)

            result = router.route("")

            # Should default to LITE tier for empty queries
            assert result.tier == ModelTier.LITE
            assert result.confidence == 0.5
            assert result.metadata["empty_query"] is True

    def test_route_whitespace_query(self):
        """Test routing with whitespace-only query."""
        strategy = MockStrategy()

        with patch("cogents_core.routing.router.get_llm_client") as mock_get_client:
            mock_get_client.side_effect = Exception("Should not be called")

            router = ModelRouter(strategy=strategy)

            result = router.route("   \n\t   ")

            # Should default to LITE tier for empty queries
            assert result.tier == ModelTier.LITE

    def test_route_strips_whitespace(self):
        """Test that route strips whitespace from query."""
        strategy = MockStrategy(return_tier=ModelTier.FAST)

        with patch("cogents_core.routing.router.get_llm_client") as mock_get_client:
            mock_get_client.side_effect = Exception("Should not be called")

            router = ModelRouter(strategy=strategy)

            result = router.route("  test query  ")

            # Should process the query (not treat as empty)
            assert result.tier == ModelTier.FAST
            assert "empty_query" not in result.metadata

    @patch("cogents_core.utils.logging.get_logger")
    def test_route_strategy_exception(self, mock_logger):
        """Test route method when strategy raises exception."""

        # Create a proper mock strategy that inherits from BaseRoutingStrategy
        class ErrorStrategy(BaseRoutingStrategy):
            def route(self, query: str) -> RoutingResult:
                raise Exception("Strategy error")

            def get_strategy_name(self) -> str:
                return "error_strategy"

        with patch("cogents_core.routing.router.get_llm_client") as mock_get_client:
            mock_get_client.side_effect = Exception("Should not be called")

            router = ModelRouter(strategy=ErrorStrategy())

            result = router.route("test query")

            # Should fallback to FAST tier
            assert result.tier == ModelTier.FAST
            assert result.confidence == 0.0
            assert result.metadata["fallback"] is True
            assert "error" in result.metadata

    def test_get_recommended_model_params_default(self):
        """Test getting default model parameters."""
        with patch("cogents_core.routing.router.get_llm_client") as mock_get_client:
            mock_get_client.side_effect = Exception("Should not be called")

            router = ModelRouter(strategy=MockStrategy())
            result = RoutingResult(
                tier=ModelTier.FAST, confidence=0.8, complexity_score=ComplexityScore(total=0.5), strategy="test"
            )

            params = router.get_recommended_model_params(result)

            expected = {
                "provider": "openai",
                "chat_model": "gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 1024,
            }
            assert params == expected

    def test_get_recommended_model_params_all_tiers(self):
        """Test getting model parameters for all tiers."""
        with patch("cogents_core.routing.router.get_llm_client") as mock_get_client:
            mock_get_client.side_effect = Exception("Should not be called")

            router = ModelRouter(strategy=MockStrategy())

            # Test LITE tier
            result_lite = RoutingResult(
                tier=ModelTier.LITE, confidence=0.8, complexity_score=ComplexityScore(total=0.2), strategy="test"
            )
            params_lite = router.get_recommended_model_params(result_lite)
            assert params_lite["provider"] == "llamacpp"

            # Test FAST tier
            result_fast = RoutingResult(
                tier=ModelTier.FAST, confidence=0.8, complexity_score=ComplexityScore(total=0.5), strategy="test"
            )
            params_fast = router.get_recommended_model_params(result_fast)
            assert params_fast["chat_model"] == "gpt-4o-mini"

            # Test POWER tier
            result_power = RoutingResult(
                tier=ModelTier.POWER, confidence=0.8, complexity_score=ComplexityScore(total=0.8), strategy="test"
            )
            params_power = router.get_recommended_model_params(result_power)
            assert params_power["chat_model"] == "gpt-4o"

    def test_get_recommended_model_params_custom_configs(self):
        """Test getting model parameters with custom configs."""
        with patch("cogents_core.routing.router.get_llm_client") as mock_get_client:
            mock_get_client.side_effect = Exception("Should not be called")

            router = ModelRouter(strategy=MockStrategy())

            custom_configs = {
                ModelTier.LITE: {"provider": "custom", "model": "lite-model"},
                ModelTier.FAST: {"provider": "custom", "model": "fast-model"},
            }

            result = RoutingResult(
                tier=ModelTier.LITE, confidence=0.8, complexity_score=ComplexityScore(total=0.2), strategy="test"
            )

            params = router.get_recommended_model_params(result, custom_configs)

            assert params == {"provider": "custom", "model": "lite-model"}

    def test_route_and_configure(self):
        """Test route_and_configure method."""
        strategy = MockStrategy(return_tier=ModelTier.POWER)

        with patch("cogents_core.routing.router.get_llm_client") as mock_get_client:
            mock_get_client.side_effect = Exception("Should not be called")

            router = ModelRouter(strategy=strategy)

            result, config = router.route_and_configure("Complex task")

            assert isinstance(result, RoutingResult)
            assert result.tier == ModelTier.POWER
            assert isinstance(config, dict)
            assert config["chat_model"] == "gpt-4o"  # POWER tier default

    def test_update_strategy_config(self, mock_llm_client):
        """Test updating strategy configuration."""
        router = ModelRouter(
            strategy="self_assessment", lite_client=mock_llm_client, strategy_config={"temperature": 0.1}
        )

        # Update config
        new_config = {"temperature": 0.5, "max_tokens": 10}
        router.update_strategy_config(new_config)

        # Config should be updated
        assert router.strategy_config["temperature"] == 0.5
        assert router.strategy_config["max_tokens"] == 10

        # Strategy should be reinitialized
        assert isinstance(router.strategy, SelfAssessmentStrategy)

    def test_get_strategy_info(self, mock_llm_client):
        """Test getting strategy information."""
        config = {"temperature": 0.3}
        router = ModelRouter(strategy="self_assessment", lite_client=mock_llm_client, strategy_config=config)

        info = router.get_strategy_info()

        assert info["name"] == "self_assessment"
        assert info["config"] == config
        assert info["requires_lite_client"] is True

    def test_get_available_strategies(self):
        """Test getting available strategies."""
        strategies = ModelRouter.get_available_strategies()

        assert "self_assessment" in strategies
        assert "dynamic_complexity" in strategies
        assert isinstance(strategies, list)

    def test_register_strategy(self):
        """Test registering a new strategy."""

        class CustomStrategy(BaseRoutingStrategy):
            def route(self, query: str) -> RoutingResult:
                return self._create_result(
                    tier=ModelTier.FAST, confidence=1.0, complexity_score=ComplexityScore(total=0.5)
                )

            def get_strategy_name(self) -> str:
                return "custom_strategy"

        # Register the strategy
        ModelRouter.register_strategy("custom", CustomStrategy)

        # Should be available
        assert "custom" in ModelRouter.get_available_strategies()

        # Should be able to use it
        router = ModelRouter(strategy="custom")
        assert isinstance(router.strategy, CustomStrategy)

    def test_register_strategy_invalid_class(self):
        """Test registering invalid strategy class."""

        class InvalidStrategy:
            """Not a BaseRoutingStrategy subclass."""

        with pytest.raises(ValueError, match="must inherit from BaseRoutingStrategy"):
            ModelRouter.register_strategy("invalid", InvalidStrategy)

    def test_setup_strategy_invalid_type(self):
        """Test _setup_strategy with invalid strategy type."""
        router = ModelRouter(strategy=MockStrategy())

        with pytest.raises(ValueError, match="Invalid strategy type"):
            router._setup_strategy(123)  # Invalid type

    def test_strategy_initialization_logging(self):
        """Test that strategy initialization is logged."""
        with patch("cogents_core.routing.router.get_llm_client"):
            with patch("cogents_core.routing.router.logger") as mock_logger:
                router = ModelRouter(strategy="self_assessment")

                # Should log initialization
                mock_logger.info.assert_called()
                log_message = mock_logger.info.call_args[0][0]
                assert "Initialized ModelRouter" in log_message
                assert "self_assessment" in log_message

    def test_config_isolation(self):
        """Test that strategy config changes don't affect original dict."""
        original_config = {"param": "original"}
        router = ModelRouter(strategy=MockStrategy(), strategy_config=original_config)

        # Modify router's config
        router.strategy_config["param"] = "modified"
        router.strategy_config["new_param"] = "added"

        # Original config should be unchanged
        assert original_config == {"param": "original"}
        assert "new_param" not in original_config

    def test_multiple_routers_independent(self, mock_llm_client):
        """Test that multiple router instances are independent."""
        router1 = ModelRouter(
            strategy="self_assessment", lite_client=mock_llm_client, strategy_config={"temperature": 0.1}
        )

        router2 = ModelRouter(strategy="dynamic_complexity", strategy_config={"alpha": 0.5})

        # Routers should have different strategies
        assert isinstance(router1.strategy, SelfAssessmentStrategy)
        assert isinstance(router2.strategy, DynamicComplexityStrategy)

        # Configs should be independent
        assert router1.strategy_config != router2.strategy_config

        # Updating one shouldn't affect the other
        router1.update_strategy_config({"new_param": "value"})
        assert "new_param" not in router2.strategy_config
