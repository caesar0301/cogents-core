"""Main router class for LLM model selection."""

from typing import Any, Dict, Optional, Type, Union

from cogents_core.llm import get_llm_client
from cogents_core.llm.base import BaseLLMClient
from cogents_core.utils.logging import get_logger

from .base import BaseRoutingStrategy
from .strategies import DynamicComplexityStrategy, SelfAssessmentStrategy
from .types import ModelTier, RoutingResult

logger = get_logger(__name__)


class ModelRouter:
    """
    Main router class for determining appropriate model tier for queries.

    This router uses pluggable strategies to analyze query complexity
    and recommend the most suitable model tier (lite/fast/power).
    """

    # Registry of available strategies
    STRATEGIES = {
        "self_assessment": SelfAssessmentStrategy,
        "dynamic_complexity": DynamicComplexityStrategy,
    }

    def __init__(
        self,
        strategy: Union[str, BaseRoutingStrategy] = "dynamic_complexity",
        lite_client: Optional[BaseLLMClient] = None,
        lite_client_config: Optional[Dict[str, Any]] = None,
        strategy_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the model router.

        Args:
            strategy: Routing strategy name or instance
            lite_client: Pre-configured lite model client
            lite_client_config: Configuration for lite client creation
            strategy_config: Strategy-specific configuration
        """
        self.strategy_config = (strategy_config or {}).copy()  # Make a copy to avoid modifying original

        # Setup lite client if needed
        self.lite_client = self._setup_lite_client(lite_client, lite_client_config)

        # Initialize routing strategy
        self.strategy = self._setup_strategy(strategy)

        logger.info(f"Initialized ModelRouter with strategy: {self.strategy.get_strategy_name()}")

    def route(self, query: str) -> RoutingResult:
        """
        Route a query to the appropriate model tier.

        Args:
            query: Input query to analyze

        Returns:
            RoutingResult with tier recommendation and analysis
        """
        if not query or not query.strip():
            # Handle empty queries
            logger.warning("Empty query provided, defaulting to LITE tier")
            from .types import ComplexityScore

            return RoutingResult(
                tier=ModelTier.LITE,
                confidence=0.5,
                complexity_score=ComplexityScore(total=0.0),
                strategy=self.strategy.get_strategy_name(),
                metadata={"empty_query": True},
            )

        try:
            return self.strategy.route(query.strip())
        except Exception as e:
            logger.error(f"Error in routing: {e}")
            # Fallback to FAST tier on error
            from .types import ComplexityScore

            return RoutingResult(
                tier=ModelTier.FAST,
                confidence=0.0,
                complexity_score=ComplexityScore(total=0.5),
                strategy=self.strategy.get_strategy_name(),
                metadata={"error": str(e), "fallback": True},
            )

    def get_recommended_model_params(
        self, routing_result: RoutingResult, model_configs: Optional[Dict[ModelTier, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Get recommended model parameters based on routing result.

        Args:
            routing_result: Result from route() call
            model_configs: Optional mapping of tiers to model configurations

        Returns:
            Dictionary of model parameters for the recommended tier
        """
        tier = routing_result.tier

        if model_configs and tier in model_configs:
            return model_configs[tier].copy()

        # Default configurations for each tier
        default_configs = {
            ModelTier.LITE: {
                "provider": "llamacpp",  # Fast local inference
                "temperature": 0.3,
                "max_tokens": 512,
            },
            ModelTier.FAST: {
                "provider": "openai",
                "chat_model": "gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            ModelTier.POWER: {
                "provider": "openai",
                "chat_model": "gpt-4o",
                "temperature": 0.7,
                "max_tokens": 2048,
            },
        }

        return default_configs.get(tier, default_configs[ModelTier.FAST])

    def route_and_configure(
        self, query: str, model_configs: Optional[Dict[ModelTier, Dict[str, Any]]] = None
    ) -> tuple[RoutingResult, Dict[str, Any]]:
        """
        Route query and return both result and recommended model configuration.

        Args:
            query: Input query to analyze
            model_configs: Optional mapping of tiers to model configurations

        Returns:
            Tuple of (RoutingResult, model_config_dict)
        """
        routing_result = self.route(query)
        model_config = self.get_recommended_model_params(routing_result, model_configs)

        return routing_result, model_config

    def update_strategy_config(self, config: Dict[str, Any]) -> None:
        """
        Update strategy configuration.

        Args:
            config: New configuration parameters
        """
        self.strategy_config.update(config)

        # Reinitialize strategy with new config
        strategy_name = self.strategy.get_strategy_name()
        self.strategy = self._setup_strategy(strategy_name)

        logger.info(f"Updated strategy configuration for {strategy_name}")

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about the current routing strategy.

        Returns:
            Dictionary with strategy information
        """
        return {
            "name": self.strategy.get_strategy_name(),
            "config": self.strategy_config.copy(),
            "requires_lite_client": self.lite_client is not None,
        }

    @classmethod
    def get_available_strategies(cls) -> list[str]:
        """Get list of available strategy names."""
        return list(cls.STRATEGIES.keys())

    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[BaseRoutingStrategy]) -> None:
        """
        Register a new routing strategy.

        Args:
            name: Strategy name
            strategy_class: Strategy class
        """
        if not issubclass(strategy_class, BaseRoutingStrategy):
            raise ValueError("Strategy class must inherit from BaseRoutingStrategy")

        cls.STRATEGIES[name] = strategy_class
        logger.info(f"Registered new routing strategy: {name}")

    def _setup_lite_client(
        self, lite_client: Optional[BaseLLMClient], lite_client_config: Optional[Dict[str, Any]]
    ) -> Optional[BaseLLMClient]:
        """Setup lite model client if needed."""
        if lite_client:
            return lite_client

        if lite_client_config:
            try:
                return get_llm_client(**lite_client_config)
            except Exception as e:
                logger.warning(f"Failed to create lite client: {e}")
                return None

        # Try to create a default lite client
        try:
            # Try llamacpp first (fastest for lite operations)
            return get_llm_client(provider="llamacpp")
        except Exception:
            try:
                # Fallback to OpenAI with a fast model
                return get_llm_client(provider="openai", chat_model="gpt-4o-mini")
            except Exception as e:
                logger.warning(f"Could not create default lite client: {e}")
                return None

    def _setup_strategy(self, strategy: Union[str, BaseRoutingStrategy]) -> BaseRoutingStrategy:
        """Setup the routing strategy."""
        if isinstance(strategy, BaseRoutingStrategy):
            return strategy

        if isinstance(strategy, str):
            if strategy not in self.STRATEGIES:
                raise ValueError(
                    f"Unknown strategy: {strategy}. " f"Available strategies: {list(self.STRATEGIES.keys())}"
                )

            strategy_class = self.STRATEGIES[strategy]
            return strategy_class(lite_client=self.lite_client, config=self.strategy_config)

        raise ValueError(f"Invalid strategy type: {type(strategy)}")
