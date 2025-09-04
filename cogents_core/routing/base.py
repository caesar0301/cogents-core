"""Base classes for routing strategies."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from cogents_core.llm.base import BaseLLMClient

from .types import ComplexityScore, ModelTier, RoutingResult


class BaseRoutingStrategy(ABC):
    """Abstract base class for routing strategies."""

    def __init__(self, lite_client: Optional[BaseLLMClient] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the routing strategy.

        Args:
            lite_client: Optional LLM client for lite model operations
            config: Strategy-specific configuration parameters
        """
        self.lite_client = lite_client
        self.config = (config or {}).copy()  # Make a copy to avoid modifying original

    @abstractmethod
    def route(self, query: str) -> RoutingResult:
        """
        Route a query to the appropriate model tier.

        Args:
            query: The input query to route

        Returns:
            RoutingResult with tier recommendation and analysis
        """

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the name of this routing strategy."""

    def _create_result(
        self,
        tier: ModelTier,
        confidence: float,
        complexity_score: ComplexityScore,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RoutingResult:
        """
        Helper method to create a RoutingResult.

        Args:
            tier: Recommended model tier
            confidence: Confidence in the decision
            complexity_score: Complexity analysis
            metadata: Additional metadata

        Returns:
            RoutingResult instance
        """
        return RoutingResult(
            tier=tier,
            confidence=confidence,
            complexity_score=complexity_score,
            strategy=self.get_strategy_name(),
            metadata=metadata,
        )
