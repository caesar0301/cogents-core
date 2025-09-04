"""Model routing module for determining appropriate LLM tiers based on query complexity."""

from .base import BaseRoutingStrategy
from .router import ModelRouter
from .strategies import DynamicComplexityStrategy, SelfAssessmentStrategy
from .types import ComplexityScore, ModelTier, RoutingResult

__all__ = [
    # Main router class
    "ModelRouter",
    # Base classes for extensibility
    "BaseRoutingStrategy",
    # Types and enums
    "ModelTier",
    "ComplexityScore",
    "RoutingResult",
    # Built-in strategies
    "SelfAssessmentStrategy",
    "DynamicComplexityStrategy",
]
