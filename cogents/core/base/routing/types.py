"""Type definitions and enums for the routing module."""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class ModelTier(str, Enum):
    """Enumeration of model tiers based on capability and resource requirements."""

    LITE = "lite"  # Fast, low-resource models for simple tasks
    FAST = "fast"  # Balanced models for moderate complexity
    POWER = "power"  # High-capability models for complex reasoning


class ComplexityScore(BaseModel):
    """Represents a complexity score with breakdown of different factors."""

    total: float = Field(..., ge=0.0, le=1.0, description="Overall complexity score (0.0-1.0)")
    linguistic: Optional[float] = Field(None, ge=0.0, le=1.0, description="Linguistic complexity component")
    reasoning: Optional[float] = Field(None, ge=0.0, le=1.0, description="Reasoning depth component")
    uncertainty: Optional[float] = Field(None, ge=0.0, le=1.0, description="Knowledge uncertainty component")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class RoutingResult(BaseModel):
    """Result of a routing decision."""

    tier: ModelTier = Field(..., description="Recommended model tier")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the routing decision (0.0-1.0)")
    complexity_score: ComplexityScore = Field(..., description="Detailed complexity breakdown")
    strategy: str = Field(..., description="Name of the strategy used")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional routing metadata")

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
