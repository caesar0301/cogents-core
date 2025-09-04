from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ConflictType(str, Enum):
    """Types of conflicts that can occur."""

    CYCLE_DETECTED = "cycle_detected"
    CONCURRENT_UPDATE = "concurrent_update"
    STATUS_INCONSISTENCY = "status_inconsistency"
    PRIORITY_CONFLICT = "priority_conflict"
    DEPENDENCY_VIOLATION = "dependency_violation"
    RESOURCE_CONFLICT = "resource_conflict"
    SEMANTIC_INCONSISTENCY = "semantic_inconsistency"


class Conflict(BaseModel):
    """
    Represents a conflict in the system.
    """

    # Core identification
    id: str = Field(default_factory=lambda: str(uuid4()))
    conflict_type: ConflictType = Field(..., description="Type of conflict")

    # Conflict details
    affected_nodes: List[str] = Field(..., description="IDs of affected nodes")
    description: str = Field(default="", description="Human-readable description")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional conflict context")
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="When the conflict was detected"
    )

    # Resolution status
    resolved: bool = Field(default=False, description="Whether the conflict has been resolved")
    resolution_strategy: Optional[str] = Field(default=None, description="Strategy used to resolve the conflict")
    resolution_action: Optional[Dict[str, Any]] = Field(
        default=None, description="Resolution action returned by resolver"
    )
    resolved_at: Optional[datetime] = Field(default=None, description="When the conflict was resolved")

    class Config:
        """Pydantic configuration."""

        use_enum_values = True

    def __eq__(self, other: object) -> bool:
        """
        Compare two Conflicts for equality based on meaningful content.

        Excludes timestamp fields as they are automatically generated.
        """
        if not isinstance(other, Conflict):
            return False

        return (
            self.id == other.id
            and self.conflict_type == other.conflict_type
            and self.affected_nodes == other.affected_nodes
            and self.description == other.description
            and self.context == other.context
            and self.resolved == other.resolved
            and self.resolution_strategy == other.resolution_strategy
            and self.resolution_action == other.resolution_action
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "conflict_type": str(self.conflict_type),
            "affected_nodes": self.affected_nodes,
            "description": self.description,
            "context": self.context,
            "detected_at": self.detected_at.isoformat() if self.detected_at else None,
            "resolved": self.resolved,
            "resolution_strategy": self.resolution_strategy,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


class ConflictResolver(ABC):
    """
    Abstract interface for conflict resolvers.

    Can be implemented by LLM-based reasoning, human input, or rule-based systems.
    """

    @abstractmethod
    def resolve(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        """
        Resolve a conflict.

        Args:
            conflict: The conflict to resolve

        Returns:
            Resolution action description, or None if cannot resolve
        """
