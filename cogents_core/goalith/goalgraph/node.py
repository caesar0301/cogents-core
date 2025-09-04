"""
Goal Node data model for the GoalithService DAG-based goal management system.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field


class NodeStatus(str, Enum):
    """Status of a node in the goal DAG."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class GoalNode(BaseModel):
    """
    Lightweight DTO representing a node in the goal DAG.

    Contains all metadata needed for goal tracking, decomposition,
    and dependency management.
    """

    # Node metadata
    id: str = Field(default_factory=lambda: str(uuid4()))
    description: str = Field(..., description="Human-readable description of the goal/task")
    status: NodeStatus = Field(default=NodeStatus.PENDING)
    priority: float = Field(default=1.0, description="Priority score (higher = more important)")
    decomposer_name: Optional[str] = Field(default=None, description="Name of decomposer used")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context data")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    estimated_effort: Optional[str] = Field(default=None, description="Estimated effort for this task")

    # Relationships (managed by GraphStore, but stored here for serialization)
    dependencies: Set[str] = Field(default_factory=set, description="IDs of nodes this depends on")
    children: Set[str] = Field(default_factory=set, description="IDs of child nodes")
    parent: Optional[str] = Field(default=None, description="ID of parent node")

    # Timing
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    deadline: Optional[datetime] = Field(default=None)

    # Execution tracking
    assigned_to: Optional[str] = Field(default=None, description="Agent or user assigned to this node")
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=3)
    error_message: Optional[str] = Field(default=None)
    execution_notes: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        use_enum_values = True

    def is_ready(self) -> bool:
        """Check if this node is ready for execution (all dependencies completed)."""
        return self.status == NodeStatus.PENDING

    def is_terminal(self) -> bool:
        """Check if this node is in a terminal state."""
        return self.status in {
            NodeStatus.COMPLETED,
            NodeStatus.FAILED,
            NodeStatus.CANCELLED,
        }

    def can_retry(self) -> bool:
        """Check if this node can be retried after failure."""
        return self.status == NodeStatus.FAILED and self.retry_count < self.max_retries

    def mark_started(self) -> None:
        """Mark this node as started."""
        self.status = NodeStatus.IN_PROGRESS
        self.started_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def mark_completed(self) -> None:
        """Mark this node as completed."""
        self.status = NodeStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def mark_failed(self, error_message: Optional[str] = None) -> None:
        """Mark this node as failed with optional error message."""
        self.status = NodeStatus.FAILED
        self.retry_count += 1
        if error_message:
            self.error_message = error_message
        self.updated_at = datetime.now(timezone.utc)

    def mark_cancelled(self) -> None:
        """Mark this node as cancelled."""
        self.status = NodeStatus.CANCELLED
        self.updated_at = datetime.now(timezone.utc)

    def add_note(self, note: str) -> None:
        """Add an execution note."""
        self.execution_notes[datetime.now(timezone.utc).isoformat()] = note
        self.updated_at = datetime.now(timezone.utc)

    def update_context(self, key: str, value: Any) -> None:
        """Update context with a key-value pair."""
        self.context[key] = value
        self.updated_at = datetime.now(timezone.utc)

    def add_dependency(self, node_id: str) -> None:
        """Add a dependency to this node."""
        self.dependencies.add(node_id)
        self.updated_at = datetime.now(timezone.utc)

    def remove_dependency(self, node_id: str) -> None:
        """Remove a dependency from this node."""
        self.dependencies.discard(node_id)
        self.updated_at = datetime.now(timezone.utc)

    def add_child(self, node_id: str) -> None:
        """Add a child to this node."""
        self.children.add(node_id)
        self.updated_at = datetime.now(timezone.utc)

    def remove_child(self, node_id: str) -> None:
        """Remove a child from this node."""
        self.children.discard(node_id)
        self.updated_at = datetime.now(timezone.utc)

    def __hash__(self) -> int:
        """Make GoalNode hashable based on its ID."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """
        Compare two GoalNodes for equality based on meaningful content.

        Excludes timestamp fields (created_at, updated_at, started_at, completed_at)
        as these are automatically generated and not part of the logical state.
        """
        if not isinstance(other, GoalNode):
            return False

        # Compare all fields except timestamps
        return (
            self.id == other.id
            and self.description == other.description
            and self.status == other.status
            and self.priority == other.priority
            and self.estimated_effort == other.estimated_effort
            and self.dependencies == other.dependencies
            and self.children == other.children
            and self.parent == other.parent
            and self.context == other.context
            and self.tags == other.tags
            and self.decomposer_name == other.decomposer_name
            and self.assigned_to == other.assigned_to
            and self.deadline == other.deadline
            and self.retry_count == other.retry_count
            and self.max_retries == other.max_retries
            and self.error_message == other.error_message
            and self.execution_notes == other.execution_notes
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the GoalNode to a dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoalNode":
        """Create a GoalNode from a dictionary."""
        return cls(**data)
