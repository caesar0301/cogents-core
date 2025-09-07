"""
Pydantic models for memory system data structures.

This module defines the core data models used by the simple memory system,
providing type safety and validation using Pydantic.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class BaseMemoryItem(BaseModel):
    """
    Base class for all memory items with common fields and functionality.

    This class provides the foundation for all memory items, ensuring
    consistent metadata and identification across different memory types.
    """

    model_config = ConfigDict(from_attributes=True, validate_default=True, arbitrary_types_allowed=True, extra="allow")

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the memory item")

    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when the memory was created")

    updated_at: Optional[datetime] = Field(default=None, description="Timestamp when the memory was last updated")

    version: int = Field(default=1, description="Version number for tracking memory updates", ge=1)

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the memory item")

    tags: List[str] = Field(default_factory=list, description="Tags for categorizing and filtering memory items")

    def add_tag(self, tag: str) -> None:
        """Add a tag to the memory item if it doesn't already exist."""
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the memory item."""
        if tag in self.tags:
            self.tags.remove(tag)

    def update_metadata(self, key: str, value: Any) -> None:
        """Update a metadata field."""
        self.metadata[key] = value
        self.updated_at = datetime.utcnow()


class MemoryItem(BaseMemoryItem):
    """
    Standard memory item for storing general information.

    This is the primary memory item type for storing conversations,
    facts, or any textual information with context.
    """

    content: str = Field(description="The main content/text of the memory item")

    memory_type: Literal["message", "fact", "note"] = Field(
        default="message", description="Type of memory item for categorization"
    )

    user_id: Optional[str] = Field(default=None, description="ID of the user associated with this memory")

    agent_id: Optional[str] = Field(default=None, description="ID of the agent associated with this memory")

    session_id: Optional[str] = Field(default=None, description="ID of the session/conversation this memory belongs to")

    importance: float = Field(
        default=0.5, description="Importance score of the memory item (0.0 to 1.0)", ge=0.0, le=1.0
    )

    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional context information for the memory"
    )


class MemoryFilter(BaseModel):
    """
    Filter model for querying memory items.

    This model provides a structured way to filter memory items
    based on various criteria.
    """

    user_id: Optional[str] = Field(default=None, description="Filter by user ID")

    agent_id: Optional[str] = Field(default=None, description="Filter by agent ID")

    session_id: Optional[str] = Field(default=None, description="Filter by session/conversation ID")

    memory_type: Optional[str] = Field(default=None, description="Filter by memory type")

    tags: Optional[List[str]] = Field(default=None, description="Filter by tags (items must have all specified tags)")

    date_from: Optional[datetime] = Field(default=None, description="Filter items created after this date")

    date_to: Optional[datetime] = Field(default=None, description="Filter items created before this date")

    min_importance: Optional[float] = Field(
        default=None, description="Filter items with importance above this threshold", ge=0.0, le=1.0
    )

    metadata_filters: Dict[str, Any] = Field(default_factory=dict, description="Filter by metadata key-value pairs")


class SearchResult(BaseModel):
    """
    Model for search results with relevance scoring.

    This model wraps memory items with relevance scores
    for search and retrieval operations.
    """

    memory_item: MemoryItem = Field(description="The retrieved memory item")

    relevance_score: float = Field(description="Relevance score for the search query", ge=0.0, le=1.0)

    distance: Optional[float] = Field(default=None, description="Distance metric from vector search (if applicable)")

    search_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the search result"
    )


class MemoryStats(BaseModel):
    """
    Statistics model for memory system analytics.

    This model provides insights into memory usage and performance.
    """

    total_items: int = Field(description="Total number of memory items")

    items_by_type: Dict[str, int] = Field(default_factory=dict, description="Count of items by memory type")

    items_by_user: Dict[str, int] = Field(default_factory=dict, description="Count of items by user ID")

    oldest_item_date: Optional[datetime] = Field(default=None, description="Date of the oldest memory item")

    newest_item_date: Optional[datetime] = Field(default=None, description="Date of the newest memory item")

    average_importance: float = Field(default=0.0, description="Average importance score across all items")

    storage_size_bytes: Optional[int] = Field(default=None, description="Approximate storage size in bytes")
