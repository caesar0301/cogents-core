"""
Abstract base interface for the simple memory system.

This module defines the core abstract interfaces that all memory implementations
must implement, providing a consistent API for memory operations across different
storage backends and use cases.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from .models import MemoryFilter, MemoryItem, MemoryStats, SearchResult


class BaseMemoryStore(ABC):
    """
    Abstract base class for memory storage systems.

    This interface defines the core memory operations that all memory store
    implementations must support. It provides a consistent API for storing,
    retrieving, searching, and managing memory items across different backends.

    The interface supports both synchronous and asynchronous operations,
    allowing implementations to choose the most appropriate approach for
    their storage backend.
    """

    # ==========================================
    # Core CRUD Operations
    # ==========================================

    @abstractmethod
    async def add(self, memory_item: MemoryItem, **kwargs) -> str:
        """
        Add a new memory item to the store.

        This method stores a new memory item and returns its unique identifier.
        The implementation should handle ID generation if not provided and
        ensure the item is properly indexed for search operations.

        Args:
            memory_item: The memory item to store
            **kwargs: Additional implementation-specific parameters

        Returns:
            The unique identifier of the stored memory item

        Raises:
            MemoryError: If the item cannot be stored
            ValidationError: If the memory item is invalid
        """

    @abstractmethod
    async def get(self, memory_id: str, **kwargs) -> Optional[MemoryItem]:
        """
        Retrieve a memory item by its unique identifier.

        This method fetches a specific memory item from the store using its ID.
        Returns None if the item is not found.

        Args:
            memory_id: Unique identifier of the memory item
            **kwargs: Additional implementation-specific parameters

        Returns:
            The memory item if found, None otherwise

        Raises:
            MemoryError: If retrieval fails due to storage issues
        """

    @abstractmethod
    async def update(self, memory_id: str, updates: Dict[str, Any], **kwargs) -> bool:
        """
        Update an existing memory item.

        This method updates specific fields of a memory item. The implementation
        should validate updates and maintain version tracking if supported.

        Args:
            memory_id: Unique identifier of the memory item to update
            updates: Dictionary of field updates to apply
            **kwargs: Additional implementation-specific parameters

        Returns:
            True if the update was successful, False if item not found

        Raises:
            MemoryError: If update fails due to storage issues
            ValidationError: If updates are invalid
        """

    @abstractmethod
    async def delete(self, memory_id: str, **kwargs) -> bool:
        """
        Delete a memory item from the store.

        This method permanently removes a memory item and all associated
        index entries. The operation cannot be undone.

        Args:
            memory_id: Unique identifier of the memory item to delete
            **kwargs: Additional implementation-specific parameters

        Returns:
            True if deletion was successful, False if item not found

        Raises:
            MemoryError: If deletion fails due to storage issues
        """

    # ==========================================
    # Batch Operations
    # ==========================================

    @abstractmethod
    async def add_many(self, memory_items: List[MemoryItem], **kwargs) -> List[str]:
        """
        Add multiple memory items in a batch operation.

        This method efficiently stores multiple memory items in a single
        operation, which is useful for bulk imports or conversation logging.

        Args:
            memory_items: List of memory items to store
            **kwargs: Additional implementation-specific parameters

        Returns:
            List of unique identifiers for the stored items

        Raises:
            MemoryError: If batch operation fails
            ValidationError: If any memory item is invalid
        """

    @abstractmethod
    async def delete_many(self, memory_ids: List[str], **kwargs) -> int:
        """
        Delete multiple memory items in a batch operation.

        This method efficiently removes multiple memory items in a single
        operation. Useful for cleanup and bulk deletion operations.

        Args:
            memory_ids: List of memory item IDs to delete
            **kwargs: Additional implementation-specific parameters

        Returns:
            Number of items actually deleted

        Raises:
            MemoryError: If batch deletion fails
        """

    # ==========================================
    # Query and Filtering Operations
    # ==========================================

    @abstractmethod
    async def get_all(
        self,
        filters: Optional[MemoryFilter] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: str = "desc",
        **kwargs,
    ) -> List[MemoryItem]:
        """
        Retrieve multiple memory items with optional filtering and pagination.

        This method provides flexible querying capabilities with support for
        filtering, sorting, and pagination. Essential for browsing and
        managing large memory collections.

        Args:
            filters: Optional filter criteria to apply
            limit: Maximum number of items to return
            offset: Number of items to skip (for pagination)
            sort_by: Field to sort by (e.g., 'created_at', 'importance')
            sort_order: Sort order ('asc' or 'desc')
            **kwargs: Additional implementation-specific parameters

        Returns:
            List of memory items matching the criteria

        Raises:
            MemoryError: If query fails due to storage issues
            ValidationError: If filter criteria are invalid
        """

    @abstractmethod
    async def count(self, filters: Optional[MemoryFilter] = None, **kwargs) -> int:
        """
        Count memory items matching the given filters.

        This method provides efficient counting of memory items without
        retrieving the actual data, useful for pagination and analytics.

        Args:
            filters: Optional filter criteria to apply
            **kwargs: Additional implementation-specific parameters

        Returns:
            Number of memory items matching the criteria

        Raises:
            MemoryError: If count operation fails
        """

    # ==========================================
    # Search Operations
    # ==========================================

    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.7,
        memory_types: Optional[List[str]] = None,
        filters: Optional[MemoryFilter] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """
        Perform semantic search across memory items.

        This method enables intelligent search across memory content using
        semantic similarity. The implementation may use vector embeddings,
        full-text search, or hybrid approaches.

        Args:
            query: Search query string
            limit: Maximum number of results to return
            threshold: Minimum relevance score (0.0 to 1.0)
            memory_types: Optional list of memory types to search
            filters: Optional additional filter criteria
            **kwargs: Additional implementation-specific parameters

        Returns:
            List of search results with relevance scores

        Raises:
            MemoryError: If search operation fails
            ValidationError: If search parameters are invalid
        """

    @abstractmethod
    async def similarity_search(
        self,
        reference_items: List[MemoryItem],
        limit: int = 10,
        threshold: float = 0.7,
        filters: Optional[MemoryFilter] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """
        Find memory items similar to a list of reference items.

        This method finds memory items that are semantically similar to
        a list of reference memory items. Useful for finding related
        conversations or duplicate detection.

        Args:
            reference_items: List of reference memory items
            limit: Maximum number of results to return
            threshold: Minimum similarity score (0.0 to 1.0)
            filters: Optional additional filter criteria
            **kwargs: Additional implementation-specific parameters

        Returns:
            List of similar memory items with similarity scores

        Raises:
            MemoryError: If similarity search fails
            ValidationError: If search parameters are invalid
        """

    # ==========================================
    # Memory Management Operations
    # ==========================================

    @abstractmethod
    async def get_stats(self, filters: Optional[MemoryFilter] = None, **kwargs) -> MemoryStats:
        """
        Get statistics about the memory store.

        This method provides insights into memory usage, distribution,
        and performance metrics for monitoring and optimization.

        Args:
            filters: Optional filters to scope statistics
            **kwargs: Additional implementation-specific parameters

        Returns:
            Memory statistics object

        Raises:
            MemoryError: If statistics calculation fails
        """

    @abstractmethod
    async def cleanup_old_memories(
        self,
        older_than: datetime,
        memory_types: Optional[List[str]] = None,
        preserve_important: bool = True,
        dry_run: bool = True,
        **kwargs,
    ) -> int:
        """
        Clean up old memory items based on age and criteria.

        This method provides maintenance functionality to manage
        memory store size and remove outdated information.

        Args:
            older_than: Delete items older than this datetime
            memory_types: Optional list of memory types to clean
            preserve_important: Whether to preserve high-importance items
            dry_run: If True, return count without actually deleting
            **kwargs: Additional implementation-specific parameters

        Returns:
            Number of items that would be/were deleted

        Raises:
            MemoryError: If cleanup operation fails
        """

    # ==========================================
    # Stream Operations (Optional)
    # ==========================================

    async def stream_memories(
        self, filters: Optional[MemoryFilter] = None, chunk_size: int = 100, **kwargs
    ) -> AsyncGenerator[List[MemoryItem], None]:
        """
        Stream memory items in chunks for large dataset processing.

        This method provides efficient streaming access to large memory
        collections without loading everything into memory at once.

        Args:
            filters: Optional filter criteria to apply
            chunk_size: Number of items per chunk
            **kwargs: Additional implementation-specific parameters

        Yields:
            Chunks of memory items

        Raises:
            MemoryError: If streaming fails
        """
        # Default implementation using get_all with pagination
        offset = 0
        while True:
            chunk = await self.get_all(filters=filters, limit=chunk_size, offset=offset, **kwargs)
            if not chunk:
                break
            yield chunk
            if len(chunk) < chunk_size:
                break
            offset += chunk_size

    # ==========================================
    # Context Manager Support
    # ==========================================

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""


class BaseMemoryManager(ABC):
    """
    Abstract base class for high-level memory management.

    This interface defines higher-level operations for memory management,
    including automatic memory extraction, summarization, and intelligent
    retrieval based on context and user patterns.
    """

    @abstractmethod
    async def extract_and_store_conversation(
        self,
        conversation_messages: List[Dict[str, str]],
        conversation_id: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        extract_profiles: bool = True,
        extract_experiences: bool = True,
        **kwargs,
    ) -> Dict[str, List[str]]:
        """
        Extract memories from a conversation and store them.

        This method processes a conversation to extract various types of
        memories including user profiles and agent experiences, then
        stores them in the memory store.

        Args:
            conversation_messages: List of conversation messages
            conversation_id: ID of the conversation
            user_id: Optional user ID
            agent_id: Optional agent ID
            extract_profiles: Whether to extract user profiles
            extract_experiences: Whether to extract agent experiences
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping memory types to lists of created memory IDs

        Raises:
            MemoryError: If extraction or storage fails
        """

    @abstractmethod
    async def get_relevant_context(
        self, query: str, user_id: Optional[str] = None, agent_id: Optional[str] = None, max_items: int = 5, **kwargs
    ) -> List[SearchResult]:
        """
        Get relevant memory context for a query.

        This method intelligently retrieves the most relevant memories
        for a given query, considering user and agent context.

        Args:
            query: Query to find relevant context for
            user_id: Optional user ID for personalization
            agent_id: Optional agent ID for agent-specific context
            max_items: Maximum number of context items to return
            **kwargs: Additional parameters

        Returns:
            List of relevant memory items with relevance scores

        Raises:
            MemoryError: If context retrieval fails
        """

    @abstractmethod
    async def summarize_conversation_history(self, conversation_id: str, max_length: int = 500, **kwargs) -> str:
        """
        Generate a summary of conversation history.

        This method creates a concise summary of a conversation,
        useful for context compression and memory management.

        Args:
            conversation_id: ID of the conversation to summarize
            max_length: Maximum length of summary in characters
            **kwargs: Additional parameters

        Returns:
            Summary text of the conversation

        Raises:
            MemoryError: If summarization fails
        """
