"""
Base classes and exceptions for the GoalithService.

This module contains the foundational classes and exceptions used across
the goalith_service module to avoid circular imports.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from cogents_core.goalith.goalgraph.node import GoalNode


class GoalDecomposer(ABC):
    """
    Abstract interface for goal decomposers.

    Decomposers can be humans, AI agents (LLMs), symbolic planners,
    or any callable entity that can break down goals into subgoals/tasks.
    """

    @abstractmethod
    def decompose(self, goal_node: GoalNode, context: Optional[Dict[str, Any]] = None) -> List[GoalNode]:
        """
        Decompose a goal into subgoals or tasks.

        Args:
            goal_node: The goal node to decompose
            context: Optional context for decomposition

        Returns:
            List of subgoal/task nodes

        Raises:
            DecompositionError: If decomposition fails
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of this decomposer."""

    @property
    def description(self) -> str:
        """Get the description of this decomposer."""
        return f"Decomposer: {self.name}"
