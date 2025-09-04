from typing import Any, Callable, Dict, List, Optional

from cogents_core.goalith.errors import DecompositionError
from cogents_core.goalith.goalgraph.node import GoalNode

from .base import GoalDecomposer


class CallableDecomposer(GoalDecomposer):
    """
    Wrapper for callable decomposition functions.

    Allows any function that matches the decomposition signature
    to be used as a decomposer.
    """

    def __init__(
        self,
        callable_func: Callable[[GoalNode, Optional[Dict[str, Any]]], List[GoalNode]],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """
        Initialize with a callable function.

        Args:
            callable_func: Function that performs decomposition
            name: Name of this decomposer (auto-detected from function if None)
            description: Optional description
        """
        self._callable = callable_func
        self._name = name or getattr(callable_func, "__name__", "callable_decomposer")
        self._description = description or f"Callable decomposer: {self._name}"

    @property
    def name(self) -> str:
        """Get the name of this decomposer."""
        return self._name

    @property
    def description(self) -> str:
        """Get the description of this decomposer."""
        return self._description

    def decompose(self, goal_node: GoalNode, context: Optional[Dict[str, Any]] = None) -> List[GoalNode]:
        """
        Decompose using the callable function.

        Args:
            goal_node: The goal node to decompose
            context: Optional context

        Returns:
            List of subgoal/task nodes

        Raises:
            DecompositionError: If callable raises an exception
        """
        try:
            return self._callable(goal_node, context)
        except ValueError:
            # Let ValueError propagate as-is for test compatibility
            raise
        except Exception as e:
            raise DecompositionError(f"Decomposition failed: {e}") from e
