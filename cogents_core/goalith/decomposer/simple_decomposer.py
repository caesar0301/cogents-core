from typing import Any, Dict, List, Optional

from cogents_core.goalith.goalgraph.node import GoalNode

from .base import GoalDecomposer


class SimpleListDecomposer(GoalDecomposer):
    """
    Simple decomposer that takes a list of subtask descriptions.

    Useful for manual decomposition or simple cases.
    """

    def __init__(self, subtasks: List[str], name: str = "simple_list"):
        """
        Initialize with a list of subtask descriptions.

        Args:
            subtasks: List of subtask descriptions
            name: Name of this decomposer instance
        """
        self._subtasks = subtasks
        self._name = name

    @property
    def name(self) -> str:
        """Get the name of this decomposer."""
        return self._name

    def decompose(self, goal_node: GoalNode, context: Optional[Dict[str, Any]] = None) -> List[GoalNode]:
        """
        Decompose goal into the predefined subtasks.

        Args:
            goal_node: The goal node to decompose
            context: Optional context (unused)

        Returns:
            List of task nodes
        """
        import copy

        nodes = []
        for i, subtask_desc in enumerate(self._subtasks):
            # Deep copy the context to avoid shared references
            context_copy = copy.deepcopy(goal_node.context) if goal_node.context else {}

            task_node = GoalNode(
                description=subtask_desc,
                parent=goal_node.id,
                priority=goal_node.priority,
                context=context_copy,
                tags=goal_node.tags.copy() if goal_node.tags else [],
                estimated_effort=goal_node.estimated_effort,
                assigned_to=goal_node.assigned_to,
                decomposer_name=self.name,
            )
            nodes.append(task_node)

        return nodes
