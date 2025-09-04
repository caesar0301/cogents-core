from cogents_core.goalith.decomposer.base import GoalDecomposer
from cogents_core.goalith.goalgraph.graph import GoalGraph
from cogents_core.goalith.goalgraph.node import GoalNode

from .base import BaseReplanner


class Replanner(BaseReplanner):
    """
    Handles replanning operations when triggers fire.

    Hooks into decomposition and scheduling modules to adjust plans.
    """

    def __init__(self, graph: GoalGraph, decomposer: GoalDecomposer, **kwargs):
        """
        Initialize replanner.

        Args:
            graph_store: The graph store
            decomposer: The decomposer to use
        """
        super().__init__(graph, decomposer, **kwargs)

    def replan(self, node: GoalNode, **kwargs) -> bool:
        """
        Replan a node in the graph store.

        Args:
            node: The node to replan
            **kwargs: Additional arguments

        Returns:
            True if replanning was successful, False otherwise
        """
        raise NotImplementedError("Replanner is an abstract class and cannot be instantiated directly.")
