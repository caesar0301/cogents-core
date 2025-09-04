from abc import ABC, abstractmethod

from cogents_core.goalith.decomposer.base import GoalDecomposer
from cogents_core.goalith.goalgraph.graph import GoalGraph
from cogents_core.goalith.goalgraph.node import GoalNode


class BaseReplanner(ABC):
    """
    Base class for replanners.
    """

    def __init__(self, graph: GoalGraph, decomposer: GoalDecomposer):
        self._goal_graph = graph
        self._decomposer = decomposer

    @property
    def goal_graph(self):
        """Get the graph."""
        return self._goal_graph

    @property
    def decomposer(self):
        """Get the decomposer."""
        return self._decomposer

    @abstractmethod
    def replan(self, node: GoalNode, **kwargs) -> bool:
        """
        Replan a node in the graph.
        """
