"""
Unit tests for base GoalDecomposer class.
"""

from abc import ABC

import pytest

from cogents.core.goalith.decomposer.base import GoalDecomposer
from cogents.core.goalith.goalgraph.node import GoalNode


class TestGoalDecomposer:
    """Test cases for GoalDecomposer abstract class."""

    def test_goal_decomposer_is_abstract(self):
        """Test that GoalDecomposer is an abstract base class."""
        assert issubclass(GoalDecomposer, ABC)

    def test_goal_decomposer_has_abstract_methods(self):
        """Test that GoalDecomposer has the required abstract methods."""
        # Check that decompose method exists and is abstract
        assert hasattr(GoalDecomposer, "decompose")

        # Check that name property exists and is abstract
        assert hasattr(GoalDecomposer, "name")

        # Check that description property exists (not abstract)
        assert hasattr(GoalDecomposer, "description")

    def test_goal_decomposer_description_property(self):
        """Test the description property of GoalDecomposer."""

        # Create a concrete implementation for testing
        class TestDecomposer(GoalDecomposer):
            @property
            def name(self) -> str:
                return "test_decomposer"

            def decompose(self, goal_node, context=None):
                return []

        decomposer = TestDecomposer()
        assert decomposer.description == "Decomposer: test_decomposer"

    def test_goal_decomposer_interface_contract(self):
        """Test that the interface contract is properly defined."""
        # The abstract methods should be defined with the correct signatures
        import inspect

        # Check decompose method signature
        decompose_sig = inspect.signature(GoalDecomposer.decompose)
        params = list(decompose_sig.parameters.keys())
        assert params == ["self", "goal_node", "context"]

        # Check that context has a default value
        assert decompose_sig.parameters["context"].default is None

    def test_goal_decomposer_cannot_be_instantiated(self):
        """Test that GoalDecomposer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            GoalDecomposer()

    def test_concrete_decomposer_implementation(self):
        """Test that a concrete implementation works correctly."""

        class ConcreteDecomposer(GoalDecomposer):
            def __init__(self, name: str = "concrete"):
                self._name = name

            @property
            def name(self) -> str:
                return self._name

            def decompose(self, goal_node: GoalNode, context=None):
                return []

        decomposer = ConcreteDecomposer("test")
        assert decomposer.name == "test"
        assert decomposer.description == "Decomposer: test"

        # Test decompose method
        goal_node = GoalNode(description="Test goal")
        result = decomposer.decompose(goal_node)
        assert result == []

    def test_decomposer_with_context(self):
        """Test that decomposer can handle context parameter."""

        class ContextAwareDecomposer(GoalDecomposer):
            @property
            def name(self) -> str:
                return "context_aware"

            def decompose(self, goal_node: GoalNode, context=None):
                if context and context.get("test_key") == "test_value":
                    return [GoalNode(description="Context-aware subtask")]
                return []

        decomposer = ContextAwareDecomposer()
        goal_node = GoalNode(description="Test goal")

        # Test without context
        result = decomposer.decompose(goal_node)
        assert result == []

        # Test with context
        context = {"test_key": "test_value"}
        result = decomposer.decompose(goal_node, context)
        assert len(result) == 1
        assert result[0].description == "Context-aware subtask"
