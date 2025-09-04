"""
Unit tests for CallableDecomposer class.
"""

import pytest

from cogents_core.goalith.decomposer.callable_decomposer import CallableDecomposer
from cogents_core.goalith.errors import DecompositionError
from cogents_core.goalith.goalgraph.node import GoalNode


class TestCallableDecomposer:
    """Test cases for CallableDecomposer class."""

    def test_callable_decomposer_initialization(self):
        """Test CallableDecomposer initialization."""

        def test_decompose(goal_node, context=None):
            return [GoalNode(description="Test task")]

        decomposer = CallableDecomposer(test_decompose)

        assert decomposer._callable == test_decompose
        assert decomposer._name == "test_decompose"
        assert decomposer._description == "Callable decomposer: test_decompose"

    def test_callable_decomposer_with_custom_name(self):
        """Test CallableDecomposer initialization with custom name."""

        def test_decompose(goal_node, context=None):
            return []

        decomposer = CallableDecomposer(test_decompose, name="custom_name")

        assert decomposer._name == "custom_name"
        assert decomposer._description == "Callable decomposer: custom_name"

    def test_callable_decomposer_with_custom_description(self):
        """Test CallableDecomposer initialization with custom description."""

        def test_decompose(goal_node, context=None):
            return []

        decomposer = CallableDecomposer(test_decompose, name="test_name", description="Custom description")

        assert decomposer._description == "Custom description"

    def test_callable_decomposer_name_property(self):
        """Test the name property of CallableDecomposer."""

        def test_decompose(goal_node, context=None):
            return []

        decomposer = CallableDecomposer(test_decompose, name="test_name")
        assert decomposer.name == "test_name"

    def test_callable_decomposer_description_property(self):
        """Test the description property of CallableDecomposer."""

        def test_decompose(goal_node, context=None):
            return []

        decomposer = CallableDecomposer(test_decompose, name="test_name")
        assert decomposer.description == "Callable decomposer: test_name"

    def test_callable_decomposer_decompose_basic(self):
        """Test basic decomposition functionality."""

        def test_decompose(goal_node, context=None):
            return [GoalNode(description="Task 1"), GoalNode(description="Task 2")]

        decomposer = CallableDecomposer(test_decompose)
        goal_node = GoalNode(description="Parent goal")

        result = decomposer.decompose(goal_node)

        assert len(result) == 2
        assert result[0].description == "Task 1"
        assert result[1].description == "Task 2"

    def test_callable_decomposer_decompose_with_context(self):
        """Test decomposition with context parameter."""

        def test_decompose(goal_node, context=None):
            if context and context.get("create_task"):
                return [GoalNode(description="Context-aware task")]
            return []

        decomposer = CallableDecomposer(test_decompose)
        goal_node = GoalNode(description="Parent goal")

        # Test without context
        result = decomposer.decompose(goal_node)
        assert len(result) == 0

        # Test with context
        context = {"create_task": True}
        result = decomposer.decompose(goal_node, context)
        assert len(result) == 1
        assert result[0].description == "Context-aware task"

    def test_callable_decomposer_decompose_with_none_context(self):
        """Test decomposition with None context."""

        def test_decompose(goal_node, context=None):
            if context is None:
                return [GoalNode(description="None context task")]
            return []

        decomposer = CallableDecomposer(test_decompose)
        goal_node = GoalNode(description="Parent goal")

        result = decomposer.decompose(goal_node, None)
        assert len(result) == 1
        assert result[0].description == "None context task"

    def test_callable_decomposer_decompose_empty_result(self):
        """Test decomposition that returns empty list."""

        def test_decompose(goal_node, context=None):
            return []

        decomposer = CallableDecomposer(test_decompose)
        goal_node = GoalNode(description="Parent goal")

        result = decomposer.decompose(goal_node)
        assert result == []

    def test_callable_decomposer_decompose_with_goal_node_properties(self):
        """Test decomposition that uses goal node properties."""

        def test_decompose(goal_node, context=None):
            return [
                GoalNode(
                    description=f"Task for {goal_node.description}",
                    priority=goal_node.priority * 0.5,
                    context=goal_node.context.copy() if goal_node.context else {},
                )
            ]

        decomposer = CallableDecomposer(test_decompose)
        goal_node = GoalNode(description="Parent goal", priority=2.0, context={"key": "value"})

        result = decomposer.decompose(goal_node)

        assert len(result) == 1
        assert result[0].description == "Task for Parent goal"
        assert result[0].priority == 1.0
        assert result[0].context == {"key": "value"}

    def test_callable_decomposer_decompose_with_lambda(self):
        """Test decomposition with lambda function."""
        decomposer = CallableDecomposer(lambda goal_node, context=None: [GoalNode(description="Lambda task")])
        goal_node = GoalNode(description="Parent goal")

        result = decomposer.decompose(goal_node)

        assert len(result) == 1
        assert result[0].description == "Lambda task"

    def test_callable_decomposer_decompose_with_function_without_name(self):
        """Test decomposition with function that has no __name__ attribute."""

        # Create a function without __name__ attribute
        def test_decompose(goal_node, context=None):
            return [GoalNode(description="No name task")]

        # Remove __name__ attribute by creating a mock object
        class MockCallable:
            def __call__(self, goal_node, context=None):
                return [GoalNode(description="No name task")]

        mock_callable = MockCallable()
        # The callable doesn't have a __name__ attribute

        decomposer = CallableDecomposer(mock_callable)
        assert decomposer._name == "callable_decomposer"

    def test_callable_decomposer_decompose_raises_value_error(self):
        """Test that ValueError is propagated as-is."""

        def test_decompose(goal_node, context=None):
            raise ValueError("Test value error")

        decomposer = CallableDecomposer(test_decompose)
        goal_node = GoalNode(description="Parent goal")

        with pytest.raises(ValueError, match="Test value error"):
            decomposer.decompose(goal_node)

    def test_callable_decomposer_decompose_raises_general_exception(self):
        """Test that general exceptions are wrapped in DecompositionError."""

        def test_decompose(goal_node, context=None):
            raise RuntimeError("Test runtime error")

        decomposer = CallableDecomposer(test_decompose)
        goal_node = GoalNode(description="Parent goal")

        with pytest.raises(DecompositionError, match="Decomposition failed: Test runtime error"):
            decomposer.decompose(goal_node)

    def test_callable_decomposer_decompose_raises_type_error(self):
        """Test that TypeError is wrapped in DecompositionError."""

        def test_decompose(goal_node, context=None):
            raise TypeError("Test type error")

        decomposer = CallableDecomposer(test_decompose)
        goal_node = GoalNode(description="Parent goal")

        with pytest.raises(DecompositionError, match="Decomposition failed: Test type error"):
            decomposer.decompose(goal_node)

    def test_callable_decomposer_decompose_raises_attribute_error(self):
        """Test that AttributeError is wrapped in DecompositionError."""

        def test_decompose(goal_node, context=None):
            raise AttributeError("Test attribute error")

        decomposer = CallableDecomposer(test_decompose)
        goal_node = GoalNode(description="Parent goal")

        with pytest.raises(DecompositionError, match="Decomposition failed: Test attribute error"):
            decomposer.decompose(goal_node)

    def test_callable_decomposer_decompose_with_complex_return_value(self):
        """Test decomposition that returns complex GoalNode objects."""

        def test_decompose(goal_node, context=None):
            return [
                GoalNode(
                    description="Complex task 1",
                    priority=1.5,
                    context={"task_type": "complex"},
                    tags=["complex", "task1"],
                    estimated_effort="2 hours",
                ),
                GoalNode(
                    description="Complex task 2",
                    priority=2.5,
                    context={"task_type": "complex"},
                    tags=["complex", "task2"],
                    estimated_effort="3 hours",
                ),
            ]

        decomposer = CallableDecomposer(test_decompose)
        goal_node = GoalNode(description="Parent goal")

        result = decomposer.decompose(goal_node)

        assert len(result) == 2

        # Check first task
        assert result[0].description == "Complex task 1"
        assert result[0].priority == 1.5
        assert result[0].context == {"task_type": "complex"}
        assert result[0].tags == ["complex", "task1"]
        assert result[0].estimated_effort == "2 hours"

        # Check second task
        assert result[1].description == "Complex task 2"
        assert result[1].priority == 2.5
        assert result[1].context == {"task_type": "complex"}
        assert result[1].tags == ["complex", "task2"]
        assert result[1].estimated_effort == "3 hours"

    def test_callable_decomposer_decompose_with_conditional_logic(self):
        """Test decomposition with conditional logic based on goal node."""

        def test_decompose(goal_node, context=None):
            if "urgent" in goal_node.description.lower():
                return [GoalNode(description="Urgent task 1"), GoalNode(description="Urgent task 2")]
            elif "simple" in goal_node.description.lower():
                return [GoalNode(description="Simple task")]
            else:
                return []

        decomposer = CallableDecomposer(test_decompose)

        # Test urgent goal
        urgent_goal = GoalNode(description="Urgent goal")
        result = decomposer.decompose(urgent_goal)
        assert len(result) == 2
        assert result[0].description == "Urgent task 1"
        assert result[1].description == "Urgent task 2"

        # Test simple goal
        simple_goal = GoalNode(description="Simple goal")
        result = decomposer.decompose(simple_goal)
        assert len(result) == 1
        assert result[0].description == "Simple task"

        # Test other goal
        other_goal = GoalNode(description="Other goal")
        result = decomposer.decompose(other_goal)
        assert result == []

    def test_callable_decomposer_inherits_goal_decomposer(self):
        """Test that CallableDecomposer inherits from GoalDecomposer."""
        from cogents_core.goalith.decomposer.base import GoalDecomposer

        def test_decompose(goal_node, context=None):
            return []

        decomposer = CallableDecomposer(test_decompose)
        assert isinstance(decomposer, GoalDecomposer)

    def test_callable_decomposer_implements_required_methods(self):
        """Test that CallableDecomposer implements all required methods."""

        def test_decompose(goal_node, context=None):
            return []

        decomposer = CallableDecomposer(test_decompose)

        # Should have name property
        assert hasattr(decomposer, "name")
        assert decomposer.name == "test_decompose"

        # Should have description property
        assert hasattr(decomposer, "description")
        assert decomposer.description == "Callable decomposer: test_decompose"

        # Should have decompose method
        assert hasattr(decomposer, "decompose")
        assert callable(decomposer.decompose)

    def test_callable_decomposer_with_class_method(self):
        """Test CallableDecomposer with a class method."""

        class TestClass:
            @staticmethod
            def static_decompose(goal_node, context=None):
                return [GoalNode(description="Static method task")]

            @classmethod
            def class_decompose(cls, goal_node, context=None):
                return [GoalNode(description="Class method task")]

        # Test with static method
        static_decomposer = CallableDecomposer(TestClass.static_decompose)
        goal_node = GoalNode(description="Parent goal")
        result = static_decomposer.decompose(goal_node)
        assert len(result) == 1
        assert result[0].description == "Static method task"

        # Test with class method
        class_decomposer = CallableDecomposer(TestClass.class_decompose)
        result = class_decomposer.decompose(goal_node)
        assert len(result) == 1
        assert result[0].description == "Class method task"
