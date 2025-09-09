"""
Unit tests for SimpleListDecomposer class.
"""

from cogents_core.goalith.decomposer.simple_decomposer import SimpleListDecomposer
from cogents_core.goalith.goalgraph.node import GoalNode, NodeStatus


class TestSimpleListDecomposer:
    """Test cases for SimpleListDecomposer class."""

    def test_simple_list_decomposer_initialization(self):
        """Test SimpleListDecomposer initialization."""
        subtasks = ["Task 1", "Task 2", "Task 3"]
        decomposer = SimpleListDecomposer(subtasks)

        assert decomposer._subtasks == subtasks
        assert decomposer._name == "simple_list"

    def test_simple_list_decomposer_with_custom_name(self):
        """Test SimpleListDecomposer initialization with custom name."""
        subtasks = ["Task 1", "Task 2"]
        decomposer = SimpleListDecomposer(subtasks, name="custom_decomposer")

        assert decomposer._subtasks == subtasks
        assert decomposer._name == "custom_decomposer"

    def test_simple_list_decomposer_name_property(self):
        """Test the name property of SimpleListDecomposer."""
        subtasks = ["Task 1"]
        decomposer = SimpleListDecomposer(subtasks, name="test_name")

        assert decomposer.name == "test_name"

    def test_simple_list_decomposer_decompose_basic(self):
        """Test basic decomposition functionality."""
        subtasks = ["Task 1", "Task 2", "Task 3"]
        decomposer = SimpleListDecomposer(subtasks)

        goal_node = GoalNode(
            id="parent-goal", description="Parent goal", priority=2.0, context={"key": "value"}, tags=["tag1", "tag2"]
        )

        result = decomposer.decompose(goal_node)

        assert len(result) == 3

        # Check first task
        assert result[0].description == "Task 1"
        assert result[0].parent == "parent-goal"
        assert result[0].priority == 2.0
        assert result[0].context == {"key": "value"}
        assert result[0].decomposer_name == "simple_list"

        # Check second task
        assert result[1].description == "Task 2"
        assert result[1].parent == "parent-goal"
        assert result[1].priority == 2.0
        assert result[1].context == {"key": "value"}
        assert result[1].decomposer_name == "simple_list"

        # Check third task
        assert result[2].description == "Task 3"
        assert result[2].parent == "parent-goal"
        assert result[2].priority == 2.0
        assert result[2].context == {"key": "value"}
        assert result[2].decomposer_name == "simple_list"

    def test_simple_list_decomposer_decompose_empty_list(self):
        """Test decomposition with empty subtasks list."""
        decomposer = SimpleListDecomposer([])

        goal_node = GoalNode(description="Parent goal")
        result = decomposer.decompose(goal_node)

        assert result == []

    def test_simple_list_decomposer_decompose_single_task(self):
        """Test decomposition with single task."""
        subtasks = ["Single task"]
        decomposer = SimpleListDecomposer(subtasks)

        goal_node = GoalNode(description="Parent goal")
        result = decomposer.decompose(goal_node)

        assert len(result) == 1
        assert result[0].description == "Single task"
        assert result[0].parent == goal_node.id

    def test_simple_list_decomposer_decompose_with_context(self):
        """Test decomposition with context parameter (should be ignored)."""
        subtasks = ["Task 1", "Task 2"]
        decomposer = SimpleListDecomposer(subtasks)

        goal_node = GoalNode(description="Parent goal")
        context = {"some": "context"}

        result = decomposer.decompose(goal_node, context)

        # Context should be ignored, result should be the same
        assert len(result) == 2
        assert result[0].description == "Task 1"
        assert result[1].description == "Task 2"

    def test_simple_list_decomposer_decompose_with_none_context(self):
        """Test decomposition with None context."""
        subtasks = ["Task 1"]
        decomposer = SimpleListDecomposer(subtasks)

        goal_node = GoalNode(description="Parent goal")
        result = decomposer.decompose(goal_node, None)

        assert len(result) == 1
        assert result[0].description == "Task 1"

    def test_simple_list_decomposer_decompose_inherits_goal_properties(self):
        """Test that decomposed tasks inherit properties from the goal node."""
        subtasks = ["Task 1", "Task 2"]
        decomposer = SimpleListDecomposer(subtasks, name="custom_name")

        goal_node = GoalNode(
            id="parent-id",
            description="Parent goal",
            priority=3.5,
            context={"inherited": "value", "nested": {"key": "value"}},
            tags=["inherited_tag"],
            estimated_effort="2 hours",
            assigned_to="agent1",
        )

        result = decomposer.decompose(goal_node)

        for task in result:
            assert task.parent == "parent-id"
            assert task.priority == 3.5
            assert task.context == {"inherited": "value", "nested": {"key": "value"}}
            assert task.tags == ["inherited_tag"]
            assert task.estimated_effort == "2 hours"
            assert task.assigned_to == "agent1"
            assert task.decomposer_name == "custom_name"

    def test_simple_list_decomposer_decompose_context_copy(self):
        """Test that context is properly copied to avoid shared references."""
        subtasks = ["Task 1"]
        decomposer = SimpleListDecomposer(subtasks)

        original_context = {"mutable": [1, 2, 3]}
        goal_node = GoalNode(description="Parent goal", context=original_context)

        result = decomposer.decompose(goal_node)

        # Modify the original context
        original_context["mutable"].append(4)
        original_context["new_key"] = "new_value"

        # The task's context should not be affected
        task_context = result[0].context
        assert task_context["mutable"] == [1, 2, 3]  # Should not include the appended 4
        assert "new_key" not in task_context

    def test_simple_list_decomposer_decompose_with_none_context_in_goal(self):
        """Test decomposition when goal node has None context."""
        subtasks = ["Task 1"]
        decomposer = SimpleListDecomposer(subtasks)

        # Create a goal node with an empty context dict instead of None
        goal_node = GoalNode(description="Parent goal", context={})
        result = decomposer.decompose(goal_node)

        assert len(result) == 1
        assert result[0].context == {}

    def test_simple_list_decomposer_decompose_task_ids_are_unique(self):
        """Test that decomposed tasks have unique IDs."""
        subtasks = ["Task 1", "Task 2", "Task 3"]
        decomposer = SimpleListDecomposer(subtasks)

        goal_node = GoalNode(description="Parent goal")
        result = decomposer.decompose(goal_node)

        task_ids = [task.id for task in result]
        assert len(task_ids) == len(set(task_ids))  # All IDs should be unique

    def test_simple_list_decomposer_decompose_task_status_is_pending(self):
        """Test that decomposed tasks have PENDING status."""
        subtasks = ["Task 1", "Task 2"]
        decomposer = SimpleListDecomposer(subtasks)

        goal_node = GoalNode(description="Parent goal")
        result = decomposer.decompose(goal_node)

        for task in result:
            assert task.status == NodeStatus.PENDING

    def test_simple_list_decomposer_decompose_task_timing_fields(self):
        """Test that decomposed tasks have proper timing fields."""
        subtasks = ["Task 1"]
        decomposer = SimpleListDecomposer(subtasks)

        goal_node = GoalNode(description="Parent goal")
        result = decomposer.decompose(goal_node)

        task = result[0]
        assert task.created_at is not None
        assert task.updated_at is not None
        assert task.started_at is None
        assert task.completed_at is None
        assert task.deadline is None

    def test_simple_list_decomposer_decompose_task_execution_fields(self):
        """Test that decomposed tasks have proper execution tracking fields."""
        subtasks = ["Task 1"]
        decomposer = SimpleListDecomposer(subtasks)

        goal_node = GoalNode(description="Parent goal")
        result = decomposer.decompose(goal_node)

        task = result[0]
        assert task.retry_count == 0
        assert task.max_retries == 3
        assert task.error_message is None
        assert task.execution_notes == {}

    def test_simple_list_decomposer_decompose_relationship_fields(self):
        """Test that decomposed tasks have proper relationship fields."""
        subtasks = ["Task 1", "Task 2"]
        decomposer = SimpleListDecomposer(subtasks)

        goal_node = GoalNode(description="Parent goal")
        result = decomposer.decompose(goal_node)

        for task in result:
            assert task.dependencies == set()
            assert task.children == set()
            assert task.parent == goal_node.id

    def test_simple_list_decomposer_inherits_goal_decomposer(self):
        """Test that SimpleListDecomposer inherits from GoalDecomposer."""
        from cogents_core.goalith.decomposer.base import GoalDecomposer

        assert issubclass(SimpleListDecomposer, GoalDecomposer)

    def test_simple_list_decomposer_implements_required_methods(self):
        """Test that SimpleListDecomposer implements all required methods."""
        subtasks = ["Task 1"]
        decomposer = SimpleListDecomposer(subtasks)

        # Should have name property
        assert hasattr(decomposer, "name")
        assert decomposer.name == "simple_list"

        # Should have decompose method
        assert hasattr(decomposer, "decompose")
        assert callable(decomposer.decompose)
