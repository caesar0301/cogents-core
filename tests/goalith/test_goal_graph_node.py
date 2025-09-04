"""
Unit tests for GoalNode class.
"""

from datetime import datetime, timezone
from unittest.mock import patch

from cogents_core.goalith.goalgraph.node import GoalNode, NodeStatus


class TestGoalNode:
    """Test cases for GoalNode class."""

    def test_goal_node_creation(self):
        """Test basic GoalNode creation."""
        node = GoalNode(description="Test goal")

        assert node.description == "Test goal"
        assert node.status == NodeStatus.PENDING
        assert node.priority == 1.0
        assert node.id is not None
        assert len(node.id) > 0
        assert node.dependencies == set()
        assert node.children == set()
        assert node.parent is None
        assert node.context == {}
        assert node.tags == []
        assert node.decomposer_name is None
        assert node.estimated_effort is None
        assert node.assigned_to is None
        assert node.retry_count == 0
        assert node.max_retries == 3
        assert node.error_message is None
        assert node.execution_notes == {}

    def test_goal_node_with_all_parameters(self):
        """Test GoalNode creation with all parameters."""
        node = GoalNode(
            id="test-id",
            description="Test goal",
            status=NodeStatus.IN_PROGRESS,
            priority=2.5,
            decomposer_name="test_decomposer",
            context={"key": "value"},
            tags=["tag1", "tag2"],
            estimated_effort="2 hours",
            dependencies={"dep1", "dep2"},
            children={"child1"},
            parent="parent1",
            assigned_to="agent1",
            retry_count=1,
            max_retries=5,
            error_message="Test error",
            execution_notes={"note1": "value1"},
        )

        assert node.id == "test-id"
        assert node.description == "Test goal"
        assert node.status == NodeStatus.IN_PROGRESS
        assert node.priority == 2.5
        assert node.decomposer_name == "test_decomposer"
        assert node.context == {"key": "value"}
        assert node.tags == ["tag1", "tag2"]
        assert node.estimated_effort == "2 hours"
        assert node.dependencies == {"dep1", "dep2"}
        assert node.children == {"child1"}
        assert node.parent == "parent1"
        assert node.assigned_to == "agent1"
        assert node.retry_count == 1
        assert node.max_retries == 5
        assert node.error_message == "Test error"
        assert node.execution_notes == {"note1": "value1"}

    def test_is_ready(self):
        """Test is_ready method."""
        # Pending node should be ready
        node = GoalNode(description="Test goal")
        assert node.is_ready() is True

        # Other statuses should not be ready
        node.status = NodeStatus.IN_PROGRESS
        assert node.is_ready() is False

        node.status = NodeStatus.COMPLETED
        assert node.is_ready() is False

        node.status = NodeStatus.FAILED
        assert node.is_ready() is False

        node.status = NodeStatus.CANCELLED
        assert node.is_ready() is False

        node.status = NodeStatus.BLOCKED
        assert node.is_ready() is False

    def test_is_terminal(self):
        """Test is_terminal method."""
        # Terminal statuses
        node = GoalNode(description="Test goal")

        node.status = NodeStatus.COMPLETED
        assert node.is_terminal() is True

        node.status = NodeStatus.FAILED
        assert node.is_terminal() is True

        node.status = NodeStatus.CANCELLED
        assert node.is_terminal() is True

        # Non-terminal statuses
        node.status = NodeStatus.PENDING
        assert node.is_terminal() is False

        node.status = NodeStatus.IN_PROGRESS
        assert node.is_terminal() is False

        node.status = NodeStatus.BLOCKED
        assert node.is_terminal() is False

    def test_can_retry(self):
        """Test can_retry method."""
        node = GoalNode(description="Test goal")

        # Failed node with retries remaining should be retryable
        node.status = NodeStatus.FAILED
        node.retry_count = 0
        node.max_retries = 3
        assert node.can_retry() is True

        # Failed node with no retries remaining should not be retryable
        node.retry_count = 3
        assert node.can_retry() is False

        # Non-failed nodes should not be retryable
        node.status = NodeStatus.PENDING
        assert node.can_retry() is False

        node.status = NodeStatus.IN_PROGRESS
        assert node.can_retry() is False

        node.status = NodeStatus.COMPLETED
        assert node.can_retry() is False

    @patch("cogents_core.goalith.goalgraph.node.datetime")
    def test_mark_started(self, mock_datetime):
        """Test mark_started method."""
        mock_now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = mock_now

        node = GoalNode(description="Test goal")
        node.mark_started()

        assert node.status == NodeStatus.IN_PROGRESS
        assert node.started_at == mock_now
        assert node.updated_at == mock_now

    @patch("cogents_core.goalith.goalgraph.node.datetime")
    def test_mark_completed(self, mock_datetime):
        """Test mark_completed method."""
        mock_now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = mock_now

        node = GoalNode(description="Test goal")
        node.mark_completed()

        assert node.status == NodeStatus.COMPLETED
        assert node.completed_at == mock_now
        assert node.updated_at == mock_now

    @patch("cogents_core.goalith.goalgraph.node.datetime")
    def test_mark_failed(self, mock_datetime):
        """Test mark_failed method."""
        mock_now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = mock_now

        node = GoalNode(description="Test goal")
        node.retry_count = 1

        # Test with error message
        node.mark_failed("Test error")

        assert node.status == NodeStatus.FAILED
        assert node.retry_count == 2
        assert node.error_message == "Test error"
        assert node.updated_at == mock_now

        # Test without error message
        node.mark_failed()

        assert node.status == NodeStatus.FAILED
        assert node.retry_count == 3
        assert node.error_message == "Test error"  # Should not change
        assert node.updated_at == mock_now

    @patch("cogents_core.goalith.goalgraph.node.datetime")
    def test_mark_cancelled(self, mock_datetime):
        """Test mark_cancelled method."""
        mock_now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = mock_now

        node = GoalNode(description="Test goal")
        node.mark_cancelled()

        assert node.status == NodeStatus.CANCELLED
        assert node.updated_at == mock_now

    @patch("cogents_core.goalith.goalgraph.node.datetime")
    def test_add_note(self, mock_datetime):
        """Test add_note method."""
        mock_now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = mock_now

        node = GoalNode(description="Test goal")
        node.add_note("Test note")

        expected_key = mock_now.isoformat()
        assert node.execution_notes[expected_key] == "Test note"
        assert node.updated_at == mock_now

    @patch("cogents_core.goalith.goalgraph.node.datetime")
    def test_update_context(self, mock_datetime):
        """Test update_context method."""
        mock_now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = mock_now

        node = GoalNode(description="Test goal")
        node.update_context("key", "value")

        assert node.context["key"] == "value"
        assert node.updated_at == mock_now

    @patch("cogents_core.goalith.goalgraph.node.datetime")
    def test_add_dependency(self, mock_datetime):
        """Test add_dependency method."""
        mock_now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = mock_now

        node = GoalNode(description="Test goal")
        node.add_dependency("dep1")

        assert "dep1" in node.dependencies
        assert node.updated_at == mock_now

    @patch("cogents_core.goalith.goalgraph.node.datetime")
    def test_remove_dependency(self, mock_datetime):
        """Test remove_dependency method."""
        mock_now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = mock_now

        node = GoalNode(description="Test goal")
        node.dependencies = {"dep1", "dep2"}

        node.remove_dependency("dep1")

        assert "dep1" not in node.dependencies
        assert "dep2" in node.dependencies
        assert node.updated_at == mock_now

    @patch("cogents_core.goalith.goalgraph.node.datetime")
    def test_add_child(self, mock_datetime):
        """Test add_child method."""
        mock_now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = mock_now

        node = GoalNode(description="Test goal")
        node.add_child("child1")

        assert "child1" in node.children
        assert node.updated_at == mock_now

    @patch("cogents_core.goalith.goalgraph.node.datetime")
    def test_remove_child(self, mock_datetime):
        """Test remove_child method."""
        mock_now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = mock_now

        node = GoalNode(description="Test goal")
        node.children = {"child1", "child2"}

        node.remove_child("child1")

        assert "child1" not in node.children
        assert "child2" in node.children
        assert node.updated_at == mock_now

    def test_hash(self):
        """Test __hash__ method."""
        node1 = GoalNode(id="test-id", description="Test goal")
        node2 = GoalNode(id="test-id", description="Different description")

        # Same ID should have same hash
        assert hash(node1) == hash(node2)

        # Different ID should have different hash
        node3 = GoalNode(id="different-id", description="Test goal")
        assert hash(node1) != hash(node3)

    def test_equality(self):
        """Test __eq__ method."""
        # Create two nodes with same content
        node1 = GoalNode(
            id="test-id",
            description="Test goal",
            status=NodeStatus.PENDING,
            priority=1.0,
            context={"key": "value"},
            tags=["tag1"],
            dependencies={"dep1"},
            children={"child1"},
            parent="parent1",
        )

        node2 = GoalNode(
            id="test-id",
            description="Test goal",
            status=NodeStatus.PENDING,
            priority=1.0,
            context={"key": "value"},
            tags=["tag1"],
            dependencies={"dep1"},
            children={"child1"},
            parent="parent1",
        )

        assert node1 == node2

        # Different ID should not be equal
        node3 = GoalNode(id="different-id", description="Test goal", status=NodeStatus.PENDING, priority=1.0)
        assert node1 != node3

        # Different description should not be equal
        node4 = GoalNode(id="test-id", description="Different goal", status=NodeStatus.PENDING, priority=1.0)
        assert node1 != node4

        # Different type should not be equal
        assert node1 != "not a node"

    def test_to_dict(self):
        """Test to_dict method."""
        node = GoalNode(
            id="test-id",
            description="Test goal",
            status=NodeStatus.PENDING,
            priority=1.0,
            context={"key": "value"},
            tags=["tag1"],
            dependencies={"dep1"},
            children={"child1"},
            parent="parent1",
        )

        node_dict = node.to_dict()

        assert node_dict["id"] == "test-id"
        assert node_dict["description"] == "Test goal"
        assert node_dict["status"] == "pending"
        assert node_dict["priority"] == 1.0
        assert node_dict["context"] == {"key": "value"}
        assert node_dict["tags"] == ["tag1"]
        assert node_dict["dependencies"] == {"dep1"}
        assert node_dict["children"] == {"child1"}
        assert node_dict["parent"] == "parent1"

    def test_from_dict(self):
        """Test from_dict method."""
        node_data = {
            "id": "test-id",
            "description": "Test goal",
            "status": "pending",
            "priority": 1.0,
            "context": {"key": "value"},
            "tags": ["tag1"],
            "dependencies": {"dep1"},
            "children": {"child1"},
            "parent": "parent1",
        }

        node = GoalNode.from_dict(node_data)

        assert node.id == "test-id"
        assert node.description == "Test goal"
        assert node.status == NodeStatus.PENDING
        assert node.priority == 1.0
        assert node.context == {"key": "value"}
        assert node.tags == ["tag1"]
        assert node.dependencies == {"dep1"}
        assert node.children == {"child1"}
        assert node.parent == "parent1"
