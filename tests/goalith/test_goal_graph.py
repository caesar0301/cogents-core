"""
Unit tests for GoalGraph class.
"""

import tempfile
from pathlib import Path

import pytest

from cogents_core.goalith.errors import CycleDetectedError, NodeNotFoundError
from cogents_core.goalith.goalgraph.graph import GoalGraph
from cogents_core.goalith.goalgraph.node import GoalNode, NodeStatus


class TestGoalGraph:
    """Test cases for GoalGraph class."""

    def test_goal_graph_initialization(self):
        """Test GoalGraph initialization."""
        graph = GoalGraph()

        assert graph._graph is not None
        assert graph._nodes == {}
        assert len(graph._graph.nodes) == 0
        assert len(graph._graph.edges) == 0

    def test_add_node(self):
        """Test adding a node to the graph."""
        graph = GoalGraph()
        node = GoalNode(description="Test goal")

        graph.add_node(node)

        assert node.id in graph._nodes
        assert graph._nodes[node.id] == node
        assert node.id in graph._graph.nodes

    def test_add_duplicate_node(self):
        """Test adding a duplicate node raises ValueError."""
        graph = GoalGraph()
        node = GoalNode(id="test-id", description="Test goal")

        graph.add_node(node)

        with pytest.raises(ValueError, match="Node test-id already exists"):
            graph.add_node(node)

    def test_get_node(self):
        """Test getting a node by ID."""
        graph = GoalGraph()
        node = GoalNode(id="test-id", description="Test goal")

        graph.add_node(node)
        retrieved_node = graph.get_node("test-id")

        assert retrieved_node == node

    def test_get_nonexistent_node(self):
        """Test getting a nonexistent node raises NodeNotFoundError."""
        graph = GoalGraph()

        with pytest.raises(NodeNotFoundError, match="Node nonexistent not found"):
            graph.get_node("nonexistent")

    def test_update_node(self):
        """Test updating an existing node."""
        graph = GoalGraph()
        node = GoalNode(id="test-id", description="Test goal")

        graph.add_node(node)

        # Update the node
        node.description = "Updated goal"
        node.status = NodeStatus.IN_PROGRESS
        graph.update_node(node)

        retrieved_node = graph.get_node("test-id")
        assert retrieved_node.description == "Updated goal"
        assert retrieved_node.status == NodeStatus.IN_PROGRESS

    def test_update_nonexistent_node(self):
        """Test updating a nonexistent node raises NodeNotFoundError."""
        graph = GoalGraph()
        node = GoalNode(id="test-id", description="Test goal")

        with pytest.raises(NodeNotFoundError, match="Node test-id not found"):
            graph.update_node(node)

    def test_remove_node(self):
        """Test removing a node from the graph."""
        graph = GoalGraph()
        node = GoalNode(id="test-id", description="Test goal")

        graph.add_node(node)
        graph.remove_node("test-id")

        assert "test-id" not in graph._nodes
        assert "test-id" not in graph._graph.nodes

    def test_remove_nonexistent_node(self):
        """Test removing a nonexistent node raises NodeNotFoundError."""
        graph = GoalGraph()

        with pytest.raises(NodeNotFoundError, match="Node nonexistent not found"):
            graph.remove_node("nonexistent")

    def test_remove_node_updates_relationships(self):
        """Test that removing a node updates relationships in other nodes."""
        graph = GoalGraph()

        # Create nodes
        parent = GoalNode(id="parent", description="Parent")
        child = GoalNode(id="child", description="Child")
        dep = GoalNode(id="dep", description="Dependency")

        graph.add_node(parent)
        graph.add_node(child)
        graph.add_node(dep)

        # Set up relationships
        graph.add_parent_child("parent", "child")
        graph.add_dependency("child", "dep")

        # Remove parent
        graph.remove_node("parent")

        # Check that child's parent is cleared
        child_node = graph.get_node("child")
        assert child_node.parent is None

        # Check that dep's children is updated
        dep_node = graph.get_node("dep")
        assert "child" in dep_node.children  # This should still exist as it's a dependency

    def test_add_dependency(self):
        """Test adding a dependency between nodes."""
        graph = GoalGraph()

        dep = GoalNode(id="dep", description="Dependency")
        dependent = GoalNode(id="dependent", description="Dependent")

        graph.add_node(dep)
        graph.add_node(dependent)

        graph.add_dependency("dependent", "dep")

        # Check that the edge was added to the graph
        assert graph._graph.has_edge("dep", "dependent")

        # Check that the dependency was added to the dependent node
        dependent_node = graph.get_node("dependent")
        assert "dep" in dependent_node.dependencies

        # Check that the child was added to the dependency node
        dep_node = graph.get_node("dep")
        assert "dependent" in dep_node.children

    def test_add_dependency_nonexistent_nodes(self):
        """Test adding dependency with nonexistent nodes raises NodeNotFoundError."""
        graph = GoalGraph()

        with pytest.raises(NodeNotFoundError, match="Node dependent not found"):
            graph.add_dependency("dependent", "dep")

        # Add one node and try again
        dep = GoalNode(id="dep", description="Dependency")
        graph.add_node(dep)

        with pytest.raises(NodeNotFoundError, match="Node dependent not found"):
            graph.add_dependency("dependent", "dep")

    def test_add_dependency_creates_cycle(self):
        """Test that adding a dependency that creates a cycle raises CycleDetectedError."""
        graph = GoalGraph()

        # Create a chain: A -> B -> C
        node_a = GoalNode(id="A", description="Node A")
        node_b = GoalNode(id="B", description="Node B")
        node_c = GoalNode(id="C", description="Node C")

        graph.add_node(node_a)
        graph.add_node(node_b)
        graph.add_node(node_c)

        graph.add_dependency("B", "A")
        graph.add_dependency("C", "B")

        # Try to add A -> C, which would create a cycle
        with pytest.raises(CycleDetectedError, match="Adding dependency A -> C would create a cycle"):
            graph.add_dependency("A", "C")

    def test_remove_dependency(self):
        """Test removing a dependency between nodes."""
        graph = GoalGraph()

        dep = GoalNode(id="dep", description="Dependency")
        dependent = GoalNode(id="dependent", description="Dependent")

        graph.add_node(dep)
        graph.add_node(dependent)
        graph.add_dependency("dependent", "dep")

        # Remove the dependency
        graph.remove_dependency("dependent", "dep")

        # Check that the edge was removed from the graph
        assert not graph._graph.has_edge("dep", "dependent")

        # Check that the dependency was removed from the dependent node
        dependent_node = graph.get_node("dependent")
        assert "dep" not in dependent_node.dependencies

        # Check that the child was removed from the dependency node
        dep_node = graph.get_node("dep")
        assert "dependent" not in dep_node.children

    def test_remove_nonexistent_dependency(self):
        """Test removing a nonexistent dependency does nothing."""
        graph = GoalGraph()

        dep = GoalNode(id="dep", description="Dependency")
        dependent = GoalNode(id="dependent", description="Dependent")

        graph.add_node(dep)
        graph.add_node(dependent)

        # Remove a dependency that doesn't exist
        graph.remove_dependency("dependent", "dep")

        # Should not raise any error and nodes should remain unchanged
        assert len(graph._graph.edges) == 0
        assert len(dependent.dependencies) == 0
        assert len(dep.children) == 0

    def test_add_parent_child(self):
        """Test adding a parent-child relationship."""
        graph = GoalGraph()

        parent = GoalNode(id="parent", description="Parent")
        child = GoalNode(id="child", description="Child")

        graph.add_node(parent)
        graph.add_node(child)

        graph.add_parent_child("parent", "child")

        # Check that the child's parent is set
        child_node = graph.get_node("child")
        assert child_node.parent == "parent"

        # Check that the parent's children includes the child
        parent_node = graph.get_node("parent")
        assert "child" in parent_node.children

    def test_add_parent_child_nonexistent_nodes(self):
        """Test adding parent-child with nonexistent nodes raises NodeNotFoundError."""
        graph = GoalGraph()

        with pytest.raises(NodeNotFoundError, match="Node parent not found"):
            graph.add_parent_child("parent", "child")

        # Add one node and try again
        parent = GoalNode(id="parent", description="Parent")
        graph.add_node(parent)

        with pytest.raises(NodeNotFoundError, match="Node child not found"):
            graph.add_parent_child("parent", "child")

    def test_remove_parent_child(self):
        """Test removing a parent-child relationship."""
        graph = GoalGraph()

        parent = GoalNode(id="parent", description="Parent")
        child = GoalNode(id="child", description="Child")

        graph.add_node(parent)
        graph.add_node(child)
        graph.add_parent_child("parent", "child")

        # Remove the relationship
        graph.remove_parent_child("parent", "child")

        # Check that the child's parent is cleared
        child_node = graph.get_node("child")
        assert child_node.parent is None

        # Check that the parent's children no longer includes the child
        parent_node = graph.get_node("parent")
        assert "child" not in parent_node.children

    def test_get_ready_nodes(self):
        """Test getting nodes that are ready for execution."""
        graph = GoalGraph()

        # Create nodes with different statuses
        ready_node = GoalNode(id="ready", description="Ready", status=NodeStatus.PENDING)
        in_progress_node = GoalNode(id="in_progress", description="In Progress", status=NodeStatus.IN_PROGRESS)
        completed_node = GoalNode(id="completed", description="Completed", status=NodeStatus.COMPLETED)

        graph.add_node(ready_node)
        graph.add_node(in_progress_node)
        graph.add_node(completed_node)

        ready_nodes = graph.get_ready_nodes()

        assert len(ready_nodes) == 1
        assert ready_nodes[0].id == "ready"

    def test_get_ready_nodes_with_dependencies(self):
        """Test getting ready nodes considers dependencies."""
        graph = GoalGraph()

        # Create a dependency chain: A -> B -> C
        node_a = GoalNode(id="A", description="Node A", status=NodeStatus.PENDING)
        node_b = GoalNode(id="B", description="Node B", status=NodeStatus.PENDING)
        node_c = GoalNode(id="C", description="Node C", status=NodeStatus.PENDING)

        graph.add_node(node_a)
        graph.add_node(node_b)
        graph.add_node(node_c)

        graph.add_dependency("B", "A")
        graph.add_dependency("C", "B")

        # Initially, only A should be ready
        ready_nodes = graph.get_ready_nodes()
        assert len(ready_nodes) == 1
        assert ready_nodes[0].id == "A"

        # Mark A as completed
        node_a.mark_completed()
        graph.update_node(node_a)

        # Now B should be ready
        ready_nodes = graph.get_ready_nodes()
        assert len(ready_nodes) == 1
        assert ready_nodes[0].id == "B"

        # Mark B as completed
        node_b.mark_completed()
        graph.update_node(node_b)

        # Now C should be ready
        ready_nodes = graph.get_ready_nodes()
        assert len(ready_nodes) == 1
        assert ready_nodes[0].id == "C"

    def test_get_all_nodes(self):
        """Test getting all nodes in the graph."""
        graph = GoalGraph()

        node1 = GoalNode(id="node1", description="Node 1")
        node2 = GoalNode(id="node2", description="Node 2")
        node3 = GoalNode(id="node3", description="Node 3")

        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)

        all_nodes = graph.get_all_nodes()

        assert len(all_nodes) == 3
        node_ids = {node.id for node in all_nodes}
        assert node_ids == {"node1", "node2", "node3"}

    def test_get_nodes_by_status(self):
        """Test getting nodes filtered by status."""
        graph = GoalGraph()

        pending_node = GoalNode(id="pending", description="Pending", status=NodeStatus.PENDING)
        in_progress_node = GoalNode(id="in_progress", description="In Progress", status=NodeStatus.IN_PROGRESS)
        completed_node = GoalNode(id="completed", description="Completed", status=NodeStatus.COMPLETED)
        failed_node = GoalNode(id="failed", description="Failed", status=NodeStatus.FAILED)

        graph.add_node(pending_node)
        graph.add_node(in_progress_node)
        graph.add_node(completed_node)
        graph.add_node(failed_node)

        # Test filtering by different statuses
        pending_nodes = graph.get_nodes_by_status(NodeStatus.PENDING)
        assert len(pending_nodes) == 1
        assert pending_nodes[0].id == "pending"

        completed_nodes = graph.get_nodes_by_status(NodeStatus.COMPLETED)
        assert len(completed_nodes) == 1
        assert completed_nodes[0].id == "completed"

        # Test with non-existent status
        non_existent_nodes = graph.get_nodes_by_status(NodeStatus.BLOCKED)
        assert len(non_existent_nodes) == 0

    def test_get_leaf_nodes(self):
        """Test getting leaf nodes (nodes with no children)."""
        graph = GoalGraph()

        # Create a tree structure: A -> B, A -> C, B -> D
        node_a = GoalNode(id="A", description="Root")
        node_b = GoalNode(id="B", description="Child 1")
        node_c = GoalNode(id="C", description="Child 2")
        node_d = GoalNode(id="D", description="Grandchild")

        graph.add_node(node_a)
        graph.add_node(node_b)
        graph.add_node(node_c)
        graph.add_node(node_d)

        graph.add_parent_child("A", "B")
        graph.add_parent_child("A", "C")
        graph.add_parent_child("B", "D")

        leaf_nodes = graph.get_leaf_nodes()

        assert len(leaf_nodes) == 2
        leaf_ids = {node.id for node in leaf_nodes}
        assert leaf_ids == {"C", "D"}

    def test_get_root_nodes(self):
        """Test getting root nodes (nodes with no parents)."""
        graph = GoalGraph()

        # Create a tree structure: A -> B, A -> C, B -> D
        node_a = GoalNode(id="A", description="Root")
        node_b = GoalNode(id="B", description="Child 1")
        node_c = GoalNode(id="C", description="Child 2")
        node_d = GoalNode(id="D", description="Grandchild")

        graph.add_node(node_a)
        graph.add_node(node_b)
        graph.add_node(node_c)
        graph.add_node(node_d)

        graph.add_parent_child("A", "B")
        graph.add_parent_child("A", "C")
        graph.add_parent_child("B", "D")

        root_nodes = graph.get_root_nodes()

        assert len(root_nodes) == 1
        assert root_nodes[0].id == "A"

    def test_get_ancestors(self):
        """Test getting ancestors of a node."""
        graph = GoalGraph()

        # Create a chain: A -> B -> C -> D
        node_a = GoalNode(id="A", description="A")
        node_b = GoalNode(id="B", description="B")
        node_c = GoalNode(id="C", description="C")
        node_d = GoalNode(id="D", description="D")

        graph.add_node(node_a)
        graph.add_node(node_b)
        graph.add_node(node_c)
        graph.add_node(node_d)

        graph.add_dependency("B", "A")
        graph.add_dependency("C", "B")
        graph.add_dependency("D", "C")

        # Get ancestors of D
        ancestors = graph.get_ancestors("D")
        ancestor_ids = {node.id for node in ancestors}
        assert ancestor_ids == {"A", "B", "C"}

        # Get ancestors of A (should be empty)
        ancestors = graph.get_ancestors("A")
        assert len(ancestors) == 0

    def test_get_descendants(self):
        """Test getting descendants of a node."""
        graph = GoalGraph()

        # Create a tree: A -> B, A -> C, B -> D
        node_a = GoalNode(id="A", description="A")
        node_b = GoalNode(id="B", description="B")
        node_c = GoalNode(id="C", description="C")
        node_d = GoalNode(id="D", description="D")

        graph.add_node(node_a)
        graph.add_node(node_b)
        graph.add_node(node_c)
        graph.add_node(node_d)

        graph.add_parent_child("A", "B")
        graph.add_parent_child("A", "C")
        graph.add_parent_child("B", "D")

        # Get descendants of A
        descendants = graph.get_descendants("A")
        descendant_ids = {node.id for node in descendants}
        assert descendant_ids == {"B", "C", "D"}

        # Get descendants of D (should be empty)
        descendants = graph.get_descendants("D")
        assert len(descendants) == 0

    def test_is_dag(self):
        """Test checking if the graph is a DAG."""
        graph = GoalGraph()

        # Empty graph is a DAG
        assert graph.is_dag() is True

        # Linear chain is a DAG
        node_a = GoalNode(id="A", description="A")
        node_b = GoalNode(id="B", description="B")
        node_c = GoalNode(id="C", description="C")

        graph.add_node(node_a)
        graph.add_node(node_b)
        graph.add_node(node_c)

        graph.add_dependency("B", "A")
        graph.add_dependency("C", "B")

        assert graph.is_dag() is True

        # Try to add a cycle (should raise an exception)
        with pytest.raises(CycleDetectedError):
            graph.add_dependency("A", "C")

        # Graph should still be a DAG
        assert graph.is_dag() is True

    def test_save_and_load_graph(self):
        """Test saving and loading a graph to/from JSON."""
        graph = GoalGraph()

        # Create some nodes with relationships
        node_a = GoalNode(id="A", description="Node A", status=NodeStatus.PENDING)
        node_b = GoalNode(id="B", description="Node B", status=NodeStatus.IN_PROGRESS)

        graph.add_node(node_a)
        graph.add_node(node_b)
        graph.add_dependency("B", "A")

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            graph.save_to_json(temp_path)

            # Load from the same file
            loaded_graph = GoalGraph()
            loaded_graph.load_from_json(temp_path)

            # Verify the loaded graph has the same structure
            assert len(loaded_graph._nodes) == 2
            assert "A" in loaded_graph._nodes
            assert "B" in loaded_graph._nodes

            # Verify relationships
            assert loaded_graph._graph.has_edge("A", "B")

            # Verify node properties
            node_a_loaded = loaded_graph.get_node("A")
            assert node_a_loaded.description == "Node A"
            assert node_a_loaded.status == NodeStatus.PENDING

            node_b_loaded = loaded_graph.get_node("B")
            assert node_b_loaded.description == "Node B"
            assert node_b_loaded.status == NodeStatus.IN_PROGRESS

        finally:
            # Clean up
            temp_path.unlink(missing_ok=True)

    def test_graph_serialization_edge_cases(self):
        """Test graph serialization with edge cases."""
        graph = GoalGraph()

        # Test with empty graph
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            graph.save_to_json(temp_path)

            loaded_graph = GoalGraph()
            loaded_graph.load_from_json(temp_path)

            assert len(loaded_graph._nodes) == 0
            assert len(loaded_graph._graph.nodes) == 0
            assert len(loaded_graph._graph.edges) == 0

        finally:
            temp_path.unlink(missing_ok=True)

    def test_load_nonexistent_file(self):
        """Test loading from a nonexistent file raises FileNotFoundError."""
        graph = GoalGraph()

        with pytest.raises(FileNotFoundError):
            graph.load_from_json(Path("nonexistent.json"))

    def test_save_to_nonexistent_directory(self):
        """Test saving to a nonexistent directory raises FileNotFoundError."""
        graph = GoalGraph()

        with pytest.raises(FileNotFoundError):
            graph.save_to_json(Path("/nonexistent/directory/graph.json"))
