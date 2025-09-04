"""
Integration tests for goalith module components.
"""

import pytest

from cogents_core.goalith.decomposer.callable_decomposer import CallableDecomposer
from cogents_core.goalith.decomposer.simple_decomposer import SimpleListDecomposer
from cogents_core.goalith.errors import CycleDetectedError, DecompositionError
from cogents_core.goalith.goalgraph.graph import GoalGraph
from cogents_core.goalith.goalgraph.node import GoalNode, NodeStatus


class TestGoalithIntegration:
    """Integration tests for goalith module components."""

    def test_goal_graph_with_simple_decomposer(self):
        """Test integration between GoalGraph and SimpleListDecomposer."""
        # Create a goal graph
        graph = GoalGraph()

        # Create a root goal
        root_goal = GoalNode(
            id="root", description="Complete project setup", priority=2.0, context={"project": "test_project"}
        )
        graph.add_node(root_goal)

        # Create a simple decomposer
        subtasks = ["Install dependencies", "Configure environment", "Run initial tests"]
        decomposer = SimpleListDecomposer(subtasks, name="setup_decomposer")

        # Decompose the root goal
        subgoals = decomposer.decompose(root_goal)

        # Add subgoals to the graph
        for subgoal in subgoals:
            graph.add_node(subgoal)
            graph.add_parent_child("root", subgoal.id)

        # Verify the graph structure
        assert len(graph._nodes) == 4  # root + 3 subgoals
        assert "root" in graph._nodes

        # Verify subgoals
        subgoal_ids = [subgoal.id for subgoal in subgoals]
        for subgoal_id in subgoal_ids:
            assert subgoal_id in graph._nodes
            subgoal = graph.get_node(subgoal_id)
            assert subgoal.parent == "root"
            assert subgoal.priority == 2.0
            assert subgoal.context == {"project": "test_project"}
            assert subgoal.decomposer_name == "setup_decomposer"

        # Verify parent-child relationships
        root_node = graph.get_node("root")
        assert len(root_node.children) == 3
        for subgoal_id in subgoal_ids:
            assert subgoal_id in root_node.children

    def test_goal_graph_with_callable_decomposer(self):
        """Test integration between GoalGraph and CallableDecomposer."""
        # Create a goal graph
        graph = GoalGraph()

        # Create a root goal
        root_goal = GoalNode(id="root", description="Build application", priority=3.0, context={"app_type": "web"})
        graph.add_node(root_goal)

        # Create a callable decomposer
        def build_decompose(goal_node, context=None):
            if "web" in goal_node.context.get("app_type", ""):
                return [
                    GoalNode(
                        description="Setup frontend",
                        priority=goal_node.priority * 0.8,
                        decomposer_name="build_decomposer",
                    ),
                    GoalNode(
                        description="Setup backend",
                        priority=goal_node.priority * 0.9,
                        decomposer_name="build_decomposer",
                    ),
                    GoalNode(
                        description="Setup database",
                        priority=goal_node.priority * 0.7,
                        decomposer_name="build_decomposer",
                    ),
                ]
            return [GoalNode(description="Generic build task", decomposer_name="build_decomposer")]

        decomposer = CallableDecomposer(build_decompose, name="build_decomposer")

        # Decompose the root goal
        subgoals = decomposer.decompose(root_goal)

        # Add subgoals to the graph
        for subgoal in subgoals:
            graph.add_node(subgoal)
            graph.add_parent_child("root", subgoal.id)

        # Verify the graph structure
        assert len(graph._nodes) == 4  # root + 3 subgoals

        # Verify subgoals have correct priorities
        subgoal_descriptions = [subgoal.description for subgoal in subgoals]
        assert "Setup frontend" in subgoal_descriptions
        assert "Setup backend" in subgoal_descriptions
        assert "Setup database" in subgoal_descriptions

        # Check priorities (using approximate equality for floating point)
        for subgoal in subgoals:
            if subgoal.description == "Setup frontend":
                assert abs(subgoal.priority - 2.4) < 1e-10  # 3.0 * 0.8
            elif subgoal.description == "Setup backend":
                assert abs(subgoal.priority - 2.7) < 1e-10  # 3.0 * 0.9
            elif subgoal.description == "Setup database":
                assert abs(subgoal.priority - 2.1) < 1e-10  # 3.0 * 0.7

    def test_goal_graph_dependency_management(self):
        """Test dependency management in goal graph with decomposers."""
        # Create a goal graph
        graph = GoalGraph()

        # Create a complex goal
        complex_goal = GoalNode(id="complex", description="Deploy application", priority=2.0)
        graph.add_node(complex_goal)

        # Create a decomposer that creates dependent tasks
        def deploy_decompose(goal_node, context=None):
            return [
                GoalNode(description="Build application", decomposer_name="deploy_decomposer"),
                GoalNode(description="Run tests", decomposer_name="deploy_decomposer"),
                GoalNode(description="Deploy to staging", decomposer_name="deploy_decomposer"),
                GoalNode(description="Deploy to production", decomposer_name="deploy_decomposer"),
            ]

        decomposer = CallableDecomposer(deploy_decompose, name="deploy_decomposer")
        subgoals = decomposer.decompose(complex_goal)

        # Add subgoals to the graph
        for subgoal in subgoals:
            graph.add_node(subgoal)
            graph.add_parent_child("complex", subgoal.id)

        # Add dependencies between subgoals
        build_task = None
        test_task = None
        staging_task = None
        prod_task = None

        for subgoal in subgoals:
            if subgoal.description == "Build application":
                build_task = subgoal
            elif subgoal.description == "Run tests":
                test_task = subgoal
            elif subgoal.description == "Deploy to staging":
                staging_task = subgoal
            elif subgoal.description == "Deploy to production":
                prod_task = subgoal

        # Set up dependencies: tests depend on build, staging depends on tests, prod depends on staging
        graph.add_dependency(test_task.id, build_task.id)
        graph.add_dependency(staging_task.id, test_task.id)
        graph.add_dependency(prod_task.id, staging_task.id)

        # Verify dependencies
        assert build_task.id in test_task.dependencies
        assert test_task.id in staging_task.dependencies
        assert staging_task.id in prod_task.dependencies

        # Verify children relationships
        assert test_task.id in build_task.children
        assert staging_task.id in test_task.children
        assert prod_task.id in staging_task.children

        # Test ready nodes - initially only build should be ready
        ready_nodes = graph.get_ready_nodes()
        assert len(ready_nodes) == 1
        assert ready_nodes[0].id == build_task.id

        # Mark build as completed
        build_task.mark_completed()
        graph.update_node(build_task)

        # Now test should be ready
        ready_nodes = graph.get_ready_nodes()
        assert len(ready_nodes) == 1
        assert ready_nodes[0].id == test_task.id

    def test_goal_graph_cycle_detection_with_decomposers(self):
        """Test cycle detection when adding dependencies between decomposed tasks."""
        # Create a goal graph
        graph = GoalGraph()

        # Create a root goal
        root_goal = GoalNode(id="root", description="Process data")
        graph.add_node(root_goal)

        # Create a decomposer
        def process_decompose(goal_node, context=None):
            return [
                GoalNode(description="Load data", decomposer_name="process_decomposer"),
                GoalNode(description="Transform data", decomposer_name="process_decomposer"),
                GoalNode(description="Save data", decomposer_name="process_decomposer"),
            ]

        decomposer = CallableDecomposer(process_decompose, name="process_decomposer")
        subgoals = decomposer.decompose(root_goal)

        # Add subgoals to the graph
        for subgoal in subgoals:
            graph.add_node(subgoal)
            graph.add_parent_child("root", subgoal.id)

        # Find specific tasks
        load_task = None
        transform_task = None
        save_task = None

        for subgoal in subgoals:
            if subgoal.description == "Load data":
                load_task = subgoal
            elif subgoal.description == "Transform data":
                transform_task = subgoal
            elif subgoal.description == "Save data":
                save_task = subgoal

        # Add valid dependencies
        graph.add_dependency(transform_task.id, load_task.id)
        graph.add_dependency(save_task.id, transform_task.id)

        # Try to add a cycle (save -> load)
        with pytest.raises(CycleDetectedError):
            graph.add_dependency(load_task.id, save_task.id)

    def test_goal_graph_serialization_with_decomposers(self):
        """Test serialization of goal graph with decomposed tasks."""
        # Create a goal graph
        graph = GoalGraph()

        # Create a root goal
        root_goal = GoalNode(
            id="root", description="Complete task", priority=1.5, context={"task_type": "serialization_test"}
        )
        graph.add_node(root_goal)

        # Create a decomposer
        def test_decompose(goal_node, context=None):
            return [
                GoalNode(description="Subtask 1", priority=goal_node.priority * 0.5, decomposer_name="test_decomposer"),
                GoalNode(description="Subtask 2", priority=goal_node.priority * 0.8, decomposer_name="test_decomposer"),
            ]

        decomposer = CallableDecomposer(test_decompose, name="test_decomposer")
        subgoals = decomposer.decompose(root_goal)

        # Add subgoals to the graph
        for subgoal in subgoals:
            graph.add_node(subgoal)
            graph.add_parent_child("root", subgoal.id)

        # Add a dependency
        graph.add_dependency(subgoals[1].id, subgoals[0].id)

        # Save and load the graph
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            graph.save_to_json(temp_path)

            # Load from the same file
            loaded_graph = GoalGraph()
            loaded_graph.load_from_json(temp_path)

            # Verify the loaded graph has the same structure
            assert len(loaded_graph._nodes) == 3  # root + 2 subgoals

            # Verify root goal
            loaded_root = loaded_graph.get_node("root")
            assert loaded_root.description == "Complete task"
            assert loaded_root.priority == 1.5
            assert loaded_root.context == {"task_type": "serialization_test"}

            # Verify subgoals
            subgoal_ids = [subgoal.id for subgoal in subgoals]
            for subgoal_id in subgoal_ids:
                loaded_subgoal = loaded_graph.get_node(subgoal_id)
                assert loaded_subgoal.parent == "root"
                assert loaded_subgoal.decomposer_name == "test_decomposer"

            # Verify dependency
            assert loaded_graph._graph.has_edge(subgoals[0].id, subgoals[1].id)

        finally:
            # Clean up
            temp_path.unlink(missing_ok=True)

    def test_goal_graph_query_operations_with_decomposers(self):
        """Test query operations on goal graph with decomposed tasks."""
        # Create a goal graph
        graph = GoalGraph()

        # Create a root goal
        root_goal = GoalNode(id="root", description="Main goal")
        graph.add_node(root_goal)

        # Create a decomposer
        def query_decompose(goal_node, context=None):
            return [
                GoalNode(description="Task A", status=NodeStatus.PENDING, decomposer_name="query_decomposer"),
                GoalNode(description="Task B", status=NodeStatus.IN_PROGRESS, decomposer_name="query_decomposer"),
                GoalNode(description="Task C", status=NodeStatus.COMPLETED, decomposer_name="query_decomposer"),
                GoalNode(description="Task D", status=NodeStatus.FAILED, decomposer_name="query_decomposer"),
            ]

        decomposer = CallableDecomposer(query_decompose, name="query_decomposer")
        subgoals = decomposer.decompose(root_goal)

        # Add subgoals to the graph
        for subgoal in subgoals:
            graph.add_node(subgoal)
            graph.add_parent_child("root", subgoal.id)

        # Test get_nodes_by_status
        pending_nodes = graph.get_nodes_by_status(NodeStatus.PENDING)
        assert len(pending_nodes) == 1
        assert pending_nodes[0].description == "Task A"

        in_progress_nodes = graph.get_nodes_by_status(NodeStatus.IN_PROGRESS)
        assert len(in_progress_nodes) == 1
        assert in_progress_nodes[0].description == "Task B"

        completed_nodes = graph.get_nodes_by_status(NodeStatus.COMPLETED)
        assert len(completed_nodes) == 1
        assert completed_nodes[0].description == "Task C"

        failed_nodes = graph.get_nodes_by_status(NodeStatus.FAILED)
        assert len(failed_nodes) == 1
        assert failed_nodes[0].description == "Task D"

        # Test get_leaf_nodes
        leaf_nodes = graph.get_leaf_nodes()
        assert len(leaf_nodes) == 4  # All subgoals are leaves

        # Test get_root_nodes
        root_nodes = graph.get_root_nodes()
        assert len(root_nodes) == 1
        assert root_nodes[0].id == "root"

    def test_goal_graph_error_handling_with_decomposers(self):
        """Test error handling when decomposers fail."""
        # Create a goal graph
        graph = GoalGraph()

        # Create a root goal
        root_goal = GoalNode(id="root", description="Error test goal")
        graph.add_node(root_goal)

        # Create a decomposer that raises an exception
        def error_decompose(goal_node, context=None):
            raise RuntimeError("Decomposition failed")

        decomposer = CallableDecomposer(error_decompose, name="error_decomposer")

        # Test that decomposition error is properly handled
        with pytest.raises(DecompositionError, match="Decomposition failed: Decomposition failed"):
            decomposer.decompose(root_goal)

        # Verify the graph is unchanged
        assert len(graph._nodes) == 1
        assert "root" in graph._nodes

    def test_goal_graph_complex_workflow(self):
        """Test a complex workflow with multiple decomposers and dependencies."""
        # Create a goal graph
        graph = GoalGraph()

        # Create a main project goal
        project_goal = GoalNode(
            id="project",
            description="Complete software project",
            priority=3.0,
            context={"project_name": "test_project"},
        )
        graph.add_node(project_goal)

        # Create a decomposer for project phases
        def project_decompose(goal_node, context=None):
            return [
                GoalNode(description="Planning phase", decomposer_name="project_decomposer"),
                GoalNode(description="Development phase", decomposer_name="project_decomposer"),
                GoalNode(description="Testing phase", decomposer_name="project_decomposer"),
                GoalNode(description="Deployment phase", decomposer_name="project_decomposer"),
            ]

        project_decomposer = CallableDecomposer(project_decompose, name="project_decomposer")
        phases = project_decomposer.decompose(project_goal)

        # Add phases to the graph
        for phase in phases:
            graph.add_node(phase)
            graph.add_parent_child("project", phase.id)

        # Create a decomposer for development tasks
        def development_decompose(goal_node, context=None):
            return [
                GoalNode(description="Setup development environment", decomposer_name="dev_decomposer"),
                GoalNode(description="Implement core features", decomposer_name="dev_decomposer"),
                GoalNode(description="Write unit tests", decomposer_name="dev_decomposer"),
                GoalNode(description="Code review", decomposer_name="dev_decomposer"),
            ]

        dev_decomposer = CallableDecomposer(development_decompose, name="dev_decomposer")

        # Find the development phase and decompose it
        dev_phase = None
        for phase in phases:
            if phase.description == "Development phase":
                dev_phase = phase
                break

        dev_tasks = dev_decomposer.decompose(dev_phase)

        # Add development tasks to the graph
        for task in dev_tasks:
            graph.add_node(task)
            graph.add_parent_child(dev_phase.id, task.id)

        # Add dependencies between development tasks
        setup_task = None
        implement_task = None
        test_task = None
        review_task = None

        for task in dev_tasks:
            if task.description == "Setup development environment":
                setup_task = task
            elif task.description == "Implement core features":
                implement_task = task
            elif task.description == "Write unit tests":
                test_task = task
            elif task.description == "Code review":
                review_task = task

        # Set up dependencies
        graph.add_dependency(implement_task.id, setup_task.id)
        graph.add_dependency(test_task.id, implement_task.id)
        graph.add_dependency(review_task.id, test_task.id)

        # Verify the complete graph structure
        assert len(graph._nodes) == 9  # project + 4 phases + 4 dev tasks

        # Verify parent-child relationships
        project_node = graph.get_node("project")
        assert len(project_node.children) == 4

        dev_phase_node = graph.get_node(dev_phase.id)
        assert len(dev_phase_node.children) == 4

        # Verify dependencies
        assert setup_task.id in implement_task.dependencies
        assert implement_task.id in test_task.dependencies
        assert test_task.id in review_task.dependencies

        # Test ready nodes - initially only setup should be ready
        ready_nodes = graph.get_ready_nodes()
        ready_descriptions = [node.description for node in ready_nodes]
        assert "Setup development environment" in ready_descriptions
        assert "Planning phase" in ready_descriptions  # No dependencies

        # Mark setup as completed
        setup_task.mark_completed()
        graph.update_node(setup_task)

        # Now implement should be ready
        ready_nodes = graph.get_ready_nodes()
        ready_descriptions = [node.description for node in ready_nodes]
        assert "Implement core features" in ready_descriptions
