"""
Graph storage and operations for the GoalithService DAG-based goal management system.
"""

import json
from pathlib import Path
from typing import Dict, List

import networkx as nx

from cogents_core.goalith.errors import CycleDetectedError, NodeNotFoundError
from cogents_core.goalith.goalgraph.node import GoalNode, NodeStatus


class GoalGraph:
    """
    Wraps a NetworkX DiGraph to provide DAG storage and basic graph operations.

    Handles CRUD operations on nodes and edges, queries for ready nodes,
    and persistence/loading of graph snapshots.
    """

    def __init__(self):
        """Initialize empty DAG."""
        self._graph = nx.DiGraph()
        self._nodes: Dict[str, GoalNode] = {}

    # Core CRUD operations

    def add_node(self, node: GoalNode) -> None:
        """
        Add a node to the graph.

        Args:
            node: The GoalNode to add

        Raises:
            ValueError: If node already exists
        """
        if node.id in self._nodes:
            raise ValueError(f"Node {node.id} already exists")

        self._nodes[node.id] = node
        self._graph.add_node(node.id)

    def get_node(self, node_id: str) -> GoalNode:
        """
        Get a node by ID.

        Args:
            node_id: The node ID

        Returns:
            The GoalNode

        Raises:
            NodeNotFoundError: If node doesn't exist
        """
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Node {node_id} not found")
        return self._nodes[node_id]

    def update_node(self, node: GoalNode) -> None:
        """
        Update an existing node.

        Args:
            node: The updated GoalNode

        Raises:
            NodeNotFoundError: If node doesn't exist
        """
        if node.id not in self._nodes:
            raise NodeNotFoundError(f"Node {node.id} not found")
        self._nodes[node.id] = node

    def remove_node(self, node_id: str) -> None:
        """
        Remove a node and all its edges.

        Args:
            node_id: The node ID to remove

        Raises:
            NodeNotFoundError: If node doesn't exist
        """
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Node {node_id} not found")

        # Remove from graph (this also removes all edges)
        self._graph.remove_node(node_id)

        # Remove from nodes dict
        del self._nodes[node_id]

        # Update parent/child relationships in remaining nodes
        for node in self._nodes.values():
            node.dependencies.discard(node_id)
            node.children.discard(node_id)
            if node.parent == node_id:
                node.parent = None

    def add_dependency(self, dependent_id: str, dependency_id: str) -> None:
        """
        Add a dependency edge between two nodes.

        Args:
            dependent_id: The node that depends on the prerequisite
            dependency_id: The node that is depended upon (prerequisite)

        Raises:
            NodeNotFoundError: If either node doesn't exist
            CycleDetectedError: If adding this edge would create a cycle
        """
        if dependent_id not in self._nodes:
            raise NodeNotFoundError(f"Node {dependent_id} not found")
        if dependency_id not in self._nodes:
            raise NodeNotFoundError(f"Node {dependency_id} not found")

        # Check if adding this edge would create a cycle before actually adding it
        # Create a temporary graph to test for cycles
        temp_graph = self._graph.copy()
        temp_graph.add_edge(dependency_id, dependent_id)

        if not nx.is_directed_acyclic_graph(temp_graph):
            raise CycleDetectedError(f"Adding dependency {dependent_id} -> {dependency_id} would create a cycle")

        # Add edge to graph (dependency -> dependent)
        self._graph.add_edge(dependency_id, dependent_id)

        # Update node relationships: dependent node gets the dependency
        self._nodes[dependent_id].add_dependency(dependency_id)
        # Update parent-child relationship: dependency becomes parent of dependent
        self._nodes[dependency_id].add_child(dependent_id)
        # Note: We don't set parent here as it conflicts with hierarchical parent-child relationships

    def remove_dependency(self, dependent_id: str, dependency_id: str) -> None:
        """
        Remove a dependency edge between two nodes.

        Args:
            dependent_id: The node that depends on the prerequisite
            dependency_id: The node that is depended upon (prerequisite)

        Note:
            If the dependency relationship doesn't exist, this method does nothing.
        """
        if self._graph.has_edge(dependency_id, dependent_id):
            self._graph.remove_edge(dependency_id, dependent_id)

            if dependent_id in self._nodes:
                self._nodes[dependent_id].remove_dependency(dependency_id)

            if dependency_id in self._nodes:
                self._nodes[dependency_id].remove_child(dependent_id)

    def add_parent_child(self, parent_id: str, child_id: str) -> None:
        """
        Add a parent-child relationship (hierarchical, not dependency).

        Args:
            parent_id: The parent node ID
            child_id: The child node ID

        Raises:
            NodeNotFoundError: If either node doesn't exist
        """
        if parent_id not in self._nodes:
            raise NodeNotFoundError(f"Node {parent_id} not found")
        if child_id not in self._nodes:
            raise NodeNotFoundError(f"Node {child_id} not found")

        # Update relationships
        self._nodes[parent_id].add_child(child_id)
        self._nodes[child_id].parent = parent_id

    def remove_parent_child(self, parent_id: str, child_id: str) -> None:
        """
        Remove a parent-child relationship.

        Args:
            parent_id: The parent node ID
            child_id: The child node ID
        """
        if parent_id in self._nodes:
            self._nodes[parent_id].remove_child(child_id)
        if child_id in self._nodes:
            self._nodes[child_id].parent = None

    # Query operations

    def get_ready_nodes(self) -> List[GoalNode]:
        """
        Get all nodes that are ready for execution (all dependencies completed).

        Returns:
            List of ready GoalNodes
        """
        ready_nodes = []

        # First, find all nodes that are dependencies of other nodes
        dependency_nodes = set()
        for node in self._nodes.values():
            dependency_nodes.update(node.dependencies)

        for node in self._nodes.values():
            # Skip nodes that have children but are not dependencies of any other node
            # These are organizational containers that are not meant to be executed directly
            if node.children and node.id not in dependency_nodes:
                continue

            if node.status == NodeStatus.PENDING:
                # Check if all dependencies are completed
                all_deps_completed = True
                for dep_id in node.dependencies:
                    if dep_id in self._nodes:
                        dep_node = self._nodes[dep_id]
                        if dep_node.status != NodeStatus.COMPLETED:
                            all_deps_completed = False
                            break
                    else:
                        # Dependency node doesn't exist, consider it completed
                        pass

                if all_deps_completed:
                    ready_nodes.append(node)

        return ready_nodes

    def get_parents(self, node_id: str) -> List[GoalNode]:
        """
        Get all parent nodes of a given node.

        Args:
            node_id: The node ID

        Returns:
            List of parent GoalNode objects (including both hierarchical parents and dependencies)
        """
        if node_id not in self._nodes:
            return []

        node = self._nodes[node_id]
        parents = []

        # Add hierarchical parent if it exists
        if node.parent and node.parent in self._nodes:
            parents.append(self._nodes[node.parent])

        # Add dependency parents
        for dep_id in node.dependencies:
            if dep_id in self._nodes:
                parents.append(self._nodes[dep_id])

        return parents

    def get_children(self, node_id: str) -> List[GoalNode]:
        """
        Get all child nodes of a given node.

        Args:
            node_id: The node ID

        Returns:
            List of child GoalNode objects
        """
        if node_id not in self._nodes:
            return []

        node = self._nodes[node_id]
        return [self._nodes[child_id] for child_id in node.children if child_id in self._nodes]

    def list_nodes(self) -> List[GoalNode]:
        """
        Get list of all node objects.

        Returns:
            List of all GoalNode objects
        """
        return list(self._nodes.values())

    def get_nodes_by_status(self, status: NodeStatus) -> List[GoalNode]:
        """
        Get all nodes with a specific status.

        Args:
            status: The status to filter by

        Returns:
            List of GoalNodes with the given status
        """
        return [node for node in self._nodes.values() if node.status == status and not node.children]

    def get_dependencies(self, node_id: str) -> List[GoalNode]:
        """
        Get all dependency nodes of a given node.

        Args:
            node_id: The dependent node ID

        Returns:
            List of dependency GoalNodes

        Raises:
            NodeNotFoundError: If node doesn't exist
        """
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Node {node_id} not found")

        node = self._nodes[node_id]
        return [self._nodes[dep_id] for dep_id in node.dependencies if dep_id in self._nodes]

    def get_root_nodes(self) -> List[GoalNode]:
        """
        Get all root nodes (nodes with no parents).

        Returns:
            List of root GoalNodes
        """
        return [node for node in self._nodes.values() if node.parent is None]

    def get_descendants(self, node_id: str) -> List[GoalNode]:
        """
        Get all descendant nodes of a given node.

        Args:
            node_id: The node ID

        Returns:
            List of descendant GoalNode objects
        """
        if node_id not in self._nodes:
            return []

        descendants = []
        visited = set()
        to_process = [node_id]

        while to_process:
            current_id = to_process.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)

            current = self._nodes[current_id]
            for child_id in current.children:
                if child_id in self._nodes and child_id not in visited:
                    descendants.append(self._nodes[child_id])
                    to_process.append(child_id)
        return descendants

    def get_ancestors(self, node_id: str) -> List[GoalNode]:
        """
        Get all ancestor nodes of a given node.

        Args:
            node_id: The node ID

        Returns:
            List of ancestor GoalNode objects
        """
        if node_id not in self._nodes:
            return []

        ancestors = []
        visited = set()
        current_id = node_id

        # First, traverse parent hierarchy
        while current_id in self._nodes:
            node = self._nodes[current_id]
            if node.parent and node.parent not in visited:
                parent = self._nodes[node.parent]
                ancestors.append(parent)
                visited.add(node.parent)
                current_id = node.parent
            else:
                break

        # Then, traverse dependency hierarchy
        def add_dependency_ancestors(dep_id: str):
            if dep_id in visited or dep_id not in self._nodes:
                return
            visited.add(dep_id)
            dep_node = self._nodes[dep_id]
            ancestors.append(dep_node)
            # Recursively add ancestors of this dependency
            for parent_dep_id in dep_node.dependencies:
                add_dependency_ancestors(parent_dep_id)

        # Start recursive traversal from the original node's dependencies
        original_node = self._nodes[node_id]
        for dep_id in original_node.dependencies:
            add_dependency_ancestors(dep_id)

        return ancestors

    def is_dag(self) -> bool:
        """
        Check if the graph is a DAG.

        Returns:
            True if valid DAG, False otherwise
        """
        return nx.is_directed_acyclic_graph(self._graph)

    def get_leaf_nodes(self) -> List[GoalNode]:
        """
        Get all leaf nodes (nodes with no children).

        Returns:
            List of leaf GoalNodes
        """
        return [node for node in self._nodes.values() if not node.children]

    def get_all_nodes(self) -> List[GoalNode]:
        """
        Get all nodes in the graph.

        Returns:
            List of all GoalNodes
        """
        return list(self._nodes.values())

    def has_node(self, node_id: str) -> bool:
        """
        Check if a node exists in the graph.

        Args:
            node_id: The node ID to check

        Returns:
            True if node exists, False otherwise
        """
        return node_id in self._nodes

    # Persistence operations

    def save_to_json(self, filepath: Path) -> None:
        """
        Save the graph to a JSON file.

        Args:
            filepath: Path to save the graph
        """
        # Prepare data for serialization
        data = {
            "nodes": {node_id: node.model_dump(mode="json") for node_id, node in self._nodes.items()},
            "edges": list(self._graph.edges()),
        }

        # Save to file
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load_from_json(self, filepath: Path) -> None:
        """
        Load the graph from a JSON file.

        Args:
            filepath: Path to load the graph from
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        # Clear current graph
        self._graph.clear()
        self._nodes.clear()

        # Load nodes
        for node_id, node_data in data["nodes"].items():
            # Convert sets back from lists
            if "dependencies" in node_data:
                node_data["dependencies"] = set(node_data["dependencies"])
            if "children" in node_data:
                node_data["children"] = set(node_data["children"])
            if "tags" in node_data:
                node_data["tags"] = set(node_data["tags"])

            node = GoalNode(**node_data)
            self._nodes[node_id] = node
            self._graph.add_node(node_id)

        # Load edges
        for source, target in data["edges"]:
            self._graph.add_edge(source, target)

    def clear(self) -> None:
        """Clear all nodes and edges from the graph."""
        self._graph.clear()
        self._nodes.clear()

    def get_graph_stats(self) -> Dict[str, int]:
        """
        Get statistics about the graph.

        Returns:
            Dictionary with graph statistics
        """
        status_counts = {}
        for status in NodeStatus:
            status_counts[str(status)] = len(self.get_nodes_by_status(status))

        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._graph.edges()),
            "ready_nodes": len(self.get_ready_nodes()),
            "root_nodes": len(self.get_root_nodes()),
            "leaf_nodes": len(self.get_leaf_nodes()),
            **status_counts,
        }

    # Test-compatible methods that return IDs instead of objects
    def get_dependency_ids(self, node_id: str) -> set:
        """Get dependency IDs for a node (test-compatible version)."""
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Node {node_id} not found")
        node = self._nodes[node_id]
        return set(node.dependencies)

    def get_child_ids(self, node_id: str) -> set:
        """Get child IDs for a node (test-compatible version)."""
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Node {node_id} not found")
        node = self._nodes[node_id]
        return set(node.children)
