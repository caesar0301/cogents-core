"""
Base classes and exceptions for the GoalithService.

This module contains the foundational classes and exceptions used across
the goalith_service module to avoid circular imports.
"""


class DecompositionError(Exception):
    """Raised when decomposition fails."""


class CycleDetectedError(Exception):
    """Raised when a cycle is detected in the DAG."""


class NodeNotFoundError(Exception):
    """Raised when a node is not found in the graph."""


class SchedulingError(Exception):
    """Raised when scheduling operations fail."""
