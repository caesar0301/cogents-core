from typing import Any, Dict, List, Optional


class GoalithService:
    def __init__(self):
        pass

    def create_goal(
        self,
        description: str,
        priority: float = 1.0,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Create a new goal in the system.

        Args:
            description: Description of the goal
            priority: Priority level
            context: Additional context data
            tags: Tags for categorization

        Returns:
            ID of the created goal
        """
