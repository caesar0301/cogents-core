"""
LLM-based goal decomposer for the GoalithService.

Provides structured goal decomposition using LLM clients with instructor integration.
"""

import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from cogents_core.goalith.errors import DecompositionError
from cogents_core.goalith.goalgraph.node import GoalNode
from cogents_core.llm import get_llm_client
from cogents_core.utils.logging import get_logger

from .base import GoalDecomposer
from .prompts import get_decomposition_system_prompt, get_decomposition_user_prompt, get_fallback_prompt

logger = get_logger(__name__)


class SubgoalSpec(BaseModel):
    """Specification for a single subgoal or task."""

    description: str = Field(description="Clear, actionable description of the subgoal or task")
    context: Optional[str] = Field(description="Additional notes or context about this subgoal", default=None)
    priority: float = Field(
        description="Priority score between 0.0 and 10.0 (higher = more important)",
        ge=0.0,
        le=10.0,
    )
    estimated_effort: Optional[str] = Field(
        description="Estimated effort or duration (e.g., '2 hours', '3 days', 'low', 'medium', 'high')",
        default=None,
    )
    dependencies: List[str] = Field(
        description="List of descriptions of other subgoals this depends on (will be matched by description)",
        default_factory=list,
    )
    tags: List[str] = Field(
        description="Tags for categorization (e.g., 'research', 'planning', 'execution')",
        default_factory=list,
    )


class GoalDecomposition(BaseModel):
    """Complete decomposition of a goal into subgoals and tasks."""

    reasoning: str = Field(description="Explanation of the decomposition approach and rationale")
    subgoals: List[SubgoalSpec] = Field(
        description="List of subgoals and tasks, in logical order (maximum 6 items for focus and manageability)",
        min_length=1,
        max_length=6,
    )
    success_criteria: List[str] = Field(
        description="Criteria that indicate successful completion of the overall goal",
        default_factory=list,
    )
    potential_risks: List[str] = Field(
        description="Potential risks or challenges in executing this plan",
        default_factory=list,
    )
    estimated_timeline: Optional[str] = Field(
        description="Overall estimated timeline for goal completion", default=None
    )
    confidence: float = Field(description="Confidence in this decomposition (0.0 to 1.0)", ge=0.0, le=1.0)


class LLMDecomposer(GoalDecomposer):
    """
    Enhanced LLM-based goal decomposer using structured completion.

    Uses the project's LLM infrastructure with instructor for structured output
    to decompose goals into subgoals.
    """

    def __init__(
        self,
        provider: str = os.getenv("COGENTS_LLM_PROVIDER", "openrouter"),
        model_name: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        name: str = "llm_decomposer",
    ):
        """
        Initialize LLM decomposer.

        Args:
            model_name: LLM model to use (uses default if None)
            temperature: Sampling temperature for LLM
            max_tokens: Maximum tokens for response
            name: Name of this decomposer
        """
        self._provider = provider
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._name = name
        self._llm_client = None  # Lazy initialization

    @property
    def name(self) -> str:
        """Get the name of this decomposer."""
        return self._name

    @property
    def llm_client(self):
        """Lazily initialize and return the LLM client."""
        if self._llm_client is None:
            self._llm_client = get_llm_client(provider=self._provider, chat_model=self._model_name)
        return self._llm_client

    def decompose(self, goal_node: GoalNode, context: Optional[Dict[str, Any]] = None) -> List[GoalNode]:
        """
        Decompose a goal using LLM structured completion.

        Args:
            goal_node: The goal node to decompose
            context: Optional context for decomposition

        Returns:
            List of subgoal nodes

        Raises:
            DecompositionError: If decomposition fails
        """
        try:
            logger.info(f"Decomposing goal: {goal_node.description}")

            # Build the decomposition prompts using the new structure
            system_prompt = get_decomposition_system_prompt()
            user_prompt = get_decomposition_user_prompt(
                goal_node=goal_node,
                context=context,
            )

            # Get structured decomposition from LLM with system and user messages
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

            decomposition: GoalDecomposition = self.llm_client.structured_completion(
                messages=messages,
                response_model=GoalDecomposition,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )

            logger.info(f"LLM decomposition completed with {len(decomposition.subgoals)} subgoals")
            logger.debug(f"Decomposition reasoning: {decomposition.reasoning}")

            # Convert to GoalNode objects
            subgoal_nodes = self._convert_to_goal_nodes(decomposition, goal_node, context)

            # Set up dependencies
            self._setup_dependencies(subgoal_nodes, decomposition.subgoals)

            # Add decomposition metadata to parent goal
            goal_node.update_context(
                "llm_decomposition",
                {
                    "reasoning": decomposition.reasoning,
                    "strategy": decomposition.decomposition_strategy,
                    "success_criteria": decomposition.success_criteria,
                    "potential_risks": decomposition.potential_risks,
                    "estimated_timeline": decomposition.estimated_timeline,
                    "confidence": decomposition.confidence,
                    "subgoal_count": len(decomposition.subgoals),
                },
            )

            return subgoal_nodes

        except Exception as e:
            logger.warning(f"Structured LLM decomposition failed for goal {goal_node.id}: {e}")
            logger.info("Attempting fallback decomposition...")

            # Try fallback decomposition with simpler prompt
            try:
                return self._fallback_decomposition(goal_node, context)
            except Exception as fallback_error:
                logger.error(f"Fallback decomposition also failed: {fallback_error}")
                raise DecompositionError(f"LLM decomposition failed: {e}")

    def _fallback_decomposition(self, goal_node: GoalNode, context: Optional[Dict[str, Any]] = None) -> List[GoalNode]:
        """
        Fallback decomposition method when structured completion fails.
        Uses a simpler prompt and attempts to parse the response manually.
        """
        fallback_prompt = get_fallback_prompt()
        user_prompt = get_decomposition_user_prompt(
            goal_node=goal_node,
            context=context,
        )

        # Combine prompts for fallback
        combined_prompt = f"{fallback_prompt}\n\n{user_prompt}"

        # Get simple text response
        messages = [{"role": "user", "content": combined_prompt}]
        response = self.llm_client.completion(
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        # Create simplified subgoals from the response
        # This is a basic implementation - in practice, you might want more sophisticated parsing
        lines = response.split("\n")
        subgoals = []

        for i, line in enumerate(lines[:6]):  # Limit to 6 items max
            line = line.strip()
            if line and len(line) > 10:  # Basic filtering
                # Create a simple subgoal
                node = GoalNode(
                    description=line.lstrip("1234567890.-• "),  # Remove numbering
                    priority=5.0,  # Default priority
                    parent=goal_node.id,
                    context={
                        "llm_generated": True,
                        "fallback_mode": True,
                        "parent_goal_id": goal_node.id,
                    },
                    decomposer_name=self.name,
                )
                subgoals.append(node)

        logger.info(f"Fallback decomposition created {len(subgoals)} subgoals")
        return subgoals

    def _convert_to_goal_nodes(
        self,
        decomposition: GoalDecomposition,
        parent_goal: GoalNode,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[GoalNode]:
        """Convert LLM decomposition to GoalNode objects."""

        nodes = []

        for spec in decomposition.subgoals:
            # Create node context
            node_context = {}
            if context:
                node_context.update(context)

            # Add LLM-specific context
            node_context.update(
                {
                    "llm_generated": True,
                    "decomposition_strategy": decomposition.decomposition_strategy,
                    "estimated_effort": spec.estimated_effort,
                    "context": spec.context,
                    "parent_goal_id": parent_goal.id,
                }
            )

            # Create the node
            node = GoalNode(
                description=spec.description,
                priority=spec.priority,
                parent=parent_goal.id,
                context=node_context,
                tags=set(spec.tags),
                decomposer_name=self.name,
            )

            # Copy deadline from parent if not specified and it's a task
            if parent_goal.deadline:
                node.deadline = parent_goal.deadline

            nodes.append(node)

        return nodes

    def _setup_dependencies(self, nodes: List[GoalNode], specs: List[SubgoalSpec]) -> None:
        """Set up dependencies between nodes based on LLM specifications."""

        # Create a mapping from description to node ID
        desc_to_id = {node.description: node.id for node in nodes}

        for i, spec in enumerate(specs):
            if not spec.dependencies:
                continue

            current_node = nodes[i]

            for dep_desc in spec.dependencies:
                # Find the dependency by description (fuzzy matching)
                dep_node_id = self._find_dependency_by_description(dep_desc, desc_to_id)

                if dep_node_id:
                    current_node.add_dependency(dep_node_id)
                    logger.debug(f"Added dependency: {dep_node_id} -> {current_node.id}")
                else:
                    logger.warning(f"Could not find dependency '{dep_desc}' for node '{current_node.description}'")

    def _find_dependency_by_description(self, dep_desc: str, desc_to_id: Dict[str, str]) -> Optional[str]:
        """Find a dependency node by description with fuzzy matching."""

        dep_desc_lower = dep_desc.lower().strip()

        # Exact match first
        for desc, node_id in desc_to_id.items():
            if desc.lower().strip() == dep_desc_lower:
                return node_id

        # Partial match (dependency description is contained in node description)
        for desc, node_id in desc_to_id.items():
            if dep_desc_lower in desc.lower() or desc.lower() in dep_desc_lower:
                return node_id

        # Word overlap matching
        dep_words = set(dep_desc_lower.split())
        best_match = None
        best_overlap = 0

        for desc, node_id in desc_to_id.items():
            desc_words = set(desc.lower().split())
            overlap = len(dep_words & desc_words)

            if overlap > best_overlap and overlap >= 2:  # At least 2 words overlap
                best_match = node_id
                best_overlap = overlap

        return best_match
