"""
Example demonstrating the LLM-based goal decomposer for GoalithService.

This example shows how to use the structured LLM decomposer to break down
goals into actionable subgoals and tasks.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cogents_core.goalith import GoalithService, LLMDecomposer, NodeType


def print_goal_graph(goalith: GoalithService, root_goal_id: str, indent: int = 0):
    """
    Print the goal graph in a hierarchical tree structure.

    Args:
        goalith: GoalithService instance
        root_goal_id: ID of the root goal to start from
        indent: Current indentation level
    """
    try:
        node = goalith.get_node(root_goal_id)
        prefix = "  " * indent

        # Node status indicator
        status_emoji = {
            "pending": "⏳",
            "in_progress": "🔄",
            "completed": "✅",
            "failed": "❌",
            "cancelled": "🚫",
            "blocked": "🚧",
        }.get(node.status, "❓")

        # Node type indicator
        type_emoji = {"goal": "🎯", "subgoal": "📋", "task": "⚡"}.get(node.type, "📄")

        # Print current node
        print(f"{prefix}{status_emoji} {type_emoji} {node.description}")
        print(f"{prefix}   ID: {node.id}")
        print(f"{prefix}   Priority: {node.priority}")
        print(f"{prefix}   Status: {node.status}")

        # Show tags if any
        if node.tags:
            print(f"{prefix}   Tags: {', '.join(node.tags)}")

        # Show dependencies if any
        if node.dependencies:
            dep_descriptions = []
            for dep_id in node.dependencies:
                try:
                    dep_node = goalith.get_node(dep_id)
                    dep_descriptions.append(
                        dep_node.description[:30] + "..." if len(dep_node.description) > 30 else dep_node.description
                    )
                except:
                    dep_descriptions.append(f"Unknown ({dep_id})")
            print(f"{prefix}   Dependencies: {', '.join(dep_descriptions)}")

        # Show LLM-specific context if available
        if "estimated_effort" in node.context:
            print(f"{prefix}   Effort: {node.context['estimated_effort']}")
        if "notes" in node.context:
            print(f"{prefix}   Notes: {node.context['notes']}")

        print()  # Empty line for readability

        # Recursively print children
        children = goalith.get_children(root_goal_id)
        for child in children:
            print_goal_graph(goalith, child.id, indent + 1)

    except Exception as e:
        print(f"{prefix}❌ Error loading node {root_goal_id}: {e}")


def example_basic_llm_decomposition():
    """Basic example of LLM decomposition."""
    print("=== Basic LLM Decomposition Example ===")

    # Create an GoalithService instance
    goalith = GoalithService()

    # Create a goal
    goal_id = goalith.create_goal(
        description="Plan and execute a product launch for a new mobile app",
        goal_type=NodeType.GOAL,
        priority=8.0,
        deadline=datetime.now() + timedelta(days=90),
        context={
            "target_audience": "young professionals",
            "budget": "$50,000",
            "platform": "iOS and Android",
        },
        tags={"product", "launch", "mobile", "marketing"},
    )

    print(f"✅ Created goal: {goal_id}")

    # Decompose using the default LLM decomposer
    try:
        subgoal_ids = goalith.decompose_goal(
            goal_id,
            "llm_decomposer",  # Use the LLM decomposer
            context={
                "timeline": "3 months",
                "team_size": "5 people",
                "experience_level": "intermediate",
            },
        )

        print(f"✅ Decomposed into {len(subgoal_ids)} subgoals")

        # Display the decomposition results
        print("\n📋 Generated Subgoals:")
        for i, subgoal_id in enumerate(subgoal_ids, 1):
            subgoal = goalith.get_node(subgoal_id)
            print(f"\n{i}. {subgoal.description}")
            print(f"   Type: {subgoal.type}")
            print(f"   Priority: {subgoal.priority}")
            print(f"   Tags: {', '.join(subgoal.tags) if subgoal.tags else 'None'}")

            # Show LLM-specific context
            if "estimated_effort" in subgoal.context:
                print(f"   Estimated Effort: {subgoal.context['estimated_effort']}")
            if "notes" in subgoal.context:
                print(f"   Notes: {subgoal.context['notes']}")

        # Show decomposition metadata
        parent_goal = goalith.get_node(goal_id)
        if "llm_decomposition" in parent_goal.context:
            llm_meta = parent_goal.context["llm_decomposition"]
            print(f"\n🧠 LLM Decomposition Metadata:")
            print(f"   Strategy: {llm_meta.get('strategy', 'N/A')}")
            print(f"   Confidence: {llm_meta.get('confidence', 'N/A')}")
            print(f"   Timeline: {llm_meta.get('estimated_timeline', 'N/A')}")
            if llm_meta.get("potential_risks"):
                print(f"   Potential Risks: {', '.join(llm_meta['potential_risks'])}")

        # Print the complete goal graph
        print("\n🌳 Goal Graph Structure:")
        print("=" * 80)
        print_goal_graph(goalith, goal_id)
        print("=" * 80)

    except Exception as e:
        print(f"❌ Decomposition failed: {e}")


def example_contextual_llm_decomposition():
    """Example using the contextual LLM decomposer."""
    print("\n=== Contextual LLM Decomposition Example ===")

    # Create a contextual decomposer with domain knowledge
    domain_context = {
        "industry": "software development",
        "methodology": "agile",
        "team_structure": "cross-functional",
        "quality_standards": "high",
        "common_risks": ["scope creep", "technical debt", "resource constraints"],
        "best_practices": ["continuous integration", "code review", "user testing"],
    }

    contextual_decomposer = LLMDecomposer(
        domain_context=domain_context,
        include_historical_patterns=True,
        max_tokens=4000,  # Increase token limit for contextual decomposer
    )

    # Replace the default decomposer with our contextual one
    goalith = GoalithService()
    registry = goalith.get_decomposer_registry()
    if registry.has_decomposer("llm_decomposer"):
        registry.unregister("llm_decomposer")
    registry.register(contextual_decomposer)

    # Create a software development goal
    goal_id = goalith.create_goal(
        description="Develop a real-time chat feature for our web application",
        goal_type=NodeType.GOAL,
        priority=7.5,
        deadline=datetime.now() + timedelta(days=45),
        context={
            "technology_stack": "React, Node.js, Socket.io",
            "expected_users": "1000+ concurrent",
            "existing_codebase": "mature",
        },
        tags={"development", "feature", "real-time", "chat"},
    )

    print(f"✅ Created software development goal: {goal_id}")

    try:
        # Decompose using the contextual decomposer
        subgoal_ids = goalith.decompose_goal(
            goal_id,
            context={
                "sprint_duration": "2 weeks",
                "team_members": ["frontend dev", "backend dev", "QA engineer"],
                "constraints": [
                    "maintain backward compatibility",
                    "ensure scalability",
                ],
            },
        )

        print(f"✅ Contextual decomposition created {len(subgoal_ids)} subgoals")

        # Display results with enhanced context
        print("\n📋 Contextually Generated Subgoals:")
        for i, subgoal_id in enumerate(subgoal_ids, 1):
            subgoal = goalith.get_node(subgoal_id)
            print(f"\n{i}. {subgoal.description}")
            print(f"   Type: {subgoal.type}")
            print(f"   Priority: {subgoal.priority}")

            # Show dependencies if any
            if subgoal.dependencies:
                deps = [goalith.get_node(dep_id).description for dep_id in subgoal.dependencies]
                print(f"   Dependencies: {'; '.join(deps)}")

        # Print the complete goal graph
        print("\n🌳 Goal Graph Structure:")
        print("=" * 80)
        print_goal_graph(goalith, goal_id)
        print("=" * 80)

    except Exception as e:
        print(f"❌ Contextual decomposition failed: {e}")


def main():
    """Run all examples."""
    print("🚀 GoalithService LLM Decomposer Examples\n")

    # Check if required environment variables are set
    if not os.getenv("OPENROUTER_API_KEY"):
        print("⚠️  Warning: OPENROUTER_API_KEY not set. LLM decomposer may not work.")
        print("   Set your API key: export OPENROUTER_API_KEY='your-key-here'")

    try:
        example_basic_llm_decomposition()
        example_contextual_llm_decomposition()

        print("\n✅ All examples completed successfully!")

    except KeyboardInterrupt:
        print("\n❌ Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Examples failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
