#!/usr/bin/env python3
"""
Opik Tracing Example for Cogents

This example demonstrates how to use Opik tracing with the Cogents LLM clients.
Opik provides comprehensive observability and monitoring for LLM applications.

Requirements:
- Install opik: pip install opik
- Configure Opik: opik configure (or set environment variables)
- Set appropriate LLM API keys

Environment Variables for Opik:
- COGENTS_OPIK_TRACING: Global toggle for Opik tracing (default: false)
- OPIK_USE_LOCAL: Use local Opik deployment (default: true)
- OPIK_LOCAL_URL: Local Opik URL (default: http://localhost:5173)
- OPIK_API_KEY: Your Comet ML API key (only needed for cloud)
- OPIK_WORKSPACE: Your workspace name (optional)
- OPIK_PROJECT_NAME: Project name for organizing traces (default: cogents-llm)
- OPIK_URL: Custom Opik URL (for cloud deployment)
- OPIK_TRACING: Enable/disable tracing (default: true if enabled globally)

Local Deployment Setup:
1. Install Opik locally: pip install opik
2. Start local Opik server: opik local start
3. Set environment variables:
   export COGENTS_OPIK_TRACING=true
   export OPIK_USE_LOCAL=true
   # Optional: export OPIK_LOCAL_URL=http://localhost:5173

Usage:
    python examples/opik_tracing_example.py
"""

import os
import sys
from pathlib import Path
from typing import List

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pydantic import BaseModel, Field

from cogents_core.llm.litellm import LLMClient as LiteLLMClient
from cogents_core.llm.ollama import LLMClient as OllamaClient
from cogents_core.llm.openai import LLMClient as OpenAIClient
from cogents_core.utils.logging_config import get_logger, setup_logging

from .opik_tracing import configure_opik, get_opik_project, is_opik_enabled


class TaskAnalysis(BaseModel):
    """Structured model for task analysis."""

    task_type: str = Field(description="Type of task identified")
    complexity: str = Field(description="Task complexity level (simple, moderate, complex)")
    estimated_time: str = Field(description="Estimated time to complete")
    required_skills: List[str] = Field(description="Skills required to complete the task")
    dependencies: List[str] = Field(description="Task dependencies")
    confidence: float = Field(description="Confidence in analysis (0.0 to 1.0)")


def setup_demo_logging():
    """Set up logging for the demo."""
    setup_logging(level="INFO", enable_colors=True)
    return get_logger(__name__)


def check_opik_configuration(logger):
    """Check and display Opik configuration status."""
    logger.info("🔍 Checking Opik configuration...")

    # Check global Cogents toggle first
    cogents_opik_enabled = os.getenv("COGENTS_OPIK_TRACING", "false").lower() == "true"

    if not cogents_opik_enabled:
        logger.warning("⚠️ COGENTS_OPIK_TRACING not enabled")
        logger.info("💡 To enable Opik tracing:")
        logger.info("   1. Set: export COGENTS_OPIK_TRACING=true")
        logger.info("   2. Configure Opik: opik configure (or set environment variables)")
        logger.info("   3. For cloud: set OPIK_API_KEY")
        logger.info("   4. For local: ensure Opik server is running")
        return False

    # Check environment variables
    use_local = os.getenv("OPIK_USE_LOCAL", "true").lower() == "true"
    api_key = os.getenv("OPIK_API_KEY")
    workspace = os.getenv("OPIK_WORKSPACE")
    project = os.getenv("OPIK_PROJECT_NAME", "cogents-llm")
    tracing_enabled = os.getenv("OPIK_TRACING", "true").lower() == "true"

    if not tracing_enabled:
        logger.warning("⚠️ Opik tracing disabled via OPIK_TRACING=false")
        return False

    # For cloud deployment, API key is required
    if not use_local and not api_key:
        logger.warning("⚠️ OPIK_API_KEY not found for cloud deployment")
        logger.info("💡 To enable cloud Opik tracing:")
        logger.info("   1. Get your API key from https://www.comet.com/")
        logger.info("   2. Run: opik configure")
        logger.info("   3. Or set: export OPIK_API_KEY=your_api_key")
        logger.info("   4. Or use local deployment: export OPIK_USE_LOCAL=true")
        return False

    logger.info(f"🏠 Deployment: {'Local' if use_local else 'Cloud'}")
    if api_key:
        logger.info(f"✅ Opik API key: {'*' * 8}...{api_key[-4:]}")
    logger.info(f"📁 Workspace: {workspace or 'default'}")
    logger.info(f"📊 Project: {project}")

    # Test configuration
    configured = configure_opik()
    if configured and is_opik_enabled():
        current_project = get_opik_project()
        logger.info(f"✅ Opik tracing enabled for project: {current_project}")
        return True
    else:
        logger.error("❌ Failed to configure Opik tracing")
        return False


def demo_basic_tracing(client, client_name: str, logger):
    """Demonstrate basic LLM tracing with Opik."""
    logger.info(f"🚀 Testing basic tracing with {client_name}...")

    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides clear and concise answers."},
        {"role": "user", "content": "Explain the concept of machine learning in simple terms."},
    ]

    try:
        response = client.completion(messages=messages, temperature=0.7, max_tokens=200)

        logger.info(f"✅ {client_name} basic tracing successful!")
        print(f"\n📝 ML Explanation ({client_name}):\n{response}\n")
        return True

    except Exception as e:
        logger.error(f"❌ {client_name} basic tracing failed: {e}")
        return False


def demo_structured_tracing(client, client_name: str, logger):
    """Demonstrate structured completion tracing with Opik."""
    logger.info(f"🔧 Testing structured tracing with {client_name}...")

    messages = [
        {"role": "system", "content": "You are a project management expert who analyzes tasks."},
        {
            "role": "user",
            "content": """
        Analyze this task: "Build a web application that allows users to upload images, 
        apply various filters, and share the results on social media. The app should 
        support user authentication, have a responsive design, and include analytics."
        """,
        },
    ]

    try:
        analysis = client.structured_completion(
            messages=messages, response_model=TaskAnalysis, temperature=0.6, max_tokens=400
        )

        logger.info(f"✅ {client_name} structured tracing successful!")
        print(f"\n🔍 Task Analysis ({client_name}):")
        print(f"Type: {analysis.task_type}")
        print(f"Complexity: {analysis.complexity}")
        print(f"Estimated Time: {analysis.estimated_time}")
        print(f"Required Skills: {', '.join(analysis.required_skills)}")
        print(f"Dependencies: {', '.join(analysis.dependencies)}")
        print(f"Confidence: {analysis.confidence:.2f}\n")
        return True

    except Exception as e:
        logger.error(f"❌ {client_name} structured tracing failed: {e}")
        return False


def demo_multiple_calls_tracing(client, client_name: str, logger):
    """Demonstrate tracing multiple sequential calls."""
    logger.info(f"🔄 Testing multiple calls tracing with {client_name}...")

    topics = ["artificial intelligence", "quantum computing", "renewable energy"]
    results = []

    try:
        for topic in topics:
            messages = [
                {"role": "system", "content": "You are a technology expert who explains complex topics simply."},
                {"role": "user", "content": f"Give me a one-sentence summary of {topic}."},
            ]

            response = client.completion(messages=messages, temperature=0.5, max_tokens=100)

            results.append((topic, response))

        logger.info(f"✅ {client_name} multiple calls tracing successful!")
        print(f"\n📚 Topic Summaries ({client_name}):")
        for topic, summary in results:
            print(f"• {topic.title()}: {summary}")
        print()
        return True

    except Exception as e:
        logger.error(f"❌ {client_name} multiple calls tracing failed: {e}")
        return False


def demo_error_tracing(client, client_name: str, logger):
    """Demonstrate error tracing with Opik."""
    logger.info(f"⚠️ Testing error tracing with {client_name}...")

    # Intentionally create a scenario that might cause issues
    very_long_message = "Explain quantum mechanics " * 1000  # Very long input
    messages = [{"role": "user", "content": very_long_message}]

    try:
        response = client.completion(messages=messages, temperature=0.7, max_tokens=10)  # Very small limit

        logger.info(f"✅ {client_name} error tracing completed (no error occurred)")
        print(f"\n📝 Truncated Response ({client_name}): {response[:100]}...\n")
        return True

    except Exception as e:
        logger.info(f"✅ {client_name} error tracing working correctly: {type(e).__name__}")
        return True


def main():
    """Main demo function."""
    logger = setup_demo_logging()

    print("🔍 Opik Tracing Demo for Cogents")
    print("=" * 50)

    # Check Opik configuration
    if not check_opik_configuration(logger):
        print("\n💡 Opik tracing is not configured. The demo will run without tracing.")
        print("   LLM calls will still work, but won't be logged to Opik.")
        print("   To enable tracing, configure Opik and run the demo again.")

    print(f"\n🔧 Opik Status: {'Enabled' if is_opik_enabled() else 'Disabled'}")
    if is_opik_enabled():
        print(f"📊 Current Project: {get_opik_project()}")

    # Test different LLM clients
    clients_to_test = []

    # OpenAI client (if API key available)
    if os.getenv("OPENAI_API_KEY"):
        try:
            openai_client = OpenAIClient(chat_model="gpt-3.5-turbo", instructor=True)
            clients_to_test.append((openai_client, "OpenAI"))
        except Exception as e:
            logger.warning(f"Could not initialize OpenAI client: {e}")

    # LiteLLM client (if API key available)
    if os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
        try:
            litellm_client = LiteLLMClient(chat_model="gpt-3.5-turbo", instructor=True)
            clients_to_test.append((litellm_client, "LiteLLM"))
        except Exception as e:
            logger.warning(f"Could not initialize LiteLLM client: {e}")

    # Ollama client (if server is running)
    try:
        ollama_client = OllamaClient(chat_model="llama3.2", instructor=True)
        # Test connection
        test_response = ollama_client.completion(messages=[{"role": "user", "content": "Hello"}], max_tokens=5)
        clients_to_test.append((ollama_client, "Ollama"))
    except Exception as e:
        logger.warning(f"Could not initialize Ollama client: {e}")

    if not clients_to_test:
        logger.error("❌ No LLM clients available. Please configure at least one:")
        logger.info("   • OpenAI: Set OPENAI_API_KEY")
        logger.info("   • Ollama: Start Ollama server and pull models")
        return

    # Run demos for each available client
    for client, client_name in clients_to_test:
        print(f"\n🤖 Testing {client_name} Client")
        print("-" * 30)

        results = []
        results.append(demo_basic_tracing(client, client_name, logger))
        results.append(demo_structured_tracing(client, client_name, logger))
        results.append(demo_multiple_calls_tracing(client, client_name, logger))
        results.append(demo_error_tracing(client, client_name, logger))

        successful = sum(results)
        total = len(results)
        logger.info(f"📈 {client_name} Results: {successful}/{total} demos successful")

        # Only test one client to avoid excessive API usage
        break

    print("\n🎉 Opik Tracing Demo completed!")

    if is_opik_enabled():
        print("🔗 View your traces at: https://www.comet.com/")
        print(f"📊 Project: {get_opik_project()}")
        print("💡 Traces include:")
        print("   • Function calls and parameters")
        print("   • Input/output data")
        print("   • Execution times and metadata")
        print("   • Error information (if any)")
    else:
        print("💡 To see traces in Opik:")
        print("   1. Configure Opik with your API key")
        print("   2. Re-run this demo")
        print("   3. View traces in your Comet ML workspace")


if __name__ == "__main__":
    main()
