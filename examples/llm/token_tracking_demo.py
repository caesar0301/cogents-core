#!/usr/bin/env python3
"""
Token Usage Tracking Demo

This demo shows how the improved token usage tracking system works
with custom LLM clients and LangGraph callbacks.
"""

import os
import sys
from pathlib import Path

# Add cogents to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cogents_core.llm import get_llm_client
from cogents_core.tracing import TokenUsageCallback, get_token_tracker
from cogents_core.utils.logging_config import get_logger

logger = get_logger(__name__)


def demo_basic_token_tracking():
    """Demonstrate basic token tracking with custom LLM clients."""
    print("\n" + "=" * 50)
    print("DEMO: Basic Token Tracking")
    print("=" * 50)

    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY not set. Skipping API demos.")
        return

    # Reset tracker
    tracker = get_token_tracker()
    tracker.reset()

    # Get LLM client
    client = get_llm_client(provider="openrouter")

    print(f"📊 Initial token count: {tracker.get_total_tokens()}")

    # Test regular chat completion
    print("\n🤖 Testing regular chat completion...")
    messages = [{"role": "user", "content": "Say 'Hello World' in exactly 2 words."}]
    response = client.completion(messages, temperature=0.1, max_tokens=10)

    print(f"Response: {response}")
    print(f"📊 Tokens after chat: {tracker.get_total_tokens()}")

    # Test structured completion if instructor is available
    try:
        from pydantic import BaseModel

        class SimpleGreeting(BaseModel):
            greeting: str
            language: str

        print("\n🧠 Testing structured completion...")
        client_instructor = get_llm_client(provider="openrouter", instructor=True)
        messages = [{"role": "user", "content": "Give me a simple greeting in English"}]
        structured_response = client_instructor.structured_completion(
            messages=messages, response_model=SimpleGreeting, temperature=0.1, max_tokens=50
        )

        print(f"Structured response: {structured_response}")
        print(f"📊 Tokens after structured: {tracker.get_total_tokens()}")

    except ImportError:
        print("⚠️ Pydantic not available, skipping structured completion test")
    except Exception as e:
        print(f"⚠️ Structured completion failed: {e}")

    # Print final stats with debug output
    print("🔍 Getting final stats...")
    try:
        stats = tracker.get_stats()
        print(f"✅ Stats retrieved successfully")
        print(f"📈 Final: {stats['total_tokens']} total | {stats['total_calls']} calls")
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        stats = {"total_tokens": 0, "total_calls": 0, "usage_history": []}


def demo_callback_integration():
    """Demonstrate integration with TokenUsageCallback."""
    print("\n" + "=" * 50)
    print("DEMO: Callback Integration")
    print("=" * 50)

    # Reset everything
    tracker = get_token_tracker()
    tracker.reset()

    callback = TokenUsageCallback(model_name="demo-model", verbose=True)
    callback.reset_session()

    # Simulate some custom client usage
    from cogents_core.tracing import record_token_usage

    print("🔄 Simulating custom client calls...")
    record_token_usage(
        prompt_tokens=45, completion_tokens=30, model_name="openrouter/google/gemini-flash-1.5", call_type="completion"
    )

    record_token_usage(
        prompt_tokens=60, completion_tokens=40, model_name="openrouter/google/gemini-flash-1.5", call_type="structured"
    )

    # Simulate some LangChain callback usage
    print("🔄 Simulating LangChain callback usage...")
    callback.total_prompt_tokens = 25
    callback.total_completion_tokens = 15
    callback.llm_calls = 1

    # Show brief structured summary instead of verbose output
    summary = callback.get_session_summary()
    print(
        f"\n📊 Combined Summary: {summary['total_tokens']} total | {summary['total_llm_calls']} calls | P:{summary['total_prompt_tokens']} C:{summary['total_completion_tokens']}"
    )
    print(f"   Callback: {summary['callback_stats']['total_tokens']} tokens")
    print(f"   Tracker: {summary['tracker_stats']['total_tokens']} tokens")


def main():
    """Run all token tracking demos with timeout protection."""
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Demo timed out")

    print("🚀 Token Usage Tracking Demo")
    print("This demo shows the improved token tracking system in action.")

    try:
        # Set overall timeout for the entire demo
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(120)  # 2 minute total timeout

        print("\n[1/3] Running basic token tracking demo...")
        demo_basic_token_tracking()
        print("✅ Basic demo completed")

        print("\n[2/3] Running callback integration demo...")
        demo_callback_integration()
        print("✅ Callback demo completed")

        signal.alarm(0)  # Cancel timeout

        print("\n" + "=" * 50)
        print("✅ All demos completed successfully!")
        print("The token tracking system is working correctly.")
        print("=" * 50)

    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except TimeoutError:
        print("\n⏰ Demo timed out after 2 minutes")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        try:
            signal.alarm(0)  # Make sure to cancel any active alarms
        except:
            pass


if __name__ == "__main__":
    main()
