#!/usr/bin/env python3
"""
LlamaCpp Local LLM Demo

This demo demonstrates how to use the LlamaCpp LLM client for local inference.
It shows basic completion, structured output, and model management.

Prerequisites:
1. Install llama-cpp-python: pip install llama-cpp-python
2. Download a compatible GGUF model file
3. Set LLAMACPP_MODEL_PATH environment variable or provide model path

Example model download (optional):
    python download_model.py  # Downloads gemma-3-270m-it-GGUF model
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Add cogents to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cogents.core.base.llm.llamacpp import LLMClient
from cogents.core.base.logging import get_logger, setup_logging
from cogents.core.base.tracing import get_token_tracker

logger = get_logger(__name__)


def download_demo_model() -> Optional[str]:
    """
    Download a small demo model if no model path is configured.
    Returns the path to the downloaded model directory.
    """
    print("🔄 No model path found. Downloading a small demo model...")
    try:
        from huggingface_hub import snapshot_download

        # Download a small gemma model (around 270MB)
        local_dir = snapshot_download("ggml-org/gemma-3-270m-it-GGUF", local_dir="./models/gemma-3-270m-it-GGUF")

        # Find the .gguf file in the directory
        model_files = list(Path(local_dir).glob("*.gguf"))
        if model_files:
            model_path = str(model_files[0])
            print(f"✅ Model downloaded to: {model_path}")
            return model_path
        else:
            print("❌ No .gguf files found in downloaded directory")
            return None

    except ImportError:
        print("❌ huggingface_hub not installed. Please install it or provide a model path.")
        print("   pip install huggingface_hub")
        return None
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return None


def demo_basic_completion(client: LLMClient):
    """Demonstrate basic completion with LlamaCpp."""
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Chat Completion")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "You are a helpful assistant that gives concise answers."},
        {"role": "user", "content": "What is the capital of France? Answer in one sentence."},
    ]

    print("💬 Messages:")
    for msg in messages:
        print(f"   {msg['role'].capitalize()}: {msg['content']}")

    print("\n🤖 Generating response...")
    try:
        response = client.completion(messages=messages, temperature=0.7, max_tokens=100)
        print(f"✅ Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")


def demo_conversation(client: LLMClient):
    """Demonstrate a multi-turn conversation."""
    print("\n" + "=" * 60)
    print("DEMO 2: Multi-turn Conversation")
    print("=" * 60)

    # Initialize conversation history
    conversation = [
        {"role": "system", "content": "You are a knowledgeable assistant. Keep responses concise."},
    ]

    # Conversation turns
    user_messages = [
        "Tell me about Python programming in one sentence.",
        "What makes it popular?",
        "Give me one simple example of Python code.",
    ]

    for i, user_input in enumerate(user_messages, 1):
        print(f"\n--- Turn {i} ---")
        conversation.append({"role": "user", "content": user_input})
        print(f"👤 User: {user_input}")

        try:
            response = client.completion(messages=conversation, temperature=0.7, max_tokens=150)
            print(f"🤖 Assistant: {response}")
            conversation.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"❌ Error: {e}")
            break


def demo_structured_output(client: LLMClient):
    """Demonstrate structured output generation."""
    print("\n" + "=" * 60)
    print("DEMO 3: Structured Output")
    print("=" * 60)

    try:
        from pydantic import BaseModel

        class BookRecommendation(BaseModel):
            title: str
            author: str
            genre: str
            rating: float
            reason: str

        print("📚 Requesting structured book recommendation...")

        # Create a client with instructor enabled for structured output
        try:
            structured_client = LLMClient(
                model_path=client.model_path, instructor=True, n_ctx=2048, n_gpu_layers=client.llama.params.n_gpu_layers
            )
        except:
            print("⚠️ Could not create instructor-enabled client, using fallback approach...")
            structured_client = client

        messages = [
            {
                "role": "user",
                "content": "Recommend a good science fiction book. Include title, author, genre, rating (1-5), and brief reason why you recommend it.",
            }
        ]

        if hasattr(structured_client, "structured_completion") and structured_client.instructor_enabled:
            response = structured_client.structured_completion(
                messages=messages, response_model=BookRecommendation, temperature=0.7, max_tokens=200
            )
            print(f"✅ Structured Response:")
            print(f"   📖 Title: {response.title}")
            print(f"   ✍️  Author: {response.author}")
            print(f"   🎭 Genre: {response.genre}")
            print(f"   ⭐ Rating: {response.rating}/5")
            print(f"   💭 Reason: {response.reason}")
        else:
            # Fallback to regular completion with JSON instruction
            messages[0][
                "content"
            ] += "\n\nPlease respond with a JSON object with fields: title, author, genre, rating, reason."
            response = client.completion(messages=messages, temperature=0.7, max_tokens=200)
            print(f"✅ JSON Response: {response}")

    except ImportError:
        print("⚠️ Pydantic not available. Install it for structured output: pip install pydantic")
    except Exception as e:
        print(f"❌ Structured output error: {e}")


def demo_token_tracking(client: LLMClient):
    """Demonstrate token usage tracking."""
    print("\n" + "=" * 60)
    print("DEMO 4: Token Usage Tracking")
    print("=" * 60)

    # Reset tracker
    tracker = get_token_tracker()
    tracker.reset()

    print(f"📊 Initial token count: {tracker.get_total_tokens()}")

    # Make a few calls
    test_messages = [
        [{"role": "user", "content": "Count from 1 to 5."}],
        [{"role": "user", "content": "What is 2 + 2?"}],
        [{"role": "user", "content": "Name a color."}],
    ]

    for i, messages in enumerate(test_messages, 1):
        print(f"\n🔄 Test call {i}: {messages[0]['content']}")
        try:
            response = client.completion(messages, max_tokens=50, temperature=0.5)
            print(f"   Response: {response}")
            print(f"   📊 Tokens after call {i}: {tracker.get_total_tokens()}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    # Show final stats
    try:
        stats = tracker.get_stats()
        print(f"\n📈 Final Statistics:")
        print(f"   Total tokens: {stats['total_tokens']}")
        print(f"   Total calls: {stats['total_calls']}")
    except Exception as e:
        print(f"❌ Could not get stats: {e}")


def demo_model_parameters(client: LLMClient):
    """Demonstrate different model parameters."""
    print("\n" + "=" * 60)
    print("DEMO 5: Model Parameters")
    print("=" * 60)

    message = [{"role": "user", "content": "Write a creative one-line story about a robot."}]

    # Test different temperature values
    temperatures = [0.1, 0.7, 1.2]

    for temp in temperatures:
        print(f"\n🌡️  Temperature {temp}:")
        try:
            response = client.completion(messages=message, temperature=temp, max_tokens=100)
            print(f"   {response}")
        except Exception as e:
            print(f"   ❌ Error: {e}")


def check_model_path() -> Optional[str]:
    """
    Check for model path from environment or offer to download.
    Returns the model path or None if not available.
    """
    # Check environment variable
    model_path = os.getenv("LLAMACPP_MODEL_PATH")
    if model_path and os.path.exists(model_path):
        print(f"✅ Using model from LLAMACPP_MODEL_PATH: {model_path}")
        return model_path

    # Check if download_model.py was used
    default_paths = ["./models/gemma-3-270m-it-GGUF", "./gemma-3-270m-it-GGUF"]

    for path in default_paths:
        if os.path.exists(path):
            gguf_files = list(Path(path).glob("*.gguf"))
            if gguf_files:
                model_path = str(gguf_files[0])
                print(f"✅ Found existing model: {model_path}")
                return model_path

    # Offer to download
    print("\n📥 No model found. Options:")
    print("1. Set LLAMACPP_MODEL_PATH environment variable to your .gguf file")
    print("2. Download a demo model automatically")
    print("3. Exit and set up manually")

    choice = input("\nEnter choice (1/2/3): ").strip()

    if choice == "2":
        return download_demo_model()
    elif choice == "1":
        model_path = input("Enter path to your .gguf model file: ").strip()
        if os.path.exists(model_path):
            return model_path
        else:
            print(f"❌ File not found: {model_path}")
            return None
    else:
        print("👋 Exiting. Set up your model and try again.")
        return None


def main():
    """Run all LlamaCpp demos."""
    print("🚀 LlamaCpp Local LLM Demo")
    print("This demo shows how to use local LLMs with the LlamaCpp client.")

    # Setup logging
    setup_logging(level="INFO", enable_colors=True)

    try:
        # Check model availability
        model_path = check_model_path()
        if not model_path:
            print("❌ No model available. Exiting.")
            return

        print(f"\n🔄 Initializing LlamaCpp client with model: {Path(model_path).name}")

        # Initialize client with reasonable defaults
        client = LLMClient(
            model_path=model_path,
            n_ctx=2048,  # Context window size
            n_gpu_layers=-1,  # Use all GPU layers if available
            verbose=False,
        )

        print("✅ Client initialized successfully!")

        # Run demos
        print(f"\n{'='*80}")
        print("Running LlamaCpp demos...")
        print("Press Ctrl+C to stop at any time.")
        print("=" * 80)

        demo_basic_completion(client)
        demo_conversation(client)
        demo_structured_output(client)
        demo_token_tracking(client)
        demo_model_parameters(client)

        print("\n" + "=" * 80)
        print("🎉 All demos completed successfully!")
        print("The LlamaCpp client is working correctly with your local model.")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except FileNotFoundError as e:
        print(f"\n❌ Model file not found: {e}")
        print("Please check your model path or download a model first.")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        logger.exception("Full error details:")

    print("\n💡 Tips for using LlamaCpp:")
    print("• Set LLAMACPP_MODEL_PATH to avoid specifying the path each time")
    print("• Use n_gpu_layers=-1 to utilize all available GPU memory")
    print("• Adjust n_ctx based on your model and memory constraints")
    print("• Lower temperature (0.1-0.3) for consistent outputs")
    print("• Higher temperature (0.7-1.2) for creative outputs")


if __name__ == "__main__":
    main()
