#!/usr/bin/env python3
"""
Ollama Demo for CogentNano

This demo showcases the Ollama client capabilities including:
- Basic chat completion with local models
- Structured completion with Pydantic models
- Image understanding with vision models (gemma3:4b)
- Embeddings and reranking
- Token usage estimation
- Model management and configuration

Requirements:
- Ollama server running locally (default: http://localhost:11434)
- Models downloaded: gemma3:4b (or configure via environment variables)
- Optional: Set OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL, OLLAMA_VISION_MODEL
- Optional: Configure Opik tracing for observability (OPIK_API_KEY)

Installation:
    # Install Ollama
    curl -fsSL https://ollama.ai/install.sh | sh
    
    # Pull models
    ollama pull gemma3:4b

Usage:
    python examples/ollama_demo.py
"""

import os
import sys
import time
from pathlib import Path
from typing import List

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pydantic import BaseModel, Field

from cogents.core.base.llm.ollama import LLMClient
from cogents.core.base.logging import get_logger, setup_logging
from cogents.core.base.tracing import get_token_tracker


class CodeAnalysis(BaseModel):
    """Structured model for code analysis."""

    language: str = Field(description="Programming language identified")
    complexity: str = Field(description="Code complexity level (simple, moderate, complex)")
    main_purpose: str = Field(description="Primary purpose of the code")
    key_functions: List[str] = Field(description="Key functions or methods identified")
    potential_issues: List[str] = Field(description="Potential issues or improvements")
    code_quality: float = Field(description="Code quality score (0.0 to 10.0)")
    suggestions: List[str] = Field(description="Improvement suggestions")


class StoryOutline(BaseModel):
    """Structured model for story outline."""

    title: str = Field(description="Story title")
    genre: str = Field(description="Story genre")
    setting: str = Field(description="Story setting (time and place)")
    main_character: str = Field(description="Main character description")
    conflict: str = Field(description="Central conflict or problem")
    plot_points: List[str] = Field(description="Key plot points")
    theme: str = Field(description="Central theme or message")
    target_length: str = Field(description="Estimated story length")


def setup_demo_logging():
    """Set up logging for the demo."""
    setup_logging(level="INFO", enable_colors=True)
    return get_logger(__name__)


def check_ollama_connection(client: LLMClient, logger):
    """Check if Ollama server is running and accessible."""
    logger.info("🔍 Checking Ollama server connection...")

    try:
        # Try a simple completion to test connection
        response = client.completion(messages=[{"role": "user", "content": "Hello"}], max_tokens=10)
        logger.info("✅ Ollama server is accessible")
        return True
    except Exception as e:
        logger.error(f"❌ Cannot connect to Ollama server: {e}")
        logger.info("💡 Make sure Ollama is running: 'ollama serve'")
        logger.info("💡 And required models are installed: 'ollama pull gemma3:4b'")
        return False


def demo_basic_completion(client: LLMClient, logger):
    """Demonstrate basic chat completion with Ollama."""
    logger.info("🚀 Testing basic chat completion...")

    messages = [
        {"role": "system", "content": "You are a helpful coding assistant with expertise in Python."},
        {"role": "user", "content": "Explain the difference between lists and tuples in Python in simple terms."},
    ]

    try:
        response = client.completion(messages=messages, temperature=0.7, max_tokens=300)

        logger.info("✅ Basic completion successful!")
        print(f"\n📝 Python Explanation:\n{response}\n")
        return True

    except Exception as e:
        logger.error(f"❌ Basic completion failed: {e}")
        return False


def demo_structured_completion(client: LLMClient, logger):
    """Demonstrate structured completion with Pydantic models."""
    logger.info("🔧 Testing structured completion...")

    code_sample = """
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def main():
    for i in range(10):
        print(f"F({i}) = {fibonacci(i)}")

if __name__ == "__main__":
    main()
"""

    messages = [
        {"role": "system", "content": "You are an expert code reviewer and analyst."},
        {"role": "user", "content": f"Analyze this Python code:\n\n{code_sample}"},
    ]

    try:
        analysis = client.structured_completion(
            messages=messages, response_model=CodeAnalysis, temperature=0.5, max_tokens=2000
        )

        logger.info("✅ Structured completion successful!")
        print(f"\n🔍 Code Analysis:")
        print(f"Language: {analysis.language}")
        print(f"Complexity: {analysis.complexity}")
        print(f"Purpose: {analysis.main_purpose}")
        print(f"Key Functions: {', '.join(analysis.key_functions)}")
        print(f"Quality Score: {analysis.code_quality}/10")
        print(f"Issues: {', '.join(analysis.potential_issues)}")
        print(f"Suggestions: {', '.join(analysis.suggestions)}\n")
        return True

    except Exception as e:
        logger.error(f"❌ Structured completion failed: {e}")
        logger.info("💡 Note: Structured completion requires instructor integration")
        return False


def demo_creative_structured_completion(client: LLMClient, logger):
    """Demonstrate creative structured completion."""
    logger.info("✨ Testing creative structured completion...")

    messages = [
        {"role": "system", "content": "You are a creative writing assistant and story planner."},
        {
            "role": "user",
            "content": """
        Create a story outline for a science fiction short story about:
        - A programmer who discovers their code is creating sentient AI
        - Set in near-future (2030s)
        - Should explore themes of consciousness and responsibility
        - Target length: 5000-7000 words
        """,
        },
    ]

    try:
        outline = client.structured_completion(
            messages=messages, response_model=StoryOutline, temperature=0.8, max_tokens=3000
        )

        logger.info("✅ Creative structured completion successful!")
        print(f"\n📚 Story Outline:")
        print(f"Title: {outline.title}")
        print(f"Genre: {outline.genre}")
        print(f"Setting: {outline.setting}")
        print(f"Main Character: {outline.main_character}")
        print(f"Conflict: {outline.conflict}")
        print(f"Theme: {outline.theme}")
        print(f"Target Length: {outline.target_length}")
        print(f"Plot Points:")
        for i, point in enumerate(outline.plot_points, 1):
            print(f"  {i}. {point}")
        print()
        return True

    except Exception as e:
        logger.error(f"❌ Creative structured completion failed: {e}")
        return False


def demo_image_understanding(client: LLMClient, logger):
    """Demonstrate image understanding with gemma3:4b model."""
    logger.info("🖼️ Testing image understanding...")

    # Create a simple test image if it doesn't exist
    test_image_path = Path("test_image.png")

    if not test_image_path.exists():
        logger.info("📷 Creating test image...")
        try:
            from PIL import Image, ImageDraw, ImageFont  # type: ignore # Optional dependency

            # Create a simple test image
            img = Image.new("RGB", (400, 300), color="lightblue")
            draw = ImageDraw.Draw(img)

            # Draw some shapes
            draw.rectangle([50, 50, 150, 150], fill="red", outline="black", width=3)
            draw.ellipse([200, 50, 350, 200], fill="green", outline="black", width=3)
            draw.polygon([(100, 200), (150, 250), (200, 200), (175, 170), (125, 170)], fill="yellow", outline="black")

            # Add text
            try:
                font = ImageFont.load_default()
                draw.text((50, 260), "Test Image for Ollama Demo", fill="black", font=font)
            except:
                draw.text((50, 260), "Test Image for Ollama Demo", fill="black")

            img.save(test_image_path)
            logger.info(f"✅ Created test image: {test_image_path}")

        except ImportError:
            logger.warning("⚠️ PIL not available, skipping image creation")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to create test image: {e}")
            return False

    prompt = "Describe this image in detail. What shapes, colors, and text do you see? What is the overall composition?"

    try:
        analysis = client.understand_image(image_path=test_image_path, prompt=prompt, temperature=0.5, max_tokens=400)

        logger.info("✅ Image understanding successful!")
        print(f"\n🔍 Image Analysis:\n{analysis}\n")

        # Clean up test image
        if test_image_path.exists():
            test_image_path.unlink()
            logger.info("🧹 Cleaned up test image")

        return True

    except Exception as e:
        logger.error(f"❌ Image understanding failed: {e}")
        logger.info("💡 Note: Make sure 'gemma3:4b' model is installed: 'ollama pull gemma3:4b'")
        return False


def demo_embeddings_and_reranking(client: LLMClient, logger):
    """Demonstrate embeddings and reranking with Ollama."""
    logger.info("🔢 Testing embeddings and reranking...")

    # Sample programming-related documents
    documents = [
        "Python is an interpreted, high-level programming language with dynamic semantics and simple syntax.",
        "JavaScript is a programming language that enables interactive web pages and is an essential part of web applications.",
        "Machine learning algorithms can automatically improve through experience without being explicitly programmed.",
        "Git is a distributed version control system for tracking changes in source code during software development.",
        "Docker containers package applications with their dependencies for consistent deployment across environments.",
        "REST APIs provide a standardized way for different software applications to communicate over HTTP.",
    ]

    query = "What programming languages are good for beginners?"

    try:
        # Test single embedding
        logger.info("Testing single text embedding...")
        embedding = client.embed(query)
        logger.info(f"✅ Generated embedding with {len(embedding)} dimensions")

        # Test batch embeddings (smaller batch for Ollama)
        logger.info("Testing batch embeddings...")
        embeddings = client.embed_batch(documents[:3])
        logger.info(f"✅ Generated {len(embeddings)} embeddings")

        # Test reranking
        logger.info("Testing document reranking...")
        reranked_docs = client.rerank(query, documents)

        logger.info("✅ Reranking successful!")
        print(f"\n🔍 Query: {query}")
        print("\n📊 Reranked Documents (most relevant first):")
        for i, doc in enumerate(reranked_docs[:3], 1):
            print(f"{i}. {doc}")
        print()
        return True

    except Exception as e:
        logger.error(f"❌ Embeddings/reranking failed: {e}")
        return False


def demo_streaming_completion(client: LLMClient, logger):
    """Demonstrate streaming completion with Ollama."""
    logger.info("🌊 Testing streaming completion...")

    messages = [
        {"role": "system", "content": "You are a technical writer who explains complex concepts clearly."},
        {
            "role": "user",
            "content": "Explain how neural networks work, using simple analogies that a beginner could understand.",
        },
    ]

    try:
        print("\n🧠 Neural Networks Explanation:")
        print("-" * 50)

        response = client.completion(messages=messages, temperature=0.7, max_tokens=400, stream=True)

        # Handle Ollama streaming response format
        full_response = ""
        for chunk in response:
            if isinstance(chunk, dict) and "message" in chunk:
                content = chunk["message"].get("content", "")
                if content:
                    print(content, end="", flush=True)
                    full_response += content
            elif hasattr(chunk, "message") and hasattr(chunk.message, "content"):
                content = chunk.message.content
                if content:
                    print(content, end="", flush=True)
                    full_response += content

        print("\n" + "-" * 50)
        logger.info("✅ Streaming completion successful!")
        return True

    except Exception as e:
        logger.error(f"❌ Streaming completion failed: {e}")
        return False


def demo_model_performance(client: LLMClient, logger):
    """Demonstrate model performance characteristics."""
    logger.info("⚡ Testing model performance...")

    messages = [{"role": "user", "content": "Count from 1 to 10 and explain what each number represents in binary."}]

    try:
        start_time = time.time()

        response = client.completion(messages=messages, temperature=0.3, max_tokens=300)

        end_time = time.time()
        response_time = end_time - start_time

        logger.info("✅ Performance test successful!")
        print(f"\n⏱️ Response Time: {response_time:.2f} seconds")
        print(f"📝 Response Preview: {response[:200]}...")
        if len(response) > 200:
            print("...")
        print()
        return True

    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        return False


def demo_error_handling(client: LLMClient, logger):
    """Demonstrate error handling with Ollama."""
    logger.info("⚠️ Testing error handling...")

    # Test with a very short max_tokens to see how it handles truncation
    messages = [{"role": "user", "content": "Write a detailed essay about the history of computer science."}]

    try:
        response = client.completion(messages=messages, temperature=0.7, max_tokens=20)  # Very small limit

        logger.info("✅ Error handling test completed")
        print(f"\n📝 Truncated Response: {response}\n")
        return True

    except Exception as e:
        logger.info(f"✅ Error handling working correctly: {type(e).__name__}")
        return True


def print_token_usage_summary(logger):
    """Print estimated token usage summary."""
    tracker = get_token_tracker()
    stats = tracker.get_stats()

    if stats["total_tokens"] > 0:
        logger.info("📊 Estimated Token Usage Summary:")
        print(f"Total Tokens: {stats['total_tokens']}")
        print(f"Prompt Tokens: {stats['total_prompt_tokens']}")
        print(f"Completion Tokens: {stats['total_completion_tokens']}")
        print(f"Total Calls: {stats['total_calls']}")
        print("💡 Note: Ollama token counts are estimated")
        print()


def list_available_models(client: LLMClient, logger):
    """List available Ollama models."""
    logger.info("📋 Checking available models...")

    try:
        # Try to get model list (this is Ollama-specific)
        models_response = client.client.list()
        if "models" in models_response:
            models = [model["name"] for model in models_response["models"]]
            logger.info(f"✅ Available models: {', '.join(models)}")
            return models
        else:
            logger.info("ℹ️ Could not retrieve model list")
            return []
    except Exception as e:
        logger.info(f"ℹ️ Could not list models: {e}")
        return []


def main():
    """Main demo function."""
    logger = setup_demo_logging()

    print("🦙 Ollama Demo for CogentNano")
    print("=" * 50)

    # Configuration
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    chat_model = os.getenv("OLLAMA_CHAT_MODEL", "gemma3:4b")
    vision_model = os.getenv("OLLAMA_VISION_MODEL", "gemma3:4b")

    logger.info(f"🔗 Ollama URL: {base_url}")
    logger.info(f"💬 Chat Model: {chat_model}")
    logger.info(f"👁️ Vision Model: {vision_model}")

    try:
        # Initialize client
        client = LLMClient(
            base_url=base_url,
            chat_model=chat_model,
            vision_model=vision_model,
            instructor=True,  # Enable structured output
        )

        # Check connection first
        if not check_ollama_connection(client, logger):
            return

        # List available models
        list_available_models(client, logger)

        print(f"\n🤖 Testing with Ollama Models")
        print("-" * 30)

        # Run demos
        results = []
        results.append(demo_basic_completion(client, logger))
        results.append(demo_structured_completion(client, logger))
        results.append(demo_creative_structured_completion(client, logger))
        results.append(demo_embeddings_and_reranking(client, logger))
        results.append(demo_streaming_completion(client, logger))
        results.append(demo_model_performance(client, logger))
        results.append(demo_image_understanding(client, logger))
        results.append(demo_error_handling(client, logger))

        # Print results summary
        successful = sum(results)
        total = len(results)
        logger.info(f"📈 Ollama Results: {successful}/{total} demos successful")

        print_token_usage_summary(logger)

    except Exception as e:
        logger.error(f"❌ Failed to initialize Ollama client: {e}")
        logger.info("💡 Troubleshooting:")
        logger.info("   1. Make sure Ollama is installed and running")
        logger.info("   2. Check if models are downloaded: 'ollama list'")
        logger.info("   3. Pull required models: 'ollama pull gemma3:4b'")
        logger.info("   4. Verify server is accessible: curl http://localhost:11434")
        return

    print("🎉 Ollama Demo completed!")
    print("💡 Check the logs above for detailed results and any issues.")
    print("🔧 To install more models: 'ollama pull <model-name>'")


if __name__ == "__main__":
    main()
