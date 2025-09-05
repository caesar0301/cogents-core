#!/usr/bin/env python3
"""
LiteLLM Demo for Cogents

This demo showcases the LiteLLM client capabilities including:
- Basic chat completion
- Structured completion with Pydantic models
- Image understanding (vision)
- Embeddings and reranking
- Token usage tracking
- Error handling and retries

Requirements:
- Set appropriate environment variables for your chosen provider
- For OpenAI: OPENAI_API_KEY
- For Anthropic: ANTHROPIC_API_KEY
- For other providers: see LiteLLM documentation
- Optional: Configure Opik tracing for observability (OPIK_API_KEY)

Usage:
    python examples/litellm_demo.py
"""

import os
import sys
from pathlib import Path
from typing import List

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pydantic import BaseModel, Field

from cogents_core.llm.litellm import LLMClient
from cogents_core.tracing import get_token_tracker
from cogents_core.utils.logging_config import get_logger, setup_logging


class PersonalityAnalysis(BaseModel):
    """Structured model for personality analysis."""

    personality_type: str = Field(description="Primary personality type identified")
    traits: List[str] = Field(description="Key personality traits observed")
    strengths: List[str] = Field(description="Identified strengths")
    areas_for_growth: List[str] = Field(description="Areas that could be developed")
    confidence_score: float = Field(description="Confidence in analysis (0.0 to 1.0)")
    summary: str = Field(description="Brief summary of the personality analysis")


class TravelRecommendation(BaseModel):
    """Structured model for travel recommendations."""

    destination: str = Field(description="Recommended destination")
    best_time_to_visit: str = Field(description="Optimal time to visit")
    duration: str = Field(description="Recommended trip duration")
    budget_estimate: str = Field(description="Estimated budget range")
    top_attractions: List[str] = Field(description="Must-see attractions")
    activities: List[str] = Field(description="Recommended activities")
    travel_tips: List[str] = Field(description="Practical travel tips")
    confidence: float = Field(description="Confidence in recommendation (0.0 to 1.0)")


def setup_demo_logging():
    """Set up logging for the demo."""
    setup_logging(level="INFO", enable_colors=True)
    return get_logger(__name__)


def demo_basic_completion(client: LLMClient, logger):
    """Demonstrate basic chat completion."""
    logger.info("🚀 Testing basic chat completion...")

    messages = [
        {"role": "system", "content": "You are a helpful and creative assistant."},
        {"role": "user", "content": "Write a short poem about artificial intelligence and creativity."},
    ]

    try:
        response = client.completion(messages=messages, temperature=0.8, max_tokens=200)

        logger.info("✅ Basic completion successful!")
        print(f"\n📝 AI Poem:\n{response}\n")
        return True

    except Exception as e:
        logger.error(f"❌ Basic completion failed: {e}")
        return False


def demo_structured_completion(client: LLMClient, logger):
    """Demonstrate structured completion with Pydantic models."""
    logger.info("🔧 Testing structured completion...")

    # Test personality analysis
    messages = [
        {"role": "system", "content": "You are a professional personality analyst."},
        {
            "role": "user",
            "content": """
        Analyze the personality of someone who:
        - Loves solving complex puzzles and coding challenges
        - Prefers working alone but enjoys mentoring others
        - Gets excited about new technologies and learning
        - Sometimes procrastinates on routine tasks
        - Values efficiency and elegant solutions
        """,
        },
    ]

    try:
        analysis = client.structured_completion(
            messages=messages, response_model=PersonalityAnalysis, temperature=0.7, max_tokens=500
        )

        logger.info("✅ Structured completion successful!")
        print(f"\n🧠 Personality Analysis:")
        print(f"Type: {analysis.personality_type}")
        print(f"Traits: {', '.join(analysis.traits)}")
        print(f"Strengths: {', '.join(analysis.strengths)}")
        print(f"Growth Areas: {', '.join(analysis.areas_for_growth)}")
        print(f"Confidence: {analysis.confidence_score:.2f}")
        print(f"Summary: {analysis.summary}\n")
        return True

    except Exception as e:
        logger.error(f"❌ Structured completion failed: {e}")
        return False


def demo_travel_recommendation(client: LLMClient, logger):
    """Demonstrate another structured completion example."""
    logger.info("✈️ Testing travel recommendation...")

    messages = [
        {
            "role": "system",
            "content": "You are an expert travel advisor with extensive knowledge of global destinations.",
        },
        {
            "role": "user",
            "content": """
        I'm looking for a travel recommendation for:
        - A solo traveler who loves history and culture
        - Budget: $2000-3000 for a week
        - Interested in museums, ancient sites, and local cuisine
        - Prefers moderate climate
        - Available to travel in spring (March-May)
        """,
        },
    ]

    try:
        recommendation = client.structured_completion(
            messages=messages, response_model=TravelRecommendation, temperature=0.6, max_tokens=600
        )

        logger.info("✅ Travel recommendation successful!")
        print(f"\n🗺️ Travel Recommendation:")
        print(f"Destination: {recommendation.destination}")
        print(f"Best Time: {recommendation.best_time_to_visit}")
        print(f"Duration: {recommendation.duration}")
        print(f"Budget: {recommendation.budget_estimate}")
        print(f"Top Attractions: {', '.join(recommendation.top_attractions)}")
        print(f"Activities: {', '.join(recommendation.activities)}")
        print(f"Tips: {', '.join(recommendation.travel_tips)}")
        print(f"Confidence: {recommendation.confidence:.2f}\n")
        return True

    except Exception as e:
        logger.error(f"❌ Travel recommendation failed: {e}")
        return False


def demo_image_understanding(client: LLMClient, logger):
    """Demonstrate image understanding capabilities."""
    logger.info("🖼️ Testing image understanding...")

    # Test with a sample image URL (using a public domain image)
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
    prompt = "Describe this image in detail. What do you see? What colors and shapes are present?"

    try:
        analysis = client.understand_image_from_url(image_url=image_url, prompt=prompt, temperature=0.5, max_tokens=300)

        logger.info("✅ Image understanding successful!")
        print(f"\n🔍 Image Analysis:\n{analysis}\n")
        return True

    except Exception as e:
        logger.error(f"❌ Image understanding failed: {e}")
        logger.info("💡 Note: Image understanding requires a vision-capable model (e.g., gpt-4-vision-preview)")
        return False


def demo_embeddings_and_reranking(client: LLMClient, logger):
    """Demonstrate embeddings and reranking capabilities."""
    logger.info("🔢 Testing embeddings and reranking...")

    # Sample documents about different topics
    documents = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn without explicit programming.",
        "The Renaissance was a period of cultural rebirth in Europe from the 14th to 17th centuries.",
        "Quantum computing uses quantum mechanical phenomena to process information in fundamentally new ways.",
        "Climate change refers to long-term shifts in global temperatures and weather patterns.",
        "Blockchain technology provides a decentralized way to record transactions across multiple computers.",
    ]

    query = "What is artificial intelligence and machine learning?"

    try:
        # Test single embedding
        logger.info("Testing single text embedding...")
        embedding = client.embed(query)
        logger.info(f"✅ Generated embedding with {len(embedding)} dimensions")

        # Test batch embeddings
        logger.info("Testing batch embeddings...")
        embeddings = client.embed_batch(documents[:3])  # Test with first 3 documents
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
    """Demonstrate streaming completion."""
    logger.info("🌊 Testing streaming completion...")

    messages = [
        {"role": "system", "content": "You are a storyteller who creates engaging short stories."},
        {"role": "user", "content": "Tell me a short story about a robot who discovers emotions."},
    ]

    try:
        print("\n📖 Streaming Story:")
        print("-" * 50)

        response = client.completion(messages=messages, temperature=0.8, max_tokens=300, stream=True)

        # Handle streaming response
        full_response = ""
        for chunk in response:
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    content = delta.content
                    print(content, end="", flush=True)
                    full_response += content

        print("\n" + "-" * 50)
        logger.info("✅ Streaming completion successful!")
        return True

    except Exception as e:
        logger.error(f"❌ Streaming completion failed: {e}")
        return False


def demo_error_handling(client: LLMClient, logger):
    """Demonstrate error handling and retries."""
    logger.info("⚠️ Testing error handling...")

    # Test with an overly long message that might cause issues
    very_long_message = "Tell me about " + "artificial intelligence " * 1000
    messages = [{"role": "user", "content": very_long_message}]

    try:
        response = client.completion(
            messages=messages, temperature=0.7, max_tokens=50  # Very small limit to potentially trigger errors
        )

        logger.info("✅ Error handling test completed (no error occurred)")
        print(f"\n📝 Response: {response[:100]}...\n")
        return True

    except Exception as e:
        logger.info(f"✅ Error handling working correctly: {type(e).__name__}")
        return True


def print_token_usage_summary(logger):
    """Print token usage summary."""
    tracker = get_token_tracker()
    total_usage = tracker.get_total_usage()

    if total_usage.total_tokens > 0:
        logger.info("📊 Token Usage Summary:")
        print(f"Total Tokens: {total_usage.total_tokens}")
        print(f"Prompt Tokens: {total_usage.prompt_tokens}")
        print(f"Completion Tokens: {total_usage.completion_tokens}")
        print(f"Estimated Cost: ${total_usage.estimated_cost:.4f}")
        print()


def main():
    """Main demo function."""
    logger = setup_demo_logging()

    print("🎯 LiteLLM Demo for Cogents")
    print("=" * 50)

    # Check for API keys
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "Cohere": os.getenv("COHERE_API_KEY"),
    }

    available_providers = [name for name, key in api_keys.items() if key]

    if not available_providers:
        logger.warning("⚠️ No API keys found in environment variables")
        logger.info("💡 Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, COHERE_API_KEY")
        return

    logger.info(f"🔑 Available providers: {', '.join(available_providers)}")

    # Initialize client with different configurations
    configs = [
        {
            "name": "OpenAI GPT-3.5",
            "chat_model": "gpt-3.5-turbo",
            "vision_model": "gpt-4-vision-preview",
            "instructor": True,
        },
        {"name": "OpenAI GPT-4", "chat_model": "gpt-4", "vision_model": "gpt-4-vision-preview", "instructor": True},
        {
            "name": "Anthropic Claude",
            "chat_model": "claude-3-sonnet-20240229",
            "vision_model": "claude-3-sonnet-20240229",
            "instructor": False,  # Claude might need different structured output handling
        },
    ]

    # Try each configuration
    for config in configs:
        if config["name"].startswith("OpenAI") and not api_keys["OpenAI"]:
            continue
        if config["name"].startswith("Anthropic") and not api_keys["Anthropic"]:
            continue

        print(f"\n🤖 Testing with {config['name']}")
        print("-" * 30)

        try:
            client = LLMClient(
                chat_model=config["chat_model"], vision_model=config["vision_model"], instructor=config["instructor"]
            )

            # Run demos
            results = []
            results.append(demo_basic_completion(client, logger))

            if config["instructor"]:
                results.append(demo_structured_completion(client, logger))
                results.append(demo_travel_recommendation(client, logger))

            results.append(demo_embeddings_and_reranking(client, logger))
            results.append(demo_streaming_completion(client, logger))
            results.append(demo_image_understanding(client, logger))
            results.append(demo_error_handling(client, logger))

            # Print results summary
            successful = sum(results)
            total = len(results)
            logger.info(f"📈 {config['name']} Results: {successful}/{total} demos successful")

            print_token_usage_summary(logger)

            # Only test one working configuration to avoid excessive API usage
            if successful > 0:
                break

        except Exception as e:
            logger.error(f"❌ Failed to initialize {config['name']}: {e}")
            continue

    print("🎉 LiteLLM Demo completed!")
    print("💡 Check the logs above for detailed results and any issues.")


if __name__ == "__main__":
    main()
