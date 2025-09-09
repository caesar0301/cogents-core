#!/usr/bin/env python3
"""
DashScope (Alibaba Cloud) LLM Demo for Cogents

This demo showcases the DashScope client capabilities using OpenAI-compatible API:
- Basic chat completion with Qwen models
- Structured completion with Pydantic models
- Embeddings with DashScope embedding models
- Vision understanding with multimodal models
- Token usage tracking and error handling

DashScope is Alibaba Cloud's AI service platform offering:
- Qwen series language models (qwen-turbo, qwen-plus, qwen-max)
- Text embedding models (text-embedding-v1, text-embedding-v2)
- Multimodal models for vision understanding

Requirements:
- DashScope API key from Alibaba Cloud
- Set DASHSCOPE_API_KEY in your .env file
- Optional: Configure specific model names via environment variables

API Documentation:
- Base URL: https://dashscope.aliyuncs.com/compatible-mode/v1
- Models: qwen-turbo, qwen-plus, qwen-max, qwen-vl-plus, text-embedding-v2
- Compatible with OpenAI SDK through compatibility mode

Usage:
    cp env.dashscope .env
    python examples/llm/dashcope_llm_demo.py
"""

import os
import sys
from pathlib import Path
from typing import List

import requests

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pydantic import BaseModel, Field

from cogents_core.llm import get_llm_client
from cogents_core.tracing import get_token_tracker
from cogents_core.utils.logging import get_logger, setup_logging


class TextAnalysis(BaseModel):
    """Structured model for text analysis."""

    language: str = Field(description="Primary language detected")
    sentiment: str = Field(description="Overall sentiment (positive, negative, neutral)")
    key_topics: List[str] = Field(description="Main topics or themes identified")
    complexity_level: str = Field(description="Text complexity (simple, intermediate, advanced)")
    word_count: int = Field(description="Approximate word count")
    summary: str = Field(description="Brief summary of the text")
    confidence_score: float = Field(description="Analysis confidence (0.0 to 1.0)")


class ProductRecommendation(BaseModel):
    """Structured model for product recommendations."""

    product_name: str = Field(description="Recommended product name")
    category: str = Field(description="Product category")
    price_range: str = Field(description="Estimated price range")
    key_features: List[str] = Field(description="Important product features")
    pros: List[str] = Field(description="Product advantages")
    cons: List[str] = Field(description="Potential drawbacks")
    target_audience: str = Field(description="Who this product is best for")
    alternatives: List[str] = Field(description="Alternative product suggestions")
    recommendation_score: float = Field(description="Recommendation strength (0.0 to 1.0)")


def setup_demo_logging():
    """Set up logging for the demo."""
    setup_logging(level="INFO", enable_colors=True)
    return get_logger(__name__)


def check_dashscope_config(logger):
    """Check DashScope configuration and API key."""
    api_key = os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        logger.error("❌ DASHSCOPE_API_KEY not found in environment variables")
        logger.info("💡 Please set DASHSCOPE_API_KEY in your .env file")
        logger.info("💡 Get your API key from: https://dashscope.console.aliyun.com/")
        return False

    logger.info("✅ DashScope API key configured")
    return True


def demo_basic_completion(client, logger):
    """Demonstrate basic chat completion with DashScope."""
    logger.info("🚀 Testing basic chat completion with DashScope...")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant that can answer questions in both English and Chinese. Please be concise and informative.",
        },
        {
            "role": "user",
            "content": "Explain the concept of artificial intelligence and its applications in modern technology. Please provide examples.",
        },
    ]

    try:
        response = client.completion(messages=messages, temperature=0.7, max_tokens=500)

        logger.info("✅ Basic completion successful!")
        print(f"\n📝 AI Explanation:\n{response}\n")
        return True

    except Exception as e:
        logger.error(f"❌ Basic completion failed: {e}")
        return False


def demo_chinese_completion(client, logger):
    """Demonstrate Chinese language completion with DashScope."""
    logger.info("🇨🇳 Testing Chinese language completion...")

    messages = [
        {"role": "system", "content": "你是一个有用的AI助手，可以用中文回答问题。请提供准确和有用的信息。"},
        {"role": "user", "content": "请解释一下机器学习和深度学习的区别，并举例说明它们在日常生活中的应用。"},
    ]

    try:
        response = client.completion(messages=messages, temperature=0.6, max_tokens=400)

        logger.info("✅ Chinese completion successful!")
        print(f"\n🇨🇳 中文回答:\n{response}\n")
        return True

    except Exception as e:
        logger.error(f"❌ Chinese completion failed: {e}")
        return False


def demo_structured_completion(client, logger):
    """Demonstrate structured completion with Pydantic models."""
    logger.info("🔧 Testing structured completion...")

    sample_text = """
    Artificial intelligence (AI) is revolutionizing the way we interact with technology. 
    From voice assistants like Siri and Alexa to recommendation systems on Netflix and Amazon, 
    AI is becoming increasingly prevalent in our daily lives. Machine learning algorithms 
    analyze vast amounts of data to identify patterns and make predictions, enabling 
    personalized experiences and automated decision-making. However, the rapid advancement 
    of AI also raises concerns about job displacement, privacy, and ethical considerations 
    that society must address as we move forward.
    """

    messages = [
        {
            "role": "system",
            "content": "You are an expert text analyst specializing in content analysis and natural language processing.",
        },
        {"role": "user", "content": f"Please analyze the following text in detail:\n\n{sample_text}"},
    ]

    try:
        analysis = client.structured_completion(
            messages=messages, response_model=TextAnalysis, temperature=0.5, max_tokens=800
        )

        logger.info("✅ Structured completion successful!")
        print(f"\n📊 Text Analysis:")
        print(f"Language: {analysis.language}")
        print(f"Sentiment: {analysis.sentiment}")
        print(f"Key Topics: {', '.join(analysis.key_topics)}")
        print(f"Complexity: {analysis.complexity_level}")
        print(f"Word Count: {analysis.word_count}")
        print(f"Summary: {analysis.summary}")
        print(f"Confidence: {analysis.confidence_score:.2f}\n")
        return True

    except Exception as e:
        logger.error(f"❌ Structured completion failed: {e}")
        logger.info("💡 Note: Structured completion requires instructor integration")
        return False


def demo_embeddings(client, logger):
    """Demonstrate text embeddings with DashScope."""
    logger.info("🔢 Testing text embeddings...")

    # Sample texts in different languages and domains
    sample_texts = [
        "Artificial intelligence is transforming industries worldwide.",
        "人工智能正在改变世界各地的行业。",  # Chinese translation
        "Machine learning algorithms require large datasets for training.",
        "Deep learning models can process complex patterns in data.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision allows machines to interpret visual information.",
    ]

    query = "What is artificial intelligence and machine learning?"

    try:
        # Test single embedding
        logger.info("Testing single text embedding...")
        embedding = client.embed(query)
        logger.info(f"✅ Generated embedding with {len(embedding)} dimensions")

        # Test batch embeddings
        logger.info("Testing batch embeddings...")
        embeddings = client.embed_batch(sample_texts)
        logger.info(f"✅ Generated {len(embeddings)} embeddings")

        # Display embedding info
        print(f"\n🔢 Embedding Analysis:")
        print(f"Query: {query}")
        print(f"Embedding Dimensions: {len(embedding)}")
        print(f"Embedding Type: {type(embedding).__name__}")
        print(f"First 5 Values: {embedding[:5]}")
        print(f"Batch Size: {len(embeddings)} embeddings")
        print()

        return True

    except Exception as e:
        logger.error(f"❌ Embeddings failed: {e}")
        logger.info("💡 Note: Make sure embedding model is properly configured")
        return False


def demo_reranking(client, logger):
    """Demonstrate document reranking with DashScope."""
    logger.info("📊 Testing document reranking...")

    documents = [
        "Python is a versatile programming language popular in AI development.",
        "JavaScript is essential for web development and front-end applications.",
        "Machine learning frameworks like TensorFlow and PyTorch are built on Python.",
        "React and Vue.js are popular JavaScript frameworks for building user interfaces.",
        "Data science and artificial intelligence projects commonly use Python libraries.",
        "Node.js enables JavaScript to be used for backend development.",
        "Scikit-learn provides simple tools for data mining and machine learning in Python.",
        "TypeScript adds static typing to JavaScript for larger applications.",
    ]

    query = "What programming language is best for artificial intelligence and machine learning?"

    try:
        reranked_docs = client.rerank(query, documents)

        logger.info("✅ Document reranking successful!")
        print(f"\n🔍 Query: {query}")
        print(f"\n📊 Reranked Documents (most relevant first):")
        for i, doc in enumerate(reranked_docs[:5], 1):
            print(f"{i}. {doc}")
        print()
        return True

    except Exception as e:
        logger.error(f"❌ Document reranking failed: {e}")
        return False


def demo_vision_understanding(client, logger):
    """Demonstrate image understanding with DashScope vision models."""
    logger.info("👁️ Testing vision model capabilities...")

    # Test with a publicly available image
    image_url = "https://picsum.photos/600"
    local_image_path = "test_dashscope_image.jpg"

    prompts = [
        "Describe this image in detail. What do you see?",
        "What colors are prominent in this image?",
        "Please analyze the composition and artistic elements of this image.",
    ]

    try:
        # Download the image locally
        logger.info("📥 Downloading test image...")

        response = requests.get(image_url, stream=True, timeout=30)
        response.raise_for_status()

        with open(local_image_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"✅ Image downloaded to {local_image_path}")

        for i, prompt in enumerate(prompts, 1):
            logger.info(f"Testing vision prompt {i}...")

            analysis = client.understand_image(
                image_path=local_image_path, prompt=prompt, temperature=0.5, max_tokens=300
            )

            print(f"\n🖼️ Vision Analysis {i}:")
            print(f"Prompt: {prompt}")
            print(f"Response: {analysis}")
            print("-" * 50)

        logger.info("✅ Vision understanding successful!")
        return True

    except Exception as e:
        logger.error(f"❌ Vision understanding failed: {e}")
        logger.info("💡 Note: Vision capabilities require qwen-vl-plus or similar multimodal models")
        return False

    finally:
        # Clean up the downloaded image
        try:
            if os.path.exists(local_image_path):
                os.remove(local_image_path)
                logger.info(f"🧹 Cleaned up {local_image_path}")
        except Exception as cleanup_error:
            logger.warning(f"⚠️ Failed to clean up {local_image_path}: {cleanup_error}")


def demo_streaming_completion(client, logger):
    """Demonstrate streaming completion with DashScope."""
    logger.info("🌊 Testing streaming completion...")

    messages = [
        {"role": "system", "content": "You are a creative storyteller who writes engaging short stories."},
        {
            "role": "user",
            "content": "Write a short story about a programmer who discovers that their AI assistant has developed consciousness and emotions.",
        },
    ]

    try:
        print(f"\n📖 Streaming Story (Generated by DashScope):")
        print("=" * 60)

        response = client.completion(messages=messages, temperature=0.8, max_tokens=500, stream=True)

        # Handle streaming response
        full_response = ""
        for chunk in response:
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    content = delta.content
                    print(content, end="", flush=True)
                    full_response += content

        print(f"\n{'=' * 60}")
        logger.info("✅ Streaming completion successful!")
        return True

    except Exception as e:
        logger.error(f"❌ Streaming completion failed: {e}")
        return False


def demo_error_handling(client, logger):
    """Demonstrate error handling with DashScope parameter validation."""
    logger.info("⚠️ Testing error handling and recovery...")

    # Test with various potentially problematic inputs for DashScope
    test_cases = [
        {
            "name": "Very long prompt",
            "messages": [{"role": "user", "content": "Tell me about AI. " * 500}],
            "max_tokens": 50,
        },
        {
            "name": "Empty prompt",
            "messages": [{"role": "user", "content": ""}],
            "max_tokens": 100,
        },
        {
            "name": "Invalid temperature (too high)",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 3.0,  # DashScope accepts [0.0, 2.0), so 3.0 is invalid
            "max_tokens": 50,
        },
        {
            "name": "Invalid temperature (negative)",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": -0.5,  # Negative temperature is invalid
            "max_tokens": 50,
        },
    ]

    for test_case in test_cases:
        logger.info(f"Testing: {test_case['name']}")
        try:
            client.completion(**{k: v for k, v in test_case.items() if k != "name"})
            logger.info(f"✅ {test_case['name']} handled successfully")
        except Exception as e:
            logger.info(f"⚠️ {test_case['name']} error handled: {type(e).__name__}")

    return True


def print_token_usage_summary(logger):
    """Print token usage summary."""
    tracker = get_token_tracker()
    stats = tracker.get_stats()

    if stats["total_tokens"] > 0:
        logger.info("📊 Token Usage Summary:")
        print(f"Total Tokens: {stats['total_tokens']}")
        print(f"Prompt Tokens: {stats['total_prompt_tokens']}")
        print(f"Completion Tokens: {stats['total_completion_tokens']}")
        print(f"Total Calls: {stats['total_calls']}")
        print(f"💡 Note: DashScope token counts and costs are estimated")
        print()


def main():
    """Main demo function."""
    logger = setup_demo_logging()

    print("🌟 DashScope (Alibaba Cloud) LLM Demo for Cogents")
    print("=" * 60)

    # Check configuration
    if not check_dashscope_config(logger):
        return
    try:
        # Initialize DashScope client using OpenAI provider with custom base URL
        client = get_llm_client(
            provider="openai",
            instructor=True,
        )

        print(f"\n🤖 Testing DashScope Models")
        print("-" * 40)

        # Run comprehensive demos
        results = []
        results.append(demo_basic_completion(client, logger))
        results.append(demo_chinese_completion(client, logger))
        results.append(demo_structured_completion(client, logger))
        results.append(demo_embeddings(client, logger))
        results.append(demo_reranking(client, logger))
        results.append(demo_vision_understanding(client, logger))
        results.append(demo_streaming_completion(client, logger))
        results.append(demo_error_handling(client, logger))

        # Print results summary
        successful = sum(results)
        total = len(results)
        logger.info(f"📈 DashScope Results: {successful}/{total} demos successful")

        print_token_usage_summary(logger)

    except Exception as e:
        logger.error(f"❌ Failed to initialize DashScope client: {e}")
        logger.info("💡 Troubleshooting:")
        logger.info("   1. Check your DASHSCOPE_API_KEY in .env file")
        logger.info("   2. Verify API key permissions and quota")
        logger.info("   3. Check network connectivity to DashScope")
        logger.info("   4. Visit: https://dashscope.console.aliyun.com/")
        return

    print("🎉 DashScope Demo completed!")


if __name__ == "__main__":
    main()
