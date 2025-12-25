# Cogents-core

[![CI](https://github.com/caesar0301/cogents-core/actions/workflows/ci.yml/badge.svg)](https://github.com/caesar0301/cogents-core/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/cogents-core.svg)](https://pypi.org/project/cogents-core/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/caesar0301/cogents-core)

This is part of [project COGENTS](https://github.com/caesar0301/COGENTS), an initiative to develop a computation-driven, cognitive agentic system. This repo contains the foundational abstractions (Agent, Memory, Tool, Goal, Orchestration, and more) along with essential modules such as LLM clients, logging, message buses, model routing, and observability. For the underlying philosophy, refer to my talk on MAS ([link](https://github.com/caesar0301/mas-talk-2508/blob/master/mas-talk-xmingc.pdf)).

## Installation

```bash
pip install -U cogents-core
```

## Core Modules

Cogents offers a comprehensive set of modules for creating intelligent agent-based applications:

### LLM Integration & Management (`cogents_core.llm`)
- **Multi-model support**: OpenAI, OpenRouter, Ollama, LlamaCPP, and LiteLLM
- **Advanced routing**: Dynamic complexity-based and self-assessment routing strategies
- **Tracing & monitoring**: Built-in token tracking and Opik tracing integration
- **Extensible architecture**: Easy to add new LLM providers

### Goal Management & Planning (`cogents_core.goalith`) - *In Development*
- **Goal decomposition**: LLM-based, callable, and simple goal decomposition strategies
- **Graph-based structure**: DAG-based goal management with dependencies
- **Node management**: Goal, subgoal, and task node creation and tracking
- **Conflict detection**: Framework for automated goal conflict identification (planned)
- **Replanning**: Dynamic goal replanning capabilities (planned)

### Tool Management (`cogents_core.toolify`)
- **Tool registry**: Centralized tool registration and management
- **MCP integration**: Model Context Protocol support for tool discovery
- **Execution engine**: Robust tool execution with error handling
- **Toolkit system**: Organized tool collections and configurations

### Memory Management (`cogents_core.memory`)
- **MemU integration**: Advanced memory agent with categorization
- **Embedding support**: Vector-based memory retrieval and linking
- **Multi-category storage**: Activity, event, and profile memory types
- **Memory linking**: Automatic relationship discovery between memories

### Vector Storage (`cogents_core.vector_store`)
- **PGVector support**: PostgreSQL with pgvector extension
- **Weaviate integration**: Cloud-native vector database
- **Semantic search**: Embedding-based document retrieval
- **Flexible indexing**: HNSW and DiskANN indexing strategies

### Message Bus (`cogents_core.msgbus`)
- **Event-driven architecture**: Inter-component communication
- **Watchdog patterns**: Monitoring and reactive behaviors
- **Flexible routing**: Message filtering and delivery

### Routing & Tracing (`cogents_core.routing`, `cogents_core.tracing`)
- **Smart routing**: Dynamic model selection based on complexity
- **Token tracking**: Comprehensive usage monitoring
- **Opik integration**: Production-ready observability
- **LangGraph hooks**: Workflow tracing and debugging

## Project Structure

```
cogents_core/
├── agent/           # Base agent classes and models
├── goalith/         # Goal management and planning system
│   ├── decomposer/  # Goal decomposition strategies
│   ├── goalgraph/   # Graph data structures
│   ├── conflict/    # Conflict detection
│   └── replanner/   # Dynamic replanning
├── llm/             # LLM provider implementations
├── memory/          # Memory management system
│   └── memu/        # MemU memory agent integration
├── toolify/         # Tool management and execution
├── vector_store/    # Vector database integrations
├── msgbus/          # Message bus system
├── routing/         # LLM routing strategies
├── tracing/         # Token tracking and observability
└── utils/           # Utilities and logging
```

## Quick Start

### 1. LLM Client Usage

```python
from cogents_core.llm import get_llm_client

# OpenAI/OpenRouter providers
client = get_llm_client(provider="openai", api_key="sk-...")
client = get_llm_client(provider="openrouter", api_key="sk-...")

# Local providers
client = get_llm_client(provider="ollama", base_url="http://localhost:11434")
client = get_llm_client(provider="llamacpp", model_path="/path/to/model.gguf")

# Basic chat completion
response = client.completion([
    {"role": "user", "content": "Hello!"}
])

# Structured output (requires structured_output=True)
from pydantic import BaseModel

class Response(BaseModel):
    answer: str
    confidence: float

client = get_llm_client(provider="openai", structured_output=True)
result = client.structured_completion(messages, Response)
```

### 2. Goal Management with Goalith

**Note**: The Goalith goal management system is currently under development. The core components are available but the full service integration is not yet complete.

```python
# Basic goal node creation and management
from cogents_core.goalith.goalgraph.node import GoalNode, NodeStatus
from cogents_core.goalith.goalgraph.graph import GoalGraph
from cogents_core.goalith.decomposer import LLMDecomposer

# Create a goal node
goal_node = GoalNode(
    description="Plan and execute a product launch",
    priority=8.0,
    context={
        "budget": "$50,000",
        "timeline": "3 months",
        "target_audience": "young professionals"
    },
    tags=["product", "launch", "marketing"]
)

# Create goal graph for management
graph = GoalGraph()
graph.add_node(goal_node)

# Use LLM decomposer directly
decomposer = LLMDecomposer()
subgoals = decomposer.decompose(goal_node, context={
    "team_size": "5 people",
    "experience_level": "intermediate"
})

print(f"Goal: {goal_node.description}")
print(f"Status: {goal_node.status}")
print(f"Generated {len(subgoals)} subgoals")
```

### 3. Memory Management

```python
from cogents_core.memory.memu import MemoryAgent

# Initialize memory agent
memory_agent = MemoryAgent(
    agent_id="my_agent",
    user_id="user123",
    memory_dir="/tmp/memory_storage",
    enable_embeddings=True
)

# Add activity memory
activity_content = """
USER: Hi, I'm Sarah and I work as a software engineer.
ASSISTANT: Nice to meet you Sarah! What kind of projects do you work on?
USER: I mainly work on web applications using Python and React.
"""

result = memory_agent.call_function(
    "add_activity_memory",
    {
        "character_name": "Sarah",
        "content": activity_content
    }
)

# Generate memory suggestions
if result.get("success"):
    memory_items = result.get("memory_items", [])
    suggestions = memory_agent.call_function(
        "generate_memory_suggestions",
        {
            "character_name": "Sarah",
            "new_memory_items": memory_items
        }
    )
```

### 4. Vector Store Operations

```python
from cogents_core.vector_store import PGVectorStore
from cogents_core.llm import get_llm_client

# Initialize vector store
vector_store = PGVectorStore(
    collection_name="my_documents",
    embedding_model_dims=768,
    dbname="vectordb",
    user="postgres",
    password="postgres",
    host="localhost",
    port=5432
)

# Initialize embedding client
embed_client = get_llm_client(provider="ollama", embed_model="nomic-embed-text")

# Prepare documents
documents = [
    {
        "id": "doc1",
        "content": "Machine learning is a subset of AI...",
        "metadata": {"category": "AI", "type": "definition"}
    }
]

# Generate embeddings and store
vectors = []
payloads = []
ids = []

for doc in documents:
    embedding = embed_client.embed(doc["content"])
    vectors.append(embedding)
    payloads.append(doc["metadata"])
    ids.append(doc["id"])

# Insert into vector store
vector_store.insert(vectors=vectors, payloads=payloads, ids=ids)

# Search
query = "What is artificial intelligence?"
query_embedding = embed_client.embed(query)
results = vector_store.search(query=query, vectors=query_embedding, limit=5)
```

### 5. Tool Management

```python
from cogents_core.toolify import BaseToolkit, ToolkitConfig, ToolkitRegistry, register_toolkit
from typing import Dict, Callable

# Create a custom toolkit using decorator
@register_toolkit("calculator")
class CalculatorToolkit(BaseToolkit):
    def get_tools_map(self) -> Dict[str, Callable]:
        return {
            "add": self.add,
            "multiply": self.multiply
        }

    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b

# Alternative: Manual registration
config = ToolkitConfig(name="calculator", description="Basic math operations")
ToolkitRegistry.register("calculator", CalculatorToolkit)

# Create and use toolkit
calculator = ToolkitRegistry.create_toolkit("calculator", config)
result = calculator.call_tool("add", a=5, b=3)
print(f"5 + 3 = {result}")
```

### 6. Message Bus and Events

```python
from cogents_core.msgbus import EventBus, BaseEvent, BaseWatchdog

# Define custom event
class TaskCompleted(BaseEvent):
    def __init__(self, task_id: str, result: str):
        super().__init__()
        self.task_id = task_id
        self.result = result

# Create event bus
bus = EventBus()

# Define watchdog
class TaskWatchdog(BaseWatchdog):
    def handle_event(self, event: BaseEvent):
        if isinstance(event, TaskCompleted):
            print(f"Task {event.task_id} completed with result: {event.result}")

# Register watchdog and publish event
watchdog = TaskWatchdog()
bus.register_watchdog(watchdog)
bus.publish(TaskCompleted("task_1", "success"))
```

### 7. Token Tracking and Tracing

```python
from cogents_core.tracing import get_token_tracker
from cogents_core.llm import get_llm_client

# Initialize client and tracker
client = get_llm_client(provider="openai")
tracker = get_token_tracker()

# Reset tracker
tracker.reset()

# Make LLM calls (automatically tracked)
response1 = client.completion([{"role": "user", "content": "Hello"}])
response2 = client.completion([{"role": "user", "content": "How are you?"}])

# Get usage statistics
stats = tracker.get_stats()
print(f"Total tokens: {stats['total_tokens']}")
print(f"Total calls: {stats['total_calls']}")
print(f"Average tokens per call: {stats.get('avg_tokens_per_call', 0)}")
```

## Environment Variables

Set these environment variables for different providers:

```bash
# Default LLM provider
export COGENTS_LLM_PROVIDER="openai"

# OpenAI
export OPENAI_API_KEY="sk-..."

# OpenRouter
export OPENROUTER_API_KEY="sk-..."

# LlamaCPP
export LLAMACPP_MODEL_PATH="/path/to/model.gguf"

# Ollama
export OLLAMA_BASE_URL="http://localhost:11434"

# PostgreSQL (for vector store)
export POSTGRES_HOST="localhost"
export POSTGRES_PORT="5432"
export POSTGRES_DB="vectordb"
export POSTGRES_USER="postgres"
export POSTGRES_PASSWORD="postgres"
```

## Advanced Usage

### Custom Goal Decomposer

```python
from cogents_core.goalith.decomposer.base import GoalDecomposer
from cogents_core.goalith.goalgraph.node import GoalNode
from typing import List, Dict, Any, Optional
import copy

class CustomDecomposer(GoalDecomposer):
    @property
    def name(self) -> str:
        return "custom_decomposer"

    def decompose(self, goal_node: GoalNode, context: Optional[Dict[str, Any]] = None) -> List[GoalNode]:
        # Custom decomposition logic
        subtasks = [
            "Research requirements",
            "Design solution",
            "Implement features",
            "Test and deploy"
        ]

        nodes = []
        for i, subtask in enumerate(subtasks):
            # Deep copy context to avoid shared references
            context_copy = copy.deepcopy(goal_node.context) if goal_node.context else {}

            node = GoalNode(
                description=subtask,
                parent=goal_node.id,
                priority=goal_node.priority - i * 0.1,
                context=context_copy,
                tags=goal_node.tags.copy() if goal_node.tags else [],
                decomposer_name=self.name
            )
            nodes.append(node)

        return nodes

# Use the decomposer directly
custom_decomposer = CustomDecomposer()
goal_node = GoalNode(description="Build a web application")
subgoals = custom_decomposer.decompose(goal_node)
```

### LLM Routing Strategies

```python
from cogents_core.routing import ModelRouter, DynamicComplexityStrategy
from cogents_core.llm import get_llm_client

# Create a lite client for complexity assessment
lite_client = get_llm_client(provider="ollama", chat_model="llama3.2:1b")

# Create router with dynamic complexity strategy
router = ModelRouter(
    strategy="dynamic_complexity",
    lite_client=lite_client,
    strategy_config={
        "complexity_threshold_low": 0.3,
        "complexity_threshold_high": 0.7
    }
)

# Route queries to get tier recommendations
simple_query = "What is 2+2?"
result = router.route(simple_query)
print(f"Query: {simple_query}")
print(f"Recommended tier: {result.tier}")  # Likely ModelTier.LITE
print(f"Confidence: {result.confidence}")

complex_query = "Explain quantum computing and its applications"
result = router.route(complex_query)
print(f"Query: {complex_query}")
print(f"Recommended tier: {result.tier}")  # Likely ModelTier.POWER
print(f"Confidence: {result.confidence}")

# Get recommended model configuration
routing_result, model_config = router.route_and_configure(complex_query)
print(f"Recommended config: {model_config}")
```

## Examples

Check the `examples/` directory for comprehensive usage examples:

- **LLM Examples**: `examples/llm/` - OpenAI, Ollama, LlamaCPP, token tracking
- **Goal Management**: `examples/goals/` - Goal decomposition and planning
- **Memory Examples**: `examples/memory/` - Memory agent operations
- **Vector Store**: `examples/vector_store/` - PGVector and Weaviate usage
- **Message Bus**: `examples/msgbus/` - Event-driven patterns
- **Tools**: Various toolkit implementations

## Development

```bash
# Install development dependencies
make install

# Run tests
make test

# Run specific test categories
make test-unit          # Unit tests only
make test-integration   # Integration tests only

# Format code
make format
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
