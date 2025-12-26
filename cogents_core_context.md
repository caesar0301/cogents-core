# Cogents Core: Comprehensive Developer Guide for AI Coding Agents

**Version:** 1.0.0  
**License:** MIT  
**Project:** [Cogents Core](https://github.com/caesar0301/cogents-core)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Core Philosophy](#core-philosophy)
3. [Installation & Setup](#installation--setup)
4. [Architecture Overview](#architecture-overview)
5. [LLM Integration System](#llm-integration-system)
6. [Agent Framework](#agent-framework)
7. [Goal Management (Goalith)](#goal-management-goalith)
8. [Tool Management (Toolify)](#tool-management-toolify)
9. [Memory Management](#memory-management)
10. [Vector Store](#vector-store)
11. [Message Bus & Events](#message-bus--events)
12. [Model Routing](#model-routing)
13. [Tracing & Observability](#tracing--observability)
14. [Advanced Patterns](#advanced-patterns)
15. [Best Practices](#best-practices)
16. [API Reference](#api-reference)

---

## Introduction

**Cogents Core** is a foundational framework for building computation-driven, cognitive agentic systems. It provides a comprehensive set of abstractions and implementations for developing sophisticated AI agents with support for:

- **Multi-model LLM integration** (OpenAI, Ollama, LlamaCPP, LiteLLM, OpenRouter)
- **Hierarchical goal management** with DAG-based planning (Goalith)
- **Extensible tool management** with MCP support (Toolify)
- **Advanced memory systems** with MemU integration
- **Vector storage** for semantic retrieval (PGVector, Weaviate)
- **Event-driven architecture** for inter-component communication
- **Smart model routing** based on query complexity
- **Production-ready observability** with token tracking and tracing

This guide focuses on **public APIs** that developers use to build AI coding agents and other intelligent applications.

---

## Core Philosophy

Cogents Core is built around these key principles:

1. **Modularity**: Each component can be used independently or composed together
2. **Extensibility**: Plugin architecture allows custom implementations
3. **Multi-Model Support**: Provider-agnostic abstractions for LLM integration
4. **Computation-Driven**: Emphasis on structured reasoning and planning
5. **Production-Ready**: Built-in observability, error handling, and monitoring

---

## Installation & Setup

### Basic Installation

```bash
pip install -U cogents-core
```

### Environment Configuration

```bash
# Default LLM provider
export COGENTS_LLM_PROVIDER="openai"

# Provider-specific configuration
export OPENAI_API_KEY="sk-..."
export OPENROUTER_API_KEY="sk-..."
export OLLAMA_BASE_URL="http://localhost:11434"
export LLAMACPP_MODEL_PATH="/path/to/model.gguf"

# Vector store configuration
export POSTGRES_HOST="localhost"
export POSTGRES_PORT="5432"
export POSTGRES_DB="vectordb"
export POSTGRES_USER="postgres"
export POSTGRES_PASSWORD="postgres"

# Observability
export COGENTS_ENABLE_TRACING="true"
export COGENTS_DEBUG="false"
```

### Verification

```python
from cogents_core.llm import get_llm_client

# Test LLM client
client = get_llm_client(provider="openai")
response = client.completion([
    {"role": "user", "content": "Hello, world!"}
])
print(response)
```

---

## Architecture Overview

Cogents Core consists of several interconnected modules:

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
│              (Your AI Agent Implementation)              │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                   Cogents Core Framework                 │
├─────────────────────────────────────────────────────────┤
│  Agent Framework (BaseAgent, BaseGraphicAgent)          │
├─────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │ Goalith  │  │ Toolify  │  │  Memory  │             │
│  │ (Goals)  │  │ (Tools)  │  │  (MemU)  │             │
│  └──────────┘  └──────────┘  └──────────┘             │
├─────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │   LLM    │  │ Routing  │  │ Tracing  │             │
│  │ Clients  │  │ Strategy │  │  Token   │             │
│  └──────────┘  └──────────┘  └──────────┘             │
├─────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │  Vector  │  │ Message  │  │ Utilities│             │
│  │  Store   │  │   Bus    │  │ Logging  │             │
│  └──────────┘  └──────────┘  └──────────┘             │
└─────────────────────────────────────────────────────────┘
```

---

## LLM Integration System

The LLM integration system provides a unified interface for interacting with multiple LLM providers.

### Core Concepts

- **`BaseLLMClient`**: Abstract base class defining the LLM interface
- **Provider implementations**: OpenAI, Ollama, LlamaCPP, LiteLLM, OpenRouter
- **Structured output**: Pydantic model support via instructor
- **Vision support**: Image understanding capabilities
- **Embeddings**: Text embedding generation for semantic search

### Getting an LLM Client

```python
from cogents_core.llm import get_llm_client

# OpenAI/OpenRouter (cloud providers)
client = get_llm_client(
    provider="openai",
    api_key="sk-...",
    chat_model="gpt-4o",
    structured_output=True  # Enable Pydantic support
)

# Ollama (local)
client = get_llm_client(
    provider="ollama",
    base_url="http://localhost:11434",
    chat_model="llama3.2:latest"
)

# LlamaCPP (local)
client = get_llm_client(
    provider="llamacpp",
    model_path="/path/to/model.gguf",
    n_ctx=2048,
    n_gpu_layers=32
)
```

### Basic Completion

```python
# Simple text completion
response = client.completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing"}
    ],
    temperature=0.7,
    max_tokens=500
)
print(response)  # Returns string
```

### Structured Completion

```python
from pydantic import BaseModel, Field
from typing import List

class CodeAnalysis(BaseModel):
    """Analysis of code quality and suggestions."""
    issues: List[str] = Field(description="List of identified issues")
    suggestions: List[str] = Field(description="Improvement suggestions")
    complexity_score: float = Field(ge=0, le=10, description="Complexity rating")

# Enable structured output
client = get_llm_client(provider="openai", structured_output=True)

# Get structured response
result = client.structured_completion(
    messages=[
        {"role": "user", "content": "Analyze this code: def foo(): pass"}
    ],
    response_model=CodeAnalysis,
    temperature=0.3
)

print(result.issues)
print(result.suggestions)
print(result.complexity_score)
```

### Vision Understanding

```python
# From local file
analysis = client.understand_image(
    image_path="/path/to/image.jpg",
    prompt="Describe the architecture diagram in this image",
    max_tokens=1000
)

# From URL
analysis = client.understand_image_from_url(
    image_url="https://example.com/diagram.png",
    prompt="Extract all text from this image"
)
```

### Embeddings

```python
# Single text embedding
embedding = client.embed("This is a sample text")
print(len(embedding))  # Embedding dimensions

# Batch embeddings
texts = [
    "First document",
    "Second document",
    "Third document"
]
embeddings = client.embed_batch(texts)

# Reranking
query = "What is machine learning?"
chunks = [
    "Machine learning is a subset of AI",
    "Python is a programming language",
    "Deep learning uses neural networks"
]
reranked = client.rerank(query, chunks)
# Returns: [(score, index, text), ...] sorted by relevance
```

### Streaming Responses

```python
# Enable streaming
response_stream = client.completion(
    messages=[{"role": "user", "content": "Write a long story"}],
    stream=True
)

for chunk in response_stream:
    print(chunk, end="", flush=True)
```

### Token Tracking

All LLM calls are automatically tracked for observability:

```python
from cogents_core.tracing import get_token_tracker

# Make some LLM calls
client.completion([{"role": "user", "content": "Hello"}])
client.completion([{"role": "user", "content": "Goodbye"}])

# Get usage statistics
tracker = get_token_tracker()
stats = tracker.get_stats()

print(f"Total tokens: {stats['total_tokens']}")
print(f"Total calls: {stats['total_calls']}")
print(f"Prompt tokens: {stats['total_prompt_tokens']}")
print(f"Completion tokens: {stats['total_completion_tokens']}")
print(f"Average tokens/call: {stats['avg_tokens_per_call']}")
```

### Provider-Specific Features

#### OpenAI Client

```python
from cogents_core.llm import OpenAIClient

client = OpenAIClient(
    api_key="sk-...",
    chat_model="gpt-4o",
    vision_model="gpt-4o",  # For image understanding
    embed_model="text-embedding-3-small",
    instructor=True
)
```

#### Ollama Client

```python
from cogents_core.llm import OllamaClient

client = OllamaClient(
    base_url="http://localhost:11434",
    chat_model="llama3.2:latest",
    embed_model="nomic-embed-text"
)
```

#### LlamaCPP Client

```python
from cogents_core.llm import LlamaCppClient

client = LlamaCppClient(
    model_path="/models/llama-3.2-3b-Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=32,
    temperature=0.7
)
```

---

## Agent Framework

Cogents provides a hierarchy of agent base classes for different use cases.

### Agent Hierarchy

```
BaseAgent (ABC)
  ├── BaseGraphicAgent (LangGraph support)
  │     ├── BaseConversationAgent (session management)
  │     └── BaseResearcher (research workflows)
  └── Custom agent implementations
```

### BaseAgent

Foundation for all agents with LLM integration and token tracking.

```python
from cogents_core.agent import BaseAgent
from typing import Dict, Any

class MyAgent(BaseAgent):
    """Custom agent implementation."""
    
    def __init__(self, llm_provider: str = "openai", model_name: str = "gpt-4o"):
        super().__init__(llm_provider, model_name)
        # Custom initialization
        
    def run(self, user_message: str, context: Dict[str, Any] = None, config = None) -> Any:
        """Execute agent logic."""
        # Use self.llm for LLM operations
        response = self.llm.completion([
            {"role": "user", "content": user_message}
        ])
        return response

# Usage
agent = MyAgent()
result = agent.run("Explain dependency injection")
print(result)

# Get token usage
stats = agent.get_token_usage_stats()
agent.print_token_usage_summary()
```

### BaseGraphicAgent

For agents using LangGraph workflows.

```python
from cogents_core.agent import BaseGraphicAgent
from langgraph.graph import StateGraph
from typing import TypedDict, Type
from pydantic import BaseModel, Field

class AgentState(TypedDict):
    """State for the agent graph."""
    messages: list
    current_step: str
    result: str

class MyGraphAgent(BaseGraphicAgent):
    """Agent with LangGraph workflow."""
    
    def get_state_class(self) -> Type:
        return AgentState
    
    def _build_graph(self) -> StateGraph:
        """Build the agent's graph."""
        graph = StateGraph(AgentState)
        
        # Define nodes
        def process_node(state: AgentState) -> AgentState:
            # Use self.llm for LLM operations
            result = self.llm.completion([
                {"role": "user", "content": state["messages"][-1]}
            ])
            state["result"] = result
            return state
        
        # Add nodes
        graph.add_node("process", process_node)
        
        # Set entry and finish
        graph.set_entry_point("process")
        graph.set_finish_point("process")
        
        return graph.compile()
    
    def run(self, user_message: str, context = None, config = None) -> str:
        """Run the graph agent."""
        if not self.graph:
            self.graph = self._build_graph()
        
        initial_state = AgentState(
            messages=[user_message],
            current_step="start",
            result=""
        )
        
        final_state = self.graph.invoke(initial_state, config=config)
        return final_state["result"]

# Usage
agent = MyGraphAgent(llm_provider="openai", model_name="gpt-4o")
result = agent.run("What is reactive programming?")

# Export graph visualization
agent.export_graph("agent_graph.png", format="png")
```

### BaseConversationAgent

For conversational agents with session management.

```python
from cogents_core.agent import BaseConversationAgent

class ChatAgent(BaseConversationAgent):
    """Conversational agent with memory."""
    
    def start_conversation(self, user_id: str, initial_message: str = None):
        """Start a new conversation session."""
        session_id = f"session_{user_id}_{int(time.time())}"
        self._session_states[session_id] = {
            "messages": [],
            "user_id": user_id
        }
        
        if initial_message:
            return self.process_user_message(user_id, session_id, initial_message)
        
        return {"session_id": session_id, "message": "Conversation started"}
    
    def process_user_message(self, user_id: str, session_id: str, message: str):
        """Process a message in the conversation."""
        state = self._session_states.get(session_id)
        if not state:
            return {"error": "Session not found"}
        
        # Add user message
        state["messages"].append({"role": "user", "content": message})
        
        # Get LLM response
        response = self.llm.completion(state["messages"])
        
        # Add assistant message
        state["messages"].append({"role": "assistant", "content": response})
        
        return {"message": response, "session_id": session_id}

# Usage
agent = ChatAgent()
session = agent.start_conversation("user123", "Hello!")
print(session["message"])

response = agent.process_user_message(
    "user123",
    session["session_id"],
    "Tell me about AI"
)
print(response["message"])

# List active sessions
sessions = agent.list_sessions()

# Clear specific session
agent.clear_session(session["session_id"])
```

### BaseResearcher

For research-oriented agents.

```python
from cogents_core.agent import BaseResearcher, ResearchOutput
from typing import Dict, Any

class WebResearcher(BaseResearcher):
    """Research agent that searches and synthesizes information."""
    
    def research(self, user_message: str, context: Dict[str, Any] = None, config = None) -> ResearchOutput:
        """Perform research on a topic."""
        # Use self.llm for research operations
        
        # 1. Generate search queries
        queries = self._generate_queries(user_message)
        
        # 2. Search and collect sources
        sources = self._search_sources(queries)
        
        # 3. Synthesize findings
        summary = self._synthesize_findings(user_message, sources)
        
        return ResearchOutput(
            content=summary,
            sources=sources,
            summary=summary[:200] + "...",
            timestamp=datetime.now()
        )
    
    def _generate_queries(self, topic: str) -> list:
        # Implementation
        pass
    
    def _search_sources(self, queries: list) -> list:
        # Implementation
        pass
    
    def _synthesize_findings(self, topic: str, sources: list) -> str:
        # Implementation
        pass

# Usage
researcher = WebResearcher(llm_provider="openai")
output = researcher.research("Latest advances in quantum computing")

print(output.summary)
print(f"Found {len(output.sources)} sources")
for source in output.sources:
    print(f"- {source}")
```

---

## Goal Management (Goalith)

Goalith is a DAG-based goal and task management system that enables hierarchical planning and execution tracking.

### Core Concepts

- **GoalNode**: Represents a goal, subgoal, or task with metadata
- **GoalGraph**: DAG structure managing nodes and dependencies
- **GoalDecomposer**: Strategy for breaking goals into subgoals
- **NodeStatus**: Lifecycle states (PENDING, IN_PROGRESS, COMPLETED, FAILED, etc.)

### GoalNode Model

```python
from cogents_core.goalith.goalgraph.node import GoalNode, NodeStatus
from datetime import datetime, timezone

# Create a goal node
goal = GoalNode(
    description="Build a web application",
    priority=8.0,
    context={
        "tech_stack": "Python, FastAPI, React",
        "deadline": "2025-03-01",
        "team_size": 3
    },
    tags=["web", "fullstack", "api"],
    estimated_effort="2 months",
    deadline=datetime(2025, 3, 1, tzinfo=timezone.utc)
)

print(f"Goal ID: {goal.id}")
print(f"Status: {goal.status}")  # PENDING by default
print(f"Priority: {goal.priority}")

# Update goal status
goal.mark_started()
print(f"Started at: {goal.started_at}")

# Add execution notes
goal.add_note("Initial planning completed")
goal.update_context("progress", "25%")

# Mark as completed
goal.mark_completed()
print(f"Completed at: {goal.completed_at}")

# Check node state
print(f"Is ready: {goal.is_ready()}")
print(f"Is terminal: {goal.is_terminal()}")
print(f"Can retry: {goal.can_retry()}")
```

### GoalGraph Operations

```python
from cogents_core.goalith.goalgraph.graph import GoalGraph
from cogents_core.goalith.goalgraph.node import GoalNode, NodeStatus

# Create graph
graph = GoalGraph()

# Create and add nodes
main_goal = GoalNode(
    description="Launch product",
    priority=10.0
)
graph.add_node(main_goal)

research = GoalNode(
    description="Market research",
    priority=9.0,
    parent=main_goal.id
)
graph.add_node(research)

design = GoalNode(
    description="Product design",
    priority=8.0,
    parent=main_goal.id
)
graph.add_node(design)

development = GoalNode(
    description="Development",
    priority=7.0,
    parent=main_goal.id
)
graph.add_node(development)

# Add dependencies (design depends on research)
graph.add_dependency(design.id, research.id)

# Development depends on design
graph.add_dependency(development.id, design.id)

# Query operations
ready_nodes = graph.get_ready_nodes()
print(f"Ready to execute: {[n.description for n in ready_nodes]}")

# Get node relationships
children = graph.get_children(main_goal.id)
print(f"Children: {[c.description for c in children]}")

dependencies = graph.get_dependencies(development.id)
print(f"Dependencies: {[d.description for d in dependencies]}")

# Update node status
research_node = graph.get_node(research.id)
research_node.mark_completed()
graph.update_node(research_node)

# Check ready nodes again
ready_nodes = graph.get_ready_nodes()
print(f"Now ready: {[n.description for n in ready_nodes]}")

# Graph statistics
stats = graph.get_graph_stats()
print(f"Total nodes: {stats['total_nodes']}")
print(f"Ready nodes: {stats['ready_nodes']}")
print(f"Completed: {stats[str(NodeStatus.COMPLETED)]}")
```

### Goal Decomposition

Goalith supports multiple decomposition strategies.

#### Simple List Decomposer

```python
from cogents_core.goalith.decomposer import SimpleListDecomposer
from cogents_core.goalith.goalgraph.node import GoalNode

# Define subtasks as strings
subtasks = [
    "Set up development environment",
    "Design database schema",
    "Implement authentication",
    "Build API endpoints",
    "Create frontend components",
    "Write tests",
    "Deploy to production"
]

# Create decomposer
decomposer = SimpleListDecomposer(subtasks)

# Decompose goal
goal = GoalNode(description="Build REST API")
subgoals = decomposer.decompose(goal)

print(f"Decomposed into {len(subgoals)} subgoals:")
for sg in subgoals:
    print(f"- {sg.description} (Priority: {sg.priority})")
```

#### LLM-Based Decomposer

```python
from cogents_core.goalith.decomposer import LLMDecomposer
from cogents_core.llm import get_llm_client

# Create LLM client
llm_client = get_llm_client(provider="openai", structured_output=True)

# Create LLM decomposer
decomposer = LLMDecomposer(llm_client=llm_client)

# Decompose with context
goal = GoalNode(
    description="Create machine learning pipeline",
    context={
        "data_source": "CSV files",
        "model_type": "classification",
        "deployment_target": "AWS Lambda"
    }
)

subgoals = decomposer.decompose(
    goal,
    context={
        "team_experience": "intermediate",
        "timeline": "4 weeks"
    }
)

print(f"LLM generated {len(subgoals)} subgoals:")
for sg in subgoals:
    print(f"- {sg.description}")
    print(f"  Priority: {sg.priority}")
    print(f"  Effort: {sg.estimated_effort}")
```

#### Callable Decomposer

```python
from cogents_core.goalith.decomposer import CallableDecomposer

def custom_decomposition_logic(goal_node: GoalNode, context: dict = None):
    """Custom decomposition function."""
    subtasks = []
    
    # Custom logic based on goal description
    if "web" in goal_node.description.lower():
        subtasks = [
            "Frontend development",
            "Backend development",
            "Database setup",
            "Testing and QA"
        ]
    else:
        subtasks = [
            "Planning",
            "Implementation",
            "Review"
        ]
    
    # Create nodes
    nodes = []
    for i, task in enumerate(subtasks):
        node = GoalNode(
            description=task,
            parent=goal_node.id,
            priority=goal_node.priority - i * 0.1,
            context=goal_node.context.copy() if goal_node.context else {}
        )
        nodes.append(node)
    
    return nodes

# Create callable decomposer
decomposer = CallableDecomposer(custom_decomposition_logic)

# Use it
goal = GoalNode(description="Build web dashboard")
subgoals = decomposer.decompose(goal)
```

### Complete Goalith Workflow

```python
from cogents_core.goalith.goalgraph.graph import GoalGraph
from cogents_core.goalith.goalgraph.node import GoalNode, NodeStatus
from cogents_core.goalith.decomposer import LLMDecomposer
from cogents_core.llm import get_llm_client
from pathlib import Path

# Initialize components
graph = GoalGraph()
llm_client = get_llm_client(provider="openai", structured_output=True)
decomposer = LLMDecomposer(llm_client=llm_client)

# Create main goal
main_goal = GoalNode(
    description="Develop AI-powered code review system",
    priority=10.0,
    context={
        "tech_stack": "Python, FastAPI, LangChain",
        "features": ["syntax analysis", "security checks", "suggestions"]
    }
)
graph.add_node(main_goal)

# Decompose into subgoals
subgoals = decomposer.decompose(main_goal, context={
    "team_size": "3 developers",
    "timeline": "8 weeks"
})

# Add subgoals to graph
for subgoal in subgoals:
    graph.add_node(subgoal)
    graph.add_parent_child(main_goal.id, subgoal.id)

# Execute goals
while True:
    ready = graph.get_ready_nodes()
    if not ready:
        break
    
    # Get highest priority ready node
    task = max(ready, key=lambda n: n.priority)
    print(f"\nExecuting: {task.description}")
    
    # Mark as in progress
    task.mark_started()
    graph.update_node(task)
    
    # Simulate execution
    # ... actual work happens here ...
    
    # Mark as completed
    task.mark_completed()
    graph.update_node(task)
    print(f"Completed: {task.description}")

# Save graph
graph.save_to_json(Path("goal_graph.json"))

# Load graph
new_graph = GoalGraph()
new_graph.load_from_json(Path("goal_graph.json"))

# Get statistics
stats = graph.get_graph_stats()
print(f"\nGoal Graph Statistics:")
print(f"Total nodes: {stats['total_nodes']}")
print(f"Completed: {stats[str(NodeStatus.COMPLETED)]}")
print(f"Pending: {stats[str(NodeStatus.PENDING)]}")
```

---

## Tool Management (Toolify)

Toolify provides a unified toolkit system with support for LangChain tools, MCP integration, and custom tool implementations.

### Core Concepts

- **BaseToolkit**: Base class for synchronous toolkits
- **AsyncBaseToolkit**: Base class for asynchronous toolkits
- **ToolkitConfig**: Configuration for toolkit initialization
- **ToolkitRegistry**: Central registry for discovering and creating toolkits
- **MCP Integration**: Support for Model Context Protocol tools

### Creating a Custom Toolkit

```python
from cogents_core.toolify import BaseToolkit, ToolkitConfig
from typing import Dict, Callable

class CalculatorToolkit(BaseToolkit):
    """Simple calculator toolkit."""
    
    def get_tools_map(self) -> Dict[str, Callable]:
        """Return mapping of tool names to functions."""
        return {
            "add": self.add,
            "subtract": self.subtract,
            "multiply": self.multiply,
            "divide": self.divide
        }
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a."""
        return a - b
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b
    
    def divide(self, a: float, b: float) -> float:
        """Divide a by b."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

# Usage
config = ToolkitConfig(name="calculator")
toolkit = CalculatorToolkit(config)

# Call tools
result = toolkit.call_tool("add", a=5, b=3)
print(f"5 + 3 = {result}")

result = toolkit.call_tool("multiply", a=4, b=7)
print(f"4 * 7 = {result}")
```

### Toolkit with LLM Integration

```python
from cogents_core.toolify import BaseToolkit, ToolkitConfig
from typing import Dict, Callable

class TextAnalysisToolkit(BaseToolkit):
    """Toolkit with LLM-powered text analysis."""
    
    def get_tools_map(self) -> Dict[str, Callable]:
        return {
            "summarize": self.summarize,
            "extract_keywords": self.extract_keywords,
            "sentiment_analysis": self.sentiment_analysis
        }
    
    def summarize(self, text: str, max_words: int = 100) -> str:
        """Summarize text using LLM."""
        response = self.llm_client.completion([
            {
                "role": "user",
                "content": f"Summarize in {max_words} words:\n\n{text}"
            }
        ])
        return response
    
    def extract_keywords(self, text: str, count: int = 10) -> list:
        """Extract keywords from text."""
        from pydantic import BaseModel, Field
        
        class Keywords(BaseModel):
            keywords: list[str] = Field(description="List of keywords")
        
        # Use structured output
        result = self.llm_client.structured_completion(
            messages=[
                {
                    "role": "user",
                    "content": f"Extract {count} keywords from:\n\n{text}"
                }
            ],
            response_model=Keywords
        )
        return result.keywords
    
    def sentiment_analysis(self, text: str) -> dict:
        """Analyze sentiment of text."""
        from pydantic import BaseModel, Field
        
        class Sentiment(BaseModel):
            sentiment: str = Field(description="positive, negative, or neutral")
            confidence: float = Field(ge=0, le=1, description="Confidence score")
        
        result = self.llm_client.structured_completion(
            messages=[
                {"role": "user", "content": f"Analyze sentiment:\n\n{text}"}
            ],
            response_model=Sentiment
        )
        return {"sentiment": result.sentiment, "confidence": result.confidence}

# Usage with LLM configuration
config = ToolkitConfig(
    name="text_analysis",
    llm_provider="openai",
    llm_model="gpt-4o"
)
toolkit = TextAnalysisToolkit(config)

text = "This product is amazing! I love it."
summary = toolkit.call_tool("summarize", text=text, max_words=20)
keywords = toolkit.call_tool("extract_keywords", text=text, count=5)
sentiment = toolkit.call_tool("sentiment_analysis", text=text)

print(f"Summary: {summary}")
print(f"Keywords: {keywords}")
print(f"Sentiment: {sentiment}")
```

### Async Toolkit

```python
from cogents_core.toolify import AsyncBaseToolkit, ToolkitConfig
from typing import Dict, Callable
import asyncio
import aiohttp

class WebScraperToolkit(AsyncBaseToolkit):
    """Async toolkit for web scraping."""
    
    async def build(self):
        """Initialize async resources."""
        await super().build()
        self.session = aiohttp.ClientSession()
    
    async def cleanup(self):
        """Cleanup async resources."""
        if hasattr(self, 'session'):
            await self.session.close()
        await super().cleanup()
    
    async def get_tools_map(self) -> Dict[str, Callable]:
        return {
            "fetch_url": self.fetch_url,
            "fetch_multiple": self.fetch_multiple
        }
    
    async def fetch_url(self, url: str) -> str:
        """Fetch content from URL."""
        async with self.session.get(url) as response:
            return await response.text()
    
    async def fetch_multiple(self, urls: list) -> dict:
        """Fetch multiple URLs concurrently."""
        tasks = [self.fetch_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return dict(zip(urls, results))

# Usage with async context manager
async def main():
    config = ToolkitConfig(name="web_scraper")
    
    async with WebScraperToolkit(config) as toolkit:
        # Fetch single URL
        content = await toolkit.call_tool(
            "fetch_url",
            url="https://example.com"
        )
        print(f"Fetched {len(content)} bytes")
        
        # Fetch multiple URLs
        urls = [
            "https://example.com/page1",
            "https://example.com/page2"
        ]
        results = await toolkit.call_tool(
            "fetch_multiple",
            urls=urls
        )
        print(f"Fetched {len(results)} pages")

asyncio.run(main())
```

### Toolkit Registry

```python
from cogents_core.toolify import (
    ToolkitRegistry,
    register_toolkit,
    get_toolkit,
    get_toolkits_map,
    BaseToolkit,
    ToolkitConfig
)

# Register using decorator
@register_toolkit("calculator")
class CalculatorToolkit(BaseToolkit):
    def get_tools_map(self):
        return {
            "add": lambda a, b: a + b,
            "multiply": lambda a, b: a * b
        }

# Manual registration
class StringToolkit(BaseToolkit):
    def get_tools_map(self):
        return {
            "upper": lambda s: s.upper(),
            "lower": lambda s: s.lower()
        }

ToolkitRegistry.register("string_tools", StringToolkit)

# List available toolkits
print("Available toolkits:", ToolkitRegistry.list_toolkits())

# Create toolkit instance
calc = get_toolkit("calculator")
result = calc.call_tool("add", a=10, b=20)
print(f"10 + 20 = {result}")

# Get multiple toolkits
toolkits = get_toolkits_map(
    names=["calculator", "string_tools"],
    configs={
        "calculator": ToolkitConfig(name="calc"),
        "string_tools": ToolkitConfig(name="str")
    }
)

# Use multiple toolkits
calc_result = toolkits["calculator"].call_tool("multiply", a=5, b=6)
str_result = toolkits["string_tools"].call_tool("upper", s="hello")
print(f"5 * 6 = {calc_result}")
print(f"HELLO = {str_result}")
```

### LangChain Integration

```python
from cogents_core.toolify import BaseToolkit, ToolkitConfig
from langchain.agents import initialize_agent, AgentType
from cogents_core.llm import get_llm_client

class MyToolkit(BaseToolkit):
    def get_tools_map(self):
        return {
            "calculator": lambda x: eval(x),
            "length": lambda s: len(s)
        }

# Create toolkit
toolkit = MyToolkit(ToolkitConfig())

# Get LangChain tools
langchain_tools = toolkit.get_langchain_tools()

# Use with LangChain agent
# Note: You'll need to wrap the cogents LLM client for LangChain
# or use a native LangChain LLM
print(f"Available tools: {[t.name for t in langchain_tools]}")
```

### Tool Configuration

```python
from cogents_core.toolify import ToolkitConfig

# Basic configuration
config = ToolkitConfig(
    name="my_toolkit",
    mode="builtin",  # or "mcp"
    activated_tools=["tool1", "tool2"],  # Only activate specific tools
)

# With LLM integration
config = ToolkitConfig(
    name="ai_toolkit",
    llm_provider="openai",
    llm_model="gpt-4o",
    llm_config={
        "temperature": 0.3,
        "max_tokens": 500
    }
)

# With custom configuration
config = ToolkitConfig(
    name="api_toolkit",
    config={
        "api_key": "...",
        "base_url": "https://api.example.com",
        "timeout": 30
    }
)

# Access configuration in toolkit
class APIToolkit(BaseToolkit):
    def get_tools_map(self):
        api_key = self.config.get_tool_config("api_key")
        # Use configuration
        return {}
```

---

## Memory Management

Cogents includes the MemU memory system for advanced memory management with categorization and embedding support.

### Core Concepts

- **MemoryAgent**: Main interface for memory operations
- **Memory Categories**: Activity, Event, Profile memories
- **Embeddings**: Semantic search and linking
- **Memory Suggestions**: AI-generated memory categorization

### Basic Memory Agent Usage

```python
from cogents_core.memory.memu import MemoryAgent
from cogents_core.llm import get_llm_client

# Initialize LLM client
llm_client = get_llm_client(provider="openai", structured_output=True)

# Create memory agent
memory_agent = MemoryAgent(
    llm_client=llm_client,
    agent_id="agent_001",
    user_id="user_123",
    memory_dir="/tmp/memory_storage",
    enable_embeddings=True  # Enable semantic search
)

print(f"Memory types: {memory_agent.memory_types}")
```

### Adding Activity Memory

```python
# Activity memory tracks conversations and interactions
activity_content = """
USER: Hi, my name is Alice and I'm a software engineer.
ASSISTANT: Nice to meet you Alice! What kind of projects do you work on?
USER: I mainly work on backend systems using Python and Go.
ASSISTANT: That's interesting! Do you prefer microservices or monolithic architectures?
USER: I prefer microservices for scalability.
"""

result = memory_agent.call_function(
    "add_activity_memory",
    {
        "character_name": "Alice",
        "content": activity_content
    }
)

if result["success"]:
    print(f"Added {len(result['memory_items'])} memory items")
    for item in result["memory_items"]:
        print(f"- {item['content'][:50]}...")
```

### Generating Memory Suggestions

```python
# Generate suggestions for categorizing memories
if result["success"]:
    memory_items = result["memory_items"]
    
    suggestions = memory_agent.call_function(
        "generate_memory_suggestions",
        {
            "character_name": "Alice",
            "new_memory_items": memory_items
        }
    )
    
    if suggestions["success"]:
        print("\nMemory Suggestions:")
        for suggestion in suggestions["suggestions"]:
            print(f"Category: {suggestion['category']}")
            print(f"Content: {suggestion['content']}")
            print(f"Reasoning: {suggestion['reasoning']}")
            print()
```

### Updating Memory with Suggestions

```python
# Apply suggestions to categorize memories
if suggestions["success"]:
    update_result = memory_agent.call_function(
        "update_memory_with_suggestions",
        {
            "character_name": "Alice",
            "suggestions": suggestions["suggestions"]
        }
    )
    
    if update_result["success"]:
        print(f"Updated {update_result['updated_count']} memories")
        print(f"Skipped {update_result['skipped_count']} memories")
```

### Linking Related Memories

```python
# Find and link semantically related memories using embeddings
linking_result = memory_agent.call_function(
    "link_related_memories",
    {
        "character_name": "Alice",
        "threshold": 0.7,  # Similarity threshold
        "max_links": 5     # Max links per memory
    }
)

if linking_result["success"]:
    print(f"Created {linking_result['links_created']} memory links")
    for link in linking_result["sample_links"][:3]:
        print(f"\nLinked memories:")
        print(f"Source: {link['source_content'][:50]}...")
        print(f"Target: {link['target_content'][:50]}...")
        print(f"Similarity: {link['similarity']:.2f}")
```

### Retrieving Memories

```python
# Get available memory categories
categories = memory_agent.call_function(
    "get_available_categories",
    {"character_name": "Alice"}
)

print(f"Available categories: {categories['categories']}")

# Retrieve memories by category
event_memories = memory_agent.storage_manager.get_memories(
    category="event",
    character_name="Alice"
)

print(f"Found {len(event_memories)} event memories")
for memory in event_memories[:3]:
    print(f"- {memory['content']}")
```

### Memory Workflow Example

```python
from cogents_core.memory.memu import MemoryAgent
from cogents_core.llm import get_llm_client

# Initialize
llm_client = get_llm_client(provider="openai", structured_output=True)
agent = MemoryAgent(
    llm_client=llm_client,
    agent_id="chatbot_01",
    user_id="customer_456",
    memory_dir="./memory_data",
    enable_embeddings=True
)

# Conversation flow
conversation = """
USER: I'm planning a trip to Japan next month.
ASSISTANT: That sounds exciting! What cities are you planning to visit?
USER: I'm thinking Tokyo, Kyoto, and Osaka.
ASSISTANT: Great choices! Have you booked your flights yet?
USER: Yes, I'm flying out on March 15th.
"""

# Store activity memory
activity_result = agent.call_function(
    "add_activity_memory",
    {
        "character_name": "customer_456",
        "content": conversation
    }
)

# Generate and apply suggestions
if activity_result["success"]:
    suggestions_result = agent.call_function(
        "generate_memory_suggestions",
        {
            "character_name": "customer_456",
            "new_memory_items": activity_result["memory_items"]
        }
    )
    
    if suggestions_result["success"]:
        agent.call_function(
            "update_memory_with_suggestions",
            {
                "character_name": "customer_456",
                "suggestions": suggestions_result["suggestions"]
            }
        )

# Link related memories
agent.call_function(
    "link_related_memories",
    {
        "character_name": "customer_456",
        "threshold": 0.75,
        "max_links": 3
    }
)

# Later: retrieve relevant memories for context
# This would be used in your agent's retrieval logic
memories = agent.storage_manager.get_memories(
    category="event",
    character_name="customer_456"
)

print("Retrieved memories for context:")
for memory in memories[:5]:
    print(f"- {memory['content']}")
```

---

## Vector Store

Cogents provides vector storage integrations for semantic search and retrieval.

### Supported Providers

- **PGVector**: PostgreSQL with pgvector extension
- **Weaviate**: Cloud-native vector database

### PGVector Setup

```python
from cogents_core.vector_store import PGVectorStore
from cogents_core.llm import get_llm_client

# Initialize vector store
vector_store = PGVectorStore(
    collection_name="my_documents",
    embedding_model_dims=768,  # Match your embedding model
    dbname="vectordb",
    user="postgres",
    password="postgres",
    host="localhost",
    port=5432,
    # Index configuration
    diskann=False,  # Use HNSW by default
    hnsw={
        "m": 16,  # Number of connections
        "ef_construction": 64
    }
)

# Initialize embedding client
embed_client = get_llm_client(
    provider="ollama",
    embed_model="nomic-embed-text"
)

print(f"Embedding dimensions: {embed_client.get_embedding_dimensions()}")
```

### Inserting Documents

```python
# Prepare documents
documents = [
    {
        "id": "doc1",
        "content": "Machine learning is a subset of artificial intelligence.",
        "metadata": {"category": "AI", "type": "definition"}
    },
    {
        "id": "doc2",
        "content": "Deep learning uses neural networks with multiple layers.",
        "metadata": {"category": "AI", "type": "definition"}
    },
    {
        "id": "doc3",
        "content": "Python is a popular programming language for data science.",
        "metadata": {"category": "programming", "type": "fact"}
    }
]

# Generate embeddings and prepare for insertion
vectors = []
payloads = []
ids = []

for doc in documents:
    # Generate embedding
    embedding = embed_client.embed(doc["content"])
    vectors.append(embedding)
    payloads.append(doc["metadata"])
    ids.append(doc["id"])

# Insert into vector store
vector_store.insert(
    vectors=vectors,
    payloads=payloads,
    ids=ids
)

print(f"Inserted {len(documents)} documents")
```

### Searching

```python
# Query
query = "What is artificial intelligence?"
query_embedding = embed_client.embed(query)

# Search for similar documents
results = vector_store.search(
    query=query,
    vectors=query_embedding,
    limit=5,
    filter_conditions={"category": "AI"}  # Optional filter
)

print(f"\nSearch results for: '{query}'")
for result in results:
    print(f"\nID: {result.id}")
    print(f"Score: {result.score:.4f}")
    print(f"Metadata: {result.metadata}")
```

### Batch Operations

```python
# Batch embedding generation
texts = [
    "First document content",
    "Second document content",
    "Third document content"
]

embeddings = embed_client.embed_batch(texts)

# Batch insert
vector_store.insert(
    vectors=embeddings,
    payloads=[{"index": i} for i in range(len(texts))],
    ids=[f"batch_{i}" for i in range(len(texts))]
)
```

### Weaviate Integration

```python
from cogents_core.vector_store import WeaviateVectorStore

# Initialize Weaviate
vector_store = WeaviateVectorStore(
    collection_name="MyDocuments",
    embedding_model_dims=768,
    cluster_url="https://your-cluster.weaviate.network",
    auth_client_secret="your-api-key",
    additional_headers={
        "X-OpenAI-Api-Key": "sk-..."  # If using OpenAI embeddings
    }
)

# Similar API to PGVector
vector_store.insert(vectors=vectors, payloads=payloads, ids=ids)
results = vector_store.search(query=query, vectors=query_embedding, limit=5)
```

### Complete RAG Example

```python
from cogents_core.vector_store import PGVectorStore
from cogents_core.llm import get_llm_client

# Initialize components
embed_client = get_llm_client(provider="ollama", embed_model="nomic-embed-text")
llm_client = get_llm_client(provider="openai", chat_model="gpt-4o")

vector_store = PGVectorStore(
    collection_name="knowledge_base",
    embedding_model_dims=embed_client.get_embedding_dimensions(),
    dbname="vectordb"
)

# Index documents
knowledge_docs = [
    "Cogents is a cognitive agentic framework for building AI systems.",
    "The framework includes LLM integration, goal management, and tools.",
    "Vector stores enable semantic search and retrieval augmented generation.",
]

for i, doc in enumerate(knowledge_docs):
    embedding = embed_client.embed(doc)
    vector_store.insert(
        vectors=[embedding],
        payloads=[{"source": "docs", "index": i}],
        ids=[f"kb_{i}"]
    )

# RAG Query function
def rag_query(question: str, k: int = 3) -> str:
    """Answer question using RAG."""
    # 1. Embed query
    query_embedding = embed_client.embed(question)
    
    # 2. Retrieve relevant documents
    results = vector_store.search(
        query=question,
        vectors=query_embedding,
        limit=k
    )
    
    # 3. Build context from results
    context = "\n".join([
        f"Document {i+1}: {result.metadata}"
        for i, result in enumerate(results)
    ])
    
    # 4. Generate answer with LLM
    response = llm_client.completion([
        {
            "role": "system",
            "content": "Answer based on the provided context."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ])
    
    return response

# Use RAG
answer = rag_query("What is Cogents?")
print(f"Answer: {answer}")
```

---

## Message Bus & Events

Event-driven architecture for inter-component communication.

### Core Concepts

- **BaseEvent**: Base class for events
- **EventBus**: Central event dispatcher
- **BaseWatchdog**: Observer pattern for event handling
- **EventProcessor**: Process events with specific logic

### Basic Event System

```python
from cogents_core.msgbus import EventBus, BaseEvent, BaseWatchdog

# Define custom event
class TaskCreated(BaseEvent):
    """Event emitted when a task is created."""
    def __init__(self, task_id: str, task_name: str, priority: int):
        super().__init__()
        self.task_id = task_id
        self.task_name = task_name
        self.priority = priority

class TaskCompleted(BaseEvent):
    """Event emitted when a task completes."""
    def __init__(self, task_id: str, result: str):
        super().__init__()
        self.task_id = task_id
        self.result = result

# Create event bus
bus = EventBus()

# Define watchdog/handler
class TaskWatchdog(BaseWatchdog):
    """Monitor task-related events."""
    
    def handle_event(self, event: BaseEvent):
        """Handle incoming events."""
        if isinstance(event, TaskCreated):
            print(f"📝 Task created: {event.task_name} (ID: {event.task_id})")
            print(f"   Priority: {event.priority}")
        
        elif isinstance(event, TaskCompleted):
            print(f"✅ Task completed: {event.task_id}")
            print(f"   Result: {event.result}")

# Register watchdog
watchdog = TaskWatchdog()
bus.register_watchdog(watchdog)

# Publish events
bus.publish(TaskCreated("task_001", "Build API", priority=8))
bus.publish(TaskCompleted("task_001", "success"))
```

### Event Filtering

```python
from cogents_core.msgbus import BaseWatchdog, BaseEvent

class FilteredWatchdog(BaseWatchdog):
    """Watchdog that filters events."""
    
    def __init__(self, event_types: list):
        super().__init__()
        self.event_types = event_types
    
    def handle_event(self, event: BaseEvent):
        """Only handle specific event types."""
        if type(event) in self.event_types:
            print(f"Handling {event.__class__.__name__}")
            self.process_event(event)
    
    def process_event(self, event: BaseEvent):
        """Override to process specific events."""
        pass

# Use filtered watchdog
class HighPriorityWatchdog(FilteredWatchdog):
    """Only handle high priority task events."""
    
    def __init__(self):
        super().__init__([TaskCreated])
    
    def process_event(self, event: BaseEvent):
        if isinstance(event, TaskCreated) and event.priority >= 8:
            print(f"🚨 High priority task: {event.task_name}")

# Register
high_priority_watchdog = HighPriorityWatchdog()
bus.register_watchdog(high_priority_watchdog)

# Test
bus.publish(TaskCreated("task_002", "Critical fix", priority=10))
bus.publish(TaskCreated("task_003", "Minor update", priority=3))
```

### Multi-Component Coordination

```python
from cogents_core.msgbus import EventBus, BaseEvent, BaseWatchdog
from cogents_core.goalith.goalgraph.node import GoalNode, NodeStatus

class GoalStatusChanged(BaseEvent):
    """Event for goal status changes."""
    def __init__(self, goal_id: str, old_status: NodeStatus, new_status: NodeStatus):
        super().__init__()
        self.goal_id = goal_id
        self.old_status = old_status
        self.new_status = new_status

class GoalMonitor(BaseWatchdog):
    """Monitor and log goal status changes."""
    
    def handle_event(self, event: BaseEvent):
        if isinstance(event, GoalStatusChanged):
            print(f"Goal {event.goal_id}: {event.old_status} → {event.new_status}")

class GoalExecutor(BaseWatchdog):
    """Execute goals when they become ready."""
    
    def __init__(self, event_bus: EventBus):
        super().__init__()
        self.event_bus = event_bus
    
    def handle_event(self, event: BaseEvent):
        if isinstance(event, GoalStatusChanged):
            if event.new_status == NodeStatus.PENDING:
                self.check_and_execute(event.goal_id)
    
    def check_and_execute(self, goal_id: str):
        print(f"Checking if goal {goal_id} is ready for execution...")
        # Execute goal logic here
        # ...
        # Publish completion event
        self.event_bus.publish(
            GoalStatusChanged(goal_id, NodeStatus.PENDING, NodeStatus.COMPLETED)
        )

# Setup
bus = EventBus()
monitor = GoalMonitor()
executor = GoalExecutor(bus)

bus.register_watchdog(monitor)
bus.register_watchdog(executor)

# Trigger workflow
bus.publish(GoalStatusChanged("goal_1", NodeStatus.BLOCKED, NodeStatus.PENDING))
```

### Event Processor Pattern

```python
from cogents_core.msgbus import EventProcessor, BaseEvent
from typing import Dict, Any

class DataProcessor(EventProcessor):
    """Process data-related events."""
    
    def process(self, event: BaseEvent) -> Dict[str, Any]:
        """Process event and return result."""
        if isinstance(event, TaskCompleted):
            return {
                "status": "processed",
                "task_id": event.task_id,
                "result": event.result,
                "processed_at": event.timestamp
            }
        return {"status": "unhandled"}

# Usage
processor = DataProcessor()
event = TaskCompleted("task_001", "success")
result = processor.process(event)
print(result)
```

---

## Model Routing

Smart routing system to select appropriate LLM tiers based on query complexity.

### Core Concepts

- **ModelTier**: LITE, FAST, POWER tiers
- **ModelRouter**: Main routing coordinator
- **BaseRoutingStrategy**: Strategy interface
- **ComplexityScore**: Detailed complexity analysis
- **RoutingResult**: Routing decision with metadata

### Model Tiers

```python
from cogents_core.routing import ModelTier

# Three tiers available
print(ModelTier.LITE)   # Fast, local models
print(ModelTier.FAST)   # Balanced cloud models
print(ModelTier.POWER)  # High-capability models
```

### Basic Router Usage

```python
from cogents_core.routing import ModelRouter
from cogents_core.llm import get_llm_client

# Create lite client for complexity assessment
lite_client = get_llm_client(provider="ollama", chat_model="llama3.2:1b")

# Create router with default strategy
router = ModelRouter(
    strategy="dynamic_complexity",
    lite_client=lite_client
)

# Route queries
simple_query = "What is 2+2?"
result = router.route(simple_query)

print(f"Query: {simple_query}")
print(f"Recommended tier: {result.tier}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Complexity: {result.complexity_score.total:.2f}")

complex_query = "Explain the implications of quantum entanglement on the EPR paradox"
result = router.route(complex_query)

print(f"\nQuery: {complex_query}")
print(f"Recommended tier: {result.tier}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Complexity: {result.complexity_score.total:.2f}")
```

### Dynamic Complexity Strategy

```python
from cogents_core.routing import ModelRouter, DynamicComplexityStrategy
from cogents_core.llm import get_llm_client

lite_client = get_llm_client(provider="ollama", chat_model="llama3.2:1b")

# Configure strategy thresholds
router = ModelRouter(
    strategy="dynamic_complexity",
    lite_client=lite_client,
    strategy_config={
        "complexity_threshold_low": 0.3,   # Below: LITE
        "complexity_threshold_high": 0.7   # Above: POWER
    }
)

# Test different complexity queries
queries = [
    "Hello",
    "Calculate the sum of 1 to 100",
    "Explain machine learning algorithms",
    "Design a distributed system architecture with fault tolerance"
]

for query in queries:
    result = router.route(query)
    print(f"\nQuery: {query}")
    print(f"Tier: {result.tier.value}")
    print(f"Complexity: {result.complexity_score.total:.2f}")
```

### Self-Assessment Strategy

```python
from cogents_core.routing import ModelRouter

router = ModelRouter(
    strategy="self_assessment",
    lite_client=lite_client,
    strategy_config={
        "use_structured_output": True
    }
)

query = "What are the ethical implications of AGI?"
result = router.route(query)

print(f"Strategy: {result.strategy}")
print(f"Recommended tier: {result.tier}")
print(f"Confidence: {result.confidence}")
```

### Route and Configure

```python
from cogents_core.routing import ModelRouter, ModelTier

router = ModelRouter(strategy="dynamic_complexity", lite_client=lite_client)

# Define tier configurations
tier_configs = {
    ModelTier.LITE: {
        "provider": "ollama",
        "chat_model": "llama3.2:1b",
        "temperature": 0.3
    },
    ModelTier.FAST: {
        "provider": "openai",
        "chat_model": "gpt-4o-mini",
        "temperature": 0.7
    },
    ModelTier.POWER: {
        "provider": "openai",
        "chat_model": "gpt-4o",
        "temperature": 0.7,
        "max_tokens": 2048
    }
}

# Route and get configuration
query = "Analyze this codebase architecture"
routing_result, model_config = router.route_and_configure(
    query,
    model_configs=tier_configs
)

print(f"Using tier: {routing_result.tier}")
print(f"Model config: {model_config}")

# Create client with recommended config
from cogents_core.llm import get_llm_client
client = get_llm_client(**model_config)
response = client.completion([{"role": "user", "content": query}])
```

### Custom Routing Strategy

```python
from cogents_core.routing import BaseRoutingStrategy, RoutingResult, ModelTier, ComplexityScore
from cogents_core.llm.base import BaseLLMClient
from typing import Optional, Dict, Any

class KeywordRoutingStrategy(BaseRoutingStrategy):
    """Route based on keywords in query."""
    
    def __init__(self, lite_client: Optional[BaseLLMClient] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(lite_client, config)
        self.power_keywords = self.config.get("power_keywords", [
            "architecture", "design", "complex", "advanced"
        ])
        self.lite_keywords = self.config.get("lite_keywords", [
            "hello", "hi", "thanks", "simple"
        ])
    
    def get_strategy_name(self) -> str:
        return "keyword_routing"
    
    def route(self, query: str) -> RoutingResult:
        """Route based on keyword matching."""
        query_lower = query.lower()
        
        # Check for power keywords
        power_score = sum(1 for kw in self.power_keywords if kw in query_lower)
        lite_score = sum(1 for kw in self.lite_keywords if kw in query_lower)
        
        # Determine tier
        if power_score > 0:
            tier = ModelTier.POWER
            complexity = 0.9
        elif lite_score > 0:
            tier = ModelTier.LITE
            complexity = 0.1
        else:
            tier = ModelTier.FAST
            complexity = 0.5
        
        return RoutingResult(
            tier=tier,
            confidence=0.8,
            complexity_score=ComplexityScore(total=complexity),
            strategy=self.get_strategy_name()
        )

# Register and use custom strategy
from cogents_core.routing import ModelRouter

ModelRouter.register_strategy("keyword", KeywordRoutingStrategy)

router = ModelRouter(
    strategy="keyword",
    strategy_config={
        "power_keywords": ["design", "architecture", "implement"],
        "lite_keywords": ["hello", "hi", "thanks"]
    }
)

# Test
result = router.route("Design a microservices architecture")
print(f"Tier: {result.tier}")  # Should be POWER
```

---

## Tracing & Observability

Production-ready observability with token tracking and Opik integration.

### Token Tracking

```python
from cogents_core.tracing import (
    get_token_tracker,
    TokenUsageTracker,
    record_token_usage
)
from cogents_core.llm import get_llm_client

# Get global tracker
tracker = get_token_tracker()

# Reset tracker for new session
tracker.reset()

# Make LLM calls (automatically tracked)
client = get_llm_client(provider="openai")

response1 = client.completion([
    {"role": "user", "content": "Hello"}
])

response2 = client.completion([
    {"role": "user", "content": "Tell me a story"}
])

# Get comprehensive statistics
stats = tracker.get_stats()
print(f"Total tokens: {stats['total_tokens']}")
print(f"Total calls: {stats['total_calls']}")
print(f"Prompt tokens: {stats['total_prompt_tokens']}")
print(f"Completion tokens: {stats['total_completion_tokens']}")
print(f"Average tokens per call: {stats['avg_tokens_per_call']:.2f}")

# Get detailed call history
for call in tracker.get_calls():
    print(f"\nCall at {call.timestamp}")
    print(f"  Model: {call.model_name}")
    print(f"  Tokens: {call.total_tokens}")
    print(f"  Prompt: {call.prompt_tokens}")
    print(f"  Completion: {call.completion_tokens}")
```

### Manual Token Recording

```python
from cogents_core.tracing import record_token_usage, TokenUsage

# Record custom token usage
usage = TokenUsage(
    prompt_tokens=100,
    completion_tokens=50,
    total_tokens=150,
    model_name="custom-model",
    timestamp=datetime.now()
)

record_token_usage(usage)
```

### Opik Tracing Integration

```python
from cogents_core.tracing import (
    configure_opik,
    is_opik_enabled,
    create_opik_trace,
    get_opik_project
)

# Configure Opik
configure_opik(
    api_key="your-opik-api-key",
    project_name="my-ai-project",
    workspace="my-workspace"
)

# Check if enabled
if is_opik_enabled():
    print(f"Opik project: {get_opik_project()}")

# Create trace
with create_opik_trace("agent_execution", input_data={"query": "test"}) as trace:
    # Your agent logic here
    result = client.completion([{"role": "user", "content": "test"}])
    trace.update(output={"result": result})
```

### LangGraph Callbacks

```python
from cogents_core.tracing import NodeLoggingCallback, TokenUsageCallback
from langgraph.graph import StateGraph

# Token tracking callback
token_callback = TokenUsageCallback()

# Node logging callback
logging_callback = NodeLoggingCallback()

# Use with LangGraph
config = {
    "callbacks": [token_callback, logging_callback]
}

# Run graph with callbacks
graph.invoke(initial_state, config=config)

# Get tracked tokens
stats = token_callback.get_usage_stats()
print(f"Graph execution used {stats['total_tokens']} tokens")
```

### Comprehensive Monitoring Example

```python
from cogents_core.tracing import get_token_tracker
from cogents_core.llm import get_llm_client
from cogents_core.routing import ModelRouter
import time

# Setup
tracker = get_token_tracker()
tracker.reset()

lite_client = get_llm_client(provider="ollama", chat_model="llama3.2:1b")
router = ModelRouter(strategy="dynamic_complexity", lite_client=lite_client)

# Track multi-query session
queries = [
    "Hello",
    "Explain async programming",
    "Design a distributed caching system"
]

results = []
for query in queries:
    start = time.time()
    
    # Route query
    routing = router.route(query)
    
    # Get appropriate client
    if routing.tier.value == "lite":
        client = lite_client
    else:
        client = get_llm_client(provider="openai")
    
    # Execute
    response = client.completion([{"role": "user", "content": query}])
    
    elapsed = time.time() - start
    
    results.append({
        "query": query,
        "tier": routing.tier.value,
        "complexity": routing.complexity_score.total,
        "response_length": len(response),
        "elapsed": elapsed
    })

# Generate report
stats = tracker.get_stats()

print("\n" + "="*60)
print("EXECUTION REPORT")
print("="*60)

print(f"\nTotal Queries: {len(queries)}")
print(f"Total Tokens Used: {stats['total_tokens']}")
print(f"Total API Calls: {stats['total_calls']}")
print(f"Average Tokens/Call: {stats['avg_tokens_per_call']:.2f}")

print("\nPer-Query Breakdown:")
for i, result in enumerate(results, 1):
    print(f"\n{i}. {result['query']}")
    print(f"   Tier: {result['tier']}")
    print(f"   Complexity: {result['complexity']:.2f}")
    print(f"   Response: {result['response_length']} chars")
    print(f"   Time: {result['elapsed']:.2f}s")

print("\n" + "="*60)
```

---

## Advanced Patterns

### Multi-Agent Coordination

```python
from cogents_core.agent import BaseAgent
from cogents_core.msgbus import EventBus, BaseEvent, BaseWatchdog
from typing import Dict, Any

class AgentMessage(BaseEvent):
    """Message between agents."""
    def __init__(self, sender: str, recipient: str, content: str):
        super().__init__()
        self.sender = sender
        self.recipient = recipient
        self.content = content

class CoordinatedAgent(BaseAgent):
    """Agent that communicates via event bus."""
    
    def __init__(self, agent_id: str, event_bus: EventBus, **kwargs):
        super().__init__(**kwargs)
        self.agent_id = agent_id
        self.event_bus = event_bus
        
        # Setup message handler
        self.watchdog = AgentWatchdog(agent_id, self)
        event_bus.register_watchdog(self.watchdog)
    
    def send_message(self, recipient: str, content: str):
        """Send message to another agent."""
        self.event_bus.publish(
            AgentMessage(self.agent_id, recipient, content)
        )
    
    def receive_message(self, sender: str, content: str):
        """Handle incoming message."""
        self.logger.info(f"Received from {sender}: {content}")
        # Process message and optionally respond
        response = self.process_message(content)
        if response:
            self.send_message(sender, response)
    
    def process_message(self, content: str) -> str:
        """Process message and generate response."""
        # Use LLM to process
        result = self.llm.completion([
            {"role": "user", "content": f"Process this message: {content}"}
        ])
        return result
    
    def run(self, user_message: str, context: Dict[str, Any] = None, config = None) -> Any:
        """Execute agent task."""
        return self.llm.completion([
            {"role": "user", "content": user_message}
        ])

class AgentWatchdog(BaseWatchdog):
    """Watch for messages directed at specific agent."""
    
    def __init__(self, agent_id: str, agent: CoordinatedAgent):
        super().__init__()
        self.agent_id = agent_id
        self.agent = agent
    
    def handle_event(self, event: BaseEvent):
        if isinstance(event, AgentMessage):
            if event.recipient == self.agent_id:
                self.agent.receive_message(event.sender, event.content)

# Usage
bus = EventBus()

agent1 = CoordinatedAgent("researcher", bus, llm_provider="openai")
agent2 = CoordinatedAgent("writer", bus, llm_provider="openai")

# Agent 1 sends message to Agent 2
agent1.send_message("writer", "Research quantum computing")

# Agent 2 processes and responds
# This happens automatically via event bus
```

### RAG with Goalith Pipeline

```python
from cogents_core.goalith.goalgraph.graph import GoalGraph
from cogents_core.goalith.goalgraph.node import GoalNode, NodeStatus
from cogents_core.vector_store import PGVectorStore
from cogents_core.llm import get_llm_client

class RAGPipeline:
    """RAG pipeline with goal-driven processing."""
    
    def __init__(self):
        self.llm = get_llm_client(provider="openai")
        self.embed = get_llm_client(provider="ollama", embed_model="nomic-embed-text")
        self.vector_store = PGVectorStore(
            collection_name="rag_docs",
            embedding_model_dims=self.embed.get_embedding_dimensions()
        )
        self.graph = GoalGraph()
    
    def index_documents(self, documents: list):
        """Create indexing goals and execute."""
        # Create main goal
        main_goal = GoalNode(
            description="Index documents",
            priority=10.0
        )
        self.graph.add_node(main_goal)
        
        # Create subgoal for each document
        for i, doc in enumerate(documents):
            subgoal = GoalNode(
                description=f"Index document {i}",
                priority=9.0,
                parent=main_goal.id,
                context={"document": doc, "index": i}
            )
            self.graph.add_node(subgoal)
            self.graph.add_parent_child(main_goal.id, subgoal.id)
        
        # Execute indexing
        while True:
            ready = self.graph.get_ready_nodes()
            if not ready:
                break
            
            for task in ready:
                doc = task.context.get("document")
                if doc:
                    # Generate embedding and store
                    embedding = self.embed.embed(doc)
                    self.vector_store.insert(
                        vectors=[embedding],
                        payloads=[{"content": doc}],
                        ids=[task.id]
                    )
                
                task.mark_completed()
                self.graph.update_node(task)
    
    def query(self, question: str, k: int = 3) -> str:
        """Answer question using RAG."""
        # Create query goal
        query_goal = GoalNode(
            description=f"Answer: {question}",
            context={"question": question, "k": k}
        )
        self.graph.add_node(query_goal)
        
        # Retrieve
        query_embedding = self.embed.embed(question)
        results = self.vector_store.search(
            query=question,
            vectors=query_embedding,
            limit=k
        )
        
        # Build context
        context = "\n".join([r.metadata.get("content", "") for r in results])
        
        # Generate answer
        response = self.llm.completion([
            {
                "role": "system",
                "content": "Answer based on provided context."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ])
        
        query_goal.mark_completed()
        query_goal.update_context("answer", response)
        self.graph.update_node(query_goal)
        
        return response

# Usage
pipeline = RAGPipeline()

# Index knowledge base
documents = [
    "Cogents is a cognitive agentic framework.",
    "It includes goal management with Goalith.",
    "Vector stores enable semantic search."
]
pipeline.index_documents(documents)

# Query
answer = pipeline.query("What is Goalith?")
print(answer)

# Review execution graph
stats = pipeline.graph.get_graph_stats()
print(f"Executed {stats['total_nodes']} goals")
```

### Adaptive Tool Selection

```python
from cogents_core.toolify import BaseToolkit, ToolkitRegistry, ToolkitConfig
from cogents_core.routing import ModelRouter
from cogents_core.llm import get_llm_client

class AdaptiveToolExecutor:
    """Executes tools with adaptive model selection."""
    
    def __init__(self):
        self.router = ModelRouter(
            strategy="dynamic_complexity",
            lite_client=get_llm_client(provider="ollama", chat_model="llama3.2:1b")
        )
    
    def execute_with_routing(self, toolkit_name: str, tool_name: str, **kwargs):
        """Execute tool with adaptive model selection."""
        # Route based on task description
        task_desc = f"{toolkit_name}.{tool_name} with {kwargs}"
        routing = self.router.route(task_desc)
        
        # Get toolkit with appropriate model
        config = ToolkitConfig(
            name=toolkit_name,
            llm_provider="openai" if routing.tier.value != "lite" else "ollama",
            llm_model="gpt-4o" if routing.tier.value == "power" else "gpt-4o-mini"
        )
        
        toolkit = ToolkitRegistry.create_toolkit(toolkit_name, config)
        
        # Execute tool
        result = toolkit.call_tool(tool_name, **kwargs)
        
        return {
            "result": result,
            "tier_used": routing.tier.value,
            "complexity": routing.complexity_score.total
        }

# Usage
executor = AdaptiveToolExecutor()

result = executor.execute_with_routing(
    "text_analysis",
    "summarize",
    text="Long document content...",
    max_words=50
)

print(f"Result: {result['result']}")
print(f"Used tier: {result['tier_used']}")
```

---

## Best Practices

### 1. LLM Client Management

```python
# DO: Reuse clients
client = get_llm_client(provider="openai")
for query in queries:
    response = client.completion([{"role": "user", "content": query}])

# DON'T: Create new clients in loops
for query in queries:
    client = get_llm_client(provider="openai")  # Wasteful
    response = client.completion([{"role": "user", "content": query}])
```

### 2. Error Handling

```python
from cogents_core.llm import get_llm_client

client = get_llm_client(provider="openai")

try:
    response = client.completion(messages)
except Exception as e:
    logger.error(f"LLM call failed: {e}")
    # Implement fallback logic
    response = fallback_response()
```

### 3. Goal Graph Organization

```python
# DO: Use descriptive names and context
goal = GoalNode(
    description="Implement user authentication system",
    context={
        "auth_method": "JWT",
        "database": "PostgreSQL",
        "security_level": "high"
    },
    tags=["security", "backend", "auth"]
)

# DON'T: Vague descriptions
goal = GoalNode(description="Do auth")  # Too vague
```

### 4. Memory Management

```python
# DO: Batch operations when possible
memory_items = []
for conversation in conversations:
    items = extract_memory_items(conversation)
    memory_items.extend(items)

# Process in batch
memory_agent.call_function(
    "update_memory_with_suggestions",
    {"character_name": "user", "suggestions": memory_items}
)

# DON'T: Process one at a time
for item in memory_items:  # Inefficient
    memory_agent.call_function(...)
```

### 5. Token Tracking

```python
# DO: Reset tracker for new sessions
from cogents_core.tracing import get_token_tracker

tracker = get_token_tracker()
tracker.reset()  # Start fresh

# ... perform operations ...

stats = tracker.get_stats()
print(f"Session used {stats['total_tokens']} tokens")
```

### 6. Structured Outputs

```python
# DO: Use Pydantic models for structured data
from pydantic import BaseModel, Field

class CodeReview(BaseModel):
    issues: list[str]
    severity: str
    suggestions: list[str]

client = get_llm_client(provider="openai", structured_output=True)
review = client.structured_completion(messages, CodeReview)

# DON'T: Parse unstructured text
response = client.completion(messages)  # Returns string
# Manual parsing is error-prone
```

### 7. Vector Store Indexing

```python
# DO: Batch embeddings
texts = ["doc1", "doc2", "doc3"]
embeddings = embed_client.embed_batch(texts)  # Efficient

# DON'T: Generate one at a time
embeddings = [embed_client.embed(text) for text in texts]  # Slower
```

### 8. Event-Driven Architecture

```python
# DO: Use events for loose coupling
bus.publish(TaskCompleted(task_id="123", result="success"))

# DON'T: Direct coupling
task_monitor.on_task_completed(task_id, result)  # Tight coupling
```

---

## API Reference

### Quick Reference Table

| Module | Key Classes | Purpose |
|--------|-------------|---------|
| `cogents_core.llm` | `get_llm_client()`, `BaseLLMClient` | LLM integration |
| `cogents_core.agent` | `BaseAgent`, `BaseGraphicAgent` | Agent framework |
| `cogents_core.goalith` | `GoalNode`, `GoalGraph`, `GoalDecomposer` | Goal management |
| `cogents_core.toolify` | `BaseToolkit`, `ToolkitRegistry` | Tool management |
| `cogents_core.memory` | `MemoryAgent`, `MemoryItem` | Memory systems |
| `cogents_core.vector_store` | `PGVectorStore`, `WeaviateVectorStore` | Vector storage |
| `cogents_core.msgbus` | `EventBus`, `BaseWatchdog` | Event messaging |
| `cogents_core.routing` | `ModelRouter`, `ModelTier` | Model routing |
| `cogents_core.tracing` | `get_token_tracker()`, `TokenUsage` | Observability |

### Import Patterns

```python
# LLM
from cogents_core.llm import get_llm_client, BaseLLMClient

# Agent
from cogents_core.agent import BaseAgent, BaseGraphicAgent, BaseConversationAgent

# Goalith
from cogents_core.goalith.goalgraph.node import GoalNode, NodeStatus
from cogents_core.goalith.goalgraph.graph import GoalGraph
from cogents_core.goalith.decomposer import LLMDecomposer, SimpleListDecomposer

# Toolify
from cogents_core.toolify import (
    BaseToolkit,
    AsyncBaseToolkit,
    ToolkitConfig,
    ToolkitRegistry,
    register_toolkit,
    get_toolkit
)

# Memory
from cogents_core.memory.memu import MemoryAgent

# Vector Store
from cogents_core.vector_store import PGVectorStore, WeaviateVectorStore, get_vector_store

# Message Bus
from cogents_core.msgbus import EventBus, BaseEvent, BaseWatchdog

# Routing
from cogents_core.routing import ModelRouter, ModelTier, RoutingResult

# Tracing
from cogents_core.tracing import (
    get_token_tracker,
    TokenUsage,
    configure_opik,
    create_opik_trace
)
```

---

## Conclusion

Cogents Core provides a comprehensive foundation for building sophisticated AI agents with:

- **Flexible LLM integration** across multiple providers
- **Hierarchical goal management** with DAG-based planning
- **Extensible tool system** with MCP support
- **Advanced memory management** with semantic retrieval
- **Production-ready observability** and monitoring
- **Event-driven architecture** for scalable systems

This guide covered the essential public APIs for AI coding agents. For additional examples, see the `examples/` directory in the repository.

### Additional Resources

- **GitHub**: https://github.com/caesar0301/cogents-core
- **PyPI**: https://pypi.org/project/cogents-core/
- **Documentation**: Design specs in `specs/` directory
- **Examples**: Comprehensive examples in `examples/` directory

### Contributing

Cogents Core is open source (MIT License). Contributions are welcome!

---

**Last Updated**: December 2025  
**Version**: 1.0.0
