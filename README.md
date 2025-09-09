# Cogents-core

[![CI](https://github.com/mirasurf/cogents-core/actions/workflows/ci.yml/badge.svg)](https://github.com/mirasurf/cogents-core/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/cogents-core.svg)](https://pypi.org/project/cogents-core/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mirasurf/cogents-core)

This is part of [project Cogents](https://github.com/mirasurf/cogents), an initiative to develop a cognitive, computation-driven agentic system. This repo the foundational abstractions (Agent, Memory, Tool, Goal, Orchestration, and more) along with essential modules such as LLM clients, logging, message buses, model routing, and observability. For the underlying philosophy, refer to my talk on MAS ([link](https://github.com/caesar0301/mas-talk-2508/blob/master/mas-talk-xmingc.pdf)).

## Core Modules

Cogents offers a comprehensive set of modules for creating intelligent agent-based applications:

### LLM Integration & Management
- **Multi-model support**: OpenAI, Google GenAI (via OpenRouter), Ollama, and LlamaCPP
- **Advanced routing**: Dynamic complexity-based and self-assessment routing strategies
- **Tracing & monitoring**: Built-in token tracking and Opik tracing integration
- **Extensible architecture**: Easy to add new LLM providers

### Goal Management & Planning
- **Goal decomposition**: LLM-based and callable goal decomposition strategies
- **Conflict detection**: Automated goal conflict identification and resolution
- **Replanning**: Dynamic goal replanning capabilities

### Tool Management
- **Tool registry**: Centralized tool registration and management
- **Execution engine**: Robust tool execution with error handling
- **Repository system**: Organized tool storage and retrieval

### Memory Management
- Under development

### Orchestration
- Under development


## Project Structure

```
cogents/core
├── base/            # Base classes and models
├── goalith/         # Goal management and planning
├── memory/          # Memory management (on plan)
├── orchestrix/      # Global orchestration (on plan)
└── toolify/         # Tool management and execution
```

## Creating a New Agent

### From Base Classes
Start with the base agent classes in `cogents_core.base` to create custom agents with full control over behavior and capabilities.

#### Base Agent Class Hierarchy

```
BaseAgent (abstract)
├── Core functionality
│   ├── LLM client management
│   ├── Token usage tracking
│   ├── Logging capabilities
│   └── Configuration management
│
├── BaseGraphicAgent (abstract)
│   ├── LangGraph integration
│   ├── State management
│   ├── Graph visualization
│   └── Error handling patterns
│   │
│   ├── BaseConversationAgent (abstract)
│   │   ├── Session management
│   │   ├── Message handling
│   │   ├── Conversation state
│   │   └── Response generation
│   │
│   └── BaseResearcher (abstract)
│       ├── Research workflow
│       ├── Source management
│       ├── Query generation
│       └── Result compilation
│           └── Uses ResearchOutput model
│               ├── content: str
│               ├── sources: List[Dict]
│               ├── summary: str
│               └── timestamp: datetime
```

**Key Inheritance Paths:**
- **BaseAgent**: Core functionality (LLM client, token tracking, logging)
- **BaseGraphicAgent**: LangGraph integration and visualization
- **BaseConversationAgent**: Session management and conversation patterns
- **BaseResearcher**: Research workflow and structured output patterns

### From Existing Agents
Use well-constructed agents like Seekra Agent as templates:

```python
from cogents_core.agents.seekra_agent import SeekraAgent

# Extend Seekra Agent for custom research tasks
class CustomResearchAgent(SeekraAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add custom functionality
        
    def custom_research_method(self):
        # Implement custom research logic
        pass
```

## Install

```
pip install -U cogents-core
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
