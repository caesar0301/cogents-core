# Cogents

[![CI](https://github.com/mirasurf/cogents-core/actions/workflows/ci.yml/badge.svg)](https://github.com/mirasurf/cogents-core/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/cogents.svg)](https://pypi.org/project/cogents/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mirasurf/cogents-core)

A comprehensive collection of essential building blocks for constructing cognitive multi-agent systems (MAS). Rather than building a full agent framework, Cogents provides a lightweight repository of key components designed to bridge the final mile in MAS development. Our philosophy focuses on modular, composable components that can be easily integrated into existing systems or used to build new ones from the ground up. For the underlying philosophy, refer to my talk on MAS ([link](https://github.com/caesar0301/mas-talk-2508/blob/master/mas-talk-xmingc.pdf)).

## Core Modules

Cogents offers a comprehensive set of modules for creating intelligent agent-based applications:

### LLM Integration & Management
- **Multi-model support**: OpenAI, Google GenAI (via OpenRouter), Ollama, and LlamaCPP
- **Advanced routing**: Dynamic complexity-based and self-assessment routing strategies
- **Tracing & monitoring**: Built-in token tracking and Opik tracing integration
- **Extensible architecture**: Easy to add new LLM providers

### Extensible Resources & Capabilities
- **Web search**: Tavily and Google AI Search integration
- **Vector stores**: Weaviate and PostgreSQL with pgvector support
- **Semantic search**: Integrated semantic web search with document processing
- **Voice processing**: Smart voice transcription and processing features

### Goal Management & Planning
- **Goal decomposition**: LLM-based and callable goal decomposition strategies
- **Conflict detection**: Automated goal conflict identification and resolution
- **Replanning**: Dynamic goal replanning capabilities

### Tool Management
- **Tool registry**: Centralized tool registration and management
- **Execution engine**: Robust tool execution with error handling
- **Repository system**: Organized tool storage and retrieval

### Agent Gallery
- **Askura Agent**: Advanced conversation and memory management agent
- **Seekra Agent**: Research-focused agent with web search capabilities
- **Mem Agent**: Memory-focused agent (coming soon)
- **Cogito Agent**: Cognitive reasoning agent (coming soon)

## Project Structure

```
cogents/
├── agents/           # Agent implementations
├── base/            # Base classes and models
├── common/          # Shared utilities and LLM integrations
├── goalith/         # Goal management and planning
├── memory/          # Memory management (on plan)
├── orchestrix/      # Global orchestration (on plan)
├── resources/       # External service integrations
└── toolify/         # Tool management and execution
```

## Creating a New Agent

### From Base Classes
Start with the base agent classes in `cogents.base` to create custom agents with full control over behavior and capabilities.

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
from cogents.core.agents.seekra_agent import SeekraAgent

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
pip install -U cogents
```

## Documentation

For detailed documentation, visit: https://cogents.readthedocs.io/ (under construction)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgment

- Tencent [Youtu-agent](https://github.com/Tencent/Youtu-agent) toolkits integration.
